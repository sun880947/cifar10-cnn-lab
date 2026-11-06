"""Fine-tune a trained model on corrupted CIFAR-10 for the deployment study.

This is a small domain-adaptation experiment: starting from the clean baseline
checkpoint, continue training with random synthetic corruptions applied to half
of the batches.  The clean and corrupted test accuracies are compared against
the unmodified model.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.corruptions import CORRUPTIONS, apply_corruption, random_corruption_batch  # noqa: E402
from src.data import (  # noqa: E402
    augment_batch,
    load_npz,
    prepare_input,
    stratified_split,
)
from src.engine import cosine_learning_rate, evaluate_model  # noqa: E402
from src.models import MODEL_LABELS, MODEL_SPECS, build_model  # noqa: E402
from src.robustness import evaluate_corruption_suite  # noqa: E402
from src.utils import configure_threads, ensure_dir, save_json, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet20_ds_se")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--cutout", type=int, default=8)
    p.add_argument("--corruption-prob", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="robust")
    return p.parse_args()


def build_mixed_validation(
    images: torch.Tensor,
    labels: torch.Tensor,
    val_idx: np.ndarray,
    *,
    per_corruption: int = 1000,
    seed: int = 999,
):
    """Clean val subset plus a fixed severity-3 corrupted subset per corruption."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(val_idx, size=min(per_corruption, len(val_idx)), replace=False)
    xs = [images[pick]]
    ys = [labels[pick]]
    for name in CORRUPTIONS:
        corrupted = apply_corruption(images[pick], name, 3, seed=seed + 17)
        xs.append(corrupted)
        ys.append(labels[pick])
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def main() -> None:
    args = parse_args()
    configure_threads(args.threads)
    set_seed(args.seed)
    ensure_dir(os.path.join(args.results_dir, "metrics"))
    ensure_dir(os.path.join(args.results_dir, "history"))

    checkpoint = args.checkpoint or os.path.join(args.checkpoint_dir, f"{args.model}_best.pt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(args.model)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(memory_format=torch.channels_last)
    print(f"base checkpoint {checkpoint} (epoch {ckpt.get('epoch')}, val {ckpt.get('val_accuracy', 0):.2%})")

    train_data = load_npz(os.path.join(args.data_dir, "cifar10_train.npz"))
    test_data = load_npz(os.path.join(args.data_dir, "cifar10_test.npz"))
    train_idx, val_idx = stratified_split(train_data.labels, val_ratio=0.1, seed=args.seed)

    base_metrics, _, _ = evaluate_model(
        model, test_data.images, test_data.labels, np.arange(len(test_data)), batch_size=512
    )
    print(f"before fine-tune: clean test acc={base_metrics['accuracy']:.2%}")
    base_suite = evaluate_corruption_suite(
        model, test_data.images, test_data.labels, batch_size=512, verbose=False
    )
    print(f"before fine-tune: corruption mean={base_suite['mean_accuracy']:.2%}")

    mixed_x, mixed_y = build_mixed_validation(train_data.images, train_data.labels, val_idx)
    print(f"mixed validation set: {len(mixed_y)} images")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True
    )
    rng = np.random.default_rng(args.seed + 7)
    history = []
    best_mixed = -1.0
    best_epoch = 0
    best_path = os.path.join(args.checkpoint_dir, f"{args.model}_{args.tag}_best.pt")
    history_path = os.path.join(args.results_dir, "history", f"{args.model}_{args.tag}.json")

    for epoch in range(1, args.epochs + 1):
        lr = cosine_learning_rate(epoch - 1, args.epochs, args.lr, args.min_lr, args.warmup_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        order = train_idx.copy()
        rng.shuffle(order)
        total_loss = 0.0
        correct = 0
        total = 0
        start = time.perf_counter()
        for offset in range(0, len(order), args.batch_size):
            batch_idx = torch.from_numpy(order[offset : offset + args.batch_size])
            xb = train_data.images[batch_idx]
            yb = train_data.labels[batch_idx]
            if rng.random() < args.corruption_prob:
                xb = random_corruption_batch(xb, rng, max_severity=5, corruption_prob=1.0)
            xb = augment_batch(xb, cutout_size=args.cutout)
            xb = prepare_input(xb)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=args.label_smoothing)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * yb.numel()
            correct += int((logits.argmax(1) == yb).sum())
            total += int(yb.numel())
        train_seconds = time.perf_counter() - start
        mixed_metrics, _, _ = evaluate_model(
            model, mixed_x, mixed_y, np.arange(len(mixed_y)), batch_size=512
        )
        mixed_acc = float(mixed_metrics["accuracy"])
        record = {
            "epoch": epoch,
            "lr": round(lr, 8),
            "train_loss": round(total_loss / max(1, total), 6),
            "train_accuracy": round(correct / max(1, total), 6),
            "mixed_val_accuracy": round(mixed_acc, 6),
            "train_seconds": round(train_seconds, 2),
        }
        history.append(record)
        save_json(history, history_path)
        improved = mixed_acc > best_mixed
        if improved:
            best_mixed = mixed_acc
            best_epoch = epoch
            torch.save(
                {
                    "model": args.model,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "mixed_val_accuracy": mixed_acc,
                    "base_checkpoint": checkpoint,
                    "args": vars(args),
                },
                best_path,
            )
        print(
            f"epoch {epoch:2d}/{args.epochs} lr={lr:.5f} loss={record['train_loss']:.4f} "
            f"train_acc={record['train_accuracy']:.2%} mixed_val={mixed_acc:.2%} "
            f"({train_seconds:.0f}s){' *' if improved else ''}",
            flush=True,
        )

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics, _, _ = evaluate_model(
        model, test_data.images, test_data.labels, np.arange(len(test_data)), batch_size=512
    )
    suite = evaluate_corruption_suite(
        model, test_data.images, test_data.labels, batch_size=512, verbose=True
    )
    summary = {
        "model": args.model,
        "label": MODEL_LABELS[args.model],
        "base_checkpoint": checkpoint,
        "finetuned_checkpoint": best_path,
        "best_epoch": best_epoch,
        "best_mixed_val_accuracy": round(best_mixed, 6),
        "epochs": args.epochs,
        "clean_before": float(base_metrics["accuracy"]),
        "clean_after": float(test_metrics["accuracy"]),
        "corruption_mean_before": float(base_suite["mean_accuracy"]),
        "corruption_mean_after": float(suite["mean_accuracy"]),
        "corruption_worst_before": float(base_suite["worst_accuracy"]),
        "corruption_worst_after": float(suite["worst_accuracy"]),
        "test_after": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in test_metrics.items()},
        "robustness_before": base_suite,
        "robustness_after": suite,
        "history": history,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = os.path.join(args.results_dir, "metrics", f"{args.model}_{args.tag}_summary.json")
    save_json(summary, out)
    print(
        f"AFTER: clean={summary['clean_after']:.2%} (delta {summary['clean_after'] - summary['clean_before']:+.2%}), "
        f"corruption mean={summary['corruption_mean_after']:.2%} "
        f"(delta {summary['corruption_mean_after'] - summary['corruption_mean_before']:+.2%})"
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()