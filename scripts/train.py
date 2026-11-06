"""Train one CIFAR-10 ResNet variant with a fixed, reproducible recipe.

Example
-------
    python -m scripts.train --model resnet20 --epochs 60 --threads 16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

# Allow running as a plain script from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_npz, stratified_split  # noqa: E402
from src.engine import cosine_learning_rate, evaluate_model, train_one_epoch  # noqa: E402
from src.flops import model_profile  # noqa: E402
from src.models import MODEL_LABELS, MODEL_SPECS, build_model, count_parameters  # noqa: E402
from src.utils import configure_threads, ensure_dir, load_json, save_json, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="resnet20", choices=sorted(MODEL_SPECS))
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--cutout", type=int, default=0)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--resume", action="store_true", help="resume from {model}_last.pt if present")
    p.add_argument("--max-train-batches", type=int, default=0, help="debug: limit batches per epoch")
    p.add_argument("--max-val-samples", type=int, default=0, help="debug: limit validation samples")
    p.add_argument("--tag", default="", help="optional suffix for output file names")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_threads(args.threads)
    set_seed(args.seed)

    suffix = f"_{args.tag}" if args.tag else ""
    metrics_dir = os.path.join(args.results_dir, "metrics")
    history_dir = os.path.join(args.results_dir, "history")
    log_dir = os.path.join(args.results_dir, "logs")
    for d in (metrics_dir, history_dir, log_dir, args.checkpoint_dir):
        ensure_dir(d)

    print(f"[{datetime.now():%H:%M:%S}] model={args.model} device=cpu threads={args.threads}")
    train_data = load_npz(os.path.join(args.data_dir, "cifar10_train.npz"))
    test_data = load_npz(os.path.join(args.data_dir, "cifar10_test.npz"))
    train_idx, val_idx = stratified_split(train_data.labels, val_ratio=args.val_ratio, seed=args.seed)
    if args.max_val_samples > 0:
        val_idx = val_idx[: args.max_val_samples]
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_data)}")

    model = build_model(args.model)
    model = model.to(memory_format=torch.channels_last)
    params = count_parameters(model)
    profile = model_profile(model)
    print(f"params={params:,} MACs={profile['macs_m']:.2f}M")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True
    )

    history = []
    start_epoch = 1
    best_val = -1.0
    best_epoch = 0
    last_path = os.path.join(args.checkpoint_dir, f"{args.model}{suffix}_last.pt")
    best_path = os.path.join(args.checkpoint_dir, f"{args.model}{suffix}_best.pt")

    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        history = ckpt.get("history", [])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        print(f"resumed from {last_path} at epoch {start_epoch - 1} (best val {best_val:.2%})")

    rng = np.random.default_rng(args.seed + 1000)
    history_path = os.path.join(history_dir, f"{args.model}{suffix}.json")
    run_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        lr = cosine_learning_rate(epoch - 1, args.epochs, args.lr, args.min_lr, args.warmup_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        epoch_start = time.perf_counter()
        train_stats = train_one_epoch(
            model,
            train_data.images,
            train_data.labels,
            train_idx,
            optimizer,
            args.batch_size,
            epoch=epoch,
            label_smoothing=args.label_smoothing,
            cutout_size=args.cutout,
            rng=rng,
            max_batches=args.max_train_batches or None,
        )
        val_metrics, _, _ = evaluate_model(
            model,
            train_data.images,
            train_data.labels,
            val_idx,
            batch_size=512,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        record: Dict[str, float | int] = {
            "epoch": epoch,
            "lr": round(lr, 8),
            "train_loss": round(train_stats["train_loss"], 6),
            "train_accuracy": round(train_stats["train_accuracy"], 6),
            "val_loss": round(float(val_metrics["loss"]), 6),
            "val_accuracy": round(float(val_metrics["accuracy"]), 6),
            "val_macro_f1": round(float(val_metrics["macro_f1"]), 6),
            "train_seconds": round(train_stats["train_seconds"], 2),
            "epoch_seconds": round(epoch_seconds, 2),
        }
        history.append(record)
        save_json(history, history_path)

        improved = float(val_metrics["accuracy"]) > best_val
        if improved:
            best_val = float(val_metrics["accuracy"])
            best_epoch = epoch
            torch.save(
                {
                    "model": args.model,
                    "spec": MODEL_SPECS[args.model],
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": best_val,
                    "args": vars(args),
                },
                best_path,
            )
        torch.save(
            {
                "model": args.model,
                "spec": MODEL_SPECS[args.model],
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "best_epoch": best_epoch,
                "history": history,
                "args": vars(args),
            },
            last_path,
        )
        print(
            f"epoch {epoch:3d}/{args.epochs} lr={lr:.4f} "
            f"train_loss={record['train_loss']:.4f} train_acc={record['train_accuracy']:.2%} "
            f"val_loss={record['val_loss']:.4f} val_acc={record['val_accuracy']:.2%} "
            f"({epoch_seconds:.1f}s){' *' if improved else ''}",
            flush=True,
        )

    train_seconds = time.perf_counter() - run_start
    print(f"training finished in {train_seconds / 60:.1f} min; best val={best_val:.2%} @ epoch {best_epoch}")

    # Final test evaluation with the best validation checkpoint.
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics, logits, labels = evaluate_model(
        model, test_data.images, test_data.labels, np.arange(len(test_data)), batch_size=512, return_predictions=True
    )
    np.savez_compressed(
        os.path.join(metrics_dir, f"{args.model}{suffix}_test_logits.npz"),
        logits=logits,
        labels=labels,
    )
    test_metrics = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in test_metrics.items()}
    summary = {
        "model": args.model,
        "label": MODEL_LABELS[args.model],
        "spec": MODEL_SPECS[args.model],
        "params": int(params),
        "macs": int(profile["macs"]),
        "flops": int(profile["flops"]),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val, 6),
        "test": test_metrics,
        "train_seconds": round(train_seconds, 2),
        "args": vars(args),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(summary, os.path.join(metrics_dir, f"{args.model}{suffix}_summary.json"))
    save_json(history, history_path)
    print(
        f"TEST accuracy={test_metrics['accuracy']:.2%} loss={test_metrics['loss']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} ece={test_metrics['ece']:.4f}"
    )


if __name__ == "__main__":
    main()