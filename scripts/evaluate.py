"""Evaluate a saved checkpoint on the clean CIFAR-10 test set."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import CIFAR10_CLASSES, load_npz  # noqa: E402
from src.engine import evaluate_model  # noqa: E402
from src.models import MODEL_LABELS, build_model  # noqa: E402
from src.utils import configure_threads, ensure_dir, save_json  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet20")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", default="")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()
    configure_threads(args.threads)

    checkpoint = args.checkpoint or os.path.join("checkpoints", f"{args.model}_best.pt")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(args.model)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test = load_npz(os.path.join(args.data_dir, "cifar10_test.npz"))
    metrics, logits, labels = evaluate_model(
        model,
        test.images,
        test.labels,
        np.arange(len(test)),
        batch_size=args.batch_size,
        return_predictions=True,
    )
    clean = {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics.items()}
    result = {
        "model": args.model,
        "label": MODEL_LABELS.get(args.model, args.model),
        "checkpoint": checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "test": clean,
    }
    print(
        f"{args.model}: acc={clean['accuracy']:.2%} nll={clean['nll']:.4f} "
        f"macro_f1={clean['macro_f1']:.4f} ece={clean['ece']:.4f}"
    )
    for m in clean["per_class"]:
        name = CIFAR10_CLASSES[m["class_id"]]
        print(f"  {name:12s} precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")
    if args.output:
        ensure_dir(os.path.dirname(os.path.abspath(args.output)))
        save_json(result, args.output)
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()