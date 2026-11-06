"""Evaluate all trained variants on clean and corrupted CIFAR-10 test data.

The corrupted tensors are generated once per (corruption, severity) and shared by
all models, which keeps the comparison paired and avoids regenerating JPEG /
motion-blur images six times.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.corruptions import CORRUPTIONS, apply_corruption  # noqa: E402
from src.data import load_npz  # noqa: E402
from src.engine import evaluate_model  # noqa: E402
from src.models import MODEL_LABELS, MODEL_SPECS, build_model  # noqa: E402
from src.utils import configure_threads, ensure_dir, save_json  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=sorted(MODEL_SPECS))
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--suffix", default="")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--severities", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    args = p.parse_args()
    configure_threads(args.threads)

    test = load_npz(os.path.join(args.data_dir, "cifar10_test.npz"))
    indices = np.arange(args.max_samples or len(test.labels))

    models = {}
    results: Dict[str, dict] = {}
    for name in args.models:
        checkpoint = os.path.join(args.checkpoint_dir, f"{name}{args.suffix}_best.pt")
        if not os.path.exists(checkpoint):
            print(f"SKIP {name}: {checkpoint} not found")
            continue
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_model(name)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models[name] = model
        results[name] = {
            "label": MODEL_LABELS[name],
            "checkpoint": checkpoint,
            "corruptions": {},
            "severities": args.severities,
        }
        print(f"[{name}] checkpoint epoch {ckpt.get('epoch')} val={ckpt.get('val_accuracy', 0):.2%}", flush=True)

    rows: List[dict] = []
    # Clean pass once per model.
    for name, model in models.items():
        metrics, _, _ = evaluate_model(
            model, test.images, test.labels, indices, batch_size=args.batch_size
        )
        results[name]["clean"] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                  for k, v in metrics.items()}
        results[name]["clean_accuracy"] = float(metrics["accuracy"])
        rows.append({"model": name, "label": MODEL_LABELS[name], "corruption": "clean",
                     "severity": 0, "accuracy": results[name]["clean_accuracy"]})
        print(f"  clean={results[name]['clean_accuracy']:.2%}", flush=True)

    # Corrupted pass: generate once, evaluate all models.
    for corruption in CORRUPTIONS:
        for severity in args.severities:
            corrupted = apply_corruption(test.images, corruption, int(severity), seed=1234 + int(severity))
            for name, model in models.items():
                metrics, _, _ = evaluate_model(
                    model, corrupted, test.labels, indices, batch_size=args.batch_size
                )
                acc = float(metrics["accuracy"])
                results[name]["corruptions"].setdefault(corruption, {})[int(severity)] = acc
                rows.append({"model": name, "label": MODEL_LABELS[name], "corruption": corruption,
                             "severity": int(severity), "accuracy": acc})
            print(f"  {corruption:14s} s={severity} " +
                  " ".join(f"{n}={results[n]['corruptions'][corruption][int(severity)]:.2%}" for n in models),
                  flush=True)

    for name in models:
        suite = results[name]
        all_accs = [suite["corruptions"][c][s] for c in CORRUPTIONS for s in args.severities]
        suite["mean_accuracy"] = float(np.mean(all_accs)) if all_accs else 0.0
        suite["worst_accuracy"] = float(np.min(all_accs)) if all_accs else 0.0
        suite["mean_drop"] = suite["clean_accuracy"] - suite["mean_accuracy"]
        print(f"[{name}] clean={suite['clean_accuracy']:.2%} mean={suite['mean_accuracy']:.2%} "
              f"worst={suite['worst_accuracy']:.2%} drop={suite['mean_drop']:.2%}", flush=True)

    out_dir = os.path.join(args.results_dir, "metrics")
    ensure_dir(out_dir)
    suffix = f"_{args.suffix}" if args.suffix else ""
    json_path = os.path.join(out_dir, f"robustness{suffix}.json")
    save_json(
        {
            "corruptions": list(CORRUPTIONS),
            "severities": args.severities,
            "max_samples": args.max_samples or "all",
            "models": results,
        },
        json_path,
    )
    csv_path = os.path.join(out_dir, f"robustness{suffix}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "label", "corruption", "severity", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.join(out_dir, f"robustness{suffix}_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "label", "clean_accuracy", "mean_corruption_accuracy",
                         "worst_case_accuracy", "mean_drop", "relative_drop_pct"])
        for name, suite in results.items():
            writer.writerow([
                name, suite["label"], round(suite["clean_accuracy"], 6),
                round(suite["mean_accuracy"], 6), round(suite["worst_accuracy"], 6),
                round(suite["mean_drop"], 6),
                round(100.0 * suite["mean_drop"] / max(1e-9, suite["clean_accuracy"]), 2),
            ])
    print(f"saved {json_path}\nsaved {csv_path}\nsaved {summary_path}")


if __name__ == "__main__":
    main()