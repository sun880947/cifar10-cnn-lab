"""Corruption robustness evaluation shared by scripts and notebooks."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from .corruptions import CORRUPTIONS, apply_corruption
from .engine import evaluate_model


@torch.inference_mode()
def evaluate_corruption_suite(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    severities: Iterable[int] = (1, 2, 3, 4, 5),
    batch_size: int = 512,
    max_samples: Optional[int] = None,
    seed: int = 1234,
    verbose: bool = False,
) -> Dict:
    """Evaluate a model on clean and corrupted test data.

    Returns a nested dict with per-corruption/per-severity accuracy and the
    aggregate mean / worst-case accuracy.  The same corruption seed is used for
    every model so comparisons are paired.
    """
    if max_samples is not None and max_samples < len(labels):
        indices = np.arange(max_samples)
    else:
        indices = np.arange(len(labels))
    clean, _, _ = evaluate_model(model, images, labels, indices, batch_size=batch_size)
    result: Dict = {
        "clean": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in clean.items()},
        "corruptions": {},
        "severities": list(severities),
    }
    all_accs: List[float] = []
    for name in CORRUPTIONS:
        per_severity = {}
        for severity in severities:
            corrupted = apply_corruption(images, name, int(severity), seed=seed + int(severity))
            metrics, _, _ = evaluate_model(
                model, corrupted, labels, indices, batch_size=batch_size
            )
            acc = float(metrics["accuracy"])
            per_severity[int(severity)] = acc
            all_accs.append(acc)
            if verbose:
                print(f"  {name:14s} s={severity} acc={acc:.2%}", flush=True)
        result["corruptions"][name] = per_severity
    result["mean_accuracy"] = float(np.mean(all_accs)) if all_accs else 0.0
    result["worst_accuracy"] = float(np.min(all_accs)) if all_accs else 0.0
    result["clean_accuracy"] = float(clean["accuracy"])
    result["mean_drop"] = result["clean_accuracy"] - result["mean_accuracy"]
    return result