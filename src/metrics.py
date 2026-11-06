"""Classification metrics implemented with NumPy only."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(np.int64), y_pred.astype(np.int64)):
        cm[t, p] += 1
    return cm


def per_class_metrics(cm: np.ndarray) -> list[dict]:
    rows = []
    for c in range(cm.shape[0]):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "class_id": c,
                "support": int(cm[c, :].sum()),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return rows


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(confidences)) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_metrics(logits: np.ndarray, y_true: np.ndarray, num_classes: int = 10) -> Dict:
    """Compute top-1/top-5, macro F1, per-class metrics, ECE and confusion matrix."""
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[0] != y_true.shape[0]:
        raise ValueError("logits and labels must have matching first dimension")

    # Stable softmax for probabilities / NLL.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)

    top1 = float((preds == y_true).mean())
    k = min(5, num_classes)
    top5_idx = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    top5 = float(np.any(top5_idx == y_true[:, None], axis=1).mean())
    nll = float(-np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, None)).mean())

    cm = confusion_matrix(y_true, preds, num_classes)
    per_class = per_class_metrics(cm)
    macro_f1 = float(np.mean([row["f1"] for row in per_class]))
    ece = expected_calibration_error(probs, y_true)

    return {
        "accuracy": top1,
        "top5_accuracy": top5,
        "nll": nll,
        "macro_f1": macro_f1,
        "ece": ece,
        "num_samples": int(len(y_true)),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }