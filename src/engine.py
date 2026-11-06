"""Training and evaluation loops."""
from __future__ import annotations

import math
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import iterate_batches
from .metrics import compute_metrics


def cosine_learning_rate(
    epoch: int, epochs: int, base_lr: float, min_lr: float = 0.0, warmup_epochs: int = 0
) -> float:
    """Cosine decay with an optional linear warm-up (epoch is zero-based)."""
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    train_idx: np.ndarray,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    *,
    epoch: int,
    label_smoothing: float = 0.1,
    cutout_size: int = 0,
    rng: Optional[np.random.Generator] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    start = time.perf_counter()
    for step, (xb, yb) in enumerate(
        iterate_batches(
            images,
            labels,
            train_idx,
            batch_size,
            train=True,
            shuffle=True,
            cutout_size=cutout_size,
            rng=rng,
        )
    ):
        if max_batches is not None and step >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach()) * yb.numel()
        total_correct += int((logits.argmax(dim=1) == yb).sum())
        total += int(yb.numel())
    elapsed = time.perf_counter() - start
    return {
        "train_loss": total_loss / max(1, total),
        "train_accuracy": total_correct / max(1, total),
        "train_seconds": elapsed,
        "num_train": total,
    }


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int] | np.ndarray,
    batch_size: int = 512,
    *,
    num_classes: int = 10,
    return_predictions: bool = False,
) -> Tuple[Dict, Optional[np.ndarray], Optional[np.ndarray]]:
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    total = 0
    for xb, yb in iterate_batches(
        images, labels, indices, batch_size, train=False, shuffle=False
    ):
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, reduction="sum")
        total_loss += float(loss)
        total += int(yb.numel())
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(yb.numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(logits_np, labels_np, num_classes=num_classes)
    metrics["loss"] = total_loss / max(1, total)
    if return_predictions:
        return metrics, logits_np, labels_np
    return metrics, None, None


def collect_predictions(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int] | np.ndarray,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    metrics, logits, y = evaluate_model(
        model, images, labels, indices, batch_size, return_predictions=True
    )
    return logits, y