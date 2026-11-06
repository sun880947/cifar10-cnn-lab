"""Data loading and augmentation for the pre-processed CIFAR-10 arrays.

The repository stores CIFAR-10 as two compressed ``.npz`` files with keys
``images`` (uint8, NHWC) and ``labels`` (int64).  This avoids a heavyweight
dataset dependency at training time and makes the experiments reproducible on
machines without network access.  ``scripts/prepare_data.py`` creates these
files from either torchvision or the HuggingFace parquet mirror.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

CIFAR10_CLASSES: Tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CIFAR10_CLASSES_ZH: Tuple[str, ...] = (
    "飞机",
    "汽车",
    "鸟",
    "猫",
    "鹿",
    "狗",
    "青蛙",
    "马",
    "船",
    "卡车",
)

# CIFAR-10 channel statistics.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


@dataclass
class CIFAR10Arrays:
    images: torch.Tensor  # uint8 [N, 32, 32, 3]
    labels: torch.Tensor  # int64 [N]

    def __len__(self) -> int:
        return int(self.labels.numel())


def load_npz(path: str) -> CIFAR10Arrays:
    with np.load(path) as data:
        images = torch.from_numpy(data["images"].astype(np.uint8, copy=False))
        labels = torch.from_numpy(data["labels"].astype(np.int64, copy=False))
    if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError(f"Unexpected image shape in {path}: {tuple(images.shape)}")
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError(f"Image/label count mismatch in {path}")
    return CIFAR10Arrays(images, labels)


def stratified_split(
    labels: torch.Tensor,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_classes: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic class-balanced train/validation indices."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")
    y = labels.numpy()
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for c in range(num_classes):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_ratio))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.sort(np.concatenate(train_parts))
    val_idx = np.sort(np.concatenate(val_parts))
    return train_idx, val_idx


def augment_batch(images: torch.Tensor, cutout_size: int = 0) -> torch.Tensor:
    """Random crop with 4-pixel zero padding and random horizontal flip.

    Args:
        images: uint8 tensor [B, 32, 32, 3].
        cutout_size: if > 0, apply cutout to half of the samples.
    """
    if images.ndim != 4:
        raise ValueError("images must be [B, H, W, C]")
    # NHWC: pad width and height only (the last two dims are W and C).
    padded = F.pad(images, (0, 0, 4, 4, 4, 4), mode="constant", value=0)
    batch = padded.shape[0]
    offsets_y = torch.randint(0, 9, (batch,))
    offsets_x = torch.randint(0, 9, (batch,))
    crops = [
        padded[i, offsets_y[i] : offsets_y[i] + 32, offsets_x[i] : offsets_x[i] + 32]
        for i in range(batch)
    ]
    out = torch.stack(crops, dim=0)
    flip_mask = torch.rand(batch) < 0.5
    out[flip_mask] = out[flip_mask].flip(-1)

    if cutout_size > 0:
        apply = torch.rand(batch) < 0.5
        for i in torch.nonzero(apply, as_tuple=False).flatten().tolist():
            cy = int(torch.randint(cutout_size // 2, 32 - cutout_size // 2, (1,)).item())
            cx = int(torch.randint(cutout_size // 2, 32 - cutout_size // 2, (1,)).item())
            y0 = max(0, cy - cutout_size // 2)
            y1 = min(32, cy + cutout_size // 2)
            x0 = max(0, cx - cutout_size // 2)
            x1 = min(32, cx + cutout_size // 2)
            out[i, y0:y1, x0:x1, :] = 0
    return out


def prepare_input(images_uint8: torch.Tensor) -> torch.Tensor:
    """Normalize NHWC uint8 images and return channels-last NCHW float."""
    x = images_uint8.float().div_(255.0)
    mean = torch.tensor(MEAN, dtype=x.dtype).view(1, 1, 1, 3)
    std = torch.tensor(STD, dtype=x.dtype).view(1, 1, 1, 3)
    x = (x - mean) / std
    x = x.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    return x


def iterate_batches(
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int] | np.ndarray,
    batch_size: int,
    *,
    train: bool = False,
    shuffle: bool = False,
    normalize: bool = True,
    cutout_size: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield batches without a DataLoader (all data already lives in RAM)."""
    idx = np.asarray(indices, dtype=np.int64)
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        idx = idx.copy()
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch_idx = torch.from_numpy(idx[start : start + batch_size])
        xb = images[batch_idx]
        yb = labels[batch_idx]
        if train:
            xb = augment_batch(xb, cutout_size=cutout_size)
        if normalize:
            xb = prepare_input(xb)
        yield xb, yb


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Convert normalized NCHW images back to [0, 1] for visualization."""
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (images.cpu() * std + mean).clamp(0, 1)