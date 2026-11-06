"""Synthetic image corruptions used for the robustness / deployment study.

All functions accept a uint8 tensor in NHWC layout and return the same shape
and dtype.  Severity is an integer from 1 (mild) to 5 (severe).  The
corruptions emulate visibility and sensor degradation in outdoor vision:
fog, low light, motion blur, sensor noise and JPEG compression.
"""
from __future__ import annotations

import io
import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CORRUPTIONS: Tuple[str, ...] = ("fog", "low_light", "motion_blur", "gaussian_noise", "jpeg")
CORRUPTION_LABELS: Dict[str, str] = {
    "fog": "Fog",
    "low_light": "Low light",
    "motion_blur": "Motion blur",
    "gaussian_noise": "Gaussian noise",
    "jpeg": "JPEG compression",
}


def _validate(images: torch.Tensor, severity: int) -> None:
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("images must be uint8 NHWC [B, 32, 32, 3]")
    if not 1 <= severity <= 5:
        raise ValueError("severity must be in [1, 5]")


def _generator(seed: int | None) -> torch.Generator:
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(int(seed))
    return g


def fog(images: torch.Tensor, severity: int, seed: int | None = None) -> torch.Tensor:
    """Atmospheric scattering with a bright, slightly blue veil."""
    _validate(images, severity)
    g = _generator(seed)
    x = images.float()
    base = 0.10 + 0.105 * severity
    alpha = torch.clamp(
        torch.tensor(base) + (torch.rand(x.shape[0], 1, 1, 1, generator=g) - 0.5) * 0.08,
        0.0,
        0.9,
    )
    color = torch.tensor([232.0, 236.0, 242.0]).view(1, 1, 1, 3)
    out = x * (1.0 - alpha) + color * alpha
    return out.clamp(0, 255).round().to(torch.uint8)


def low_light(images: torch.Tensor, severity: int, seed: int | None = None) -> torch.Tensor:
    """Gamma darkening plus read noise, as in a dim outdoor scene."""
    _validate(images, severity)
    g = _generator(seed)
    x = images.float() / 255.0
    gamma = 1.0 + 0.24 * severity
    x = torch.pow(x.clamp_min(0), gamma) * 255.0
    sigma = 1.5 + 2.0 * severity
    noise = torch.randn(x.shape, generator=g) * sigma
    return (x + noise).clamp(0, 255).round().to(torch.uint8)


def _blur_kernel(length: int, angle_deg: int) -> torch.Tensor:
    """Return a normalized 2-D motion-blur kernel."""
    length = max(3, length | 1)  # odd
    kernel = torch.zeros(length, length, dtype=torch.float32)
    center = length // 2
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    for t in torch.linspace(-center, center, length):
        x = int(round(center + float(t) * dx))
        y = int(round(center + float(t) * dy))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    if kernel.sum() == 0:
        kernel[center, center] = 1.0
    return kernel / kernel.sum()


def motion_blur(images: torch.Tensor, severity: int, seed: int | None = None) -> torch.Tensor:
    """Directional blur with a per-image angle from four canonical directions."""
    _validate(images, severity)
    g = _generator(seed)
    b = images.shape[0]
    k = 2 * severity + 1
    angles = torch.tensor([0, 45, 90, 135])
    base = torch.stack([_blur_kernel(k, int(a)) for a in angles])  # [4, k, k]
    choice = torch.randint(0, 4, (b,), generator=g)
    kernels = base[choice]  # [B, k, k]
    kernels = kernels.repeat_interleave(3, dim=0).unsqueeze(1)  # [3B, 1, k, k]
    x = images.permute(0, 3, 1, 2).float().reshape(1, 3 * b, 32, 32)
    pad = k // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    y = F.conv2d(x, kernels, groups=3 * b)
    y = y.reshape(b, 3, 32, 32).permute(0, 2, 3, 1)
    return y.clamp(0, 255).round().to(torch.uint8)


def gaussian_noise(images: torch.Tensor, severity: int, seed: int | None = None) -> torch.Tensor:
    """Additive white Gaussian noise (sensor / low-light noise)."""
    _validate(images, severity)
    g = _generator(seed)
    sigma = 6.0 * severity
    noise = torch.randn(images.shape, generator=g) * sigma
    return (images.float() + noise).clamp(0, 255).round().to(torch.uint8)


def jpeg(images: torch.Tensor, severity: int, seed: int | None = None) -> torch.Tensor:
    """JPEG compression artefacts (quality decreases with severity)."""
    _validate(images, severity)
    quality = max(5, 62 - 10 * severity)
    outputs = []
    for img in images:
        arr = img.numpy().astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(arr, mode="RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        decoded = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)
        outputs.append(torch.from_numpy(decoded.copy()))
    return torch.stack(outputs, dim=0)


CORRUPTION_FUNCS = {
    "fog": fog,
    "low_light": low_light,
    "motion_blur": motion_blur,
    "gaussian_noise": gaussian_noise,
    "jpeg": jpeg,
}


def apply_corruption(
    images: torch.Tensor, name: str, severity: int, seed: int | None = None
) -> torch.Tensor:
    if name not in CORRUPTION_FUNCS:
        raise KeyError(f"Unknown corruption '{name}', expected one of {CORRUPTIONS}")
    return CORRUPTION_FUNCS[name](images, severity, seed=seed)


def random_corruption_batch(
    images: torch.Tensor,
    rng: np.random.Generator,
    *,
    max_severity: int = 5,
    corruption_prob: float = 0.7,
) -> torch.Tensor:
    """Apply one randomly selected corruption to a whole training batch.

    Batch-level randomization is used deliberately: it is cheap, keeps the
    augmentation pipeline deterministic under a given seed, and over an epoch
    exposes the model to all corruption types and severities.
    """
    if float(rng.random()) >= corruption_prob:
        return images
    name = str(rng.choice(CORRUPTIONS))
    severity = int(rng.integers(1, max_severity + 1))
    return apply_corruption(images, name, severity, seed=int(rng.integers(0, 2**31 - 1)))