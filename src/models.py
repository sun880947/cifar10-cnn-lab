"""CIFAR-10 ResNet models and controlled structural variants.

The baseline is a CIFAR-style ResNet-20 (He et al., 2016): a 3x3 stem, three
stages of residual blocks, and a global average pooling classifier.  Six
variants are exposed through ``build_model`` so that every experiment uses the
same training recipe and data split.

Variants
--------
resnet20       : standard 3x3 convolutions, ReLU, BatchNorm (baseline)
resnet20_ds    : 3x3 convolutions replaced by depthwise separable convolutions
resnet20_se    : squeeze-and-excitation channel attention in each block
resnet20_gelu  : GELU instead of ReLU
resnet20_gn    : GroupNorm (8 groups) instead of BatchNorm
resnet20_ds_se : depthwise separable convolutions + SE attention
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 convolution followed by a pointwise 1x1 convolution."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, use_ds: bool = False) -> nn.Module:
    if use_ds:
        return DepthwiseSeparableConv(in_channels, out_channels, stride)
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def make_norm(norm: str, channels: int, num_groups: int = 8) -> nn.Module:
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "gn":
        groups = min(num_groups, channels)
        if channels % groups != 0:
            raise ValueError(f"GroupNorm needs channels % groups == 0, got {channels} and {groups}")
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unknown norm: {norm}")


def make_act(act: str) -> nn.Module:
    act = act.lower()
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "gelu":
        return nn.GELU(approximate="none")
    if act == "mish":
        return nn.Mish(inplace=False)
    raise ValueError(f"Unknown activation: {act}")


class SqueezeExcitation(nn.Module):
    """Channel attention: global average pool -> bottleneck MLP -> sigmoid gate."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class BasicBlock(nn.Module):
    """Pre-activation-free residual block used in the original CIFAR ResNet."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm: str = "bn",
        act: str = "relu",
        use_se: bool = False,
        use_ds: bool = False,
        se_reduction: int = 8,
        gn_groups: int = 8,
    ):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride, use_ds=use_ds)
        self.bn1 = make_norm(norm, out_channels, gn_groups)
        self.act1 = make_act(act)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1, use_ds=use_ds)
        self.bn2 = make_norm(norm, out_channels, gn_groups)
        self.se = SqueezeExcitation(out_channels, se_reduction) if use_se else nn.Identity()
        self.act2 = make_act(act)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=stride),
                make_norm(norm, out_channels, gn_groups),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + self.shortcut(x)
        return self.act2(out)


class CifarResNet(nn.Module):
    """ResNet for 32x32 CIFAR images with configurable depth and components.

    Args:
        depth: total number of parameterised layers (6n + 2). 20, 32 and 44
            correspond to the standard CIFAR ResNet family.
        width: base channel width (16 in the standard CIFAR ResNet).
    """

    def __init__(
        self,
        depth: int = 20,
        width: int = 16,
        num_classes: int = 10,
        norm: str = "bn",
        act: str = "relu",
        use_se: bool = False,
        use_ds: bool = False,
        se_reduction: int = 8,
        gn_groups: int = 8,
    ):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("depth must satisfy depth = 6n + 2 (e.g. 20, 32, 44)")
        blocks_per_stage = (depth - 2) // 6
        if blocks_per_stage < 1:
            raise ValueError("depth is too small")

        self.in_channels = width
        self.stem = nn.Sequential(
            conv3x3(3, width, stride=1, use_ds=False),
            make_norm(norm, width, gn_groups),
            make_act(act),
        )
        self.layer1 = self._make_layer(
            width, blocks_per_stage, stride=1, norm=norm, act=act,
            use_se=use_se, use_ds=use_ds, se_reduction=se_reduction, gn_groups=gn_groups,
        )
        self.layer2 = self._make_layer(
            width * 2, blocks_per_stage, stride=2, norm=norm, act=act,
            use_se=use_se, use_ds=use_ds, se_reduction=se_reduction, gn_groups=gn_groups,
        )
        self.layer3 = self._make_layer(
            width * 4, blocks_per_stage, stride=2, norm=norm, act=act,
            use_se=use_se, use_ds=use_ds, se_reduction=se_reduction, gn_groups=gn_groups,
        )
        self.num_features = width * 4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.num_features, num_classes)
        self._initialize_weights()

    def _make_layer(
        self,
        out_channels: int,
        blocks: int,
        stride: int,
        norm: str,
        act: str,
        use_se: bool,
        use_ds: bool,
        se_reduction: int,
        gn_groups: int,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        for i in range(blocks):
            layers.append(
                BasicBlock(
                    self.in_channels,
                    out_channels,
                    stride=stride if i == 0 else 1,
                    norm=norm,
                    act=act,
                    use_se=use_se,
                    use_ds=use_ds,
                    se_reduction=se_reduction,
                    gn_groups=gn_groups,
                )
            )
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
MODEL_SPECS: Dict[str, dict] = {
    "resnet20": dict(depth=20, width=16, norm="bn", act="relu", use_se=False, use_ds=False),
    "resnet20_ds": dict(depth=20, width=16, norm="bn", act="relu", use_se=False, use_ds=True),
    "resnet20_se": dict(depth=20, width=16, norm="bn", act="relu", use_se=True, use_ds=False),
    "resnet20_gelu": dict(depth=20, width=16, norm="bn", act="gelu", use_se=False, use_ds=False),
    "resnet20_gn": dict(depth=20, width=16, norm="gn", act="relu", use_se=False, use_ds=False),
    "resnet20_ds_se": dict(depth=20, width=16, norm="bn", act="relu", use_se=True, use_ds=True),
}

MODEL_LABELS: Dict[str, str] = {
    "resnet20": "R20 (baseline)",
    "resnet20_ds": "R20-DS",
    "resnet20_se": "R20-SE",
    "resnet20_gelu": "R20-GELU",
    "resnet20_gn": "R20-GN",
    "resnet20_ds_se": "R20-DS-SE",
}


def build_model(name: str, num_classes: int = 10) -> CifarResNet:
    if name not in MODEL_SPECS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_SPECS)}")
    spec = dict(MODEL_SPECS[name])
    return CifarResNet(num_classes=num_classes, **spec)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> str:
    total = count_parameters(model)
    lines = [
        f"{type(model).__name__}: {total:,} trainable parameters",
        f"layers: {len(list(model.modules()))}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    for name in MODEL_SPECS:
        m = build_model(name)
        print(f"{name:18s} params={count_parameters(m):,}")