"""Analytical parameter and multiply-accumulate (MAC) counting.

Only Conv2d and Linear layers contribute a non-trivial number of operations for
these small ResNets, so a forward hook over those modules is sufficient and
avoids an external dependency such as fvcore/thop.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class FlopCounter:
    def __init__(self) -> None:
        self.macs = 0
        self._handles = []

    def _conv_hook(self, module: nn.Conv2d, inputs, output) -> None:
        out = output
        kernel_ops = module.kernel_size[0] * module.kernel_size[1]
        # input channels per output channel (groups split the input)
        in_per_group = module.in_channels // module.groups
        self.macs += int(out.shape[0] * out.shape[1] * out.shape[2] * out.shape[3] * kernel_ops * in_per_group)

    def _linear_hook(self, module: nn.Linear, inputs, output) -> None:
        self.macs += int(output.shape[0] * module.in_features * module.out_features)

    def attach(self, model: nn.Module) -> None:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                self._handles.append(m.register_forward_hook(self._conv_hook))
            elif isinstance(m, nn.Linear):
                self._handles.append(m.register_forward_hook(self._linear_hook))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


@torch.inference_mode()
def count_macs(model: nn.Module, input_shape=(1, 3, 32, 32)) -> int:
    model.eval()
    counter = FlopCounter()
    counter.attach(model)
    x = torch.randn(*input_shape)
    model(x)
    counter.detach()
    return counter.macs


def model_profile(model: nn.Module, input_shape=(1, 3, 32, 32)) -> Dict[str, int | float]:
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs = count_macs(model, input_shape)
    return {
        "params": int(params),
        "macs": int(macs),
        "flops": int(2 * macs),
        "params_m": round(params / 1e6, 4),
        "macs_m": round(macs / 1e6, 4),
    }