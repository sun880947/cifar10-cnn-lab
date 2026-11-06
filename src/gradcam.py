"""Minimal Grad-CAM implementation for the CIFAR ResNet variants."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or model.layer3[-1]
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = []
        self._register()

    def _register(self) -> None:
        self._handles.append(self.target_layer.register_forward_hook(self._forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __call__(
        self, image: torch.Tensor, class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """Return (cam in [0,1] HxW, predicted class, softmax probabilities).

        Args:
            image: normalized NCHW tensor with batch size 1.
            class_idx: target class; defaults to the model prediction.
        """
        if image.shape[0] != 1:
            raise ValueError("Grad-CAM expects a batch of one image")
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred = int(probs.argmax())
        target = pred if class_idx is None else int(class_idx)
        score = logits[0, target]
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        if was_training:
            self.model.train()
        return cam, pred, probs