"""Fast smoke tests: model shapes / parameter counts, corruption, Grad-CAM.

Run with:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.corruptions import CORRUPTIONS, apply_corruption  # noqa: E402
from src.gradcam import GradCAM  # noqa: E402
from src.models import MODEL_SPECS, build_model, count_parameters  # noqa: E402

EXPECTED_PARAMS = {
    "resnet20": 272474,
    "resnet20_ds": 40522,
    "resnet20_se": 277082,
    "resnet20_gelu": 272474,
    "resnet20_gn": 272474,
    "resnet20_ds_se": 45130,
}


class ModelSmokeTest(unittest.TestCase):
    def test_forward_shape_and_parameters(self):
        x = torch.randn(2, 3, 32, 32)
        for name in MODEL_SPECS:
            with self.subTest(model=name):
                model = build_model(name).eval()
                with torch.no_grad():
                    y = model(x)
                self.assertEqual(tuple(y.shape), (2, 10))
                self.assertEqual(count_parameters(model), EXPECTED_PARAMS[name])

    def test_depthwise_separable_reduces_parameters(self):
        self.assertLess(EXPECTED_PARAMS["resnet20_ds"], EXPECTED_PARAMS["resnet20"] // 4)
        self.assertLess(EXPECTED_PARAMS["resnet20_ds_se"], EXPECTED_PARAMS["resnet20_ds"] * 2)


class CorruptionSmokeTest(unittest.TestCase):
    def test_every_corruption_keeps_shape_and_dtype(self):
        rng = np.random.default_rng(0)
        images = torch.from_numpy(
            rng.integers(0, 256, size=(8, 32, 32, 3), dtype=np.uint8)
        )
        for name in CORRUPTIONS:
            with self.subTest(corruption=name):
                out = apply_corruption(images, name, severity=3, seed=7)
                self.assertEqual(tuple(out.shape), tuple(images.shape))
                self.assertEqual(out.dtype, torch.uint8)


class GradCamSmokeTest(unittest.TestCase):
    def test_cam_shape(self):
        model = build_model("resnet20").eval()
        cam = GradCAM(model)
        image = torch.rand(1, 3, 32, 32)
        heatmap, pred, probs = cam(image)
        self.assertEqual(tuple(heatmap.shape), (32, 32))
        self.assertGreaterEqual(pred, 0)
        self.assertLess(pred, 10)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()