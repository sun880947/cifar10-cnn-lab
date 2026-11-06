"""Flask demo: CIFAR-10 recognition under simulated low-visibility conditions.

Run from the repository root:

    python demo/app.py --port 5000
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import uuid

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.corruptions import CORRUPTIONS, apply_corruption  # noqa: E402
from src.data import CIFAR10_CLASSES, CIFAR10_CLASSES_ZH, denormalize, prepare_input  # noqa: E402
from src.gradcam import GradCAM  # noqa: E402
from src.models import MODEL_LABELS, build_model  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
os.makedirs(os.path.join(ROOT, "demo", "static", "results"), exist_ok=True)
_MODEL_CACHE = {}


def get_model(name: str):
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    path = os.path.join(ROOT, "checkpoints", f"{name}_best.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(name)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _MODEL_CACHE[name] = model
    return model


def center_crop_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def cam_overlay(image: Image.Image, cam: np.ndarray, size: int = 160) -> Image.Image:
    base = np.asarray(image.resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    cam_np = np.asarray(cam_img, dtype=np.float32) / 255.0
    try:
        import matplotlib

        cmap = matplotlib.colormaps["jet"]
        heat = cmap(cam_np)[..., :3]
    except Exception:
        heat = np.stack([cam_np, 1.0 - cam_np, np.zeros_like(cam_np)], axis=-1)
    overlay = np.clip(0.55 * base + 0.45 * heat, 0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8))


@app.route("/")
def index():
    return render_template(
        "index.html",
        corruptions=CORRUPTIONS,
        labels=MODEL_LABELS,
        classes_zh=CIFAR10_CLASSES_ZH,
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/sample/<int:index>")
def sample(index: int):
    """Serve one bundled demo image, used by the `?demo=1` auto-run."""
    path = os.path.join(ROOT, "demo", "samples", f"sample_{index}.png")
    if not os.path.exists(path):
        return jsonify({"error": "sample not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "请选择一张图片"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "文件名为空"}), 400
    model_name = request.form.get("model", "resnet20_ds_se")
    corruption = request.form.get("corruption", "none")
    severity = int(request.form.get("severity", "3"))
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "无法读取该图片，请上传 JPG/PNG"}), 400
    image = center_crop_square(image).resize((32, 32), Image.BICUBIC)
    raw = np.asarray(image, dtype=np.uint8)[None, ...].copy()
    if corruption != "none":
        raw = apply_corruption(torch.from_numpy(raw), corruption, max(1, min(5, severity)), seed=2026).numpy()

    try:
        model = get_model(model_name)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    tensor = prepare_input(torch.from_numpy(raw))
    cam, pred, probs = GradCAM(model)(tensor)

    order = np.argsort(-probs)[:3]
    top3 = [
        {
            "class_id": int(i),
            "name": CIFAR10_CLASSES[int(i)],
            "name_zh": CIFAR10_CLASSES_ZH[int(i)],
            "prob": float(probs[int(i)]),
        }
        for i in order
    ]
    input_img = Image.fromarray(raw[0])
    overlay = cam_overlay(input_img, cam)
    filename = f"{uuid.uuid4().hex}.png"
    overlay.save(os.path.join(ROOT, "demo", "static", "results", filename))
    return jsonify(
        {
            "prediction": top3[0],
            "top3": top3,
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "corruption": corruption,
            "severity": severity,
            "input_image": image_to_data_url(input_img.resize((160, 160), Image.NEAREST)),
            "cam_image": image_to_data_url(overlay),
            "cam_url": f"/static/results/{filename}",
        }
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()