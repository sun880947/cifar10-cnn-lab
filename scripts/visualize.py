"""Build every figure used by the report from the recorded experiment outputs.

Run after training / robustness / fine-tuning:

    python -m scripts.visualize
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter, NullFormatter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.corruptions import CORRUPTION_LABELS, CORRUPTIONS, apply_corruption  # noqa: E402
from src.data import (  # noqa: E402
    CIFAR10_CLASSES,
    denormalize,
    load_npz,
    prepare_input,
    stratified_split,
)
from src.gradcam import GradCAM  # noqa: E402
from src.models import MODEL_LABELS, build_model  # noqa: E402

ORDER = ["resnet20", "resnet20_ds", "resnet20_se", "resnet20_gelu", "resnet20_gn", "resnet20_ds_se"]
SHORT = {
    "resnet20": "R20",
    "resnet20_ds": "R20-DS",
    "resnet20_se": "R20-SE",
    "resnet20_gelu": "R20-GELU",
    "resnet20_gn": "R20-GN",
    "resnet20_ds_se": "R20-DS-SE",
}
COLORS = {
    "resnet20": "#222222",
    "resnet20_ds": "#1f77b4",
    "resnet20_se": "#d62728",
    "resnet20_gelu": "#2ca02c",
    "resnet20_gn": "#9467bd",
    "resnet20_ds_se": "#ff7f0e",
}
CLASS_SHORT = ["airplane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def available_summaries(results_dir: str) -> Dict[str, dict]:
    out = {}
    for name in ORDER:
        path = os.path.join(results_dir, "metrics", f"{name}_summary.json")
        if os.path.exists(path):
            out[name] = load_json(path)
    return out


def fig_dataset(data_dir: str, out_dir: str) -> None:
    test = load_npz(os.path.join(data_dir, "cifar10_test.npz"))
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for c, ax in enumerate(axes.flat):
        idx = int(np.flatnonzero(test.labels.numpy() == c)[0])
        ax.imshow(test.images[idx].numpy())
        ax.set_title(f"{c}: {CIFAR10_CLASSES[c]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("CIFAR-10 test samples (one per class)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dataset_samples.png"), bbox_inches="tight")
    plt.close(fig)


def fig_corruptions(data_dir: str, out_dir: str, severity: int = 3) -> None:
    test = load_npz(os.path.join(data_dir, "cifar10_test.npz"))
    image = test.images[3:4]
    fig, axes = plt.subplots(len(CORRUPTIONS), 4, figsize=(9.2, 10.2))
    for row, name in enumerate(CORRUPTIONS):
        panels = [("clean", image)]
        for s in (1, severity, 5):
            panels.append((f"severity {s}", apply_corruption(image, name, s, seed=100 + s)))
        for col, (title, img) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img[0].numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if row == 0:
                ax.set_title(title, fontsize=9)
        axes[row, 0].set_ylabel(CORRUPTION_LABELS[name], fontsize=10)
    fig.suptitle("Synthetic low-visibility corruptions used in the robustness study", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "corruption_examples.png"), bbox_inches="tight")
    plt.close(fig)


def fig_training_curves(results_dir: str, summaries: Dict[str, dict], out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in ORDER:
        path = os.path.join(results_dir, "history", f"{name}.json")
        if not os.path.exists(path):
            continue
        hist = load_json(path)
        epochs = [h["epoch"] for h in hist]
        loss = [h["train_loss"] for h in hist]
        acc = [100.0 * h["val_accuracy"] for h in hist]
        lw = 2.0 if name == "resnet20" else 1.2
        axes[0].plot(epochs, loss, color=COLORS[name], lw=lw, label=SHORT[name])
        axes[1].plot(epochs, acc, color=COLORS[name], lw=lw, label=SHORT[name])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss (label smoothing 0.1)")
    axes[0].set_title("(a) Training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation accuracy (%)")
    axes[1].set_title("(b) Validation accuracy")
    axes[1].legend(ncol=2, loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), bbox_inches="tight")
    plt.close(fig)


def fig_cost_accuracy(summaries: Dict[str, dict], benchmark_path: str, out_dir: str) -> None:
    bench = {}
    if os.path.exists(benchmark_path):
        bench = {row["model"]: row for row in load_json(benchmark_path)["models"]}
    metrics = [
        ("params", "Parameters (M)", [0.05, 0.1, 0.2, 0.3]),
        ("macs", "MACs per image (M)", [7, 10, 20, 40]),
        ("latency_b1_median", "CPU latency, batch 1 (ms)", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, (key, label, log_ticks) in zip(axes, metrics):
        for name in ORDER:
            if name not in summaries:
                continue
            if key.startswith("latency"):
                if name not in bench:
                    continue
                x = bench[name]["latency_b1"]["latency_ms_median"]
            else:
                x = summaries[name][key] / 1e6
            y = 100.0 * summaries[name]["test"]["accuracy"]
            ax.scatter([x], [y], s=42, color=COLORS[name], zorder=3)
            ax.annotate(SHORT[name], (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7.5)
        if log_ticks:
            ax.set_xscale("log")
            ax.set_xticks(log_ticks)
            ax.get_xaxis().set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
            ax.get_xaxis().set_minor_formatter(NullFormatter())
        ax.set_xlabel(label)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"Accuracy vs {label.split(' (')[0].lower()}")
        ax.margins(x=0.18, y=0.12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_vs_cost.png"), bbox_inches="tight")
    plt.close(fig)


def fig_per_class(summaries: Dict[str, dict], out_dir: str) -> None:
    if "resnet20" not in summaries:
        return
    compare = [n for n in ("resnet20", "resnet20_ds_se") if n in summaries]
    x = np.arange(len(CIFAR10_CLASSES))
    width = 0.8 / len(compare)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, name in enumerate(compare):
        pc = summaries[name]["test"]["per_class"]
        vals = [100.0 * row["recall"] for row in pc]
        ax.bar(x + (i - (len(compare) - 1) / 2) * width, vals, width, label=SHORT[name], color=COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_SHORT, rotation=20, ha="right")
    ax.set_ylabel("Per-class recall (%)")
    ax.set_ylim(60, 100)
    ax.legend()
    ax.set_title("Per-class test accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_class_accuracy.png"), bbox_inches="tight")
    plt.close(fig)


def fig_confusion(results_dir: str, summaries: Dict[str, dict], out_dir: str) -> None:
    models = [n for n in ("resnet20", "resnet20_ds_se") if n in summaries]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.6))
    if len(models) == 1:
        axes = [axes]
    for ax, name in zip(axes, models):
        path = os.path.join(results_dir, "metrics", f"{name}_test_logits.npz")
        if not os.path.exists(path):
            continue
        data = np.load(path)
        logits, labels = data["logits"], data["labels"]
        pred = logits.argmax(1)
        cm = np.zeros((10, 10), dtype=np.float64)
        for t, p in zip(labels, pred):
            cm[t, p] += 1
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if cm[i, j] > 0.01:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6.3,
                            color="white" if cm[i, j] > 0.55 else "#222222")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CLASS_SHORT, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(CLASS_SHORT, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(SHORT[name])
        ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Row-normalised fraction")
    fig.savefig(os.path.join(out_dir, "confusion_matrices.png"), bbox_inches="tight")
    plt.close(fig)


def fig_robustness(robust: dict, out_dir: str) -> None:
    models = robust["models"]
    names = [n for n in ORDER if n in models]
    if not names:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), sharey=True)
    for ax, name in zip(axes.flat[:5], CORRUPTIONS):
        for m in names:
            suite = models[m]
            severities = sorted(int(s) for s in suite["corruptions"][name])
            accs = [100.0 * suite["corruptions"][name][str(s)] if str(s) in suite["corruptions"][name]
                    else 100.0 * suite["corruptions"][name][s] for s in severities]
            xs = [0] + severities
            ys = [100.0 * suite["clean_accuracy"]] + accs
            ax.plot(xs, ys, marker="o", ms=3, lw=1.3, color=COLORS[m], label=SHORT[m])
        ax.set_title(CORRUPTION_LABELS[name])
        ax.set_xlabel("Severity (0 = clean)")
        ax.set_xticks(range(0, 6))
        ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy (%)")
    ax = axes.flat[5]
    for m in names:
        suite = models[m]
        severities = sorted(int(s) for s in suite["severities"])
        mean_acc = []
        for s in severities:
            vals = [suite["corruptions"][c][str(s)] for c in CORRUPTIONS if str(s) in suite["corruptions"][c]]
            mean_acc.append(100.0 * float(np.mean(vals)))
        xs = [0] + severities
        ys = [100.0 * suite["clean_accuracy"]] + mean_acc
        ax.plot(xs, ys, marker="o", ms=3, lw=1.3, color=COLORS[m], label=SHORT[m])
    ax.set_title("Mean over five corruptions")
    ax.set_xlabel("Severity (0 = clean)")
    ax.set_xticks(range(0, 6))
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out_dir, "robustness_curves.png"), bbox_inches="tight")
    plt.close(fig)


def fig_robustness_heatmap(robust: dict, out_dir: str) -> None:
    models = robust["models"]
    names = [n for n in ORDER if n in models]
    if not names:
        return
    matrix = []
    for m in names:
        suite = models[m]
        row = []
        for c in CORRUPTIONS:
            vals = [suite["corruptions"][c][str(s)] for s in suite["severities"] if str(s) in suite["corruptions"][c]]
            row.append(100.0 * float(np.mean(vals)))
        matrix.append(row)
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=20, vmax=90, aspect="auto")
    ax.set_xticks(range(len(CORRUPTIONS)))
    ax.set_xticklabels([CORRUPTION_LABELS[c] for c in CORRUPTIONS], rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([SHORT[n] for n in names], fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Mean accuracy over severities (%)")
    ax.set_title("Corruption robustness (mean over severity 1-5)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "robustness_heatmap.png"), bbox_inches="tight")
    plt.close(fig)


def _overlay(image_hwc: np.ndarray, cam: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    heat = cmap(cam)[..., :3]
    return np.clip(0.55 * image_hwc + 0.45 * heat, 0, 1)


def fig_gradcam(data_dir: str, checkpoint_dir: str, out_dir: str) -> None:
    test = load_npz(os.path.join(data_dir, "cifar10_test.npz"))
    labels = test.labels.numpy()
    models = {}
    for name in ("resnet20", "resnet20_ds_se"):
        path = os.path.join(checkpoint_dir, f"{name}_best.pt")
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(name)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models[name] = model
    if len(models) < 2:
        return

    chosen = []
    for c in (0, 1, 3, 8):
        idx = int(np.flatnonzero(labels == c)[0])
        chosen.append(idx)

    fig, axes = plt.subplots(4, len(chosen), figsize=(11.2, 10.4))
    for col, idx in enumerate(chosen):
        raw = test.images[idx : idx + 1]
        foggy = apply_corruption(raw, "fog", 4, seed=77)
        for row, name in enumerate(("resnet20", "resnet20_ds_se")):
            cam_gen = GradCAM(models[name])
            for cond, (img, tag) in enumerate(((raw, "clean"), (foggy, "fog s=4"))):
                r = row * 2 + cond
                cam, pred, probs = cam_gen(prepare_input(img))
                base = denormalize(prepare_input(img))[0].permute(1, 2, 0).numpy()
                axes[r, col].imshow(_overlay(base, cam))
                axes[r, col].set_xticks([])
                axes[r, col].set_yticks([])
                axes[r, col].grid(False)
                if r == 0:
                    axes[r, col].set_title(f"true: {CLASS_SHORT[labels[idx]]}", fontsize=8)
                if col == 0:
                    axes[r, col].set_ylabel(f"{SHORT[name]}\n{tag}", fontsize=8)
                axes[r, col].set_xlabel(f"pred: {CLASS_SHORT[pred]} ({probs[pred]:.2f})", fontsize=7)
            cam_gen.remove()
    fig.suptitle("Grad-CAM: clean vs. foggy images (last-stage ResNet features)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gradcam_examples.png"), bbox_inches="tight")
    plt.close(fig)


def fig_finetune(results_dir: str, out_dir: str) -> None:
    path = os.path.join(results_dir, "metrics", "resnet20_ds_se_robust_summary.json")
    if not os.path.exists(path):
        return
    s = load_json(path)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    ax = axes[0]
    labels = ["Clean", "Corruption mean"]
    before = [100 * s["clean_before"], 100 * s["corruption_mean_before"]]
    after = [100 * s["clean_after"], 100 * s["corruption_mean_after"]]
    x = np.arange(2)
    ax.bar(x - 0.18, before, 0.36, label="Before fine-tune", color="#9ecae1")
    ax.bar(x + 0.18, after, 0.36, label="After fine-tune", color="#e6550d")
    for i in range(2):
        ax.text(x[i] - 0.18, before[i] + 0.8, f"{before[i]:.1f}", ha="center", fontsize=8)
        ax.text(x[i] + 0.18, after[i] + 0.8, f"{after[i]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(a) Overall accuracy")
    ax.legend()

    ax = axes[1]
    before_c, after_c = s["robustness_before"]["corruptions"], s["robustness_after"]["corruptions"]
    xs = np.arange(len(CORRUPTIONS))
    b = [100 * np.mean(list(before_c[c].values())) for c in CORRUPTIONS]
    a = [100 * np.mean(list(after_c[c].values())) for c in CORRUPTIONS]
    ax.bar(xs - 0.18, b, 0.36, label="Before", color="#9ecae1")
    ax.bar(xs + 0.18, a, 0.36, label="After", color="#e6550d")
    ax.set_xticks(xs)
    ax.set_xticklabels([CORRUPTION_LABELS[c] for c in CORRUPTIONS], rotation=20, ha="right")
    ax.set_ylabel("Mean accuracy, severity 1-5 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(b) Per-corruption accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "finetune_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--out-dir", default="report/figures")
    p.add_argument("--skip-gradcam", action="store_true")
    args = p.parse_args()
    style()
    os.makedirs(args.out_dir, exist_ok=True)

    summaries = available_summaries(args.results_dir)
    fig_dataset(args.data_dir, args.out_dir)
    fig_corruptions(args.data_dir, args.out_dir)
    if summaries:
        fig_training_curves(args.results_dir, summaries, args.out_dir)
        fig_cost_accuracy(summaries, os.path.join(args.results_dir, "metrics", "benchmark.json"), args.out_dir)
        fig_per_class(summaries, args.out_dir)
        fig_confusion(args.results_dir, summaries, args.out_dir)
    robust_path = os.path.join(args.results_dir, "metrics", "robustness.json")
    if os.path.exists(robust_path):
        robust = load_json(robust_path)
        fig_robustness(robust, args.out_dir)
        fig_robustness_heatmap(robust, args.out_dir)
    fig_finetune(args.results_dir, args.out_dir)
    if not args.skip_gradcam:
        fig_gradcam(args.data_dir, args.checkpoint_dir, args.out_dir)
    print(f"figures written to {args.out_dir}")
    for name in sorted(os.listdir(args.out_dir)):
        print(" ", name)


if __name__ == "__main__":
    main()