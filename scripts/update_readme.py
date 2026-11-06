"""Fill the README results section from the experiment JSON files."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["resnet20", "resnet20_ds", "resnet20_se", "resnet20_gelu", "resnet20_gn", "resnet20_ds_se"]
SHORT = {
    "resnet20": "R20",
    "resnet20_ds": "R20-DS",
    "resnet20_se": "R20-SE",
    "resnet20_gelu": "R20-GELU",
    "resnet20_gn": "R20-GN",
    "resnet20_ds_se": "R20-DS-SE",
}
MARKER = "<!-- RESULTS_TABLE -->"
START = "<!-- RESULTS_START -->"
END = "<!-- RESULTS_END -->"


def load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    metrics = os.path.join(ROOT, "results", "metrics")
    missing = [f"{m}_summary.json" for m in ORDER if not os.path.exists(os.path.join(metrics, f"{m}_summary.json"))]
    if missing:
        raise SystemExit("missing: " + ", ".join(missing))
    summaries = {m: load(os.path.join(metrics, f"{m}_summary.json")) for m in ORDER}
    bench = {row["model"]: row for row in load(os.path.join(metrics, "benchmark.json"))["models"]}
    robust = load(os.path.join(metrics, "robustness.json"))["models"]
    ft_path = os.path.join(metrics, "resnet20_ds_se_robust_summary.json")
    ft = load(ft_path) if os.path.exists(ft_path) else None

    base = summaries["resnet20"]
    acc_base = base["test"]["accuracy"]
    best = max(ORDER, key=lambda m: summaries[m]["test"]["accuracy"])
    best_rob = max(ORDER, key=lambda m: robust[m]["mean_accuracy"])
    ds = summaries["resnet20_ds"]
    ds_lat = bench["resnet20_ds"]["latency_b1"]["latency_ms_median"] / bench["resnet20"]["latency_b1"]["latency_ms_median"] - 1.0
    if ds_lat > 0.02:
        ds_lat_note = f"batch=1 延迟反而变化 **{ds_lat:+.1%}**"
    elif ds_lat < -0.02:
        ds_lat_note = f"batch=1 延迟也变化 **{ds_lat:+.1%}**（降幅小于 MACs 降幅）"
    else:
        ds_lat_note = f"batch=1 延迟基本持平（**{ds_lat:+.1%}**）"

    lines = [
        "| 模型 | 参数量 | MACs | Test Acc | ΔAcc | CPU 延迟 (batch=1) | 退化平均 Acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in ORDER:
        t = summaries[m]["test"]
        lines.append(
            f"| `{m}` | {summaries[m]['params']:,} | {summaries[m]['macs'] / 1e6:.2f}M | "
            f"{t['accuracy'] * 100:.2f}% | {(t['accuracy'] - acc_base) * 100:+.2f} pp | "
            f"{bench[m]['latency_b1']['latency_ms_median']:.2f} ms | {robust[m]['mean_accuracy'] * 100:.2f}% |"
        )
    lines += [
        "",
        f"- 基线 R20 测试准确率 **{acc_base * 100:.2f}%**；最佳模型 **{SHORT[best]} "
        f"{summaries[best]['test']['accuracy'] * 100:.2f}%**。",
        f"- 深度可分离卷积把参数量降至 {ds['params']:,}（−{(1 - ds['params'] / base['params']) * 100:.1f}%）、"
        f"MACs 降至 {ds['macs'] / 1e6:.2f}M（−{(1 - ds['macs'] / base['macs']) * 100:.1f}%），"
        f"{ds_lat_note}。",
        f"- 25 个退化条件上，R20-DS-SE 平均准确率 {robust['resnet20_ds_se']['mean_accuracy'] * 100:.2f}%，"
        f"基线 {robust['resnet20']['mean_accuracy'] * 100:.2f}%。",
        f"- 退化平均准确率最高的是 **{SHORT[best_rob]} {robust[best_rob]['mean_accuracy'] * 100:.2f}%**"
        f"（干净准确率 {robust[best_rob]['clean_accuracy'] * 100:.2f}%），说明干净精度与退化鲁棒性并不总是同向。",
    ]
    if ft:
        lines.append(
            f"- 退化微调后 R20-DS-SE 退化平均准确率 {ft['corruption_mean_before'] * 100:.2f}% → "
            f"{ft['corruption_mean_after'] * 100:.2f}%，干净准确率 {ft['clean_before'] * 100:.2f}% → "
            f"{ft['clean_after'] * 100:.2f}%。"
        )
    table = "\n".join(lines)

    readme_path = os.path.join(ROOT, "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        text = f.read()
    if START in text and END in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        text = before + START + "\n" + table + "\n" + END + after
    elif MARKER in text:
        before, after = text.split(MARKER, 1)
        text = before + START + "\n" + table + "\n" + END + after
    else:
        raise SystemExit(f"markers not found in {readme_path}")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"README results section updated ({len(lines)} lines)")


if __name__ == "__main__":
    main()