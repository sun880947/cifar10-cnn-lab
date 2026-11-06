"""Cross-check the numbers in the generated artifacts against the raw outputs.

Checks, in order:
1. every ``*_summary.json`` test metric can be recomputed from ``*_test_logits.npz``;
2. ``benchmark.json`` parameters / MACs agree with the summaries;
3. ``robustness.json`` clean accuracy agrees with the summaries, and its mean /
   worst values are the mean / minimum of the stored per-condition matrix;
4. the fine-tuning summary's "before" suite agrees with the baseline evaluation;
5. every key test accuracy and parameter count appears in README.md (when the
   results table has been inserted);
6. every key test accuracy and parameter count appears in the DOCX report.

The script exits non-zero on the first failed category and prints a compact
report, so it can be used as the final reproducibility gate.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, Iterable, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import compute_metrics  # noqa: E402

ORDER = ["resnet20", "resnet20_ds", "resnet20_se", "resnet20_gelu", "resnet20_gn", "resnet20_ds_se"]
TOL = 5e-5


def fail(msg: str) -> None:
    print("FAIL:", msg)
    raise SystemExit(1)


def close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(float(a) - float(b)) <= tol


def load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_summaries(root: str) -> Dict[str, dict]:
    summaries = {}
    for name in ORDER:
        sp = os.path.join(root, "results", "metrics", f"{name}_summary.json")
        lp = os.path.join(root, "results", "metrics", f"{name}_test_logits.npz")
        if not os.path.exists(sp) or not os.path.exists(lp):
            fail(f"missing {sp} or {lp}")
        s = load(sp)
        d = np.load(lp)
        m = compute_metrics(d["logits"], d["labels"])
        for key in ("accuracy", "top5_accuracy", "nll", "macro_f1", "ece"):
            if not close(m[key], s["test"][key]):
                fail(f"{name}: {key} summary={s['test'][key]} recomputed={m[key]}")
        if int(m["num_samples"]) != int(s["test"]["num_samples"]):
            fail(f"{name}: num_samples mismatch")
        summaries[name] = s
        print(f"  summary/logits OK  {name:16s} acc={s['test']['accuracy']:.4f} "
              f"params={s['params']:,} macs={s['macs'] / 1e6:.4f}M")
    return summaries


def check_benchmark(root: str, summaries: Dict[str, dict]) -> None:
    path = os.path.join(root, "results", "metrics", "benchmark.json")
    if not os.path.exists(path):
        fail(f"missing {path}")
    b = load(path)
    by_model = {row["model"]: row for row in b["models"]}
    for name in ORDER:
        row = by_model[name]
        s = summaries[name]
        if int(row["params"]) != int(s["params"]):
            fail(f"{name}: benchmark params {row['params']} != summary {s['params']}")
        if not close(row["macs"], s["macs"]):
            fail(f"{name}: benchmark MACs mismatch")
        if not (0 < row["latency_b1"]["latency_ms_median"] < 1e5):
            fail(f"{name}: nonsensical latency")
        print(f"  benchmark OK      {name:16s} b1={row['latency_b1']['latency_ms_median']:.2f}ms "
              f"size={row['size_mb']:.2f}MB")


def check_robustness(root: str, summaries: Dict[str, dict]) -> None:
    path = os.path.join(root, "results", "metrics", "robustness.json")
    if not os.path.exists(path):
        fail(f"missing {path}")
    r = load(path)
    names = ORDER
    for name in names:
        suite = r["models"][name]
        if not close(suite["clean_accuracy"], summaries[name]["test"]["accuracy"]):
            fail(f"{name}: robustness clean != summary test")
        accs: List[float] = []
        for c in r["corruptions"]:
            for sev in r["severities"]:
                accs.append(float(suite["corruptions"][c][str(sev)]))
        if not close(suite["mean_accuracy"], float(np.mean(accs))):
            fail(f"{name}: robustness mean mismatch")
        if not close(suite["worst_accuracy"], float(np.min(accs))):
            fail(f"{name}: robustness worst mismatch")
        if not close(suite["mean_drop"], suite["clean_accuracy"] - suite["mean_accuracy"]):
            fail(f"{name}: robustness mean_drop mismatch")
        print(f"  robustness OK     {name:16s} clean={suite['clean_accuracy']:.4f} "
              f"mean={suite['mean_accuracy']:.4f} worst={suite['worst_accuracy']:.4f}")


def check_finetune(root: str, summaries: Dict[str, dict]) -> None:
    path = os.path.join(root, "results", "metrics", "resnet20_ds_se_robust_summary.json")
    if not os.path.exists(path):
        fail(f"missing {path}")
    f = load(path)
    before = f["robustness_before"]
    after = f["robustness_after"]
    if not close(before["clean_accuracy"], summaries["resnet20_ds_se"]["test"]["accuracy"]):
        fail("finetune: robustness_before.clean != clean baseline")
    if not close(f["clean_before"], before["clean_accuracy"]):
        fail("finetune: clean_before mismatch")
    if not close(f["corruption_mean_before"], before["mean_accuracy"]):
        fail("finetune: corruption_mean_before mismatch")
    if not close(f["clean_after"], after["clean_accuracy"]):
        fail("finetune: clean_after mismatch")
    if not close(f["corruption_mean_after"], after["mean_accuracy"]):
        fail("finetune: corruption_mean_after mismatch")
    print(f"  finetune OK       clean {f['clean_before']:.4f}->{f['clean_after']:.4f} "
          f"corr mean {f['corruption_mean_before']:.4f}->{f['corruption_mean_after']:.4f}")


def check_readme(root: str, summaries: Dict[str, dict]) -> None:
    path = os.path.join(root, "README.md")
    if not os.path.exists(path):
        fail(f"missing {path}")
    text = open(path, "r", encoding="utf-8").read()
    if "<!-- RESULTS_TABLE -->" in text:
        fail("README still contains the RESULTS_TABLE placeholder")
    for name in ORDER:
        acc = f"{summaries[name]['test']['accuracy'] * 100:.2f}%"
        params = f"{summaries[name]['params']:,}"
        if acc not in text:
            fail(f"README missing test accuracy {acc} for {name}")
        if params not in text:
            fail(f"README missing parameter count {params} for {name}")
    print("  README OK         all six accuracy/parameter pairs present")


def check_docx(root: str, summaries: Dict[str, dict]) -> None:
    path = os.path.join(root, "report", "CIFAR10_CNN_course_report.docx")
    if not os.path.exists(path):
        fail(f"missing {path}")
    from docx import Document

    doc = Document(path)
    chunks: List[str] = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                chunks.append(cell.text)
    text = "\n".join(chunks)
    # Normalise thin spaces that Word may insert in numbers.
    flat = re.sub(r"\s+", "", text)
    for name in ORDER:
        acc = f"{summaries[name]['test']['accuracy'] * 100:.2f}%"
        params = f"{summaries[name]['params']:,}"
        if acc not in flat:
            fail(f"DOCX missing test accuracy {acc} for {name}")
        if params.replace(",", "") not in flat.replace(",", ""):
            fail(f"DOCX missing parameter count {params} for {name}")
    print(f"  DOCX OK           {len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables, "
          f"all six accuracy/parameter pairs present")


def main() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"root = {root}")
    print("[1/6] summaries vs saved logits")
    summaries = check_summaries(root)
    print("[2/6] benchmark vs summaries")
    check_benchmark(root, summaries)
    print("[3/6] robustness matrix")
    check_robustness(root, summaries)
    print("[4/6] fine-tuning summary")
    check_finetune(root, summaries)
    print("[5/6] README results")
    check_readme(root, summaries)
    print("[6/6] DOCX report")
    check_docx(root, summaries)
    print("\nALL CONSISTENCY CHECKS PASSED")


if __name__ == "__main__":
    main()