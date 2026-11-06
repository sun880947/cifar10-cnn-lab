"""Read chapter start pages from the rendered PDF and save a TOC page map."""
from __future__ import annotations

import json
import os
import re
import sys

import fitz

LABELS = [
    "摘要",
    "第 1 章  绪论",
    "第 2 章  数据集与实验设置",
    "第 3 章  模型设计",
    "第 4 章  训练结果与结构分析",
    "第 5 章  低能见度鲁棒性与场景应用",
    "第 6 章  结论与展望",
    "参考文献",
    "附录 A  代码结构",
    "附录 B  复现命令",
    "附录 C  结果文件说明",
]


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "report/CIFAR10_CNN_course_report.pdf"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "results/metrics/report_pages.json"
    doc = fitz.open(pdf_path)
    page_map = {"摘要": 2}
    for label in LABELS[1:]:
        target = norm(label)
        found = None
        for i in range(3, doc.page_count):
            lines = [norm(x) for x in doc[i].get_text().splitlines() if x.strip()]
            if target in lines:
                found = i + 1
                break
        if found is None:
            raise SystemExit(f"heading not found in PDF: {label}")
        page_map[label] = found
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(page_map, f, ensure_ascii=False, indent=2)
    print(f"pages={doc.page_count}")
    for k, v in page_map.items():
        print(f"{v:>3}  {k}")


if __name__ == "__main__":
    main()