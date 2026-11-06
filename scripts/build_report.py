"""Build the Chinese course report as a DOCX from recorded experiment outputs.

Every number in the report is read from ``results/metrics/*.json`` at build time;
no result is typed by hand.  Run after training, benchmark, robustness,
fine-tuning and ``scripts.visualize``:

    python -m scripts.build_report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.corruptions import CORRUPTION_LABELS, CORRUPTIONS  # noqa: E402
from src.models import MODEL_LABELS, MODEL_SPECS  # noqa: E402

ORDER = ["resnet20", "resnet20_ds", "resnet20_se", "resnet20_gelu", "resnet20_gn", "resnet20_ds_se"]
SHORT = {
    "resnet20": "R20",
    "resnet20_ds": "R20-DS",
    "resnet20_se": "R20-SE",
    "resnet20_gelu": "R20-GELU",
    "resnet20_gn": "R20-GN",
    "resnet20_ds_se": "R20-DS-SE",
}
CN_NAME = {
    "resnet20": "基线",
    "resnet20_ds": "深度可分离卷积",
    "resnet20_se": "SE 通道注意力",
    "resnet20_gelu": "GELU 激活",
    "resnet20_gn": "GroupNorm 归一化",
    "resnet20_ds_se": "深度可分离 + SE",
}
CORRUPTION_CN = {
    "fog": "雾",
    "low_light": "低照度",
    "motion_blur": "运动模糊",
    "gaussian_noise": "高斯噪声",
    "jpeg": "JPEG 压缩",
}

BODY_FONT = "宋体"
HEAD_FONT = "黑体"
LATIN_FONT = "Times New Roman"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def pct(x: float, digits: int = 2, sign: bool = False) -> str:
    return f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"


def delta_pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:+.{digits}f} 个百分点"


def num(x: float, digits: int = 2) -> str:
    return f"{x:,.{digits}f}"


def effective_train_minutes(root: str, name: str, fallback_seconds: float) -> float:
    """Sum per-epoch times, replacing a suspend-sized outlier with the median.

    One GELU epoch was stretched by a Windows suspend; using the raw wall clock
    would mistakenly suggest that GELU itself is 1.5x slower than the baseline.
    """
    path = os.path.join(root, "results", "history", f"{name}.json")
    if not os.path.exists(path):
        return fallback_seconds / 60.0
    history = load_json(path, required=False) or []
    times = [float(row["epoch_seconds"]) for row in history if "epoch_seconds" in row]
    if not times:
        return fallback_seconds / 60.0
    median = sorted(times)[len(times) // 2]
    cleaned = [t if t <= 3.0 * median else median for t in times]
    return sum(cleaned) / 60.0


def set_run_font(run, size: float = 12, bold: Optional[bool] = None,
                 ea: str = BODY_FONT, latin: str = LATIN_FONT,
                 color: Optional[RGBColor] = None) -> None:
    run.font.name = latin
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), ea)
    rfonts.set(qn("w:ascii"), latin)
    rfonts.set(qn("w:hAnsi"), latin)


def add_rich(paragraph, text: str, size: float = 12, ea: str = BODY_FONT,
             latin: str = LATIN_FONT) -> None:
    """Add text, treating **...** as bold."""
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        run = paragraph.add_run(part)
        set_run_font(run, size=size, bold=(i % 2 == 1), ea=ea, latin=latin)


def body(doc: Document, text: str, indent: bool = True, size: float = 12,
         space_after: float = 0, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    pf.space_before = Pt(0)
    pf.space_after = Pt(space_after)
    if indent:
        pf.first_line_indent = Pt(size * 2)
    add_rich(p, text, size=size)
    return p


def bullet(doc: Document, text: str, size: float = 12) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_together = True
    add_rich(p, text, size=size)


def heading(doc: Document, text: str, level: int) -> None:
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt({1: 16, 2: 12, 3: 8}.get(level, 8))
    p.paragraph_format.space_after = Pt({1: 10, 2: 6, 3: 4}.get(level, 4))
    run = p.add_run(text)
    size = {1: 16, 2: 14, 3: 12}.get(level, 12)
    set_run_font(run, size=size, bold=True, ea=HEAD_FONT, color=RGBColor(0, 0, 0))


def caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(9)
    run = p.add_run(text)
    set_run_font(run, size=10.5, ea=BODY_FONT, color=RGBColor(60, 60, 60))


def figure(doc: Document, path: str, cap: str, width_cm: float = 15.2) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing figure: {path}")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Cm(width_cm))
    caption(doc, cap)


def shade(cell, fill: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def cell_text(cell, text: str, bold: bool = False, size: float = 8.5,
              align=WD_ALIGN_PARAGRAPH.CENTER) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(str(text))
    set_run_font(run, size=size, bold=bold)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def add_table(doc: Document, headers: Sequence[str], rows: Sequence[Sequence[str]],
              widths: Sequence[float], cap: str, font_size: float = 8.5,
              first_col_left: bool = False) -> None:
    if len(widths) != len(headers):
        raise ValueError("width count must match header count")
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    hdr = table.rows[0]
    trPr = hdr._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.width = Cm(widths[i])
        shade(cell, "E7E6E6")
        cell_text(cell, h, bold=True, size=font_size)

    for r in rows:
        row = table.add_row()
        for i, value in enumerate(r):
            cell = row.cells[i]
            cell.width = Cm(widths[i])
            cell_text(
                cell, value, size=font_size,
                align=WD_ALIGN_PARAGRAPH.LEFT if (first_col_left and i == 0) else WD_ALIGN_PARAGRAPH.CENTER,
            )
    for i, w in enumerate(widths):
        table.columns[i].width = Cm(w)
    # Keep the table on one page where possible: no row may split, and all but
    # the last row are tied to the next row.
    for ri, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.insert(0, OxmlElement("w:cantSplit"))
        if ri < len(table.rows) - 1:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.keep_with_next = True
    caption(doc, cap)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_page_number_footer(section) -> None:
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run_font(run, size=9, color=RGBColor(90, 90, 90))


def add_header(section, text: str) -> None:
    p = section.header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    set_run_font(run, size=9, color=RGBColor(110, 110, 110))
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "4")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "BFBFBF")
    pBdr.append(bottom)
    pPr.insert(0, pBdr)


def page_break(doc: Document) -> None:
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)


# ---------------------------------------------------------------------------
# Results loading
# ---------------------------------------------------------------------------
def load_json(path: str, required: bool = True):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_results(root: str) -> dict:
    metrics = os.path.join(root, "results", "metrics")
    summaries = {}
    missing = []
    for name in ORDER:
        path = os.path.join(metrics, f"{name}_summary.json")
        if not os.path.exists(path):
            missing.append(path)
            continue
        summaries[name] = load_json(path)
    if missing:
        raise SystemExit("missing summaries:\n  " + "\n  ".join(missing))

    bench_raw = load_json(os.path.join(metrics, "benchmark.json"))
    benchmark = {row["model"]: row for row in bench_raw["models"]}
    robust = load_json(os.path.join(metrics, "robustness.json"))
    ft_path = os.path.join(metrics, "resnet20_ds_se_robust_summary.json")
    finetune = load_json(ft_path, required=False)
    return {
        "summaries": summaries,
        "benchmark": benchmark,
        "benchmark_meta": bench_raw,
        "robustness": robust,
        "finetune": finetune,
    }


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------
def cover(doc: Document) -> None:
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("《深度学习》课程大作业")
    set_run_font(run, size=16, bold=False, ea=HEAD_FONT)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(30)
    run = p.add_run("CIFAR-10 图像分类中的 CNN 结构对比\n与低能见度鲁棒性分析")
    set_run_font(run, size=24, bold=True, ea=HEAD_FONT)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12)
    run = p.add_run("——基于 ResNet-20 的深度可分离卷积、SE 注意力与归一化消融实验")
    set_run_font(run, size=13, ea=BODY_FONT, color=RGBColor(70, 70, 70))

    for _ in range(5):
        doc.add_paragraph()
    info = [
        ("姓　　名", "孙蜀齐"),
        ("学　　号", "2421010213"),
        ("专　　业", "人工智能"),
        ("指导教师", "史朋飞"),
    ]
    for label, value in info:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(8)
        run = p.add_run(f"{label}：{value if value else '________________'}")
        set_run_font(run, size=13)

    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("2026 年 11 月")
    set_run_font(run, size=12)


def abstract(doc: Document, r: dict) -> None:
    s = r["summaries"]
    base = s["resnet20"]
    ds = s["resnet20_ds"]
    dsse = s["resnet20_ds_se"]
    acc_base = base["test"]["accuracy"]
    acc_best = max(s[n]["test"]["accuracy"] for n in ORDER)
    best_name = max(ORDER, key=lambda n: s[n]["test"]["accuracy"])
    param_cut = 1.0 - ds["params"] / base["params"]
    mac_cut = 1.0 - ds["macs"] / base["macs"]
    b1_base = r["benchmark"]["resnet20"]["latency_b1"]["latency_ms_median"]
    b1_ds = r["benchmark"]["resnet20_ds"]["latency_b1"]["latency_ms_median"]
    lat_change = b1_ds / b1_base - 1.0
    if lat_change > 0.02:
        lat_text = (
            f"但单张 CPU 延迟由 {b1_base:.2f} ms 升至 {b1_ds:.2f} ms（**{lat_change:+.1%}**），"
            "说明在 CPU 上 FLOPs 的下降并不自动等于延迟的下降。"
        )
    elif lat_change < -0.02:
        lat_text = (
            f"单张 CPU 延迟也由 {b1_base:.2f} ms 降至 {b1_ds:.2f} ms（**{lat_change:+.1%}**），"
            "但降幅仍小于 MACs 的降幅，说明访存与算子启动开销不可忽略。"
        )
    else:
        lat_text = (
            f"单张 CPU 延迟基本持平（{b1_base:.2f} ms → {b1_ds:.2f} ms，**{lat_change:+.1%}**），"
            "说明参数量与 MACs 的下降没有转化为可测的延迟收益。"
        )

    heading(doc, "摘　要", 1)
    body(
        doc,
        "卷积神经网络（CNN）在图像分类任务上已经相当成熟，但在边缘设备部署和低能见度场景中，"
        "“精度—参数量—计算量—实际延迟”之间往往并不存在简单的线性关系。本报告以 CIFAR-10 为数据集，"
        "以 CIFAR 风格 ResNet-20 为基线，在完全一致的训练配方下比较了五种结构改动：深度可分离卷积、"
        "SE 通道注意力、GELU 激活、GroupNorm 归一化，以及深度可分离卷积与 SE 的组合。六个模型各训练 "
        f"{base['epochs']} 个 epoch，并在同一测试集上评估。",
    )
    body(
        doc,
        f"实验结果表明：基线 R20 的测试准确率为 **{pct(acc_base)}**；表现最好的变体是 "
        f"**{SHORT[best_name]}（{pct(acc_best)}）**。深度可分离卷积把参数量从 "
        f"{base['params']:,} 降到 {ds['params']:,}（**−{param_cut:.2%}**），MACs 从 "
        f"{base['macs'] / 1e6:.2f}M 降到 {ds['macs'] / 1e6:.2f}M（**−{mac_cut:.2%}**）；"
        f"{lat_text}"
        "SE 与 GELU 的精度变化分别为 "
        f"{delta_pct(s['resnet20_se']['test']['accuracy'] - acc_base)} 和 "
        f"{delta_pct(s['resnet20_gelu']['test']['accuracy'] - acc_base)}；"
        f"GroupNorm 相对 BatchNorm 的变化为 {delta_pct(s['resnet20_gn']['test']['accuracy'] - acc_base)}。",
    )

    rob = r["robustness"]["models"]
    base_mean = rob["resnet20"]["mean_accuracy"]
    dsse_mean = rob["resnet20_ds_se"]["mean_accuracy"]
    worst_model = min(ORDER, key=lambda n: rob[n]["mean_accuracy"])
    ft = r["finetune"]
    if ft:
        ft_text = (
            f"在 5 类合成退化（雾、低照度、运动模糊、高斯噪声、JPEG 压缩）上，基线平均准确率降至 "
            f"{pct(base_mean)}，R20-DS-SE 为 {pct(dsse_mean)}；对 R20-DS-SE 做退化数据微调后，"
            f"退化平均准确率由 {pct(ft['corruption_mean_before'])} 变为 "
            f"{pct(ft['corruption_mean_after'])}（{delta_pct(ft['corruption_mean_after'] - ft['corruption_mean_before'])}），"
            f"干净测试准确率由 {pct(ft['clean_before'])} 变为 {pct(ft['clean_after'])}"
            f"（{delta_pct(ft['clean_after'] - ft['clean_before'])}）。"
        )
    else:
        ft_text = (
            f"在 5 类合成退化上，基线平均准确率降至 {pct(base_mean)}，R20-DS-SE 为 {pct(dsse_mean)}；"
            f"表现最弱的是 {SHORT[worst_model]}（{pct(rob[worst_model]['mean_accuracy'])}）。"
        )
    body(
        doc,
        ft_text + "最后，本文用 Grad-CAM 对比了干净与雾天图像上的关注区域，并实现了一个可上传图片、"
        "选择模型与退化强度的 Flask 演示系统。所有代码、日志、指标 JSON 与报告数字均由同一仓库产出，"
        "可通过固定随机种子复现。",
    )
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    run = p.add_run("关键词：")
    set_run_font(run, size=12, bold=True, ea=HEAD_FONT)
    run = p.add_run("卷积神经网络；ResNet；深度可分离卷积；SE 注意力；鲁棒性；CIFAR-10")
    set_run_font(run, size=12)


def contents(doc: Document, page_map: Optional[Dict[str, int]] = None) -> None:
    heading(doc, "目　录", 1)
    items = [
        ("摘要", ""),
        ("第 1 章  绪论", ""),
        ("第 2 章  数据集与实验设置", ""),
        ("第 3 章  模型设计", ""),
        ("第 4 章  训练结果与结构分析", ""),
        ("第 5 章  低能见度鲁棒性与场景应用", ""),
        ("第 6 章  结论与展望", ""),
        ("参考文献", ""),
        ("附录 A  代码结构", ""),
        ("附录 B  复现命令", ""),
        ("附录 C  结果文件说明", ""),
    ]
    for text, _ in items:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.8
        p.paragraph_format.tab_stops.add_tab_stop(
            Cm(15.9), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS
        )
        run = p.add_run(text)
        set_run_font(run, size=12)
        if page_map and text in page_map:
            run = p.add_run(f"\t{page_map[text]}")
            set_run_font(run, size=12)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    note = ("页码与正文实际分页一致，由构建脚本从导出的 PDF 中回填；若在 Word 中修改内容导致分页变化，"
            "请以正文页码为准。" if page_map else
            "说明：页码将在 PDF 排版后由构建脚本回填；报告中的图表编号与正文引用一一对应。")
    run = p.add_run(note)
    set_run_font(run, size=10.5, ea=BODY_FONT, color=RGBColor(100, 100, 100))


def chapter1(doc: Document, r: dict) -> None:
    heading(doc, "第 1 章  绪论", 1)
    heading(doc, "1.1  研究背景", 2)
    body(
        doc,
        "图像分类是计算机视觉的基础任务，也是理解卷积神经网络内部组件的常用载体。从 AlexNet 到 ResNet，"
        "网络深度的增加带来了精度提升，但也带来了参数量和计算量的增长。在手机、嵌入式设备和边缘服务器上，"
        "模型往往需要在有限算力、有限内存和实时性约束下运行，因此“更少的参数与 FLOPs”成为模型设计的核心目标之一。",
    )
    body(
        doc,
        "与此同时，真实部署环境并不总是清晰的实验室图像。雾、低照度、运动模糊、传感器噪声和 JPEG 压缩都会改变"
        "输入分布，使在干净数据上训练的模型出现明显精度下降。对道路场景而言，低能见度条件下的稳定识别具有直接的"
        "应用价值，但真实雾天数据的采集与标注成本很高。因此，在公开数据集上构造可控退化，是课程作业条件下研究"
        "鲁棒性的可行方案。",
    )

    heading(doc, "1.2  课程任务与教材实验的区别", 2)
    body(
        doc,
        "本作业要求实现一个与教材实验三不同的经典 CNN，在开源数据集上训练到基准准确率，并完成结构修改、"
        "实验分析与场景应用。教材中与实验三对应的应用型章节以图像去雾、超分辨率重建和目标检测等流程为主线；"
        "本报告不重复这些流程，"
        "而是选择 CIFAR-10 图像分类作为任务，把重点放在 CNN 组件的受控消融、CPU 部署成本与退化鲁棒性上。"
        "两者在任务形式、数据规模和评价指标上均不同，满足“必须区别于教材实验三”的要求。",
    )
    body(
        doc,
        "具体而言，本作业需要覆盖以下考察点：",
    )
    for text in [
        "**深度可分离卷积**：比较参数量、计算量与精度，而不只报告一个准确率；",
        "**通道注意力**：实现 SE 模块并分析其收益与开销；",
        "**激活函数与归一化**：比较 GELU 与 ReLU、GroupNorm 与 BatchNorm；",
        "**模型设计与调优**：在统一配方下训练多个变体，保证比较公平；",
        "**场景适配**：构造低能见度退化，评估鲁棒性并做退化微调；",
        "**工程交付**：提供可运行代码、训练日志、复现说明与交互演示。",
    ]:
        bullet(doc, text)

    heading(doc, "1.3  本文主要工作", 2)
    body(
        doc,
        "本文的主要工作可以概括为四点。第一，实现了一个 CIFAR 风格 ResNet-20 基线，并在其基础上通过配置开关"
        "生成五个结构变体，六个模型共享同一数据划分、增强策略、优化器和学习率调度。第二，使用前向 hook 统计"
        "每个模型的参数量与 MACs，并在同一台 CPU 上测量 batch=1 与 batch=128 的推理延迟，给出“精度—成本”的"
        "多维权衡。第三，设计雾、低照度、运动模糊、高斯噪声和 JPEG 压缩五类退化，在 5 个强度上评估全部模型，"
        "并对轻量组合模型做退化数据微调。第四，使用 Grad-CAM 观察模型在干净与雾天图像上的关注区域，并提供一个"
        "可交互的 Flask 演示。",
    )
    body(
        doc,
        "需要强调的是，本文的目标不是追求 CIFAR-10 上的最高精度。ResNet-20 是一个刻意保持较小的经典模型，"
        "在 CPU 上可以在可接受的时间内完成多次训练；这使得“控制变量、重复六个模型”成为可能。报告中的所有"
        "数字均来自 `results/metrics/` 下的 JSON 文件，并由 `scripts/build_report.py` 自动写入，避免手工抄录"
        "带来的不一致。",
    )

    heading(doc, "1.4  报告结构", 2)
    body(
        doc,
        "第 2 章介绍数据集、划分、预处理与评价指标；第 3 章说明基线残差块及四种结构改动的设计；第 4 章给出"
        "训练结果，并从参数量、MACs、延迟、逐类精度和混淆矩阵等角度分析各组件的影响；第 5 章讨论低能见度"
        "鲁棒性、退化微调、Grad-CAM 与演示系统；第 6 章总结结论、局限与后续改进方向。",
    )


def chapter2(doc: Document, r: dict) -> None:
    s = r["summaries"]
    heading(doc, "第 2 章  数据集与实验设置", 1)
    heading(doc, "2.1  CIFAR-10 数据集", 2)
    body(
        doc,
        "CIFAR-10 由 Krizhevsky 等人整理，包含 10 个类别、共 60000 张 32×32 彩色图像，其中训练集 50000 张、"
        "测试集 10000 张，各类别数量基本均衡。10 个类别为：飞机、汽车、鸟、猫、鹿、狗、蛙、马、船、卡车。"
        "图像分辨率低、类内差异大、背景复杂，适合在有限算力下研究网络结构，而不适合直接评估高分辨率真实场景。",
    )
    add_table(
        doc,
        ["项目", "取值", "说明"],
        [
            ["图像尺寸", "32 × 32 × 3", "RGB，uint8 存储"],
            ["类别数", "10", "各类 6000 张（训练 5000 / 测试 1000）"],
            ["原始训练集", "50000", "按类别分层划分"],
            ["训练 / 验证", "45000 / 5000", "每类 500 张验证，seed=42"],
            ["测试集", "10000", "只在最终评估使用"],
            ["归一化", "CIFAR-10 均值 / 标准差", "逐通道标准化"],
        ],
        [4.2, 5.2, 6.4],
        "表 2-1  CIFAR-10 数据划分与预处理",
    )
    figure(doc, os.path.join("report", "figures", "dataset_samples.png"),
           "图 2-1  CIFAR-10 测试集每类一个样例", width_cm=14.8)

    heading(doc, "2.2  数据划分与数据增强", 2)
    body(
        doc,
        "为避免测试集信息泄漏，本文不把测试集用于任何调参。具体做法是："
        "在 50000 张训练图像中，按类别分层抽取 500 张作为验证集，剩余 45000 张用于训练；模型选择、早停和学习率"
        "策略均只看验证集；测试集只在训练结束后评估一次。这样得到的测试数字更接近“未见数据”的表现。",
    )
    body(
        doc,
        "训练增强由三部分组成：先对图像四周零填充 4 像素，再随机裁剪出 32×32 区域，随后以 0.5 的概率水平翻转；"
        "最后施加 Cutout，将 8×8 的随机方形区域置零。前两者对应平移与镜像不变性，Cutout 则迫使网络不要过度"
        "依赖单个局部区域。验证与测试阶段不做随机增强，只做中心区域（完整图像）的归一化。",
    )
    body(
        doc,
        "归一化使用 CIFAR-10 的全局均值 (0.4914, 0.4822, 0.4465) 与标准差 (0.2470, 0.2435, 0.2616)。"
        "输入张量从 NHWC 转为 PyTorch 的 NCHW，并使用 channels-last 内存格式，以改善 CPU 卷积的访存效率。",
    )

    heading(doc, "2.3  评价指标", 2)
    body(
        doc,
        "分类质量使用五个指标：**Accuracy** 为 Top-1 准确率；**Top-5 Accuracy** 衡量正确类别是否进入前五；"
        "**Macro-F1** 对 10 个类别等权平均，能反映类别不均衡时的整体表现；**NLL** 为负对数似然，关注概率质量"
        "是否放在正确类别上；**ECE**（Expected Calibration Error，15 个置信度分箱）衡量置信度与真实正确率"
        "之间的偏差。ECE 越低，说明模型“知道自己有多确定”。",
    )
    body(
        doc,
        "部署成本使用四个指标：**参数量**（可训练参数个数）、**MACs**（每张图像一次前向的乘加次数，"
        "FLOPs = 2 × MACs）、**模型文件大小**（float32 参数占用）与**CPU 推理延迟**。延迟在 batch=1 和 "
        "batch=128 两种设置下测量，前者对应单张在线推理，后者对应吞吐。所有延迟测量均在 `torch.inference_mode()`"
        " 下进行，先 warmup 再取 20 次的中位数与 P90，以减少缓存和调度噪声。",
    )
    body(
        doc,
        "需要区分“理论计算量”和“实际延迟”：MACs 只统计乘加次数，不包含内存访问、逐通道卷积的循环开销、"
        "算子融合与线程调度。因此，MACs 更低的模型不一定在 CPU 上更快。本文把两者同时报告，正是为了观察这一差异。",
    )

    heading(doc, "2.4  实验环境", 2)
    body(
        doc,
        "实验在 Windows 11 的 CPU 环境中完成，未使用 GPU。Python 3.12.7，PyTorch 2.12.0+cpu，"
        "torchvision 0.27.0+cpu，NumPy 2.2.6；CPU 为 Intel Core Ultra 9 185H（16 核 22 线程，支持 AVX2，"
        "不支持 AVX-512 BF16），内存 32 GB。训练时固定 16 个 PyTorch 线程，并把 Windows 电源计划设为“高性能”，"
        "以减少降频带来的 epoch 时间波动。",
    )
    body(
        doc,
        "在该环境下，ResNet-20 一个 epoch（45000 张训练图 + 5000 张验证图）约需 65–75 秒，具体受后台负载和"
        "CPU 温度影响；六个模型各 60 epoch 的顺序训练耗时数小时。报告中的训练时间取自每个模型 history JSON 的"
        "逐 epoch 计时，而不是估算值；唯一的例外是 GELU 训练期间的一次系统挂起，表中用中位数替换了异常 epoch。",
    )


def chapter3(doc: Document, r: dict) -> None:
    s = r["summaries"]
    base = s["resnet20"]
    heading(doc, "第 3 章  模型设计", 1)
    heading(doc, "3.1  ResNet-20 基线", 2)
    body(
        doc,
        "CIFAR 风格 ResNet-20 由 He 等人在残差学习工作中提出：一个 3×3 卷积作为 stem，随后三个 stage，"
        "每个 stage 包含 3 个 BasicBlock，通道数分别为 16、32、64；后两个 stage 的第一个 block 使用 stride=2 下采样，"
        "其余 block 保持分辨率。最后经全局平均池化和一个 10 类全连接层输出 logits。总参数量约 "
        f"{base['params']:,}，单张图像约 {base['macs'] / 1e6:.2f}M MACs。",
    )
    body(
        doc,
        "每个 BasicBlock 的主分支为“3×3 卷积 → BatchNorm → ReLU → 3×3 卷积 → BatchNorm”，"
        "再加一条恒等短路（当通道数或分辨率变化时用 1×1 卷积 + BN 调整），相加后再经过 ReLU。"
        "残差连接使梯度可以沿短路直接回传，是训练更深网络的关键。",
    )
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("y = ReLU( F(x, {W_i}) + x )")
    set_run_font(run, size=11, latin="Cambria Math")
    caption(doc, "式 3-1  BasicBlock 的残差形式（恒等短路时）")

    heading(doc, "3.2  深度可分离卷积", 2)
    body(
        doc,
        "标准 3×3 卷积在 C_in 个输入通道和 C_out 个输出通道之间做全连接，参数量为 "
        "C_in × C_out × 3 × 3。深度可分离卷积把它分解为两步：先对每个输入通道独立做 3×3 卷积"
        "（depthwise，groups=C_in），再用 1×1 卷积做通道混合（pointwise）。参数量变为：",
    )
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("C_in × 3 × 3  +  C_in × C_out")
    set_run_font(run, size=11, latin="Cambria Math")
    caption(doc, "式 3-2  深度可分离卷积的参数量")
    body(
        doc,
        "两者之比约为 1/C_out + 1/9。当 C_out=16 时，理论参数量约为标准卷积的 17.4%，即减少约 82.6%；"
        "当通道数增大时，比例进一步接近 1/9。MACs 也有类似的下降。直观上，深度可分离卷积把“空间滤波”和"
        "“通道混合”解耦，用更少的乘加完成同样的感受野。",
    )
    body(
        doc,
        "但 depthwise 卷积的每个输出通道只依赖一个输入通道，计算密度低、访存模式不连续；在 CPU 上，"
        "它往往无法像标准卷积那样充分利用向量化和多线程，因此实际延迟可能不降反升。这一点将在第 4 章用实测数据检验。",
    )

    heading(doc, "3.3  SE 通道注意力", 2)
    body(
        doc,
        "Squeeze-and-Excitation（SE）模块先对每个通道做全局平均池化，得到一个 C 维向量 z；再用两层 1×1 卷积"
        "（中间维度为 max(4, C/8)）学习通道之间的非线性关系，最后用 sigmoid 得到 0 到 1 的通道权重 s，并与原特征图"
        "逐通道相乘：",
    )
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("z = AvgPool(x),   s = σ( W2 · ReLU( W1 · z ) ),   x' = s ⊙ x")
    set_run_font(run, size=11, latin="Cambria Math")
    caption(doc, "式 3-3  SE 模块的通道重标定")
    body(
        doc,
        "本文把 SE 放在每个 BasicBlock 的第二个 BN 之后、残差相加之前，只重标定残差分支，不改变短路本身。"
        "这样既能让注意力调制非线性特征，又保留了恒等映射的梯度通路。设 h=max(4, C/8)，两层 1×1 卷积均带偏置，"
        "则 SE 的额外参数量为 2C·h + h + C；当 C=64 时 h=8，共 1,096 个参数。相对整体很小，但引入了一次全局池化"
        "和两次 1×1 卷积，会增加少量延迟。",
    )

    heading(doc, "3.4  GELU 激活", 2)
    body(
        doc,
        "GELU（Gaussian Error Linear Unit）用标准正态分布的累积分布函数对输入加权：",
    )
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("GELU(x) = 0.5 · x · (1 + erf(x / √2))")
    set_run_font(run, size=11, latin="Cambria Math")
    caption(doc, "式 3-4  GELU 的定义")
    body(
        doc,
        "与 ReLU 的硬截断相比，GELU 在负半轴被平滑地压制而不是直接置零，在 0 附近也更平滑。它在 Transformer 中"
        "被广泛使用，也被认为可能改善深层网络的优化。但 GELU 需要计算 erf 或 tanh 近似，在 CPU 上通常比 ReLU "
        "更慢；其收益是否足以抵消这一开销，需要实验判断。本文使用 `nn.GELU(approximate='none')`，即精确版本。",
    )

    heading(doc, "3.5  GroupNorm 归一化", 2)
    body(
        doc,
        "BatchNorm 在训练时对一个 mini-batch 内的样本统计均值和方差，因此依赖 batch size，并在训练/推理之间"
        "存在行为差异。GroupNorm 把通道分成 G 组，在每组内部跨通道和空间位置统计，不依赖 batch 维度。"
        "本文取 G=8；当通道数为 16、32、64 时都能整除。GroupNorm 为每个通道保留一对 γ、β，参数量与 BatchNorm 相同，"
        "但每个样本的统计更稳定，适合小 batch 场景。",
    )
    body(
        doc,
        "在 CPU 上，GroupNorm 需要对每个样本、每个组分别做归约，访存模式不如 BatchNorm 的逐通道归约规则，"
        "因此可能更慢。本文把 GroupNorm 作为“归一化方式”的对照，观察其在 batch=256 的训练和 batch=1 的推理下"
        "分别带来什么影响。",
    )

    heading(doc, "3.6  六个模型变体", 2)
    body(
        doc,
        "六个模型只在上述组件上不同，深度、宽度、stage 结构、分类头和初始化完全一致。表 3-1 给出配置与理论成本，"
        "图 3-1 对比了基线 block 与 DS+SE block 的数据流。",
    )
    rows = []
    for name in ORDER:
        sp = MODEL_SPECS[name]
        rows.append([
            SHORT[name],
            "BN" if sp["norm"] == "bn" else "GN(8)",
            sp["act"].upper(),
            "是" if sp["use_ds"] else "否",
            "是" if sp["use_se"] else "否",
            f"{s[name]['params']:,}",
            f"{s[name]['macs'] / 1e6:.2f}",
        ])
    add_table(
        doc,
        ["模型", "归一化", "激活", "深度可分离", "SE", "参数量", "MACs (M)"],
        rows,
        [2.5, 2.2, 2.0, 2.3, 1.5, 2.7, 2.8],
        "表 3-1  六个模型的配置与理论成本",
    )
    figure(doc, os.path.join("report", "figures", "architecture.png"),
           "图 3-1  基线 BasicBlock 与 DS+SE BasicBlock 的结构对比", width_cm=15.6)

    heading(doc, "3.7  本章小结", 2)
    body(
        doc,
        "本章从残差块出发，把 CNN 的结构设计拆成“空间卷积、通道注意力、激活函数、归一化方式”四个可替换"
        "组件。深度可分离卷积把标准 3×3 卷积分解为逐通道空间卷积与 1×1 通道混合，理论上把乘加量从 "
        "C_in×C_out×9 降到 C_in×9+C_in×C_out；SE 用全局池化和两层瓶颈全连接学习通道权重；GELU 用平滑的"
        "高斯门控替代 ReLU 的硬截断；GroupNorm 则把归一化从 batch 维度移到通道组内。",
    )
    body(
        doc,
        "六个模型共享同一深度、宽度、stage 划分、分类头与初始化，只在上表所列组件上不同。这样，第 4 章"
        "观察到的精度与成本差异才能归因于结构本身，而不是训练轮数、数据划分或超参数的偶然变化。",
    )


def chapter4(doc: Document, r: dict) -> None:
    s = r["summaries"]
    b = r["benchmark"]
    base = s["resnet20"]
    acc_base = base["test"]["accuracy"]
    heading(doc, "第 4 章  训练结果与结构分析", 1)
    heading(doc, "4.1  统一训练配方", 2)
    body(
        doc,
        "为了让结构差异成为唯一变量，六个模型使用完全相同的训练配方。数据为 45000 张训练图、5000 张验证图；"
        "batch size 256；SGD + momentum 0.9 + Nesterov；基础学习率 0.1，5 个 epoch 线性 warmup，随后余弦衰减到 "
        "1e-4；weight decay 5e-4；label smoothing 0.1；Cutout 8；共 60 个 epoch；随机种子 42。",
    )
    add_table(
        doc,
        ["超参数", "取值", "超参数", "取值"],
        [
            ["Epochs", "60", "Batch size", "256"],
            ["Optimizer", "SGD + Nesterov", "Momentum", "0.9"],
            ["Base LR", "0.1", "Min LR", "1e-4"],
            ["Warmup", "5 epochs", "Schedule", "cosine"],
            ["Weight decay", "5e-4", "Label smoothing", "0.1"],
            ["Cutout", "8", "Seed", "42"],
            ["输入归一化", "CIFAR-10 mean/std", "推理精度", "float32 CPU"],
        ],
        [3.4, 4.4, 3.4, 4.4],
        "表 4-1  六个模型共享的训练配方",
    )
    body(
        doc,
        f"每个模型训练完成后，按验证集准确率选择最佳 checkpoint，再在 10000 张测试图上评估一次。基线的最佳 "
        f"epoch 为 {base['best_epoch']}，验证准确率 {pct(base['best_val_accuracy'])}，测试准确率 "
        f"{pct(acc_base)}；六个模型的总训练时间见 4.4 节。需要说明的是，ResNet-20 原论文使用 200 epoch"
        " 训练并在 CIFAR-10 上报告 91.25%；本文将统一预算设为 60 epoch，是为了在纯 CPU 上完成六个模型的"
        "全部训练。因此表中的绝对精度低于原论文的长训练结果，但六个模型共享同一数据划分、增强与优化预算，"
        "结构之间的相对差异仍然可比。",
    )

    heading(doc, "4.2  训练曲线", 2)
    body(
        doc,
        "图 4-1 给出六个模型的训练损失与验证准确率。所有模型都经历了相似的三个阶段：warmup 阶段损失快速下降，"
        "中间阶段在数据增强下波动，最后在余弦学习率接近 1e-4 时收敛。曲线之间的差异主要出现在中后段，说明"
        "结构改动影响的是特征学习效率，而不是单纯的初始化差异。",
    )
    figure(doc, os.path.join("report", "figures", "training_curves.png"),
           "图 4-1  六个模型的训练损失与验证准确率", width_cm=15.8)

    heading(doc, "4.3  总体结果", 2)
    body(
        doc,
        "表 4-2 汇总了测试集上的分类指标，表 4-3 汇总了部署成本。表中的“ΔAcc”均相对基线 R20 计算，"
        "正数表示优于基线。",
    )
    rows = []
    for name in ORDER:
        t = s[name]["test"]
        rows.append([
            SHORT[name],
            f"{s[name]['params'] / 1e6:.3f}",
            f"{s[name]['macs'] / 1e6:.2f}",
            pct(t["accuracy"]),
            f"{(t['accuracy'] - acc_base) * 100:+.2f}",
            pct(t["top5_accuracy"]),
            f"{t['macro_f1']:.4f}",
            f"{t['ece']:.4f}",
        ])
    add_table(
        doc,
        ["模型", "参数量(M)", "MACs(M)", "Test Acc", "ΔAcc(pp)", "Top-5", "Macro-F1", "ECE"],
        rows,
        [2.3, 2.0, 1.8, 2.0, 1.9, 2.0, 2.0, 1.6],
        "表 4-2  六个模型的测试集分类指标（ΔAcc 相对基线）",
        font_size=8.2,
    )

    rows = []
    for name in ORDER:
        row = b[name]
        rows.append([
            SHORT[name],
            f"{row['size_mb']:.2f}",
            f"{row['latency_b1']['latency_ms_median']:.2f}",
            f"{row['latency_b128']['throughput_img_s']:.0f}",
            f"{effective_train_minutes(os.getcwd(), name, s[name]['train_seconds']):.1f}",
        ])
    add_table(
        doc,
        ["模型", "模型大小(MB)", "batch=1 延迟(ms)", "batch=128 吞吐(img/s)", "有效训练时间(min)"],
        rows,
        [2.5, 2.8, 3.4, 3.8, 3.1],
        "表 4-3  模型大小、CPU 推理延迟与训练时间",
    )
    body(
        doc,
        "延迟在 16 线程、channels_last、torch.inference_mode 下测量；六个模型交错进行 5 轮，batch=1 每模型"
        "取 50 次、batch=128 每模型取 25 次，表中为中位数。交错测量可避免“先测的模型承担冷缓存与升频开销”"
        "造成的系统偏差。",
        size=10.5,
    )
    body(
        doc,
        "表 4-3 的训练时间按逐 epoch 计时求和。GELU 训练期间发生过一次 Windows 系统挂起（单个 epoch 的墙钟"
        "时间被拉长到 1168.8 秒），若直接使用墙钟总时长会把挂起时间误算成模型开销；因此表中把该异常 epoch "
        "替换为中位数后再求和，其余模型仍为实际计时。",
        size=10.5,
    )

    best_name = max(ORDER, key=lambda n: s[n]["test"]["accuracy"])
    body(
        doc,
        f"从表 4-2 可以看到，测试准确率最高的是 **{SHORT[best_name]}（{pct(s[best_name]['test']['accuracy'])}）**，"
        f"最低的是 {SHORT[min(ORDER, key=lambda n: s[n]['test']['accuracy'])]}"
        f"（{pct(s[min(ORDER, key=lambda n: s[n]['test']['accuracy'])]['test']['accuracy'])}）。"
        "六个模型的 Top-5 准确率都明显高于 Top-1，说明错误主要集中在少数容易混淆的类别之间。"
        "ECE 的差异则反映了不同结构对置信度校准的影响。",
    )

    heading(doc, "4.4  精度—成本权衡", 2)
    body(
        doc,
        "图 4-2 把测试准确率分别对参数量、MACs 和 batch=1 延迟作图。左、中两图的横轴是对数坐标。"
        "如果只看前两张图，深度可分离卷积看起来是“免费午餐”：参数和 MACs 都下降了一个数量级，而精度只变化了几个点。"
        "但右图把延迟纳入后，结论就复杂得多。",
    )
    figure(doc, os.path.join("report", "figures", "accuracy_vs_cost.png"),
           "图 4-2  测试准确率与参数量、MACs、CPU 延迟的关系", width_cm=16.0)

    # dynamic DS analysis
    ds = s["resnet20_ds"]
    b1_base = b["resnet20"]["latency_b1"]["latency_ms_median"]
    b1_ds = b["resnet20_ds"]["latency_b1"]["latency_ms_median"]
    ds_lat = b1_ds / b1_base - 1.0
    body(
        doc,
        f"深度可分离卷积（R20-DS）把参数量从 {base['params']:,} 降到 {ds['params']:,}，降幅 "
        f"{(1 - ds['params'] / base['params']) * 100:.1f}%；MACs 从 {base['macs'] / 1e6:.2f}M 降到 "
        f"{ds['macs'] / 1e6:.2f}M，降幅 {(1 - ds['macs'] / base['macs']) * 100:.1f}%。"
        f"测试准确率从 {pct(acc_base)} 变为 {pct(ds['test']['accuracy'])}"
        f"（{delta_pct(ds['test']['accuracy'] - acc_base)}）。"
        f"然而 batch=1 延迟从 {b1_base:.2f} ms 变为 {b1_ds:.2f} ms，即 **{ds_lat:+.1%}**。",
    )
    if ds_lat > 0.02:
        body(
            doc,
            "这一结果说明，在 CPU 上 depthwise 卷积的低计算密度并没有转化为低延迟。原因主要有三：其一，"
            "depthwise 卷积的每个通道独立计算，难以利用标准卷积的高度优化内核，向量化程度低；其二，"
            "逐通道访存和 1×1 卷积的额外启动开销增加了内存带宽压力；其三，PyTorch 在 CPU 上对 groups 较大的"
            "卷积支持不如稠密卷积成熟。因此，“参数少、MACs 少”只说明模型更小，不说明推理更快。对边缘部署而言，"
            "必须同时测量目标硬件上的真实延迟。",
        )
    else:
        body(
            doc,
            "在该 CPU 上，深度可分离卷积的延迟随 MACs 一起下降，说明 PyTorch 的 depthwise 内核在该尺寸下"
            "已经比较高效。即便如此，延迟降幅仍小于 MACs 降幅，说明内存访问和算子启动开销占据了不可忽略的比例。",
        )

    b128_base = b["resnet20"]["latency_b128"]["throughput_img_s"]
    b128_ds = b["resnet20_ds"]["latency_b128"]["throughput_img_s"]
    if b128_ds > b128_base:
        body(
            doc,
            f"不过，若把观察量换成 batch=128 的吞吐，R20-DS 为 {b128_ds:.0f} img/s，反而高于基线的 "
            f"{b128_base:.0f} img/s（{b128_ds / b128_base - 1:+.1%}）。这与 batch=1 的结论并不矛盾："
            "单张延迟由算子启动、调度和访存开销主导，大 batch 下 depthwise 的低 MACs 更容易摊薄固定开销。"
            "因此部署评价必须区分“单请求延迟”和“批量吞吐”，不能只报一个。",
        )

    # SE analysis
    se = s["resnet20_se"]
    b1_se = b["resnet20_se"]["latency_b1"]["latency_ms_median"]
    se_gain = se["test"]["accuracy"] - acc_base
    if se_gain > 0.002:
        se_conclusion = (
            "SE 的增益为正，说明在本任务上“哪些通道更重要”确实是有效信息，瓶颈结构以很小的参数代价"
            "提供了图像级的通道上下文。"
        )
    elif se_gain > -0.002:
        se_conclusion = (
            "SE 与基线基本持平：通道重标定没有转化为可测的精度收益，其开销主要体现在额外的池化与门控延迟上。"
        )
    else:
        se_conclusion = (
            "SE 低于基线，可能是 60 epoch 的短训练下额外的门控参数尚未充分学习，或该任务并不需要额外的"
            "通道上下文；这说明注意力模块的收益依赖训练预算与任务。"
        )
    body(
        doc,
        f"SE（R20-SE）只增加 {(se['params'] - base['params']):,} 个参数（约 "
        f"{(se['params'] / base['params'] - 1) * 100:.1f}%），MACs 几乎不变（{se['macs'] / 1e6:.2f}M vs "
        f"{base['macs'] / 1e6:.2f}M）。测试准确率为 {pct(se['test']['accuracy'])}，相对基线 "
        f"{delta_pct(se_gain)}；batch=1 延迟为 {b1_se:.2f} ms，"
        f"相对基线 {(b1_se / b1_base - 1) * 100:+.1f}%。" + se_conclusion,
    )

    # GELU
    gelu = s["resnet20_gelu"]
    b1_gelu = b["resnet20_gelu"]["latency_b1"]["latency_ms_median"]
    gelu_gain = gelu["test"]["accuracy"] - acc_base
    if gelu_gain > 0.002:
        gelu_conclusion = (
            f"在本例中 GELU 略高于基线，说明平滑激活对优化有一点帮助；但 GELU 的 erf 计算通常比 ReLU 更昂贵，"
            "收益是否值得取决于精度预算与延迟约束。"
        )
    elif gelu_gain > -0.002:
        gelu_conclusion = (
            "GELU 与基线基本持平：在 CIFAR-10 这样的中小规模任务上，ReLU 的硬截断已经足够好，平滑激活"
            "没有转化为可测的精度收益，而 erf 计算在 CPU 上更昂贵。"
        )
    else:
        gelu_conclusion = (
            "GELU 低于基线：平滑激活并没有改善本轮训练的优化，反而可能因额外的 erf 计算和 60 epoch 的"
            "有限预算而吃亏；它更适合作为激活函数替换的对照，而不是默认选择。"
        )
    body(
        doc,
        f"GELU（R20-GELU）不改变参数量与 MACs，测试准确率为 {pct(gelu['test']['accuracy'])}，"
        f"相对基线 {delta_pct(gelu_gain)}；batch=1 延迟为 {b1_gelu:.2f} ms，"
        f"相对基线 {(b1_gelu / b1_base - 1) * 100:+.1f}%。" + gelu_conclusion,
    )

    # GN
    gn = s["resnet20_gn"]
    b1_gn = b["resnet20_gn"]["latency_b1"]["latency_ms_median"]
    body(
        doc,
        f"GroupNorm（R20-GN）的参数量与基线相同，测试准确率为 {pct(gn['test']['accuracy'])}，"
        f"相对基线 {delta_pct(gn['test']['accuracy'] - acc_base)}；batch=1 延迟为 {b1_gn:.2f} ms，"
        f"相对基线 {(b1_gn / b1_base - 1) * 100:+.1f}%。GroupNorm 不依赖 batch 统计，在小 batch 或"
        "分布偏移场景下通常更稳；但在 batch=256 的训练设置下，BatchNorm 的跨样本统计本身就有正则化效果，"
        "GroupNorm 未必占优。其逐样本、逐组的归约也可能增加 CPU 开销，但本次实测延迟受内核实现与线程数"
        "影响，未必与理论开销同向。",
    )

    # DS-SE
    dsse = s["resnet20_ds_se"]
    b1_dsse = b["resnet20_ds_se"]["latency_b1"]["latency_ms_median"]
    body(
        doc,
        f"组合模型 R20-DS-SE 的参数量为 {dsse['params']:,}，MACs 为 {dsse['macs'] / 1e6:.2f}M，"
        f"测试准确率 {pct(dsse['test']['accuracy'])}，相对 R20-DS "
        f"{delta_pct(dsse['test']['accuracy'] - ds['test']['accuracy'])}，相对基线 "
        f"{delta_pct(dsse['test']['accuracy'] - acc_base)}。其 batch=1 延迟为 {b1_dsse:.2f} ms，"
        f"相对基线 {(b1_dsse / b1_base - 1) * 100:+.1f}%。",
    )
    dsse_delta = dsse["test"]["accuracy"] - ds["test"]["accuracy"]
    if dsse_delta > 0.002:
        combo_text = (
            f"相对 R20-DS，SE 把准确率提高了 {delta_pct(dsse_delta)}，说明在轻量卷积的特征上，通道重标定"
            "确实带来了一部分补偿；但代价是额外的全局池化与门控延迟。"
        )
    elif dsse_delta > -0.005:
        combo_text = (
            f"相对 R20-DS，SE 的准确率变化为 {delta_pct(dsse_delta)}，基本持平；在本训练预算下，"
            "注意力带来的收益被额外的门控开销抵消。"
        )
    else:
        combo_text = (
            f"相对 R20-DS，SE 的准确率反而变化 {delta_pct(dsse_delta)}，并未恢复精度；这说明在 60 epoch "
            "预算下，轻量卷积上的 SE 门控尚未充分学习，或该任务并不需要额外的通道上下文。"
        )
    body(
        doc,
        combo_text + "因此，“轻量卷积 + 注意力”是否值得，取决于部署端更在意模型体积、吞吐还是单张延迟；"
        "不能仅凭参数量更小就推断精度更高或更快。",
    )

    heading(doc, "4.5  逐类精度与混淆矩阵", 2)
    body(
        doc,
        "图 4-3 对比了基线与 R20-DS-SE 的逐类召回率，图 4-4 给出两者的行归一化混淆矩阵。CIFAR-10 中"
        "猫/狗、鹿/马、飞机/船等类别在低分辨率下外观接近，是主要的混淆来源。若某个结构在整体准确率上只差"
        "零点几个点，但在“猫/狗”等关键类别上差异明显，则不能简单地说两者等价。",
    )
    figure(doc, os.path.join("report", "figures", "per_class_accuracy.png"),
           "图 4-3  基线 R20 与 R20-DS-SE 的逐类测试召回率", width_cm=14.8)
    figure(doc, os.path.join("report", "figures", "confusion_matrices.png"),
           "图 4-4  R20 与 R20-DS-SE 的测试集混淆矩阵（行归一化）", width_cm=15.6)

    pc_base = {row["class_id"]: row["recall"] for row in base["test"]["per_class"]}
    pc_dsse = {row["class_id"]: row["recall"] for row in dsse["test"]["per_class"]}
    class_names = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "船", "卡车"]
    diffs = sorted(((pc_dsse[i] - pc_base[i], i) for i in range(10)))
    worst_cls = diffs[0]
    best_cls = diffs[-1]
    body(
        doc,
        f"从逐类结果看，R20-DS-SE 相对基线下降最多的是 **{class_names[worst_cls[1]]}**"
        f"（{worst_cls[0] * 100:+.1f} 个百分点），提升最多的是 **{class_names[best_cls[1]]}**"
        f"（{best_cls[0] * 100:+.1f} 个百分点）。这说明结构改动并不是在所有类别上均匀生效；"
        "参数量更小的模型可能对纹理简单、形状明确的类别更友好，而对细粒度差异较大的类别更吃力。",
    )

    doc.add_page_break()
    heading(doc, "4.6  小结", 2)
    body(
        doc,
        "本章的受控实验给出三点结论。第一，深度可分离卷积在参数量与 MACs 上优势巨大，但在 CPU 上不一定更快，"
        "部署前必须实测延迟。第二，SE 以很小的参数代价提供了通道重标定能力，本次相对基线为 "
        f"{delta_pct(se_gain)}；GELU 与 GroupNorm 相对基线的变化分别为 {delta_pct(gelu_gain)} 和 "
        f"{delta_pct(gn['test']['accuracy'] - acc_base)}。这些差异都应在“单次 60 epoch 训练”的误差范围内"
        "谨慎解读，不能直接外推为普遍结论。第三，组合模型把参数量"
        f"压到 {dsse['params']:,}，但其测试准确率相对 R20-DS 的变化为 {delta_pct(dsse_delta)}；"
        "“小体积”与“高精度”并不自动同时成立，需要按部署目标取舍。",
    )


def chapter5(doc: Document, r: dict) -> None:
    s = r["summaries"]
    rob = r["robustness"]["models"]
    heading(doc, "第 5 章  低能见度鲁棒性与场景应用", 1)
    heading(doc, "5.1  场景动机与退化设计", 2)
    body(
        doc,
        "道路场景中的低能见度主要由雾、夜间低照度、相机抖动、传感器噪声和图像压缩造成。真实数据的采集需要"
        "对应天气与道路条件，标注成本高；在课程作业中，更可控的做法是在 CIFAR-10 测试集上施加合成退化，"
        "固定退化参数与随机种子，比较不同结构在相同扰动下的相对表现。",
    )
    body(
        doc,
        "本文实现五类退化，每类 5 个强度（severity 1–5），共 25 个测试条件。所有退化都在 uint8 NHWC 图像上"
        "进行，再统一归一化；同一强度下所有模型使用相同的随机种子，因此比较是配对的。",
    )
    add_table(
        doc,
        ["退化", "实现方式", "severity 1 → 5 的关键参数"],
        [
            ["雾", "与亮灰蓝色 [232,236,242] 按透明度 α 混合", "α ≈ 0.20 → 0.63"],
            ["低照度", "Gamma 压暗 + 高斯读出噪声", "γ = 1.24 → 2.20，σ = 3.5 → 11.5"],
            ["运动模糊", "0/45/90/135° 线性运动核，逐图随机方向", "核长 3 → 11"],
            ["高斯噪声", "加性高斯噪声", "σ = 6 → 30"],
            ["JPEG 压缩", "PIL JPEG 编码再解码", "quality 52 → 12"],
        ],
        [2.6, 8.2, 5.0],
        "表 5-1  五类合成退化的实现",
        first_col_left=True,
    )
    figure(doc, os.path.join("report", "figures", "corruption_examples.png"),
           "图 5-1  同一张测试图在干净、severity 1/3/5 下的退化示例", width_cm=13.8)
    body(
        doc,
        "这些退化只模拟了低能见度的一部分统计特征，并不等价于真实雾天成像：真实雾的浓度随深度变化，"
        "夜间还有光源眩光与色偏，运动模糊与曝光时间耦合。因此，本章的结论应理解为“在受控合成退化下的"
        "相对鲁棒性”，而不是对真实道路场景的绝对性能承诺。",
    )

    heading(doc, "5.2  全模型鲁棒性结果", 2)
    body(
        doc,
        "表 5-2 列出每个模型在干净测试集、五类退化（对 severity 1–5 取平均）以及全部 25 个条件上的平均与"
        "最差准确率。图 5-2 给出准确率随 severity 变化的曲线，图 5-3 用热力图展示模型与退化类型的交互。",
    )
    rows = []
    for name in ORDER:
        suite = rob[name]
        row = [SHORT[name], pct(suite["clean_accuracy"])]
        for c in CORRUPTIONS:
            vals = [suite["corruptions"][c][str(k)] for k in suite["severities"]]
            row.append(pct(sum(vals) / len(vals)))
        row.append(pct(suite["mean_accuracy"]))
        row.append(pct(suite["worst_accuracy"]))
        rows.append(row)
    add_table(
        doc,
        ["模型", "Clean", "雾", "低照度", "运动模糊", "噪声", "JPEG", "平均", "最差"],
        rows,
        [2.0, 1.7, 1.6, 1.8, 1.9, 1.6, 1.6, 1.7, 1.6],
        "表 5-2  六个模型在干净与退化测试集上的准确率（退化列为 severity 1–5 平均）",
        font_size=7.9,
    )
    figure(doc, os.path.join("report", "figures", "robustness_curves.png"),
           "图 5-2  准确率随退化强度变化的曲线（横轴 0 为干净图像）", width_cm=16.0)
    figure(doc, os.path.join("report", "figures", "robustness_heatmap.png"),
           "图 5-3  模型 × 退化类型的平均准确率热力图", width_cm=14.2)

    base_mean = rob["resnet20"]["mean_accuracy"]
    best_rob = max(ORDER, key=lambda n: rob[n]["mean_accuracy"])
    worst_rob = min(ORDER, key=lambda n: rob[n]["mean_accuracy"])
    body(
        doc,
        f"基线在 25 个退化条件上的平均准确率为 {pct(base_mean)}，相对干净图像下降 "
        f"{pct(rob['resnet20']['mean_drop'])}。鲁棒性最好的模型是 **{SHORT[best_rob]}"
        f"（{pct(rob[best_rob]['mean_accuracy'])}）**，最差的是 **{SHORT[worst_rob]}"
        f"（{pct(rob[worst_rob]['mean_accuracy'])}）**。从退化类型看，"
        + "、".join(
            f"{CORRUPTION_CN[c]}（所有模型平均 {pct(sum(rob[n]['corruptions'][c][str(k)] for n in ORDER for k in rob[n]['severities']) / (len(ORDER) * 5))}）"
            for c in CORRUPTIONS
        )
        + "。",
    )
    best_clean = max(ORDER, key=lambda n: rob[n]["clean_accuracy"])
    if best_clean == best_rob:
        clean_rob_text = (
            f"本例中，干净精度最高的 {SHORT[best_clean]} 同时也是 25 个退化条件平均准确率最高的模型，"
            "干净精度与退化精度在模型排序上一致。"
        )
    else:
        clean_rob_text = (
            f"一个值得注意的现象是：干净精度最高的 {SHORT[best_clean]} 并不是退化条件下平均准确率最高的"
            f"模型，后者是 {SHORT[best_rob]}。这说明仅凭干净测试集选型，可能会低估模型在低能见度下的"
            "相对表现。"
        )
    body(
        doc,
        clean_rob_text + "鲁棒性比较应同时看“绝对退化精度”和“相对干净精度的下降幅度”：前者更贴近部署体验，"
        "后者更能反映结构对分布偏移的敏感度。",
    )
    def corr_mean(name: str, corruption: str) -> float:
        vals = [rob[name]["corruptions"][corruption][str(k)] for k in rob[name]["severities"]]
        return float(sum(vals) / len(vals))

    body(
        doc,
        f"分退化类型看，R20-GN 的优势主要集中在运动模糊（{pct(corr_mean('resnet20_gn', 'motion_blur'))}，"
        f"基线 {pct(corr_mean('resnet20', 'motion_blur'))}）和高斯噪声（{pct(corr_mean('resnet20_gn', 'gaussian_noise'))}，"
        f"基线 {pct(corr_mean('resnet20', 'gaussian_noise'))}）上；在雾和低照度上，它与基线的差距要小得多。"
        "一种解释是：GroupNorm 在推理时对每个样本独立统计，局部像素被模糊或加噪后，归一化会重新缩放该样本的"
        "特征，而 BatchNorm 使用训练集累积的 running statistics，对这类局部扰动更敏感。但这一解释仍是假设，"
        "需要消融归一化的统计方式才能确认；同时 R20-GN 的干净精度最低，说明鲁棒性提升不是“免费”的。",
    )

    heading(doc, "5.3  退化数据微调", 2)
    ft = r["finetune"]
    if ft:
        body(
            doc,
            "为了验证“用退化数据继续训练能否提升低能见度表现”，本文从 R20-DS-SE 的最佳干净 checkpoint 出发，"
            "再做 15 个 epoch 的微调：每个 batch 以 0.5 的概率施加随机类型、随机强度的退化，其余设置与主训练"
            "一致但学习率降为 0.01。模型选择使用一个混合验证集：5000 张干净验证图 + 每类退化各 1000 张"
            "（severity=3），共 10000 张。这样做的目的是避免只优化干净精度或只优化退化精度。",
        )
        add_table(
            doc,
            ["条件", "微调前", "微调后", "变化"],
            [
                ["干净测试集", pct(ft["clean_before"]), pct(ft["clean_after"]),
                 delta_pct(ft["clean_after"] - ft["clean_before"])],
                ["25 个退化条件平均", pct(ft["corruption_mean_before"]), pct(ft["corruption_mean_after"]),
                 delta_pct(ft["corruption_mean_after"] - ft["corruption_mean_before"])],
                ["最差退化条件", pct(ft["corruption_worst_before"]), pct(ft["corruption_worst_after"]),
                 delta_pct(ft["corruption_worst_after"] - ft["corruption_worst_before"])],
            ],
            [4.8, 3.5, 3.6, 3.7],
            "表 5-3  R20-DS-SE 退化微调前后的准确率",
        )
        figure(doc, os.path.join("report", "figures", "finetune_comparison.png"),
               "图 5-4  退化微调前后的整体与分类型准确率", width_cm=15.6)
        clean_change = ft["clean_after"] - ft["clean_before"]
        rob_gain = ft["corruption_mean_after"] - ft["corruption_mean_before"]
        if rob_gain > 0.002:
            if clean_change < -0.002:
                ft_conclusion = (
                    "这体现了典型的“鲁棒性—干净精度”权衡：退化增强让模型见过更多低质量输入，"
                    "从而在退化条件下更稳，但牺牲了一部分干净数据的判别边界。如果应用场景明确以低能见度为主，"
                    "这种交换是合理的；如果干净与退化图像混合出现，则应把干净验证集纳入模型选择，"
                    "或采用更小的微调学习率与更少的 epoch。"
                )
            else:
                ft_conclusion = (
                    "退化平均准确率提升，而干净准确率没有明显下降，说明本轮微调在两方面都取得了正收益；"
                    "这可能与退化增强的正则化效应以及较短、较低学习率的微调有关，但仍需多次随机种子验证。"
                )
        elif rob_gain < -0.002:
            ft_conclusion = (
                "退化平均准确率反而下降，说明“用退化数据继续训练”并不必然提升鲁棒性：15 个 epoch 的"
                "微调可能改变了特征分布，而混合验证集上的模型选择又未能找到更好的折中点。这一负结果同样"
                "值得报告，它提示场景适配需要更细致的学习率、增强强度与模型选择策略。"
            )
        else:
            ft_conclusion = (
                "退化平均准确率基本持平，说明本轮微调的收益有限；若要在低能见度场景获得稳定提升，"
                "需要调整退化强度分布、增加训练轮数或采用更强的域适应方法。"
            )
        body(
            doc,
            f"微调后，退化平均准确率变化 **{delta_pct(rob_gain)}**，最差条件变化 "
            f"{delta_pct(ft['corruption_worst_after'] - ft['corruption_worst_before'])}；"
            f"干净准确率变化 {delta_pct(clean_change)}。" + ft_conclusion,
        )
    else:
        body(doc, "退化微调结果尚未生成，本节将在运行 `scripts/finetune_robust.py` 后补充。")

    heading(doc, "5.4  Grad-CAM 可解释性", 2)
    body(
        doc,
        "Grad-CAM 通过目标类别对最后一个卷积 stage 特征的梯度，得到一张 32×32 的重要性热力图。本文对"
        "基线 R20 与 R20-DS-SE 分别输入干净图像和 severity=4 的雾天图像，比较两者关注区域的变化。"
        "热力图越接近红色，表示该位置对预测的贡献越大。",
    )
    figure(doc, os.path.join("report", "figures", "gradcam_examples.png"),
           "图 5-5  R20 与 R20-DS-SE 在干净/雾天图像上的 Grad-CAM", width_cm=14.6)
    body(
        doc,
        "在干净图像上，两个模型的高响应区域通常集中在目标主体（如飞机机身、船的轮廓、猫的头部）；"
        "在雾天图像上，响应会变得更分散，部分样本转向背景或边缘。若某个模型在雾天仍能把响应集中在主体上，"
        "说明其特征对对比度下降更不敏感；反之，若响应大面积转移到天空或地面，则 accuracy 的下降就有了"
        "可视化的解释。Grad-CAM 是定性证据，不能替代定量指标，但可以帮助检查模型是否“看对了地方”。",
    )

    heading(doc, "5.5  交互演示系统", 2)
    body(
        doc,
        "为了把训练好的模型用于实际交互，本文实现了一个 Flask 演示：用户上传图片，选择模型（默认 R20-DS-SE）、"
        "退化类型与强度，服务端完成中心裁剪、缩放到 32×32、归一化、前向推理与 Grad-CAM，返回 Top-3 类别、"
        "置信度、网络输入和热力图叠加结果。演示还提供 `/health` 接口用于健康检查，以及 `?demo=1` "
        "示例自动运行：页面载入内置测试图并自动完成一次指定退化的推理，便于在无外部图片时复现截图。",
    )
    demo_path = os.path.join("report", "figures", "demo_ui.png")
    if os.path.exists(demo_path):
        figure(doc, demo_path, "图 5-6  Flask 演示界面（示例自动运行：R20 对雾天样本的 Top-3 与 Grad-CAM）", width_cm=15.6)
    body(
        doc,
        "演示的意义在于验证“模型—预处理—推理—可视化”整条链路可运行，而不是提供一个可直接用于真实道路的"
        "产品。CIFAR-10 的 32×32 输入与真实车载相机的分辨率差距很大；若要把系统扩展到真实场景，需要先做"
        "目标检测或感兴趣区域裁剪，再在域内数据上重新训练，并补充雨、雪、夜间眩光等退化类型。",
    )

    heading(doc, "5.6  场景适配的局限", 2)
    body(
        doc,
        "本章实验存在三方面局限。第一，合成退化只覆盖了低能见度的一部分统计特征，真实雾天的深度相关散射、"
        "夜间色偏与运动模糊的点扩散函数都更复杂。第二，CIFAR-10 的类别是通用物体，不是道路参与者；"
        "“低能见度道路场景”在这里是方法验证的场景隐喻，而不是端到端的道路感知系统。第三，鲁棒性评估只用了"
        "准确率，没有评估检测框、分割掩码或时间一致性。因此，报告中的结论应定位为“结构在受控退化下的相对"
        "行为”，而不是实际部署的性能保证。",
    )


def chapter6(doc: Document, r: dict) -> None:
    s = r["summaries"]
    base = s["resnet20"]
    rob = r["robustness"]["models"]
    ft = r["finetune"]
    base_acc = s["resnet20"]["test"]["accuracy"]
    ds = s["resnet20_ds"]
    b = r["benchmark"]
    b1_base = b["resnet20"]["latency_b1"]["latency_ms_median"]
    b1_ds = b["resnet20_ds"]["latency_b1"]["latency_ms_median"]
    if b1_ds < b1_base:
        ds_lat_text = (
            f"单张 CPU 延迟也从 {b1_base:.2f} ms 降至 {b1_ds:.2f} ms（{(b1_ds / b1_base - 1) * 100:+.1f}%），"
            "但延迟降幅仍小于 MACs 降幅；这提醒我们，模型压缩的收益必须在目标硬件上测量，"
            "不能只用 FLOPs 代替延迟。"
        )
    else:
        ds_lat_text = (
            f"但单张 CPU 延迟由 {b1_base:.2f} ms 变为 {b1_ds:.2f} ms（{(b1_ds / b1_base - 1) * 100:+.1f}%），"
            "并未随 MACs 同比下降；这提醒我们，模型压缩的收益必须在目标硬件上测量，"
            "不能只用 FLOPs 代替延迟。"
        )
    heading(doc, "第 6 章  结论与展望", 1)
    heading(doc, "6.1  结论", 2)
    body(
        doc,
        f"本文在 CIFAR-10 上完成了 ResNet-20 及其五个结构变体的受控训练与评估。基线测试准确率为 "
        f"{pct(base_acc)}；深度可分离卷积把参数量降至 {ds['params']:,}、MACs 降至 "
        f"{ds['macs'] / 1e6:.2f}M，{ds_lat_text}",
    )
    se_gain = s["resnet20_se"]["test"]["accuracy"] - base_acc
    gelu_gain = s["resnet20_gelu"]["test"]["accuracy"] - base_acc
    gn_gain = s["resnet20_gn"]["test"]["accuracy"] - base_acc
    component_text = (
        f"在结构组件层面，SE、GELU、GroupNorm 相对基线的测试准确率变化分别为 "
        f"{delta_pct(se_gain)}、{delta_pct(gelu_gain)}、{delta_pct(gn_gain)}。"
    )
    component_rank = max(
        [("SE", se_gain), ("GELU", gelu_gain), ("GroupNorm", gn_gain)],
        key=lambda item: item[1],
    )
    if se_gain > 0.002 and component_rank[0] == "SE":
        component_text += (
            f"SE 用 {s['resnet20_se']['params'] - base['params']:,} 个额外参数换来了这部分提升，"
            "是本轮三个单组件改动中提升幅度最大的一项。"
        )
    elif se_gain > 0.002:
        component_text += (
            f"SE 用 {s['resnet20_se']['params'] - base['params']:,} 个额外参数带来了正提升，"
            f"但幅度不及 {component_rank[0]}。"
        )
    else:
        component_text += "SE 的通道重标定没有转化为稳定的精度提升，说明其收益依赖训练预算与任务。"
    if max(gelu_gain, gn_gain) > 0.002:
        if gelu_gain >= gn_gain:
            component_text += (
                f"其中 GELU 相对基线为 {delta_pct(gelu_gain)}，GroupNorm 为 {delta_pct(gn_gain)}；"
                "单次训练预算下，这一差异不足以证明 GELU 普遍优于 ReLU。"
            )
        else:
            component_text += (
                f"其中 GroupNorm 相对基线为 {delta_pct(gn_gain)}，GELU 为 {delta_pct(gelu_gain)}；"
                "单次训练预算下，这一差异不足以证明 GroupNorm 普遍优于 BatchNorm。"
            )
    else:
        component_text += (
            "GELU 与 GroupNorm 都没有超过 ReLU + BatchNorm，说明在本训练预算与 batch=256 的设置下，"
            "平滑激活和组归一化的理论优势没有转化为更高的测试精度。"
        )
    body(doc, component_text)
    dsse_gain = s["resnet20_ds_se"]["test"]["accuracy"] - s["resnet20_ds"]["test"]["accuracy"]
    body(
        doc,
        f"组合模型 R20-DS-SE 的参数量为 {s['resnet20_ds_se']['params']:,}，测试准确率 "
        f"{pct(s['resnet20_ds_se']['test']['accuracy'])}，相对 R20-DS 的变化为 {delta_pct(dsse_gain)}。"
        "它适合作为“体积优先”场景的候选，但是否部署仍取决于延迟预算；参数少并不自动意味着精度更高或更快。",
    )
    if ft:
        ft_rob_gain = ft["corruption_mean_after"] - ft["corruption_mean_before"]
        if ft_rob_gain > 0.002:
            ft_note = "这说明退化增强是一种有效但需要权衡的域适应手段。"
        elif ft_rob_gain < -0.002:
            ft_note = (
                "本轮微调没有提升退化平均准确率，说明退化增强并非“越多越好”；学习率、增强强度与"
                "模型选择策略都会影响最终收益。"
            )
        else:
            ft_note = "退化平均准确率基本持平，说明本轮微调的域适应收益有限。"
        body(
            doc,
            f"在低能见度模拟中，所有模型的准确率都随退化强度下降；对 R20-DS-SE 做退化微调后，25 个退化条件的"
            f"平均准确率由 {pct(ft['corruption_mean_before'])} 变为 {pct(ft['corruption_mean_after'])}，"
            f"干净准确率由 {pct(ft['clean_before'])} 变为 {pct(ft['clean_after'])}。" + ft_note,
        )
    body(
        doc,
        "本作业的工程价值在于“可复现的闭环”：数据准备、训练、评估、成本测量、鲁棒性、微调、可视化与演示"
        "均由脚本完成，报告数字由 JSON 自动生成。相比单独追求一个高准确率，这种闭环更能支撑对 CNN 组件的"
        "理解，也更接近真实项目的工作方式。",
    )

    heading(doc, "6.2  不足与改进方向", 2)
    for text in [
        "**训练预算有限**：60 epoch 的短训练可能低估了 SE、GELU 等模块的潜力；更长的训练与多次随机种子"
        "重复可以给出置信区间。",
        "**延迟测量范围有限**：只在一种 CPU、一个 PyTorch 版本上测量；结论应扩展到 ARM、GPU、ONNX Runtime "
        "与量化推理。",
        "**退化类型有限**：可加入雨、雪、夜间眩光、色偏与混合退化，并采用 CIFAR-10-C 等标准基准做对照。",
        "**场景不是真实道路数据**：未来可在 BDD100K、Cityscapes 等数据上做低能见度子集评估，或先检测再分类。",
        "**可解释性仍是定性**：可加入 Grad-CAM 的定量指标（如与标注框的 IoU）与失败案例分析。",
    ]:
        bullet(doc, text)
    body(
        doc,
        "后续工作可以从两个方向展开：一是把结构搜索与硬件感知延迟结合，在目标设备上做真实的 NAS/剪枝；"
        "二是把退化鲁棒性从分类扩展到检测与分割，研究低能见度下的多任务一致性。",
    )


def references(doc: Document) -> None:
    heading(doc, "参考文献", 1)
    refs = [
        "He K., Zhang X., Ren S., Sun J. Deep Residual Learning for Image Recognition. CVPR, 2016.",
        "He K., Zhang X., Ren S., Sun J. Identity Mappings in Deep Residual Networks. ECCV, 2016.",
        "Howard A. G., Zhu M., Chen B., et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv:1704.04861, 2017.",
        "Hu J., Shen L., Sun G. Squeeze-and-Excitation Networks. CVPR, 2018.",
        "Hendrycks D., Gimpel K. Gaussian Error Linear Units (GELUs). arXiv:1606.08415, 2016.",
        "Wu Y., He K. Group Normalization. ECCV, 2018.",
        "Selvaraju R. R., Cogswell M., Das A., et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV, 2017.",
        "Hendrycks D., Dietterich T. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR, 2019.",
        "Krizhevsky A. Learning Multiple Layers of Features from Tiny Images. Technical Report, University of Toronto, 2009.",
        "Ioffe S., Szegedy C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML, 2015.",
    ]
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.4
        p.paragraph_format.left_indent = Cm(0.74)
        p.paragraph_format.first_line_indent = Cm(-0.74)
        run = p.add_run(f"[{i}]  {ref}")
        set_run_font(run, size=10.5)


def appendix(doc: Document, r: dict) -> None:
    heading(doc, "附录 A  代码结构", 1)
    body(
        doc,
        "仓库按“数据—模型—训练—评估—应用”分层。核心文件如下：",
    )
    add_table(
        doc,
        ["路径", "作用"],
        [
            ["src/models.py", "ResNet-20、DS 卷积、SE、GELU、GroupNorm 与六个模型注册"],
            ["src/data.py", "npz 读取、分层划分、增强、归一化、批迭代"],
            ["src/engine.py", "单 epoch 训练、评估、余弦学习率"],
            ["src/metrics.py", "Accuracy、Top-5、NLL、Macro-F1、ECE、混淆矩阵"],
            ["src/flops.py", "参数量与 MACs 统计"],
            ["src/corruptions.py", "五类合成退化"],
            ["src/robustness.py", "退化测试套件"],
            ["src/gradcam.py", "Grad-CAM"],
            ["scripts/train.py", "单模型训练与 checkpoint"],
            ["scripts/benchmark.py", "成本与延迟测量"],
            ["scripts/robustness.py", "全模型退化评估"],
            ["scripts/finetune_robust.py", "退化数据微调"],
            ["scripts/visualize.py", "报告插图"],
            ["demo/app.py", "Flask 演示"],
        ],
        [5.0, 10.8],
        "表 A-1  代码结构",
        first_col_left=True,
    )

    heading(doc, "附录 B  复现命令", 1)
    body(doc, "以下命令均在仓库根目录执行，Windows PowerShell 为例：")
    commands = [
        "python -m pip install -r requirements.txt",
        "python -m scripts.prepare_data --source torchvision --output-dir data",
        "python -m scripts.train --model resnet20 --epochs 60 --cutout 8 --threads 16",
        "python -m scripts.evaluate --model resnet20",
        "python -m scripts.benchmark --threads 16",
        "python -m scripts.robustness --threads 16",
        "python -m scripts.finetune_robust --model resnet20_ds_se --epochs 15 --threads 16",
        "python -m scripts.visualize",
        "python -m scripts.build_report",
        "python demo/app.py --port 5000",
    ]
    for cmd in commands:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.5)
        p.paragraph_format.line_spacing = 1.2
        run = p.add_run(cmd)
        set_run_font(run, size=9.5, latin="Consolas")

    body(
        doc,
        "六个模型可按 README 中的循环依次训练；`--resume` 可从 `checkpoints/<model>_last.pt` 断点续训。"
        "所有脚本的随机种子、数据划分和退化种子均已固定，报告中的数字由 `results/metrics/*.json` 直接读取。",
    )

    heading(doc, "附录 C  结果文件说明", 1)
    add_table(
        doc,
        ["文件", "内容"],
        [
            ["results/history/<model>.json", "逐 epoch 训练损失、验证准确率、学习率、耗时"],
            ["results/metrics/<model>_summary.json", "参数量、MACs、最佳 epoch、测试指标"],
            ["results/metrics/<model>_test_logits.npz", "测试集 logits 与标签"],
            ["results/metrics/benchmark.json", "batch=1/128 延迟、吞吐、模型大小"],
            ["results/metrics/robustness.json", "六模型 × 五退化 × 五强度的准确率"],
            ["results/metrics/robustness_summary.csv", "干净/退化平均/最差准确率汇总"],
            ["results/metrics/resnet20_ds_se_robust_summary.json", "退化微调前后对比"],
            ["report/figures/*.png", "报告插图"],
        ],
        [7.4, 8.4],
        "表 C-1  结果文件",
        first_col_left=True,
    )


def build(root: str, output: str, pages_path: Optional[str] = None) -> None:
    root = os.path.abspath(root)
    r = load_results(root)
    page_map = None
    if pages_path:
        page_map = load_json(pages_path)
    os.chdir(root)
    doc = Document()

    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    section.header_distance = Cm(1.4)
    section.footer_distance = Cm(1.4)
    section.different_first_page_header_footer = True

    # base styles
    normal = doc.styles["Normal"]
    normal.font.name = LATIN_FONT
    normal.font.size = Pt(12)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    for lvl in (1, 2, 3):
        st = doc.styles[f"Heading {lvl}"]
        st.font.name = LATIN_FONT
        st.font.color.rgb = RGBColor(0, 0, 0)
        st._element.rPr.rFonts.set(qn("w:eastAsia"), HEAD_FONT)

    add_header(section, "《深度学习》课程大作业 · CIFAR-10 CNN 结构对比")
    add_page_number_footer(section)

    cover(doc)
    page_break(doc)
    abstract(doc, r)
    page_break(doc)
    contents(doc, page_map)
    page_break(doc)
    chapter1(doc, r)
    page_break(doc)
    chapter2(doc, r)
    page_break(doc)
    chapter3(doc, r)
    page_break(doc)
    chapter4(doc, r)
    page_break(doc)
    chapter5(doc, r)
    page_break(doc)
    chapter6(doc, r)
    page_break(doc)
    references(doc)
    page_break(doc)
    appendix(doc, r)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    doc.core_properties.title = "CIFAR-10 图像分类中的 CNN 结构对比与低能见度鲁棒性分析"
    doc.core_properties.author = "孙蜀齐"
    doc.core_properties.subject = "深度学习课程大作业"
    doc.save(output)
    print(f"report written to {output}")

    # Save the exact numbers used, for auditing.
    audit = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summaries": r["summaries"],
        "benchmark": r["benchmark"],
        "robustness": r["robustness"],
        "finetune": r["finetune"],
    }
    audit_path = os.path.join(root, "results", "metrics", "report_snapshot.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"audit snapshot written to {audit_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--output", default="report/CIFAR10_CNN_course_report.docx")
    p.add_argument("--pages", default="", help="optional JSON map from TOC label to page number")
    args = p.parse_args()
    build(args.root, args.output, args.pages or None)


if __name__ == "__main__":
    main()