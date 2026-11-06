# CIFAR-10 图像分类与低能见度鲁棒性实验

> GitHub 仓库：<https://github.com/sun880947/cifar10-cnn-lab>

本项目是《深度学习》课程大作业：以 **CIFAR-10** 为数据集，以 **ResNet-20** 为基线，系统比较 **深度可分离卷积、SE 通道注意力、GELU 激活、GroupNorm** 等结构改动对参数量、计算量、精度与 CPU 推理延迟的影响；并把轻量组合模型用于**低能见度道路场景**的模拟鲁棒性评估与 Grad-CAM 可解释性分析，最后提供一个可本地上传图片的 Flask 演示。

> 与教材实验三的区别：本项目不采用教材中的 UNet 去雾 / GAN 超分 / 目标检测流程，而是独立设计“CIFAR-10 分类 + 结构消融 + 退化鲁棒性 + 可解释性”的闭环实验，所有代码、数据划分、训练曲线和报告数字均来自本仓库的真实运行结果。

---

## 1. 主要特性

- **统一配方、受控消融**：6 个模型共用同一数据划分、增强、优化器、学习率调度与训练轮数，避免“调参差异”污染结构对比。
- **参数量 / MACs / 延迟三重度量**：不只看精度，还分析深度可分离卷积在 CPU 上的“理论省算力、实际不一定更快”的工程现象。
- **低能见度鲁棒性**：雾、低照度、运动模糊、高斯噪声、JPEG 压缩 5 类退化 × 5 个强度，共 25 个测试条件。
- **退化数据微调**：从干净模型出发，用随机退化数据继续训练，量化“干净精度损失 vs 退化精度收益”。
- **Grad-CAM 可视化**：对比基线模型与 R20-DS-SE 在清晰 / 雾天图像上的关注区域。
- **可交互演示**：Flask 上传图片，选择模型与退化类型，实时显示 Top-3 与 Grad-CAM。

---

## 2. 目录结构

```text
cifar10-cnn-lab/
├─ configs/recipe.json          # 固定训练配方（可直接对照报告）
├─ data/
│  ├─ cifar10_train.npz         # 50000 张，uint8 NHWC + int64 label（由脚本生成）
│  └─ cifar10_test.npz          # 10000 张
├─ src/
│  ├─ models.py                 # ResNet-20 与 5 个结构变体
│  ├─ data.py                   # 划分、增强、归一化、批迭代
│  ├─ engine.py                 # 训练 / 评估循环
│  ├─ metrics.py                # Accuracy、Top-5、NLL、Macro-F1、ECE、混淆矩阵
│  ├─ flops.py                  # 参数量与 MACs 统计（Conv2d/Linear 前向 hook）
│  ├─ corruptions.py            # 5 类合成退化
│  ├─ robustness.py             # 退化测试套件
│  ├─ gradcam.py                # Grad-CAM
│  └─ utils.py
├─ scripts/
│  ├─ prepare_data.py           # torchvision / parquet -> npz
│  ├─ train.py                  # 单模型训练
│  ├─ evaluate.py               # 测试集评估
│  ├─ benchmark.py              # 参数、MACs、CPU 延迟
│  ├─ robustness.py             # 全模型退化评估
│  ├─ finetune_robust.py        # 退化数据微调
│  ├─ visualize.py              # 报告全部插图
│  ├─ build_report.py           # 由 JSON 生成中文 DOCX 报告
│  ├─ toc_pages.py              # 从 PDF 回填目录页码
│  ├─ check_consistency.py      # 报告 / JSON / 日志数字一致性自检
│  ├─ make_architecture.py      # 结构示意图
│  └─ update_readme.py          # 用真实指标填 README 结果表
├─ checkpoints/                 # *_best.pt / *_last.pt（运行后生成）
├─ results/
│  ├─ metrics/                  # summary / benchmark / robustness / 混淆矩阵 logits
│  ├─ history/                  # 逐 epoch 训练历史
│  └─ logs/                     # 训练日志
├─ tests/test_smoke.py          # 结构 / 退化 / Grad-CAM 冒烟测试
├─ report/                      # 报告 docx / pdf 与 figures
├─ demo/                        # Flask 演示
└─ requirements.txt
```

---

## 3. 环境

实验在 Windows 11 + CPU 上完成，未使用 GPU：

| 项目 | 版本 / 配置 |
|---|---|
| Python | 3.12.7 |
| PyTorch | 2.12.0+cpu |
| torchvision | 0.27.0+cpu |
| NumPy | 2.2.6 |
| CPU | Intel Core Ultra 9 185H（16 核 22 线程，AVX2） |
| 内存 | 32 GB |
| 线程 | `torch.set_num_threads(16)`，channels-last |

安装：

```powershell
python -m pip install -r requirements.txt
```

> 没有 GPU 也能完整复现；文中所有训练时间均指上述 CPU 环境。`bf16`  autocast 在该 CPU 上极慢，因此全程使用 float32。

---

## 4. 数据准备

仓库不强制提交 170 MB 的 npz 数据，可任选一种方式生成：

### 方式 A：torchvision 自动下载（需联网）

```powershell
python -m scripts.prepare_data --source torchvision --output-dir data
```

### 方式 B：离线 HuggingFace parquet 镜像

若已有 `train-00000-of-00001.parquet` 与 `test-00000-of-00001.parquet`（字段 `img.bytes`、`label`）：

```powershell
python -m scripts.prepare_data --source parquet --parquet-dir path\to\parquet --output-dir data
```

生成后 `data/cifar10_train.npz` 与 `data/cifar10_test.npz` 的 `images` 形状为 `[N,32,32,3]`（uint8），`labels` 为 `[N]`（int64）。

### 数据划分

- 训练集 50000 张中，每类分层抽取 500 张作为验证集：**45000 训练 / 5000 验证**，`seed=42`；
- 官方测试集 10000 张只用于最终评估，不参与调参；
- 归一化均值 `(0.4914, 0.4822, 0.4465)`，标准差 `(0.2470, 0.2435, 0.2616)`；
- 训练增强：4 像素零填充 + 随机裁剪 32×32 + 随机水平翻转 + Cutout(8)。

---

## 5. 模型与训练配方

### 5.1 六个模型

| 名称 | 说明 |
|---|---|
| `resnet20` | 基线：标准 3×3 卷积 + BN + ReLU |
| `resnet20_ds` | 将残差块内 3×3 卷积替换为深度可分离卷积（DW 3×3 + PW 1×1） |
| `resnet20_se` | 每个残差块加入 SE 通道注意力（reduction=8） |
| `resnet20_gelu` | ReLU 替换为 GELU |
| `resnet20_gn` | BatchNorm 替换为 GroupNorm（8 组） |
| `resnet20_ds_se` | 深度可分离卷积 + SE，作为轻量部署候选 |

所有模型均为 CIFAR 风格 ResNet-20（3 个 stage，每 stage 3 个 BasicBlock，base width 16），仅改变上述组件；分类头为全局平均池化 + 全连接。

### 5.2 统一训练配方

| 超参数 | 值 |
|---|---|
| Epochs | 60 |
| Batch size | 256 |
| Optimizer | SGD, momentum 0.9, Nesterov |
| Base LR | 0.1 |
| Min LR | 1e-4 |
| Warmup | 5 epochs 线性 |
| Schedule | cosine decay |
| Weight decay | 5e-4 |
| Label smoothing | 0.1 |
| Cutout | 8 |
| Seed | 42 |

训练命令（六个模型依次执行）：

```powershell
$models = @('resnet20','resnet20_ds','resnet20_se','resnet20_gelu','resnet20_gn','resnet20_ds_se')
foreach ($m in $models) {
  python -m scripts.train --model $m --epochs 60 --cutout 8 --threads 16
}
```

每个模型会输出：

```text
results/history/<model>.json           # 逐 epoch loss / accuracy / 时间
results/metrics/<model>_summary.json   # 参数量、MACs、最佳 epoch、测试指标
results/metrics/<model>_test_logits.npz
checkpoints/<model>_best.pt            # 按验证集准确率选择
checkpoints/<model>_last.pt            # 断点续训
```

断点续训：

```powershell
python -m scripts.train --model resnet20 --epochs 60 --cutout 8 --threads 16 --resume
```

---

## 6. 评估与实验

### 6.1 单模型测试

```powershell
python -m scripts.evaluate --model resnet20 --output results/metrics/resnet20_eval.json
```

### 6.2 参数量、MACs 与 CPU 延迟

```powershell
python -m scripts.benchmark --threads 16
```

- MACs 通过 `Conv2d` / `Linear` 前向 hook 解析统计（1 MAC = 1 次乘加，FLOPs = 2×MACs）；
- 延迟在 16 线程、`channels_last`、`torch.inference_mode()` 下测量；六个模型交错进行 5 轮，batch=1 每模型 50 次、batch=128 每模型 25 次，报告 median / P90，避免先测模型承担冷缓存偏差。

### 6.3 低能见度鲁棒性

```powershell
python -m scripts.robustness --threads 16
```

5 类退化及实现：

| 退化 | 实现要点 | severity 1 → 5 |
|---|---|---|
| `fog` | 与亮灰蓝色 `[232,236,242]` 按透明度混合 | α ≈ 0.20 → 0.63 |
| `low_light` | Gamma 压暗 + 高斯读出噪声 | γ=1.24→2.20，σ=3.5→11.5 |
| `motion_blur` | 0/45/90/135° 线性运动核，逐图随机方向 | 核长 3→11 |
| `gaussian_noise` | 加性高斯噪声 | σ=6→30 |
| `jpeg` | PIL JPEG 编码/解码 | quality 52→12 |

输出 `results/metrics/robustness.json`、`robustness.csv`、`robustness_summary.csv`。

### 6.4 退化数据微调

以 `resnet20_ds_se` 为例：

```powershell
python -m scripts.finetune_robust --model resnet20_ds_se --epochs 15 --threads 16
```

训练时 50% 的 batch 施加随机退化（类型与强度随机），再走正常增强；以“干净验证 + 5 类 severity-3 退化验证”的混合集选择 checkpoint。

### 6.5 生成报告插图

```powershell
python -m scripts.visualize
```

输出到 `report/figures/`：数据样例、退化样例、训练曲线、精度-成本散点、逐类精度、混淆矩阵、鲁棒性曲线/热力图、Grad-CAM、微调前后对比。

### 6.6 生成中文报告

报告由 `scripts/build_report.py` 从 `results/metrics/*.json` 自动生成，不手工抄数字：

```powershell
python -m scripts.build_report
```

DOCX 位于 `report/CIFAR10_CNN_course_report.docx`。若要让目录页码与 PDF 实际分页一致，先导出一次 PDF，回填页码后再生成一次：

```powershell
python -m scripts.build_report
python -m scripts.toc_pages report/CIFAR10_CNN_course_report.pdf results/metrics/report_pages.json
python -m scripts.build_report --pages results/metrics/report_pages.json
```

PDF 可由 Word 打开 DOCX 后“另存为 PDF”，或用 LibreOffice 转换（见 `scripts/build_report.py` 顶部说明）。

### 6.7 数字一致性自检

```powershell
python -m scripts.check_consistency
```

该脚本从 `*_test_logits.npz` 重算 Accuracy / Top-5 / NLL / Macro-F1 / ECE，与 `*_summary.json` 对比；并核对 benchmark、robustness、finetune 以及 README / DOCX 中的关键数字，任一项超差都会报错。

### 6.8 冒烟测试

不加载大数据、只检查结构形状/参数量、五类退化与 Grad-CAM 的最小测试：

```powershell
python -m unittest discover -s tests -v
```

---

## 7. 演示系统

```powershell
python demo/app.py --port 5000
```

浏览器打开 `http://127.0.0.1:5000`：

1. 上传图片（自动中心裁剪并缩放为 32×32）；
2. 选择模型（默认 R20-DS-SE）与退化类型 / 强度；
3. 返回 Top-3 类别、置信度、网络输入与 Grad-CAM。

如果手头没有 32×32 测试图，可直接打开示例自动运行地址，页面会载入内置的 `demo/samples/sample_0.png` 并自动完成一次“雾天 + 强度 3”的推理：

```text
http://127.0.0.1:5000/?demo=1&model=resnet20_ds_se&corruption=fog&severity=3
```

接口：

```text
GET  /health
GET  /sample/<index>   # 返回内置示例图
POST /predict          # multipart: image, model, corruption, severity
```

演示仅用于课程教学，输入分辨率与 CIFAR-10 一致；对高分辨率真实道路图像，应先做目标检测/裁剪，这不是本项目的研究范围。

---

## 8. 结果摘要

> 完整数字与讨论见 `report/`。下表在六模型训练完成后由 `results/metrics/*_summary.json` 汇总。

<!-- RESULTS_START -->
| 模型 | 参数量 | MACs | Test Acc | ΔAcc | CPU 延迟 (batch=1) | 退化平均 Acc |
|---|---:|---:|---:|---:|---:|---:|
| `resnet20` | 272,474 | 40.81M | 88.54% | +0.00 pp | 1.95 ms | 57.48% |
| `resnet20_ds` | 40,522 | 6.65M | 86.68% | -1.86 pp | 3.08 ms | 53.93% |
| `resnet20_se` | 277,082 | 40.82M | 89.07% | +0.53 pp | 2.96 ms | 59.32% |
| `resnet20_gelu` | 272,474 | 40.81M | 89.34% | +0.80 pp | 2.37 ms | 59.19% |
| `resnet20_gn` | 272,474 | 40.81M | 85.76% | -2.78 pp | 2.19 ms | 65.63% |
| `resnet20_ds_se` | 45,130 | 6.66M | 86.45% | -2.09 pp | 3.88 ms | 54.11% |

- 基线 R20 测试准确率 **88.54%**；最佳模型 **R20-GELU 89.34%**。
- 深度可分离卷积把参数量降至 40,522（−85.1%）、MACs 降至 6.65M（−83.7%），batch=1 延迟反而变化 **+57.4%**。
- 25 个退化条件上，R20-DS-SE 平均准确率 54.11%，基线 57.48%。
- 退化平均准确率最高的是 **R20-GN 65.63%**（干净准确率 85.76%），说明干净精度与退化鲁棒性并不总是同向。
- 退化微调后 R20-DS-SE 退化平均准确率 54.11% → 65.70%，干净准确率 86.45% → 85.65%。
<!-- RESULTS_END -->

---

## 9. 复现注意事项

- **随机性**：数据划分、增强、初始化、退化均固定 seed；PyTorch CPU 卷积在不同版本/线程数下可能有极小数值差异，报告中的数字以本仓库 `results/` 为准。
- **模型选择**：只用 5000 张验证集选择 best checkpoint，测试集仅在最后评估一次。
- **公平比较**：六个模型共用配方；没有为某个模型单独调参。若继续调参，应重新跑全部变体。
- **CPU 延迟**：受电源计划、后台负载、线程数影响，`benchmark.py` 的数字用于相对比较，不宜当作绝对值。
- **合成退化**：雾/低光等是 CIFAR-10 上的模拟，不等价于真实雾天道路数据；结论应理解为“受控退化下的相对鲁棒性”。

---

## 10. 参考文献

1. He K., Zhang X., Ren S., Sun J. Deep Residual Learning for Image Recognition. CVPR, 2016.
2. He K., Zhang X., Ren S., Sun J. Identity Mappings in Deep Residual Networks. ECCV, 2016.
3. Howard A. G., et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv:1704.04861, 2017.
4. Hu J., Shen L., Sun G. Squeeze-and-Excitation Networks. CVPR, 2018.
5. Hendrycks D., Gimpel K. Gaussian Error Linear Units (GELUs). arXiv:1606.08415, 2016.
6. Wu Y., He K. Group Normalization. ECCV, 2018.
7. Selvaraju R. R., et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV, 2017.
8. Hendrycks D., Dietterich T. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR, 2019.
9. Krizhevsky A. Learning Multiple Layers of Features from Tiny Images. Technical report, 2009.