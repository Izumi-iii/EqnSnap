# EqnSnap 公式识别评测集

## 数据来源

前两层评测集由 `prepare_im2latex_eval.py` 自动生成，原始数据来自 [im2latex-100k](https://zenodo.org/records/56198)。该数据集采用 CC0-1.0，包含官方 train、validation 和 test 拆分。

数据和生成图片位于 `Tools/model-conversion/datasets/`，该目录已被 Git 忽略。

## 第一层：官方 test 子集

默认从 10,355 条官方 test 记录中，以随机种子 42 固定抽取 1,000 条：

```text
datasets/evaluation/layer1-im2latex-test/
├── images/
└── manifest.jsonl
```

这一层用于验证：

- PyTorch 与 Core ML 的结果差异。
- FP32、FP16 和量化模型的质量变化。
- Encoder、Decoder 和 Tokenizer 回归。

manifest 同时保存原始 LaTeX、去除不可见 `\label{...}` 后的规范化 LaTeX，以及启发式的 v0.1 范围标记。

## 第一层 A：严格简单单行子集

`in_v0_1_scope` 只排除了明显的多行、矩阵和超长表达式，仍然包含不少复杂公式。运行下面的命令，可建立一个更严格且与模型预测无关的固定子集：

```bash
.venv/bin/python build_simple_single_line_subset.py
```

默认保留所有符合严格规则的样本；如只需快速冒烟测试，可通过 `--sample-size 50` 使用固定种子抽样：

```text
datasets/evaluation/layer1a-simple-single-line/
├── manifest.jsonl
└── summary.json
```

筛选规则包括：LaTeX 不超过 60 字符、公式区域不超过 550×90 px、命令不超过 8 个、花括号嵌套不超过 2 层、上下标不超过 5 个，并且分式、根号和求和/积分等大型算子分别不超过 1 个。文本排版命令、换行和导出残留会被排除。子集直接引用第一层图片，不重复保存文件。

评测命令：

```bash
.venv/bin/python evaluate_pix2tex.py \
  --manifest datasets/evaluation/layer1a-simple-single-line/manifest.jsonl \
  --output test-output/layer1a-pix2tex-evaluation.json \
  --csv test-output/layer1a-pix2tex-samples.csv \
  --device cpu
```

筛选过程不能读取 pix2tex 预测或既有评测分数，否则会形成挑选容易样本的数据泄漏。具体阈值和排除数量记录在 `summary.json`。

## 第一层 B：复杂和多行公式子集

运行下面的命令，从固定 Layer 1 中建立模型无关的复杂公式压力集：

```bash
.venv/bin/python build_complex_formula_subset.py
```

生成目录：

```text
datasets/evaluation/layer1b-complex-formulas/
├── manifest.jsonl
└── summary.json
```

`complex-formulas-v1` 当前包含 246 条：

- 157 条复杂单行公式。
- 64 条带多行环境、显式换行或堆叠结构的公式。
- 25 条超长或过高、但未归入明确多行的公式。

复杂单行公式按 LaTeX 长度、命令数、上下标数、分式数、大型算子数和公式高度筛选。多行和超长类别复用 Layer 1 已记录的范围原因。筛选不读取任何模型预测；图片直接引用 Layer 1，不重复保存。

这一层是来自 im2latex-100k 的合成压力集，适合做同源模型对比，但不能代替真实复杂截图，也不能排除训练集重叠造成的偏差。

### UniMERNet Tiny Python 基线

UniMERNet 0.2.3 固定依赖 `transformers==4.42.4`，与 pix2tex 环境中的版本不同，因此使用独立环境：

```bash
~/.local/bin/uv venv --python 3.11 .venv-unimernet
~/.local/bin/uv pip sync \
  --python .venv-unimernet/bin/python \
  requirements-unimernet.txt
```

`requirements-unimernet.txt` 还将 `pyarrow` 固定为 20.0.0，避免 UniMERNet 间接依赖的 `datasets==2.14.4` 在新版 pyarrow 上导入失败。

首次评测会把 `wanderkid/unimernet_tiny` 下载到被 Git 忽略的 `.cache/unimernet_tiny`。Tiny checkpoint 为 430,075,701 字节；评测时记录 SHA-256，并检查加载后的 missing/unexpected keys。

```bash
NO_ALBUMENTATIONS_UPDATE=1 \
.venv-unimernet/bin/python evaluate_unimernet.py \
  --manifest datasets/evaluation/layer1b-complex-formulas/manifest.jsonl \
  --device mps \
  --max-tokens 1536 \
  --offline
```

运行 pix2tex 对照并使用共同的模型无关 LaTeX 词法指标比较：

```bash
.venv/bin/python evaluate_pix2tex.py \
  --manifest datasets/evaluation/layer1b-complex-formulas/manifest.jsonl \
  --output test-output/layer1b-pix2tex-evaluation.json \
  --csv test-output/layer1b-pix2tex-samples.csv

.venv/bin/python compare_formula_model_reports.py
```

2026-08-21 的本机 PyTorch 基线结果：

| 数据 | pix2tex 共同词法相似度 | UniMERNet Tiny 共同词法相似度 |
|---|---:|---:|
| 全部 246 条 | 0.658 | 0.702 |
| 复杂单行 157 条 | 0.713 | 0.735 |
| 超长或过高 25 条 | 0.661 | 0.696 |
| 多行 64 条 | 0.521 | 0.622 |

逐条比较中，UniMERNet Tiny 在 136 条上更高，pix2tex 在 69 条上更高，41 条持平。UniMERNet 的优势主要集中在多行公式，但仍存在欠生成样本；生成长度 P50 为 215 Token、P95 为 448 Token、最大为 1022 Token，其中 10 条超过 512 Token。

同机 Python 测试中，UniMERNet MPS 的延迟 P50/P95 为 3.90/7.85 秒，最长约 21.06 秒。pix2tex 的完整报告使用 CPU，不能与该延迟作严格倍数比较；额外的 15 条 pix2tex MPS 冒烟测试也表明 PyTorch MPS 路径并不代表 Core ML 性能。因此速度、内存和功耗必须在 UniMERNet 转换成 Core ML 后重新测量。

在现有 35 张真实截图上，共同词法相似度为 pix2tex 0.742、UniMERNet Tiny 0.711；pix2tex 赢 15 条、UniMERNet 赢 6 条、14 条持平。当前证据支持保留 pix2tex 作为简单单行默认模型，把 UniMERNet 作为复杂/多行候选，而不是全面替换。

共同词法指标忽略空白、纯排版间距命令，并把 `\rm` 视为 `\mathrm`。它比原始字符串距离更公平，但仍不能证明数学或渲染语义等价；发布决策还需要 CDM、渲染比较和人工复核。

### 短公式失败诊断

使用既有 Layer 1 报告选择长度膨胀、严重错识别和稳定对照样本，并比较缩放与 Decoder 策略：

```bash
.venv/bin/python compare_pix2tex_strategies.py
```

默认比较：关闭 image resizer、固定 0.75 缩放、temperature 0.1、Argmax、128 Token 上限和重复保护。输出位于：

```text
test-output/pix2tex-strategy-comparison.json
test-output/pix2tex-strategy-comparison.csv
```

在完整严格子集上只验证固定缩放策略：

```bash
.venv/bin/python compare_pix2tex_strategies.py \
  --all-samples \
  --only fixed_scale_0_75 \
  --output test-output/layer1a-fixed-scale-comparison.json \
  --csv test-output/layer1a-fixed-scale-comparison.csv
```

固定缩放属于诊断结论，不应直接写死为最终 Swift 预处理规则。还需要在真实截图和深色、退化截图上验证，并确定基于公式像素高度或目标画布尺寸的稳定规则。

### Core ML 输入预处理比较

比较固定 `448×64` 画布上的前景高度，并复核相对缩放与可变 shape：

```bash
.venv/bin/python compare_pix2tex_preprocessing.py
```

输出位于：

```text
test-output/pix2tex-preprocessing-comparison.json
test-output/pix2tex-preprocessing-comparison.csv
```

当前严格子集结果否定了“统一目标前景高度 + 单一 `448×64` 画布”。有效候选规则是：前景按 0.75 相对缩放，保持宽高比，左上角放置，并将宽高分别补到 32 的倍数。实际观察到的枚举 shape 会写入 JSON 报告；在 Layer 2、Layer 3 通过前，这仍是发布候选而不是最终产品常量。

### 0.75 相对缩放的 Layer 2 / Layer 3 验证

Layer 2 只验证与严格单行子集同源的 176 个公式，共 528 张：

```bash
.venv/bin/python evaluate_pix2tex_relative_preprocessing.py \
  --manifest datasets/evaluation/layer2-screenshot-styles/manifest.jsonl \
  --source-subset-manifest datasets/evaluation/layer1a-simple-single-line/manifest.jsonl \
  --output test-output/layer2-strict-relative-preprocessing-evaluation.json \
  --csv test-output/layer2-strict-relative-preprocessing-samples.csv
```

Layer 3：

```bash
.venv/bin/python evaluate_pix2tex_relative_preprocessing.py \
  --manifest datasets/evaluation/layer3-real-screenshots/manifest.jsonl \
  --baseline-report test-output/layer3-pix2tex-evaluation.json \
  --output test-output/layer3-relative-preprocessing-evaluation.json \
  --csv test-output/layer3-relative-preprocessing-samples.csv
```

验证结论：Layer 3 基本持平且无长度膨胀，通过稳定性门槛；Layer 2 的 light、dark 和 degraded 均未通过，尤其 light/dark 出现大量长度膨胀。因此 0.75 不能作为跨来源的通用规则，Core ML 生产模型转换暂停。下一项实验应使用“目标前景高度 + 可变 32 倍数 shape”，不能再把目标高度和固定 `448×64` 画布绑定在一起。

### 固定前景高度 + 可变 shape

运行跨三层比较：

```bash
.venv/bin/python compare_pix2tex_target_height.py
```

脚本先使用 263 张诊断样本扫描 `24/28/32/36/40`，再把诊断最优高度跑完整 739 张。输出位于：

```text
test-output/pix2tex-target-height-comparison.json
test-output/pix2tex-target-height-comparison.csv
test-output/target-height-validation-summary.json
```

24 px 是诊断集上的最优值，并能让 Layer 1 与 Layer 2 三种风格消除长度膨胀；但 Layer 3 范围内字符相似度降至约 0.536，并出现 1 条长度膨胀。更高目标高度会改善部分真实截图，却会使合成层快速退化。因此不存在通过三层验证的单一固定前景高度，24 px 不能写入生产契约，Core ML 改造继续暂停。

## 第二层：截图风格

对第一层中符合 v0.1 单行范围的样本生成三种确定性变体：

- `light`：浅色页面截图。
- `dark`：深色页面截图。
- `degraded`：较小字号、轻度模糊和 JPEG 压缩。

```text
datasets/evaluation/layer2-screenshot-styles/
├── images/
└── manifest.jsonl
```

第二层仍是合成数据，不能替代真实 macOS 截图。第三层真实截图由人工收集，并应保持独立，避免为了通过现有测试而调整标签。

## 第三层：真实截图

把人工截取的 PNG 导入第三层：

```bash
.venv/bin/python import_real_screenshots.py ~/Documents/final_test_images
```

生成目录：

```text
datasets/evaluation/layer3-real-screenshots/
├── images/
├── labels.csv
├── manifest.jsonl
└── summary.json
```

在 `labels.csv` 中填写：

- `expected_latex`：人工确认的正确 LaTeX。
- `scope`：填写 `in`、`out` 或留空待确认。
- `notes`：模糊、裁切、文字混排等备注。

重新运行导入脚本时，会按原始文件名保留已经填写的标签。模型预测不能直接写入 `expected_latex`，除非人工核对后确认正确。

## 运行 pix2tex 评测

第三层标签确认后运行：

```bash
.venv/bin/python evaluate_pix2tex.py
```

默认输出：

```text
test-output/layer3-pix2tex-evaluation.json
test-output/layer3-pix2tex-samples.csv
```

报告分别统计全部样本、v0.1 范围内样本和范围外样本。字符串不一致不一定代表数学语义错误，因此错误样本仍需要人工或渲染结果复核。

## 生成命令

```bash
cd Tools/model-conversion
.venv/bin/python prepare_im2latex_eval.py
```

改变固定子集大小：

```bash
.venv/bin/python prepare_im2latex_eval.py --sample-size 2000
```

生成结果和限制记录在：

```text
datasets/evaluation/summary.json
```
