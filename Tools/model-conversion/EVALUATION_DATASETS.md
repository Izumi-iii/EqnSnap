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
