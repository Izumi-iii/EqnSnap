# EqnSnap 模型选择与按需加载

## 1. 模型定位

EqnSnap 保留两个完全离线的识别模式：

| 模型 | 默认状态 | 目标场景 |
| --- | --- | --- |
| pix2tex | 默认 | 清晰、简单、单行印刷公式 |
| UniMERNet Tiny | 可选 | 更复杂或多行的印刷公式 |

模型选择是显式设置，不根据截图来源或图像尺寸自动切换。这样可以避免
不可解释的延迟变化，也便于分别记录两个模型的准确率和性能。

## 2. 运行时结构

```text
RecognitionModelSettings
  -> FormulaCaptureWorkflow
  -> FormulaRecognitionService actor
       -> pix2tex Engine
       -> UniMERNet Engine
```

`RecognitionModelSettings` 使用 `UserDefaults` 保存选择，默认值为
`pix2tex`。`FormulaCaptureWorkflow` 在每次识别开始时读取一次选择；当前
任务不会在推理中途换模型，原有的 Session ID 和取消检查继续阻止旧结果
覆盖新会话。

`FormulaRecognitionService` 最多只强引用一个已加载 Engine：

1. 第一次使用某模型时才加载 Core ML 和 Tokenizer。
2. 连续使用同一模型时复用已加载实例。
3. 设置切换后释放不匹配的缓存实例。
4. 新模型在下一次识别请求到来时加载，不在设置窗口中预热。

这项策略优先控制常驻内存。它不同时缓存两个大模型，因此切换后的第一次
识别会包含模型加载时间。

## 3. UniMERNet 资源契约

`UniMERNetModelBundleLoader` 支持从 App Bundle、资源目录或显式 URL
加载。发布包中的资源名必须是：

```text
UniMERNetTinyEncoder-FP16.mlmodelc
UniMERNetTinyDecoder-CachedStep-SelfKV-FP16.mlmodelc
UniMERNetTokenizer.json
```

开发目录也可以使用同名 `.mlpackage`。加载器会在创建 Pipeline 前检查
Encoder 固定输入输出、Decoder 输入、8 层 KV Cache 特征和 Tokenizer。
动态 Cache 输出的最终 shape 仍由 Decoder Runner 在每一步验证。

## 4. 当前打包边界

UniMERNet FP16 Encoder 约 50 MB，Cached Decoder 约 156 MB，其中
Decoder 的单个 `weight.bin` 超过 GitHub 普通仓库 100 MB 单文件限制。
当前源码仓库不提交这两个发布权重。

当 Bundle 中缺少完整 UniMERNet 资源时：

- 设置窗口显示 UniMERNet，但该 segment 不可用。
- 已保存的无效 UniMERNet 选择会自动回退到 pix2tex。
- pix2tex 的现有离线识别不受影响。

发布前需要选择一种模型交付方式：使用 Git LFS 管理权重，或在受控发布
构建阶段把已校验哈希的模型注入 App Bundle。无论采用哪种方式，正式 App
必须包含模型并保持识别离线，不允许运行时上传截图或依赖远程 OCR。
