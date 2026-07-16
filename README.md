# EqnSnap

EqnSnap 是一款面向 macOS 的原生公式截图识别工具：用户通过全局快捷键框选屏幕中的**印刷体数学公式**，应用在本机完成识别并输出 LaTeX。

## 产品边界

- 只处理屏幕截图、电子文档和扫描材料中清晰、常规、单行的印刷体公式。
- 不提供手写公式识别。
- 首版不承诺复杂多行公式和矩阵识别。
- 首要输出格式为 LaTeX；Markdown 等包装格式可在后续按实际粘贴场景扩展。
- 不做整篇论文编辑、通用文档 OCR 或 AI 对话。
- 产品计划闭源并可能商业发行。

## 技术方向

- 使用 Swift、SwiftUI 开发 macOS 原生应用，必要时使用 AppKit 完成全局快捷键、截屏和多显示器交互。
- 识别过程完全在本机完成，不依赖 Python 运行时或远程 OCR 服务。
- 采用开源模型 [pix2tex / LaTeX-OCR](https://github.com/lukas-blecher/LaTeX-OCR)，当前测试权重约 116 MB。
- 在 Python 端完成权重整理和 Core ML 转换，在 Swift 端实现图像预处理、模型调用、自回归解码、Tokenizer 和 LaTeX 输出。
- 目标系统为 macOS 13.0+；模型能否由 Apple Neural Engine 加速，以 Core ML 转换和真机性能测试结果为准。

## 当前阶段

项目已创建 Xcode 工程。pix2tex 已完成 PyTorch 本机基线测试并确定为首版模型，当前工作重点是复用 Snapio 的截图模块，并验证 pix2tex 转换为 Core ML 后的正确性、速度、内存和安装包体积。

## 文档

- [同类产品调研](Docs/调研.md)
- [pix2tex 与 Core ML 技术方案](Docs/技术方案.md)
- [项目目标与 Snapio 复用方案](Docs/target.md)
