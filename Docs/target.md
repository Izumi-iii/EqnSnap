# EqnSnap 项目目标

> 更新日期：2026-07-16<br>
> 文档状态：首版目标<br>
> 当前阶段：Xcode 工程已创建，准备搭建截图与模型推理主链路

具体实施拆分见 [v0.1 实施计划](v0.1.md)。

## 1. 项目概述

EqnSnap 是一款 macOS 原生菜单栏工具。用户通过全局快捷键框选屏幕中的单行印刷公式，应用在本机完成识别，将结果转换为 LaTeX，并允许用户预览、修正和复制。

一句话描述：

> **在 macOS 上截取一行公式，立即得到可复制的 LaTeX。**

产品计划闭源并可能商业发行。用户截图和识别结果默认不离开设备。

## 2. 我们要做什么

EqnSnap 首版需要完成以下闭环：

```text
菜单栏或全局快捷键
    → 进入截图框选
    → 用户选择单行公式
    → 本地 Core ML 识别
    → 显示 LaTeX 和渲染结果
    → 用户确认、修正或复制
```

### 2.1 菜单栏应用

- 应用常驻 macOS 菜单栏。
- 菜单中提供“截取公式”、设置和退出入口。
- 支持可配置的全局快捷键。
- 应用启动后不强制显示普通主窗口。

### 2.2 截图框选

- 通过快捷键快速进入截图状态。
- 支持多显示器及不同缩放比例；首版在鼠标所在显示器内框选，不做跨显示器选区拼接。
- 截图时显示透明遮罩、十字光标和选区边框。
- 用户拖动鼠标框选公式区域。
- 按 `Esc` 立即取消，不产生识别记录。
- 对空白、过小或无效选区给出轻量提示。

### 2.3 公式识别

- 只处理清晰、常规、单行的印刷体数学公式。
- 使用 pix2tex 模型，并转换为 Core ML。
- 推理完全在本机执行，不依赖网络和 Python 运行时。
- 模型输出 LaTeX Token，由 Swift 完成自回归循环和 Tokenizer 解码。

### 2.4 结果确认

- 显示用户截取的原始公式图片。
- 显示可编辑的 LaTeX 源码。
- 显示 LaTeX 实时渲染结果。
- 支持一键复制、重新识别和关闭。
- 复制成功后提供明确但不打扰用户的反馈。

### 2.5 后续版本：本地历史

历史记录不进入 v0.1。完成截图、识别、校正和复制主闭环后，再评估：

- 保存最近识别记录，包括截图、LaTeX 和时间。
- 支持再次复制和删除。
- 历史数据只保存在本机。
- 限制保存数量或自动清理，避免截图长期占用大量空间。

## 3. 希望达到的使用效果

### 3.1 基本体验

用户在任意应用中看到公式后：

1. 按下全局快捷键。
2. 屏幕进入截图框选状态。
3. 框选一行公式并松开鼠标。
4. 短暂显示识别中状态。
5. 弹出紧凑的结果浮层。
6. 用户确认渲染结果后复制 LaTeX。
7. 回到原应用完成粘贴。

整个过程应尽量通过键盘和鼠标连续完成，不要求用户切换到完整主窗口。

### 3.2 视觉效果

- 菜单栏图标简洁，能够表达“公式”或“截图识别”。
- 截图遮罩接近原生 macOS 截图体验，不遮挡用户对公式位置的判断。
- 选区边框清晰，拖动过程稳定，没有明显闪烁。
- 结果浮层尺寸紧凑，重点突出公式图片、LaTeX 和复制按钮。
- 错误、权限请求和模型加载状态使用清晰的系统化提示。

### 3.3 性能目标

以下是首版目标，需要在 Core ML 转换后实测确认：

| 指标 | 目标 |
|---|---:|
| 截图界面出现时间 | 用户感知为即时 |
| 普通单行公式识别 P50 | 不超过 1 秒 |
| 普通单行公式识别 P95 | 不超过 2 秒 |
| 模型首次加载 | 不阻塞菜单栏基本操作 |
| FP16 模型包体 | 目标约 60 MB |
| 网络依赖 | 无 |

## 4. 首版业务边界

### 4.1 首版支持

- 清晰的单行印刷公式。
- 常见积分、分式、上下标、根号、希腊字母和关系符号。
- 从网页、PDF、幻灯片和电子教材中截取的公式。
- LaTeX 源码输出、预览、编辑和复制。

### 4.2 首版不承诺

- 手写公式。
- 复杂矩阵和多行公式。
- 低分辨率、严重压缩或背景噪声很强的截图。
- 整篇 PDF、表格、文字和公式混排识别。
- MathML、Typst、DOCX 等完整多格式转换。
- 云端高精度识别或账号同步。
- 通用 AI 解释和对话功能。

如果用户选择超出首版范围的内容，应用应允许识别，但不能对结果准确性作出承诺。

## 5. 与 Snapio 的代码复用

截图软件项目为同级目录中的 Snapio：

```text
/Users/wangjie/dddd/Snapio
```

Snapio 是原生 macOS AppKit 截图工具，核心业务已拆分到本地 Swift Package：

```text
/Users/wangjie/dddd/Snapio/Packages/SnapioKit
```

SnapioKit 使用 Swift 5.9 Package、Swift 5 语言模式，最低支持 macOS 11。EqnSnap 目标为 macOS 13，因此其系统兼容路径可以覆盖 EqnSnap。Snapio 使用 AppKit，EqnSnap 主界面使用 SwiftUI，但截图覆盖窗口本来就适合继续使用 AppKit，两者并不冲突。

### 5.1 可以直接或小改复用的模块

| 能力 | Snapio 现有文件 | EqnSnap 用法 |
|---|---|---|
| 强类型坐标和标识 | `SnapioCore/Shared/Identifiers.swift`、`GeometryTypes.swift` | 保留显示器局部点、全局点和像素坐标的区分 |
| 点到像素映射 | `SnapioCore/Capture/GeometryMapper.swift` | 直接复用 Retina、分数缩放和裁切计算 |
| 捕获模型和协议 | `SnapioCore/Capture/CaptureModels.swift`、`CapturePorts.swift` | 作为截图 Core 与系统后端之间的接口 |
| 显示器定位 | `SnapioPlatform/Capture/DisplayGeometryAdapter.swift` | 获取鼠标所在显示器及实际像素尺寸 |
| macOS 13 截图 | `SnapioPlatform/Capture/LegacyCaptureBackend.swift` | EqnSnap 在 macOS 13 使用 Core Graphics 静态截图路径 |
| macOS 14+ 截图 | `SnapioPlatform/Capture/ScreenCaptureKitBackend.swift` | EqnSnap 在 macOS 14+ 使用 `SCScreenshotManager` |
| 全局快捷键 | `SnapioPlatform/Shortcuts/CarbonHotKeyClient.swift` | 修改签名和命名后注册公式截图快捷键 |
| 快捷键事务 | `SnapioCore/Shortcuts/ShortcutCoordinator.swift` | 复用注册新值、保存成功后释放旧值的回滚逻辑 |
| 全屏覆盖 Panel | `SnapioPresentation/Selection/CaptureOverlayPanel.swift` | 可基本直接复用 |
| 覆盖窗口生命周期 | `SelectionOverlayWindowController.swift` | 修改窗口标题和 EqnSnap 路由后复用 |
| 选区绘制与输入 | `SnapioUI/Selection/SelectionOverlayView.swift` | 复用遮罩、边框、鼠标事件、尺寸标签和 `Esc` 处理 |
| 临时图片持有 | `SnapioCore/Assets/TransientAssetStore.swift` | 可改造成截图到识别结果浮层之间的内存资产存储 |

### 5.2 需要针对 EqnSnap 改造的模块

#### 权限

Snapio 的 `PermissionGate` 把屏幕录制和辅助功能权限都设为截图硬门槛。EqnSnap 首版只做区域框选，不做窗口自动吸附，也不需要读取其他应用的辅助功能信息，因此不应直接沿用双权限硬门槛。

EqnSnap 首版原则：

- 必需权限只有屏幕录制。
- Carbon 全局快捷键不应被错误地解释为必须获取辅助功能权限。
- 是否启用 App Sandbox 和采用何种分发方式，应根据实际 API 和商业发行渠道单独验证，不能照搬 Snapio 的非沙盒结论。

可以复用 `MacPermissionClient` 中的屏幕录制检查、申请和系统设置跳转，但需要拆掉 Accessibility 依赖。

#### 选区交互

Snapio 的 `SelectionReducer` 支持窗口悬停、单击窗口捕获、选区移动、八向缩放和再次确认。EqnSnap 的目标流程更短：

```text
鼠标按下
    → 拖动单行公式区域
    → 鼠标松开
    → 立即提交截图并开始识别
```

因此需要从 Snapio 状态机中保留：

- 新建矩形选区。
- 反向拖动和显示器边界裁切。
- 点到像素坐标转换。
- `Esc` 取消。

首版可以移除：

- 窗口目录和窗口悬停。
- 单击捕获整个窗口。
- 选区移动及八向缩放。
- Capture/Cancel 工具栏和二次确认。

如果实际使用发现用户需要修正选区，再恢复 Snapio 已有的移动和缩放逻辑。

#### CaptureCoordinator

Snapio 的 `CaptureCoordinator` 在捕获完成后进入标注、快捷浮层和贴图资产链路。EqnSnap 应保留权限检查、session ID、旧异步结果丢弃、冻结显示器帧和区域裁切，但把结果出口改为公式识别：

```text
CaptureResult
    → 获取临时 CGImage
    → FormulaRecognitionCoordinator
    → LaTeXResult
    → 结果浮层
```

`TransientAssetStore.AssetOwner` 需要增加公式识别或结果浮层的 owner，不能借用 Snapio 的 `.editor`、`.pin` 等无关语义。

#### 菜单栏与设置

Snapio 的 `StatusItemController` 和快捷键设置可以参考，但 v0.1 菜单内容只需要 EqnSnap 的“截取公式”“设置”和“退出”。EqnSnap 使用 SwiftUI `MenuBarExtra` 还是复用 Snapio 的 `NSStatusItem`，应以全局快捷键、权限窗口和结果浮层的生命周期是否容易统一为判断依据。

### 5.3 不应复制到 EqnSnap 的 Snapio 功能

- 标注系统：`AnnotationDocument`、`AnnotationRenderer`、undo/redo。
- 图片安全遮挡。
- PNG 保存、另存为和默认目录书签。
- 贴图窗口与 `PinSessionStore`。
- Snapio 的 Quick Action 复制/保存浮层。
- 窗口目录和窗口自动吸附。
- 辅助功能双权限硬门槛。
- Snapio 完整的 `AppCoordinator`。

这些功能会扩大 EqnSnap 首版范围，并引入与公式识别无关的状态和窗口生命周期。

### 5.4 EqnSnap 独有能力

- 公式图片预处理。
- pix2tex Core ML Encoder 和 Decoder。
- Swift 自回归解码。
- LaTeX Tokenizer 和结果规范化。
- LaTeX 渲染、编辑，以及后续版本的识别历史。
- 公式识别相关的质量检测和错误提示。

### 5.5 推荐的集成方式

SnapioKit 当前只对外声明 `SnapioApplication` 产品，而该产品组合了截图、标注、保存、贴图和完整 Snapio 应用生命周期。EqnSnap 不应直接依赖整个 `SnapioApplication`。

推荐顺序：

1. 先从 Snapio 复制并改名最小截图子集到 EqnSnap 的本地 Package，例如 `Packages/EqnSnapKit`。
2. 保留 Snapio 的 Core → Platform / Presentation 依赖方向，不把 Core ML 或 SwiftUI 类型放入截图 Core。
3. 为复用的坐标映射、选区裁切和捕获后端同步迁移对应测试。
4. 两个项目截图逻辑稳定后，再评估抽取独立的共享 `ScreenSelectionKit`，由 Snapio 和 EqnSnap 共同依赖。

不建议让 EqnSnap 通过相对路径长期依赖正在频繁开发的完整 SnapioKit，否则 Snapio 的标注和贴图改动可能无意中破坏 EqnSnap 构建。

### 5.6 复用原则

- 先确认另一个项目中的代码所有权和许可证适合闭源商业使用。
- 优先抽取边界清晰的通用模块，不直接复制整个应用结构。
- 通用截图逻辑不应依赖公式识别模块。
- 两个项目对多显示器、权限和坐标系统应使用同一套测试用例。
- 如果两个项目会长期共同维护，可以把稳定后的截图核心抽取为独立 Swift Package。
- 如果只是一次性复用，应在 EqnSnap 中记录来源和后续修改，避免以后无法同步修复。
- 每次从 Snapio 同步修复时，应注明来源文件和 Snapio commit，避免无法判断两个版本的差异。
- Snapio 当前工作区存在未提交修改，复制代码时应以明确的 commit 或经确认的工作区版本为基准，不能无意混入未完成代码。

## 6. 技术架构

```text
EqnSnapApp
├── App
│   ├── 菜单栏入口
│   ├── 应用生命周期
│   └── 设置
├── Capture
│   ├── 权限管理
│   ├── 全局快捷键
│   ├── 多显示器截图
│   └── 选区遮罩
├── Recognition
│   ├── 图像预处理
│   ├── pix2tex Image Resizer
│   ├── Core ML Encoder
│   ├── Core ML Decoder
│   └── Tokenizer
├── Result
│   ├── LaTeX 编辑
│   ├── 公式渲染
│   └── 剪贴板
```

SwiftUI 负责菜单、设置和结果界面。AppKit 负责全局快捷键、透明覆盖窗口、截图框选、多显示器坐标以及需要精确控制的窗口行为。History 模块不进入 v0.1，等主闭环稳定后再单独设计。

## 7. 基本项目信息

| 项目 | 信息 |
|---|---|
| 产品名称 | EqnSnap |
| 平台 | macOS |
| 目标最低系统 | macOS 13.0 |
| Xcode Deployment Target | macOS 13.0 |
| 开发语言 | Swift |
| Xcode 工程 Swift Version | 5.0 |
| UI | SwiftUI，必要时使用 AppKit |
| 应用形态 | 菜单栏常驻工具 |
| Bundle Identifier | `com.izumiiii.EqnSnap` |
| 截图代码来源 | `/Users/wangjie/dddd/Snapio/Packages/SnapioKit` |
| 首版权限 | 屏幕录制；不默认要求辅助功能权限 |
| 识别模型 | pix2tex / LaTeX-OCR 0.1.4 |
| 模型运行方式 | Core ML，本地离线 |
| 主要输出 | LaTeX |
| 测试框架 | Swift Testing + XCTest UI Testing |
| 数据存储 | v0.1 仅保留当前会话内存，不使用 CloudKit |
| 源码发布 | 闭源 |
| 商业计划 | 可能商业发行 |
| 模型许可证 | MIT，发行前需完成第三方许可证核查 |

## 8. 当前状态

已经完成：

- Xcode macOS SwiftUI 工程创建。
- 产品和竞品调研。
- pix2tex 与候选模型的 Python 基线对比。
- 首版模型选择：pix2tex。
- pix2tex CPU 基线：项目公式平均约 0.56 秒。
- 模型规模确认：约 30.35M 参数、115.9 MB FP32 权重。
- 已浏览 Snapio 的 Core、Platform、Selection、权限和快捷键实现，并确定可复用边界。

尚未完成：

- 菜单栏应用结构。
- 从 Snapio 迁移最小截图子集并接入 EqnSnap。
- pix2tex Core ML 转换。
- Swift 图像预处理和 Tokenizer。
- 结果浮层和 LaTeX 渲染。
- 后续版本的历史记录设计。
- 完整的产品测试集和性能验收。

## 9. 首版完成标准

当以下条件全部满足时，可以认为首版核心闭环完成：

- 应用能够从菜单栏或快捷键进入截图状态。
- 多显示器下能够正确框选并获得截图。
- `Esc` 能随时取消截图。
- 清晰单行公式能够在本机转换为 LaTeX。
- LaTeX 能够正确渲染、编辑和复制。
- 普通公式识别速度达到目标范围。
- 应用离线运行，截图不会上传。
- 最低支持系统上功能正常。
- 第三方模型和代码的许可证声明完整。
