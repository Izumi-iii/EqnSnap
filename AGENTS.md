# EqnSnap Agent Guide

## Product

EqnSnap is a native macOS menu-bar app that captures a user-selected printed
single-line formula and converts it to editable LaTeX entirely on-device.
The v0.1 product boundary excludes handwriting, general OCR, document parsing,
cloud recognition, history, and complex multi-line formula guarantees.

## Supported Toolchain

- Xcode 15.x or newer.
- macOS deployment target 13.0.
- Swift language mode 5.
- The `.xcodeproj` must remain readable by Xcode 15. Do not introduce
  `PBXFileSystemSynchronizedRootGroup`, object version 77, or other newer-only
  project structures.
- Unit tests use XCTest for Xcode 15 compatibility. Do not migrate them to the
  Swift Testing `Testing` framework unless the minimum Xcode version changes.

## Runtime Flow

```text
EqnSnapApp
  -> FormulaCaptureWorkflow
  -> screen permission and frozen display capture
  -> FormulaSelectionReducer and overlay
  -> crop selected pixels
  -> Pix2TexRecognitionService
  -> preprocess -> Core ML Encoder -> Decoder -> Tokenizer
  -> FormulaResultWindowController
  -> editable LaTeX, local preview, copy
```

Only one capture/recognition/result session may be active. Preserve
`CaptureSessionID` checks and task cancellation so stale Core ML results cannot
overwrite a newer session.

## Architecture Boundaries

- `EqnSnap/Application`: app lifecycle, workflow, menu bar, hot key, result UI.
- `EqnSnap/Capture`: permission, display capture, coordinates, selection UI.
- `EqnSnap/Recognition`: image preprocessing, model loading, inference,
  decoding, tokenizer, and LaTeX post-processing.
- `Tools/model-conversion`: Python-only conversion and evaluation tools. These
  are not runtime dependencies.
- `EqnSnapTests`: XCTest unit and integration tests.

Keep capture independent from recognition. Keep the result UI independent
from the concrete LaTeX library through `FormulaPreviewRendering`. A future
renderer replacement should not require changes to capture or recognition.

## Model Contract

- Bundled model: pix2tex / LaTeX-OCR 0.1.4 converted to Core ML.
- Encoder input is grayscale Float32 NCHW, height 32 or 64, width 32...672 in
  multiples of 32.
- Preprocessing chooses a 24 px or 40 px target foreground height using the
  stroke-profile policy.
- Decoder uses a fixed 128-token prefix, context length 169, Argmax decoding,
  EOS termination, repetition protection, and cancellation checks.
- Do not change preprocessing constants, tensor shapes, token IDs, or feature
  names without updating Python fixtures and Swift regression tests together.

## Change Rules

- Reproduce the exact reported behavior before fixing it.
- Make the smallest change that satisfies the request; preserve unrelated
  workspace changes.
- New UI or services should expose a narrow boundary only when replacement or
  testing is a real requirement.
- Core functionality must remain offline. Do not add CDN-backed LaTeX
  rendering, remote OCR, analytics, or screenshot uploads.
- Do not add `$` or `$$` around copied LaTeX.
- Update `THIRD_PARTY_NOTICES.md` when adding distributable dependencies.

## Verification

Run from the repository root:

```bash
xcodebuild build \
  -project EqnSnap.xcodeproj \
  -scheme EqnSnap \
  -destination 'platform=macOS' \
  CODE_SIGNING_ALLOWED=NO

xcodebuild test \
  -project EqnSnap.xcodeproj \
  -scheme EqnSnap \
  -destination 'platform=macOS' \
  -skip-testing:EqnSnapUITests
```

For recognition changes, also run the bundled-model end-to-end tests and the
relevant scripts under `Tools/model-conversion`. UI tests are currently only a
basic launch scaffold and should not be treated as product-flow coverage.

For result-window visual QA in a Debug build, launch the built app with:

```bash
open EqnSnap.app --args --preview-result-window
```

The argument is compiled only in Debug and opens a deterministic sample result
without requiring screen-capture permission or model inference.

## Known Gaps

- Shortcut configuration and a real settings screen are not implemented.
- Product-level latency, memory, and accuracy release thresholds are not yet
  fully recorded.
- Documentation written before the basic v0.1 implementation may lag behind
  the current code; verify claims against the implementation and git history.
