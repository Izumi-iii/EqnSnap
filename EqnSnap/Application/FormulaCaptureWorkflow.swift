import AppKit
import Combine
import CoreGraphics

@MainActor
final class FormulaCaptureWorkflow {
    private enum State {
        case idle
        case preparing
        case selecting
        case recognizing
        case presenting
        case failed
    }

    private let permissionClient = MacScreenCapturePermissionClient()
    private let captureService = MacScreenCaptureService()
    private let recognitionService = FormulaRecognitionService()
    private let modelSettings = RecognitionModelSettings.shared
    private let hotKeyClient = CarbonHotKeyClient()
    private let statusItemController = EqnSnapStatusItemController()
    private let resultWindowController = FormulaResultWindowController()
    private let settingsWindowController = RecognitionSettingsWindowController(
        settings: .shared
    )

    private var state = State.idle
    private var activeSessionID: CaptureSessionID?
    private var capturedFrame: CapturedDisplayFrame?
    private var selectedImage: CGImage?
    private var overlayViewController: FormulaSelectionOverlayViewController?
    private var overlayWindowController: FormulaSelectionOverlayWindowController?
    private var captureTask: Task<Void, Never>?
    private var recognitionTask: Task<Void, Never>?
    private var modelSelectionCancellable: AnyCancellable?

    func start() {
        modelSelectionCancellable = modelSettings.$selectedModel
            .removeDuplicates()
            .dropFirst()
            .sink { [weak self] model in
                guard let self else { return }
                Task {
                    await self.recognitionService.discardLoadedEngine(
                        except: model
                    )
                }
            }
        statusItemController.onCapture = { [weak self] in
            self?.beginCapture()
        }
        statusItemController.onOpenSettings = { [weak self] in
            self?.settingsWindowController.show()
        }
        statusItemController.onOpenScreenCaptureSettings = { [weak self] in
            self?.permissionClient.openSystemSettings()
        }
        statusItemController.install()
        do {
            try hotKeyClient.registerDefault { [weak self] in
                self?.beginCapture()
            }
        } catch {
            NSLog(
                "EqnSnap could not register the default hot key: %@",
                String(describing: error)
            )
        }
    }

    func stop() {
        invalidateSession()
        modelSelectionCancellable = nil
        hotKeyClient.unregister()
        statusItemController.uninstall()
        resultWindowController.dismiss()
        settingsWindowController.dismiss()
    }

    func beginCapture() {
        if resultWindowController.isVisible {
            resultWindowController.focus()
            return
        }
        guard state == .idle else { return }

        guard permissionClient.isAuthorized else {
            if permissionClient.requestAccess() {
                beginCapture()
            } else {
                showPermissionAlert()
            }
            return
        }

        let sessionID = CaptureSessionID()
        activeSessionID = sessionID
        state = .preparing
        statusItemController.setCaptureEnabled(false)
        captureTask = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let frame = try await captureService
                    .captureDisplayContainingPointer()
                guard isCurrent(sessionID), !Task.isCancelled else { return }
                capturedFrame = frame
                presentSelection(frame: frame, sessionID: sessionID)
            } catch {
                guard isCurrent(sessionID) else { return }
                finishCaptureFailure(error)
            }
        }
    }

    private func presentSelection(
        frame: CapturedDisplayFrame,
        sessionID: CaptureSessionID
    ) {
        state = .selecting
        let viewController = FormulaSelectionOverlayViewController(
            sessionID: sessionID,
            frame: frame,
            onCommit: { [weak self] pixelRect in
                self?.commitSelection(
                    pixelRect,
                    sessionID: sessionID
                )
            },
            onCancel: { [weak self] in
                self?.cancelSelection(sessionID: sessionID)
            }
        )
        let windowController = FormulaSelectionOverlayWindowController(
            contentViewController: viewController
        )
        overlayViewController = viewController
        overlayWindowController = windowController
        windowController.present(
            on: frame.geometry.bottomLeftGlobalFrame.rawValue
        )
    }

    private func commitSelection(
        _ pixelRect: PixelRect,
        sessionID: CaptureSessionID
    ) {
        guard isCurrent(sessionID),
              let capturedFrame else {
            return
        }
        do {
            let image = try capturedFrame.crop(to: pixelRect)
            dismissOverlay()
            self.capturedFrame = nil
            selectedImage = image
            recognize(image, sessionID: sessionID)
        } catch {
            finishCaptureFailure(error)
        }
    }

    private func recognize(
        _ image: CGImage,
        sessionID: CaptureSessionID
    ) {
        state = .recognizing
        resultWindowController.showRecognizing(
            screenshot: image,
            onRetry: { [weak self] in
                self?.retryRecognition(sessionID: sessionID)
            },
            onClose: { [weak self] in
                self?.closeResult(sessionID: sessionID)
            }
        )
        runRecognition(image, sessionID: sessionID)
    }

    private func runRecognition(
        _ image: CGImage,
        sessionID: CaptureSessionID
    ) {
        recognitionTask?.cancel()
        recognitionTask = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let output = try await recognitionService.recognize(
                    CapturedFormulaImage(image: image),
                    using: modelSettings.selectedModel
                )
                guard isCurrent(sessionID), !Task.isCancelled else { return }
                state = .presenting
                statusItemController.setCaptureEnabled(true)
                resultWindowController.showResult(output.latex)
            } catch is CancellationError {
                return
            } catch {
                guard isCurrent(sessionID) else { return }
                state = .failed
                statusItemController.setCaptureEnabled(true)
                resultWindowController.showFailure(
                    recognitionMessage(for: error)
                )
            }
        }
    }

    private func retryRecognition(sessionID: CaptureSessionID) {
        guard isCurrent(sessionID),
              state == .failed,
              let selectedImage else {
            return
        }
        state = .recognizing
        statusItemController.setCaptureEnabled(false)
        resultWindowController.showRetrying()
        runRecognition(selectedImage, sessionID: sessionID)
    }

    private func cancelSelection(sessionID: CaptureSessionID) {
        guard isCurrent(sessionID) else { return }
        invalidateSession()
    }

    private func closeResult(sessionID: CaptureSessionID) {
        guard isCurrent(sessionID) else { return }
        invalidateSession()
        resultWindowController.dismiss()
    }

    private func finishCaptureFailure(_ error: Error) {
        invalidateSession()
        if (error as? FormulaCaptureError)
            == .screenCapturePermissionDenied {
            showPermissionAlert()
        } else {
            showAlert(
                title: "无法截取公式",
                message: "截图失败，请确认目标显示器仍然可用后重试。"
            )
        }
    }

    private func invalidateSession() {
        activeSessionID = nil
        state = .idle
        captureTask?.cancel()
        captureTask = nil
        recognitionTask?.cancel()
        recognitionTask = nil
        capturedFrame = nil
        selectedImage = nil
        dismissOverlay()
        statusItemController.setCaptureEnabled(true)
    }

    private func dismissOverlay() {
        overlayViewController?.stop()
        overlayWindowController?.dismiss()
        overlayViewController = nil
        overlayWindowController = nil
    }

    private func isCurrent(_ sessionID: CaptureSessionID) -> Bool {
        activeSessionID == sessionID
    }

    private func showPermissionAlert() {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "EqnSnap 需要屏幕录制权限"
        alert.informativeText = (
            "权限仅用于截取你主动框选的公式区域。授权后如果仍无法截图，请重新启动 EqnSnap。"
        )
        alert.addButton(withTitle: "打开系统设置")
        alert.addButton(withTitle: "取消")
        NSApp.activate(ignoringOtherApps: true)
        if alert.runModal() == .alertFirstButtonReturn {
            permissionClient.openSystemSettings()
        }
    }

    private func showAlert(title: String, message: String) {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = title
        alert.informativeText = message
        alert.addButton(withTitle: "确定")
        NSApp.activate(ignoringOtherApps: true)
        alert.runModal()
    }

    private func recognitionMessage(for error: Error) -> String {
        switch error {
        case Pix2TexDecoderError.maximumTokenLengthReached:
            return "公式可能过长或超出当前单行公式支持范围。"
        case Pix2TexDecoderError.repetitionDetected:
            return "模型产生了重复输出，请调整选区后重试。"
        case UniMERNetCachedDecoderError.maximumTokenLengthReached:
            return "公式过长，UniMERNet 已达到当前 512 Token 上限。"
        case UniMERNetCachedDecoderError.repetitionDetected:
            return "UniMERNet 产生了重复输出，请调整选区后重试。"
        case UniMERNetModelBundleLoaderError.missingResource:
            return "UniMERNet 模型资源不完整，请重新安装应用。"
        default:
            return "本地模型未能完成识别，你可以使用同一截图重试。"
        }
    }
}
