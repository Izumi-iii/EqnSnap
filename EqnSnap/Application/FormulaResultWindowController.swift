import AppKit
import SwiftUI

@MainActor
final class FormulaResultWindowController:
    NSWindowController,
    NSWindowDelegate
{
    private let viewModel: FormulaResultViewModel
    private var closeHandler: (() -> Void)?

    init() {
        let viewModel = FormulaResultViewModel()
        self.viewModel = viewModel
        let hostingController = NSHostingController(
            rootView: ContentView(viewModel: viewModel)
        )
        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 600, height: 430),
            styleMask: [.titled, .closable, .resizable, .utilityWindow],
            backing: .buffered,
            defer: false
        )
        panel.title = "EqnSnap"
        panel.level = .floating
        panel.isReleasedWhenClosed = false
        panel.collectionBehavior = [.moveToActiveSpace, .fullScreenAuxiliary]
        panel.contentViewController = hostingController
        super.init(window: panel)

        panel.delegate = self
        viewModel.onClose = { [weak self] in
            self?.requestClose()
        }
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    var isVisible: Bool {
        window?.isVisible == true
    }

    func showRecognizing(
        screenshot: CGImage,
        onRetry: @escaping () -> Void,
        onClose: @escaping () -> Void
    ) {
        closeHandler = onClose
        viewModel.screenshot = NSImage(
            cgImage: screenshot,
            size: NSSize(width: screenshot.width, height: screenshot.height)
        )
        viewModel.latex = ""
        viewModel.copyConfirmation = nil
        viewModel.phase = .recognizing
        viewModel.onRetry = onRetry
        present()
    }

    func showResult(_ latex: String) {
        viewModel.latex = latex
        viewModel.copyConfirmation = nil
        viewModel.phase = .result
        focus()
    }

    func showFailure(_ message: String) {
        viewModel.copyConfirmation = nil
        viewModel.phase = .failed(message)
        focus()
    }

    func showRetrying() {
        viewModel.copyConfirmation = nil
        viewModel.phase = .recognizing
        focus()
    }

    func focus() {
        guard let window else { return }
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
    }

    func dismiss() {
        closeHandler = nil
        window?.orderOut(nil)
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        requestClose()
        return false
    }

    private func present() {
        guard let window else { return }
        window.center()
        NSApp.activate(ignoringOtherApps: true)
        showWindow(nil)
        window.makeKeyAndOrderFront(nil)
    }

    private func requestClose() {
        closeHandler?()
    }
}
