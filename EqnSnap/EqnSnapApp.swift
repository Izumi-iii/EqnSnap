//
//  EqnSnapApp.swift
//  EqnSnap
//
//  Created by wangjie on 2026/7/16.
//

import SwiftUI

@main
struct EqnSnapApp: App {
    @NSApplicationDelegateAdaptor(EqnSnapAppDelegate.self)
    private var appDelegate

    var body: some Scene {
        Settings {
            RecognitionSettingsView(settings: .shared)
        }
    }
}

@MainActor
final class EqnSnapAppDelegate: NSObject, NSApplicationDelegate {
    private var workflow: FormulaCaptureWorkflow?
#if DEBUG
    private var previewResultWindowController: FormulaResultWindowController?
    private var previewSettingsWindowController: RecognitionSettingsWindowController?
#endif

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        let workflow = FormulaCaptureWorkflow()
        workflow.start()
        self.workflow = workflow
#if DEBUG
        if ProcessInfo.processInfo.arguments.contains("--preview-result-window") {
            showResultWindowPreview()
        }
        if ProcessInfo.processInfo.arguments.contains("--preview-settings-window") {
            DispatchQueue.main.async {
                let controller = RecognitionSettingsWindowController(
                    settings: .shared
                )
                self.previewSettingsWindowController = controller
                controller.show()
            }
        }
#endif
    }

    func applicationWillTerminate(_ notification: Notification) {
        workflow?.stop()
    }

    func applicationSupportsSecureRestorableState(
        _ app: NSApplication
    ) -> Bool {
        true
    }

#if DEBUG
    private func showResultWindowPreview() {
        let image = NSImage(size: NSSize(width: 560, height: 92))
        image.lockFocus()
        NSColor.white.setFill()
        NSRect(origin: .zero, size: image.size).fill()
        let formula = "∫₀ˣ (1 − t²) dt = x − x³⁄3"
        formula.draw(
            at: NSPoint(x: 54, y: 28),
            withAttributes: [
                .font: NSFont.systemFont(ofSize: 28),
                .foregroundColor: NSColor.black,
            ]
        )
        image.unlockFocus()

        var rect = NSRect(origin: .zero, size: image.size)
        guard let screenshot = image.cgImage(
            forProposedRect: &rect,
            context: nil,
            hints: nil
        ) else {
            return
        }

        let controller = FormulaResultWindowController()
        previewResultWindowController = controller
        controller.showRecognizing(
            screenshot: screenshot,
            onRetry: {},
            onClose: { [weak self, weak controller] in
                controller?.dismiss()
                self?.previewResultWindowController = nil
            }
        )
        controller.showResult(
            #"\int_{0}^{x}\left(1-t^{2}\right)\,dt=x-\frac{x^{3}}{3}"#
        )
    }
#endif
}
