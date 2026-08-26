import AppKit
import SwiftUI

@MainActor
final class RecognitionSettingsWindowController {
    private let settings: RecognitionModelSettings
    private var windowController: NSWindowController?

    init(settings: RecognitionModelSettings) {
        self.settings = settings
    }

    func show() {
        let controller: NSWindowController
        if let windowController {
            controller = windowController
        } else {
            let hostingController = NSHostingController(
                rootView: RecognitionSettingsView(settings: settings)
            )
            let window = NSWindow(contentViewController: hostingController)
            window.title = "EqnSnap 设置"
            window.styleMask = [.titled, .closable]
            window.isReleasedWhenClosed = false
            window.setContentSize(NSSize(width: 420, height: 140))
            window.center()
            let created = NSWindowController(window: window)
            windowController = created
            controller = created
        }

        NSApp.activate(ignoringOtherApps: true)
        controller.showWindow(nil)
        controller.window?.makeKeyAndOrderFront(nil)
    }

    func dismiss() {
        windowController?.close()
    }
}
