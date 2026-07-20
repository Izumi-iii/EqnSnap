import AppKit

// EqnSnap-specific reduction of SnapioPresentation/StatusItemController.swift.
@MainActor
final class EqnSnapStatusItemController: NSObject {
    var onCapture: (() -> Void)?
    var onOpenScreenCaptureSettings: (() -> Void)?

    private var statusItem: NSStatusItem?
    private let menu = NSMenu()
    private let captureItem = NSMenuItem()

    override init() {
        super.init()
        buildMenu()
    }

    func install() {
        guard statusItem == nil else { return }
        let item = NSStatusBar.system.statusItem(
            withLength: NSStatusItem.variableLength
        )
        if let button = item.button {
            button.toolTip = "EqnSnap"
            button.image = nil
            button.title = "ƒx"
            button.font = .systemFont(
                ofSize: NSFont.systemFontSize,
                weight: .semibold
            )
            button.setAccessibilityLabel("EqnSnap")
        }
        item.menu = menu
        statusItem = item
    }

    func setCaptureEnabled(_ enabled: Bool) {
        captureItem.isEnabled = enabled
    }

    func uninstall() {
        guard let statusItem else { return }
        NSStatusBar.system.removeStatusItem(statusItem)
        self.statusItem = nil
    }

    private func buildMenu() {
        captureItem.title = "截取公式…"
        captureItem.target = self
        captureItem.action = #selector(startCapture)
        captureItem.keyEquivalent = "e"
        captureItem.keyEquivalentModifierMask = [.command, .control]

        let permissionItem = NSMenuItem(
            title: "屏幕录制权限…",
            action: #selector(openScreenCaptureSettings),
            keyEquivalent: ""
        )
        permissionItem.target = self

        let quitItem = NSMenuItem(
            title: "退出 EqnSnap",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        )
        quitItem.target = NSApplication.shared

        menu.addItem(captureItem)
        menu.addItem(.separator())
        menu.addItem(permissionItem)
        menu.addItem(.separator())
        menu.addItem(quitItem)
    }

    @objc private func startCapture() {
        onCapture?()
    }

    @objc private func openScreenCaptureSettings() {
        onOpenScreenCaptureSettings?()
    }
}
