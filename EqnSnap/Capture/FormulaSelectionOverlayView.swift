import AppKit
import CoreGraphics

@MainActor
protocol FormulaSelectionOverlayViewDelegate: AnyObject {
    func selectionOverlayView(
        _ view: FormulaSelectionOverlayView,
        didSend action: FormulaSelectionAction
    )
}

struct FormulaSelectionOverlaySnapshot: @unchecked Sendable {
    let displayFrame: CGImage
    let selectionRect: CGRect?
    let pixelSize: PixelSize?
}

// Region-only adaptation of SnapioUI/SelectionOverlayView.swift.
@MainActor
final class FormulaSelectionOverlayView: NSView {
    weak var delegate: FormulaSelectionOverlayViewDelegate?

    override var isFlipped: Bool { true }
    override var isOpaque: Bool { true }
    override var acceptsFirstResponder: Bool { true }

    private let pixelSizeLabel = NSTextField(labelWithString: "")
    private var snapshot: FormulaSelectionOverlaySnapshot?
    private var backgroundImage: NSImage?

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        buildView()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func render(_ snapshot: FormulaSelectionOverlaySnapshot) {
        self.snapshot = snapshot
        if backgroundImage == nil {
            let image = snapshot.displayFrame
            backgroundImage = NSImage(
                cgImage: image,
                size: NSSize(width: image.width, height: image.height)
            )
        }
        if let pixelSize = snapshot.pixelSize {
            pixelSizeLabel.stringValue = (
                "\(pixelSize.width) × \(pixelSize.height) px"
            )
            pixelSizeLabel.isHidden = false
        } else {
            pixelSizeLabel.isHidden = true
        }
        needsDisplay = true
        needsLayout = true
    }

    override func draw(_ dirtyRect: NSRect) {
        super.draw(dirtyRect)
        drawBackground()

        guard let selectionRect = snapshot?.selectionRect else {
            NSColor.black.withAlphaComponent(0.38).setFill()
            bounds.fill()
            return
        }
        let clipped = selectionRect.standardized.intersection(bounds)
        guard !clipped.isNull, clipped.width > 0, clipped.height > 0 else {
            NSColor.black.withAlphaComponent(0.38).setFill()
            bounds.fill()
            return
        }

        let mask = NSBezierPath()
        mask.appendRect(bounds)
        mask.appendRect(clipped)
        mask.windingRule = .evenOdd
        NSColor.black.withAlphaComponent(0.42).setFill()
        mask.fill()

        let borderRect = clipped.insetBy(dx: 1, dy: 1)
        if borderRect.width > 0, borderRect.height > 0 {
            let border = NSBezierPath(rect: borderRect)
            border.lineWidth = 2.5
            NSColor.controlAccentColor.setStroke()
            border.stroke()
        }
    }

    override func layout() {
        super.layout()
        guard !pixelSizeLabel.isHidden,
              let selectionRect = snapshot?.selectionRect else {
            return
        }
        pixelSizeLabel.sizeToFit()
        let padding = CGSize(width: 12, height: 7)
        let size = CGSize(
            width: pixelSizeLabel.frame.width + padding.width,
            height: pixelSizeLabel.frame.height + padding.height
        )
        var origin = CGPoint(
            x: selectionRect.minX,
            y: selectionRect.maxY + 8
        )
        if origin.y + size.height > bounds.maxY - 8 {
            origin.y = selectionRect.minY - size.height - 8
        }
        origin.x = min(
            max(origin.x, bounds.minX + 8),
            bounds.maxX - size.width - 8
        )
        origin.y = min(
            max(origin.y, bounds.minY + 8),
            bounds.maxY - size.height - 8
        )
        pixelSizeLabel.frame = CGRect(origin: origin, size: size).integral
    }

    override func mouseDown(with event: NSEvent) {
        if event.modifierFlags.contains(.control) {
            send(.cancel)
        } else {
            send(.primaryDown(displayLocalPoint(for: event)))
        }
    }

    override func mouseDragged(with event: NSEvent) {
        send(.primaryDragged(displayLocalPoint(for: event)))
    }

    override func mouseUp(with event: NSEvent) {
        send(.primaryUp(displayLocalPoint(for: event)))
    }

    override func rightMouseDown(with event: NSEvent) {
        send(.cancel)
    }

    override func keyDown(with event: NSEvent) {
        if event.keyCode == 53 {
            send(.cancel)
        } else {
            super.keyDown(with: event)
        }
    }

    override func resetCursorRects() {
        addCursorRect(bounds, cursor: .crosshair)
    }

    private func buildView() {
        toolTip = "拖动选择公式区域，松开后立即识别；按 Esc 取消。"
        pixelSizeLabel.font = .monospacedDigitSystemFont(
            ofSize: NSFont.smallSystemFontSize,
            weight: .medium
        )
        pixelSizeLabel.textColor = .white
        pixelSizeLabel.alignment = .center
        pixelSizeLabel.backgroundColor = NSColor.black.withAlphaComponent(0.72)
        pixelSizeLabel.drawsBackground = true
        pixelSizeLabel.wantsLayer = true
        pixelSizeLabel.layer?.cornerRadius = 5
        pixelSizeLabel.layer?.masksToBounds = true
        pixelSizeLabel.isHidden = true
        addSubview(pixelSizeLabel)
    }

    private func drawBackground() {
        guard let backgroundImage else {
            NSColor.black.setFill()
            bounds.fill()
            return
        }
        NSGraphicsContext.current?.imageInterpolation = .high
        backgroundImage.draw(
            in: bounds,
            from: .zero,
            operation: .copy,
            fraction: 1,
            respectFlipped: true,
            hints: nil
        )
    }

    private func displayLocalPoint(
        for event: NSEvent
    ) -> Point2<DisplayLocalPoints> {
        let point = convert(event.locationInWindow, from: nil)
        return Point2(
            rawValue: CGPoint(
                x: min(max(point.x, bounds.minX), bounds.maxX),
                y: min(max(point.y, bounds.minY), bounds.maxY)
            )
        )
    }

    private func send(_ action: FormulaSelectionAction) {
        delegate?.selectionOverlayView(self, didSend: action)
    }
}
