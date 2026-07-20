import AppKit

@MainActor
final class FormulaSelectionOverlayViewController:
    NSViewController,
    FormulaSelectionOverlayViewDelegate
{
    let sessionID: CaptureSessionID

    private let frame: CapturedDisplayFrame
    private let onCommit: (PixelRect) -> Void
    private let onCancel: () -> Void
    private let geometryMapper = CaptureGeometryMapper()
    private var reducer = FormulaSelectionReducer()
    private var selectionState = FormulaSelectionState.ready
    private var isFinished = false

    init(
        sessionID: CaptureSessionID,
        frame: CapturedDisplayFrame,
        onCommit: @escaping (PixelRect) -> Void,
        onCancel: @escaping () -> Void
    ) {
        self.sessionID = sessionID
        self.frame = frame
        self.onCommit = onCommit
        self.onCancel = onCancel
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func loadView() {
        view = FormulaSelectionOverlayView(frame: .zero)
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        overlayView.delegate = self
        render()
    }

    func stop() {
        isFinished = true
        if isViewLoaded {
            overlayView.delegate = nil
        }
    }

    func selectionOverlayView(
        _ view: FormulaSelectionOverlayView,
        didSend action: FormulaSelectionAction
    ) {
        guard !isFinished else { return }
        let effect = reducer.reduce(
            state: &selectionState,
            action: action,
            geometry: frame.geometry
        )
        switch effect {
        case .none:
            render()
        case .commit(let pixelRect):
            isFinished = true
            onCommit(pixelRect)
        case .cancel:
            isFinished = true
            onCancel()
        }
    }

    private func render() {
        let selectionRect = reducer.selectionRect(
            for: selectionState
        )?.rawValue
        let pixelSize = selectionRect.flatMap {
            try? geometryMapper.pixelSize(
                of: PointRect(rawValue: $0),
                geometry: frame.geometry
            )
        }
        overlayView.render(
            FormulaSelectionOverlaySnapshot(
                displayFrame: frame.image,
                selectionRect: selectionRect,
                pixelSize: pixelSize
            )
        )
    }

    private var overlayView: FormulaSelectionOverlayView {
        guard let view = view as? FormulaSelectionOverlayView else {
            preconditionFailure("Unexpected overlay view type")
        }
        return view
    }
}

@MainActor
private final class FormulaCaptureOverlayPanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}

// Adapted from SnapioPresentation/SelectionOverlayWindowController.swift.
@MainActor
final class FormulaSelectionOverlayWindowController: NSWindowController {
    init(contentViewController: FormulaSelectionOverlayViewController) {
        let panel = FormulaCaptureOverlayPanel(
            contentRect: NSRect(x: 0, y: 0, width: 1, height: 1),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        panel.title = "EqnSnap Formula Selection"
        panel.isOpaque = true
        panel.backgroundColor = .black
        panel.hasShadow = false
        panel.level = .screenSaver
        panel.animationBehavior = .none
        panel.hidesOnDeactivate = false
        panel.isReleasedWhenClosed = false
        panel.acceptsMouseMovedEvents = true
        panel.sharingType = .none
        panel.collectionBehavior = [
            .moveToActiveSpace,
            .fullScreenAuxiliary,
            .transient,
            .ignoresCycle,
        ]
        panel.contentViewController = contentViewController
        super.init(window: panel)
        shouldCascadeWindows = false
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func present(on displayFrame: CGRect) {
        guard displayFrame.isFinite,
              !displayFrame.isNull,
              !displayFrame.isEmpty,
              let window else {
            return
        }
        window.setFrame(displayFrame.standardized, display: true)
        NSApp.activate(ignoringOtherApps: true)
        showWindow(nil)
        window.makeKeyAndOrderFront(nil)
        window.makeFirstResponder(contentViewController?.view)
    }

    func dismiss() {
        window?.orderOut(nil)
    }
}

private extension CGRect {
    var isFinite: Bool {
        origin.x.isFinite
            && origin.y.isFinite
            && size.width.isFinite
            && size.height.isFinite
    }
}
