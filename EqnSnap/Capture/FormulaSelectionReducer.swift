import CoreGraphics

// Region-only adaptation of SnapioCore/SelectionReducer.swift.
nonisolated enum FormulaSelectionState: Equatable, Sendable {
    case ready
    case dragging(
        anchor: Point2<DisplayLocalPoints>,
        current: Point2<DisplayLocalPoints>
    )
}

nonisolated enum FormulaSelectionAction: Equatable, Sendable {
    case primaryDown(Point2<DisplayLocalPoints>)
    case primaryDragged(Point2<DisplayLocalPoints>)
    case primaryUp(Point2<DisplayLocalPoints>)
    case cancel
}

nonisolated enum FormulaSelectionEffect: Equatable, Sendable {
    case none
    case commit(PixelRect)
    case cancel
}

nonisolated struct FormulaSelectionReducer: Sendable {
    private let geometryMapper = CaptureGeometryMapper()
    private let minimumDragDistanceSquared: CGFloat = 9

    mutating func reduce(
        state: inout FormulaSelectionState,
        action: FormulaSelectionAction,
        geometry: CaptureDisplayGeometry
    ) -> FormulaSelectionEffect {
        if case .cancel = action {
            return .cancel
        }

        switch state {
        case .ready:
            guard case .primaryDown(let point) = action else {
                return .none
            }
            let clamped = clamp(point, to: geometry)
            state = .dragging(anchor: clamped, current: clamped)
            return .none

        case .dragging(let anchor, _):
            switch action {
            case .primaryDragged(let point):
                state = .dragging(
                    anchor: anchor,
                    current: clamp(point, to: geometry)
                )
                return .none
            case .primaryUp(let point):
                let current = clamp(point, to: geometry)
                guard exceededMinimumDistance(from: anchor, to: current) else {
                    return .cancel
                }
                let rect = selectionRect(from: anchor, to: current)
                guard let pixelRect = try? geometryMapper.pixelRect(
                    from: rect,
                    geometry: geometry
                ) else {
                    return .cancel
                }
                state = .dragging(anchor: anchor, current: current)
                return .commit(pixelRect)
            case .primaryDown, .cancel:
                return .none
            }
        }
    }

    func selectionRect(
        for state: FormulaSelectionState
    ) -> PointRect<DisplayLocalPoints>? {
        guard case .dragging(let anchor, let current) = state else {
            return nil
        }
        return selectionRect(from: anchor, to: current)
    }

    private func selectionRect(
        from anchor: Point2<DisplayLocalPoints>,
        to current: Point2<DisplayLocalPoints>
    ) -> PointRect<DisplayLocalPoints> {
        PointRect(
            rawValue: CGRect(
                x: min(anchor.rawValue.x, current.rawValue.x),
                y: min(anchor.rawValue.y, current.rawValue.y),
                width: abs(current.rawValue.x - anchor.rawValue.x),
                height: abs(current.rawValue.y - anchor.rawValue.y)
            )
        )
    }

    private func clamp(
        _ point: Point2<DisplayLocalPoints>,
        to geometry: CaptureDisplayGeometry
    ) -> Point2<DisplayLocalPoints> {
        let bounds = geometry.localBounds.rawValue
        return Point2(
            rawValue: CGPoint(
                x: min(max(point.rawValue.x, 0), bounds.width),
                y: min(max(point.rawValue.y, 0), bounds.height)
            )
        )
    }

    private func exceededMinimumDistance(
        from anchor: Point2<DisplayLocalPoints>,
        to current: Point2<DisplayLocalPoints>
    ) -> Bool {
        let deltaX = current.rawValue.x - anchor.rawValue.x
        let deltaY = current.rawValue.y - anchor.rawValue.y
        return deltaX * deltaX + deltaY * deltaY
            >= minimumDragDistanceSquared
    }
}
