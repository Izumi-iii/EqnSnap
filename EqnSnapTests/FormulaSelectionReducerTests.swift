import CoreGraphics
import XCTest
@testable import EqnSnap

final class FormulaSelectionReducerTests: XCTestCase {
    private let geometry = CaptureDisplayGeometry(
        displayID: 7,
        bottomLeftGlobalFrame: PointRect(
            rawValue: CGRect(x: -100, y: 40, width: 100, height: 80)
        ),
        localBounds: PointRect(
            rawValue: CGRect(x: 0, y: 0, width: 100, height: 80)
        ),
        pixelSize: PixelSize(width: 200, height: 160)
    )

    func testCommitsReverseDragAndClipsToDisplay() {
        var reducer = FormulaSelectionReducer()
        var state = FormulaSelectionState.ready
        _ = reducer.reduce(
            state: &state,
            action: .primaryDown(point(90, 70)),
            geometry: geometry
        )
        _ = reducer.reduce(
            state: &state,
            action: .primaryDragged(point(-10, -20)),
            geometry: geometry
        )

        let effect = reducer.reduce(
            state: &state,
            action: .primaryUp(point(-10, -20)),
            geometry: geometry
        )

        XCTAssert(
            effect == .commit(
                PixelRect(x: 0, y: 0, width: 180, height: 140)
            )
        )
    }

    func testClickWithoutDragCancelsSelection() {
        var reducer = FormulaSelectionReducer()
        var state = FormulaSelectionState.ready
        _ = reducer.reduce(
            state: &state,
            action: .primaryDown(point(10, 10)),
            geometry: geometry
        )

        let effect = reducer.reduce(
            state: &state,
            action: .primaryUp(point(11, 11)),
            geometry: geometry
        )

        XCTAssert(effect == .cancel)
    }

    func testEscapeCancelsActiveDrag() {
        var reducer = FormulaSelectionReducer()
        var state = FormulaSelectionState.ready
        _ = reducer.reduce(
            state: &state,
            action: .primaryDown(point(10, 10)),
            geometry: geometry
        )

        let effect = reducer.reduce(
            state: &state,
            action: .cancel,
            geometry: geometry
        )

        XCTAssert(effect == .cancel)
    }

    private func point(
        _ x: CGFloat,
        _ y: CGFloat
    ) -> Point2<DisplayLocalPoints> {
        Point2(rawValue: CGPoint(x: x, y: y))
    }
}
