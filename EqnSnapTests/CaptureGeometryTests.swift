import CoreGraphics
import XCTest
@testable import EqnSnap

final class CaptureGeometryTests: XCTestCase {
    func testMapsRetinaSelectionUsingActualCapturedPixelSize() throws {
        let geometry = makeGeometry(
            pointSize: CGSize(width: 1_440, height: 900),
            pixelSize: PixelSize(width: 2_880, height: 1_800)
        )
        let rect = PointRect<DisplayLocalPoints>(
            rawValue: CGRect(x: 10.25, y: 20.25, width: 100.5, height: 50.5)
        )

        let result = try CaptureGeometryMapper().pixelRect(
            from: rect,
            geometry: geometry
        )

        XCTAssert(result == PixelRect(x: 20, y: 40, width: 202, height: 102))
    }

    func testClipsSelectionAtFractionalDisplayScale() throws {
        let geometry = makeGeometry(
            pointSize: CGSize(width: 1_280, height: 800),
            pixelSize: PixelSize(width: 1_920, height: 1_200)
        )
        let rect = PointRect<DisplayLocalPoints>(
            rawValue: CGRect(x: -2.2, y: 799.2, width: 10, height: 5)
        )

        let result = try CaptureGeometryMapper().pixelRect(
            from: rect,
            geometry: geometry
        )

        XCTAssert(result == PixelRect(x: 0, y: 1_198, width: 12, height: 2))
    }

    private func makeGeometry(
        pointSize: CGSize,
        pixelSize: PixelSize
    ) -> CaptureDisplayGeometry {
        CaptureDisplayGeometry(
            displayID: 7,
            bottomLeftGlobalFrame: PointRect(
                rawValue: CGRect(origin: .zero, size: pointSize)
            ),
            localBounds: PointRect(
                rawValue: CGRect(origin: .zero, size: pointSize)
            ),
            pixelSize: pixelSize
        )
    }
}
