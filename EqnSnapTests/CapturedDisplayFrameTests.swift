import CoreGraphics
import XCTest
@testable import EqnSnap

final class CapturedDisplayFrameTests: XCTestCase {
    func testCropsSelectedPixelRectangle() throws {
        let context = try XCTUnwrap(
            CGContext(
                data: nil,
                width: 8,
                height: 6,
                bitsPerComponent: 8,
                bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
        )
        let image = try XCTUnwrap(context.makeImage())
        let frame = CapturedDisplayFrame(
            image: image,
            geometry: CaptureDisplayGeometry(
                displayID: 1,
                bottomLeftGlobalFrame: PointRect(
                    rawValue: CGRect(x: 0, y: 0, width: 4, height: 3)
                ),
                localBounds: PointRect(
                    rawValue: CGRect(x: 0, y: 0, width: 4, height: 3)
                ),
                pixelSize: PixelSize(width: 8, height: 6)
            )
        )

        let cropped = try frame.crop(
            to: PixelRect(x: 2, y: 1, width: 4, height: 3)
        )

        XCTAssert(cropped.width == 4)
        XCTAssert(cropped.height == 3)
    }

    func testRejectsOutOfBoundsCrop() throws {
        let context = try XCTUnwrap(
            CGContext(
                data: nil,
                width: 4,
                height: 4,
                bitsPerComponent: 8,
                bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
        )
        let image = try XCTUnwrap(context.makeImage())
        let frame = CapturedDisplayFrame(
            image: image,
            geometry: CaptureDisplayGeometry(
                displayID: 1,
                bottomLeftGlobalFrame: PointRect(rawValue: .zero),
                localBounds: PointRect(
                    rawValue: CGRect(x: 0, y: 0, width: 4, height: 4)
                ),
                pixelSize: PixelSize(width: 4, height: 4)
            )
        )

        XCTAssertThrowsError(
            try frame.crop(
                to: PixelRect(x: 3, y: 3, width: 2, height: 2)
            )
        ) { error in
            XCTAssertEqual(
                error as? FormulaCaptureError,
                .invalidGeometry
            )
        }
    }
}
