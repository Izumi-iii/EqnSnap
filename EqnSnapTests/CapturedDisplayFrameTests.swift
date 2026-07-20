import CoreGraphics
import Testing
@testable import EqnSnap

struct CapturedDisplayFrameTests {
    @Test func cropsSelectedPixelRectangle() throws {
        let context = try #require(
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
        let image = try #require(context.makeImage())
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

        #expect(cropped.width == 4)
        #expect(cropped.height == 3)
    }

    @Test func rejectsOutOfBoundsCrop() throws {
        let context = try #require(
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
        let image = try #require(context.makeImage())
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

        #expect(throws: FormulaCaptureError.invalidGeometry) {
            _ = try frame.crop(
                to: PixelRect(x: 3, y: 3, width: 2, height: 2)
            )
        }
    }
}
