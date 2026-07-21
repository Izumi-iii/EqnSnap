import CoreGraphics
import CoreML
import Foundation
import ImageIO
import XCTest
@testable import EqnSnap

final class Pix2TexFormulaImagePreprocessorTests: XCTestCase {
    func testMatchesPythonStrokeProfileAndPreparedShape() throws {
        let root = repositoryRoot()
        let image = try loadCGImage(
            root.appendingPathComponent(
                "Tools/model-conversion/test_images/formula.png"
            )
        )

        let prepared = try Pix2TexFormulaImagePreprocessor().prepare(image)

        XCTAssert(prepared.sourcePixelSize == CGSize(width: 726, height: 109))
        XCTAssert(
            prepared.croppedPixelRect
                == CGRect(x: 9, y: 11, width: 708, height: 78)
        )
        XCTAssert(prepared.strokeMetrics.otsuThreshold == 140)
        XCTAssert(prepared.strokeMetrics.foregroundArea == 3_671)
        XCTAssert(prepared.strokeMetrics.foregroundPerimeter == 3_582)
        XCTAssert(prepared.targetForegroundHeight == 40)
        XCTAssert(prepared.actualForegroundSize == CGSize(width: 363, height: 40))
        XCTAssert(prepared.canvasSize == CGSize(width: 384, height: 64))
        XCTAssert(prepared.tensor.shape.map(\.intValue) == [1, 1, 64, 384])
    }

    func testStaysNumericallyCloseToPythonTensorFixture() throws {
        let root = repositoryRoot()
        let fixtureURL = root.appendingPathComponent(
            "Tools/model-conversion/artifacts/fixtures/encoder-variable32/target-height-40/input.npy"
        )
        guard FileManager.default.fileExists(atPath: fixtureURL.path) else {
            return
        }
        let image = try loadCGImage(
            root.appendingPathComponent(
                "Tools/model-conversion/test_images/formula.png"
            )
        )
        let actual = try Pix2TexFormulaImagePreprocessor().prepare(image).tensor
        let expected = try loadFloat32NPY(
            fixtureURL,
            shape: [1, 1, 64, 384]
        )

        let comparison = compare(actual, expected)
        // vImage high-quality resampling is not bit-identical to Pillow's
        // Lanczos implementation, but it must stay close enough to preserve
        // the model's generated token sequence.
        XCTAssert(comparison.maximumAbsoluteError < 0.5)
        XCTAssert(
            comparison.meanAbsoluteError < 0.02,
            "mean error: \(comparison.meanAbsoluteError)"
        )
    }

    func testRecognizesScreenshotUsingBundledModels() async throws {
        let image = try loadCGImage(
            repositoryRoot().appendingPathComponent(
                "Tools/model-conversion/test_images/formula.png"
            )
        )
        let recognizer = try Pix2TexScreenshotRecognizer(
            bundle: .main,
            computeUnits: .all
        )

        let result = try await recognizer.recognize(image)

        XCTAssert(result.decoderSteps == 72)
        XCTAssert(
            result.latex
                == "=-\\int_{0}^{x}{\\frac{1-t-1}{1-t}}d t=-\\int_{0}^{x}(1-{\\frac{1}{1-t}})d t=-\\ln(1-x)-x"
        )
    }

    private func repositoryRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func loadCGImage(_ url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw PreprocessingFixtureError.unreadableImage
        }
        return image
    }

    private func loadFloat32NPY(
        _ url: URL,
        shape: [Int]
    ) throws -> MLMultiArray {
        let data = try Data(contentsOf: url)
        guard data.count >= 10,
              Array(data.prefix(6)) == [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59],
              data[6] == 1,
              data[7] == 0
        else {
            throw PreprocessingFixtureError.unsupportedNPYHeader
        }
        let headerLength = Int(data[8]) | (Int(data[9]) << 8)
        let dataOffset = 10 + headerLength
        let count = shape.reduce(1, *)
        let byteCount = count * MemoryLayout<Float32>.size
        guard dataOffset + byteCount == data.count else {
            throw PreprocessingFixtureError.invalidNPYDataLength
        }

        let array = try MLMultiArray(
            shape: shape.map { NSNumber(value: $0) },
            dataType: .float32
        )
        data.copyBytes(
            to: UnsafeMutableRawBufferPointer(
                start: array.dataPointer,
                count: byteCount
            ),
            from: dataOffset..<data.count
        )
        return array
    }

    private func compare(
        _ actual: MLMultiArray,
        _ expected: MLMultiArray
    ) -> (maximumAbsoluteError: Float32, meanAbsoluteError: Float32) {
        let actualPointer = actual.dataPointer.assumingMemoryBound(
            to: Float32.self
        )
        let expectedPointer = expected.dataPointer.assumingMemoryBound(
            to: Float32.self
        )
        var maximum: Float32 = 0
        var sum: Float32 = 0
        for index in 0..<actual.count {
            let difference = abs(actualPointer[index] - expectedPointer[index])
            maximum = max(maximum, difference)
            sum += difference
        }
        return (maximum, sum / Float32(actual.count))
    }
}

private enum PreprocessingFixtureError: Error {
    case unreadableImage
    case unsupportedNPYHeader
    case invalidNPYDataLength
}
