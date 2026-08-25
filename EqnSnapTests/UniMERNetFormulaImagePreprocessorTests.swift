import CoreGraphics
import CoreML
import Foundation
import ImageIO
import XCTest
@testable import EqnSnap

final class UniMERNetFormulaImagePreprocessorTests: XCTestCase {
    func testProducesFixedUniMERNetInputGeometry() throws {
        let prepared = try UniMERNetFormulaImagePreprocessor().prepare(
            loadFormulaImage()
        )

        XCTAssertEqual(prepared.sourcePixelSize, CGSize(width: 726, height: 109))
        XCTAssertEqual(
            prepared.croppedPixelRect,
            CGRect(x: 9, y: 10, width: 708, height: 79)
        )
        XCTAssertEqual(prepared.contentPixelSize, CGSize(width: 672, height: 75))
        XCTAssertEqual(prepared.canvasSize, CGSize(width: 672, height: 192))
        XCTAssertEqual(prepared.tensor.shape.map(\.intValue), [1, 3, 192, 672])
        XCTAssertEqual(prepared.tensor.dataType, .float32)
    }

    func testStaysNumericallyCloseToPythonFixture() throws {
        let actual = try UniMERNetFormulaImagePreprocessor().prepare(
            loadFormulaImage()
        ).tensor
        let expected = try loadFloat32NPY(
            repositoryRoot().appendingPathComponent(
                "EqnSnapTests/Fixtures/UniMERNet/preprocessor-input.npy"
            ),
            shape: [1, 3, 192, 672]
        )

        let comparison = compare(actual, expected)
        XCTAssertLessThan(
            comparison.maximumAbsoluteError,
            1.5,
            "max error: \(comparison.maximumAbsoluteError)"
        )
        XCTAssertLessThan(
            comparison.meanAbsoluteError,
            0.025,
            "mean error: \(comparison.meanAbsoluteError)"
        )
    }

    private func loadFormulaImage() throws -> CGImage {
        let url = repositoryRoot().appendingPathComponent(
            "Tools/model-conversion/test_images/formula.png"
        )
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw UniMERNetFixtureError.unreadableImage
        }
        return image
    }

    private func repositoryRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func loadFloat32NPY(_ url: URL, shape: [Int]) throws
        -> MLMultiArray
    {
        let data = try Data(contentsOf: url)
        guard data.count >= 10,
              Array(data.prefix(6)) == [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59],
              data[6] == 1,
              data[7] == 0
        else {
            throw UniMERNetFixtureError.unsupportedNPYHeader
        }
        let headerLength = Int(data[8]) | (Int(data[9]) << 8)
        let dataOffset = 10 + headerLength
        let valueCount = shape.reduce(1, *)
        let byteCount = valueCount * MemoryLayout<Float32>.size
        guard dataOffset + byteCount == data.count else {
            throw UniMERNetFixtureError.invalidNPYDataLength
        }

        let array = try MLMultiArray(
            shape: shape.map(NSNumber.init(value:)),
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

    private func compare(_ actual: MLMultiArray, _ expected: MLMultiArray)
        -> (maximumAbsoluteError: Float32, meanAbsoluteError: Float32)
    {
        let actualPointer = actual.dataPointer.assumingMemoryBound(to: Float32.self)
        let expectedPointer = expected.dataPointer.assumingMemoryBound(to: Float32.self)
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

private enum UniMERNetFixtureError: Error {
    case unreadableImage
    case unsupportedNPYHeader
    case invalidNPYDataLength
}
