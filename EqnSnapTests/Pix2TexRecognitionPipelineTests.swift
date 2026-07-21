import CoreML
import Foundation
import XCTest
@testable import EqnSnap

final class Pix2TexRecognitionPipelineTests: XCTestCase {
    func testConnectsEncoderDecoderAndTokenizer() async throws {
        let tokenizer = try loadTokenizer()
        let encoder = PipelineEncoderPredictor(marker: 42)
        let decoder = PipelineDecoderPredictor(
            expectedMarker: 42,
            tokens: [30, 2]
        )
        let pipeline = Pix2TexRecognitionPipeline(
            encoderRunner: Pix2TexEncoderRunner(predictor: encoder),
            decoderRunner: Pix2TexDecoderRunner(predictor: decoder),
            tokenizer: tokenizer
        )

        let result = try await pipeline.recognize(
            pixelValues: makeEmptyInput(height: 32, width: 32)
        )

        XCTAssert(result.latex == "=")
        XCTAssert(result.tokenIDs == [30])
        XCTAssert(result.decoderSteps == 2)
        XCTAssert(decoder.observedContextMarker == 42)
    }

    func testRunsConvertedCoreMLModelsEndToEnd() async throws {
        let repositoryRoot = repositoryRoot()
        let artifacts = repositoryRoot.appendingPathComponent(
            "Tools/model-conversion/artifacts"
        )
        let encoderURL = artifacts.appendingPathComponent(
            "Pix2TexEncoder-Variable32.mlpackage"
        )
        let decoderURL = artifacts.appendingPathComponent(
            "Pix2TexDecoder-Prefix128-Context169.mlpackage"
        )
        guard FileManager.default.fileExists(atPath: encoderURL.path),
              FileManager.default.fileExists(atPath: decoderURL.path)
        else {
            return
        }

        let loader = Pix2TexModelBundleLoader()
        let loaded = try loader.load(
            encoderURL: encoderURL,
            decoderURL: decoderURL,
            tokenizerURL: repositoryRoot.appendingPathComponent(
                "EqnSnap/Resources/Models/Pix2Tex/tokenizer.json"
            )
        )
        let input = try loadFloat32NPY(
            artifacts.appendingPathComponent(
                "fixtures/encoder-variable32/target-height-24/input.npy"
            ),
            shape: [1, 1, 32, 224]
        )

        let result = try await loaded.makeRecognitionPipeline().recognize(
            pixelValues: input
        )

        XCTAssert(result.decoderSteps == 79)
        XCTAssert(
            result.latex
                == "=-\\int_{0}^{1-\\ell-\\ell-1}\\!\\!\\!d t=-\\int_{0}^{1}(1-{\\frac{1}{1-\\ell}})d t=-\\mathrm{lin}(1-x)-{\\bar{x}}"
        )
    }

    private func loadTokenizer() throws -> Pix2TexTokenizer {
        try Pix2TexTokenizer(
            contentsOf: repositoryRoot().appendingPathComponent(
                "EqnSnap/Resources/Models/Pix2Tex/tokenizer.json"
            )
        )
    }

    private func repositoryRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func makeEmptyInput(
        height: Int,
        width: Int
    ) throws -> MLMultiArray {
        try MLMultiArray(
            shape: [1, 1, NSNumber(value: height), NSNumber(value: width)],
            dataType: .float32
        )
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
            throw NPYFixtureError.unsupportedHeader
        }
        let headerLength = Int(data[8]) | (Int(data[9]) << 8)
        let dataOffset = 10 + headerLength
        let valueCount = shape.reduce(1, *)
        let byteCount = valueCount * MemoryLayout<Float32>.size
        guard dataOffset + byteCount == data.count else {
            throw NPYFixtureError.invalidDataLength
        }

        let array = try MLMultiArray(
            shape: shape.map(NSNumber.init(value:)),
            dataType: .float32
        )
        let destination = UnsafeMutableRawBufferPointer(
            start: array.dataPointer,
            count: byteCount
        )
        data.copyBytes(to: destination, from: dataOffset..<data.count)
        return array
    }
}

private enum NPYFixtureError: Error {
    case unsupportedHeader
    case invalidDataLength
}

private final class PipelineEncoderPredictor:
    Pix2TexEncoderPredicting
{
    private let marker: Float32

    init(marker: Float32) {
        self.marker = marker
    }

    func predict(pixelValues: MLMultiArray) throws -> MLMultiArray {
        let shape = pixelValues.shape.map(\.intValue)
        let contextLength = (shape[2] / 16) * (shape[3] / 16) + 1
        let context = try MLMultiArray(
            shape: [1, NSNumber(value: contextLength), 256],
            dataType: .float32
        )
        context.dataPointer.assumingMemoryBound(to: Float32.self)[0] = marker
        return context
    }
}

private final class PipelineDecoderPredictor:
    Pix2TexDecoderPredicting
{
    private let expectedMarker: Float32
    private let tokens: [Int32]
    private var callIndex = 0
    private(set) var observedContextMarker: Float32?

    init(expectedMarker: Float32, tokens: [Int32]) {
        self.expectedMarker = expectedMarker
        self.tokens = tokens
    }

    func predict(
        inputIDs: MLMultiArray,
        tokenMask: MLMultiArray,
        encoderContext: MLMultiArray,
        contextMask: MLMultiArray
    ) throws -> MLMultiArray {
        let marker = encoderContext.dataPointer.assumingMemoryBound(
            to: Float32.self
        )[0]
        observedContextMarker = marker
        guard marker == expectedMarker else {
            throw PipelineFixtureError.contextWasNotForwarded
        }

        let logits = try MLMultiArray(
            shape: [1, 1, 8_000],
            dataType: .float32
        )
        let pointer = logits.dataPointer.assumingMemoryBound(to: Float32.self)
        for index in 0..<logits.count {
            pointer[index] = -1_000
        }
        let token = tokens[min(callIndex, tokens.count - 1)]
        pointer[Int(token) * logits.strides[2].intValue] = 1_000
        callIndex += 1
        return logits
    }
}

private enum PipelineFixtureError: Error {
    case contextWasNotForwarded
}
