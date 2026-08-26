import CoreML
import CoreGraphics
import Foundation
import ImageIO
import XCTest
@testable import EqnSnap

final class UniMERNetRecognitionPipelineTests: XCTestCase {
    func testConnectsEncoderDecoderAndTokenizer() async throws {
        let encoderConfiguration = UniMERNetEncoderConfiguration(
            inputShape: [1, 3, 2, 2],
            outputShape: [1, 3, 4]
        )
        let decoderConfiguration = makeDecoderConfiguration()
        let encoder = PipelineUniMERNetEncoderPredictor(
            output: try markedContext(marker: 42)
        )
        let decoder = PipelineUniMERNetDecoderPredictor(
            expectedMarker: 42,
            tokens: [110, 2],
            configuration: decoderConfiguration
        )
        let pipeline = UniMERNetRecognitionPipeline(
            encoderRunner: UniMERNetEncoderRunner(
                predictor: encoder,
                configuration: encoderConfiguration
            ),
            decoderRunner: UniMERNetCachedDecoderRunner(
                predictor: decoder,
                configuration: decoderConfiguration
            ),
            tokenizer: try loadTokenizer()
        )

        let result = try await pipeline.recognize(
            pixelValues: MLMultiArray(
                shape: [1, 3, 2, 2],
                dataType: .float32
            )
        )

        XCTAssertEqual(result.latex, "x")
        XCTAssertEqual(result.tokenIDs, [110])
        XCTAssertEqual(result.decoderSteps, 2)
        XCTAssertTrue(decoder.observedExpectedMarker)
    }

    func testRunsConvertedModelsFromScreenshotEndToEnd() async throws {
        let recognizer = UniMERNetScreenshotRecognizer(
            preprocessor: UniMERNetFormulaImagePreprocessor(),
            pipeline: try makeConvertedPipeline()
        )

        let result = try await recognizer.recognize(loadFormulaImage())

        XCTAssertEqual(result.decoderSteps, 133)
        XCTAssertEqual(
            result.latex,
            "= - \\! \\! \\int _ { 0 } ^ { x } \\! \\frac { 1 - t - 1 } { 1 - t } d t = - \\! \\! \\int _ { 0 } ^ { x } ( 1 - \\frac { 1 } { 1 - t } ) d t = - \\ln ( 1 - x ) - x"
        )
    }

    func testConvertedModelsDecodePythonPreprocessingFixture() async throws {
        let input = try loadFloat32NPY(
            repositoryRoot().appendingPathComponent(
                "EqnSnapTests/Fixtures/UniMERNet/preprocessor-input.npy"
            ),
            shape: [1, 3, 192, 672]
        )

        let result = try await makeConvertedPipeline().recognize(
            pixelValues: input
        )

        XCTAssertEqual(result.decoderSteps, 166)
        XCTAssertEqual(
            result.latex,
            "= - \\! \\! \\int _ { 0 } ^ { x } \\! \\frac { 1 \\! - \\! t - 1 } { 1 \\! - \\! t } d t = - \\! \\! \\int _ { 0 } ^ { x } ( 1 \\! - \\! \\frac { 1 } { 1 \\! - \\! t } ) d t = - \\! \\ln ( 1 \\! - \\! x ) - x"
        )
    }

    private func makeDecoderConfiguration()
        -> UniMERNetCachedDecoderConfiguration
    {
        UniMERNetCachedDecoderConfiguration(
            maximumTokenLength: 4,
            contextLength: 3,
            embeddingDimension: 4,
            decoderOutputSize: 50_000,
            tokenizerVocabularySize: 50_000,
            decoderLayers: 1,
            attentionHeads: 1,
            keyDimension: 1,
            valueDimension: 1,
            bosTokenID: 0,
            eosTokenID: 2,
            padTokenID: 1,
            maximumRepeatedPatternLength: 2,
            minimumPatternRepetitions: 3
        )
    }

    private func markedContext(marker: Float32) throws -> MLMultiArray {
        let context = try MLMultiArray(shape: [1, 3, 4], dataType: .float32)
        context.dataPointer.assumingMemoryBound(to: Float32.self)[0] = marker
        return context
    }

    private func loadTokenizer() throws -> UniMERNetTokenizer {
        try UniMERNetTokenizer(
            contentsOf: repositoryRoot().appendingPathComponent(
                "EqnSnap/Resources/Models/UniMERNet/UniMERNetTokenizer.json"
            )
        )
    }

    private func loadModel(at packageURL: URL) throws -> MLModel {
        let compiledURL = try MLModel.compileModel(at: packageURL)
        addTeardownBlock { try? FileManager.default.removeItem(at: compiledURL) }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        return try MLModel(contentsOf: compiledURL, configuration: configuration)
    }

    private func makeConvertedPipeline() throws
        -> UniMERNetRecognitionPipeline
    {
        let artifacts = repositoryRoot().appendingPathComponent(
            "Tools/model-conversion/artifacts"
        )
        let encoderURL = artifacts.appendingPathComponent(
            "UniMERNetTinyEncoder-FP16.mlpackage"
        )
        let decoderURL = artifacts.appendingPathComponent(
            "UniMERNetTinyDecoder-CachedStep-SelfKV-FP16.mlpackage"
        )
        guard FileManager.default.fileExists(atPath: encoderURL.path),
              FileManager.default.fileExists(atPath: decoderURL.path)
        else {
            throw XCTSkip("Converted UniMERNet models are not available")
        }
        let encoder = try loadModel(at: encoderURL)
        let decoder = try loadModel(at: decoderURL)
        return UniMERNetRecognitionPipeline(
            encoderRunner: UniMERNetEncoderRunner(
                predictor: CoreMLUniMERNetEncoderModel(model: encoder)
            ),
            decoderRunner: UniMERNetCachedDecoderRunner(
                predictor: CoreMLUniMERNetCachedDecoderModel(model: decoder)
            ),
            tokenizer: try loadTokenizer()
        )
    }

    private func loadFormulaImage() throws -> CGImage {
        let url = repositoryRoot().appendingPathComponent(
            "Tools/model-conversion/test_images/formula.png"
        )
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw UniMERNetPipelineFixtureError.unreadableImage
        }
        return image
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
            throw UniMERNetPipelineFixtureError.unsupportedNPYHeader
        }
        let headerLength = Int(data[8]) | (Int(data[9]) << 8)
        let dataOffset = 10 + headerLength
        let byteCount = shape.reduce(1, *) * MemoryLayout<Float32>.size
        guard dataOffset + byteCount == data.count else {
            throw UniMERNetPipelineFixtureError.invalidNPYDataLength
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

    private func repositoryRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}

private enum UniMERNetPipelineFixtureError: Error {
    case unreadableImage
    case unsupportedNPYHeader
    case invalidNPYDataLength
}

private final class PipelineUniMERNetEncoderPredictor:
    UniMERNetEncoderPredicting
{
    private let output: MLMultiArray

    init(output: MLMultiArray) {
        self.output = output
    }

    func predict(pixelValues: MLMultiArray) throws -> MLMultiArray { output }
}

private final class PipelineUniMERNetDecoderPredictor:
    UniMERNetCachedDecoderPredicting
{
    private let expectedMarker: Float32
    private let tokens: [Int32]
    private let configuration: UniMERNetCachedDecoderConfiguration
    private var callIndex = 0
    private(set) var observedExpectedMarker = false

    init(
        expectedMarker: Float32,
        tokens: [Int32],
        configuration: UniMERNetCachedDecoderConfiguration
    ) {
        self.expectedMarker = expectedMarker
        self.tokens = tokens
        self.configuration = configuration
    }

    func predict(
        inputID: MLMultiArray,
        encoderContext: MLMultiArray,
        selfCache: [MLMultiArray]
    ) throws -> UniMERNetDecoderPrediction {
        let context = encoderContext.dataPointer.assumingMemoryBound(
            to: Float32.self
        )
        observedExpectedMarker = context[0] == expectedMarker

        let logits = try MLMultiArray(
            shape: [1, 1, NSNumber(value: configuration.decoderOutputSize)],
            dataType: .float32
        )
        let logitsPointer = logits.dataPointer.assumingMemoryBound(to: Float32.self)
        for index in 0..<logits.count { logitsPointer[index] = -1_000 }
        logitsPointer[Int(tokens[callIndex])] = 1_000

        let nextLength = selfCache[0].shape[2].intValue + 1
        var nextCache: [MLMultiArray] = []
        for _ in 0..<configuration.decoderLayers {
            nextCache.append(try makeCache(length: nextLength))
            nextCache.append(try makeCache(length: nextLength))
        }
        callIndex += 1
        return UniMERNetDecoderPrediction(logits: logits, selfCache: nextCache)
    }

    private func makeCache(length: Int) throws -> MLMultiArray {
        try MLMultiArray(
            shape: [1, 1, NSNumber(value: length), 1],
            dataType: .float32
        )
    }
}
