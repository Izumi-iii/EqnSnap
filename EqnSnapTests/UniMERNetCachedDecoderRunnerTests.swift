import CoreML
import XCTest
@testable import EqnSnap

final class UniMERNetCachedDecoderRunnerTests: XCTestCase {
    func testConvertedCoreMLModelAcceptsMaskedDummyCache() throws {
        let packageURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Tools/model-conversion/artifacts")
            .appendingPathComponent(
                "UniMERNetTinyDecoder-CachedStep-SelfKV-FP16.mlpackage"
            )
        guard FileManager.default.fileExists(atPath: packageURL.path) else {
            throw XCTSkip("Converted UniMERNet Decoder is not available")
        }

        let compiledURL = try MLModel.compileModel(at: packageURL)
        defer { try? FileManager.default.removeItem(at: compiledURL) }
        let modelConfiguration = MLModelConfiguration()
        modelConfiguration.computeUnits = .cpuOnly
        let predictor = CoreMLUniMERNetCachedDecoderModel(
            model: try MLModel(
                contentsOf: compiledURL,
                configuration: modelConfiguration
            )
        )
        let inputID = try MLMultiArray(shape: [1, 1], dataType: .int32)
        inputID[0] = NSNumber(value: Int32(0))
        let context = try MLMultiArray(
            shape: [1, 126, 512],
            dataType: .float32
        )
        var dummyCache: [MLMultiArray] = []
        for _ in 0..<8 {
            dummyCache.append(
                try MLMultiArray(
                    shape: [1, 16, 1, 16],
                    dataType: .float32
                )
            )
            dummyCache.append(
                try MLMultiArray(
                    shape: [1, 16, 1, 32],
                    dataType: .float32
                )
            )
        }

        let prediction = try predictor.predict(
            inputID: inputID,
            encoderContext: context,
            selfCache: dummyCache
        )

        XCTAssert(prediction.logits.shape.map(\.intValue) == [1, 1, 50_000])
        XCTAssert(prediction.selfCache.count == 16)
        XCTAssert(
            prediction.selfCache[0].shape.map(\.intValue) == [1, 16, 2, 16]
        )
        XCTAssert(
            prediction.selfCache[1].shape.map(\.intValue) == [1, 16, 2, 32]
        )
    }

    func testStartsWithMaskedDummyCacheAndStopsAtEOS() async throws {
        let predictor = ScriptedUniMERNetDecoderPredictor(
            tokens: [10, 11, 2],
            configuration: configuration()
        )
        let runner = UniMERNetCachedDecoderRunner(
            predictor: predictor,
            configuration: configuration()
        )

        let result = try await runner.decode(encoderContext: makeContext())

        XCTAssert(result.tokenIDs == [10, 11])
        XCTAssert(result.decoderSteps == 3)
        XCTAssert(predictor.observedInputIDs == [0, 10, 11])
        XCTAssert(predictor.observedPastLengths == [1, 2, 3])
    }

    func testThrowsWhenMaximumTokenLengthIsReached() async throws {
        let config = configuration(maximumTokenLength: 3)
        let predictor = ScriptedUniMERNetDecoderPredictor(
            tokens: [10, 11, 12],
            configuration: config
        )
        let runner = UniMERNetCachedDecoderRunner(
            predictor: predictor,
            configuration: config
        )

        do {
            _ = try await runner.decode(encoderContext: makeContext())
            XCTFail("Expected maximum token length failure")
        } catch let error as UniMERNetCachedDecoderError {
            XCTAssert(error == .maximumTokenLengthReached)
        }
        XCTAssert(predictor.observedPastLengths == [1, 2, 3])
    }

    func testDetectsRepeatedSuffixPattern() async throws {
        let config = configuration(minimumPatternRepetitions: 3)
        let predictor = ScriptedUniMERNetDecoderPredictor(
            tokens: [10, 10, 10],
            configuration: config
        )
        let runner = UniMERNetCachedDecoderRunner(
            predictor: predictor,
            configuration: config
        )

        do {
            _ = try await runner.decode(encoderContext: makeContext())
            XCTFail("Expected repetition failure")
        } catch let error as UniMERNetCachedDecoderError {
            XCTAssert(error == .repetitionDetected([10]))
        }
    }

    func testRejectsUnexpectedCacheGrowth() async throws {
        let config = configuration()
        let predictor = ScriptedUniMERNetDecoderPredictor(
            tokens: [10],
            configuration: config,
            outputLengthOffset: 1
        )
        let runner = UniMERNetCachedDecoderRunner(
            predictor: predictor,
            configuration: config
        )

        do {
            _ = try await runner.decode(encoderContext: makeContext())
            XCTFail("Expected invalid Cache shape")
        } catch let error as UniMERNetCachedDecoderError {
            XCTAssert(
                error == .invalidCacheShape(
                    index: 0,
                    expected: [1, 2, 2, 2],
                    actual: [1, 2, 3, 2]
                )
            )
        }
    }

    func testHonorsTaskCancellationBeforePrediction() async throws {
        let config = configuration()
        let predictor = ScriptedUniMERNetDecoderPredictor(
            tokens: [2],
            configuration: config
        )
        let runner = UniMERNetCachedDecoderRunner(
            predictor: predictor,
            configuration: config
        )
        let context = try makeContext()
        let task = Task {
            withUnsafeCurrentTask { currentTask in
                currentTask?.cancel()
            }
            return try await runner.decode(encoderContext: context)
        }

        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch is CancellationError {
            XCTAssert(predictor.observedInputIDs.isEmpty)
        }
    }

    private func configuration(
        maximumTokenLength: Int = 8,
        minimumPatternRepetitions: Int = 6
    ) -> UniMERNetCachedDecoderConfiguration {
        UniMERNetCachedDecoderConfiguration(
            maximumTokenLength: maximumTokenLength,
            contextLength: 3,
            embeddingDimension: 4,
            decoderOutputSize: 64,
            tokenizerVocabularySize: 64,
            decoderLayers: 2,
            attentionHeads: 2,
            keyDimension: 2,
            valueDimension: 3,
            bosTokenID: 0,
            eosTokenID: 2,
            padTokenID: 1,
            maximumRepeatedPatternLength: 4,
            minimumPatternRepetitions: minimumPatternRepetitions
        )
    }

    private func makeContext() throws -> MLMultiArray {
        try MLMultiArray(shape: [1, 3, 4], dataType: .float32)
    }
}

private final class ScriptedUniMERNetDecoderPredictor:
    UniMERNetCachedDecoderPredicting
{
    private let tokens: [Int32]
    private let configuration: UniMERNetCachedDecoderConfiguration
    private let outputLengthOffset: Int
    private var callIndex = 0

    private(set) var observedInputIDs: [Int32] = []
    private(set) var observedPastLengths: [Int] = []

    init(
        tokens: [Int32],
        configuration: UniMERNetCachedDecoderConfiguration,
        outputLengthOffset: Int = 0
    ) {
        self.tokens = tokens
        self.configuration = configuration
        self.outputLengthOffset = outputLengthOffset
    }

    func predict(
        inputID: MLMultiArray,
        encoderContext: MLMultiArray,
        selfCache: [MLMultiArray]
    ) throws -> UniMERNetDecoderPrediction {
        let inputPointer = inputID.dataPointer.assumingMemoryBound(
            to: Int32.self
        )
        observedInputIDs.append(inputPointer[0])
        let pastLength = selfCache[0].shape[2].intValue
        observedPastLengths.append(pastLength)

        let logits = try MLMultiArray(
            shape: [1, 1, NSNumber(value: configuration.decoderOutputSize)],
            dataType: .float32
        )
        let logitsPointer = logits.dataPointer.assumingMemoryBound(
            to: Float32.self
        )
        for index in 0..<logits.count {
            logitsPointer[index] = -1_000
        }
        let token = tokens[min(callIndex, tokens.count - 1)]
        logitsPointer[Int(token) * logits.strides[2].intValue] = 1_000

        let outputLength = pastLength + 1 + outputLengthOffset
        var nextCache: [MLMultiArray] = []
        for _ in 0..<configuration.decoderLayers {
            nextCache.append(
                try makeCache(length: outputLength, dimension: configuration.keyDimension)
            )
            nextCache.append(
                try makeCache(length: outputLength, dimension: configuration.valueDimension)
            )
        }
        callIndex += 1
        return UniMERNetDecoderPrediction(
            logits: logits,
            selfCache: nextCache
        )
    }

    private func makeCache(length: Int, dimension: Int) throws
        -> MLMultiArray
    {
        try MLMultiArray(
            shape: [
                1,
                NSNumber(value: configuration.attentionHeads),
                NSNumber(value: length),
                NSNumber(value: dimension),
            ],
            dataType: .float32
        )
    }
}
