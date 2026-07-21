import CoreML
import Foundation
import XCTest
@testable import EqnSnap

final class Pix2TexDecoderRunnerTests: XCTestCase {
    func testStopsAtEOSAndBuildsContiguousMasks() async throws {
        let predictor = ScriptedDecoderPredictor(tokens: [10, 11, 2])
        let runner = Pix2TexDecoderRunner(predictor: predictor)

        let result = try await runner.decode(
            encoderContext: try makeContext(length: 29)
        )

        XCTAssert(result.tokenIDs == [10, 11])
        XCTAssert(result.decoderSteps == 3)
        XCTAssert(predictor.observedTokenLengths == [1, 2, 3])
        XCTAssert(predictor.observedContextLengths == [29, 29, 29])
        XCTAssert(predictor.observedPrefixes == [[1], [1, 10], [1, 10, 11]])
    }

    func testThrowsWhenMaximumTokenLengthIsReached() async throws {
        let predictor = ScriptedDecoderPredictor(tokens: [10, 11, 12])
        let runner = Pix2TexDecoderRunner(
            predictor: predictor,
            configuration: configuration(
                maximumTokenLength: 4,
                minimumPatternRepetitions: 6
            )
        )

        do {
            _ = try await runner.decode(
                encoderContext: try makeContext(length: 5)
            )
            XCTFail("Expected maximum token length failure")
        } catch let error as Pix2TexDecoderError {
            XCTAssert(error == .maximumTokenLengthReached)
        }
        XCTAssert(predictor.observedTokenLengths == [1, 2, 3])
    }

    func testDetectsRepeatedSuffixPattern() async throws {
        let predictor = ScriptedDecoderPredictor(tokens: [10, 10, 10])
        let runner = Pix2TexDecoderRunner(
            predictor: predictor,
            configuration: configuration(
                maximumTokenLength: 16,
                minimumPatternRepetitions: 3
            )
        )

        do {
            _ = try await runner.decode(
                encoderContext: try makeContext(length: 9)
            )
            XCTFail("Expected repetition failure")
        } catch let error as Pix2TexDecoderError {
            XCTAssert(error == .repetitionDetected([10]))
        }
    }

    func testHonorsTaskCancellationBeforePrediction() async throws {
        let predictor = ScriptedDecoderPredictor(tokens: [2])
        let runner = Pix2TexDecoderRunner(predictor: predictor)
        let context = try makeContext(length: 5)
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
            XCTAssert(predictor.observedTokenLengths.isEmpty)
        }
    }

    private func configuration(
        maximumTokenLength: Int,
        minimumPatternRepetitions: Int
    ) -> Pix2TexDecoderConfiguration {
        Pix2TexDecoderConfiguration(
            maximumTokenLength: maximumTokenLength,
            maximumContextLength: 169,
            embeddingDimension: 256,
            decoderOutputSize: 8_000,
            tokenizerVocabularySize: 1_175,
            bosTokenID: 1,
            eosTokenID: 2,
            padTokenID: 0,
            maximumRepeatedPatternLength: 8,
            minimumPatternRepetitions: minimumPatternRepetitions
        )
    }

    private func makeContext(length: Int) throws -> MLMultiArray {
        let context = try MLMultiArray(
            shape: [1, NSNumber(value: length), 256],
            dataType: .float32
        )
        let pointer = context.dataPointer.assumingMemoryBound(to: Float32.self)
        for index in 0..<context.count {
            pointer[index] = Float32(index) / 100
        }
        return context
    }
}

private final class ScriptedDecoderPredictor: Pix2TexDecoderPredicting {
    private let tokens: [Int32]
    private var callIndex = 0

    private(set) var observedTokenLengths: [Int] = []
    private(set) var observedContextLengths: [Int] = []
    private(set) var observedPrefixes: [[Int32]] = []

    init(tokens: [Int32]) {
        self.tokens = tokens
    }

    func predict(
        inputIDs: MLMultiArray,
        tokenMask: MLMultiArray,
        encoderContext: MLMultiArray,
        contextMask: MLMultiArray
    ) throws -> MLMultiArray {
        let tokenLength = sum(mask: tokenMask)
        let contextLength = sum(mask: contextMask)
        observedTokenLengths.append(tokenLength)
        observedContextLengths.append(contextLength)

        let inputPointer = inputIDs.dataPointer.assumingMemoryBound(
            to: Int32.self
        )
        let inputStride = inputIDs.strides[1].intValue
        observedPrefixes.append(
            (0..<tokenLength).map { inputPointer[$0 * inputStride] }
        )

        let logits = try MLMultiArray(
            shape: [1, 1, 8_000],
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
        callIndex += 1
        return logits
    }

    private func sum(mask: MLMultiArray) -> Int {
        let pointer = mask.dataPointer.assumingMemoryBound(to: Int32.self)
        let stride = mask.strides[1].intValue
        return (0..<mask.shape[1].intValue).reduce(into: 0) {
            $0 += Int(pointer[$1 * stride])
        }
    }
}
