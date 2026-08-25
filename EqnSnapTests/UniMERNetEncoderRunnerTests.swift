import CoreML
import XCTest
@testable import EqnSnap

final class UniMERNetEncoderRunnerTests: XCTestCase {
    func testReturnsExpectedContext() async throws {
        let predictor = StubUniMERNetEncoderPredictor(
            output: try makeArray(shape: [1, 126, 512])
        )
        let runner = UniMERNetEncoderRunner(predictor: predictor)

        let context = try await runner.encode(
            makeArray(shape: [1, 3, 192, 672])
        )

        XCTAssert(context.shape.map(\.intValue) == [1, 126, 512])
        XCTAssert(predictor.predictionCount == 1)
    }

    func testRejectsUnexpectedInputShape() async throws {
        let predictor = StubUniMERNetEncoderPredictor(
            output: try makeArray(shape: [1, 126, 512])
        )
        let runner = UniMERNetEncoderRunner(predictor: predictor)

        do {
            _ = try await runner.encode(makeArray(shape: [1, 3, 192, 640]))
            XCTFail("Expected invalid input shape")
        } catch let error as UniMERNetEncoderError {
            XCTAssert(error == .invalidInputShape([1, 3, 192, 640]))
        }
        XCTAssert(predictor.predictionCount == 0)
    }

    func testRejectsUnexpectedContextShape() async throws {
        let predictor = StubUniMERNetEncoderPredictor(
            output: try makeArray(shape: [1, 125, 512])
        )
        let runner = UniMERNetEncoderRunner(predictor: predictor)

        do {
            _ = try await runner.encode(makeArray(shape: [1, 3, 192, 672]))
            XCTFail("Expected invalid context shape")
        } catch let error as UniMERNetEncoderError {
            XCTAssert(error == .invalidContextShape([1, 125, 512]))
        }
    }

    func testHonorsCancellationBeforePrediction() async throws {
        let predictor = StubUniMERNetEncoderPredictor(
            output: try makeArray(shape: [1, 126, 512])
        )
        let runner = UniMERNetEncoderRunner(predictor: predictor)
        let input = try makeArray(shape: [1, 3, 192, 672])
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await runner.encode(input)
        }

        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch is CancellationError {
            XCTAssert(predictor.predictionCount == 0)
        }
    }

    private func makeArray(shape: [Int]) throws -> MLMultiArray {
        try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float32)
    }
}

private final class StubUniMERNetEncoderPredictor: UniMERNetEncoderPredicting {
    private let output: MLMultiArray
    private(set) var predictionCount = 0

    init(output: MLMultiArray) {
        self.output = output
    }

    func predict(pixelValues: MLMultiArray) throws -> MLMultiArray {
        predictionCount += 1
        return output
    }
}
