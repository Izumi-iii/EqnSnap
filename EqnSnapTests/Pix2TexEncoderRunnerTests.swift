import CoreML
import Testing
@testable import EqnSnap

struct Pix2TexEncoderRunnerTests {
    @Test func returnsDynamicEncoderContext() async throws {
        let predictor = RecordingEncoderPredictor()
        let runner = Pix2TexEncoderRunner(predictor: predictor)
        let input = try makeInput(height: 32, width: 224)

        let context = try await runner.encode(input)

        #expect(context.shape.map(\.intValue) == [1, 29, 256])
        #expect(predictor.observedShapes == [[1, 1, 32, 224]])
    }

    @Test func rejectsUnsupportedInputShapeBeforePrediction() async throws {
        let predictor = RecordingEncoderPredictor()
        let runner = Pix2TexEncoderRunner(predictor: predictor)

        do {
            _ = try await runner.encode(
                makeInput(height: 96, width: 224)
            )
            Issue.record("Expected invalid input shape")
        } catch let error as Pix2TexEncoderError {
            #expect(error == .invalidInputShape([1, 1, 96, 224]))
        }
        #expect(predictor.observedShapes.isEmpty)
    }

    @Test func rejectsUnexpectedContextShape() async throws {
        let predictor = RecordingEncoderPredictor(contextLengthOverride: 30)
        let runner = Pix2TexEncoderRunner(predictor: predictor)

        do {
            _ = try await runner.encode(
                makeInput(height: 32, width: 224)
            )
            Issue.record("Expected invalid context shape")
        } catch let error as Pix2TexEncoderError {
            #expect(
                error == .invalidContextShape(
                    expected: [1, 29, 256],
                    actual: [1, 30, 256]
                )
            )
        }
    }

    @Test func honorsTaskCancellationBeforePrediction() async throws {
        let predictor = RecordingEncoderPredictor()
        let runner = Pix2TexEncoderRunner(predictor: predictor)
        let input = try makeInput(height: 32, width: 224)
        let task = Task {
            withUnsafeCurrentTask { currentTask in
                currentTask?.cancel()
            }
            return try await runner.encode(input)
        }

        do {
            _ = try await task.value
            Issue.record("Expected cancellation")
        } catch is CancellationError {
            #expect(predictor.observedShapes.isEmpty)
        }
    }

    private func makeInput(height: Int, width: Int) throws -> MLMultiArray {
        try MLMultiArray(
            shape: [1, 1, NSNumber(value: height), NSNumber(value: width)],
            dataType: .float32
        )
    }
}

nonisolated private final class RecordingEncoderPredictor:
    Pix2TexEncoderPredicting
{
    private let contextLengthOverride: Int?
    private(set) var observedShapes: [[Int]] = []

    init(contextLengthOverride: Int? = nil) {
        self.contextLengthOverride = contextLengthOverride
    }

    func predict(pixelValues: MLMultiArray) throws -> MLMultiArray {
        let shape = pixelValues.shape.map(\.intValue)
        observedShapes.append(shape)
        let contextLength = contextLengthOverride
            ?? (shape[2] / 16) * (shape[3] / 16) + 1
        return try MLMultiArray(
            shape: [1, NSNumber(value: contextLength), 256],
            dataType: .float32
        )
    }
}
