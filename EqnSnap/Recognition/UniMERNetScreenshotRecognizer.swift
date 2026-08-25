import CoreGraphics

final class UniMERNetScreenshotRecognizer {
    private let preprocessor: UniMERNetFormulaImagePreprocessor
    private let pipeline: UniMERNetRecognitionPipeline

    init(
        preprocessor: UniMERNetFormulaImagePreprocessor,
        pipeline: UniMERNetRecognitionPipeline
    ) {
        self.preprocessor = preprocessor
        self.pipeline = pipeline
    }

    func recognize(_ screenshot: CGImage) async throws
        -> UniMERNetRecognitionOutput
    {
        try Task.checkCancellation()
        let prepared = try preprocessor.prepare(screenshot)
        try Task.checkCancellation()
        return try await pipeline.recognize(pixelValues: prepared.tensor)
    }
}
