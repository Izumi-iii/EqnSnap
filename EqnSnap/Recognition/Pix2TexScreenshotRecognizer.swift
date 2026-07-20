import CoreGraphics
import CoreML
import Foundation

nonisolated final class Pix2TexScreenshotRecognizer {
    private let preprocessor: Pix2TexFormulaImagePreprocessor
    private let pipeline: Pix2TexRecognitionPipeline

    init(
        preprocessor: Pix2TexFormulaImagePreprocessor,
        pipeline: Pix2TexRecognitionPipeline
    ) {
        self.preprocessor = preprocessor
        self.pipeline = pipeline
    }

    convenience init(
        bundle: Bundle = .main,
        computeUnits: MLComputeUnits = .all
    ) throws {
        let models = try Pix2TexModelBundleLoader(
            computeUnits: computeUnits
        ).load(from: bundle)
        self.init(
            preprocessor: Pix2TexFormulaImagePreprocessor(),
            pipeline: models.makeRecognitionPipeline()
        )
    }

    func recognize(_ screenshot: CGImage) async throws
        -> Pix2TexRecognitionOutput
    {
        try Task.checkCancellation()
        let prepared = try preprocessor.prepare(screenshot)
        try Task.checkCancellation()
        return try await pipeline.recognize(pixelValues: prepared.tensor)
    }
}
