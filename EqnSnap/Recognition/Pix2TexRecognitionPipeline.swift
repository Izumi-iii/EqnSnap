import CoreML

nonisolated struct Pix2TexRecognitionOutput: Sendable, Equatable {
    let latex: String
    let tokenIDs: [Int32]
    let decoderSteps: Int
}

nonisolated final class Pix2TexRecognitionPipeline {
    private let encoderRunner: Pix2TexEncoderRunner
    private let decoderRunner: Pix2TexDecoderRunner
    private let tokenizer: Pix2TexTokenizer

    init(
        encoderRunner: Pix2TexEncoderRunner,
        decoderRunner: Pix2TexDecoderRunner,
        tokenizer: Pix2TexTokenizer
    ) {
        self.encoderRunner = encoderRunner
        self.decoderRunner = decoderRunner
        self.tokenizer = tokenizer
    }

    func recognize(pixelValues: MLMultiArray) async throws
        -> Pix2TexRecognitionOutput
    {
        try Task.checkCancellation()
        let context = try await encoderRunner.encode(pixelValues)
        let sequence = try await decoderRunner.decode(
            encoderContext: context
        )
        try Task.checkCancellation()
        let latex = try tokenizer.decode(sequence.tokenIDs)
        return Pix2TexRecognitionOutput(
            latex: latex,
            tokenIDs: sequence.tokenIDs,
            decoderSteps: sequence.decoderSteps
        )
    }
}
