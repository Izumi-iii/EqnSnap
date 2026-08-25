import CoreML

struct UniMERNetRecognitionOutput: Sendable, Equatable {
    let latex: String
    let tokenIDs: [Int32]
    let decoderSteps: Int
}

final class UniMERNetRecognitionPipeline {
    private let encoderRunner: UniMERNetEncoderRunner
    private let decoderRunner: UniMERNetCachedDecoderRunner
    private let tokenizer: UniMERNetTokenizer

    init(
        encoderRunner: UniMERNetEncoderRunner,
        decoderRunner: UniMERNetCachedDecoderRunner,
        tokenizer: UniMERNetTokenizer
    ) {
        self.encoderRunner = encoderRunner
        self.decoderRunner = decoderRunner
        self.tokenizer = tokenizer
    }

    func recognize(pixelValues: MLMultiArray) async throws
        -> UniMERNetRecognitionOutput
    {
        try Task.checkCancellation()
        let context = try await encoderRunner.encode(pixelValues)
        let sequence = try await decoderRunner.decode(
            encoderContext: context
        )
        try Task.checkCancellation()
        return UniMERNetRecognitionOutput(
            latex: try tokenizer.decode(sequence.tokenIDs),
            tokenIDs: sequence.tokenIDs,
            decoderSteps: sequence.decoderSteps
        )
    }
}
