import CoreML
import Foundation

nonisolated protocol Pix2TexDecoderPredicting: AnyObject {
    func predict(
        inputIDs: MLMultiArray,
        tokenMask: MLMultiArray,
        encoderContext: MLMultiArray,
        contextMask: MLMultiArray
    ) throws -> MLMultiArray
}

nonisolated struct Pix2TexDecoderFeatureNames: Sendable, Equatable {
    static let standard = Pix2TexDecoderFeatureNames(
        inputIDs: "input_ids",
        tokenMask: "token_mask",
        encoderContext: "encoder_context",
        contextMask: "context_mask",
        logits: "logits"
    )

    let inputIDs: String
    let tokenMask: String
    let encoderContext: String
    let contextMask: String
    let logits: String
}

nonisolated final class CoreMLPix2TexDecoderModel: Pix2TexDecoderPredicting {
    private let model: MLModel
    private let names: Pix2TexDecoderFeatureNames

    init(
        model: MLModel,
        names: Pix2TexDecoderFeatureNames = .standard
    ) {
        self.model = model
        self.names = names
    }

    func predict(
        inputIDs: MLMultiArray,
        tokenMask: MLMultiArray,
        encoderContext: MLMultiArray,
        contextMask: MLMultiArray
    ) throws -> MLMultiArray {
        let input = try MLDictionaryFeatureProvider(
            dictionary: [
                names.inputIDs: MLFeatureValue(multiArray: inputIDs),
                names.tokenMask: MLFeatureValue(multiArray: tokenMask),
                names.encoderContext: MLFeatureValue(
                    multiArray: encoderContext
                ),
                names.contextMask: MLFeatureValue(multiArray: contextMask),
            ]
        )
        let output = try model.prediction(from: input)
        guard let logits = output.featureValue(
            for: names.logits
        )?.multiArrayValue else {
            throw Pix2TexDecoderError.missingLogitsOutput(names.logits)
        }
        return logits
    }
}

nonisolated struct Pix2TexDecoderConfiguration: Sendable, Equatable {
    static let pix2texV1 = Pix2TexDecoderConfiguration(
        maximumTokenLength: 128,
        maximumContextLength: 169,
        embeddingDimension: 256,
        decoderOutputSize: 8_000,
        tokenizerVocabularySize: 1_175,
        bosTokenID: 1,
        eosTokenID: 2,
        padTokenID: 0,
        maximumRepeatedPatternLength: 8,
        minimumPatternRepetitions: 6
    )

    let maximumTokenLength: Int
    let maximumContextLength: Int
    let embeddingDimension: Int
    let decoderOutputSize: Int
    let tokenizerVocabularySize: Int
    let bosTokenID: Int32
    let eosTokenID: Int32
    let padTokenID: Int32
    let maximumRepeatedPatternLength: Int
    let minimumPatternRepetitions: Int
}

nonisolated struct Pix2TexDecodedSequence: Sendable, Equatable {
    let tokenIDs: [Int32]
    let decoderSteps: Int
}

nonisolated enum Pix2TexDecoderError: Error, Sendable, Equatable {
    case invalidEncoderContextShape([Int])
    case invalidEncoderContextDataType
    case invalidLogitsShape([Int])
    case invalidLogitsDataType
    case nonFiniteLogits
    case missingLogitsOutput(String)
    case invalidGeneratedToken(Int32)
    case maximumTokenLengthReached
    case repetitionDetected([Int32])
}

nonisolated final class Pix2TexDecoderRunner {
    private let predictor: Pix2TexDecoderPredicting
    private let configuration: Pix2TexDecoderConfiguration

    init(
        predictor: Pix2TexDecoderPredicting,
        configuration: Pix2TexDecoderConfiguration = .pix2texV1
    ) {
        self.predictor = predictor
        self.configuration = configuration
    }

    func decode(encoderContext: MLMultiArray) async throws
        -> Pix2TexDecodedSequence
    {
        try Task.checkCancellation()
        let preparedContext = try prepareContext(encoderContext)
        let tokenInputs = try makeTokenInputs()
        let inputIDs = tokenInputs.inputIDs
        let tokenMask = tokenInputs.tokenMask
        let inputIDPointer = inputIDs.dataPointer.assumingMemoryBound(
            to: Int32.self
        )
        let tokenMaskPointer = tokenMask.dataPointer.assumingMemoryBound(
            to: Int32.self
        )
        let inputStride = inputIDs.strides[1].intValue
        let maskStride = tokenMask.strides[1].intValue

        inputIDPointer[0] = configuration.bosTokenID
        tokenMaskPointer[0] = 1

        var generatedTokens: [Int32] = []
        var prefixLength = 1
        var decoderSteps = 0

        while prefixLength < configuration.maximumTokenLength {
            try Task.checkCancellation()
            let logits = try predictor.predict(
                inputIDs: inputIDs,
                tokenMask: tokenMask,
                encoderContext: preparedContext.context,
                contextMask: preparedContext.mask
            )
            try Task.checkCancellation()
            decoderSteps += 1

            let nextToken = try argmax(logits)
            if nextToken == configuration.eosTokenID {
                return Pix2TexDecodedSequence(
                    tokenIDs: generatedTokens,
                    decoderSteps: decoderSteps
                )
            }
            guard nextToken != configuration.padTokenID,
                  nextToken != configuration.bosTokenID,
                  nextToken >= 0,
                  Int(nextToken) < configuration.tokenizerVocabularySize
            else {
                throw Pix2TexDecoderError.invalidGeneratedToken(nextToken)
            }

            generatedTokens.append(nextToken)
            if let pattern = repeatedSuffix(in: generatedTokens) {
                throw Pix2TexDecoderError.repetitionDetected(pattern)
            }

            inputIDPointer[prefixLength * inputStride] = nextToken
            tokenMaskPointer[prefixLength * maskStride] = 1
            prefixLength += 1
        }

        throw Pix2TexDecoderError.maximumTokenLengthReached
    }

    private func prepareContext(_ source: MLMultiArray) throws
        -> (context: MLMultiArray, mask: MLMultiArray)
    {
        let shape = source.shape.map(\.intValue)
        guard shape.count == 3,
              shape[0] == 1,
              shape[1] > 0,
              shape[1] <= configuration.maximumContextLength,
              shape[2] == configuration.embeddingDimension
        else {
            throw Pix2TexDecoderError.invalidEncoderContextShape(shape)
        }
        guard source.dataType == .float32 else {
            throw Pix2TexDecoderError.invalidEncoderContextDataType
        }

        let context = try MLMultiArray(
            shape: [
                1,
                NSNumber(value: configuration.maximumContextLength),
                NSNumber(value: configuration.embeddingDimension),
            ],
            dataType: .float32
        )
        let mask = try MLMultiArray(
            shape: [
                1,
                NSNumber(value: configuration.maximumContextLength),
            ],
            dataType: .int32
        )
        let sourcePointer = source.dataPointer.assumingMemoryBound(
            to: Float32.self
        )
        let destinationPointer = context.dataPointer.assumingMemoryBound(
            to: Float32.self
        )
        let maskPointer = mask.dataPointer.assumingMemoryBound(
            to: Int32.self
        )
        for index in 0..<context.count {
            destinationPointer[index] = 0
        }
        for index in 0..<mask.count {
            maskPointer[index] = 0
        }

        let sourceRowStride = source.strides[1].intValue
        let sourceColumnStride = source.strides[2].intValue
        let destinationRowStride = context.strides[1].intValue
        let destinationColumnStride = context.strides[2].intValue
        let maskRowStride = mask.strides[1].intValue
        for row in 0..<shape[1] {
            maskPointer[row * maskRowStride] = 1
            for column in 0..<configuration.embeddingDimension {
                destinationPointer[
                    row * destinationRowStride
                        + column * destinationColumnStride
                ] = sourcePointer[
                    row * sourceRowStride
                        + column * sourceColumnStride
                ]
            }
        }
        return (context, mask)
    }

    private func makeTokenInputs() throws
        -> (inputIDs: MLMultiArray, tokenMask: MLMultiArray)
    {
        let inputIDs = try MLMultiArray(
            shape: [1, NSNumber(value: configuration.maximumTokenLength)],
            dataType: .int32
        )
        let tokenMask = try MLMultiArray(
            shape: [1, NSNumber(value: configuration.maximumTokenLength)],
            dataType: .int32
        )
        let inputPointer = inputIDs.dataPointer.assumingMemoryBound(
            to: Int32.self
        )
        let maskPointer = tokenMask.dataPointer.assumingMemoryBound(
            to: Int32.self
        )
        for index in 0..<inputIDs.count {
            inputPointer[index] = configuration.padTokenID
            maskPointer[index] = 0
        }
        return (inputIDs, tokenMask)
    }

    private func argmax(_ logits: MLMultiArray) throws -> Int32 {
        let shape = logits.shape.map(\.intValue)
        guard shape == [1, 1, configuration.decoderOutputSize] else {
            throw Pix2TexDecoderError.invalidLogitsShape(shape)
        }
        guard logits.dataType == .float32 else {
            throw Pix2TexDecoderError.invalidLogitsDataType
        }

        let pointer = logits.dataPointer.assumingMemoryBound(to: Float32.self)
        let tokenStride = logits.strides[2].intValue
        var bestToken = 0
        var bestValue = -Float.greatestFiniteMagnitude
        for token in 0..<configuration.decoderOutputSize {
            let value = pointer[token * tokenStride]
            guard value.isFinite else {
                throw Pix2TexDecoderError.nonFiniteLogits
            }
            if value > bestValue {
                bestValue = value
                bestToken = token
            }
        }
        return Int32(bestToken)
    }

    private func repeatedSuffix(in tokens: [Int32]) -> [Int32]? {
        let maximumPatternLength = min(
            configuration.maximumRepeatedPatternLength,
            tokens.count / configuration.minimumPatternRepetitions
        )
        guard maximumPatternLength > 0 else {
            return nil
        }

        for patternLength in 1...maximumPatternLength {
            let repeatedLength = (
                patternLength * configuration.minimumPatternRepetitions
            )
            let suffixStart = tokens.count - repeatedLength
            let patternStart = tokens.count - patternLength
            let pattern = Array(tokens[patternStart..<tokens.count])
            var matches = true
            for index in suffixStart..<tokens.count {
                if tokens[index] != pattern[(index - suffixStart) % patternLength] {
                    matches = false
                    break
                }
            }
            if matches {
                return pattern
            }
        }
        return nil
    }
}
