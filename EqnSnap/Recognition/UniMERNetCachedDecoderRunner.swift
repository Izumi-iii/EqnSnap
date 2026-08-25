import CoreML
import Foundation

struct UniMERNetDecoderPrediction {
    let logits: MLMultiArray
    let selfCache: [MLMultiArray]
}

protocol UniMERNetCachedDecoderPredicting: AnyObject {
    func predict(
        inputID: MLMultiArray,
        encoderContext: MLMultiArray,
        selfCache: [MLMultiArray]
    ) throws -> UniMERNetDecoderPrediction
}

struct UniMERNetCachedDecoderFeatureNames: Sendable, Equatable {
    static let standard = UniMERNetCachedDecoderFeatureNames(
        inputID: "input_id",
        encoderContext: "encoder_context",
        selfMask: "self_mask",
        logits: "logits"
    )

    let inputID: String
    let encoderContext: String
    let selfMask: String
    let logits: String

    func inputKey(layer: Int) -> String { "self_key_\(layer)" }
    func inputValue(layer: Int) -> String { "self_value_\(layer)" }
    func outputKey(layer: Int) -> String { "next_self_key_\(layer)" }
    func outputValue(layer: Int) -> String { "next_self_value_\(layer)" }
}

final class CoreMLUniMERNetCachedDecoderModel:
    UniMERNetCachedDecoderPredicting
{
    private let model: MLModel
    private let names: UniMERNetCachedDecoderFeatureNames
    private let layerCount: Int

    init(
        model: MLModel,
        names: UniMERNetCachedDecoderFeatureNames = .standard,
        layerCount: Int = 8
    ) {
        self.model = model
        self.names = names
        self.layerCount = layerCount
    }

    func predict(
        inputID: MLMultiArray,
        encoderContext: MLMultiArray,
        selfCache: [MLMultiArray]
    ) throws -> UniMERNetDecoderPrediction {
        guard selfCache.count == layerCount * 2 else {
            throw UniMERNetCachedDecoderError.invalidCacheCount(
                expected: layerCount * 2,
                actual: selfCache.count
            )
        }

        var features: [String: MLFeatureValue] = [
            names.inputID: MLFeatureValue(multiArray: inputID),
            names.encoderContext: MLFeatureValue(multiArray: encoderContext),
            names.selfMask: MLFeatureValue(
                multiArray: try makeSelfMask(cacheLength: selfCache[0].shape[2].intValue)
            ),
        ]
        for layer in 0..<layerCount {
            features[names.inputKey(layer: layer)] = MLFeatureValue(
                multiArray: selfCache[layer * 2]
            )
            features[names.inputValue(layer: layer)] = MLFeatureValue(
                multiArray: selfCache[layer * 2 + 1]
            )
        }

        let input = try MLDictionaryFeatureProvider(dictionary: features)
        let output = try model.prediction(from: input)
        guard let logits = output.featureValue(
            for: names.logits
        )?.multiArrayValue else {
            throw UniMERNetCachedDecoderError.missingOutput(names.logits)
        }

        var nextCache: [MLMultiArray] = []
        nextCache.reserveCapacity(layerCount * 2)
        for layer in 0..<layerCount {
            for name in [
                names.outputKey(layer: layer),
                names.outputValue(layer: layer),
            ] {
                guard let array = output.featureValue(
                    for: name
                )?.multiArrayValue else {
                    throw UniMERNetCachedDecoderError.missingOutput(name)
                }
                nextCache.append(array)
            }
        }
        return UniMERNetDecoderPrediction(
            logits: logits,
            selfCache: nextCache
        )
    }

    private func makeSelfMask(cacheLength: Int) throws -> MLMultiArray {
        let mask = try MLMultiArray(
            shape: [1, NSNumber(value: cacheLength)],
            dataType: .int32
        )
        let pointer = mask.dataPointer.assumingMemoryBound(to: Int32.self)
        let stride = mask.strides[1].intValue
        for index in 0..<cacheLength {
            pointer[index * stride] = index == 0 ? 0 : 1
        }
        return mask
    }
}

struct UniMERNetCachedDecoderConfiguration: Sendable, Equatable {
    static let tinyV1 = UniMERNetCachedDecoderConfiguration(
        maximumTokenLength: 512,
        contextLength: 126,
        embeddingDimension: 512,
        decoderOutputSize: 50_000,
        tokenizerVocabularySize: 50_000,
        decoderLayers: 8,
        attentionHeads: 16,
        keyDimension: 16,
        valueDimension: 32,
        bosTokenID: 0,
        eosTokenID: 2,
        padTokenID: 1,
        maximumRepeatedPatternLength: 8,
        minimumPatternRepetitions: 6
    )

    let maximumTokenLength: Int
    let contextLength: Int
    let embeddingDimension: Int
    let decoderOutputSize: Int
    let tokenizerVocabularySize: Int
    let decoderLayers: Int
    let attentionHeads: Int
    let keyDimension: Int
    let valueDimension: Int
    let bosTokenID: Int32
    let eosTokenID: Int32
    let padTokenID: Int32
    let maximumRepeatedPatternLength: Int
    let minimumPatternRepetitions: Int
}

struct UniMERNetDecodedSequence: Sendable, Equatable {
    let tokenIDs: [Int32]
    let decoderSteps: Int
}

enum UniMERNetCachedDecoderError: Error, Sendable, Equatable {
    case invalidEncoderContextShape([Int])
    case invalidEncoderContextDataType
    case invalidCacheCount(expected: Int, actual: Int)
    case invalidCacheShape(index: Int, expected: [Int], actual: [Int])
    case invalidCacheDataType(index: Int)
    case invalidLogitsShape([Int])
    case invalidLogitsDataType
    case nonFiniteLogits
    case missingOutput(String)
    case invalidGeneratedToken(Int32)
    case maximumTokenLengthReached
    case repetitionDetected([Int32])
}

final class UniMERNetCachedDecoderRunner {
    private let predictor: UniMERNetCachedDecoderPredicting
    private let configuration: UniMERNetCachedDecoderConfiguration

    init(
        predictor: UniMERNetCachedDecoderPredicting,
        configuration: UniMERNetCachedDecoderConfiguration = .tinyV1
    ) {
        self.predictor = predictor
        self.configuration = configuration
    }

    func decode(encoderContext: MLMultiArray) async throws
        -> UniMERNetDecodedSequence
    {
        try Task.checkCancellation()
        try validateEncoderContext(encoderContext)

        let inputID = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let inputPointer = inputID.dataPointer.assumingMemoryBound(
            to: Int32.self
        )
        var cache = try makeInitialCache()
        var currentToken = configuration.bosTokenID
        var generatedTokens: [Int32] = []

        for step in 0..<configuration.maximumTokenLength {
            try Task.checkCancellation()
            inputPointer[0] = currentToken
            let prediction = try predictor.predict(
                inputID: inputID,
                encoderContext: encoderContext,
                selfCache: cache
            )
            try Task.checkCancellation()
            try validateCache(prediction.selfCache, pastLength: step + 2)
            let nextToken = try argmax(prediction.logits)

            if nextToken == configuration.eosTokenID {
                return UniMERNetDecodedSequence(
                    tokenIDs: generatedTokens,
                    decoderSteps: step + 1
                )
            }
            guard nextToken != configuration.padTokenID,
                  nextToken != configuration.bosTokenID,
                  nextToken >= 0,
                  Int(nextToken) < configuration.tokenizerVocabularySize
            else {
                throw UniMERNetCachedDecoderError.invalidGeneratedToken(
                    nextToken
                )
            }

            generatedTokens.append(nextToken)
            if let pattern = repeatedSuffix(in: generatedTokens) {
                throw UniMERNetCachedDecoderError.repetitionDetected(pattern)
            }
            currentToken = nextToken
            cache = prediction.selfCache
        }

        throw UniMERNetCachedDecoderError.maximumTokenLengthReached
    }

    private func validateEncoderContext(_ context: MLMultiArray) throws {
        let shape = context.shape.map(\.intValue)
        guard shape == [
            1,
            configuration.contextLength,
            configuration.embeddingDimension,
        ] else {
            throw UniMERNetCachedDecoderError.invalidEncoderContextShape(shape)
        }
        guard context.dataType == .float32 else {
            throw UniMERNetCachedDecoderError.invalidEncoderContextDataType
        }
    }

    private func makeInitialCache() throws -> [MLMultiArray] {
        var cache: [MLMultiArray] = []
        cache.reserveCapacity(configuration.decoderLayers * 2)
        for _ in 0..<configuration.decoderLayers {
            cache.append(
                try makeCacheArray(
                    pastLength: 1,
                    dimension: configuration.keyDimension
                )
            )
            cache.append(
                try makeCacheArray(
                    pastLength: 1,
                    dimension: configuration.valueDimension
                )
            )
        }
        return cache
    }

    private func makeCacheArray(
        pastLength: Int,
        dimension: Int
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [
                1,
                NSNumber(value: configuration.attentionHeads),
                NSNumber(value: pastLength),
                NSNumber(value: dimension),
            ],
            dataType: .float32
        )
        let pointer = array.dataPointer.assumingMemoryBound(to: Float32.self)
        pointer.initialize(repeating: 0, count: array.count)
        return array
    }

    private func validateCache(
        _ cache: [MLMultiArray],
        pastLength: Int
    ) throws {
        let expectedCount = configuration.decoderLayers * 2
        guard cache.count == expectedCount else {
            throw UniMERNetCachedDecoderError.invalidCacheCount(
                expected: expectedCount,
                actual: cache.count
            )
        }
        for index in 0..<cache.count {
            let dimension = index.isMultiple(of: 2)
                ? configuration.keyDimension
                : configuration.valueDimension
            let expectedShape = [
                1,
                configuration.attentionHeads,
                pastLength,
                dimension,
            ]
            let actualShape = cache[index].shape.map(\.intValue)
            guard actualShape == expectedShape else {
                throw UniMERNetCachedDecoderError.invalidCacheShape(
                    index: index,
                    expected: expectedShape,
                    actual: actualShape
                )
            }
            guard cache[index].dataType == .float32 else {
                throw UniMERNetCachedDecoderError.invalidCacheDataType(
                    index: index
                )
            }
        }
    }

    private func argmax(_ logits: MLMultiArray) throws -> Int32 {
        let shape = logits.shape.map(\.intValue)
        guard shape == [1, 1, configuration.decoderOutputSize] else {
            throw UniMERNetCachedDecoderError.invalidLogitsShape(shape)
        }
        guard logits.dataType == .float32 else {
            throw UniMERNetCachedDecoderError.invalidLogitsDataType
        }

        let pointer = logits.dataPointer.assumingMemoryBound(to: Float32.self)
        let tokenStride = logits.strides[2].intValue
        var bestToken = 0
        var bestValue = -Float.greatestFiniteMagnitude
        for token in 0..<configuration.decoderOutputSize {
            let value = pointer[token * tokenStride]
            guard value.isFinite else {
                throw UniMERNetCachedDecoderError.nonFiniteLogits
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
            let matches = (suffixStart..<tokens.count).allSatisfy { index in
                tokens[index] == pattern[(index - suffixStart) % patternLength]
            }
            if matches {
                return pattern
            }
        }
        return nil
    }
}
