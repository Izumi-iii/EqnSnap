import CoreML
import Foundation

nonisolated struct Pix2TexModelResourceNames: Sendable, Equatable {
    static let standard = Pix2TexModelResourceNames(
        encoder: "Pix2TexEncoder-Variable32",
        decoder: "Pix2TexDecoder-Prefix128-Context169",
        tokenizer: "tokenizer"
    )

    let encoder: String
    let decoder: String
    let tokenizer: String
}

nonisolated struct Pix2TexLoadedModelBundle {
    let encoder: MLModel
    let decoder: MLModel
    let tokenizer: Pix2TexTokenizer

    func makeRecognitionPipeline() -> Pix2TexRecognitionPipeline {
        Pix2TexRecognitionPipeline(
            encoderRunner: Pix2TexEncoderRunner(
                predictor: CoreMLPix2TexEncoderModel(model: encoder)
            ),
            decoderRunner: Pix2TexDecoderRunner(
                predictor: CoreMLPix2TexDecoderModel(model: decoder)
            ),
            tokenizer: tokenizer
        )
    }
}

nonisolated enum Pix2TexModelBundleLoaderError: Error, Sendable, Equatable {
    case missingResource(String)
    case unsupportedModelResource(URL)
    case missingFeature(model: String, feature: String)
    case invalidFeatureType(model: String, feature: String)
    case invalidFeatureDataType(model: String, feature: String)
    case invalidFeatureShape(
        model: String,
        feature: String,
        expected: [Int],
        actual: [Int]
    )
    case incompatibleTokenizer
}

nonisolated final class Pix2TexModelBundleLoader {
    private enum FeatureLocation {
        case input
        case output
    }

    private let resourceNames: Pix2TexModelResourceNames
    private let modelConfiguration: MLModelConfiguration

    init(
        resourceNames: Pix2TexModelResourceNames = .standard,
        computeUnits: MLComputeUnits = .all
    ) {
        self.resourceNames = resourceNames
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        self.modelConfiguration = configuration
    }

    func load(from bundle: Bundle = .main) throws
        -> Pix2TexLoadedModelBundle
    {
        let encoderURL = try modelResourceURL(
            named: resourceNames.encoder,
            in: bundle
        )
        let decoderURL = try modelResourceURL(
            named: resourceNames.decoder,
            in: bundle
        )
        guard let tokenizerURL = bundle.url(
            forResource: resourceNames.tokenizer,
            withExtension: "json"
        ) else {
            throw Pix2TexModelBundleLoaderError.missingResource(
                "\(resourceNames.tokenizer).json"
            )
        }
        return try load(
            encoderURL: encoderURL,
            decoderURL: decoderURL,
            tokenizerURL: tokenizerURL
        )
    }

    func load(from directory: URL) throws -> Pix2TexLoadedModelBundle {
        let encoderURL = try modelResourceURL(
            named: resourceNames.encoder,
            in: directory
        )
        let decoderURL = try modelResourceURL(
            named: resourceNames.decoder,
            in: directory
        )
        let tokenizerURL = directory
            .appendingPathComponent(resourceNames.tokenizer)
            .appendingPathExtension("json")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw Pix2TexModelBundleLoaderError.missingResource(
                tokenizerURL.lastPathComponent
            )
        }
        return try load(
            encoderURL: encoderURL,
            decoderURL: decoderURL,
            tokenizerURL: tokenizerURL
        )
    }

    func load(
        encoderURL: URL,
        decoderURL: URL,
        tokenizerURL: URL
    ) throws -> Pix2TexLoadedModelBundle {
        let encoder = try loadModel(at: encoderURL)
        let decoder = try loadModel(at: decoderURL)
        let tokenizer = try Pix2TexTokenizer(contentsOf: tokenizerURL)

        try validateEncoder(encoder)
        try validateDecoder(decoder)
        try validateTokenizer(tokenizer.resource)

        return Pix2TexLoadedModelBundle(
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer
        )
    }

    private func loadModel(at url: URL) throws -> MLModel {
        let modelURL: URL
        switch url.pathExtension {
        case "mlmodelc":
            modelURL = url
        case "mlmodel", "mlpackage":
            modelURL = try MLModel.compileModel(at: url)
        default:
            throw Pix2TexModelBundleLoaderError.unsupportedModelResource(url)
        }
        return try MLModel(
            contentsOf: modelURL,
            configuration: modelConfiguration
        )
    }

    private func modelResourceURL(
        named name: String,
        in bundle: Bundle
    ) throws -> URL {
        for fileExtension in ["mlmodelc", "mlpackage", "mlmodel"] {
            if let url = bundle.url(
                forResource: name,
                withExtension: fileExtension
            ) {
                return url
            }
        }
        throw Pix2TexModelBundleLoaderError.missingResource(name)
    }

    private func modelResourceURL(
        named name: String,
        in directory: URL
    ) throws -> URL {
        for fileExtension in ["mlmodelc", "mlpackage", "mlmodel"] {
            let url = directory
                .appendingPathComponent(name)
                .appendingPathExtension(fileExtension)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }
        throw Pix2TexModelBundleLoaderError.missingResource(name)
    }

    private func validateEncoder(_ model: MLModel) throws {
        try validateMultiArrayFeature(
            model: model,
            modelName: "encoder",
            location: .input,
            name: Pix2TexEncoderFeatureNames.standard.pixelValues,
            dataType: .float32,
            expectedShape: nil,
            expectedRank: 4
        )
        try validateMultiArrayFeature(
            model: model,
            modelName: "encoder",
            location: .output,
            name: Pix2TexEncoderFeatureNames.standard.encoderContext,
            dataType: .float32,
            expectedShape: nil,
            expectedRank: nil
        )
    }

    private func validateDecoder(_ model: MLModel) throws {
        let names = Pix2TexDecoderFeatureNames.standard
        let configuration = Pix2TexDecoderConfiguration.pix2texV1
        try validateMultiArrayFeature(
            model: model,
            modelName: "decoder",
            location: .input,
            name: names.inputIDs,
            dataType: .int32,
            expectedShape: [1, configuration.maximumTokenLength]
        )
        try validateMultiArrayFeature(
            model: model,
            modelName: "decoder",
            location: .input,
            name: names.tokenMask,
            dataType: .int32,
            expectedShape: [1, configuration.maximumTokenLength]
        )
        try validateMultiArrayFeature(
            model: model,
            modelName: "decoder",
            location: .input,
            name: names.encoderContext,
            dataType: .float32,
            expectedShape: [
                1,
                configuration.maximumContextLength,
                configuration.embeddingDimension,
            ]
        )
        try validateMultiArrayFeature(
            model: model,
            modelName: "decoder",
            location: .input,
            name: names.contextMask,
            dataType: .int32,
            expectedShape: [1, configuration.maximumContextLength]
        )
        try validateMultiArrayFeature(
            model: model,
            modelName: "decoder",
            location: .output,
            name: names.logits,
            dataType: .float32,
            expectedShape: [1, 1, configuration.decoderOutputSize]
        )
    }

    private func validateMultiArrayFeature(
        model: MLModel,
        modelName: String,
        location: FeatureLocation,
        name: String,
        dataType: MLMultiArrayDataType,
        expectedShape: [Int]?,
        expectedRank: Int? = nil
    ) throws {
        let features: [String: MLFeatureDescription]
        switch location {
        case .input:
            features = model.modelDescription.inputDescriptionsByName
        case .output:
            features = model.modelDescription.outputDescriptionsByName
        }
        guard let feature = features[name] else {
            throw Pix2TexModelBundleLoaderError.missingFeature(
                model: modelName,
                feature: name
            )
        }
        guard feature.type == .multiArray,
              let constraint = feature.multiArrayConstraint
        else {
            throw Pix2TexModelBundleLoaderError.invalidFeatureType(
                model: modelName,
                feature: name
            )
        }
        guard constraint.dataType == dataType else {
            throw Pix2TexModelBundleLoaderError.invalidFeatureDataType(
                model: modelName,
                feature: name
            )
        }

        let actualShape = constraint.shape.map(\.intValue)
        if let expectedShape, actualShape != expectedShape {
            throw Pix2TexModelBundleLoaderError.invalidFeatureShape(
                model: modelName,
                feature: name,
                expected: expectedShape,
                actual: actualShape
            )
        }
        if let expectedRank, actualShape.count != expectedRank {
            throw Pix2TexModelBundleLoaderError.invalidFeatureShape(
                model: modelName,
                feature: name,
                expected: [Int](repeating: -1, count: expectedRank),
                actual: actualShape
            )
        }
    }

    private func validateTokenizer(
        _ resource: Pix2TexTokenizerResource
    ) throws {
        let configuration = Pix2TexDecoderConfiguration.pix2texV1
        guard resource.vocabularySize
                == configuration.tokenizerVocabularySize,
              resource.decoderOutputSize == configuration.decoderOutputSize,
              resource.specialTokens.bos == configuration.bosTokenID,
              resource.specialTokens.eos == configuration.eosTokenID,
              resource.specialTokens.pad == configuration.padTokenID
        else {
            throw Pix2TexModelBundleLoaderError.incompatibleTokenizer
        }
    }
}
