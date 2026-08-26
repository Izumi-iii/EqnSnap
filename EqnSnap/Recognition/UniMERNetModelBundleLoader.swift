import CoreML
import Foundation

struct UniMERNetModelResourceNames: Sendable, Equatable {
    static let standard = UniMERNetModelResourceNames(
        encoder: "UniMERNetTinyEncoder-FP16",
        decoder: "UniMERNetTinyDecoder-CachedStep-SelfKV-FP16",
        tokenizer: "UniMERNetTokenizer"
    )

    let encoder: String
    let decoder: String
    let tokenizer: String
}

struct UniMERNetLoadedModelBundle {
    let encoder: MLModel
    let decoder: MLModel
    let tokenizer: UniMERNetTokenizer

    func makeRecognitionPipeline() -> UniMERNetRecognitionPipeline {
        UniMERNetRecognitionPipeline(
            encoderRunner: UniMERNetEncoderRunner(
                predictor: CoreMLUniMERNetEncoderModel(model: encoder)
            ),
            decoderRunner: UniMERNetCachedDecoderRunner(
                predictor: CoreMLUniMERNetCachedDecoderModel(model: decoder)
            ),
            tokenizer: tokenizer
        )
    }
}

enum UniMERNetModelBundleLoaderError: Error, Sendable, Equatable {
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
}

final class UniMERNetModelBundleLoader {
    private enum FeatureLocation {
        case input
        case output
    }

    private let resourceNames: UniMERNetModelResourceNames
    private let modelConfiguration: MLModelConfiguration

    init(
        resourceNames: UniMERNetModelResourceNames = .standard,
        computeUnits: MLComputeUnits = .all
    ) {
        self.resourceNames = resourceNames
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        self.modelConfiguration = configuration
    }

    static func resourcesAvailable(
        in bundle: Bundle = .main,
        resourceNames: UniMERNetModelResourceNames = .standard
    ) -> Bool {
        modelResourceURL(named: resourceNames.encoder, in: bundle) != nil
            && modelResourceURL(named: resourceNames.decoder, in: bundle) != nil
            && bundle.url(
                forResource: resourceNames.tokenizer,
                withExtension: "json"
            ) != nil
    }

    func load(from bundle: Bundle = .main) throws
        -> UniMERNetLoadedModelBundle
    {
        guard let encoderURL = Self.modelResourceURL(
            named: resourceNames.encoder,
            in: bundle
        ) else {
            throw UniMERNetModelBundleLoaderError.missingResource(
                resourceNames.encoder
            )
        }
        guard let decoderURL = Self.modelResourceURL(
            named: resourceNames.decoder,
            in: bundle
        ) else {
            throw UniMERNetModelBundleLoaderError.missingResource(
                resourceNames.decoder
            )
        }
        guard let tokenizerURL = bundle.url(
            forResource: resourceNames.tokenizer,
            withExtension: "json"
        ) else {
            throw UniMERNetModelBundleLoaderError.missingResource(
                "\(resourceNames.tokenizer).json"
            )
        }
        return try load(
            encoderURL: encoderURL,
            decoderURL: decoderURL,
            tokenizerURL: tokenizerURL
        )
    }

    func load(from directory: URL) throws -> UniMERNetLoadedModelBundle {
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
            throw UniMERNetModelBundleLoaderError.missingResource(
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
    ) throws -> UniMERNetLoadedModelBundle {
        let encoder = try loadModel(at: encoderURL)
        let decoder = try loadModel(at: decoderURL)
        let tokenizer = try UniMERNetTokenizer(contentsOf: tokenizerURL)

        try validateEncoder(encoder)
        try validateDecoder(decoder)
        return UniMERNetLoadedModelBundle(
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
            throw UniMERNetModelBundleLoaderError.unsupportedModelResource(url)
        }
        return try MLModel(
            contentsOf: modelURL,
            configuration: modelConfiguration
        )
    }

    private static func modelResourceURL(
        named name: String,
        in bundle: Bundle
    ) -> URL? {
        for fileExtension in ["mlmodelc", "mlpackage", "mlmodel"] {
            if let url = bundle.url(
                forResource: name,
                withExtension: fileExtension
            ) {
                return url
            }
        }
        return nil
    }

    private func modelResourceURL(named name: String, in directory: URL) throws
        -> URL
    {
        for fileExtension in ["mlmodelc", "mlpackage", "mlmodel"] {
            let url = directory
                .appendingPathComponent(name)
                .appendingPathExtension(fileExtension)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }
        throw UniMERNetModelBundleLoaderError.missingResource(name)
    }

    private func validateEncoder(_ model: MLModel) throws {
        let names = UniMERNetEncoderFeatureNames.standard
        let configuration = UniMERNetEncoderConfiguration.tinyV1
        try validateFeature(
            model: model,
            modelName: "encoder",
            location: .input,
            name: names.pixelValues,
            dataType: .float32,
            expectedShape: configuration.inputShape
        )
        try validateFeature(
            model: model,
            modelName: "encoder",
            location: .output,
            name: names.encoderContext,
            dataType: .float32,
            expectedShape: configuration.outputShape
        )
    }

    private func validateDecoder(_ model: MLModel) throws {
        let names = UniMERNetCachedDecoderFeatureNames.standard
        let configuration = UniMERNetCachedDecoderConfiguration.tinyV1
        try validateFeature(
            model: model,
            modelName: "decoder",
            location: .input,
            name: names.inputID,
            dataType: .int32,
            expectedShape: [1, 1]
        )
        try validateFeature(
            model: model,
            modelName: "decoder",
            location: .input,
            name: names.encoderContext,
            dataType: .float32,
            expectedShape: [
                1,
                configuration.contextLength,
                configuration.embeddingDimension,
            ]
        )
        try validateFeature(
            model: model,
            modelName: "decoder",
            location: .input,
            name: names.selfMask,
            dataType: .int32,
            expectedShape: [1, 1]
        )
        for layer in 0..<configuration.decoderLayers {
            try validateFeature(
                model: model,
                modelName: "decoder",
                location: .input,
                name: names.inputKey(layer: layer),
                dataType: .float32,
                expectedShape: [
                    1,
                    configuration.attentionHeads,
                    1,
                    configuration.keyDimension,
                ]
            )
            try validateFeature(
                model: model,
                modelName: "decoder",
                location: .input,
                name: names.inputValue(layer: layer),
                dataType: .float32,
                expectedShape: [
                    1,
                    configuration.attentionHeads,
                    1,
                    configuration.valueDimension,
                ]
            )
            try validateFeature(
                model: model,
                modelName: "decoder",
                location: .output,
                name: names.outputKey(layer: layer),
                dataType: .float32,
                expectedShape: nil
            )
            try validateFeature(
                model: model,
                modelName: "decoder",
                location: .output,
                name: names.outputValue(layer: layer),
                dataType: .float32,
                expectedShape: nil
            )
        }
        try validateFeature(
            model: model,
            modelName: "decoder",
            location: .output,
            name: names.logits,
            dataType: .float32,
            expectedShape: nil
        )
    }

    private func validateFeature(
        model: MLModel,
        modelName: String,
        location: FeatureLocation,
        name: String,
        dataType: MLMultiArrayDataType,
        expectedShape: [Int]?
    ) throws {
        let features: [String: MLFeatureDescription]
        switch location {
        case .input:
            features = model.modelDescription.inputDescriptionsByName
        case .output:
            features = model.modelDescription.outputDescriptionsByName
        }
        guard let feature = features[name] else {
            throw UniMERNetModelBundleLoaderError.missingFeature(
                model: modelName,
                feature: name
            )
        }
        guard feature.type == .multiArray,
              let constraint = feature.multiArrayConstraint
        else {
            throw UniMERNetModelBundleLoaderError.invalidFeatureType(
                model: modelName,
                feature: name
            )
        }
        guard constraint.dataType == dataType else {
            throw UniMERNetModelBundleLoaderError.invalidFeatureDataType(
                model: modelName,
                feature: name
            )
        }
        let actualShape = constraint.shape.map(\.intValue)
        if let expectedShape, actualShape != expectedShape {
            throw UniMERNetModelBundleLoaderError.invalidFeatureShape(
                model: modelName,
                feature: name,
                expected: expectedShape,
                actual: actualShape
            )
        }
    }
}
