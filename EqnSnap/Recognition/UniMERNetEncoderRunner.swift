import CoreML
import Foundation

protocol UniMERNetEncoderPredicting: AnyObject {
    func predict(pixelValues: MLMultiArray) throws -> MLMultiArray
}

struct UniMERNetEncoderFeatureNames: Sendable, Equatable {
    static let standard = UniMERNetEncoderFeatureNames(
        pixelValues: "pixel_values",
        encoderContext: "encoder_context"
    )

    let pixelValues: String
    let encoderContext: String
}

final class CoreMLUniMERNetEncoderModel: UniMERNetEncoderPredicting {
    private let model: MLModel
    private let names: UniMERNetEncoderFeatureNames

    init(
        model: MLModel,
        names: UniMERNetEncoderFeatureNames = .standard
    ) {
        self.model = model
        self.names = names
    }

    func predict(pixelValues: MLMultiArray) throws -> MLMultiArray {
        let input = try MLDictionaryFeatureProvider(
            dictionary: [
                names.pixelValues: MLFeatureValue(multiArray: pixelValues),
            ]
        )
        let output = try model.prediction(from: input)
        guard let context = output.featureValue(
            for: names.encoderContext
        )?.multiArrayValue else {
            throw UniMERNetEncoderError.missingContextOutput(
                names.encoderContext
            )
        }
        return context
    }
}

struct UniMERNetEncoderConfiguration: Sendable, Equatable {
    static let tinyV1 = UniMERNetEncoderConfiguration(
        inputShape: [1, 3, 192, 672],
        outputShape: [1, 126, 512]
    )

    let inputShape: [Int]
    let outputShape: [Int]
}

enum UniMERNetEncoderError: Error, Sendable, Equatable {
    case invalidInputShape([Int])
    case invalidInputDataType
    case invalidContextShape([Int])
    case invalidContextDataType
    case missingContextOutput(String)
}

final class UniMERNetEncoderRunner {
    private let predictor: UniMERNetEncoderPredicting
    private let configuration: UniMERNetEncoderConfiguration

    init(
        predictor: UniMERNetEncoderPredicting,
        configuration: UniMERNetEncoderConfiguration = .tinyV1
    ) {
        self.predictor = predictor
        self.configuration = configuration
    }

    func encode(_ pixelValues: MLMultiArray) async throws -> MLMultiArray {
        try Task.checkCancellation()
        guard pixelValues.shape.map(\.intValue) == configuration.inputShape else {
            throw UniMERNetEncoderError.invalidInputShape(
                pixelValues.shape.map(\.intValue)
            )
        }
        guard pixelValues.dataType == .float32 else {
            throw UniMERNetEncoderError.invalidInputDataType
        }

        let context = try predictor.predict(pixelValues: pixelValues)
        try Task.checkCancellation()
        guard context.shape.map(\.intValue) == configuration.outputShape else {
            throw UniMERNetEncoderError.invalidContextShape(
                context.shape.map(\.intValue)
            )
        }
        guard context.dataType == .float32 else {
            throw UniMERNetEncoderError.invalidContextDataType
        }
        return context
    }
}
