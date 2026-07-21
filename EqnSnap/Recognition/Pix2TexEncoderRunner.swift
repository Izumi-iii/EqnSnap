import CoreML
import Foundation

protocol Pix2TexEncoderPredicting: AnyObject {
    func predict(pixelValues: MLMultiArray) throws -> MLMultiArray
}

struct Pix2TexEncoderFeatureNames: Sendable, Equatable {
    static let standard = Pix2TexEncoderFeatureNames(
        pixelValues: "pixel_values",
        encoderContext: "encoder_context"
    )

    let pixelValues: String
    let encoderContext: String
}

final class CoreMLPix2TexEncoderModel: Pix2TexEncoderPredicting {
    private let model: MLModel
    private let names: Pix2TexEncoderFeatureNames

    init(
        model: MLModel,
        names: Pix2TexEncoderFeatureNames = .standard
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
            throw Pix2TexEncoderError.missingContextOutput(
                names.encoderContext
            )
        }
        return context
    }
}

struct Pix2TexEncoderConfiguration: Sendable, Equatable {
    static let variable32 = Pix2TexEncoderConfiguration(
        supportedHeights: [32, 64],
        minimumWidth: 32,
        maximumWidth: 672,
        shapeMultiple: 32,
        patchSize: 16,
        embeddingDimension: 256
    )

    let supportedHeights: [Int]
    let minimumWidth: Int
    let maximumWidth: Int
    let shapeMultiple: Int
    let patchSize: Int
    let embeddingDimension: Int
}

enum Pix2TexEncoderError: Error, Sendable, Equatable {
    case invalidInputShape([Int])
    case invalidInputDataType
    case invalidContextShape(expected: [Int], actual: [Int])
    case invalidContextDataType
    case missingContextOutput(String)
}

final class Pix2TexEncoderRunner {
    private let predictor: Pix2TexEncoderPredicting
    private let configuration: Pix2TexEncoderConfiguration

    init(
        predictor: Pix2TexEncoderPredicting,
        configuration: Pix2TexEncoderConfiguration = .variable32
    ) {
        self.predictor = predictor
        self.configuration = configuration
    }

    func encode(_ pixelValues: MLMultiArray) async throws -> MLMultiArray {
        try Task.checkCancellation()
        let inputShape = pixelValues.shape.map(\.intValue)
        try validateInput(shape: inputShape, dataType: pixelValues.dataType)

        let context = try predictor.predict(pixelValues: pixelValues)
        try Task.checkCancellation()

        let expectedShape = contextShape(forInputShape: inputShape)
        let actualShape = context.shape.map(\.intValue)
        guard actualShape == expectedShape else {
            throw Pix2TexEncoderError.invalidContextShape(
                expected: expectedShape,
                actual: actualShape
            )
        }
        guard context.dataType == .float32 else {
            throw Pix2TexEncoderError.invalidContextDataType
        }
        return context
    }

    private func validateInput(
        shape: [Int],
        dataType: MLMultiArrayDataType
    ) throws {
        guard shape.count == 4,
              shape[0] == 1,
              shape[1] == 1,
              configuration.supportedHeights.contains(shape[2]),
              shape[3] >= configuration.minimumWidth,
              shape[3] <= configuration.maximumWidth,
              shape[3].isMultiple(of: configuration.shapeMultiple)
        else {
            throw Pix2TexEncoderError.invalidInputShape(shape)
        }
        guard dataType == .float32 else {
            throw Pix2TexEncoderError.invalidInputDataType
        }
    }

    private func contextShape(forInputShape shape: [Int]) -> [Int] {
        let patchRows = shape[2] / configuration.patchSize
        let patchColumns = shape[3] / configuration.patchSize
        return [
            1,
            patchRows * patchColumns + 1,
            configuration.embeddingDimension,
        ]
    }
}
