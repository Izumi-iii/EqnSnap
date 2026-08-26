import CoreGraphics

struct CapturedFormulaImage: @unchecked Sendable {
    let image: CGImage
}

struct FormulaRecognitionOutput: Sendable, Equatable {
    let model: FormulaRecognitionModel
    let latex: String
}

protocol FormulaRecognitionEngine: AnyObject {
    func recognize(_ input: CapturedFormulaImage) async throws -> String
}

private final class Pix2TexRecognitionEngine: FormulaRecognitionEngine {
    private let recognizer: Pix2TexScreenshotRecognizer

    init() throws {
        recognizer = try Pix2TexScreenshotRecognizer()
    }

    func recognize(_ input: CapturedFormulaImage) async throws -> String {
        try await recognizer.recognize(input.image).latex
    }
}

private final class UniMERNetRecognitionEngine: FormulaRecognitionEngine {
    private let recognizer: UniMERNetScreenshotRecognizer

    init() throws {
        recognizer = try UniMERNetScreenshotRecognizer()
    }

    func recognize(_ input: CapturedFormulaImage) async throws -> String {
        try await recognizer.recognize(input.image).latex
    }
}

actor FormulaRecognitionService {
    typealias EngineFactory = () throws -> any FormulaRecognitionEngine

    private struct LoadedEngine {
        let model: FormulaRecognitionModel
        let engine: any FormulaRecognitionEngine
    }

    private let pix2texFactory: EngineFactory
    private let uniMERNetFactory: EngineFactory
    private var loadedEngine: LoadedEngine?

    init() {
        pix2texFactory = { try Pix2TexRecognitionEngine() }
        uniMERNetFactory = { try UniMERNetRecognitionEngine() }
    }

    init(
        pix2texFactory: @escaping EngineFactory,
        uniMERNetFactory: @escaping EngineFactory
    ) {
        self.pix2texFactory = pix2texFactory
        self.uniMERNetFactory = uniMERNetFactory
    }

    func recognize(
        _ input: CapturedFormulaImage,
        using model: FormulaRecognitionModel
    ) async throws -> FormulaRecognitionOutput {
        try Task.checkCancellation()
        let engine = try engine(for: model)
        let latex = try await engine.recognize(input)
        try Task.checkCancellation()
        return FormulaRecognitionOutput(model: model, latex: latex)
    }

    func discardLoadedEngine(except model: FormulaRecognitionModel) {
        guard loadedEngine?.model != model else { return }
        loadedEngine = nil
    }

    private func engine(for model: FormulaRecognitionModel) throws
        -> any FormulaRecognitionEngine
    {
        if let loadedEngine, loadedEngine.model == model {
            return loadedEngine.engine
        }
        loadedEngine = nil

        let engine: any FormulaRecognitionEngine
        switch model {
        case .pix2tex:
            engine = try pix2texFactory()
        case .uniMERNet:
            engine = try uniMERNetFactory()
        }
        loadedEngine = LoadedEngine(model: model, engine: engine)
        return engine
    }
}
