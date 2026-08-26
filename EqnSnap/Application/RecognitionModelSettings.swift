import Combine
import Foundation

enum FormulaRecognitionModel: String, CaseIterable, Identifiable, Sendable {
    case pix2tex
    case uniMERNet = "unimernet"

    static let defaultsKey = "recognition.selectedModel"
    static let defaultModel = FormulaRecognitionModel.pix2tex

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .pix2tex:
            return "pix2tex"
        case .uniMERNet:
            return "UniMERNet"
        }
    }
}

@MainActor
final class RecognitionModelSettings: ObservableObject {
    static let shared = RecognitionModelSettings()

    @Published var selectedModel: FormulaRecognitionModel {
        didSet {
            guard isAvailable(selectedModel) else {
                if selectedModel != .defaultModel {
                    selectedModel = .defaultModel
                }
                return
            }
            defaults.set(
                selectedModel.rawValue,
                forKey: FormulaRecognitionModel.defaultsKey
            )
        }
    }

    private let defaults: UserDefaults
    private let isAvailable: (FormulaRecognitionModel) -> Bool

    init(
        defaults: UserDefaults = .standard,
        isAvailable: @escaping (FormulaRecognitionModel) -> Bool = {
            switch $0 {
            case .pix2tex:
                return true
            case .uniMERNet:
                return UniMERNetModelBundleLoader.resourcesAvailable()
            }
        }
    ) {
        self.defaults = defaults
        self.isAvailable = isAvailable
        let stored = defaults.string(
            forKey: FormulaRecognitionModel.defaultsKey
        ).flatMap(FormulaRecognitionModel.init(rawValue:))
        let selected = stored ?? .defaultModel
        if isAvailable(selected) {
            selectedModel = selected
        } else {
            selectedModel = .defaultModel
            defaults.set(
                FormulaRecognitionModel.defaultModel.rawValue,
                forKey: FormulaRecognitionModel.defaultsKey
            )
        }
    }

    func modelIsAvailable(_ model: FormulaRecognitionModel) -> Bool {
        isAvailable(model)
    }
}
