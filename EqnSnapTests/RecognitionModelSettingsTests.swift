import Foundation
import XCTest
@testable import EqnSnap

@MainActor
final class RecognitionModelSettingsTests: XCTestCase {
    func testDefaultsToPix2TexAndPersistsAvailableSelection() throws {
        let defaults = try makeDefaults()
        defer { clear(defaults) }
        let settings = RecognitionModelSettings(
            defaults: defaults,
            isAvailable: { _ in true }
        )

        XCTAssertEqual(settings.selectedModel, .pix2tex)
        settings.selectedModel = .uniMERNet

        XCTAssertEqual(
            defaults.string(forKey: FormulaRecognitionModel.defaultsKey),
            FormulaRecognitionModel.uniMERNet.rawValue
        )
    }

    func testFallsBackWhenPersistedModelIsUnavailable() throws {
        let defaults = try makeDefaults()
        defer { clear(defaults) }
        defaults.set(
            FormulaRecognitionModel.uniMERNet.rawValue,
            forKey: FormulaRecognitionModel.defaultsKey
        )

        let settings = RecognitionModelSettings(
            defaults: defaults,
            isAvailable: { $0 == .pix2tex }
        )

        XCTAssertEqual(settings.selectedModel, .pix2tex)
        XCTAssertEqual(
            defaults.string(forKey: FormulaRecognitionModel.defaultsKey),
            FormulaRecognitionModel.pix2tex.rawValue
        )
    }

    private func makeDefaults() throws -> UserDefaults {
        let name = "RecognitionModelSettingsTests.\(UUID().uuidString)"
        guard let defaults = UserDefaults(suiteName: name) else {
            throw RecognitionModelSettingsFixtureError.defaultsCreationFailed
        }
        return defaults
    }

    private func clear(_ defaults: UserDefaults) {
        defaults.removeObject(forKey: FormulaRecognitionModel.defaultsKey)
    }
}

private enum RecognitionModelSettingsFixtureError: Error {
    case defaultsCreationFailed
}
