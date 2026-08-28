import CoreGraphics
import Foundation
import ImageIO
import XCTest
@testable import EqnSnap

final class FormulaRecognitionServiceTests: XCTestCase {
    func testLoadsOnDemandReusesCurrentModelAndReleasesItOnSwitch() async throws {
        let counts = EngineFactoryCounts()
        let service = FormulaRecognitionService(
            pix2texFactory: {
                counts.pix2tex += 1
                return StubFormulaRecognitionEngine(latex: "pix")
            },
            uniMERNetFactory: {
                counts.uniMERNet += 1
                return StubFormulaRecognitionEngine(latex: "uni")
            }
        )
        let input = CapturedFormulaImage(image: try makeImage())

        let first = try await service.recognize(input, using: .pix2tex)
        let second = try await service.recognize(input, using: .pix2tex)
        let third = try await service.recognize(input, using: .uniMERNet)
        let fourth = try await service.recognize(input, using: .pix2tex)

        XCTAssertEqual(first, FormulaRecognitionOutput(model: .pix2tex, latex: "pix"))
        XCTAssertEqual(second, first)
        XCTAssertEqual(third, FormulaRecognitionOutput(model: .uniMERNet, latex: "uni"))
        XCTAssertEqual(fourth, first)
        XCTAssertEqual(counts.pix2tex, 2)
        XCTAssertEqual(counts.uniMERNet, 1)
    }

    func testDiscardReleasesEngineForDifferentSelection() async throws {
        let counts = EngineFactoryCounts()
        let service = FormulaRecognitionService(
            pix2texFactory: {
                counts.pix2tex += 1
                return StubFormulaRecognitionEngine(latex: "pix")
            },
            uniMERNetFactory: {
                counts.uniMERNet += 1
                return StubFormulaRecognitionEngine(latex: "uni")
            }
        )
        let input = CapturedFormulaImage(image: try makeImage())

        _ = try await service.recognize(input, using: .pix2tex)
        await service.discardLoadedEngine(except: .uniMERNet)
        _ = try await service.recognize(input, using: .pix2tex)

        XCTAssertEqual(counts.pix2tex, 2)
        XCTAssertEqual(counts.uniMERNet, 0)
    }

    func testRecognizesWithBundledModelsWhenSwitchingBackAndForth() async throws {
        guard UniMERNetModelBundleLoader.resourcesAvailable() else {
            throw XCTSkip("Bundled UniMERNet models are not available")
        }
        let service = FormulaRecognitionService()
        let input = CapturedFormulaImage(image: try loadFormulaImage())

        let pix2tex = try await service.recognize(input, using: .pix2tex)
        let uniMERNet = try await service.recognize(input, using: .uniMERNet)
        let pix2texAgain = try await service.recognize(input, using: .pix2tex)

        XCTAssertEqual(pix2tex.model, .pix2tex)
        XCTAssertEqual(
            pix2tex.latex,
            "=-\\int_{0}^{x}{\\frac{1-t-1}{1-t}}d t=-\\int_{0}^{x}(1-{\\frac{1}{1-t}})d t=-\\ln(1-x)-x"
        )
        XCTAssertEqual(uniMERNet.model, .uniMERNet)
        XCTAssertEqual(
            uniMERNet.latex,
            "= - \\! \\! \\int _ { 0 } ^ { x } \\! \\frac { 1 \\! - \\! t - 1 } { 1 \\! - \\! t } d t = - \\! \\! \\int _ { 0 } ^ { x } ( 1 \\! - \\! \\frac { 1 } { 1 \\! - \\! t } ) d t = - \\! \\ln ( 1 \\! - \\! x ) - x"
        )
        XCTAssertEqual(pix2texAgain, pix2tex)
    }

    private func makeImage() throws -> CGImage {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(
            data: nil,
            width: 1,
            height: 1,
            bitsPerComponent: 8,
            bytesPerRow: 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
        guard let image = context?.makeImage() else {
            throw FormulaRecognitionServiceFixtureError.imageCreationFailed
        }
        return image
    }

    private func loadFormulaImage() throws -> CGImage {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Tools/model-conversion/test_images/formula.png")
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw FormulaRecognitionServiceFixtureError.imageCreationFailed
        }
        return image
    }
}

private final class EngineFactoryCounts {
    var pix2tex = 0
    var uniMERNet = 0
}

private final class StubFormulaRecognitionEngine: FormulaRecognitionEngine {
    private let latex: String

    init(latex: String) {
        self.latex = latex
    }

    func recognize(_ input: CapturedFormulaImage) async throws -> String {
        latex
    }
}

private enum FormulaRecognitionServiceFixtureError: Error {
    case imageCreationFailed
}
