import SwiftMath
import XCTest
@testable import EqnSnap

@MainActor
final class FormulaLatexPreviewTests: XCTestCase {
    func testSwiftMathAcceptsRecognizedFormulaSyntax() {
        let label = MTMathUILabel()

        label.latex = #"=-\int_{0}^{x}\frac{1-t-1}{1-t}\,dt"#

        XCTAssertNil(label.error)
    }

    func testSwiftMathReportsUnsupportedLatexWithoutChangingSource() {
        let label = MTMathUILabel()
        let latex = #"\commandThatDoesNotExist{x}"#

        label.latex = latex

        XCTAssertNotNil(label.error)
        XCTAssertEqual(label.latex, latex)
    }
}
