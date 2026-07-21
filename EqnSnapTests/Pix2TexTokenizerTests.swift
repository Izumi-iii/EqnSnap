import Foundation
import XCTest
@testable import EqnSnap

final class Pix2TexTokenizerTests: XCTestCase {
    private struct FixtureSuite: Decodable {
        let cases: [Fixture]
    }

    private struct Fixture: Decodable {
        let name: String
        let tokenIDs: [Int32]
        let decoded: String
        let postProcessed: String
    }

    func testMatchesPythonTokenizerFixtures() throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let tokenizerURL = repositoryRoot
            .appendingPathComponent(
                "EqnSnap/Resources/Models/Pix2Tex/tokenizer.json"
            )
        let fixtureURL = repositoryRoot
            .appendingPathComponent(
                "EqnSnapTests/Fixtures/Tokenizer/tokenizer-fixtures.json"
            )
        let tokenizer = try Pix2TexTokenizer(contentsOf: tokenizerURL)
        let fixtureData = try Data(contentsOf: fixtureURL)
        let fixtures = try JSONDecoder().decode(
            FixtureSuite.self,
            from: fixtureData
        )

        for fixture in fixtures.cases {
            let result = try tokenizer.decode(fixture.tokenIDs)
            if result != fixture.postProcessed {
                XCTFail(
                    "\(fixture.name): Swift=\(result) Python=\(fixture.postProcessed)"
                )
            }
        }
    }
}
