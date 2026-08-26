import Foundation
import XCTest
@testable import EqnSnap

final class UniMERNetTokenizerTests: XCTestCase {
    func testMatchesPythonByteLevelFixtures() throws {
        let root = repositoryRoot()
        let tokenizer = try UniMERNetTokenizer(
            contentsOf: root.appendingPathComponent(
                "EqnSnap/Resources/Models/UniMERNet/UniMERNetTokenizer.json"
            )
        )
        let fixtureData = try Data(
            contentsOf: root.appendingPathComponent(
                "EqnSnapTests/Fixtures/UniMERNet/tokenizer-fixtures.json"
            )
        )
        let fixture = try JSONDecoder().decode(
            UniMERNetTokenizerFixture.self,
            from: fixtureData
        )

        for testCase in fixture.cases {
            XCTAssertEqual(
                try tokenizer.decode(testCase.tokenIDs),
                testCase.decoded,
                testCase.name
            )
        }
    }

    func testRejectsOutOfRangeToken() throws {
        let tokenizer = try UniMERNetTokenizer(
            contentsOf: repositoryRoot().appendingPathComponent(
                "EqnSnap/Resources/Models/UniMERNet/UniMERNetTokenizer.json"
            )
        )

        XCTAssertThrowsError(try tokenizer.decode([50_000])) { error in
            XCTAssertEqual(
                error as? UniMERNetTokenizerError,
                .invalidTokenID(50_000)
            )
        }
    }

    private func repositoryRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}

private struct UniMERNetTokenizerFixture: Decodable {
    struct Case: Decodable {
        let name: String
        let tokenIDs: [Int32]
        let decoded: String
    }

    let cases: [Case]
}
