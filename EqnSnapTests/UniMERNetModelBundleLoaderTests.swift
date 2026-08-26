import CoreML
import XCTest
@testable import EqnSnap

final class UniMERNetModelBundleLoaderTests: XCTestCase {
    func testLoadsAndValidatesConvertedModels() throws {
        let root = repositoryRoot()
        let artifacts = root.appendingPathComponent(
            "Tools/model-conversion/artifacts"
        )
        let encoderURL = artifacts.appendingPathComponent(
            "UniMERNetTinyEncoder-FP16.mlpackage"
        )
        let decoderURL = artifacts.appendingPathComponent(
            "UniMERNetTinyDecoder-CachedStep-SelfKV-FP16.mlpackage"
        )
        guard FileManager.default.fileExists(atPath: encoderURL.path),
              FileManager.default.fileExists(atPath: decoderURL.path)
        else {
            throw XCTSkip("Converted UniMERNet models are not available")
        }

        let loaded = try UniMERNetModelBundleLoader(
            computeUnits: .cpuOnly
        ).load(
            encoderURL: encoderURL,
            decoderURL: decoderURL,
            tokenizerURL: root.appendingPathComponent(
                "EqnSnap/Resources/Models/UniMERNet/UniMERNetTokenizer.json"
            )
        )

        XCTAssertEqual(
            loaded.encoder.modelDescription
                .inputDescriptionsByName["pixel_values"]?
                .multiArrayConstraint?.shape.map(\.intValue),
            [1, 3, 192, 672]
        )
        XCTAssertEqual(loaded.tokenizer.resource.vocabularySize, 50_000)
    }

    func testReportsMissingDirectoryResource() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertThrowsError(
            try UniMERNetModelBundleLoader().load(from: directory)
        ) { error in
            XCTAssertEqual(
                error as? UniMERNetModelBundleLoaderError,
                .missingResource("UniMERNetTinyEncoder-FP16")
            )
        }
    }

    private func repositoryRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}
