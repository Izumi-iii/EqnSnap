//
//  EqnSnapTests.swift
//  EqnSnapTests
//
//  Created by wangjie on 2026/7/16.
//

import Foundation
import Testing
@testable import EqnSnap

struct EqnSnapTests {
    private struct FixtureSuite: Decodable {
        let policy: Policy
        let cases: [Fixture]
    }

    private struct Policy: Decodable {
        let relativeStrokeWidthThreshold: Double
        let smallTargetHeight: Int
        let largeTargetHeight: Int
    }

    private struct Fixture: Decodable {
        let id: String
        let image: String
        let width: Int
        let height: Int
        let otsuThreshold: UInt8
        let foregroundArea: Int
        let foregroundPerimeter: Int
        let estimatedStrokeWidth: Double
        let relativeStrokeWidth: Double
        let selectedTargetHeight: Int
    }

    @Test func strokeProfileMatchesPythonFixtures() throws {
        let fixtureDirectory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/StrokeProfiles")
        let data = try Data(
            contentsOf: fixtureDirectory.appendingPathComponent("fixtures.json")
        )
        let suite = try JSONDecoder().decode(FixtureSuite.self, from: data)
        let policy = FormulaStrokeProfilePolicy(
            relativeStrokeWidthThreshold: suite.policy.relativeStrokeWidthThreshold,
            smallTargetHeight: suite.policy.smallTargetHeight,
            largeTargetHeight: suite.policy.largeTargetHeight
        )

        #expect(policy == .pix2texV1)
        for fixture in suite.cases {
            let image = try loadPGM(
                fixtureDirectory.appendingPathComponent(fixture.image),
                width: fixture.width,
                height: fixture.height
            )
            let metrics = try FormulaStrokeProfileAnalyzer.analyze(image)

            #expect(metrics.otsuThreshold == fixture.otsuThreshold)
            #expect(metrics.foregroundArea == fixture.foregroundArea)
            #expect(metrics.foregroundPerimeter == fixture.foregroundPerimeter)
            #expect(
                abs(metrics.estimatedStrokeWidth - fixture.estimatedStrokeWidth)
                    < 1e-12
            )
            #expect(
                abs(metrics.relativeStrokeWidth - fixture.relativeStrokeWidth)
                    < 1e-12
            )
            #expect(
                policy.targetForegroundHeight(for: metrics)
                    == fixture.selectedTargetHeight
            )
        }
    }

    private func loadPGM(_ url: URL, width: Int, height: Int) throws -> GrayscaleImage {
        let data = try Data(contentsOf: url)
        let header = Data("P5\n\(width) \(height)\n255\n".utf8)
        #expect(data.starts(with: header))
        return try GrayscaleImage(
            width: width,
            height: height,
            pixels: Array(data.dropFirst(header.count))
        )
    }
}
