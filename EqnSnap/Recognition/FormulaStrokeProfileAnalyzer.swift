import Foundation

nonisolated struct GrayscaleImage: Sendable, Equatable {
    let width: Int
    let height: Int
    let pixels: [UInt8]

    init(width: Int, height: Int, pixels: [UInt8]) throws {
        guard width > 0, height > 0, pixels.count == width * height else {
            throw FormulaStrokeProfileError.invalidImageDimensions
        }
        self.width = width
        self.height = height
        self.pixels = pixels
    }
}

nonisolated enum FormulaStrokeProfileError: Error, Equatable {
    case invalidImageDimensions
    case noMeasurableForeground
}

nonisolated struct FormulaStrokeMetrics: Sendable, Equatable {
    let otsuThreshold: UInt8
    let foregroundArea: Int
    let foregroundPerimeter: Int
    let estimatedStrokeWidth: Double
    let relativeStrokeWidth: Double
}

nonisolated struct FormulaStrokeProfilePolicy: Sendable, Equatable {
    static let pix2texV1 = FormulaStrokeProfilePolicy(
        relativeStrokeWidthThreshold: 0.03137623705730272,
        smallTargetHeight: 24,
        largeTargetHeight: 40
    )

    let relativeStrokeWidthThreshold: Double
    let smallTargetHeight: Int
    let largeTargetHeight: Int

    func targetForegroundHeight(for metrics: FormulaStrokeMetrics) -> Int {
        metrics.relativeStrokeWidth <= relativeStrokeWidthThreshold
            ? largeTargetHeight
            : smallTargetHeight
    }
}

nonisolated enum FormulaStrokeProfileAnalyzer {
    static func analyze(_ image: GrayscaleImage) throws -> FormulaStrokeMetrics {
        let threshold = otsuThreshold(pixels: image.pixels)
        var foreground = [UInt8](repeating: 0, count: image.pixels.count)
        var area = 0

        for index in image.pixels.indices where image.pixels[index] <= threshold {
            foreground[index] = 1
            area += 1
        }

        let perimeter = gridPerimeter(
            foreground: foreground,
            width: image.width,
            height: image.height
        )
        guard area > 0, perimeter > 0 else {
            throw FormulaStrokeProfileError.noMeasurableForeground
        }

        let strokeWidth = 2.0 * Double(area) / Double(perimeter)
        return FormulaStrokeMetrics(
            otsuThreshold: threshold,
            foregroundArea: area,
            foregroundPerimeter: perimeter,
            estimatedStrokeWidth: strokeWidth,
            relativeStrokeWidth: strokeWidth / Double(image.height)
        )
    }

    private static func otsuThreshold(pixels: [UInt8]) -> UInt8 {
        var histogram = [Int64](repeating: 0, count: 256)
        for pixel in pixels {
            histogram[Int(pixel)] += 1
        }

        let totalCount = Int64(pixels.count)
        var totalSum: Int64 = 0
        for level in 0..<256 {
            totalSum += Int64(level) * histogram[level]
        }

        var leftCount: Int64 = 0
        var leftSum: Int64 = 0
        var bestScore = -Double.infinity
        var bestThreshold = 0

        for threshold in 0..<256 {
            let count = histogram[threshold]
            leftCount += count
            leftSum += Int64(threshold) * count
            let rightCount = totalCount - leftCount
            guard leftCount > 0, rightCount > 0 else {
                continue
            }

            let difference = totalSum * leftCount - leftSum * totalCount
            let score = Double(difference) * Double(difference)
                / Double(leftCount * rightCount)
            if score > bestScore {
                bestScore = score
                bestThreshold = threshold
            }
        }

        return UInt8(bestThreshold)
    }

    private static func gridPerimeter(
        foreground: [UInt8],
        width: Int,
        height: Int
    ) -> Int {
        var perimeter = 0
        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                guard foreground[index] != 0 else {
                    continue
                }
                if y == 0 || foreground[index - width] == 0 { perimeter += 1 }
                if y == height - 1 || foreground[index + width] == 0 { perimeter += 1 }
                if x == 0 || foreground[index - 1] == 0 { perimeter += 1 }
                if x == width - 1 || foreground[index + 1] == 0 { perimeter += 1 }
            }
        }
        return perimeter
    }
}
