import Accelerate
import CoreGraphics
import CoreML
import Foundation

struct Pix2TexPreprocessingConfiguration: Sendable, Equatable {
    static let pix2texV1 = Pix2TexPreprocessingConfiguration(
        foregroundThreshold: 128,
        tightCropThreshold: 250,
        canvasMultiple: 32,
        maximumCanvasWidth: 672,
        maximumCanvasHeight: 64,
        normalizationMean: 0.7931,
        normalizationStandardDeviation: 0.1738,
        strokeProfilePolicy: .pix2texV1
    )

    let foregroundThreshold: Double
    let tightCropThreshold: UInt8
    let canvasMultiple: Int
    let maximumCanvasWidth: Int
    let maximumCanvasHeight: Int
    let normalizationMean: Float32
    let normalizationStandardDeviation: Float32
    let strokeProfilePolicy: FormulaStrokeProfilePolicy
}

struct PreparedFormulaInput {
    let tensor: MLMultiArray
    let sourcePixelSize: CGSize
    let croppedPixelRect: CGRect
    let targetForegroundHeight: Int
    let actualForegroundSize: CGSize
    let canvasSize: CGSize
    let strokeMetrics: FormulaStrokeMetrics
}

enum Pix2TexFormulaImagePreprocessorError: Error, Equatable {
    case invalidImageDimensions
    case unableToCreateBitmapContext
    case noVisibleForeground
    case preparedShapeExceedsModelLimit(width: Int, height: Int)
    case imageResizeFailed(Int)
}

final class Pix2TexFormulaImagePreprocessor {
    private struct CroppedImage {
        let image: GrayscaleImage
        let rect: CGRect
    }

    private let configuration: Pix2TexPreprocessingConfiguration

    init(
        configuration: Pix2TexPreprocessingConfiguration = .pix2texV1
    ) {
        self.configuration = configuration
    }

    func prepare(_ image: CGImage) throws -> PreparedFormulaInput {
        guard image.width > 0, image.height > 0 else {
            throw Pix2TexFormulaImagePreprocessorError.invalidImageDimensions
        }

        let cropped = try normalizeAndCrop(image)
        let metrics = try FormulaStrokeProfileAnalyzer.analyze(cropped.image)
        let targetHeight = configuration.strokeProfilePolicy
            .targetForegroundHeight(for: metrics)
        let prepared = try resizeAndPad(
            cropped.image,
            targetForegroundHeight: targetHeight
        )
        let tensor = try makeTensor(
            pixels: prepared.pixels,
            width: prepared.canvasWidth,
            height: prepared.canvasHeight
        )

        return PreparedFormulaInput(
            tensor: tensor,
            sourcePixelSize: CGSize(width: image.width, height: image.height),
            croppedPixelRect: cropped.rect,
            targetForegroundHeight: targetHeight,
            actualForegroundSize: CGSize(
                width: prepared.foregroundWidth,
                height: prepared.foregroundHeight
            ),
            canvasSize: CGSize(
                width: prepared.canvasWidth,
                height: prepared.canvasHeight
            ),
            strokeMetrics: metrics
        )
    }

    private func normalizeAndCrop(_ image: CGImage) throws -> CroppedImage {
        let width = image.width
        let height = image.height
        let rgba = try rgbaPixels(from: image)
        let alphaIsConstant = alphaChannelIsConstant(
            rgba,
            pixelCount: width * height
        )

        var source = [UInt8](repeating: 0, count: width * height)
        for index in source.indices {
            let component = index * 4
            if alphaIsConstant {
                let red = Int(rgba[component])
                let green = Int(rgba[component + 1])
                let blue = Int(rgba[component + 2])
                source[index] = UInt8(
                    (red * 299 + green * 587 + blue * 114 + 500) / 1_000
                )
            } else {
                source[index] = 255 - rgba[component + 3]
            }
        }

        guard let minimum = source.min(),
              let maximum = source.max(),
              minimum != maximum
        else {
            throw Pix2TexFormulaImagePreprocessorError.noVisibleForeground
        }

        let range = Double(Int(maximum) - Int(minimum))
        var normalized = [Double](repeating: 0, count: source.count)
        var normalizedSum = 0.0
        for index in source.indices {
            let value = Double(Int(source[index]) - Int(minimum)) / range * 255
            normalized[index] = value
            normalizedSum += value
        }
        let lightBackground = normalizedSum / Double(normalized.count)
            > configuration.foregroundThreshold

        guard let initialBounds = bounds(
            width: width,
            height: height,
            where: { value in
                lightBackground
                    ? value < configuration.foregroundThreshold
                    : value > configuration.foregroundThreshold
            },
            valueAt: { normalized[$0] }
        ) else {
            throw Pix2TexFormulaImagePreprocessorError.noVisibleForeground
        }

        let initialWidth = initialBounds.maxX - initialBounds.minX + 1
        let initialHeight = initialBounds.maxY - initialBounds.minY + 1
        var initiallyCropped = [UInt8](
            repeating: 255,
            count: initialWidth * initialHeight
        )
        for y in 0..<initialHeight {
            for x in 0..<initialWidth {
                let sourceIndex = (initialBounds.minY + y) * width
                    + initialBounds.minX + x
                let normalizedValue = lightBackground
                    ? normalized[sourceIndex]
                    : 255 - normalized[sourceIndex]
                initiallyCropped[y * initialWidth + x] = UInt8(
                    max(0, min(255, normalizedValue))
                )
            }
        }

        guard let tightBounds = bounds(
            width: initialWidth,
            height: initialHeight,
            where: { $0 < configuration.tightCropThreshold },
            valueAt: { initiallyCropped[$0] }
        ) else {
            throw Pix2TexFormulaImagePreprocessorError.noVisibleForeground
        }

        let tightWidth = tightBounds.maxX - tightBounds.minX + 1
        let tightHeight = tightBounds.maxY - tightBounds.minY + 1
        var pixels = [UInt8](repeating: 255, count: tightWidth * tightHeight)
        for y in 0..<tightHeight {
            let sourceStart = (tightBounds.minY + y) * initialWidth
                + tightBounds.minX
            let destinationStart = y * tightWidth
            pixels.replaceSubrange(
                destinationStart..<(destinationStart + tightWidth),
                with: initiallyCropped[
                    sourceStart..<(sourceStart + tightWidth)
                ]
            )
        }

        let originX = initialBounds.minX + tightBounds.minX
        let originY = initialBounds.minY + tightBounds.minY
        return CroppedImage(
            image: try GrayscaleImage(
                width: tightWidth,
                height: tightHeight,
                pixels: pixels
            ),
            rect: CGRect(
                x: originX,
                y: originY,
                width: tightWidth,
                height: tightHeight
            )
        )
    }

    private func rgbaPixels(from image: CGImage) throws -> [UInt8] {
        let bytesPerRow = image.width * 4
        var pixels = [UInt8](
            repeating: 0,
            count: bytesPerRow * image.height
        )
        let created = pixels.withUnsafeMutableBytes { bytes -> Bool in
            guard let baseAddress = bytes.baseAddress,
                  let context = CGContext(
                    data: baseAddress,
                    width: image.width,
                    height: image.height,
                    bitsPerComponent: 8,
                    bytesPerRow: bytesPerRow,
                    space: image.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
                        | CGImageAlphaInfo.premultipliedLast.rawValue
                  )
            else {
                return false
            }
            context.draw(
                image,
                in: CGRect(x: 0, y: 0, width: image.width, height: image.height)
            )
            return true
        }
        guard created else {
            throw Pix2TexFormulaImagePreprocessorError
                .unableToCreateBitmapContext
        }
        return pixels
    }

    private func alphaChannelIsConstant(
        _ rgba: [UInt8],
        pixelCount: Int
    ) -> Bool {
        let firstAlpha = rgba[3]
        for index in 1..<pixelCount where rgba[index * 4 + 3] != firstAlpha {
            return false
        }
        return true
    }

    private func bounds<T>(
        width: Int,
        height: Int,
        where isForeground: (T) -> Bool,
        valueAt: (Int) -> T
    ) -> (minX: Int, minY: Int, maxX: Int, maxY: Int)? {
        var minX = width
        var minY = height
        var maxX = -1
        var maxY = -1
        for y in 0..<height {
            for x in 0..<width where isForeground(valueAt(y * width + x)) {
                minX = min(minX, x)
                minY = min(minY, y)
                maxX = max(maxX, x)
                maxY = max(maxY, y)
            }
        }
        guard maxX >= minX, maxY >= minY else {
            return nil
        }
        return (minX, minY, maxX, maxY)
    }

    private func resizeAndPad(
        _ image: GrayscaleImage,
        targetForegroundHeight: Int
    ) throws -> (
        pixels: [UInt8],
        foregroundWidth: Int,
        foregroundHeight: Int,
        canvasWidth: Int,
        canvasHeight: Int
    ) {
        let targetScale = Double(targetForegroundHeight) / Double(image.height)
        let fitScale = min(
            Double(configuration.maximumCanvasWidth) / Double(image.width),
            Double(configuration.maximumCanvasHeight) / Double(image.height)
        )
        let scale = min(targetScale, fitScale)
        let foregroundWidth = max(
            1,
            Int((Double(image.width) * scale).rounded(.toNearestOrEven))
        )
        let foregroundHeight = max(
            1,
            Int((Double(image.height) * scale).rounded(.toNearestOrEven))
        )
        let canvasWidth = nextCanvasDimension(foregroundWidth)
        let canvasHeight = nextCanvasDimension(foregroundHeight)
        guard canvasWidth <= configuration.maximumCanvasWidth,
              canvasHeight <= configuration.maximumCanvasHeight
        else {
            throw Pix2TexFormulaImagePreprocessorError
                .preparedShapeExceedsModelLimit(
                    width: canvasWidth,
                    height: canvasHeight
                )
        }

        let resized = try resize(
            image,
            width: foregroundWidth,
            height: foregroundHeight
        )
        var canvas = [UInt8](
            repeating: 255,
            count: canvasWidth * canvasHeight
        )
        for y in 0..<foregroundHeight {
            let sourceStart = y * foregroundWidth
            let destinationStart = y * canvasWidth
            canvas.replaceSubrange(
                destinationStart..<(destinationStart + foregroundWidth),
                with: resized[sourceStart..<(sourceStart + foregroundWidth)]
            )
        }
        return (
            canvas,
            foregroundWidth,
            foregroundHeight,
            canvasWidth,
            canvasHeight
        )
    }

    private func nextCanvasDimension(_ value: Int) -> Int {
        max(
            configuration.canvasMultiple,
            ((value + configuration.canvasMultiple - 1)
                / configuration.canvasMultiple) * configuration.canvasMultiple
        )
    }

    private func resize(
        _ image: GrayscaleImage,
        width: Int,
        height: Int
    ) throws -> [UInt8] {
        if width == image.width, height == image.height {
            return image.pixels
        }

        var sourcePixels = image.pixels
        var destinationPixels = [UInt8](repeating: 255, count: width * height)
        let error = sourcePixels.withUnsafeMutableBytes { sourceBytes in
            destinationPixels.withUnsafeMutableBytes { destinationBytes in
                var sourceBuffer = vImage_Buffer(
                    data: sourceBytes.baseAddress,
                    height: vImagePixelCount(image.height),
                    width: vImagePixelCount(image.width),
                    rowBytes: image.width
                )
                var destinationBuffer = vImage_Buffer(
                    data: destinationBytes.baseAddress,
                    height: vImagePixelCount(height),
                    width: vImagePixelCount(width),
                    rowBytes: width
                )
                return vImageScale_Planar8(
                    &sourceBuffer,
                    &destinationBuffer,
                    nil,
                    vImage_Flags(kvImageHighQualityResampling)
                )
            }
        }
        guard error == kvImageNoError else {
            throw Pix2TexFormulaImagePreprocessorError.imageResizeFailed(error)
        }
        return destinationPixels
    }

    private func makeTensor(
        pixels: [UInt8],
        width: Int,
        height: Int
    ) throws -> MLMultiArray {
        let tensor = try MLMultiArray(
            shape: [1, 1, NSNumber(value: height), NSNumber(value: width)],
            dataType: .float32
        )
        let pointer = tensor.dataPointer.assumingMemoryBound(to: Float32.self)
        let rowStride = tensor.strides[2].intValue
        let columnStride = tensor.strides[3].intValue
        for y in 0..<height {
            for x in 0..<width {
                let value = Float32(pixels[y * width + x]) / 255
                pointer[y * rowStride + x * columnStride] = (
                    value - configuration.normalizationMean
                ) / configuration.normalizationStandardDeviation
            }
        }
        return tensor
    }
}
