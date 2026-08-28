import Accelerate
import CoreGraphics
import CoreML
import Foundation

struct UniMERNetPreprocessingConfiguration: Sendable, Equatable {
    static let tinyV1 = UniMERNetPreprocessingConfiguration(
        canvasWidth: 672,
        canvasHeight: 192,
        cropThreshold: 200,
        normalizationMean: 0.7931,
        normalizationStandardDeviation: 0.1738
    )

    let canvasWidth: Int
    let canvasHeight: Int
    let cropThreshold: Double
    let normalizationMean: Float32
    let normalizationStandardDeviation: Float32
}

struct UniMERNetPreparedFormulaInput {
    let tensor: MLMultiArray
    let sourcePixelSize: CGSize
    let croppedPixelRect: CGRect
    let contentPixelSize: CGSize
    let canvasSize: CGSize
}

enum UniMERNetFormulaImagePreprocessorError: Error, Sendable, Equatable {
    case invalidImageDimensions
    case unableToCreateBitmapContext
    case imageResizeFailed(Int)
}

final class UniMERNetFormulaImagePreprocessor {
    private struct RGBAImage {
        let width: Int
        let height: Int
        let pixels: [UInt8]
    }

    private struct CroppedImage {
        let image: RGBAImage
        let rect: CGRect
    }

    private let configuration: UniMERNetPreprocessingConfiguration

    init(
        configuration: UniMERNetPreprocessingConfiguration = .tinyV1
    ) {
        self.configuration = configuration
    }

    func prepare(_ image: CGImage) throws -> UniMERNetPreparedFormulaInput {
        guard image.width > 0, image.height > 0 else {
            throw UniMERNetFormulaImagePreprocessorError.invalidImageDimensions
        }

        let source = try rgbaImage(from: image)
        let cropped = cropMargin(source)
        let firstSize = shortEdgeResizeSize(
            width: cropped.image.width,
            height: cropped.image.height,
            shortEdge: min(configuration.canvasWidth, configuration.canvasHeight)
        )
        let firstResize = try resize(
            cropped.image,
            width: firstSize.width,
            height: firstSize.height,
            highQuality: false
        )
        let contentSize = thumbnailSize(
            width: firstResize.width,
            height: firstResize.height
        )
        let content = try resize(
            firstResize,
            width: contentSize.width,
            height: contentSize.height,
            highQuality: true
        )
        let tensor = try makeTensor(content)

        return UniMERNetPreparedFormulaInput(
            tensor: tensor,
            sourcePixelSize: CGSize(width: source.width, height: source.height),
            croppedPixelRect: cropped.rect,
            contentPixelSize: CGSize(
                width: content.width,
                height: content.height
            ),
            canvasSize: CGSize(
                width: configuration.canvasWidth,
                height: configuration.canvasHeight
            )
        )
    }

    private func rgbaImage(from image: CGImage) throws -> RGBAImage {
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
            throw UniMERNetFormulaImagePreprocessorError
                .unableToCreateBitmapContext
        }
        return RGBAImage(width: image.width, height: image.height, pixels: pixels)
    }

    private func cropMargin(_ image: RGBAImage) -> CroppedImage {
        var grayscale = [UInt8](repeating: 0, count: image.width * image.height)
        var minimum = UInt8.max
        var maximum = UInt8.min
        for index in grayscale.indices {
            let component = index * 4
            let value = grayscaleValue(
                red: image.pixels[component],
                green: image.pixels[component + 1],
                blue: image.pixels[component + 2]
            )
            grayscale[index] = value
            minimum = min(minimum, value)
            maximum = max(maximum, value)
        }
        guard minimum != maximum else {
            return CroppedImage(
                image: image,
                rect: CGRect(x: 0, y: 0, width: image.width, height: image.height)
            )
        }

        let range = Double(Int(maximum) - Int(minimum))
        var minX = image.width
        var minY = image.height
        var maxX = -1
        var maxY = -1
        for y in 0..<image.height {
            for x in 0..<image.width {
                let value = grayscale[y * image.width + x]
                let normalized = Double(Int(value) - Int(minimum)) / range * 255
                if normalized < configuration.cropThreshold {
                    minX = min(minX, x)
                    minY = min(minY, y)
                    maxX = max(maxX, x)
                    maxY = max(maxY, y)
                }
            }
        }
        guard maxX >= minX, maxY >= minY else {
            return CroppedImage(
                image: image,
                rect: CGRect(x: 0, y: 0, width: image.width, height: image.height)
            )
        }

        let width = maxX - minX + 1
        let height = maxY - minY + 1
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            let sourceStart = ((minY + y) * image.width + minX) * 4
            let destinationStart = y * width * 4
            let byteCount = width * 4
            pixels.replaceSubrange(
                destinationStart..<(destinationStart + byteCount),
                with: image.pixels[sourceStart..<(sourceStart + byteCount)]
            )
        }
        return CroppedImage(
            image: RGBAImage(width: width, height: height, pixels: pixels),
            rect: CGRect(x: minX, y: minY, width: width, height: height)
        )
    }

    private func shortEdgeResizeSize(
        width: Int,
        height: Int,
        shortEdge: Int
    ) -> (width: Int, height: Int) {
        if width <= height {
            return (shortEdge, max(1, height * shortEdge / width))
        }
        return (max(1, width * shortEdge / height), shortEdge)
    }

    private func thumbnailSize(width: Int, height: Int)
        -> (width: Int, height: Int)
    {
        let scale = min(
            1,
            min(
                Double(configuration.canvasWidth) / Double(width),
                Double(configuration.canvasHeight) / Double(height)
            )
        )
        return (
            max(1, Int((Double(width) * scale).rounded(.toNearestOrEven))),
            max(1, Int((Double(height) * scale).rounded(.toNearestOrEven)))
        )
    }

    private func resize(
        _ image: RGBAImage,
        width: Int,
        height: Int,
        highQuality: Bool
    ) throws -> RGBAImage {
        if image.width == width, image.height == height {
            return image
        }
        if !highQuality {
            return resizeBilinear(image, width: width, height: height)
        }
        var sourcePixels = image.pixels
        var destinationPixels = [UInt8](
            repeating: 0,
            count: width * height * 4
        )
        let error = sourcePixels.withUnsafeMutableBytes { sourceBytes in
            destinationPixels.withUnsafeMutableBytes { destinationBytes in
                var sourceBuffer = vImage_Buffer(
                    data: sourceBytes.baseAddress,
                    height: vImagePixelCount(image.height),
                    width: vImagePixelCount(image.width),
                    rowBytes: image.width * 4
                )
                var destinationBuffer = vImage_Buffer(
                    data: destinationBytes.baseAddress,
                    height: vImagePixelCount(height),
                    width: vImagePixelCount(width),
                    rowBytes: width * 4
                )
                return vImageScale_ARGB8888(
                    &sourceBuffer,
                    &destinationBuffer,
                    nil,
                    vImage_Flags(kvImageHighQualityResampling)
                )
            }
        }
        guard error == kvImageNoError else {
            throw UniMERNetFormulaImagePreprocessorError.imageResizeFailed(error)
        }
        return RGBAImage(width: width, height: height, pixels: destinationPixels)
    }

    private func resizeBilinear(
        _ image: RGBAImage,
        width: Int,
        height: Int
    ) -> RGBAImage {
        let scaleX = Double(image.width) / Double(width)
        let scaleY = Double(image.height) / Double(height)
        var pixels = [UInt8](repeating: 0, count: width * height * 4)

        for y in 0..<height {
            let sourceY = min(
                Double(image.height - 1),
                max(0, (Double(y) + 0.5) * scaleY - 0.5)
            )
            let top = Int(sourceY.rounded(.down))
            let bottom = min(top + 1, image.height - 1)
            let verticalWeight = sourceY - Double(top)

            for x in 0..<width {
                let sourceX = min(
                    Double(image.width - 1),
                    max(0, (Double(x) + 0.5) * scaleX - 0.5)
                )
                let left = Int(sourceX.rounded(.down))
                let right = min(left + 1, image.width - 1)
                let horizontalWeight = sourceX - Double(left)

                for channel in 0..<4 {
                    let topLeft = Double(
                        image.pixels[(top * image.width + left) * 4 + channel]
                    )
                    let topRight = Double(
                        image.pixels[(top * image.width + right) * 4 + channel]
                    )
                    let bottomLeft = Double(
                        image.pixels[(bottom * image.width + left) * 4 + channel]
                    )
                    let bottomRight = Double(
                        image.pixels[(bottom * image.width + right) * 4 + channel]
                    )
                    let topValue = topLeft
                        + (topRight - topLeft) * horizontalWeight
                    let bottomValue = bottomLeft
                        + (bottomRight - bottomLeft) * horizontalWeight
                    let value = topValue
                        + (bottomValue - topValue) * verticalWeight
                    pixels[(y * width + x) * 4 + channel] = UInt8(
                        clamping: Int(value.rounded(.toNearestOrEven))
                    )
                }
            }
        }
        return RGBAImage(width: width, height: height, pixels: pixels)
    }

    private func makeTensor(_ content: RGBAImage) throws -> MLMultiArray {
        let tensor = try MLMultiArray(
            shape: [
                1,
                3,
                NSNumber(value: configuration.canvasHeight),
                NSNumber(value: configuration.canvasWidth),
            ],
            dataType: .float32
        )
        let pointer = tensor.dataPointer.assumingMemoryBound(to: Float32.self)
        for index in 0..<tensor.count {
            pointer[index] = -configuration.normalizationMean
                / configuration.normalizationStandardDeviation
        }

        let originX = (configuration.canvasWidth - content.width) / 2
        let originY = (configuration.canvasHeight - content.height) / 2
        let channelStride = tensor.strides[1].intValue
        let rowStride = tensor.strides[2].intValue
        let columnStride = tensor.strides[3].intValue
        for y in 0..<content.height {
            for x in 0..<content.width {
                let component = (y * content.width + x) * 4
                let grayscale = grayscaleValue(
                    red: content.pixels[component],
                    green: content.pixels[component + 1],
                    blue: content.pixels[component + 2]
                )
                let normalized = (
                    Float32(grayscale) / 255 - configuration.normalizationMean
                ) / configuration.normalizationStandardDeviation
                let canvasOffset = (originY + y) * rowStride
                    + (originX + x) * columnStride
                for channel in 0..<3 {
                    pointer[channel * channelStride + canvasOffset] = normalized
                }
            }
        }
        return tensor
    }

    private func grayscaleValue(
        red: UInt8,
        green: UInt8,
        blue: UInt8
    ) -> UInt8 {
        let weightedRed = Int(red) * 19_595
        let weightedGreen = Int(green) * 38_470
        let weightedBlue = Int(blue) * 7_471
        let rounded = weightedRed + weightedGreen + weightedBlue + 32_768
        return UInt8(min(255, rounded >> 16))
    }
}
