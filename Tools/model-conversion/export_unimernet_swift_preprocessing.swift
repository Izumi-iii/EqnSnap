import CoreGraphics
import CoreML
import Foundation
import ImageIO

enum ExportError: Error, CustomStringConvertible {
    case invalidArguments
    case unableToLoadImage(String)
    case unsupportedTensorShape([Int])

    var description: String {
        switch self {
        case .invalidArguments:
            return "Usage: export-unimernet-preprocessing <image> <tensor.raw> <metadata.json>"
        case let .unableToLoadImage(path):
            return "Unable to load image: \(path)"
        case let .unsupportedTensorShape(shape):
            return "Expected a four-dimensional tensor, got \(shape)"
        }
    }
}

@main
struct UniMERNetSwiftPreprocessingExporter {
    static func main() throws {
        guard CommandLine.arguments.count == 4 else {
            throw ExportError.invalidArguments
        }

        let imageURL = URL(fileURLWithPath: CommandLine.arguments[1])
        let tensorURL = URL(fileURLWithPath: CommandLine.arguments[2])
        let metadataURL = URL(fileURLWithPath: CommandLine.arguments[3])
        guard let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw ExportError.unableToLoadImage(imageURL.path)
        }

        let prepared = try UniMERNetFormulaImagePreprocessor().prepare(image)
        let tensor = prepared.tensor
        let shape = tensor.shape.map(\.intValue)
        guard shape.count == 4 else {
            throw ExportError.unsupportedTensorShape(shape)
        }

        let strides = tensor.strides.map(\.intValue)
        let pointer = tensor.dataPointer.assumingMemoryBound(to: Float32.self)
        var values = [Float32]()
        values.reserveCapacity(tensor.count)
        for batch in 0..<shape[0] {
            for channel in 0..<shape[1] {
                for row in 0..<shape[2] {
                    for column in 0..<shape[3] {
                        let offset = batch * strides[0]
                            + channel * strides[1]
                            + row * strides[2]
                            + column * strides[3]
                        values.append(pointer[offset])
                    }
                }
            }
        }
        let tensorData = values.withUnsafeBytes { Data($0) }
        try tensorData.write(to: tensorURL, options: .atomic)

        let metadata: [String: Any] = [
            "source_pixel_size": sizeDictionary(prepared.sourcePixelSize),
            "cropped_pixel_rect": rectDictionary(prepared.croppedPixelRect),
            "content_pixel_size": sizeDictionary(prepared.contentPixelSize),
            "canvas_size": sizeDictionary(prepared.canvasSize),
            "tensor_shape": shape,
            "tensor_strides": strides,
            "tensor_data_type": "float32",
            "tensor_byte_order": "little_endian",
        ]
        let json = try JSONSerialization.data(
            withJSONObject: metadata,
            options: [.prettyPrinted, .sortedKeys]
        )
        try json.write(to: metadataURL, options: .atomic)
    }

    private static func sizeDictionary(_ size: CGSize) -> [String: Double] {
        ["width": size.width, "height": size.height]
    }

    private static func rectDictionary(_ rect: CGRect) -> [String: Double] {
        [
            "x": rect.origin.x,
            "y": rect.origin.y,
            "width": rect.size.width,
            "height": rect.size.height,
        ]
    }
}
