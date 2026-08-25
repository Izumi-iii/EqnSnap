import Foundation

struct UniMERNetTokenizerSpecialTokens: Decodable, Sendable, Equatable {
    let bos: Int32
    let pad: Int32
    let eos: Int32
    let unk: Int32
}

struct UniMERNetTokenizerResource: Decodable, Sendable, Equatable {
    let schemaVersion: Int
    let modelID: String
    let vocabularySize: Int
    let specialTokens: UniMERNetTokenizerSpecialTokens
    let specialTokenIDs: [Int32]
    let tokensByID: [String]
}

enum UniMERNetTokenizerError: Error, Sendable, Equatable {
    case unsupportedSchemaVersion(Int)
    case invalidVocabularySize
    case incompatibleSpecialTokens
    case invalidTokenID(Int32)
    case invalidByteLevelScalar(UInt32)
}

final class UniMERNetTokenizer {
    let resource: UniMERNetTokenizerResource

    private let specialTokenIDs: Set<Int32>
    private let byteDecoder: [UnicodeScalar: UInt8]

    init(data: Data) throws {
        let resource = try JSONDecoder().decode(
            UniMERNetTokenizerResource.self,
            from: data
        )
        guard resource.schemaVersion == 1 else {
            throw UniMERNetTokenizerError.unsupportedSchemaVersion(
                resource.schemaVersion
            )
        }
        guard resource.vocabularySize == resource.tokensByID.count else {
            throw UniMERNetTokenizerError.invalidVocabularySize
        }
        guard resource.vocabularySize == 50_000,
              resource.specialTokens.bos == 0,
              resource.specialTokens.pad == 1,
              resource.specialTokens.eos == 2,
              resource.specialTokens.unk == 3
        else {
            throw UniMERNetTokenizerError.incompatibleSpecialTokens
        }
        self.resource = resource
        self.specialTokenIDs = Set(resource.specialTokenIDs)
        self.byteDecoder = Self.makeByteDecoder()
    }

    convenience init(contentsOf url: URL) throws {
        try self.init(data: Data(contentsOf: url))
    }

    func decode(_ tokenIDs: [Int32]) throws -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(tokenIDs.count * 2)
        for tokenID in tokenIDs {
            guard tokenID >= 0,
                  Int(tokenID) < resource.tokensByID.count
            else {
                throw UniMERNetTokenizerError.invalidTokenID(tokenID)
            }
            guard !specialTokenIDs.contains(tokenID) else {
                continue
            }
            for scalar in resource.tokensByID[Int(tokenID)].unicodeScalars {
                guard let byte = byteDecoder[scalar] else {
                    throw UniMERNetTokenizerError.invalidByteLevelScalar(
                        scalar.value
                    )
                }
                bytes.append(byte)
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    private static func makeByteDecoder() -> [UnicodeScalar: UInt8] {
        var bytes = Array(UInt8(33)...UInt8(126))
        bytes.append(contentsOf: Array(UInt8(161)...UInt8(172)))
        bytes.append(contentsOf: Array(UInt8(174)...UInt8(255)))
        let directBytes = Set(bytes)
        var scalarValues = bytes.map(UInt32.init)
        var extraScalar: UInt32 = 256
        for value in 0...255 {
            let byte = UInt8(value)
            guard !directBytes.contains(byte) else { continue }
            bytes.append(byte)
            scalarValues.append(extraScalar)
            extraScalar += 1
        }

        var decoder: [UnicodeScalar: UInt8] = [:]
        for (byte, scalarValue) in zip(bytes, scalarValues) {
            decoder[UnicodeScalar(scalarValue)!] = byte
        }
        return decoder
    }
}
