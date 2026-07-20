import Foundation

nonisolated struct Pix2TexTokenizerSpecialTokens: Decodable, Sendable, Equatable {
    let pad: Int
    let bos: Int
    let eos: Int
}

nonisolated struct Pix2TexTokenizerResource: Decodable, Sendable, Equatable {
    let schemaVersion: Int
    let modelID: String
    let vocabularySize: Int
    let decoderOutputSize: Int
    let specialTokens: Pix2TexTokenizerSpecialTokens
    let tokensByID: [String]
}

nonisolated enum Pix2TexTokenizerError: Error, Sendable, Equatable {
    case unsupportedSchemaVersion(Int)
    case invalidVocabularySize
    case invalidTokenID(Int32)
}

nonisolated final class Pix2TexTokenizer {
    let resource: Pix2TexTokenizerResource

    init(data: Data) throws {
        let resource = try JSONDecoder().decode(
            Pix2TexTokenizerResource.self,
            from: data
        )
        guard resource.schemaVersion == 1 else {
            throw Pix2TexTokenizerError.unsupportedSchemaVersion(
                resource.schemaVersion
            )
        }
        guard resource.vocabularySize == resource.tokensByID.count else {
            throw Pix2TexTokenizerError.invalidVocabularySize
        }
        self.resource = resource
    }

    convenience init(contentsOf url: URL) throws {
        try self.init(data: Data(contentsOf: url))
    }

    func decode(_ tokenIDs: [Int32]) throws -> String {
        var decoded = ""
        decoded.reserveCapacity(tokenIDs.count * 2)
        for tokenID in tokenIDs {
            guard tokenID >= 0,
                  Int(tokenID) < resource.tokensByID.count
            else {
                throw Pix2TexTokenizerError.invalidTokenID(tokenID)
            }
            decoded += resource.tokensByID[Int(tokenID)]
        }
        decoded = decoded
            .replacingOccurrences(of: "Ġ", with: " ")
            .replacingOccurrences(of: "[EOS]", with: "")
            .replacingOccurrences(of: "[BOS]", with: "")
            .replacingOccurrences(of: "[PAD]", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return Pix2TexLatexPostProcessor.process(decoded)
    }
}

nonisolated enum Pix2TexLatexPostProcessor {
    static func process(_ input: String) -> String {
        var text = compactTextCommands(input)
        // Python's `\w` treats Unicode letters and numbers as word
        // characters. ICU's `\W` differs for characters such as `½`, so
        // spell out the equivalent class instead of relying on `\W`.
        let noLetter = #"(?:[^\p{L}\p{N}]|\p{Nd})"#
        let patterns = [
            #"(?!\\ )("# + noLetter + #")\s+?("# + noLetter + #")"#,
            #"(?!\\ )("# + noLetter + #")\s+?([a-zA-Z])"#,
            #"([a-zA-Z])\s+?("# + noLetter + #")"#,
        ]

        while true {
            let previous = text
            for pattern in patterns {
                text = replacing(
                    pattern: pattern,
                    in: text,
                    template: "$1$2"
                )
            }
            if text == previous {
                return text
            }
        }
    }

    private static func compactTextCommands(_ input: String) -> String {
        let pattern = #"(\\(operatorname|mathrm|text|mathbf)\s?\*? \{.*?\})"#
        let expression = try! NSRegularExpression(pattern: pattern)
        var output = input
        let matches = expression.matches(
            in: output,
            range: NSRange(output.startIndex..., in: output)
        )
        for match in matches.reversed() {
            let value = (output as NSString).substring(with: match.range)
            output = (output as NSString).replacingCharacters(
                in: match.range,
                with: value.replacingOccurrences(of: " ", with: "")
            )
        }
        return output
    }

    private static func replacing(
        pattern: String,
        in input: String,
        template: String
    ) -> String {
        let expression = try! NSRegularExpression(pattern: pattern)
        return expression.stringByReplacingMatches(
            in: input,
            range: NSRange(input.startIndex..., in: input),
            withTemplate: template
        )
    }
}
