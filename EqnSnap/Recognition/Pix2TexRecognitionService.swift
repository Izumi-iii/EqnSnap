import CoreGraphics

nonisolated struct CapturedFormulaImage: @unchecked Sendable {
    let image: CGImage
}

actor Pix2TexRecognitionService {
    private var recognizer: Pix2TexScreenshotRecognizer?

    func recognize(_ input: CapturedFormulaImage) async throws
        -> Pix2TexRecognitionOutput
    {
        try Task.checkCancellation()
        let recognizer: Pix2TexScreenshotRecognizer
        if let loaded = self.recognizer {
            recognizer = loaded
        } else {
            let loaded = try Pix2TexScreenshotRecognizer()
            self.recognizer = loaded
            recognizer = loaded
        }
        return try await recognizer.recognize(input.image)
    }
}
