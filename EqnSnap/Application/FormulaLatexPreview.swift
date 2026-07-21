import AppKit
import SwiftMath
import SwiftUI

enum FormulaPreviewStatus: Equatable {
    case empty
    case rendered
    case unsupported(String)
}

@MainActor
protocol FormulaPreviewRendering {
    func makePreview(
        latex: String,
        status: Binding<FormulaPreviewStatus>
    ) -> AnyView
}

@MainActor
struct SwiftMathFormulaPreviewRenderer: FormulaPreviewRendering {
    func makePreview(
        latex: String,
        status: Binding<FormulaPreviewStatus>
    ) -> AnyView {
        AnyView(
            SwiftMathFormulaPreview(
                latex: latex,
                status: status
            )
        )
    }
}

private struct SwiftMathFormulaPreview: NSViewRepresentable {
    let latex: String
    @Binding var status: FormulaPreviewStatus

    func makeCoordinator() -> Coordinator {
        Coordinator(status: $status)
    }

    func makeNSView(context: Context) -> MTMathUILabel {
        let view = MTMathUILabel()
        view.labelMode = .display
        view.textAlignment = .center
        view.fontSize = 24
        view.textColor = .labelColor
        view.displayErrorInline = false
        return view
    }

    func updateNSView(_ view: MTMathUILabel, context: Context) {
        view.textColor = .labelColor
        view.latex = latex
        context.coordinator.report(
            latex: latex,
            error: view.error
        )
    }

    final class Coordinator {
        private let status: Binding<FormulaPreviewStatus>

        init(status: Binding<FormulaPreviewStatus>) {
            self.status = status
        }

        func report(latex: String, error: NSError?) {
            let nextStatus: FormulaPreviewStatus
            if latex.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                nextStatus = .empty
            } else if let error {
                nextStatus = .unsupported(error.localizedDescription)
            } else {
                nextStatus = .rendered
            }
            guard status.wrappedValue != nextStatus else { return }
            DispatchQueue.main.async { [status] in
                status.wrappedValue = nextStatus
            }
        }
    }
}
