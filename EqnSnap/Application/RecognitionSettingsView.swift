import AppKit
import SwiftUI

struct RecognitionSettingsView: View {
    @ObservedObject var settings: RecognitionModelSettings

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("识别模型")
                .font(.system(size: 13, weight: .medium))

            RecognitionModelSegmentedControl(
                selection: $settings.selectedModel,
                isAvailable: settings.modelIsAvailable
            )
            .frame(height: 26)

            if !settings.modelIsAvailable(.uniMERNet) {
                Label(
                    "UniMERNet 模型尚未加入应用包",
                    systemImage: "externaldrive.badge.exclamationmark"
                )
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
            }

            Spacer(minLength: 0)
        }
        .frame(width: 380, height: 100)
        .padding(20)
    }
}

private struct RecognitionModelSegmentedControl: NSViewRepresentable {
    @Binding var selection: FormulaRecognitionModel
    let isAvailable: (FormulaRecognitionModel) -> Bool

    func makeCoordinator() -> Coordinator {
        Coordinator(selection: $selection)
    }

    func makeNSView(context: Context) -> NSSegmentedControl {
        let control = NSSegmentedControl(
            labels: FormulaRecognitionModel.allCases.map(\.displayName),
            trackingMode: .selectOne,
            target: context.coordinator,
            action: #selector(Coordinator.selectionChanged(_:))
        )
        return control
    }

    func updateNSView(_ control: NSSegmentedControl, context: Context) {
        let models = FormulaRecognitionModel.allCases
        control.selectedSegment = models.firstIndex(of: selection) ?? 0
        for (index, model) in models.enumerated() {
            control.setEnabled(isAvailable(model), forSegment: index)
        }
    }

    final class Coordinator: NSObject {
        private var selection: Binding<FormulaRecognitionModel>

        init(selection: Binding<FormulaRecognitionModel>) {
            self.selection = selection
        }

        @objc func selectionChanged(_ sender: NSSegmentedControl) {
            let models = FormulaRecognitionModel.allCases
            guard models.indices.contains(sender.selectedSegment) else { return }
            selection.wrappedValue = models[sender.selectedSegment]
        }
    }
}
