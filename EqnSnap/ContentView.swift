//
//  ContentView.swift
//  EqnSnap
//
//  Created by wangjie on 2026/7/16.
//

import AppKit
import Combine
import SwiftUI

@MainActor
final class FormulaResultViewModel: ObservableObject {
    enum Phase: Equatable {
        case recognizing
        case result
        case failed(String)
    }

    @Published var phase: Phase = .recognizing
    @Published var latex = ""
    @Published var screenshot: NSImage?
    @Published var copyConfirmation: String?

    var onRetry: (() -> Void)?
    var onClose: (() -> Void)?

    func copyLatex() {
        guard !latex.isEmpty else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        if pasteboard.setString(latex, forType: .string) {
            copyConfirmation = "已复制"
        } else {
            copyConfirmation = "复制失败"
        }
    }
}

struct ContentView: View {
    @ObservedObject var viewModel: FormulaResultViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let screenshot = viewModel.screenshot {
                Image(nsImage: screenshot)
                    .resizable()
                    .interpolation(.high)
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: 130)
                    .background(Color(nsColor: .textBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            }

            switch viewModel.phase {
            case .recognizing:
                HStack(spacing: 10) {
                    ProgressView()
                        .controlSize(.small)
                    Text("正在本地识别公式…")
                }
                .frame(maxWidth: .infinity, minHeight: 120)

            case .result:
                Text("LaTeX")
                    .font(.headline)
                TextEditor(text: $viewModel.latex)
                    .font(.system(.body, design: .monospaced))
                    .frame(minHeight: 150)
                    .border(Color(nsColor: .separatorColor))

            case .failed(let message):
                VStack(alignment: .leading, spacing: 10) {
                    Label("识别失败", systemImage: "exclamationmark.triangle")
                        .font(.headline)
                    Text(message)
                        .foregroundStyle(.secondary)
                    Button("使用同一截图重试") {
                        viewModel.onRetry?()
                    }
                }
                .frame(maxWidth: .infinity, minHeight: 120, alignment: .leading)
            }

            HStack {
                if let confirmation = viewModel.copyConfirmation {
                    Text(confirmation)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("关闭") {
                    viewModel.onClose?()
                }
                if case .result = viewModel.phase {
                    Button("复制 LaTeX") {
                        viewModel.copyLatex()
                    }
                    .keyboardShortcut(.return, modifiers: .command)
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.latex.isEmpty)
                }
            }
        }
        .padding(18)
        .frame(minWidth: 560, minHeight: 390)
    }
}
