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
    @Published var previewStatus = FormulaPreviewStatus.empty

    var onRetry: (() -> Void)?
    var onClose: (() -> Void)?

    private var copyConfirmationTask: Task<Void, Never>?

    func copyLatex() {
        guard !latex.isEmpty else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        copyConfirmation = pasteboard.setString(latex, forType: .string)
            ? "已复制 LaTeX"
            : "复制失败"

        copyConfirmationTask?.cancel()
        copyConfirmationTask = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 1_800_000_000)
            guard !Task.isCancelled else { return }
            self?.copyConfirmation = nil
        }
    }
}

struct ContentView: View {
    @ObservedObject var viewModel: FormulaResultViewModel
    let previewRenderer: any FormulaPreviewRendering

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
            Divider()
            footer
        }
        .frame(minWidth: 620, minHeight: 560)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var header: some View {
        HStack(spacing: 11) {
            ZStack {
                Circle()
                    .fill(headerTint.opacity(0.13))
                Image(systemName: headerSymbol)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(headerTint)
            }
            .frame(width: 30, height: 30)

            VStack(alignment: .leading, spacing: 2) {
                Text(headerTitle)
                    .font(.system(size: 14, weight: .semibold))
                Text(headerSubtitle)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if case .result = viewModel.phase {
                Label("本地识别", systemImage: "lock.fill")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    @ViewBuilder
    private var content: some View {
        switch viewModel.phase {
        case .recognizing:
            recognizingContent
        case .result:
            resultContent
        case .failed(let message):
            failureContent(message: message)
        }
    }

    private var recognizingContent: some View {
        VStack(spacing: 18) {
            screenshotSection(maxHeight: 130)
            Spacer()
            ProgressView()
                .controlSize(.regular)
            VStack(spacing: 5) {
                Text("正在识别公式")
                    .font(.headline)
                Text("模型在本机运行，截图不会离开这台 Mac。")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(20)
    }

    private var resultContent: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                screenshotSection(maxHeight: 64)

                ResultSection(title: "公式预览", systemImage: "function") {
                    previewRenderer.makePreview(
                        latex: viewModel.latex,
                        status: $viewModel.previewStatus
                    )
                    .frame(maxWidth: .infinity, minHeight: 90)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                }

                if case .unsupported = viewModel.previewStatus {
                    Label(
                        "预览器暂不支持部分语法，但不会影响源码编辑和复制。",
                        systemImage: "exclamationmark.triangle.fill"
                    )
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                }

                ResultSection(title: "LaTeX 源码", systemImage: "chevron.left.forwardslash.chevron.right") {
                    TextEditor(text: $viewModel.latex)
                        .font(.system(size: 13, design: .monospaced))
                        .scrollContentBackground(.hidden)
                        .padding(8)
                        .frame(minHeight: 128)
                }
            }
            .padding(16)
        }
    }

    private func failureContent(message: String) -> some View {
        VStack(spacing: 18) {
            screenshotSection(maxHeight: 130)
            Spacer()
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 28))
                .foregroundStyle(.orange)
            VStack(spacing: 6) {
                Text("未能识别这张截图")
                    .font(.headline)
                Text(message)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 360)
            }
            Button("使用同一截图重试") {
                viewModel.onRetry?()
            }
            .controlSize(.large)
            Spacer()
        }
        .padding(20)
    }

    @ViewBuilder
    private func screenshotSection(maxHeight: CGFloat) -> some View {
        if let screenshot = viewModel.screenshot {
            ResultSection(title: "原始截图", systemImage: "viewfinder") {
                Image(nsImage: screenshot)
                    .resizable()
                    .interpolation(.high)
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: maxHeight)
                    .padding(8)
            }
        }
    }

    private var footer: some View {
        HStack(spacing: 10) {
            Group {
                if let confirmation = viewModel.copyConfirmation {
                    Label(
                        confirmation,
                        systemImage: confirmation == "复制失败"
                            ? "xmark.circle.fill"
                            : "checkmark.circle.fill"
                    )
                    .foregroundStyle(
                        confirmation == "复制失败" ? .red : .secondary
                    )
                } else if case .result = viewModel.phase {
                    Text("⌘↩ 复制")
                        .foregroundStyle(.tertiary)
                }
            }
            .font(.system(size: 11))

            Spacer()

            Button("关闭") {
                viewModel.onClose?()
            }
            .keyboardShortcut(.cancelAction)

            if case .result = viewModel.phase {
                Button {
                    viewModel.copyLatex()
                } label: {
                    Label("复制 LaTeX", systemImage: "doc.on.doc")
                }
                .keyboardShortcut(.return, modifiers: .command)
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.latex.isEmpty)
                .help("复制纯 LaTeX，不添加 $ 或 $$")
            }
        }
        .controlSize(.regular)
        .padding(.horizontal, 20)
        .padding(.vertical, 13)
    }

    private var headerSymbol: String {
        switch viewModel.phase {
        case .recognizing: return "ellipsis"
        case .result: return "checkmark"
        case .failed: return "exclamationmark"
        }
    }

    private var headerTitle: String {
        switch viewModel.phase {
        case .recognizing: return "正在识别"
        case .result: return "识别完成"
        case .failed: return "识别失败"
        }
    }

    private var headerSubtitle: String {
        switch viewModel.phase {
        case .recognizing: return "正在分析选中的公式"
        case .result: return "检查预览，也可以直接修改源码"
        case .failed: return "可以保留当前截图并重新尝试"
        }
    }

    private var headerTint: Color {
        switch viewModel.phase {
        case .recognizing: return .accentColor
        case .result: return .green
        case .failed: return .orange
        }
    }
}

private struct ResultSection<Content: View>: View {
    let title: String
    let systemImage: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(title, systemImage: systemImage)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)

            content
                .frame(maxWidth: .infinity)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(Color(nsColor: .separatorColor).opacity(0.72))
                }
        }
    }
}
