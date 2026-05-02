//
// ContentView.swift
//
// The whole UI:
//  - voice picker
//  - multiline text editor
//  - Generate / Play / Stream / Stop buttons
//  - status label
//

import SwiftUI

struct ContentView: View {
    @State private var viewModel = TTSViewModel()
    // StreamingPlayer's @Published state drives button enablement; observe
    // it as a StateObject so the view re-renders on playback transitions.
    @StateObject private var playerBinder: PlayerBinder

    init() {
        let vm = TTSViewModel()
        _viewModel = State(initialValue: vm)
        _playerBinder = StateObject(wrappedValue: PlayerBinder(player: vm.player))
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if let err = viewModel.errorMessage {
                        errorBanner(err)
                    }
                    voiceSection
                    textSection
                    actionRow
                    statusSection
                }
                .padding()
            }
            .navigationTitle("PocketTTS Demo")
            .task {
                await viewModel.load()
            }
        }
    }

    // MARK: - Sections

    private var voiceSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Voice").font(.subheadline).foregroundStyle(.secondary)
            VoicePickerView(
                voices: viewModel.voices,
                selection: Binding(
                    get: { viewModel.selectedVoice },
                    set: { viewModel.selectedVoice = $0 }
                ),
                disabled: viewModel.isGenerating || viewModel.isStreaming
            )
        }
    }

    private var textSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Text").font(.subheadline).foregroundStyle(.secondary)
            TextEditor(text: Binding(
                get: { viewModel.text },
                set: { viewModel.text = $0 }
            ))
            .frame(minHeight: 140, maxHeight: 240)
            .padding(8)
            .background(Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .disabled(viewModel.isGenerating || viewModel.isStreaming)
        }
    }

    private var actionRow: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                Button {
                    Task { await viewModel.generate() }
                } label: {
                    Label("Generate", systemImage: "waveform")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!viewModel.isReady
                          || viewModel.isGenerating
                          || viewModel.isStreaming)

                Button {
                    Task { await viewModel.play() }
                } label: {
                    Label("Play", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(viewModel.generatedPCM == nil
                          || viewModel.isStreaming
                          || playerBinder.isPlaying)
            }

            HStack(spacing: 10) {
                Button {
                    Task { await viewModel.stream() }
                } label: {
                    Label("Stream", systemImage: "dot.radiowaves.left.and.right")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(!viewModel.isReady
                          || viewModel.isGenerating
                          || viewModel.isStreaming)

                Button(role: .destructive) {
                    viewModel.stop()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(!(viewModel.isStreaming
                            || viewModel.isGenerating
                            || playerBinder.isPlaying))
            }
        }
    }

    private var statusSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Status").font(.subheadline).foregroundStyle(.secondary)
            HStack {
                if viewModel.isGenerating || viewModel.isStreaming {
                    ProgressView().controlSize(.small)
                }
                Text(statusString)
                    .font(.callout.monospacedDigit())
            }
            .padding(.vertical, 6)
            .padding(.horizontal, 10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private var statusString: String {
        if viewModel.isStreaming && playerBinder.totalChunks > 0 {
            let cur = min(
                max(playerBinder.currentChunk, 1),
                playerBinder.totalChunks
            )
            return "Streaming (\(cur)/\(playerBinder.totalChunks))…"
        }
        if playerBinder.isPlaying && !viewModel.isStreaming {
            return "Playing…"
        }
        return viewModel.status
    }

    private func errorBanner(_ msg: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Error").font(.headline).foregroundStyle(.red)
            Text(msg).font(.caption.monospaced())
        }
        .padding()
        .background(Color.red.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

/// Adapts StreamingPlayer's ObservableObject @Published state into something
/// a SwiftUI @StateObject can observe for button-enablement purposes. We
/// can't make the @Observable TTSViewModel own it directly without losing
/// Combine publisher observability, so this thin relay wraps it.
@MainActor
final class PlayerBinder: ObservableObject {
    @Published var isPlaying: Bool = false
    @Published var currentChunk: Int = 0
    @Published var totalChunks: Int = 0

    private let player: StreamingPlayer

    init(player: StreamingPlayer) {
        self.player = player
        observe()
    }

    private func observe() {
        // Minimal KVO-free relay: poll via a periodic Task. AVAudioEngine's
        // completion handlers already update @Published properties on the
        // player; a 10 Hz poll is plenty for UI state.
        Task { @MainActor [weak self] in
            guard let self = self else { return }
            while !Task.isCancelled {
                self.isPlaying = self.player.isPlaying
                self.currentChunk = self.player.currentChunk
                self.totalChunks = self.player.totalChunks
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
    }
}

#Preview {
    ContentView()
}
