//
// ContentView.swift — redesign per mockup:
//   - large title
//   - Voice card (compact row with Menu)
//   - Text card (header + clear button, char count in bottom-right)
//   - Status card (tinted; Audio-ready stats or streaming/error state)
//   - Bottom bar: Generate / Stream / Play / Share (all icons + labels)
//

import SwiftUI
import UIKit

struct ContentView: View {
    @State private var viewModel: TTSViewModel
    @StateObject private var playerBinder: PlayerBinder
    @State private var shareItem: ShareItem? = nil
    @State private var showVoiceSheet: Bool = false
    @State private var showLanguageSheet: Bool = false

    init() {
        let vm = TTSViewModel()
        _viewModel = State(initialValue: vm)
        _playerBinder = StateObject(wrappedValue: PlayerBinder(player: vm.player))
    }

    var body: some View {
        NavigationStack {
            ZStack(alignment: .bottom) {
                // Scrollable content above the bottom bar.
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("PocketTTS CoreML")
                            .font(.system(size: 34, weight: .bold))
                            .padding(.top, 4)

                        languageCard
                        voiceCard
                        textCard
                        statusCard
                    }
                    .padding(.horizontal, 20)
                    .padding(.bottom, 120)  // room for bottom bar
                }
                .scrollDismissesKeyboard(.interactively)
                // Tap-to-dismiss: any tap that falls through to the
                // background (not on a control) resigns first responder.
                // Use `simultaneousGesture` with `.onEnded` and `.high`
                // priority so buttons/menus still receive their taps.
                .background(
                    Color.clear
                        .contentShape(Rectangle())
                        .onTapGesture { dismissKeyboard() }
                )

                bottomBar
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar(.hidden, for: .navigationBar)
            .task { await viewModel.load() }
            .sheet(item: $shareItem) { item in
                ShareSheet(items: [item.url])
            }
        }
    }

    private func dismissKeyboard() {
        UIApplication.shared.sendAction(
            #selector(UIResponder.resignFirstResponder),
            to: nil, from: nil, for: nil
        )
    }

    // MARK: - Language card

    private var languageCard: some View {
        Button {
            guard !viewModel.isGenerating,
                !viewModel.isStreaming,
                !viewModel.isSwitchingLanguage,
                !viewModel.isDownloading
            else { return }
            dismissKeyboard()
            showLanguageSheet = true
        } label: {
            HStack(spacing: 12) {
                ZStack {
                    Circle().fill(Color.white.opacity(0.08)).frame(width: 28, height: 28)
                    Image(systemName: "globe")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.white)
                }
                Text("Language")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(.primary)
                Spacer()
                Text(viewModel.selectedLanguage.displayName)
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(Color.accentColor)
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Color.accentColor)
            }
            .padding(.vertical, 14)
            .padding(.horizontal, 16)
            .frame(maxWidth: .infinity)
            .background(cardBackground)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(
            viewModel.isGenerating
                || viewModel.isStreaming
                || viewModel.isSwitchingLanguage
                || viewModel.isDownloading
        )
        .sheet(isPresented: $showLanguageSheet) {
            LanguageListSheet(
                languages: Language.all,
                selectedID: viewModel.selectedLanguage.id,
                onPick: { language in
                    showLanguageSheet = false
                    // Async switch on the MainActor-bound view model.
                    Task { await viewModel.switchLanguage(to: language) }
                },
                onCancel: { showLanguageSheet = false }
            )
        }
    }

    // MARK: - Voice card

    private var voiceCard: some View {
        Button {
            guard !viewModel.isGenerating,
                !viewModel.isStreaming,
                !viewModel.isSwitchingLanguage,
                !viewModel.isDownloading,
                !viewModel.voices.isEmpty
            else { return }
            dismissKeyboard()
            showVoiceSheet = true
        } label: {
            HStack(spacing: 12) {
                ZStack {
                    Circle().fill(Color.white.opacity(0.08)).frame(width: 28, height: 28)
                    Image(systemName: "waveform")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.white)
                }
                Text("Voice")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(.primary)
                Spacer()
                Text(viewModel.selectedVoice?.displayName ?? "—")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(Color.accentColor)
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(Color.accentColor)
            }
            .padding(.vertical, 14)
            .padding(.horizontal, 16)
            .frame(maxWidth: .infinity)
            .background(cardBackground)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(
            viewModel.isGenerating
                || viewModel.isStreaming
                || viewModel.isSwitchingLanguage
                || viewModel.isDownloading
                || viewModel.voices.isEmpty
        )
        .sheet(isPresented: $showVoiceSheet) {
            VoiceListSheet(
                viewModel: viewModel,
                onPick: { voice in
                    viewModel.selectedVoice = voice
                    showVoiceSheet = false
                },
                onCancel: { showVoiceSheet = false }
            )
        }
    }

    // MARK: - Text card

    private var textCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "text.alignleft")
                    .font(.system(size: 18, weight: .semibold))
                Text("Text to Speak")
                    .font(.system(size: 18, weight: .semibold))
                Spacer()
                Button("Clear") {
                    viewModel.text = ""
                }
                .foregroundStyle(.secondary)
                .disabled(
                    viewModel.text.isEmpty || viewModel.isGenerating || viewModel.isStreaming
                )
            }

            TextEditor(text: $viewModel.text)
                .font(.system(size: 17))
                .scrollContentBackground(.hidden)
                .frame(minHeight: 180, maxHeight: 260)
                .disabled(viewModel.isGenerating || viewModel.isStreaming)
                .autocorrectionDisabled(false)
                .textInputAutocapitalization(.sentences)

            HStack {
                Spacer()
                Text("\(viewModel.text.count) characters")
                    .font(.system(size: 14))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 14)
        .padding(.horizontal, 16)
        .frame(maxWidth: .infinity)
        .background(cardBackground)
    }

    // MARK: - Status card

    private var statusCard: some View {
        Group {
            if let err = viewModel.errorMessage {
                statusBody(
                    tint: .red,
                    icon: "exclamationmark.triangle.fill",
                    title: "Error",
                    subtitle: err
                )
            } else if viewModel.isDownloading {
                downloadCard
            } else if viewModel.isGenerating {
                statusBody(
                    tint: .blue,
                    icon: "waveform",
                    title: viewModel.status,
                    subtitle: nil,
                    showSpinner: true
                )
            } else if viewModel.isStreaming {
                let label: String = {
                    if playerBinder.totalChunks > 0 {
                        let cur = min(
                            max(playerBinder.currentChunk, 1),
                            playerBinder.totalChunks)
                        return "Streaming \(cur)/\(playerBinder.totalChunks)"
                    }
                    return "Streaming"
                }()
                statusBody(
                    tint: .orange,
                    icon: "dot.radiowaves.left.and.right",
                    title: label,
                    // Show stats live during streaming once we have any
                    // (first-audio ticks as soon as the first chunk
                    // generates; audio+gen+RTF fill in on completion).
                    subtitle: statsSubtitle,
                    showSpinner: true
                )
            } else if viewModel.generatedPCM != nil
                || viewModel.stats.audioSeconds != nil
            {
                // "Audio ready" covers both Generate (has playable PCM)
                // and Stream (no stored PCM but stats are populated).
                statusBody(
                    tint: .green,
                    icon: "checkmark",
                    title: "Audio ready",
                    subtitle: statsSubtitle
                )
            } else if !viewModel.isReady {
                statusBody(
                    tint: .gray,
                    icon: "clock",
                    title: viewModel.status,
                    subtitle: statsSubtitle,
                    showSpinner: true
                )
            } else {
                statusBody(
                    tint: .gray,
                    icon: "checkmark",
                    title: "Ready",
                    subtitle: statsSubtitle
                )
            }
        }
    }

    // MARK: - Download progress card

    /// Blue-tinted status card rendered while a language bundle is being
    /// downloaded from Hugging Face. Shows a determinate progress bar
    /// once we know the total size, plus a byte counter and the name of
    /// the currently-downloading file. Falls back to an indeterminate
    /// spinner if the tree API hasn't yielded sizes yet.
    private var downloadCard: some View {
        let progress = viewModel.downloadProgress ?? .zero
        let langName = viewModel.selectedLanguage.displayName
        let total = progress.bytesTotal
        let done = progress.bytesCompleted
        let fraction: Double? = total > 0 ? min(1.0, Double(done) / Double(total)) : nil
        let countsSubtitle: String = {
            if progress.totalFiles == 0 {
                return "Fetching file list…"
            }
            var s = "File \(progress.currentFileIndex) of \(progress.totalFiles)"
            if !progress.currentFileName.isEmpty {
                s += " • \(progress.currentFileName)"
            }
            return s
        }()
        let bytesSubtitle: String? = {
            guard total > 0 else { return nil }
            return "\(formatBytes(done)) / \(formatBytes(total))"
        }()
        return HStack(alignment: .top, spacing: 12) {
            ZStack {
                Circle().fill(Color.blue).frame(width: 34, height: 34)
                Image(systemName: "icloud.and.arrow.down")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(.white)
            }
            VStack(alignment: .leading, spacing: 6) {
                Text("Downloading \(langName) model…")
                    .font(.system(size: 17, weight: .semibold))
                if let frac = fraction {
                    ProgressView(value: frac)
                        .progressViewStyle(.linear)
                        .tint(.blue)
                } else {
                    ProgressView()
                        .progressViewStyle(.linear)
                        .tint(.blue)
                }
                if let bs = bytesSubtitle {
                    Text(bs)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.secondary)
                }
                Text(countsSubtitle)
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 14)
        .padding(.horizontal, 16)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color.blue.opacity(0.12))
        )
    }

    /// "485 MB" / "1.70 GB" — ByteCountFormatter with `.file` style.
    private func formatBytes(_ bytes: Int64) -> String {
        let fmt = ByteCountFormatter()
        fmt.allowedUnits = [.useMB, .useGB]
        fmt.countStyle = .file
        return fmt.string(fromByteCount: bytes)
    }

    private var statsSubtitle: String? {
        var parts: [String] = []
        if let init_ = viewModel.stats.initSeconds {
            parts.append(String(format: "Init: %.2fs", init_))
        }
        if let first = viewModel.stats.firstAudioSeconds {
            parts.append(String(format: "First audio: %.2fs", first))
        }
        if let gen = viewModel.stats.generateSeconds {
            parts.append(String(format: "Gen: %.2fs", gen))
        }
        if let audio = viewModel.stats.audioSeconds {
            parts.append(String(format: "Audio: %.2fs", audio))
        }
        if let rtf = viewModel.stats.rtf, rtf > 0 {
            // User-facing metric: RTFx = audio / wall (how many times
            // faster than realtime). Internal stats.rtf is wall / audio,
            // so invert for display.
            parts.append(String(format: "RTFx %.2fx", 1.0 / rtf))
        }
        if let peak = viewModel.stats.peakMemoryMB {
            if peak >= 1024 {
                parts.append(String(format: "Peak: %.2f GB", peak / 1024.0))
            } else {
                parts.append(String(format: "Peak: %.0f MB", peak))
            }
        }
        return parts.isEmpty ? nil : parts.joined(separator: " • ")
    }

    private func statusBody(
        tint: Color,
        icon: String,
        title: String,
        subtitle: String?,
        showSpinner: Bool = false
    ) -> some View {
        HStack(alignment: .top, spacing: 12) {
            ZStack {
                Circle().fill(tint).frame(width: 34, height: 34)
                if showSpinner {
                    ProgressView()
                        .tint(.white)
                        .controlSize(.small)
                } else {
                    Image(systemName: icon)
                        .font(.system(size: 16, weight: .bold))
                        .foregroundStyle(.white)
                }
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 17, weight: .semibold))
                if let subtitle = subtitle {
                    Text(subtitle)
                        .font(.system(size: 14))
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
        }
        .padding(.vertical, 14)
        .padding(.horizontal, 16)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(tint.opacity(0.12))
        )
    }

    // MARK: - Bottom bar

    private var bottomBar: some View {
        HStack(spacing: 0) {
            barButton(
                title: "Generate",
                systemImage: "waveform",
                tint: .accentColor,
                enabled: viewModel.isReady
                    && !viewModel.isGenerating
                    && !viewModel.isStreaming
                    && !viewModel.isSwitchingLanguage
            ) {
                Task { await viewModel.generate() }
            }

            barButton(
                title: "Stream",
                systemImage: "dot.radiowaves.left.and.right",
                tint: .orange,
                enabled: viewModel.isReady
                    && !viewModel.isGenerating
                    && !viewModel.isStreaming
                    && !viewModel.isSwitchingLanguage
            ) {
                Task { await viewModel.stream() }
            }

            if viewModel.isStreaming || playerBinder.isPlaying {
                barButton(
                    title: "Stop",
                    systemImage: "stop.fill",
                    tint: .red,
                    enabled: true
                ) {
                    viewModel.stop()
                }
            } else {
                barButton(
                    title: "Play",
                    systemImage: "play.fill",
                    tint: .white,
                    enabled: viewModel.generatedPCM != nil
                ) {
                    Task { await viewModel.play() }
                }
            }

            barButton(
                title: "Share",
                systemImage: "square.and.arrow.up",
                tint: .white,
                enabled: viewModel.generatedPCM != nil
            ) {
                if let url = viewModel.exportGeneratedAsWav() {
                    shareItem = ShareItem(url: url)
                }
            }
        }
        .padding(.top, 10)
        .padding(.bottom, 20)
        .background(.thinMaterial)
    }

    private func barButton(
        title: String,
        systemImage: String,
        tint: Color,
        enabled: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: systemImage)
                    .font(.system(size: 22, weight: .semibold))
                Text(title)
                    .font(.system(size: 13))
            }
            .foregroundStyle(enabled ? tint : Color.secondary)
            .frame(maxWidth: .infinity)
        }
        .disabled(!enabled)
    }

    // MARK: - Card background

    private var cardBackground: some View {
        RoundedRectangle(cornerRadius: 14, style: .continuous)
            .fill(Color(.secondarySystemBackground))
    }
}

// MARK: - Share sheet (UIActivityViewController wrapper)

// MARK: - Language picker sheet

/// Sheet with one row per bundled language. Same visual vocabulary as
/// VoiceListSheet so the picker feels familiar.
private struct LanguageListSheet: View {
    let languages: [Language]
    let selectedID: String
    let onPick: (Language) -> Void
    let onCancel: () -> Void

    var body: some View {
        NavigationStack {
            List {
                ForEach(languages) { language in
                    Button {
                        onPick(language)
                    } label: {
                        HStack {
                            Text(language.displayName)
                                .foregroundStyle(.primary)
                            Spacer()
                            if language.id == selectedID {
                                Image(systemName: "checkmark")
                                    .foregroundStyle(Color.accentColor)
                                    .fontWeight(.semibold)
                            }
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
            .navigationTitle("Select language")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done", action: onCancel)
                }
            }
        }
        .presentationDetents([.medium])
        .presentationDragIndicator(.visible)
    }
}

// MARK: - Voice picker sheet

/// Full-screen sheet with one row per voice. Sheet-based selection is
/// immune to the hit-test flakiness Menu/Picker can show when they sit
/// inside a custom card layout with nearby tap gestures.
///
/// Top row is "+ Clone new voice" — tapping it presents CloneSheet
/// modally. On successful clone, the voice list refreshes (bundled +
/// Documents/ClonedVoices/<lang>/) and the newly-cloned voice is
/// auto-selected and the sheet dismissed.
///
/// Cloned voices (`VoiceEntry.isCloned == true`) are rendered with a
/// "(cloned)" suffix and support swipe-to-delete. Bundled voices are
/// read-only and cannot be deleted.
private struct VoiceListSheet: View {
    let viewModel: TTSViewModel
    let onPick: (VoiceEntry) -> Void
    let onCancel: () -> Void

    @State private var showCloneSheet: Bool = false

    var body: some View {
        NavigationStack {
            List {
                // "+ Clone new voice" row — always the first row, regardless
                // of whether cloning is available for this language (the
                // CloneSheet itself shows a helpful message if not).
                Section {
                    Button {
                        showCloneSheet = true
                    } label: {
                        HStack(spacing: 10) {
                            Image(systemName: "plus.circle.fill")
                                .font(.system(size: 18, weight: .semibold))
                                .foregroundStyle(Color.accentColor)
                            Text("Clone new voice")
                                .font(.system(size: 17, weight: .semibold))
                                .foregroundStyle(Color.accentColor)
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.system(size: 13, weight: .semibold))
                                .foregroundStyle(.secondary)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }

                Section("Voices") {
                    ForEach(viewModel.voices) { voice in
                        Button {
                            onPick(voice)
                        } label: {
                            HStack(spacing: 8) {
                                if voice.isCloned {
                                    Image(systemName: "waveform.badge.plus")
                                        .foregroundStyle(.tint)
                                }
                                Text(voice.displayName)
                                    .foregroundStyle(.primary)
                                if voice.isCloned {
                                    Text("(cloned)")
                                        .font(.system(size: 13))
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                if voice.id == viewModel.selectedVoice?.id {
                                    Image(systemName: "checkmark")
                                        .foregroundStyle(Color.accentColor)
                                        .fontWeight(.semibold)
                                }
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        // Swipe-to-delete only on cloned voices. Bundled
                        // voices live in the read-only app bundle.
                        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                            if voice.isCloned {
                                Button(role: .destructive) {
                                    deleteCloned(voice)
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Select voice")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done", action: onCancel)
                }
            }
            .sheet(isPresented: $showCloneSheet) {
                CloneSheet(
                    viewModel: viewModel,
                    onCloned: { _ in
                        // TTSViewModel.cloneVoice already refreshed and
                        // auto-selected the new voice. Dismiss both the
                        // clone sheet and the voice picker so the user
                        // returns to the main screen with their new
                        // voice active.
                        showCloneSheet = false
                        onCancel()
                    },
                    onCancel: { showCloneSheet = false }
                )
            }
        }
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
    }

    private func deleteCloned(_ entry: VoiceEntry) {
        do {
            try VoiceCatalog.deleteCloned(entry)
            // Refresh the list. If the deleted voice was selected, fall
            // back to the first remaining voice.
            let wasSelected = viewModel.selectedVoice?.id == entry.id
            viewModel.refreshVoices(
                selectingID: wasSelected ? nil : viewModel.selectedVoice?.id
            )
        } catch {
            // Silently swallow — deletion failure is rare (permission /
            // missing file) and the refresh will naturally re-sync.
        }
    }
}

private struct ShareItem: Identifiable {
    let id = UUID()
    let url: URL
}

private struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

// MARK: - StreamingPlayer → @Published relay

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
