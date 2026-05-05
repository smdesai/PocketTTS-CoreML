//
// CloneSheet.swift
//
// Modal flow for cloning a new voice. Presented from VoiceListSheet's
// "+ Clone new voice" row. Lets the user either:
//
//   1. Record a ~4s reference clip via AVAudioRecorder (CloneRecorder.swift)
//   2. Pick any audio file via .fileImporter (UTType.audio)
//
// Both paths end up with a URL on disk that we hand to
// TTSViewModel.cloneVoice(from:named:), which runs the on-device pipeline
// (mimi_encoder → speaker_proj → flow_lm_prefill) and saves the resulting
// VoiceHandle into Documents/ClonedVoices/<language>/<name>.safetensors.
//
// State machine (pure SwiftUI @State — no separate view model since the
// flow is self-contained and short-lived):
//
//   .idle          both buttons enabled, Clone disabled
//   .recording     Stop + progress bar; Pick disabled
//   .recorded(url) shows duration + Preview; Clone enabled (if name set)
//   .cloning       everything disabled; spinner + stage label
//   .done          auto-dismiss
//
// Language implication: cloning is per-language (speaker_proj + mimi_encoder
// are per-language). The user's clone is scoped to the currently-selected
// language and will only appear in that language's voice list.
//

import AVFoundation
import SwiftUI
import UIKit
import UniformTypeIdentifiers

struct CloneSheet: View {
    /// Injected: parent view model. Sheet calls `viewModel.cloneVoice(from:named:)`
    /// and then `viewModel.refreshVoices(selectingID:)` on success.
    let viewModel: TTSViewModel
    /// Called after a successful clone with the new voice's id (filename
    /// stem). Parent is responsible for dismissing the sheet and
    /// refreshing its own voice picker if needed.
    let onCloned: (String) -> Void
    let onCancel: () -> Void

    // MARK: - State

    private enum Phase: Equatable {
        case idle
        case recording
        case recorded(URL, TimeInterval)
        case cloning
        case failed(String)
    }

    @State private var phase: Phase = .idle
    @State private var voiceName: String = ""
    @State private var showFilePicker: Bool = false
    @State private var showPermissionAlert: Bool = false
    @State private var previewPlayer: AVAudioPlayer? = nil
    @State private var isPreviewing: Bool = false

    @StateObject private var recorder = CloneRecorder()

    // MARK: - Derived flags

    /// Cloning is only possible if the current language's artifacts
    /// contain `speaker_proj.safetensors`. Computed at sheet-open time.
    private var cloningAvailable: Bool {
        VoiceCatalog.cloningAvailable(for: viewModel.selectedLanguage)
    }

    private var canClone: Bool {
        guard case .recorded = phase else { return false }
        return !voiceName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        NavigationStack {
            Group {
                if !cloningAvailable {
                    unavailableView
                } else {
                    mainForm
                }
            }
            .navigationTitle("Clone a voice")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") {
                        teardown()
                        onCancel()
                    }
                    .disabled(phase == .cloning)
                }
            }
            .fileImporter(
                isPresented: $showFilePicker,
                allowedContentTypes: [.audio],
                allowsMultipleSelection: false
            ) { result in
                handleFileImport(result)
            }
            .alert(
                "Microphone access required",
                isPresented: $showPermissionAlert
            ) {
                Button("Open Settings") { openSettings() }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Enable microphone access in Settings to record a voice sample.")
            }
            .onDisappear { teardown() }
        }
        .presentationDetents([.large])
        .presentationDragIndicator(.visible)
    }

    // MARK: - Main form

    private var mainForm: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                languageBanner

                nameField

                Text("Reference audio")
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.secondary)

                inputButtons

                stateBlock

                if case .failed(let msg) = phase {
                    Text(msg)
                        .font(.system(size: 14))
                        .foregroundStyle(.red)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .fill(Color.red.opacity(0.1))
                        )
                }

                Spacer(minLength: 8)

                cloneButton
            }
            .padding(20)
        }
    }

    private var languageBanner: some View {
        HStack(spacing: 10) {
            Image(systemName: "info.circle.fill")
                .foregroundStyle(.tint)
            Text(
                "Cloning for \(viewModel.selectedLanguage.displayName). "
                    + "The cloned voice will only work with "
                    + "\(viewModel.selectedLanguage.displayName) generation."
            )
            .font(.system(size: 13))
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.accentColor.opacity(0.1))
        )
    }

    private var nameField: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Voice name")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.secondary)
            TextField("e.g. My Voice", text: $voiceName)
                .textFieldStyle(.roundedBorder)
                .disabled(phase == .cloning)
                .autocorrectionDisabled(true)
                .textInputAutocapitalization(.words)
        }
    }

    private var inputButtons: some View {
        HStack(spacing: 12) {
            // Record / Stop button
            Button {
                switch phase {
                case .recording:
                    recorder.stop()
                default:
                    Task { await beginRecording() }
                }
            } label: {
                inputButtonLabel(
                    title: phase == .recording ? "Stop" : "Record",
                    systemImage: phase == .recording ? "stop.fill" : "mic.fill",
                    tint: phase == .recording ? .red : .accentColor
                )
            }
            .buttonStyle(.plain)
            .disabled(
                {
                    switch phase {
                    case .cloning: return true
                    default: return false
                    }
                }())

            // Pick-from-files button
            Button {
                dismissKeyboard()
                showFilePicker = true
            } label: {
                inputButtonLabel(
                    title: "Pick",
                    systemImage: "folder.fill",
                    tint: .accentColor
                )
            }
            .buttonStyle(.plain)
            .disabled(
                {
                    switch phase {
                    case .recording, .cloning: return true
                    default: return false
                    }
                }())
        }
    }

    private func inputButtonLabel(
        title: String, systemImage: String, tint: Color
    ) -> some View {
        VStack(spacing: 6) {
            Image(systemName: systemImage)
                .font(.system(size: 22, weight: .semibold))
            Text(title)
                .font(.system(size: 15, weight: .semibold))
        }
        .foregroundStyle(tint)
        .frame(maxWidth: .infinity, minHeight: 72)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(.secondarySystemBackground))
        )
    }

    // MARK: - State block (depends on phase)

    private var stateBlock: some View {
        Group {
            switch phase {
            case .idle:
                stateBody(
                    icon: "waveform",
                    tint: .gray,
                    title: "No audio yet",
                    subtitle:
                        "Record a 4-second sample or pick an audio file (.wav, .m4a, .caf, .aiff)."
                )

            case .recording:
                VStack(alignment: .leading, spacing: 10) {
                    stateBody(
                        icon: "mic.fill",
                        tint: .red,
                        title: String(
                            format: "Recording %.1fs / %.0fs",
                            recorder.elapsed, CloneRecorder.maxDuration),
                        subtitle: "Speak now. Recording stops automatically at 4 seconds."
                    )
                    ProgressView(
                        value: recorder.elapsed,
                        total: CloneRecorder.maxDuration
                    )
                    .tint(.red)
                }

            case .recorded(_, let duration):
                VStack(alignment: .leading, spacing: 12) {
                    stateBody(
                        icon: "checkmark.circle.fill",
                        tint: .green,
                        title: String(format: "Recorded %.2fs", duration),
                        subtitle: "Ready to clone. You can preview or re-record."
                    )
                    Button {
                        togglePreview()
                    } label: {
                        HStack(spacing: 8) {
                            Image(systemName: isPreviewing ? "stop.fill" : "play.fill")
                            Text(isPreviewing ? "Stop preview" : "Preview")
                        }
                        .font(.system(size: 15, weight: .semibold))
                        .frame(maxWidth: .infinity, minHeight: 36)
                    }
                    .buttonStyle(.bordered)
                }

            case .cloning:
                VStack(alignment: .leading, spacing: 10) {
                    stateBody(
                        icon: "gearshape.fill",
                        tint: .blue,
                        title: "Cloning voice…",
                        subtitle: "mimi_encoder → speaker_proj → flow_lm_prefill. "
                            + "This can take 10-30s on simulator and ~1-3s on device.",
                        showSpinner: true
                    )
                }

            case .failed:
                EmptyView()
            }
        }
    }

    private func stateBody(
        icon: String, tint: Color, title: String, subtitle: String,
        showSpinner: Bool = false
    ) -> some View {
        HStack(alignment: .top, spacing: 12) {
            ZStack {
                Circle().fill(tint).frame(width: 32, height: 32)
                if showSpinner {
                    ProgressView().tint(.white).controlSize(.small)
                } else {
                    Image(systemName: icon)
                        .font(.system(size: 15, weight: .bold))
                        .foregroundStyle(.white)
                }
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(size: 15, weight: .semibold))
                Text(subtitle)
                    .font(.system(size: 13))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(tint.opacity(0.12))
        )
    }

    // MARK: - Clone button

    private var cloneButton: some View {
        Button {
            Task { await performClone() }
        } label: {
            HStack(spacing: 8) {
                if phase == .cloning {
                    ProgressView().tint(.white).controlSize(.small)
                }
                Text(phase == .cloning ? "Cloning…" : "Clone voice")
                    .font(.system(size: 17, weight: .semibold))
            }
            .frame(maxWidth: .infinity, minHeight: 48)
        }
        .buttonStyle(.borderedProminent)
        .disabled(!canClone || phase == .cloning)
    }

    // MARK: - Unavailable view

    private var unavailableView: some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 40))
                .foregroundStyle(.orange)
            Text("Voice cloning is not available for \(viewModel.selectedLanguage.displayName).")
                .font(.system(size: 17, weight: .semibold))
                .multilineTextAlignment(.center)
            Text(
                "The speaker projection sidecar (speaker_proj.safetensors) "
                    + "is missing from this language's bundle. Re-export it via "
                    + "`python -m pockettts_coreml.convert.export_speaker_proj` "
                    + "and re-run prepare_resources.sh."
            )
            .font(.system(size: 13))
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
            .padding(.horizontal, 8)
            Spacer()
        }
        .padding(24)
    }

    // MARK: - Actions

    private func beginRecording() async {
        dismissKeyboard()
        // Stop any in-flight preview before we grab the mic.
        stopPreview()

        let granted = await CloneRecorder.requestPermission()
        guard granted else {
            showPermissionAlert = true
            return
        }
        do {
            try recorder.start()
            phase = .recording
            // Observe the recorder's completion to transition out of
            // .recording. We poll isRecording because AVAudioRecorder's
            // delegate fires on a background thread; @Published bounces
            // back to the main actor, and we drive the phase from there.
            Task { @MainActor in
                while recorder.isRecording {
                    try? await Task.sleep(nanoseconds: 100_000_000)
                }
                if let url = recorder.lastRecordingURL {
                    phase = .recorded(url, recorder.lastRecordingDuration)
                } else if case .recording = phase {
                    phase = .idle
                }
            }
        } catch {
            phase = .failed("Could not start recording: \(error.localizedDescription)")
        }
    }

    private func handleFileImport(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let picked = urls.first else { return }
            do {
                // .fileImporter URLs from outside the app sandbox are
                // security-scoped. Copy into our temp dir so we don't
                // have to hold the scope open through the async clone.
                let didStart = picked.startAccessingSecurityScopedResource()
                defer { if didStart { picked.stopAccessingSecurityScopedResource() } }

                let dest = FileManager.default.temporaryDirectory
                    .appendingPathComponent(
                        "pockettts-clone-picked-\(Int(Date().timeIntervalSince1970)).\(picked.pathExtension)"
                    )
                if FileManager.default.fileExists(atPath: dest.path) {
                    try FileManager.default.removeItem(at: dest)
                }
                try FileManager.default.copyItem(at: picked, to: dest)
                let duration = probeDuration(of: dest)
                phase = .recorded(dest, duration)
                // If the user hasn't named the voice yet, seed the name
                // with the picked filename stem (sanitized in save path).
                if voiceName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    voiceName = picked.deletingPathExtension().lastPathComponent
                }
            } catch {
                phase = .failed("Couldn't read audio file: \(error.localizedDescription)")
            }
        case .failure(let error):
            phase = .failed("Pick failed: \(error.localizedDescription)")
        }
    }

    private func probeDuration(of url: URL) -> TimeInterval {
        // Cheap duration probe that doesn't require decoding. Good
        // enough to show the user "we read X seconds"; VoiceCloner
        // will handle the truncation to 4s itself.
        if let player = try? AVAudioPlayer(contentsOf: url) {
            return player.duration
        }
        return 0
    }

    private func togglePreview() {
        if isPreviewing {
            stopPreview()
            return
        }
        guard case .recorded(let url, _) = phase else { return }
        do {
            // Ensure the session is in a playback-friendly state for the
            // preview (the record session left it in .playAndRecord
            // which is fine, but defaultToSpeaker routes audible output).
            let session = AVAudioSession.sharedInstance()
            try? session.setCategory(
                .playAndRecord, mode: .default,
                options: [.defaultToSpeaker, .allowBluetooth])
            try? session.setActive(true, options: [])

            let p = try AVAudioPlayer(contentsOf: url)
            p.prepareToPlay()
            p.play()
            previewPlayer = p
            isPreviewing = true
            // Poll for completion to flip the icon back.
            Task { @MainActor in
                while let player = previewPlayer, player.isPlaying {
                    try? await Task.sleep(nanoseconds: 100_000_000)
                }
                if previewPlayer != nil {
                    isPreviewing = false
                }
            }
        } catch {
            phase = .failed("Preview failed: \(error.localizedDescription)")
        }
    }

    private func stopPreview() {
        previewPlayer?.stop()
        previewPlayer = nil
        isPreviewing = false
    }

    private func performClone() async {
        guard case .recorded(let url, _) = phase else { return }
        let trimmedName = voiceName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty else { return }
        stopPreview()
        dismissKeyboard()
        phase = .cloning
        do {
            let newID = try await viewModel.cloneVoice(from: url, named: trimmedName)
            // Refresh parent's voice list and auto-select the new clone.
            viewModel.refreshVoices(selectingID: newID)
            teardown()
            onCloned(newID)
        } catch {
            phase = .failed("Clone failed: \(error.localizedDescription)")
        }
    }

    private func teardown() {
        stopPreview()
        recorder.teardown()
    }

    private func openSettings() {
        guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
        UIApplication.shared.open(url)
    }

    private func dismissKeyboard() {
        UIApplication.shared.sendAction(
            #selector(UIResponder.resignFirstResponder),
            to: nil, from: nil, for: nil
        )
    }
}
