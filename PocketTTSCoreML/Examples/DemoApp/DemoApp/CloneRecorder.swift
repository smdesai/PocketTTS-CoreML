//
// CloneRecorder.swift
//
// Thin wrapper around AVAudioRecorder + AVAudioSession for the voice-clone
// flow. Owns:
//
//  - microphone permission request (AVCaptureDevice.requestAccess)
//  - an AVAudioSession configured for .playAndRecord so we don't conflict
//    with the main app's TTS playback session
//  - a 4-second hard cap (mimi_encoder ingests exactly 4s; anything longer
//    is silently truncated by VoiceCloner.loadMonoFloat32_24k, but we cut
//    at the UI layer so the progress bar is honest).
//
// Target output: .m4a AAC in a temp file. VoiceCloner resamples to 24 kHz
// mono internally, so we don't care that AVAudioRecorder defaults to 44.1
// or 48 kHz — the downstream path handles it.
//

import AVFoundation
import Foundation

@MainActor
final class CloneRecorder: NSObject, ObservableObject, AVAudioRecorderDelegate {

    /// Fixed recording window (seconds). mimi_encoder expects exactly 4s
    /// of audio; we stop the recorder here so the user can't record past
    /// that and the progress bar is meaningful.
    static let maxDuration: TimeInterval = 4.0

    @Published var isRecording: Bool = false
    /// 0.0 → maxDuration; updated ~10 Hz while recording.
    @Published var elapsed: TimeInterval = 0.0
    /// Absolute URL of the last successful recording (m4a). nil until the
    /// recorder finishes naturally or is stopped.
    @Published var lastRecordingURL: URL? = nil
    /// Actual duration of `lastRecordingURL` in seconds.
    @Published var lastRecordingDuration: TimeInterval = 0.0

    private var recorder: AVAudioRecorder? = nil
    private var tickTask: Task<Void, Never>? = nil

    /// Ask for mic permission. Returns true if granted. If previously
    /// denied, returns false immediately — caller is responsible for
    /// surfacing a "Go to Settings" alert.
    static func requestPermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return true
        case .denied, .restricted:
            return false
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .audio)
        @unknown default:
            return false
        }
    }

    /// Current permission status without prompting. Used to drive the
    /// "permission denied" banner in CloneSheet.
    static var permissionStatus: AVAuthorizationStatus {
        AVCaptureDevice.authorizationStatus(for: .audio)
    }

    /// Start recording into a fresh temp .m4a. Caller is expected to have
    /// already awaited `requestPermission()` and confirmed it returned
    /// true. Throws if the audio session or recorder can't be set up.
    func start() throws {
        stop()  // idempotent guard

        let session = AVAudioSession.sharedInstance()
        // .playAndRecord lets the demo still play back TTS audio after
        // recording without tearing down the session. .defaultToSpeaker
        // avoids the routing-to-earpiece surprise when mixed with
        // a bluetooth preview.
        try session.setCategory(
            .playAndRecord, mode: .default,
            options: [.defaultToSpeaker, .allowBluetooth])
        try session.setActive(true, options: [])

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("pockettts-clone-\(Int(Date().timeIntervalSince1970)).m4a")

        // AAC in .m4a. The sample rate we request here is honored
        // best-effort by the hardware; VoiceCloner resamples anyway.
        // 24 kHz mono is requested to match the downstream expectation
        // and keep the file small.
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: 24_000.0,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
        ]

        let rec = try AVAudioRecorder(url: url, settings: settings)
        rec.delegate = self
        rec.isMeteringEnabled = false
        guard rec.prepareToRecord() else {
            throw NSError(
                domain: "CloneRecorder", code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey: "prepareToRecord failed"
                ])
        }
        // Hard-cap the duration. AVAudioRecorder will fire the delegate's
        // audioRecorderDidFinishRecording(_:successfully:) callback at
        // this point, which flips isRecording back to false.
        guard rec.record(forDuration: Self.maxDuration) else {
            throw NSError(
                domain: "CloneRecorder", code: 2,
                userInfo: [
                    NSLocalizedDescriptionKey: "record(forDuration:) returned false"
                ])
        }

        self.recorder = rec
        self.elapsed = 0
        self.isRecording = true
        self.lastRecordingURL = nil
        self.lastRecordingDuration = 0

        // 10 Hz UI tick for the progress bar. AVAudioRecorder's
        // currentTime is the authoritative elapsed value.
        tickTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                guard let self = self, let r = self.recorder, r.isRecording else { break }
                self.elapsed = r.currentTime
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }
    }

    /// Stop early (before the 4s cap). Safe to call even when not
    /// recording — it's a no-op in that case.
    func stop() {
        tickTask?.cancel()
        tickTask = nil
        if let r = recorder, r.isRecording {
            r.stop()  // triggers audioRecorderDidFinishRecording(_:successfully:)
        }
        // Leave isRecording as-is; the delegate callback flips it.
    }

    /// Drop the recorder and detach the audio session. Called when the
    /// sheet is dismissed so the main app's TTS playback session isn't
    /// left on .playAndRecord unnecessarily.
    func teardown() {
        stop()
        recorder = nil
        try? AVAudioSession.sharedInstance().setActive(
            false,
            options: [.notifyOthersOnDeactivation])
    }

    // MARK: - AVAudioRecorderDelegate

    nonisolated func audioRecorderDidFinishRecording(
        _ recorder: AVAudioRecorder, successfully flag: Bool
    ) {
        // Delegate callback may arrive on any thread.
        let url = recorder.url
        Task { @MainActor in
            self.isRecording = false
            self.tickTask?.cancel()
            self.tickTask = nil
            if flag {
                let duration = self.elapsed
                self.lastRecordingURL = url
                self.lastRecordingDuration = min(duration, Self.maxDuration)
                if self.lastRecordingDuration <= 0 {
                    // If we never ticked (recorder stopped immediately),
                    // fall back to the player probe below.
                    let player = try? AVAudioPlayer(contentsOf: url)
                    self.lastRecordingDuration = player?.duration ?? 0
                }
            } else {
                self.lastRecordingURL = nil
                self.lastRecordingDuration = 0
            }
        }
    }

    nonisolated func audioRecorderEncodeErrorDidOccur(
        _ recorder: AVAudioRecorder, error: Error?
    ) {
        Task { @MainActor in
            self.isRecording = false
            self.tickTask?.cancel()
            self.tickTask = nil
            self.lastRecordingURL = nil
            self.lastRecordingDuration = 0
        }
    }
}
