//
// StreamingPlayer.swift
//
// AVAudioEngine-backed player for both non-streaming full-utterance playback
// and streaming sentence-chunk playback. PCM input is 24 kHz int16 LE mono;
// we convert to 24 kHz float32 mono AVAudioPCMBuffer and schedule on an
// AVAudioPlayerNode. The engine runs on the device's mixer output format,
// so AVAudioEngine applies any necessary rate conversion.
//

import AVFoundation
import Foundation

@MainActor
public final class StreamingPlayer: ObservableObject {

    @Published public private(set) var isPlaying: Bool = false
    @Published public private(set) var currentChunk: Int = 0
    @Published public private(set) var totalChunks: Int = 0

    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    /// Format the buffers we schedule are in. 24 kHz mono float32.
    private let sourceFormat: AVAudioFormat
    /// Monotonic count of buffers scheduled on the current stream. Used so
    /// each completion handler knows its own finish-index; reading
    /// `currentChunk` at schedule time would be racy since all chunks can
    /// be queued before any finishes.
    private var scheduledCount: Int = 0

    public init(sampleRate: Double = 24000) {
        // Non-interleaved float32 mono at 24 kHz — AVAudioEngine handles the
        // rate/channel conversion to the output device format.
        self.sourceFormat = AVAudioFormat(
            standardFormatWithSampleRate: sampleRate,
            channels: 1
        )!
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: sourceFormat)
    }

    // MARK: - Audio session

    /// Configure the shared AVAudioSession for spoken-audio playback.
    /// Safe to call multiple times.
    public func configureAudioSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .spokenAudio, options: [])
        try session.setActive(true, options: [])
    }

    /// Start the audio engine if not already running. Must be called on a
    /// freshly-attached configureAudioSession.
    public func startEngineIfNeeded() throws {
        guard !engine.isRunning else { return }
        try engine.start()
    }

    // MARK: - Non-streaming playback

    /// Play a full PCM buffer (int16 LE 24 kHz mono). Used by the Play
    /// button after a non-streaming generate.
    public func play(fullPCM data: Data, sampleRate: Double = 24000) throws {
        try configureAudioSession()
        try startEngineIfNeeded()
        stopAndReset()

        guard let buffer = pcmBuffer(fromInt16LE: data, sampleRate: sampleRate) else {
            return
        }
        totalChunks = 1
        currentChunk = 1
        isPlaying = true
        player.scheduleBuffer(buffer, at: nil, options: []) { [weak self] in
            Task { @MainActor in
                self?.isPlaying = false
            }
        }
        player.play()
    }

    // MARK: - Streaming playback

    /// Prepare the player for a sentence-chunk stream. Must be called once
    /// before the first `pushChunk`.
    public func startStream(totalChunks: Int) throws {
        try configureAudioSession()
        try startEngineIfNeeded()
        stopAndReset()
        self.totalChunks = totalChunks
        self.currentChunk = 0
        self.scheduledCount = 0
        self.isPlaying = true
    }

    /// Schedule one sentence's worth of PCM data. AVAudioPlayerNode queues
    /// buffers in order, so we can schedule as they arrive without any
    /// manual ring buffer. The completion handler advances `currentChunk`.
    public func pushChunk(_ data: Data, sampleRate: Double = 24000) {
        guard let buffer = pcmBuffer(fromInt16LE: data, sampleRate: sampleRate) else {
            return
        }
        // Capture this chunk's 1-based finish index. All chunks can be
        // scheduled back-to-back before any finishes, so we must NOT read
        // `currentChunk` at schedule time — each buffer gets its own
        // monotonically-increasing finish index.
        scheduledCount += 1
        let finishIndex = scheduledCount
        player.scheduleBuffer(buffer, at: nil, options: []) { [weak self] in
            Task { @MainActor in
                guard let self = self else { return }
                guard self.isPlaying else { return }
                self.currentChunk = max(self.currentChunk, finishIndex)
                if self.currentChunk >= self.totalChunks {
                    self.isPlaying = false
                }
            }
        }
        // Start playback on the first scheduled buffer.
        if !player.isPlaying {
            player.play()
        }
    }

    /// Stop the stream. Any remaining buffered chunks are dropped and the
    /// player returns to an idle state immediately.
    public func stopStream() {
        stopAndReset()
        isPlaying = false
        currentChunk = 0
        totalChunks = 0
        scheduledCount = 0
    }

    // MARK: - Internals

    private func stopAndReset() {
        if player.isPlaying {
            player.stop()
        }
        player.reset()
    }

    /// Convert a Data blob of int16 LE 24 kHz mono samples to an
    /// AVAudioPCMBuffer matching `sourceFormat`.
    private func pcmBuffer(
        fromInt16LE data: Data, sampleRate: Double
    ) -> AVAudioPCMBuffer? {
        let sampleCount = data.count / MemoryLayout<Int16>.size
        guard sampleCount > 0 else { return nil }
        guard
            let buf = AVAudioPCMBuffer(
                pcmFormat: sourceFormat,
                frameCapacity: AVAudioFrameCount(sampleCount)
            )
        else { return nil }
        buf.frameLength = AVAudioFrameCount(sampleCount)

        data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let src = raw.bindMemory(to: Int16.self).baseAddress!
            let dst = buf.floatChannelData![0]
            let scale: Float = 1.0 / 32768.0
            for i in 0 ..< sampleCount {
                dst[i] = Float(src[i]) * scale
            }
        }
        return buf
    }
}
