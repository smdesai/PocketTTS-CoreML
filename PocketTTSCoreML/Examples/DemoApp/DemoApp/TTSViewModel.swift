//
// TTSViewModel.swift
//
// Main orchestrator for the demo UI. Owns:
//  - the PocketTTS actor (heavy to construct; built once at launch)
//  - a Tokenizer instance used for sentence chunking (cheap)
//  - the StreamingPlayer
//  - voice handles cached by file URL
//
// Uses `@Observable` (iOS 17+ macro) for reactive state. The StreamingPlayer
// is kept separate as an ObservableObject because its @Published state is
// driven by AVAudioEngine completion callbacks on arbitrary threads and
// we want SwiftUI to observe it directly via @StateObject in the view.
//

import Foundation
import SwiftUI
import PocketTTSCoreML

@Observable
@MainActor
public final class TTSViewModel {
    /// Max tokens per sentence-chunk. Lower than the package default (50)
    /// to keep each chunk's AR loop short enough that fp16 drift on
    /// CoreML doesn't cause audible amplitude decay. ~25 tokens ≈ 2s of
    /// audio, well under the ~40-frame drift threshold measured on device.
    static let chunkMaxTokens: Int = 25


    // MARK: - User-facing state

    public var status: String = "Loading model…"
    public var isGenerating: Bool = false
    public var isStreaming: Bool = false
    public var generatedPCM: Data? = nil
    public var text: String = """
        Pocket TTS is a lightweight text-to-speech model. \
        It runs entirely on your iPhone, streaming audio as the voice speaks. \
        Tap Stream to hear it break a longer paragraph into sentences.
        """
    public var voices: [VoiceEntry] = []
    public var selectedVoice: VoiceEntry? = nil
    public var errorMessage: String? = nil
    public var isReady: Bool = false

    /// Stats surfaced in the status card. Values are nil until populated
    /// by load() (initSeconds) or generate() (rest). Reset at the start
    /// of each generate so the card reflects the most recent run.
    public struct Stats {
        public var initSeconds: Double?         = nil  // model load + warmup (one-time)
        public var firstAudioSeconds: Double?   = nil  // wall time to first PCM chunk
        public var generateSeconds: Double?     = nil  // total wall time for this run
        public var audioSeconds: Double?        = nil  // total audio duration generated
        public var rtf: Double?                 = nil  // generateSeconds / audioSeconds
    }
    public var stats = Stats()

    /// The StreamingPlayer is observed directly by the View — we keep a
    /// strong reference here but the view sets up its own @StateObject.
    public let player: StreamingPlayer

    // MARK: - Internal

    private var tts: PocketTTS? = nil
    private var tokenizer: Tokenizer? = nil
    private var runningTask: Task<Void, Never>? = nil

    // MARK: - Init

    public init(player: StreamingPlayer = StreamingPlayer()) {
        self.player = player
        self.voices = VoiceCatalog.bundled()
        self.selectedVoice = voices.first
    }

    /// Async-load the model. Call from a `.task` modifier on the root view.
    public func load() async {
        guard !isReady && errorMessage == nil else { return }
        let initStart = Date()
        do {
            let bundle = Bundle.main
            guard let artifactsDir = bundle.url(
                forResource: "Artifacts", withExtension: nil
            ) else {
                throw DemoError.missingResource(
                    "Artifacts/ directory not bundled. Run prepare_resources.sh."
                )
            }
            guard let tokenizerURL = bundle.url(
                forResource: "tokenizer", withExtension: "model"
            ) else {
                throw DemoError.missingResource(
                    "tokenizer.model not bundled. Run prepare_resources.sh."
                )
            }

            self.tokenizer = try Tokenizer(modelURL: tokenizerURL)

            status = "Compiling CoreML models (first launch is slow)…"
            let model = try await PocketTTS(
                artifactsBundle: artifactsDir,
                tokenizerPath: tokenizerURL,
                computeUnits: .cpuAndNeuralEngine
            )
            self.tts = model

            status = "Warming up…"
            await model.warmup()

            if voices.isEmpty {
                throw DemoError.missingResource(
                    "No voices found in Resources/Voices/. Run prepare_resources.sh."
                )
            }

            self.stats.initSeconds = Date().timeIntervalSince(initStart)
            self.isReady = true
            self.status = "Idle"
        } catch let err as PocketTTSLoadError {
            self.errorMessage = err.description
            self.status = "Model load failed"
        } catch {
            self.errorMessage = error.localizedDescription
            self.status = "Load failed"
        }
    }

    // MARK: - Public actions

    /// Full-utterance generate. Splits text at sentence boundaries (same
    /// TextChunker used by stream()) and concatenates the per-chunk PCM.
    /// This avoids the KV-cache S_cap=256 overflow that hit on long inputs.
    /// Each chunk gets a fresh VoiceHandle (re-prefills voice KV + text).
    public func generate() async {
        guard isReady, !isGenerating, !isStreaming,
              let voice = selectedVoice, let tts = tts,
              let tokenizer = tokenizer else { return }
        cancelRunning()
        isGenerating = true
        errorMessage = nil
        generatedPCM = nil
        // Replace the whole stats struct (preserving one-time initSeconds)
        // so @Observable publishes a single change notification rather
        // than four separate ones. Also guards against any edge case where
        // nested-field mutation isn't picked up by the observer.
        var s = stats
        s.firstAudioSeconds = nil
        s.generateSeconds = nil
        s.audioSeconds = nil
        s.rtf = nil
        stats = s

        let text = preparedText()
        let chunks = TextChunker.splitIntoBestSentences(
            text, tokenizer: tokenizer, maxTokens: Self.chunkMaxTokens
        )
        guard !chunks.isEmpty else {
            status = "Nothing to generate"
            isGenerating = false
            return
        }
        status = chunks.count == 1 ? "Generating…"
                                   : "Generating (1/\(chunks.count))…"

        // Wrap in a plain Task<Void, Never> so it matches runningTask's
        // type. Share the result (and a first-audio timestamp) via actor.
        actor ResultBox {
            var data: Data? = nil
            var error: Error? = nil
            var firstPCMAt: Date? = nil
            func set(_ d: Data) { data = d }
            func fail(_ e: Error) { error = e }
            func markFirstPCMIfNil(_ at: Date) { if firstPCMAt == nil { firstPCMAt = at } }
            func take() -> (Data?, Error?, Date?) { (data, error, firstPCMAt) }
        }
        let box = ResultBox()
        let wallStart = Date()
        let task = Task<Void, Never> {
            do {
                var accum = Data()
                for (idx, chunkText) in chunks.enumerated() {
                    try Task.checkCancellation()
                    await MainActor.run {
                        self.status = "Generating (\(idx + 1)/\(chunks.count))…"
                    }
                    let handle = try await self.voiceHandle(for: voice, on: tts)
                    let pcmStream = await tts.generate(text: chunkText, voice: handle)
                    for try await pcm in pcmStream {
                        try Task.checkCancellation()
                        await box.markFirstPCMIfNil(Date())
                        accum.append(pcm)
                    }
                }
                await box.set(accum)
            } catch {
                await box.fail(error)
            }
        }
        self.runningTask = task
        await task.value

        let (accumOpt, errOpt, firstPCMAt) = await box.take()
        let wallSeconds = Date().timeIntervalSince(wallStart)
        do {
            if let err = errOpt { throw err }
            let accum = accumOpt ?? Data()
            self.generatedPCM = accum
            let audio = Double(accum.count / 2) / Double(PocketTTSArch.sampleRate)
            var s = self.stats
            s.audioSeconds = audio
            s.generateSeconds = wallSeconds
            s.rtf = audio > 0 ? wallSeconds / audio : nil
            if let firstAt = firstPCMAt {
                s.firstAudioSeconds = firstAt.timeIntervalSince(wallStart)
            }
            self.stats = s
            self.status = "Audio ready"
        } catch is CancellationError {
            self.status = "Stopped"
        } catch {
            self.errorMessage = error.localizedDescription
            self.status = "Generate failed"
        }
        isGenerating = false
    }

    /// Play the most recent `generatedPCM`.
    public func play() async {
        guard let pcm = generatedPCM, !pcm.isEmpty else { return }
        do {
            try player.play(fullPCM: pcm, sampleRate: Double(PocketTTSArch.sampleRate))
            status = "Playing…"
        } catch {
            errorMessage = error.localizedDescription
            status = "Playback failed"
        }
    }

    /// Sentence-chunk streaming: split, generate + play chunk-by-chunk with
    /// at most 2 chunks queued ahead of the player.
    public func stream() async {
        guard isReady, !isGenerating, !isStreaming,
              let voice = selectedVoice, let tts = tts,
              let tokenizer = tokenizer else { return }
        cancelRunning()
        errorMessage = nil
        // Reset per-run stats so the card reflects this streaming session.
        var s = stats
        s.firstAudioSeconds = nil
        s.generateSeconds = nil
        s.audioSeconds = nil
        s.rtf = nil
        stats = s

        let text = preparedText()
        // fp16 drift on-device causes AR amplitude decay after ~40 frames
        // (~3s of audio). Cap chunks at ~25 tokens (~2s audio) so each
        // utterance stays well inside the stable range. Reference Python
        // uses 50 at fp32 where drift isn't an issue.
        let chunks = TextChunker.splitIntoBestSentences(
            text, tokenizer: tokenizer, maxTokens: Self.chunkMaxTokens
        )
        guard !chunks.isEmpty else {
            status = "Nothing to stream"
            return
        }

        isStreaming = true
        status = "Streaming (1/\(chunks.count))…"
        let wallStart = Date()
        do {
            try await runStream(
                chunks: chunks, voice: voice, tts: tts, wallStart: wallStart
            )
        } catch is CancellationError {
            status = "Stopped"
        } catch {
            errorMessage = error.localizedDescription
            status = "Stream failed"
        }
        isStreaming = false
    }

    /// Stop a running stream. Drains the player to finish any already-
    /// scheduled chunk, per the task requirements.
    public func stop() {
        cancelRunning()
        player.stopStream()
        if isStreaming || isGenerating {
            status = "Stopped"
        }
        isGenerating = false
        isStreaming = false
    }

    // MARK: - Streaming machinery

    private func runStream(
        chunks: [String], voice: VoiceEntry, tts: PocketTTS, wallStart: Date
    ) async throws {
        try player.startStream(totalChunks: chunks.count)

        // Back-pressured async sequence: at most 2 chunks produced ahead
        // of consumption. We use a tiny actor-based buffer.
        let channel = BoundedChannel<(Int, Data)>(capacity: 2)

        // Shared mutable state between producer + consumer: first-pcm
        // timestamp and total bytes emitted. Use an actor so both tasks
        // can update it safely.
        actor StreamMetrics {
            var firstPCMAt: Date? = nil
            var totalBytes: Int = 0
            func markFirstPCMIfNil(_ at: Date) { if firstPCMAt == nil { firstPCMAt = at } }
            func addBytes(_ n: Int) { totalBytes += n }
            func take() -> (Date?, Int) { (firstPCMAt, totalBytes) }
        }
        let metrics = StreamMetrics()

        // Producer task: generate each chunk's full PCM, push to channel.
        // IMPORTANT: fresh VoiceHandle per chunk. The orchestrator mutates
        // the handle's KV cache in place during runTextPrefill + AR loop,
        // so reusing a handle across chunks poisons the KV for chunks 2+.
        let producer = Task<Void, Error> {
            do {
                for (idx, chunkText) in chunks.enumerated() {
                    try Task.checkCancellation()
                    let handle = try await self.voiceHandle(for: voice, on: tts)
                    let pcmStream = await tts.generate(text: chunkText, voice: handle)
                    var accum = Data()
                    for try await pcm in pcmStream {
                        try Task.checkCancellation()
                        await metrics.markFirstPCMIfNil(Date())
                        accum.append(pcm)
                    }
                    await metrics.addBytes(accum.count)
                    try await channel.send((idx, accum))
                    _ = idx  // idx used only for enumerated()
                }
                await channel.close()
            } catch {
                await channel.close()
                throw error
            }
        }

        self.runningTask = Task { [weak self] in
            _ = try? await producer.value
            _ = self  // silence capture warning
        }

        // Consumer (this coroutine): pull each chunk, push to player, update
        // status label with current chunk index.
        do {
            while let (idx, pcm) = try await channel.receive() {
                try Task.checkCancellation()
                self.status = "Streaming (\(idx + 1)/\(chunks.count))…"
                // First time we push a chunk to the player, we know the
                // first audio has been generated — publish stats now so
                // the card updates while the stream is still playing.
                let (firstAt, totalBytes) = await metrics.take()
                if let firstAt = firstAt, self.stats.firstAudioSeconds == nil {
                    var s = self.stats
                    s.firstAudioSeconds = firstAt.timeIntervalSince(wallStart)
                    self.stats = s
                }
                player.pushChunk(pcm, sampleRate: Double(PocketTTSArch.sampleRate))
                _ = totalBytes  // final totals updated after loop
            }
        } catch {
            producer.cancel()
            throw error
        }

        // All chunks scheduled. Wait for the player to drain naturally.
        while player.isPlaying {
            try await Task.sleep(nanoseconds: 100_000_000)
            if Task.isCancelled { throw CancellationError() }
        }

        // Publish final stats once playback has drained. Generation wall
        // time is measured from `wallStart` (stream() entry) to the
        // moment the producer finished its last chunk — which is
        // approximately `now - drainTime`. We use `now` as an
        // overestimate and accept the small additional drain tail as
        // part of "time until user hears last audio".
        let (firstAt, totalBytes) = await metrics.take()
        var s = self.stats
        let audio = Double(totalBytes / 2) / Double(PocketTTSArch.sampleRate)
        s.audioSeconds = audio
        s.generateSeconds = Date().timeIntervalSince(wallStart)
        s.rtf = audio > 0 ? s.generateSeconds! / audio : nil
        if let firstAt = firstAt, s.firstAudioSeconds == nil {
            s.firstAudioSeconds = firstAt.timeIntervalSince(wallStart)
        }
        self.stats = s

        status = "Done"
    }

    private func cancelRunning() {
        runningTask?.cancel()
        runningTask = nil
    }

    // MARK: - Voice handle

    /// Load a fresh VoiceHandle from disk every call. We do NOT cache
    /// handles: `PocketTTS.generate` mutates the handle's KV cache in
    /// place (runs runTextPrefill + AR loop directly on the handle's
    /// buffer), so a cached handle is single-use. The safetensors parse
    /// takes ~10 ms which is negligible compared to generation time.
    private func voiceHandle(
        for entry: VoiceEntry, on tts: PocketTTS
    ) async throws -> VoiceHandle {
        return try await tts.loadVoice(from: entry.url)
    }

    /// Write `generatedPCM` to a WAV file in the user's tmp dir and return
    /// the URL. Returns nil if there is no audio yet. Used by the Share
    /// button to hand a URL to UIActivityViewController.
    public func exportGeneratedAsWav() -> URL? {
        guard let pcm = generatedPCM, !pcm.isEmpty else { return nil }
        let sr: UInt32 = UInt32(PocketTTSArch.sampleRate)
        let stem = (selectedVoice?.id ?? "voice") + "-\(Int(Date().timeIntervalSince1970))"
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("pockettts-\(stem).wav")
        var data = Data()
        data.append(wavHeader(samples: pcm.count / 2, sampleRate: sr))
        data.append(pcm)
        do {
            try data.write(to: url)
            return url
        } catch {
            return nil
        }
    }

    private func wavHeader(samples: Int, sampleRate: UInt32) -> Data {
        // RIFF/WAV PCM16 mono header.
        let byteRate = sampleRate * 2           // 1 channel * 2 bytes/sample
        let blockAlign: UInt16 = 2
        let subchunk2: UInt32 = UInt32(samples) * 2
        let chunkSize: UInt32 = 36 + subchunk2
        var h = Data()
        h.append("RIFF".data(using: .ascii)!)
        h.append(UInt32(chunkSize).littleEndianData)
        h.append("WAVE".data(using: .ascii)!)
        h.append("fmt ".data(using: .ascii)!)
        h.append(UInt32(16).littleEndianData)   // subchunk1 size
        h.append(UInt16(1).littleEndianData)    // PCM format
        h.append(UInt16(1).littleEndianData)    // channels
        h.append(sampleRate.littleEndianData)
        h.append(byteRate.littleEndianData)
        h.append(blockAlign.littleEndianData)
        h.append(UInt16(16).littleEndianData)   // bits per sample
        h.append("data".data(using: .ascii)!)
        h.append(subchunk2.littleEndianData)
        return h
    }

    private func preparedText() -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "Hello from PocketTTS."
            : text
    }

    public enum DemoError: LocalizedError {
        case missingResource(String)
        public var errorDescription: String? {
            switch self {
            case .missingResource(let m): return "Missing resource: \(m)"
            }
        }
    }
}

// MARK: - Bounded async channel

/// Tiny bounded channel used to back-pressure sentence-chunk generation so
/// the generator is never more than `capacity` chunks ahead of the player.
/// Send awaits if full; receive awaits if empty; close wakes pending
/// receivers with nil.
actor BoundedChannel<Element: Sendable> {
    private var buffer: [Element] = []
    private let capacity: Int
    private var closed: Bool = false
    private var sendWaiters: [CheckedContinuation<Void, Error>] = []
    private var recvWaiters: [CheckedContinuation<Element?, Error>] = []

    init(capacity: Int) {
        precondition(capacity > 0)
        self.capacity = capacity
    }

    func send(_ element: Element) async throws {
        if closed { throw CancellationError() }
        if let waiter = recvWaiters.first {
            recvWaiters.removeFirst()
            waiter.resume(returning: element)
            return
        }
        if buffer.count < capacity {
            buffer.append(element)
            return
        }
        try await withCheckedThrowingContinuation { cont in
            sendWaiters.append(cont)
        }
        // After resume, re-attempt the send (guaranteed space or closed).
        if closed { throw CancellationError() }
        if buffer.count < capacity {
            buffer.append(element)
        } else if let waiter = recvWaiters.first {
            recvWaiters.removeFirst()
            waiter.resume(returning: element)
        }
    }

    func receive() async throws -> Element? {
        if !buffer.isEmpty {
            let elem = buffer.removeFirst()
            if let waiter = sendWaiters.first {
                sendWaiters.removeFirst()
                waiter.resume()
            }
            return elem
        }
        if closed { return nil }
        return try await withCheckedThrowingContinuation { cont in
            recvWaiters.append(cont)
        }
    }

    func close() {
        guard !closed else { return }
        closed = true
        for w in recvWaiters { w.resume(returning: nil) }
        recvWaiters.removeAll()
        for w in sendWaiters { w.resume(throwing: CancellationError()) }
        sendWaiters.removeAll()
    }
}

// MARK: - WAV helpers

private extension FixedWidthInteger {
    var littleEndianData: Data {
        withUnsafeBytes(of: self.littleEndian) { Data($0) }
    }
}
