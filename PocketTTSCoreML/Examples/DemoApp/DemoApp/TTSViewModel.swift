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
        // type (used by stop()/cancelRunning()). We share the result via
        // an actor-held box.
        actor ResultBox {
            var data: Data? = nil
            var error: Error? = nil
            func set(_ d: Data) { data = d }
            func fail(_ e: Error) { error = e }
            func take() -> (Data?, Error?) { (data, error) }
        }
        let box = ResultBox()
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

        let (accumOpt, errOpt) = await box.take()
        do {
            if let err = errOpt { throw err }
            let accum = accumOpt ?? Data()
            self.generatedPCM = accum
            let seconds = Double(accum.count / 2) / Double(PocketTTSArch.sampleRate)
            self.status = String(format: "Generated %.2fs of audio — tap Play", seconds)
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
        do {
            try await runStream(chunks: chunks, voice: voice, tts: tts)
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
        chunks: [String], voice: VoiceEntry, tts: PocketTTS
    ) async throws {
        try player.startStream(totalChunks: chunks.count)

        // Back-pressured async sequence: at most 2 chunks produced ahead
        // of consumption. We use a tiny actor-based buffer.
        let channel = BoundedChannel<(Int, Data)>(capacity: 2)

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
                        accum.append(pcm)
                    }
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
                player.pushChunk(pcm, sampleRate: Double(PocketTTSArch.sampleRate))
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
