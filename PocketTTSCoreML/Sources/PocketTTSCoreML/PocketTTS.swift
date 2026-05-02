import Foundation
import CoreML

/// Public entry point. Loads the 5 `.mlpackage` bundles + tokenizer + Mimi
/// state layout, and exposes `generate`/`loadVoice`/`cloneVoice`.
///
/// See `README.md` for Phase-4A caveats. Known limitation: text prefill in
/// Swift is not yet implemented — `generate(text:)` requires a pre-prefilled
/// VoiceHandle when using an arbitrary prompt. The helper
/// `tools/export_full_prefill.py` produces one from a voice + prompt pair.
public actor PocketTTS {
    public struct GenerateOptions: Sendable {
        public var lsdDecodeSteps: Int = 1
        public var temperature: Float = PocketTTSArch.defaultTemperature
        public var eosThreshold: Float = PocketTTSArch.defaultEosThreshold
        public var framesAfterEos: Int = PocketTTSArch.defaultFramesAfterEos
        public var noiseClamp: Float? = nil
        public var seed: UInt64? = nil
        public var maxGenLen: Int = 512

        public init() {}
        public static let `default` = GenerateOptions()
    }

    public let artifactsBundle: URL
    public let tokenizerPath: URL
    public let computeUnits: MLComputeUnits

    private let tokenizer: Tokenizer
    private let orchestrator: Orchestrator

    public init(
        artifactsBundle: URL,
        tokenizerPath: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws {
        self.artifactsBundle = artifactsBundle
        self.tokenizerPath = tokenizerPath
        self.computeUnits = computeUnits

        self.tokenizer = try Tokenizer(modelURL: tokenizerPath)

        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits

        let flowMain = try await Self.loadCompiled(
            artifactsBundle.appendingPathComponent("flow_lm_main.mlpackage"),
            config: cfg
        )
        let flowFlow = try await Self.loadCompiled(
            artifactsBundle.appendingPathComponent("flow_lm_flow.mlpackage"),
            config: cfg
        )
        let mimiDec = try await Self.loadCompiled(
            artifactsBundle.appendingPathComponent("mimi_decoder.mlpackage"),
            config: cfg
        )

        let layoutURL = artifactsBundle.appendingPathComponent("mimi_decoder.state_layout.json")
        let layout = try MimiStateLayout.load(from: layoutURL)

        self.orchestrator = Orchestrator(models: .init(
            flowMain: flowMain, flowFlow: flowFlow,
            mimiDecoder: mimiDec, mimiLayout: layout
        ))
    }

    // MARK: - Private model loading

    /// Compile an `.mlpackage` (or `.mlmodel`) on first load, cache the
    /// compiled `.mlmodelc` alongside the package so subsequent launches
    /// are quick.
    private static func loadCompiled(_ url: URL, config: MLModelConfiguration) async throws -> MLModel {
        let cached = url
            .deletingPathExtension()
            .appendingPathExtension("mlmodelc")
        let fm = FileManager.default
        if fm.fileExists(atPath: cached.path) {
            return try MLModel(contentsOf: cached, configuration: config)
        }
        let compiled = try await MLModel.compileModel(at: url)
        // Move/copy into `cached` for persistence.
        do {
            if fm.fileExists(atPath: cached.path) {
                try fm.removeItem(at: cached)
            }
            try fm.moveItem(at: compiled, to: cached)
            return try MLModel(contentsOf: cached, configuration: config)
        } catch {
            // Fall back to loading the tmp-located compile result.
            return try MLModel(contentsOf: compiled, configuration: config)
        }
    }

    // MARK: - API surface

    public func loadVoice(from url: URL) throws -> VoiceHandle {
        try VoiceLoader.load(url: url)
    }

    public func saveVoice(_ handle: VoiceHandle, to url: URL) throws {
        try VoiceLoader.save(handle, to: url)
    }

    /// Phase 4A: cloning returns a voice-only handle. The caller must still
    /// run the Python prefill helper to produce a prefilled bundle before
    /// invoking `generate`. This will be closed out in Phase 4B when the
    /// prefill is ported to Swift.
    public func cloneVoice(from audioURL: URL) async throws -> VoiceHandle {
        let cloner = try await VoiceCloner(
            mimiEncoderURL: artifactsBundle.appendingPathComponent("mimi_encoder.mlpackage"),
            computeUnits: computeUnits
        )
        return try await cloner.clone(from: audioURL)
    }

    /// Encode text to SentencePiece ids (exposed for tests / parity checks).
    public nonisolated func tokenize(_ text: String) -> [Int32] {
        tokenizer.encode(text)
    }

    /// Generate audio for `text`, streaming PCM16 LE 24 kHz mono.
    ///
    /// - Note: the `voice` must be a `.prefilled` handle matching the
    ///   `text`. Phase 4A does not do text prefill in Swift — see README.
    public func generate(
        text: String,
        voice: VoiceHandle,
        options: GenerateOptions = .default
    ) -> AsyncThrowingStream<Data, Error> {
        orchestrator.generate(voice: voice, options: options, maxGenLen: options.maxGenLen)
    }

    public func warmup() async {
        // Run a single zeroed-input step on flow_lm_flow to eagerly load
        // weights / JIT the ANE path. A no-op if not supported; errors
        // are swallowed. flow_lm_flow I/O is fp16.
        do {
            let c = try MLMultiArray(shape: [1, NSNumber(value: PocketTTSArch.dModel)], dataType: .float16)
            let s = try MLMultiArray(shape: [1, 1], dataType: .float16)
            let t = try MLMultiArray(shape: [1, 1], dataType: .float16)
            let x = try MLMultiArray(shape: [1, NSNumber(value: PocketTTSArch.latentDim)], dataType: .float16)
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "c": MLFeatureValue(multiArray: c),
                "s": MLFeatureValue(multiArray: s),
                "t": MLFeatureValue(multiArray: t),
                "x": MLFeatureValue(multiArray: x),
            ])
            _ = try await orchestrator.models.flowFlow.prediction(from: provider)
        } catch {
            // ignore — warmup is best-effort
        }
    }
}
