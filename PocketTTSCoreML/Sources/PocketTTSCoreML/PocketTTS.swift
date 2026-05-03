import Foundation
import CoreML

public enum PocketTTSLoadError: Error, CustomStringConvertible {
    case modelNotFound(String)
    public var description: String {
        switch self {
        case .modelNotFound(let msg): return "PocketTTSLoadError.modelNotFound: \(msg)"
        }
    }
}

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
    /// Lazy voice cloner — created on the FIRST cloneVoice() call and
    /// cached for all subsequent clones. mimi_encoder loading + ANE
    /// program prep costs ~10-25s on iPhone A19 Pro for the 24L French
    /// bundle (~5s for 6L bundles), so we don't preload at PocketTTS
    /// init — users who never clone don't pay the price. Second and
    /// later clones in the same session reuse the warm encoder.
    private var voiceCloner: VoiceCloner?

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

        guard let flowMainURL = Self.resolveArtifact(artifactsBundle, stem: "flow_lm_main"),
              let flowFlowURL = Self.resolveArtifact(artifactsBundle, stem: "flow_lm_flow"),
              let mimiDecURL  = Self.resolveArtifact(artifactsBundle, stem: "mimi_decoder")
        else {
            throw PocketTTSLoadError.modelNotFound(
                "Required mlpackage/mlmodelc not present in: \(artifactsBundle.path)"
            )
        }
        let flowMain = try await Self.loadCompiled(flowMainURL, config: cfg)
        let flowFlow = try await Self.loadCompiled(flowFlowURL, config: cfg)
        let mimiDec  = try await Self.loadCompiled(mimiDecURL,  config: cfg)

        // Resolve layer count from the loaded flow_lm_main input shape so
        // that 24L french_24l and 6L en/es/de/it/pt all work. Must happen
        // BEFORE anything that reads PocketTTSArch.flowLayers (voice load,
        // KV alloc, prefill mask builders).
        PocketTTSArch.configureFlowLayers(from: flowMain)

        let layoutURL = artifactsBundle.appendingPathComponent("mimi_decoder.state_layout.json")
        let layout = try MimiStateLayout.load(from: layoutURL)

        // Optional models enabling in-Swift text prefill (Phase 4B). If
        // either is missing, only `.prefilled` VoiceHandle loads remain
        // functional — `.voiceOnly` loads will throw at generate time.
        var flowPrefill: MLModel? = nil
        var textCond: MLModel? = nil
        var defaultBos: [Float]? = nil

        // Accept either a `.mlpackage` (compiled at runtime) or a
        // pre-compiled `.mlmodelc` sibling in the bundle. iOS bundles are
        // read-only so runtime compilation can't be cached — pre-compile
        // on macOS via `xcrun coremlcompiler compile` and ship the
        // `.mlmodelc` directory directly.
        let prefillURL = Self.resolveArtifact(artifactsBundle, stem: "flow_lm_prefill")
        if let url = prefillURL {
            flowPrefill = try await Self.loadCompiled(url, config: cfg)
        }
        let tcURL = Self.resolveArtifact(artifactsBundle, stem: "text_conditioner")
        if let url = tcURL {
            textCond = try await Self.loadCompiled(url, config: cfg)
        }
        let bosURL = artifactsBundle.appendingPathComponent("flow_lm_bos_emb.safetensors")
        if FileManager.default.fileExists(atPath: bosURL.path) {
            let reader = try SafetensorsReader(url: bosURL)
            let (bos, _) = try reader.float32Array(for: "bos_emb")
            defaultBos = bos
        }

        // Optional speaker_proj sidecar — required for voice cloning
        // (stage 2). If missing, `cloneVoice` throws but other paths work.
        let projURL = artifactsBundle.appendingPathComponent("speaker_proj.safetensors")
        let speakerProj: SpeakerProjection? = try VoiceLoader.loadSpeakerProjection(from: projURL)

        self.orchestrator = Orchestrator(models: .init(
            flowMain: flowMain, flowFlow: flowFlow,
            mimiDecoder: mimiDec, mimiLayout: layout,
            flowPrefill: flowPrefill, textConditioner: textCond,
            defaultBosEmb: defaultBos,
            speakerProjection: speakerProj
        ))

        // mimi_encoder is loaded lazily on first cloneVoice. Preloading
        // it here adds 10-25s to app startup which is unacceptable for
        // users who never clone.
        self.voiceCloner = nil
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

    /// Resolve a model artifact by stem (e.g. `"flow_lm_main"`). Prefers a
    /// pre-compiled `.mlmodelc` in the bundle; falls back to `.mlpackage`
    /// for dev builds. Returns nil if neither exists.
    private static func resolveArtifact(_ bundle: URL, stem: String) -> URL? {
        let fm = FileManager.default
        let mlmodelc = bundle.appendingPathComponent("\(stem).mlmodelc")
        if fm.fileExists(atPath: mlmodelc.path) { return mlmodelc }
        let mlpackage = bundle.appendingPathComponent("\(stem).mlpackage")
        if fm.fileExists(atPath: mlpackage.path) { return mlpackage }
        return nil
    }

    // MARK: - API surface

    public func loadVoice(from url: URL) throws -> VoiceHandle {
        try VoiceLoader.load(url: url)
    }

    public func saveVoice(_ handle: VoiceHandle, to url: URL) throws {
        try VoiceLoader.save(handle, to: url)
    }

    /// Run the full two-stage voice cloning pipeline (mimi_encoder →
    /// speaker projection → flow_lm_prefill) on a reference waveform.
    /// Returns a `.voiceOnly` handle that can be passed directly to
    /// `generate(...)` without any Python helper.
    ///
    /// Requires the artifacts bundle to contain:
    ///   - `mimi_encoder.mlpackage` (or compiled `.mlmodelc`)
    ///   - `flow_lm_prefill.mlpackage`
    ///   - `speaker_proj.safetensors`  (see `export_speaker_proj.py`)
    public func cloneVoice(from audioURL: URL) async throws -> VoiceHandle {
        let cloner = try await ensureVoiceCloner()
        return try await cloner.clone(from: audioURL)
    }

    /// Build (and cache) the VoiceCloner on first use. The first call
    /// pays the ~5-25s mimi_encoder MLModel init + ANE program prep
    /// cost (6L ~5s, 24L french ~25s on A19 Pro). Subsequent calls
    /// return the cached instance so the encoder stays warm.
    private func ensureVoiceCloner() async throws -> VoiceCloner {
        if let cached = voiceCloner { return cached }

        guard let speakerProj = orchestrator.models.speakerProjection else {
            throw PocketTTSLoadError.modelNotFound(
                "speaker_proj.safetensors not found in \(artifactsBundle.path); "
                + "voice cloning unavailable for this bundle. Regenerate via "
                + "`python -m pockettts_coreml.convert.export_speaker_proj "
                + "--language <lang> --out <artifacts_dir>`"
            )
        }
        guard let encoderURL = Self.resolveArtifact(artifactsBundle, stem: "mimi_encoder") else {
            throw PocketTTSLoadError.modelNotFound(
                "mimi_encoder not present in \(artifactsBundle.path); "
                + "voice cloning requires it."
            )
        }

        // mimi_encoder is heavy CPU-fallback (SEANet strides 6/5/4 inverse +
        // depthwise ConvTrUpsample1d are ANE-unsupported per Phase 2+3 notes).
        // Loading it with .cpuAndNeuralEngine makes iOS CoreML attempt an
        // ANE program compile that loops on unsupported ops and can hang
        // indefinitely. Force .cpuOnly — the encoder was measured at
        // ~70 ms/predict on CPU anyway, and clones are off the hot path.
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuOnly

        let t0 = Date()
        FileHandle.standardError.write(Data(
            "[clone-init] loading mimi_encoder from \(encoderURL.lastPathComponent) (computeUnits=.cpuOnly)\n".utf8
        ))
        let encoder = try await Self.loadCompiled(encoderURL, config: cfg)
        FileHandle.standardError.write(Data(
            String(format: "[clone-init] mimi_encoder MLModel ready in %.2fs\n",
                   Date().timeIntervalSince(t0)).utf8
        ))
        let cloner = VoiceCloner(
            encoder: encoder,
            orchestrator: orchestrator,
            speakerProjection: speakerProj
        )
        self.voiceCloner = cloner
        return cloner
    }

    /// Encode text to SentencePiece ids (exposed for tests / parity checks).
    public nonisolated func tokenize(_ text: String) -> [Int32] {
        tokenizer.encode(text)
    }

    /// Generate audio for `text`, streaming PCM16 LE 24 kHz mono.
    ///
    /// Behavior depends on the voice handle flavor:
    /// - `.voiceOnly`: text is tokenized and a Swift-native prefill step
    ///   (text_conditioner + flow_lm_prefill) runs before the AR loop.
    /// - `.prefilled`: the `text` argument is ignored (text prefill was
    ///   baked into the KV cache by `export_full_prefill.py`).
    public func generate(
        text: String,
        voice: VoiceHandle,
        options: GenerateOptions = .default
    ) -> AsyncThrowingStream<Data, Error> {
        let tokens: [Int32]
        switch voice.kind {
        case .voiceOnly: tokens = tokenizer.encode(text)
        case .prefilled: tokens = []  // ignored for pre-prefilled bundles
        }
        return orchestrator.generate(
            voice: voice, textTokens: tokens,
            options: options, maxGenLen: options.maxGenLen
        )
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
