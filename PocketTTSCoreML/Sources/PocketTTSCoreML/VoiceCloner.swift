import AVFoundation
import Accelerate
import CoreML
import Foundation

/// Two-stage voice cloning pipeline driven entirely in Swift.
///
/// Stage 1: `mimi_encoder.mlpackage` runs on a 4-second reference waveform
/// (resampled/padded/cropped to exactly 96 000 samples @ 24 kHz) to produce
/// Mimi latents of shape `fp16[1, 32, T_latent]` (T_latent = 50 for 4 s).
///
/// Stage 2: latents are projected to `d_model` (1024) via the learned
/// `speaker_proj_weight` linear (loaded from `speaker_proj.safetensors`),
/// optionally prepended with the `bos_before_voice` vector, and fed to
/// `flow_lm_prefill.mlpackage` to populate the voice KV cache at slots
/// `[0, T_voice)`. The returned `VoiceHandle.voiceOnly` can then be passed
/// directly to `PocketTTS.generate(...)`.
///
/// This mirrors the reference Python pipeline — see `TTSModel._encode_audio`
/// + `TTSModel.get_state_for_audio_prompt` in
/// `pockettts_coreml/reference/pocket_tts/models/tts_model.py`.
public final class VoiceCloner: @unchecked Sendable {
    public let encoder: MLModel
    public let orchestrator: Orchestrator
    public let speakerProj: SpeakerProjection

    /// Preferred initializer: reuses a pre-loaded mimi_encoder MLModel so
    /// repeat clones in the same session don't pay the ~2-3s instance-
    /// init + ANE program hand-off cost. `PocketTTS.init` loads the
    /// encoder once and caches a VoiceCloner on the actor for all
    /// subsequent clone calls.
    public init(
        encoder: MLModel,
        orchestrator: Orchestrator,
        speakerProjection: SpeakerProjection
    ) {
        self.encoder = encoder
        self.orchestrator = orchestrator
        self.speakerProj = speakerProjection
    }

    /// Legacy path: constructs its own encoder from a URL. Each call to
    /// this initializer pays MLModel load + ANE program init cost, so
    /// prefer the encoder-passed form above for per-clone reuse.
    public init(
        mimiEncoderURL: URL,
        orchestrator: Orchestrator,
        speakerProjection: SpeakerProjection,
        computeUnits: MLComputeUnits
    ) async throws {
        self.orchestrator = orchestrator
        self.speakerProj = speakerProjection

        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        let cached = mimiEncoderURL.deletingPathExtension().appendingPathExtension("mlmodelc")
        let fm = FileManager.default
        if fm.fileExists(atPath: cached.path) {
            self.encoder = try MLModel(contentsOf: cached, configuration: cfg)
            return
        }
        let compiled = try await MLModel.compileModel(at: mimiEncoderURL)
        do {
            if fm.fileExists(atPath: cached.path) { try fm.removeItem(at: cached) }
            try fm.moveItem(at: compiled, to: cached)
            self.encoder = try MLModel(contentsOf: cached, configuration: cfg)
        } catch {
            self.encoder = try MLModel(contentsOf: compiled, configuration: cfg)
        }
    }

    /// Run the full two-stage clone on a reference waveform and return a
    /// `.voiceOnly` VoiceHandle with the voice KV populated.
    public func clone(from audioURL: URL) async throws -> VoiceHandle {
        let t0 = Date()
        func log(_ phase: String, _ startedAt: Date) {
            let dt = Date().timeIntervalSince(startedAt)
            let total = Date().timeIntervalSince(t0)
            let msg = String(
                format: "[clone] %@ took %.2fs (total %.2fs)\n",
                phase, dt, total)
            FileHandle.standardError.write(Data(msg.utf8))
        }

        // --- Stage 1: load + resample + mimi_encoder ------------------------
        let tLoad = Date()
        let samples = try Self.loadMonoFloat32_24k(from: audioURL)
        log("load+resample (input \(samples.count) samples)", tLoad)

        let target = 96_000
        var buffer = [Float](repeating: 0, count: target)
        let copy = min(samples.count, target)
        for i in 0 ..< copy { buffer[i] = samples[i] }

        let waveform = try MLMultiArray(
            shape: [1, 1, NSNumber(value: target)], dataType: .float16
        )
        let dstPtr = waveform.dataPointer.bindMemory(to: UInt16.self, capacity: target)
        buffer.withUnsafeBufferPointer { src in
            Float16Ops.convertFp32ToFp16(srcPtr: src.baseAddress!, dstPtr: dstPtr, count: target)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "waveform": MLFeatureValue(multiArray: waveform)
        ])

        let tEnc = Date()
        let result = try autoreleasepool { try encoder.prediction(from: provider) }
        log("mimi_encoder predict", tEnc)

        guard let latents = result.featureValue(for: "latents")?.multiArrayValue else {
            throw NSError(
                domain: "VoiceCloner", code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey: "mimi_encoder returned no latents"
                ])
        }

        // --- Stage 2a: apply speaker projection ----------------------------
        let tProj = Date()
        let (projected, _) = try VoiceProjection.applySpeakerProjection(
            latents: latents, proj: speakerProj.weight
        )
        let (conditioning, tVoice) = try VoiceProjection.prependBosBeforeVoice(
            conditioning: projected, bosBeforeVoice: speakerProj.bosBeforeVoice
        )
        log("speaker_proj + BOS prepend (T_voice=\(tVoice))", tProj)

        // --- Stage 2b: allocate zero-filled KV, run flow_lm_prefill --------
        let tKV = Date()
        let kvShape: [NSNumber] = [
            NSNumber(value: 2 * PocketTTSArch.flowLayers), 1,
            NSNumber(value: PocketTTSArch.flowSCap),
            NSNumber(value: PocketTTSArch.flowHeads),
            NSNumber(value: PocketTTSArch.flowHeadDim),
        ]
        let voiceKV = try MLMultiArray(shape: kvShape, dataType: .float16)
        let total = voiceKV.count
        let kvPtr = voiceKV.dataPointer.bindMemory(to: UInt16.self, capacity: total)
        for i in 0 ..< total { kvPtr[i] = 0 }
        log("alloc+zero KV", tKV)

        let tPrefill = Date()
        let voiceOffset = try orchestrator.runVoicePrefill(
            voiceKV: voiceKV, voiceConditioning: conditioning
        )
        log("flow_lm_prefill predict", tPrefill)
        precondition(voiceOffset == tVoice)

        log("TOTAL clone() excl VoiceCloner init", t0)
        return VoiceHandle(
            kind: .voiceOnly(
                flowKVRank5: MLMultiArrayBox(voiceKV),
                voiceOffset: voiceOffset,
                bosEmb: nil  // orchestrator falls back to the default sidecar
            ),
            sourceURL: audioURL
        )
    }

    // MARK: - Audio loading

    static func loadMonoFloat32_24k(from url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let srcFormat = file.processingFormat
        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(PocketTTSArch.sampleRate),
            channels: 1,
            interleaved: false
        )!

        let needResample =
            srcFormat.sampleRate != targetFormat.sampleRate
            || srcFormat.channelCount != 1
        if !needResample {
            guard
                let buf = AVAudioPCMBuffer(
                    pcmFormat: srcFormat, frameCapacity: AVAudioFrameCount(file.length))
            else {
                throw NSError(domain: "VoiceCloner", code: 2)
            }
            try file.read(into: buf)
            let n = Int(buf.frameLength)
            guard let ch = buf.floatChannelData?[0] else {
                throw NSError(domain: "VoiceCloner", code: 3)
            }
            return Array(UnsafeBufferPointer(start: ch, count: n))
        }

        guard let conv = AVAudioConverter(from: srcFormat, to: targetFormat) else {
            throw NSError(
                domain: "VoiceCloner", code: 4,
                userInfo: [
                    NSLocalizedDescriptionKey: "Cannot create AVAudioConverter"
                ])
        }
        guard
            let src = AVAudioPCMBuffer(
                pcmFormat: srcFormat,
                frameCapacity: AVAudioFrameCount(file.length))
        else { throw NSError(domain: "VoiceCloner", code: 5) }
        try file.read(into: src)

        let ratio = targetFormat.sampleRate / srcFormat.sampleRate
        let outCapacity = AVAudioFrameCount(Double(src.frameLength) * ratio + 1024)
        guard let dst = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outCapacity)
        else { throw NSError(domain: "VoiceCloner", code: 6) }

        var err: NSError?
        var done = false
        conv.convert(to: dst, error: &err) { _, status in
            if done {
                status.pointee = .endOfStream
                return nil
            }
            status.pointee = .haveData
            done = true
            return src
        }
        if let e = err { throw e }

        let n = Int(dst.frameLength)
        guard let ch = dst.floatChannelData?[0] else {
            throw NSError(domain: "VoiceCloner", code: 7)
        }
        return Array(UnsafeBufferPointer(start: ch, count: n))
    }
}
