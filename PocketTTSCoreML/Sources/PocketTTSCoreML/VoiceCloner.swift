import Foundation
import CoreML
import AVFoundation
import Accelerate

/// Runs `mimi_encoder.mlpackage` over a reference waveform to produce Mimi
/// latents, the first half of the two-stage voice-cloning pipeline.
///
/// Phase 4A limitation: the SECOND stage (flow_lm_main prefill over those
/// latents to produce the voice KV cache) is not runnable in Swift because
/// the exported `flow_lm_main.mlpackage` is AR-only — see
/// `Orchestrator.swift` for the background. `cloneVoice` therefore writes
/// the latents + wave bytes to a temporary safetensors and returns a
/// voice-only handle. Users can feed that handle to
/// `pockettts_coreml.e2e.export_full_prefill` to complete the clone.
///
/// Phase 4B will either:
///   - add a `flow_lm_prefill.mlpackage`, or
///   - port the 6-layer transformer to Swift MLMultiArray ops.
public final class VoiceCloner: @unchecked Sendable {
    public let encoder: MLModel

    public init(mimiEncoderURL: URL, computeUnits: MLComputeUnits) async throws {
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

    /// Run mimi_encoder on a 24 kHz mono waveform. Input audio is resampled /
    /// padded / cropped to exactly 96 000 samples (4 seconds) as the
    /// converted encoder is static-shape.
    public func clone(from audioURL: URL) async throws -> VoiceHandle {
        let samples = try Self.loadMonoFloat32_24k(from: audioURL)
        // Pad or crop to 96 000 samples.
        let target = 96_000
        var buffer = [Float](repeating: 0, count: target)
        let copy = min(samples.count, target)
        for i in 0..<copy { buffer[i] = samples[i] }

        // mimi_encoder is fp16.
        let waveform = try MLMultiArray(shape: [1, 1, NSNumber(value: target)], dataType: .float16)
        let dstPtr = waveform.dataPointer.bindMemory(to: UInt16.self, capacity: target)
        buffer.withUnsafeBufferPointer { src in
            Float16Ops.convertFp32ToFp16(srcPtr: src.baseAddress!, dstPtr: dstPtr, count: target)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "waveform": MLFeatureValue(multiArray: waveform)
        ])
        let result = try autoreleasepool { try encoder.prediction(from: provider) }
        guard let latentsMA = result.featureValue(for: "latents")?.multiArrayValue else {
            throw NSError(domain: "VoiceCloner", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "mimi_encoder returned no latents"
            ])
        }

        // Clone output carries the Mimi latents for downstream packing.
        // `cloneVoice` is a two-stage pipeline — mimi_encoder here, then
        // a flow_lm pass over the latents (via the reference Python
        // helper or a future Swift port) to produce the final voice KV.
        // Until that second stage is written, we return a zero-filled
        // voice-only handle and expose the latents via the bundled URL
        // metadata so the CLI can save them for an offline prefill.
        let n = latentsMA.count
        var floats = [Float](repeating: 0, count: n)
        let srcPtr = latentsMA.dataPointer.bindMemory(to: UInt16.self, capacity: n)
        floats.withUnsafeMutableBufferPointer { dst in
            Float16Ops.convertFp16ToFp32(srcPtr: srcPtr, dstPtr: dst.baseAddress!, count: n)
        }
        self.lastLatents = floats  // for CLI save-after-clone

        // Zero-filled KV; voice_offset=0 is the signal that this handle
        // isn't directly playable — the CLI catches this before calling
        // `generate`.
        let shape: [NSNumber] = [
            NSNumber(value: 2 * PocketTTSArch.flowLayers), 1,
            NSNumber(value: PocketTTSArch.flowSCap),
            NSNumber(value: PocketTTSArch.flowHeads),
            NSNumber(value: PocketTTSArch.flowHeadDim),
        ]
        let placeholderKV = try MLMultiArray(shape: shape, dataType: .float16)
        let total = placeholderKV.count
        let dst = placeholderKV.dataPointer.bindMemory(to: UInt16.self, capacity: total)
        for i in 0..<total { dst[i] = 0 }
        return VoiceHandle(
            kind: .voiceOnly(
                flowKVRank5: MLMultiArrayBox(placeholderKV),
                voiceOffset: 0,
                bosEmb: nil
            ),
            sourceURL: audioURL
        )
    }

    /// Latents from the last clone(...) call. Exposed for the CLI's
    /// intermediate-bundle save path.
    public private(set) var lastLatents: [Float]? = nil

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

        let needResample = srcFormat.sampleRate != targetFormat.sampleRate
            || srcFormat.channelCount != 1
        if !needResample {
            guard let buf = AVAudioPCMBuffer(pcmFormat: srcFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
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
            throw NSError(domain: "VoiceCloner", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Cannot create AVAudioConverter"
            ])
        }
        guard let src = AVAudioPCMBuffer(pcmFormat: srcFormat,
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
