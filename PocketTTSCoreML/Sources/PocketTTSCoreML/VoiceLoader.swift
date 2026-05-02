import Foundation
import CoreML

/// Handle to a loaded voice (KV cache snapshot).
///
/// Phase 4A supports two handle flavors:
///
/// - `.prefilled(kv, offset, bosEmb)` — the KV cache already contains the
///   text prefill for a specific prompt. Produced by
///   `pockettts_coreml.e2e.export_full_prefill`. The orchestrator skips its
///   prefill step and goes straight to AR generation.
/// - `.voiceOnly(cache, offset, bosEmb)` — voice-only KV prefix at the
///   reference layer layout `[2, 1, T_voice, H, D]`. Phase 4A cannot perform
///   text prefill in Swift (see README §Limitations); using this handle
///   without an accompanying pre-prefilled bundle will fail at
///   `PocketTTS.generate(...)`.
public struct VoiceHandle: Sendable {
    public enum Kind: Sendable {
        /// Full voice + text prefill (ready for AR loop).
        case prefilled(
            flowKVRank5: MLMultiArrayBox,
            flowOffset: Int,
            bosEmb: [Float],
            promptUTF8: Data?,
            /// Per-AR-step precomputed noise. Shape [MAX_STEPS, 32]. If nil,
            /// the orchestrator falls back to its own Gaussian sampler (which
            /// WILL diverge from the Python oracle's RNG).
            noiseSeq: [[Float]]?
        )
        /// Voice-only prefill (needs a Phase-4B-supplied text prefill step).
        case voiceOnly(
            layers: [LayerCache],
            bosEmb: [Float]?
        )
    }

    public struct LayerCache: Sendable {
        public let cache: [Float]  // shape [2, 1, T_voice, H, D] flat
        public let offset: Int
        public let tVoice: Int
    }

    public let kind: Kind
    public let sourceURL: URL?

    public init(kind: Kind, sourceURL: URL? = nil) {
        self.kind = kind
        self.sourceURL = sourceURL
    }
}

/// Thin reference-semantic wrapper around MLMultiArray so it can live in a
/// `Sendable` enum case.
public final class MLMultiArrayBox: @unchecked Sendable {
    public let array: MLMultiArray
    public init(_ array: MLMultiArray) { self.array = array }
}

public enum VoiceLoader {
    /// Load a voice from `.safetensors`. Auto-detects prefilled vs voice-only
    /// by the presence of `flow_kv_rank5` (prefilled) vs the
    /// `transformer.layers.<i>.self_attn/cache` layout (voice-only).
    public static func load(url: URL) throws -> VoiceHandle {
        let reader = try SafetensorsReader(url: url)
        if reader.tensors["flow_kv_rank5"] != nil {
            return try loadPrefilled(reader: reader, sourceURL: url)
        }
        return try loadVoiceOnly(reader: reader, sourceURL: url)
    }

    static func loadPrefilled(reader: SafetensorsReader, sourceURL: URL) throws -> VoiceHandle {
        let (kvFlat, kvShape) = try reader.float32Array(for: "flow_kv_rank5")
        // Expected shape: [12, 1, 256, 16, 64]
        let expected = [
            2 * PocketTTSArch.flowLayers, 1, PocketTTSArch.flowSCap,
            PocketTTSArch.flowHeads, PocketTTSArch.flowHeadDim,
        ]
        guard kvShape == expected else {
            throw SafetensorsError.shapeMismatch(expected: expected, actual: kvShape)
        }

        // flow_lm_main expects fp16 KV.
        let arr = try MLMultiArray(
            shape: kvShape.map { NSNumber(value: $0) },
            dataType: .float16
        )
        let n = kvFlat.count
        let dstPtr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: n)
        kvFlat.withUnsafeBufferPointer { src in
            Float16Ops.convertFp32ToFp16(srcPtr: src.baseAddress!, dstPtr: dstPtr, count: n)
        }

        let (offArr, _) = try reader.int64Array(for: "flow_offset")
        let offset = Int(offArr[0])

        let (bosEmb, _) = try reader.float32Array(for: "bos_emb")

        var promptUTF8: Data? = nil
        if let _ = reader.tensors["prompt_utf8"] {
            promptUTF8 = try reader.bytes(for: "prompt_utf8")
        }

        var noiseSeq: [[Float]]? = nil
        if reader.tensors["noise_seq"] != nil {
            let (flat, shape) = try reader.float32Array(for: "noise_seq")
            precondition(shape.count == 2)
            let steps = shape[0]
            let width = shape[1]
            var seq: [[Float]] = []
            seq.reserveCapacity(steps)
            for i in 0..<steps {
                seq.append(Array(flat[(i * width) ..< ((i + 1) * width)]))
            }
            noiseSeq = seq
        }

        return VoiceHandle(
            kind: .prefilled(
                flowKVRank5: MLMultiArrayBox(arr),
                flowOffset: offset,
                bosEmb: bosEmb,
                promptUTF8: promptUTF8,
                noiseSeq: noiseSeq
            ),
            sourceURL: sourceURL
        )
    }

    static func loadVoiceOnly(reader: SafetensorsReader, sourceURL: URL) throws -> VoiceHandle {
        // Walk transformer.layers.<i>.self_attn/cache and /offset.
        var layers: [VoiceHandle.LayerCache] = []
        for layer in 0..<PocketTTSArch.flowLayers {
            let cacheKey = "transformer.layers.\(layer).self_attn/cache"
            let offKey = "transformer.layers.\(layer).self_attn/offset"
            guard reader.tensors[cacheKey] != nil, reader.tensors[offKey] != nil else {
                throw SafetensorsError.missingKey(cacheKey)
            }
            let (cache, shape) = try reader.float32Array(for: cacheKey)
            // Shape: [2, 1, T_voice, H, D]
            guard shape.count == 5, shape[0] == 2, shape[1] == 1,
                  shape[3] == PocketTTSArch.flowHeads, shape[4] == PocketTTSArch.flowHeadDim
            else {
                throw SafetensorsError.shapeMismatch(
                    expected: [2, 1, -1, PocketTTSArch.flowHeads, PocketTTSArch.flowHeadDim],
                    actual: shape
                )
            }
            let tVoice = shape[2]
            let (offArr, _) = try reader.int64Array(for: offKey)
            let offset = Int(offArr[0])
            // NaN-sanitize: reference cache contains NaNs in unwritten slots.
            var sanitized = cache
            for i in 0..<sanitized.count where sanitized[i].isNaN { sanitized[i] = 0 }
            layers.append(VoiceHandle.LayerCache(cache: sanitized, offset: offset, tVoice: tVoice))
        }
        return VoiceHandle(
            kind: .voiceOnly(layers: layers, bosEmb: nil),
            sourceURL: sourceURL
        )
    }

    /// Save a prefilled voice bundle in the format produced by
    /// `export_full_prefill.py`.
    public static func save(_ handle: VoiceHandle, to url: URL) throws {
        guard case let .prefilled(kvBox, offset, bosEmb, promptUTF8, _) = handle.kind else {
            throw NSError(domain: "VoiceLoader", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Only prefilled VoiceHandles are saveable in Phase 4A"
            ])
        }
        let kvArr = kvBox.array
        let n = kvArr.count
        var floats = [Float](repeating: 0, count: n)
        switch kvArr.dataType {
        case .float32:
            kvArr.withUnsafeBufferPointer(ofType: Float.self) { src in
                memcpy(&floats, src.baseAddress!, n * MemoryLayout<Float>.stride)
            }
        case .float16:
            let srcPtr = kvArr.dataPointer.bindMemory(to: UInt16.self, capacity: n)
            floats.withUnsafeMutableBufferPointer { dst in
                Float16Ops.convertFp16ToFp32(srcPtr: srcPtr, dstPtr: dst.baseAddress!, count: n)
            }
        default:
            throw NSError(domain: "VoiceLoader", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Unsupported kv dtype for save"
            ])
        }
        let kvData = floats.withUnsafeBufferPointer { Data(buffer: $0) }

        var offsetI64: Int64 = Int64(offset)
        let offData = withUnsafeBytes(of: &offsetI64) { Data($0) }

        let bosData = bosEmb.withUnsafeBufferPointer { Data(buffer: $0) }

        var tensors: [SafetensorsWriter.Tensor] = [
            .init(name: "flow_kv_rank5",
                  shape: [2 * PocketTTSArch.flowLayers, 1, PocketTTSArch.flowSCap,
                          PocketTTSArch.flowHeads, PocketTTSArch.flowHeadDim],
                  dtype: .F32, data: kvData),
            .init(name: "flow_offset", shape: [1], dtype: .I64, data: offData),
            .init(name: "bos_emb", shape: [PocketTTSArch.latentDim], dtype: .F32, data: bosData),
        ]
        if let p = promptUTF8 {
            tensors.append(.init(name: "prompt_utf8", shape: [p.count], dtype: .U8, data: p))
        }
        try SafetensorsWriter.write(tensors, to: url)
    }
}
