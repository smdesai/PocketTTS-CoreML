//
//  VoiceLoader.swift
//  PocketTTSCoreML
//
//  Created by Sachin Desai on 5/3/26.
//

import CoreML
import Foundation

// Handle to a loaded voice (KV cache snapshot).
//
// Two handle flavors are supported (auto-detected at load time by
// `VoiceLoader.load(url:)`):
//
// - `.prefilled(kv, offset, bosEmb, promptUTF8, noiseSeq)` — the KV cache
//   already contains the text prefill for a specific prompt. Produced by
//   `pockettts_coreml.e2e.export_full_prefill`. The orchestrator skips its
//   prefill step and goes straight to AR generation. Useful as a
//   voice+prompt cache for apps that ship fixed canned phrases.
//
// - `.voiceOnly(flowKVRank5, voiceOffset, bosEmb)` — the voice KV prefix
//   pre-packed into the CoreML rank-5 layout, ready to be fed to
//   `flow_lm_prefill.mlpackage` for in-Swift text prefill. This is the
//   usual path: ship the raw per-layer voice file (e.g. alba.safetensors
//   from the kyutai HF repo) and let the runtime handle the rest.
//   `bosEmb` is optional here — if nil, the orchestrator falls back to
//   the `flow_lm_bos_emb.safetensors` sidecar it loaded at init.
public struct VoiceHandle: Sendable {
    public enum Kind: Sendable {
        // Full voice + text prefill (ready for AR loop).
        case prefilled(
            flowKVRank5: MLMultiArrayBox,
            flowOffset: Int,
            bosEmb: [Float],
            promptUTF8: Data?,
            // Per-AR-step precomputed noise. Shape [MAX_STEPS, 32]. If nil,
            // the orchestrator falls back to its own Gaussian sampler (which
            // WILL diverge from the Python oracle's RNG).
            noiseSeq: [[Float]]?
        )
        // Voice-only prefill — orchestrator runs text prefill in Swift.
        // `flowKVRank5` is fp16 [2*L, 1, S_cap, H, D] with the voice KV at
        // slots [0, voiceOffset) and zeros elsewhere. `bosEmb` is optional
        // (the orchestrator supplies a default from the sidecar if nil).
        case voiceOnly(
            flowKVRank5: MLMultiArrayBox,
            voiceOffset: Int,
            bosEmb: [Float]?
        )
    }

    public let kind: Kind
    public let sourceURL: URL?

    public init(kind: Kind, sourceURL: URL? = nil) {
        self.kind = kind
        self.sourceURL = sourceURL
    }
}

// Thin reference-semantic wrapper around MLMultiArray so it can live in a
// `Sendable` enum case.
public final class MLMultiArrayBox: @unchecked Sendable {
    public let array: MLMultiArray
    public init(_ array: MLMultiArray) { self.array = array }
}

// Speaker projection weights + optional bos_before_voice vector used by
// voice-cloning stage 2. Loaded once at `PocketTTS.init` from the sidecar
// `<artifacts>/speaker_proj.safetensors` and handed to `VoiceCloner`.
public struct SpeakerProjection: Sendable {
    // Flattened fp32 `[d_model, ldim]` row-major weight matrix (typically
    // `[1024, 32]`).
    public let weight: [Float]
    // Optional `[1, 1, d_model]` bos_before_voice prefix. Some language
    // configs set `insert_bos_before_voice: true` and expect this vector
    // to be prepended to the projected voice conditioning before the
    // flow_lm prefill pass.
    public let bosBeforeVoice: [Float]?
    public init(weight: [Float], bosBeforeVoice: [Float]?) {
        self.weight = weight
        self.bosBeforeVoice = bosBeforeVoice
    }
}

public enum VoiceLoader {
    // Load `speaker_proj.safetensors` from the artifacts bundle. Returns
    // `nil` if the file isn't present (voice cloning unavailable for that
    // language bundle).
    public static func loadSpeakerProjection(from url: URL) throws -> SpeakerProjection? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let reader = try SafetensorsReader(url: url)
        guard let info = reader.tensors["speaker_proj"] else {
            throw SafetensorsError.missingKey("speaker_proj")
        }
        // Expected shape: [d_model, ldim] = [1024, 32].
        let expected = [PocketTTSArch.dModel, PocketTTSArch.latentDim]
        guard info.shape == expected else {
            throw SafetensorsError.shapeMismatch(expected: expected, actual: info.shape)
        }
        let (weight, _) = try reader.float32Array(for: "speaker_proj")

        var bos: [Float]? = nil
        if let bosInfo = reader.tensors["bos_before_voice"] {
            // Expected shape [1, 1, d_model]; flatten to d_model.
            guard bosInfo.elementCount == PocketTTSArch.dModel else {
                throw SafetensorsError.shapeMismatch(
                    expected: [1, 1, PocketTTSArch.dModel],
                    actual: bosInfo.shape
                )
            }
            let (b, _) = try reader.float32Array(for: "bos_before_voice")
            bos = b
        }
        return SpeakerProjection(weight: weight, bosBeforeVoice: bos)
    }

    // Load a voice from `.safetensors`. Auto-detects prefilled vs voice-only
    // by the presence of `flow_kv_rank5` (prefilled) vs the
    // `transformer.layers.<i>.self_attn/cache` layout (voice-only).
    public static func load(url: URL) throws -> VoiceHandle {
        let reader = try SafetensorsReader(url: url)
        if reader.tensors["flow_kv_rank5"] != nil {
            return try loadPrefilled(reader: reader, sourceURL: url)
        }
        return try loadVoiceOnly(reader: reader, sourceURL: url)
    }

    static func loadPrefilled(reader: SafetensorsReader, sourceURL: URL) throws -> VoiceHandle {
        guard let kvInfo = reader.tensors["flow_kv_rank5"] else {
            throw SafetensorsError.missingKey("flow_kv_rank5")
        }
        let kvShape = kvInfo.shape
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
        let n = arr.count
        let dstPtr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: n)
        try copyTensorToFp16(reader: reader, name: "flow_kv_rank5", dstPtr: dstPtr, count: n)

        let (offArr, offShape) = try reader.int64Array(for: "flow_offset")
        guard offShape == [1], offArr.count == 1 else {
            throw SafetensorsError.shapeMismatch(expected: [1], actual: offShape)
        }
        let offset = Int(offArr[0])
        guard offset >= 0 && offset <= PocketTTSArch.flowSCap else {
            throw NSError(
                domain: "VoiceLoader", code: 5,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "flow_offset \(offset) is outside 0...\(PocketTTSArch.flowSCap)"
                ])
        }

        let (bosEmb, bosShape) = try reader.float32Array(for: "bos_emb")
        guard bosShape == [PocketTTSArch.latentDim] else {
            throw SafetensorsError.shapeMismatch(
                expected: [PocketTTSArch.latentDim], actual: bosShape)
        }

        var promptUTF8: Data? = nil
        if reader.tensors["prompt_utf8"] != nil {
            promptUTF8 = try reader.bytes(for: "prompt_utf8")
        }

        var noiseSeq: [[Float]]? = nil
        if reader.tensors["noise_seq"] != nil {
            let (flat, shape) = try reader.float32Array(for: "noise_seq")
            guard shape.count == 2, shape[1] == PocketTTSArch.latentDim else {
                throw SafetensorsError.shapeMismatch(
                    expected: [-1, PocketTTSArch.latentDim], actual: shape)
            }
            let steps = shape[0]
            let width = shape[1]
            var seq: [[Float]] = []
            seq.reserveCapacity(steps)
            for i in 0 ..< steps {
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
        // Walk transformer.layers.<i>.self_attn/cache and /offset. Pack
        // into the rank-5 KV layout expected by flow_lm_prefill:
        //   [2*L, 1, S_cap, H, D]  (voice slots [0, voice_off) only)
        let shape: [NSNumber] = [
            NSNumber(value: 2 * PocketTTSArch.flowLayers),
            1,
            NSNumber(value: PocketTTSArch.flowSCap),
            NSNumber(value: PocketTTSArch.flowHeads),
            NSNumber(value: PocketTTSArch.flowHeadDim),
        ]
        let arr = try MLMultiArray(shape: shape, dataType: .float16)
        // Zero-fill first.
        let total = arr.count
        let dstPtr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: total)
        for i in 0 ..< total { dstPtr[i] = 0 }

        let rowStride = PocketTTSArch.flowSCap * PocketTTSArch.flowHeads * PocketTTSArch.flowHeadDim
        let perSlot = PocketTTSArch.flowHeads * PocketTTSArch.flowHeadDim

        var voiceOffset: Int? = nil
        for layer in 0 ..< PocketTTSArch.flowLayers {
            let cacheKey = "transformer.layers.\(layer).self_attn/cache"
            let offKey = "transformer.layers.\(layer).self_attn/offset"
            guard reader.tensors[cacheKey] != nil, reader.tensors[offKey] != nil else {
                throw SafetensorsError.missingKey(cacheKey)
            }
            guard let cacheInfo = reader.tensors[cacheKey] else {
                throw SafetensorsError.missingKey(cacheKey)
            }
            let cshape = cacheInfo.shape
            // Shape: [2, 1, T_voice, H, D]
            guard cshape.count == 5, cshape[0] == 2, cshape[1] == 1,
                cshape[3] == PocketTTSArch.flowHeads, cshape[4] == PocketTTSArch.flowHeadDim
            else {
                throw SafetensorsError.shapeMismatch(
                    expected: [2, 1, -1, PocketTTSArch.flowHeads, PocketTTSArch.flowHeadDim],
                    actual: cshape
                )
            }
            let tVoice = cshape[2]
            let (offArr, offShape) = try reader.int64Array(for: offKey)
            guard offShape == [1], offArr.count == 1 else {
                throw SafetensorsError.shapeMismatch(expected: [1], actual: offShape)
            }
            let offset = Int(offArr[0])
            guard offset >= 0 && offset <= tVoice && offset <= PocketTTSArch.flowSCap else {
                throw NSError(
                    domain: "VoiceLoader", code: 6,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "voice KV offset \(offset) is outside available slots 0...\(min(tVoice, PocketTTSArch.flowSCap))"
                    ])
            }
            if let prior = voiceOffset, prior != offset {
                throw NSError(
                    domain: "VoiceLoader", code: 4,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "voice KV has inconsistent per-layer offsets: \(prior) vs \(offset)"
                    ])
            }
            voiceOffset = offset
            // Copy first `offset` slots into rows 2*layer (K) and 2*layer+1 (V).
            // Source layout per K or V: [1, T_voice, H, D] → per slot H*D floats.
            let copyLen = offset
            if copyLen == 0 { continue }
            // Convert per-slot fp32 → fp16 on the fly.
            let dstKBase = (2 * layer) * rowStride
            let dstVBase = (2 * layer + 1) * rowStride
            try copyVoiceCacheTensorToKV(
                reader: reader,
                name: cacheKey,
                tVoice: tVoice,
                copyLen: copyLen,
                perSlot: perSlot,
                dstPtr: dstPtr,
                dstKBase: dstKBase,
                dstVBase: dstVBase
            )
        }
        guard let voff = voiceOffset else {
            throw SafetensorsError.missingKey("transformer.layers.0.self_attn/offset")
        }
        return VoiceHandle(
            kind: .voiceOnly(
                flowKVRank5: MLMultiArrayBox(arr),
                voiceOffset: voff,
                bosEmb: nil
            ),
            sourceURL: sourceURL
        )
    }

    // Save a voice bundle to `.safetensors`. Supports both flavors:
    //
    // - `.prefilled`: writes `flow_kv_rank5` + `flow_offset` + `bos_emb`
    //   (and optional `prompt_utf8`) — same format as
    //   `export_full_prefill.py`.
    // - `.voiceOnly`: writes per-layer `transformer.layers.<i>.self_attn/cache`
    //   (shape `[2, 1, voiceOffset, H, D]`) + matching `/offset` —
    //   same format as the Kyutai HF voice files (e.g. `alba.safetensors`),
    //   so a cloned voice can be round-tripped through `loadVoice(url:)`
    //   and re-used across sessions without rerunning the mimi encoder.
    public static func save(_ handle: VoiceHandle, to url: URL) throws {
        switch handle.kind {
        case .voiceOnly(let kvBox, let voiceOffset, _):
            try saveVoiceOnly(kv: kvBox.array, voiceOffset: voiceOffset, to: url)
            return
        case .prefilled:
            break  // fall through to legacy path
        }
        guard case .prefilled(let kvBox, let offset, let bosEmb, let promptUTF8, _) = handle.kind
        else {
            throw NSError(
                domain: "VoiceLoader", code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey: "Unreachable"
                ])
        }
        let kvArr = kvBox.array
        let n = kvArr.count
        var floats = [Float](repeating: 0, count: n)
        switch kvArr.dataType {
        case .float32:
            _ = kvArr.withUnsafeBufferPointer(ofType: Float.self) { src in
                floats.withUnsafeMutableBufferPointer { dst in
                    memcpy(dst.baseAddress!, src.baseAddress!, n * MemoryLayout<Float>.stride)
                }
            }
        case .float16:
            let srcPtr = kvArr.dataPointer.bindMemory(to: UInt16.self, capacity: n)
            floats.withUnsafeMutableBufferPointer { dst in
                Float16Ops.convertFp16ToFp32(srcPtr: srcPtr, dstPtr: dst.baseAddress!, count: n)
            }
        default:
            throw NSError(
                domain: "VoiceLoader", code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey: "Unsupported kv dtype for save"
                ])
        }
        let kvData = floats.withUnsafeBufferPointer { Data(buffer: $0) }

        var offsetI64: Int64 = Int64(offset)
        let offData = withUnsafeBytes(of: &offsetI64) { Data($0) }

        let bosData = bosEmb.withUnsafeBufferPointer { Data(buffer: $0) }

        var tensors: [SafetensorsWriter.Tensor] = [
            .init(
                name: "flow_kv_rank5",
                shape: [
                    2 * PocketTTSArch.flowLayers, 1, PocketTTSArch.flowSCap,
                    PocketTTSArch.flowHeads, PocketTTSArch.flowHeadDim,
                ],
                dtype: .F32, data: kvData),
            .init(name: "flow_offset", shape: [1], dtype: .I64, data: offData),
            .init(name: "bos_emb", shape: [PocketTTSArch.latentDim], dtype: .F32, data: bosData),
        ]
        if let p = promptUTF8 {
            tensors.append(.init(name: "prompt_utf8", shape: [p.count], dtype: .U8, data: p))
        }
        try SafetensorsWriter.write(tensors, to: url)
    }

    // Write a `.voiceOnly` handle's KV cache in the per-layer Kyutai HF
    // voice-file layout. Slices first `voiceOffset` slots of each K/V row
    // out of the rank-5 buffer and stacks them into `[2, 1, voiceOffset,
    // H, D]` fp32 tensors named `transformer.layers.<i>.self_attn/cache`
    // (plus a matching `/offset` I64 scalar per layer). The resulting
    // file round-trips through `VoiceLoader.loadVoiceOnly(reader:)`.
    static func saveVoiceOnly(kv: MLMultiArray, voiceOffset: Int, to url: URL) throws {
        precondition(kv.dataType == .float16, "expected fp16 rank-5 KV buffer")
        precondition(
            kv.count == 2 * PocketTTSArch.flowLayers * PocketTTSArch.flowSCap
                * PocketTTSArch.flowHeads * PocketTTSArch.flowHeadDim)
        guard voiceOffset >= 0 && voiceOffset <= PocketTTSArch.flowSCap else {
            throw NSError(
                domain: "VoiceLoader", code: 7,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "voiceOffset \(voiceOffset) is outside 0...\(PocketTTSArch.flowSCap)"
                ])
        }
        let H = PocketTTSArch.flowHeads
        let D = PocketTTSArch.flowHeadDim
        let L = PocketTTSArch.flowLayers
        let sCap = PocketTTSArch.flowSCap
        let perSlot = H * D
        let rowStride = sCap * perSlot
        let srcPtr = kv.dataPointer.bindMemory(to: UInt16.self, capacity: kv.count)

        var tensors: [SafetensorsWriter.Tensor] = []
        tensors.reserveCapacity(2 * L)
        // Scratch buffers reused per layer.
        let perLayerF32Count = 2 * voiceOffset * perSlot  // [2, 1, voiceOffset, H, D]
        var f32 = [Float](repeating: 0, count: perLayerF32Count)
        var offsetI64: Int64 = Int64(voiceOffset)
        let offData = withUnsafeBytes(of: &offsetI64) { Data($0) }

        for layer in 0 ..< L {
            let kBase = (2 * layer) * rowStride
            let vBase = (2 * layer + 1) * rowStride
            // Copy first voiceOffset slots of K then V into fp32 scratch.
            // Layout [2, 1, voiceOffset, H, D]: K occupies [0..voiceOffset*H*D),
            // V occupies [voiceOffset*H*D..2*voiceOffset*H*D).
            if voiceOffset > 0 {
                f32.withUnsafeMutableBufferPointer { dst in
                    // K rows.
                    Float16Ops.convertFp16ToFp32(
                        srcPtr: srcPtr.advanced(by: kBase),
                        dstPtr: dst.baseAddress!,
                        count: voiceOffset * perSlot
                    )
                    // V rows.
                    Float16Ops.convertFp16ToFp32(
                        srcPtr: srcPtr.advanced(by: vBase),
                        dstPtr: dst.baseAddress!.advanced(by: voiceOffset * perSlot),
                        count: voiceOffset * perSlot
                    )
                }
            }
            let cacheData = f32.withUnsafeBufferPointer { Data(buffer: $0) }
            tensors.append(
                .init(
                    name: "transformer.layers.\(layer).self_attn/cache",
                    shape: [2, 1, voiceOffset, H, D],
                    dtype: .F32, data: cacheData
                ))
            tensors.append(
                .init(
                    name: "transformer.layers.\(layer).self_attn/offset",
                    shape: [1], dtype: .I64, data: offData
                ))
        }
        try SafetensorsWriter.write(tensors, to: url)
    }

    private static func copyTensorToFp16(
        reader: SafetensorsReader,
        name: String,
        dstPtr: UnsafeMutablePointer<UInt16>,
        count: Int
    ) throws {
        try reader.withUnsafeTensorBytes(for: name) { raw, info in
            guard info.elementCount == count else {
                throw SafetensorsError.shapeMismatch(expected: [count], actual: info.shape)
            }
            switch info.dtype {
            case .F16:
                memcpy(dstPtr, raw.baseAddress!, count * MemoryLayout<UInt16>.stride)
            case .F32:
                let src = raw.bindMemory(to: Float.self)
                Float16Ops.convertFp32ToFp16(
                    srcPtr: src.baseAddress!, dstPtr: dstPtr, count: count)
            case .BF16:
                var scratch = [Float](repeating: 0, count: count)
                let src = raw.bindMemory(to: UInt16.self)
                for i in 0 ..< count {
                    let bits = UInt32(src[i]) << 16
                    scratch[i] = Float(bitPattern: bits)
                }
                scratch.withUnsafeBufferPointer { sp in
                    Float16Ops.convertFp32ToFp16(
                        srcPtr: sp.baseAddress!, dstPtr: dstPtr, count: count)
                }
            default:
                throw SafetensorsError.unsupportedDType(info.dtype.rawValue)
            }
        }
    }

    private static func copyVoiceCacheTensorToKV(
        reader: SafetensorsReader,
        name: String,
        tVoice: Int,
        copyLen: Int,
        perSlot: Int,
        dstPtr: UnsafeMutablePointer<UInt16>,
        dstKBase: Int,
        dstVBase: Int
    ) throws {
        try reader.withUnsafeTensorBytes(for: name) { raw, info in
            let kBase = 0
            let vBase = tVoice * perSlot
            switch info.dtype {
            case .F32:
                let src = raw.bindMemory(to: Float.self)
                var scratch = [Float](repeating: 0, count: perSlot)
                for s in 0 ..< copyLen {
                    copySanitizedFp32(
                        src.baseAddress!.advanced(by: kBase + s * perSlot),
                        into: &scratch,
                        count: perSlot
                    )
                    scratch.withUnsafeBufferPointer { sp in
                        Float16Ops.convertFp32ToFp16(
                            srcPtr: sp.baseAddress!,
                            dstPtr: dstPtr.advanced(by: dstKBase + s * perSlot),
                            count: perSlot
                        )
                    }
                    copySanitizedFp32(
                        src.baseAddress!.advanced(by: vBase + s * perSlot),
                        into: &scratch,
                        count: perSlot
                    )
                    scratch.withUnsafeBufferPointer { sp in
                        Float16Ops.convertFp32ToFp16(
                            srcPtr: sp.baseAddress!,
                            dstPtr: dstPtr.advanced(by: dstVBase + s * perSlot),
                            count: perSlot
                        )
                    }
                }
            case .F16:
                let src = raw.bindMemory(to: UInt16.self)
                for s in 0 ..< copyLen {
                    copySanitizedFp16(
                        src.baseAddress!.advanced(by: kBase + s * perSlot),
                        dstPtr.advanced(by: dstKBase + s * perSlot),
                        count: perSlot
                    )
                    copySanitizedFp16(
                        src.baseAddress!.advanced(by: vBase + s * perSlot),
                        dstPtr.advanced(by: dstVBase + s * perSlot),
                        count: perSlot
                    )
                }
            case .BF16:
                let src = raw.bindMemory(to: UInt16.self)
                var scratch = [Float](repeating: 0, count: perSlot)
                for s in 0 ..< copyLen {
                    copyBF16AsSanitizedFp32(
                        src.baseAddress!.advanced(by: kBase + s * perSlot),
                        into: &scratch,
                        count: perSlot
                    )
                    scratch.withUnsafeBufferPointer { sp in
                        Float16Ops.convertFp32ToFp16(
                            srcPtr: sp.baseAddress!,
                            dstPtr: dstPtr.advanced(by: dstKBase + s * perSlot),
                            count: perSlot
                        )
                    }
                    copyBF16AsSanitizedFp32(
                        src.baseAddress!.advanced(by: vBase + s * perSlot),
                        into: &scratch,
                        count: perSlot
                    )
                    scratch.withUnsafeBufferPointer { sp in
                        Float16Ops.convertFp32ToFp16(
                            srcPtr: sp.baseAddress!,
                            dstPtr: dstPtr.advanced(by: dstVBase + s * perSlot),
                            count: perSlot
                        )
                    }
                }
            default:
                throw SafetensorsError.unsupportedDType(info.dtype.rawValue)
            }
        }
    }

    private static func copySanitizedFp32(
        _ src: UnsafePointer<Float>, into dst: inout [Float], count: Int
    ) {
        for i in 0 ..< count {
            let value = src[i]
            dst[i] = value.isNaN ? 0 : value
        }
    }

    private static func copySanitizedFp16(
        _ src: UnsafePointer<UInt16>, _ dst: UnsafeMutablePointer<UInt16>, count: Int
    ) {
        for i in 0 ..< count {
            let value = src[i]
            dst[i] = isFp16NaN(value) ? 0 : value
        }
    }

    private static func copyBF16AsSanitizedFp32(
        _ src: UnsafePointer<UInt16>, into dst: inout [Float], count: Int
    ) {
        for i in 0 ..< count {
            let value = Float(bitPattern: UInt32(src[i]) << 16)
            dst[i] = value.isNaN ? 0 : value
        }
    }

    private static func isFp16NaN(_ value: UInt16) -> Bool {
        let exponent = value & 0x7C00
        let mantissa = value & 0x03FF
        return exponent == 0x7C00 && mantissa != 0
    }
}
