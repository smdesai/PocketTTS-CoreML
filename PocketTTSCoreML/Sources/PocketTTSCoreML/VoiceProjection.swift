import Accelerate
import CoreML
import Foundation

/// Speaker-projection helpers for voice-cloning stage 2.
///
/// Stage 1 (in `VoiceCloner`) runs `mimi_encoder.mlpackage` on a 4-second
/// reference waveform and produces latents of shape `fp16[1, 32, T_latent]`
/// (T_latent = 50 for 4 s of audio at Mimi's 12.5 Hz frame rate).
///
/// Stage 2 projects those 32-d latents to `d_model` (1024) via a learned
/// linear, producing `fp16[1, T_latent, 1024]` conditioning that is fed to
/// `flow_lm_prefill.mlpackage` at KV slots `[0, T_latent)`. This file
/// implements the projection; the prefill call lives in `Orchestrator`.
///
/// The projection is a simple `F.linear(latents, speaker_proj)` in the
/// reference Python:
///
/// ```python
/// latents = encoded.transpose(-1, -2).to(torch.float32)   # [1, 50, 32]
/// conditioning = F.linear(latents, flow_lm.speaker_proj_weight)
/// # -> [1, 50, 1024]
/// ```
///
/// where `speaker_proj_weight` has shape `[d_model, ldim] = [1024, 32]`.
/// `F.linear(x, W)` computes `x @ W.T`, so the effective matmul is
/// `(T_latent, 32) @ (32, 1024) = (T_latent, 1024)`.
public enum VoiceProjection {
    /// Apply the speaker projection to Mimi encoder latents.
    ///
    /// - Parameters:
    ///   - latents: `fp16[1, 32, T_latent]` MLMultiArray from `mimi_encoder`.
    ///   - proj: Flattened `[d_model * ldim]` fp32 weights, row-major
    ///     `[d_model=1024, ldim=32]`. Loaded from `speaker_proj.safetensors`.
    ///   - dModel: typically 1024 (may change for future 24L variants).
    /// - Returns: `fp16[1, T_latent, d_model]` conditioning tensor ready to
    ///   feed to `flow_lm_prefill.mlpackage` as `text_embeddings` (padded).
    public static func applySpeakerProjection(
        latents: MLMultiArray,
        proj: [Float],
        dModel: Int = PocketTTSArch.dModel
    ) throws -> (conditioning: MLMultiArray, tLatent: Int) {
        // Expect latents fp16[1, 32, T_latent]. We transpose to [T_latent, 32]
        // fp32 for the matmul, then cast the product back down to fp16.
        guard latents.shape.count == 3,
            latents.shape[0].intValue == 1,
            latents.shape[1].intValue == PocketTTSArch.latentDim
        else {
            throw NSError(
                domain: "VoiceProjection", code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "latents shape must be [1, 32, T_latent]; got \(latents.shape)"
                ])
        }
        let tLatent = latents.shape[2].intValue
        let ldim = PocketTTSArch.latentDim  // 32
        guard proj.count == dModel * ldim else {
            throw NSError(
                domain: "VoiceProjection", code: 2,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "speaker_proj size \(proj.count) != d_model(\(dModel)) * ldim(\(ldim))"
                ])
        }
        // 1. Unpack fp16 latents [1, 32, T_latent] → fp32 transposed [T_latent, 32].
        //    IMPORTANT: MLMultiArray backing storage is not guaranteed to be
        //    C-contiguous. CoreML frequently pads the channel dim for ANE
        //    alignment (observed strides `[2048, 64, 1]` on a [1, 32, 50]
        //    fp16 tensor — the channel stride of 64 includes 14 trailing
        //    padding words past the 50 real values). We must gather
        //    respecting `latents.strides`, not assume a tight layout.
        var latentsT = [Float](repeating: 0, count: tLatent * ldim)
        let strides: [Int] = latents.strides.map { $0.intValue }
        precondition(strides.count == 3, "expected rank-3 latents")
        let capacityBytes = (strides[0] * 1)  // large enough upper bound; we bind conservatively
        _ = capacityBytes
        // Bind the raw buffer. dataPointer addresses the full strided region;
        // CoreML guarantees the range [0, total_stride_bytes) is valid.
        let totalU16 = strides[0]  // outer stride covers the full allocation
        let srcPtr = latents.dataPointer.bindMemory(to: UInt16.self, capacity: totalU16)
        for d in 0 ..< ldim {
            for t in 0 ..< tLatent {
                // Logical index [0, d, t] — respecting strides.
                let srcIdx = d * strides[1] + t * strides[2]
                var one: Float = 0
                let half: UInt16 = srcPtr[srcIdx]
                withUnsafePointer(to: half) { inP in
                    Float16Ops.convertFp16ToFp32(srcPtr: inP, dstPtr: &one, count: 1)
                }
                latentsT[t * ldim + d] = one
            }
        }
        // 2. Matmul latentsT[T, 32] @ proj.T[32, 1024] = out[T, 1024].
        //    proj is stored row-major as [d_model, ldim] = [1024, 32], so
        //    cblas_sgemm with transB=Trans gives us the F.linear semantics:
        //
        //      out[t, d] = sum_k latentsT[t, k] * proj[d, k]
        //                = sum_k A[t, k] * B_T[k, d]
        //
        //    where B = proj has shape [d_model, ldim], B_T (interpreted as
        //    [ldim, d_model] after transpose) is what we multiply by.
        var outFp32 = [Float](repeating: 0, count: tLatent * dModel)
        proj.withUnsafeBufferPointer { pPtr in
            latentsT.withUnsafeBufferPointer { aPtr in
                outFp32.withUnsafeMutableBufferPointer { cPtr in

                    cblas_sgemm(
                        CblasRowMajor,
                        CblasNoTrans,  // A: [T, 32] not transposed
                        CblasTrans,  // B: [1024, 32] -> transpose to [32, 1024]
                        Int32(tLatent),  // M
                        Int32(dModel),  // N
                        Int32(ldim),  // K
                        1.0,  // alpha
                        aPtr.baseAddress!,  // A
                        Int32(ldim),  // lda (row stride of A)
                        pPtr.baseAddress!,  // B (row-major [1024, 32])
                        Int32(ldim),  // ldb (row stride of B as stored)
                        0.0,  // beta
                        cPtr.baseAddress!,  // C
                        Int32(dModel)  // ldc
                    )
                }
            }
        }
        // 3. Pack back into fp16[1, T_latent, d_model] MLMultiArray.
        let out = try MLMultiArray(
            shape: [1, NSNumber(value: tLatent), NSNumber(value: dModel)],
            dataType: .float16
        )
        let nOut = tLatent * dModel
        let dstPtr = out.dataPointer.bindMemory(to: UInt16.self, capacity: nOut)
        outFp32.withUnsafeBufferPointer { sPtr in
            Float16Ops.convertFp32ToFp16(srcPtr: sPtr.baseAddress!, dstPtr: dstPtr, count: nOut)
        }
        return (out, tLatent)
    }

    /// Prepend the per-language `bos_before_voice` vector to a voice
    /// conditioning tensor. Some configs (english/spanish/german/italian/
    /// portuguese/french_24l, per their YAML) set
    /// `insert_bos_before_voice: true`; the reference does
    ///
    /// ```python
    /// prompt = torch.cat([flow_lm.bos_before_voice, prompt], dim=1)
    /// ```
    ///
    /// which grows the sequence by one frame before the flow_lm prefill.
    /// If `bosBeforeVoice` is nil we just pass conditioning through
    /// unmodified.
    public static func prependBosBeforeVoice(
        conditioning: MLMultiArray,
        bosBeforeVoice: [Float]?,
        dModel: Int = PocketTTSArch.dModel
    ) throws -> (conditioning: MLMultiArray, tOut: Int) {
        guard let bos = bosBeforeVoice else {
            let t = conditioning.shape[1].intValue
            return (conditioning, t)
        }
        guard bos.count == dModel else {
            throw NSError(
                domain: "VoiceProjection", code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "bos_before_voice has \(bos.count) elements, expected \(dModel)"
                ])
        }
        guard conditioning.shape.count == 3,
            conditioning.shape[0].intValue == 1,
            conditioning.shape[2].intValue == dModel
        else {
            throw NSError(
                domain: "VoiceProjection", code: 4,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "conditioning shape must be [1, T, \(dModel)]; got \(conditioning.shape)"
                ])
        }
        let tIn = conditioning.shape[1].intValue
        let tOut = tIn + 1
        let out = try MLMultiArray(
            shape: [1, NSNumber(value: tOut), NSNumber(value: dModel)],
            dataType: .float16
        )
        let dstPtr = out.dataPointer.bindMemory(to: UInt16.self, capacity: tOut * dModel)
        // Row 0: bos_before_voice (fp32 → fp16).
        bos.withUnsafeBufferPointer { s in
            Float16Ops.convertFp32ToFp16(srcPtr: s.baseAddress!, dstPtr: dstPtr, count: dModel)
        }
        // Rows 1..tOut: copy fp16 conditioning verbatim.
        let srcPtr = conditioning.dataPointer.bindMemory(to: UInt16.self, capacity: tIn * dModel)
        memcpy(dstPtr.advanced(by: dModel), srcPtr, tIn * dModel * MemoryLayout<UInt16>.stride)
        return (out, tOut)
    }
}
