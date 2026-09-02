//
//  Masks.swift
//  PocketTTSCoreML
//
//  Created by Sachin Desai on 5/3/26.
//

import Accelerate
import CoreML
import Foundation

// Attention mask builders — ports of `pockettts_coreml.patches.transformer_patched`
// helpers used by the reference Python driver. These are called OUTSIDE the
// traced CoreML graph and passed in as tensor inputs (`attn_mask`,
// `offset_mask`, `scatter_mask`).
public enum Masks {
    // fp16-safe "-infinity" additive mask value.
    public static let attnMaskNeg: Float = -65_504
    public static let attnMaskNegFp16Bits: UInt16 = 0xFBFF
    public static let oneFp16Bits: UInt16 = 0x3C00

    // MARK: - fp16 variants (for fp16 flow_lm_main / mimi_encoder)

    public static func oneHotOffsetMaskFp16(offset: Int, sCapacity: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: sCapacity)]
        let arr = try MLMultiArray(shape: shape, dataType: .float16)
        try fillOneHotOffsetMaskFp16(arr, offset: offset, sCapacity: sCapacity)
        return arr
    }

    public static func fillOneHotOffsetMaskFp16(
        _ arr: MLMultiArray, offset: Int, sCapacity: Int
    ) throws {
        guard offset >= 0 && offset < sCapacity, arr.count == sCapacity,
            arr.dataType == .float16
        else {
            throw NSError(
                domain: "Masks", code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "invalid fp16 offset mask offset=\(offset), shape=\(arr.shape)"
                ])
        }
        let ptr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: sCapacity)
        for i in 0 ..< sCapacity { ptr[i] = 0 }
        ptr[offset] = oneFp16Bits
    }

    public static func additiveAttentionMaskStepFp16(
        offset: Int, sCapacity: Int, context: Int? = nil
    ) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 1, 1, NSNumber(value: sCapacity)]
        let arr = try MLMultiArray(shape: shape, dataType: .float16)
        try fillAdditiveAttentionMaskStepFp16(
            arr, offset: offset, sCapacity: sCapacity, context: context)
        return arr
    }

    public static func fillAdditiveAttentionMaskStepFp16(
        _ arr: MLMultiArray, offset: Int, sCapacity: Int, context: Int? = nil
    ) throws {
        guard offset >= 0 && offset < sCapacity, arr.count == sCapacity,
            arr.dataType == .float16
        else {
            throw NSError(
                domain: "Masks", code: 2,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "invalid fp16 attention mask offset=\(offset), shape=\(arr.shape)"
                ])
        }
        let ptr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: sCapacity)
        let zeroBits: UInt16 = 0
        for i in 0 ..< sCapacity { ptr[i] = attnMaskNegFp16Bits }
        let lo = context == nil ? 0 : max(0, offset - context! + 1)
        for i in lo ... offset { ptr[i] = zeroBits }
    }

    public static func scatterPrefillMaskFp16(
        startOffset: Int, prefillLen: Int, sCapacity: Int
    ) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: sCapacity), NSNumber(value: prefillLen)]
        let arr = try MLMultiArray(shape: shape, dataType: .float16)
        try fillScatterPrefillMaskFp16(
            arr, startOffset: startOffset, prefillLen: prefillLen, sCapacity: sCapacity)
        return arr
    }

    public static func fillScatterPrefillMaskFp16(
        _ arr: MLMultiArray, startOffset: Int, prefillLen: Int, sCapacity: Int
    ) throws {
        guard prefillLen >= 0, startOffset >= 0, startOffset + prefillLen <= sCapacity,
            arr.count == sCapacity * prefillLen, arr.dataType == .float16
        else {
            throw NSError(
                domain: "Masks", code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "invalid fp16 scatter mask start=\(startOffset), len=\(prefillLen), shape=\(arr.shape)"
                ])
        }
        let ptr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: sCapacity * prefillLen)
        for i in 0 ..< (sCapacity * prefillLen) { ptr[i] = 0 }
        for j in 0 ..< prefillLen {
            let row = startOffset + j
            ptr[row * prefillLen + j] = oneFp16Bits
        }
    }

    public static func additiveAttentionMaskPrefillFp16(
        startOffset: Int, prefillLen: Int, sCapacity: Int, context: Int? = nil
    ) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 1, NSNumber(value: prefillLen), NSNumber(value: sCapacity)]
        let arr = try MLMultiArray(shape: shape, dataType: .float16)
        try fillAdditiveAttentionMaskPrefillFp16(
            arr, startOffset: startOffset, prefillLen: prefillLen,
            sCapacity: sCapacity, context: context)
        return arr
    }

    public static func fillAdditiveAttentionMaskPrefillFp16(
        _ arr: MLMultiArray, startOffset: Int, prefillLen: Int, sCapacity: Int,
        context: Int? = nil
    ) throws {
        guard prefillLen >= 0, startOffset >= 0, startOffset + prefillLen <= sCapacity,
            arr.count == prefillLen * sCapacity, arr.dataType == .float16
        else {
            throw NSError(
                domain: "Masks", code: 4,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "invalid fp16 prefill attention mask start=\(startOffset), len=\(prefillLen), shape=\(arr.shape)"
                ])
        }
        let total = prefillLen * sCapacity
        let ptr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: total)
        let zeroBits: UInt16 = 0
        for i in 0 ..< total { ptr[i] = attnMaskNegFp16Bits }
        for i in 0 ..< prefillLen {
            let pos = startOffset + i
            let lo = context == nil ? 0 : max(0, pos - context! + 1)
            for k in lo ... pos {
                ptr[i * sCapacity + k] = zeroBits
            }
        }
    }

    // MARK: - Padded prefill helpers (flow_lm_prefill.mlpackage)

    // Scatter mask for the padded flow_lm_prefill graph.
    //
    // Shape `[1, sCapacity, sTextPad]`. Column `j` for `j in [0, sText)`
    // is one-hot at row `startOffset + j`; columns `j in [sText, sTextPad)`
    // are zero (padding — their rows in text_embeddings are also zero/
    // ignored, and a zero column leaves the KV cache untouched at every
    // slot). Matches the `_build_example_inputs` construction in
    // `convert_flow_lm_prefill.py`.
    public static func scatterPrefillMaskFp16Padded(
        startOffset: Int, sText: Int, sTextPad: Int, sCapacity: Int
    ) throws -> MLMultiArray {
        precondition(sText <= sTextPad)
        precondition(startOffset + sText <= sCapacity)
        let shape: [NSNumber] = [
            1, NSNumber(value: sCapacity), NSNumber(value: sTextPad),
        ]
        let arr = try MLMultiArray(shape: shape, dataType: .float16)
        let total = sCapacity * sTextPad
        let ptr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: total)
        for i in 0 ..< total { ptr[i] = 0 }
        for j in 0 ..< sText {
            let row = startOffset + j
            ptr[row * sTextPad + j] = oneFp16Bits
        }
        return arr
    }

    // Additive attention mask for the padded flow_lm_prefill graph.
    //
    // Shape `[1, 1, sTextPad, sCapacity]`. Rows `i in [0, sText)` are
    // causal over `[0, startOffset + i]`. Rows `i in [sText, sTextPad)`
    // copy the last real row (i = sText - 1) so softmax stays finite
    // — since their scatter columns are zero, they don't mutate the KV.
    public static func additiveAttentionMaskPrefillFp16Padded(
        startOffset: Int, sText: Int, sTextPad: Int, sCapacity: Int
    ) throws -> MLMultiArray {
        precondition(sText > 0 && sText <= sTextPad)
        precondition(startOffset + sText <= sCapacity)
        let shape: [NSNumber] = [
            1, 1, NSNumber(value: sTextPad), NSNumber(value: sCapacity),
        ]
        let arr = try MLMultiArray(shape: shape, dataType: .float16)
        let total = sTextPad * sCapacity
        let ptr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: total)
        let zeroBits: UInt16 = 0
        // Fill all masked.
        for i in 0 ..< total { ptr[i] = attnMaskNegFp16Bits }
        // Real rows: causal window.
        for i in 0 ..< sText {
            let pos = startOffset + i
            for k in 0 ... pos { ptr[i * sCapacity + k] = zeroBits }
        }
        // Padding rows: copy last real row (i = sText - 1). Use memcpy
        // for speed.
        let lastRowBase = (sText - 1) * sCapacity
        for i in sText ..< sTextPad {
            let dstBase = i * sCapacity
            memcpy(
                ptr.advanced(by: dstBase),
                ptr.advanced(by: lastRowBase),
                sCapacity * MemoryLayout<UInt16>.stride)
        }
        return arr
    }

    @inline(__always)
    static func fp32ToFp16Bits(_ f: Float) -> UInt16 {
        var input = f
        var output: UInt16 = 0
        withUnsafeMutableBytes(of: &input) { srcRaw in
            withUnsafeMutableBytes(of: &output) { dstRaw in
                var srcBuf = vImage_Buffer(
                    data: srcRaw.baseAddress!, height: 1, width: 1, rowBytes: 4)
                var dstBuf = vImage_Buffer(
                    data: dstRaw.baseAddress!, height: 1, width: 1, rowBytes: 2)
                vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
            }
        }
        return output
    }

    // `build_additive_attention_mask_step`
    // Output shape `[1, 1, 1, s_capacity]`. Visible where pos_k ∈ [lo, offset].
    public static func additiveAttentionMaskStep(
        offset: Int, sCapacity: Int, context: Int? = nil
    ) throws -> MLMultiArray {
        precondition(offset >= 0 && offset < sCapacity)
        let shape: [NSNumber] = [1, 1, 1, NSNumber(value: sCapacity)]
        let arr = try MLMultiArray(shape: shape, dataType: .float32)
        try fillAdditiveAttentionMaskStep(
            arr, offset: offset, sCapacity: sCapacity, context: context)
        return arr
    }

    public static func fillAdditiveAttentionMaskStep(
        _ arr: MLMultiArray, offset: Int, sCapacity: Int, context: Int? = nil
    ) throws {
        guard offset >= 0 && offset < sCapacity, arr.count == sCapacity,
            arr.dataType == .float32
        else {
            throw NSError(
                domain: "Masks", code: 5,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "invalid fp32 attention mask offset=\(offset), shape=\(arr.shape)"
                ])
        }
        arr.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            for i in 0 ..< sCapacity { buf[i] = attnMaskNeg }
            let lo = context == nil ? 0 : max(0, offset - context! + 1)
            for i in lo ... offset { buf[i] = 0.0 }
        }
    }

    // `build_one_hot_offset_mask`
    // Output shape `[1, s_capacity]`. 1.0 at position `offset`, 0 elsewhere.
    public static func oneHotOffsetMask(offset: Int, sCapacity: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: sCapacity)]
        let arr = try MLMultiArray(shape: shape, dataType: .float32)
        guard offset >= 0 && offset < sCapacity else {
            throw NSError(domain: "Masks", code: 6)
        }
        arr.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            for i in 0 ..< sCapacity { buf[i] = 0.0 }
            buf[offset] = 1.0
        }
        return arr
    }

    // `build_scatter_prefill_mask`
    // Output shape `[1, s_capacity, prefill_len]`.
    public static func scatterPrefillMask(
        startOffset: Int, prefillLen: Int, sCapacity: Int
    ) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: sCapacity), NSNumber(value: prefillLen)]
        let arr = try MLMultiArray(shape: shape, dataType: .float32)
        try fillScatterPrefillMask(
            arr, startOffset: startOffset, prefillLen: prefillLen, sCapacity: sCapacity)
        return arr
    }

    public static func fillScatterPrefillMask(
        _ arr: MLMultiArray, startOffset: Int, prefillLen: Int, sCapacity: Int
    ) throws {
        guard prefillLen >= 0, startOffset >= 0, startOffset + prefillLen <= sCapacity,
            arr.count == sCapacity * prefillLen, arr.dataType == .float32
        else {
            throw NSError(
                domain: "Masks", code: 7,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "invalid fp32 scatter mask start=\(startOffset), len=\(prefillLen), shape=\(arr.shape)"
                ])
        }
        let n = sCapacity * prefillLen
        arr.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            for i in 0 ..< n { buf[i] = 0.0 }
            for j in 0 ..< prefillLen {
                let row = startOffset + j
                buf[row * prefillLen + j] = 1.0
            }
        }
    }

    // `build_additive_attention_mask_prefill`
    // Output shape `[1, 1, prefill_len, s_capacity]`.
    public static func additiveAttentionMaskPrefill(
        startOffset: Int, prefillLen: Int, sCapacity: Int, context: Int? = nil
    ) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 1, NSNumber(value: prefillLen), NSNumber(value: sCapacity)]
        let arr = try MLMultiArray(shape: shape, dataType: .float32)
        try fillAdditiveAttentionMaskPrefill(
            arr, startOffset: startOffset, prefillLen: prefillLen,
            sCapacity: sCapacity, context: context)
        return arr
    }

    public static func fillAdditiveAttentionMaskPrefill(
        _ arr: MLMultiArray, startOffset: Int, prefillLen: Int, sCapacity: Int,
        context: Int? = nil
    ) throws {
        guard prefillLen >= 0, startOffset >= 0, startOffset + prefillLen <= sCapacity,
            arr.count == prefillLen * sCapacity, arr.dataType == .float32
        else {
            throw NSError(
                domain: "Masks", code: 8,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "invalid fp32 prefill attention mask start=\(startOffset), len=\(prefillLen), shape=\(arr.shape)"
                ])
        }
        let totalQ = prefillLen * sCapacity
        arr.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            for i in 0 ..< totalQ { buf[i] = attnMaskNeg }
            for i in 0 ..< prefillLen {
                let pos = startOffset + i
                let lo = context == nil ? 0 : max(0, pos - context! + 1)
                for k in lo ... pos {
                    buf[i * sCapacity + k] = 0.0
                }
            }
        }
    }
}
