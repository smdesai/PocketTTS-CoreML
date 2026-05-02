import Foundation
import CoreML
import Accelerate

/// Attention mask builders — ports of `pockettts_coreml.patches.transformer_patched`
/// helpers used by the reference Python driver. These are called OUTSIDE the
/// traced CoreML graph and passed in as tensor inputs (`attn_mask`,
/// `offset_mask`, `scatter_mask`).
public enum Masks {
    /// fp16-safe "-infinity" additive mask value.
    public static let attnMaskNeg: Float = -65_504

    // MARK: - fp16 variants (for fp16 flow_lm_main / mimi_encoder)

    public static func oneHotOffsetMaskFp16(offset: Int, sCapacity: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: sCapacity)]
        let arr = try MLMultiArray(shape: shape, dataType: .float16)
        let ptr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: sCapacity)
        for i in 0..<sCapacity { ptr[i] = 0 }
        ptr[offset] = fp32ToFp16Bits(1.0)
        return arr
    }


    public static func additiveAttentionMaskStepFp16(
        offset: Int, sCapacity: Int, context: Int? = nil
    ) throws -> MLMultiArray {
        precondition(offset >= 0 && offset < sCapacity)
        let shape: [NSNumber] = [1, 1, 1, NSNumber(value: sCapacity)]
        let arr = try MLMultiArray(shape: shape, dataType: .float16)
        let ptr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: sCapacity)
        let negBits = fp32ToFp16Bits(attnMaskNeg)
        let zeroBits: UInt16 = 0
        for i in 0..<sCapacity { ptr[i] = negBits }
        let lo = context == nil ? 0 : max(0, offset - context! + 1)
        for i in lo...offset { ptr[i] = zeroBits }
        return arr
    }

    public static func scatterPrefillMaskFp16(
        startOffset: Int, prefillLen: Int, sCapacity: Int
    ) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: sCapacity), NSNumber(value: prefillLen)]
        let arr = try MLMultiArray(shape: shape, dataType: .float16)
        let ptr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: sCapacity * prefillLen)
        for i in 0..<(sCapacity * prefillLen) { ptr[i] = 0 }
        let oneBits = fp32ToFp16Bits(1.0)
        for j in 0..<prefillLen {
            let row = startOffset + j
            ptr[row * prefillLen + j] = oneBits
        }
        return arr
    }

    public static func additiveAttentionMaskPrefillFp16(
        startOffset: Int, prefillLen: Int, sCapacity: Int, context: Int? = nil
    ) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 1, NSNumber(value: prefillLen), NSNumber(value: sCapacity)]
        let arr = try MLMultiArray(shape: shape, dataType: .float16)
        let total = prefillLen * sCapacity
        let ptr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: total)
        let negBits = fp32ToFp16Bits(attnMaskNeg)
        let zeroBits: UInt16 = 0
        for i in 0..<total { ptr[i] = negBits }
        for i in 0..<prefillLen {
            let pos = startOffset + i
            let lo = context == nil ? 0 : max(0, pos - context! + 1)
            for k in lo...pos {
                ptr[i * sCapacity + k] = zeroBits
            }
        }
        return arr
    }

    @inline(__always)
    static func fp32ToFp16Bits(_ f: Float) -> UInt16 {
        var input = f
        var output: UInt16 = 0
        var srcBuf = vImage_Buffer(data: &input, height: 1, width: 1, rowBytes: 4)
        var dstBuf = vImage_Buffer(data: &output, height: 1, width: 1, rowBytes: 2)
        vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
        return output
    }

    /// `build_additive_attention_mask_step`
    /// Output shape `[1, 1, 1, s_capacity]`. Visible where pos_k ∈ [lo, offset].
    public static func additiveAttentionMaskStep(
        offset: Int, sCapacity: Int, context: Int? = nil
    ) throws -> MLMultiArray {
        precondition(offset >= 0 && offset < sCapacity)
        let shape: [NSNumber] = [1, 1, 1, NSNumber(value: sCapacity)]
        let arr = try MLMultiArray(shape: shape, dataType: .float32)
        arr.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            for i in 0..<sCapacity { buf[i] = attnMaskNeg }
            let lo = context == nil ? 0 : max(0, offset - context! + 1)
            for i in lo...offset { buf[i] = 0.0 }
        }
        return arr
    }

    /// `build_one_hot_offset_mask`
    /// Output shape `[1, s_capacity]`. 1.0 at position `offset`, 0 elsewhere.
    public static func oneHotOffsetMask(offset: Int, sCapacity: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: sCapacity)]
        let arr = try MLMultiArray(shape: shape, dataType: .float32)
        arr.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            for i in 0..<sCapacity { buf[i] = 0.0 }
            buf[offset] = 1.0
        }
        return arr
    }

    /// `build_scatter_prefill_mask`
    /// Output shape `[1, s_capacity, prefill_len]`.
    public static func scatterPrefillMask(
        startOffset: Int, prefillLen: Int, sCapacity: Int
    ) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: sCapacity), NSNumber(value: prefillLen)]
        let arr = try MLMultiArray(shape: shape, dataType: .float32)
        let n = sCapacity * prefillLen
        arr.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            for i in 0..<n { buf[i] = 0.0 }
            for j in 0..<prefillLen {
                let row = startOffset + j
                buf[row * prefillLen + j] = 1.0
            }
        }
        return arr
    }

    /// `build_additive_attention_mask_prefill`
    /// Output shape `[1, 1, prefill_len, s_capacity]`.
    public static func additiveAttentionMaskPrefill(
        startOffset: Int, prefillLen: Int, sCapacity: Int, context: Int? = nil
    ) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 1, NSNumber(value: prefillLen), NSNumber(value: sCapacity)]
        let arr = try MLMultiArray(shape: shape, dataType: .float32)
        let totalQ = prefillLen * sCapacity
        arr.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            for i in 0..<totalQ { buf[i] = attnMaskNeg }
            for i in 0..<prefillLen {
                let pos = startOffset + i
                let lo = context == nil ? 0 : max(0, pos - context! + 1)
                for k in lo...pos {
                    buf[i * sCapacity + k] = 0.0
                }
            }
        }
        return arr
    }
}
