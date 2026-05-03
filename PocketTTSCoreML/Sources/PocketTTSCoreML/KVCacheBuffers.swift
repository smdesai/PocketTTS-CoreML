import Foundation
import CoreML

/// Architecture constants for the PocketTTS CoreML bundle.
/// Must match `CoreMLGenerator` in `pockettts_coreml/e2e/generator.py`.
///
/// NOTE: `flowLayers` is mutable because the 24-layer (`french_24l`) variant
/// has 24 layers while the 6-layer variants (english, spanish, german,
/// italian, portuguese) have 6. `PocketTTS.init` calls
/// `PocketTTSArch.configureFlowLayers(from:)` at model-load time to set
/// the right value based on the loaded `flow_lm_main` input shape. The
/// default of 6 is correct for the majority of shipped languages.
public enum PocketTTSArch {
    // FlowLM main transformer.
    public static let flowSCap: Int     = 256
    // nonisolated(unsafe) is appropriate here: the value is set exactly
    // once by PocketTTS.init (before the actor returns) and read-only
    // thereafter. The ordering is enforced by configureFlowLayers being
    // called before any KV allocation / voice load.
    public nonisolated(unsafe) static var flowLayers: Int   = 6
    public static let flowHeads: Int    = 16
    public static let flowHeadDim: Int  = 64
    public static let flowHalfDim: Int  = 32   // half of head_dim (RoPE table width)
    public static let dModel: Int       = 1024
    public static let latentDim: Int    = 32

    // Mimi decoder streaming transformer.
    public static let mimiSCap: Int     = 1024
    public static let mimiLayers: Int   = 2
    public static let mimiHeads: Int    = 8
    public static let mimiHeadDim: Int  = 64
    public static let mimiHalfDim: Int  = 32
    public static let mimiTxContext: Int = 250
    public static let mimiTStep: Int    = 16

    public static let frameSize: Int    = 1920  // samples per AR frame (80 ms @ 24 kHz)
    public static let sampleRate: Int   = 24000

    public static let sTextPad: Int     = 128   // text_conditioner input width

    // Defaults mirroring reference.
    public static let defaultTemperature: Float = 0.7
    public static let defaultEosThreshold: Float = -4.0
    public static let defaultFramesAfterEos: Int = 2

    /// Resolve `flowLayers` from a loaded flow_lm_main (or flow_lm_prefill)
    /// MLModel by inspecting the `kv_cache_in` input's expected shape.
    /// Expected shape is `[2*L, 1, S_cap, H, D]`. Called by `PocketTTS.init`.
    public static func configureFlowLayers(from flowMain: MLModel) {
        let desc = flowMain.modelDescription.inputDescriptionsByName
        guard let kv = desc["kv_cache_in"] ?? desc["kv_cache"],
              let shape = kv.multiArrayConstraint?.shape,
              let dim0 = shape.first?.intValue
        else { return }  // leave default (6) if shape can't be read
        // dim0 is 2*L; L = 6 for en/es/de/it/pt, 24 for french_24l.
        flowLayers = dim0 / 2
    }
}

/// Pre-allocated MLMultiArray pool for flow_lm_main's KV cache.
///
/// The flow_lm_main signature expects `kv_cache_in` at shape
/// `[12, 1, 256, 16, 64]` (that is `[2*L, B, S_cap, H, D]`, layer `i` occupying
/// rows `[2i, 2i+2)`).
public final class FlowKVCache: @unchecked Sendable {
    public static let shape: [NSNumber] = [
        NSNumber(value: 2 * PocketTTSArch.flowLayers),  // 12
        1,
        NSNumber(value: PocketTTSArch.flowSCap),        // 256
        NSNumber(value: PocketTTSArch.flowHeads),       // 16
        NSNumber(value: PocketTTSArch.flowHeadDim),     // 64
    ]

    public let array: MLMultiArray
    public let dtype: MLMultiArrayDataType

    public init(dtype: MLMultiArrayDataType = .float16) throws {
        self.array = try MLMultiArray(shape: Self.shape, dataType: dtype)
        self.dtype = dtype
        fill(with: 0)
    }

    public func fill(with value: Float) {
        let n = array.count
        array.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            for i in 0..<n { buf[i] = value }
        }
    }

    /// Copy float array into KV rows [2*layer, 2*layer+2) up to `sLen` slots.
    /// `kForLayer` and `vForLayer` are shape [1, S_actual, H, D] flat arrays.
    public func writeLayer(_ layer: Int, k: [Float], v: [Float], sLen: Int) {
        // row for K = 2*layer, V = 2*layer+1. Target layout per-row: [1, S_cap, H, D]
        // Source layout (per cache): [1, S_actual, H, D]; we copy first sLen slots.
        let rowStride = PocketTTSArch.flowSCap * PocketTTSArch.flowHeads * PocketTTSArch.flowHeadDim
        let perSlot = PocketTTSArch.flowHeads * PocketTTSArch.flowHeadDim
        let srcStride = perSlot  // each slot is H*D floats
        let offsetK = (2 * layer) * rowStride
        let offsetV = (2 * layer + 1) * rowStride
        array.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            k.withUnsafeBufferPointer { kp in
                for s in 0..<sLen {
                    let src = kp.baseAddress! + s * srcStride
                    let dst = buf.baseAddress! + offsetK + s * perSlot
                    memcpy(dst, src, perSlot * MemoryLayout<Float>.stride)
                }
            }
            v.withUnsafeBufferPointer { vp in
                for s in 0..<sLen {
                    let src = vp.baseAddress! + s * srcStride
                    let dst = buf.baseAddress! + offsetV + s * perSlot
                    memcpy(dst, src, perSlot * MemoryLayout<Float>.stride)
                }
            }
        }
    }

    /// Copy from another MLMultiArray in-place (shapes must match).
    public func copy(from other: MLMultiArray) {
        precondition(other.count == array.count, "shape mismatch")
        let n = array.count
        other.withUnsafeBufferPointer(ofType: Float.self) { src in
            array.withUnsafeMutableBufferPointer(ofType: Float.self) { dst, _ in
                memcpy(dst.baseAddress!, src.baseAddress!, n * MemoryLayout<Float>.stride)
            }
        }
    }
}
