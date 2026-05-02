import Foundation
import CoreML

/// Precomputed cosine/sine tables for RoPE (Rotary Position Embeddings).
///
/// Mirrors `pockettts_coreml.patches.build_rope_tables`:
/// ```python
/// half = head_dim // 2
/// inv_freq = 1.0 / (10000 ** (arange(0, half) / half))   # [half]
/// t = arange(0, max_context, dtype=float32)              # [T]
/// freqs = einsum('i,j->ij', t, inv_freq)                 # [T, half]
/// cos_t = cos(freqs)  # [T, half]
/// sin_t = sin(freqs)  # [T, half]
/// ```
/// Sliced per-AR-step at shape `[1, 1, 1, half]`.
public struct RoPECache: Sendable {
    public let maxContext: Int
    public let halfDim: Int
    /// `[maxContext * halfDim]` row-major; row `t` is cos[t, :halfDim].
    public let cos: [Float]
    public let sin: [Float]

    public init(maxContext: Int, headDim: Int, base: Float = 10_000) {
        let half = headDim / 2
        var invFreq = [Float](repeating: 0, count: half)
        for i in 0..<half {
            invFreq[i] = 1.0 / powf(base, Float(i) / Float(half))
        }
        var c = [Float](repeating: 0, count: maxContext * half)
        var s = [Float](repeating: 0, count: maxContext * half)
        for t in 0..<maxContext {
            let tf = Float(t)
            for j in 0..<half {
                let x = tf * invFreq[j]
                c[t * half + j] = Foundation.cos(x)
                s[t * half + j] = Foundation.sin(x)
            }
        }
        self.maxContext = maxContext
        self.halfDim = half
        self.cos = c
        self.sin = s
    }

    /// Slice one AR step at `offset`, writing into MLMultiArrays of shape
    /// `[1, 1, 1, halfDim]`.
    public func fillStep(offset: Int, cosOut: MLMultiArray, sinOut: MLMultiArray) {
        precondition(cosOut.count == halfDim && sinOut.count == halfDim,
                     "RoPE step buffer size must match halfDim (\(halfDim)); got \(cosOut.count)")
        let base = offset * halfDim
        cosOut.withUnsafeMutableBufferPointer(ofType: Float.self) { dst, _ in
            self.cos.withUnsafeBufferPointer { src in
                memcpy(dst.baseAddress!, src.baseAddress! + base,
                       halfDim * MemoryLayout<Float>.stride)
            }
        }
        sinOut.withUnsafeMutableBufferPointer(ofType: Float.self) { dst, _ in
            self.sin.withUnsafeBufferPointer { src in
                memcpy(dst.baseAddress!, src.baseAddress! + base,
                       halfDim * MemoryLayout<Float>.stride)
            }
        }
    }

    /// Slice `length` steps starting at `offset`, writing into MLMultiArrays of
    /// shape `[1, length, 1, halfDim]`.
    public func fillRange(
        offset: Int, length: Int,
        cosOut: MLMultiArray, sinOut: MLMultiArray
    ) {
        precondition(cosOut.count == length * halfDim)
        precondition(sinOut.count == length * halfDim)
        let src = offset * halfDim
        cosOut.withUnsafeMutableBufferPointer(ofType: Float.self) { dst, _ in
            self.cos.withUnsafeBufferPointer { s in
                memcpy(dst.baseAddress!, s.baseAddress! + src,
                       length * halfDim * MemoryLayout<Float>.stride)
            }
        }
        sinOut.withUnsafeMutableBufferPointer(ofType: Float.self) { dst, _ in
            self.sin.withUnsafeBufferPointer { s in
                memcpy(dst.baseAddress!, s.baseAddress! + src,
                       length * halfDim * MemoryLayout<Float>.stride)
            }
        }
    }
}
