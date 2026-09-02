//
//  NoiseSource.swift
//  PocketTTSCoreML
//
//  Created by Sachin Desai on 5/3/26.
//

import Foundation

// Deterministic Gaussian noise source with a minimal Xorshift backing for
// testable reproducibility. The reference uses PyTorch's Mersenne Twister
// with `torch.nn.init.normal_`; here we expose a seedable API but don't
// aim for torch-identical sequences — we only need deterministic
// Swift-side repro. For parity vs the Python reference, the Python side's
// noise samples can be precomputed and injected via `setPrecomputed`.
public final class NoiseSource: @unchecked Sendable {
    private var state0: UInt64
    private var state1: UInt64

    public var precomputed: [[Float]]?  // if non-nil, pops one per call
    private var pcIndex: Int = 0

    public let std: Float

    public init(seed: UInt64 = 42, std: Float = 1.0) {
        self.state0 = (seed << 1) | 1
        self.state1 = ~seed &+ 0xA5A5_5A5A_A5A5_5A5A
        self.std = std
    }

    public func setPrecomputed(_ seq: [[Float]]) {
        self.precomputed = seq
        self.pcIndex = 0
    }

    // Draw `count` samples from N(0, std^2).
    public func sample(count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        fillSample(&out, count: count)
        return out
    }

    public func fillSample(_ out: inout [Float], count: Int) {
        precondition(out.count >= count)
        if let pre = precomputed, pcIndex < pre.count {
            let v = pre[pcIndex]
            pcIndex += 1
            let copyCount = min(count, v.count)
            for i in 0 ..< copyCount { out[i] = v[i] }
            if copyCount < count {
                for i in copyCount ..< count { out[i] = 0 }
            }
            return
        }
        var i = 0
        while i < count {
            let u1 = max(nextUniform(), 1e-7)
            let u2 = nextUniform()
            let r = sqrt(-2 * log(u1))
            let theta = 2 * Float.pi * u2
            out[i] = r * cos(theta) * std
            if i + 1 < count { out[i + 1] = r * sin(theta) * std }
            i += 2
        }
    }

    @inline(__always)
    private func nextUniform() -> Float {
        // xoroshiro128+ minimal impl.
        let s0 = state0
        var s1 = state1
        let result = s0 &+ s1
        s1 ^= s0
        state0 = ((s0 << 24) | (s0 >> 40)) ^ s1 ^ (s1 << 16)
        state1 = (s1 << 37) | (s1 >> 27)
        let u = (result >> 11) & ((UInt64(1) << 53) - 1)
        return Float(Double(u) / Double(UInt64(1) << 53))
    }
}
