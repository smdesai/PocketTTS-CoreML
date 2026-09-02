//
//  Float16Ops.swift
//  PocketTTSCoreML
//
//  Created by Sachin Desai on 5/3/26.
//

import Accelerate
import Foundation

// Fast fp32 <-> fp16 bulk conversion via vImage. Allocates a temporary
// workspace per call; prefer using the MLMultiArray overloads below which
// operate in-place.
public enum Float16Ops {
    public static func fp32ToFp16(_ src: [Float]) -> [UInt16] {
        var input = src
        var output = [UInt16](repeating: 0, count: src.count)
        input.withUnsafeMutableBufferPointer { inputBuffer in
            output.withUnsafeMutableBufferPointer { outputBuffer in
                var srcBuf = vImage_Buffer(
                    data: inputBuffer.baseAddress,
                    height: 1,
                    width: vImagePixelCount(src.count),
                    rowBytes: src.count * 4)
                var dstBuf = vImage_Buffer(
                    data: outputBuffer.baseAddress,
                    height: 1,
                    width: vImagePixelCount(src.count),
                    rowBytes: src.count * 2)
                vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
            }
        }
        return output
    }

    public static func fp16ToFp32(_ src: [UInt16]) -> [Float] {
        var input = src
        var output = [Float](repeating: 0, count: src.count)
        input.withUnsafeMutableBufferPointer { inputBuffer in
            output.withUnsafeMutableBufferPointer { outputBuffer in
                var srcBuf = vImage_Buffer(
                    data: inputBuffer.baseAddress,
                    height: 1,
                    width: vImagePixelCount(src.count),
                    rowBytes: src.count * 2)
                var dstBuf = vImage_Buffer(
                    data: outputBuffer.baseAddress,
                    height: 1,
                    width: vImagePixelCount(src.count),
                    rowBytes: src.count * 4)
                vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
            }
        }
        return output
    }

    public static func fp32ToFp16Bits(_ value: Float) -> UInt16 {
        var input = value
        var output: UInt16 = 0
        withUnsafeMutableBytes(of: &input) { inputBuffer in
            withUnsafeMutableBytes(of: &output) { outputBuffer in
                var srcBuf = vImage_Buffer(
                    data: inputBuffer.baseAddress!, height: 1, width: 1, rowBytes: 4)
                var dstBuf = vImage_Buffer(
                    data: outputBuffer.baseAddress!, height: 1, width: 1, rowBytes: 2)
                vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
            }
        }
        return output
    }

    // Convert a planar fp32 memory buffer to fp16 in-place (dst has same length, half byte count).
    public static func convertFp32ToFp16(
        srcPtr: UnsafePointer<Float>, dstPtr: UnsafeMutablePointer<UInt16>, count: Int
    ) {
        var srcBuf = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: srcPtr),
            height: 1, width: vImagePixelCount(count), rowBytes: count * 4
        )
        var dstBuf = vImage_Buffer(
            data: UnsafeMutableRawPointer(dstPtr),
            height: 1, width: vImagePixelCount(count), rowBytes: count * 2
        )
        vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
    }

    // Convert a planar fp16 memory buffer to fp32.
    public static func convertFp16ToFp32(
        srcPtr: UnsafePointer<UInt16>, dstPtr: UnsafeMutablePointer<Float>, count: Int
    ) {
        var srcBuf = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: srcPtr),
            height: 1, width: vImagePixelCount(count), rowBytes: count * 2
        )
        var dstBuf = vImage_Buffer(
            data: UnsafeMutableRawPointer(dstPtr),
            height: 1, width: vImagePixelCount(count), rowBytes: count * 4
        )
        vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
    }
}
