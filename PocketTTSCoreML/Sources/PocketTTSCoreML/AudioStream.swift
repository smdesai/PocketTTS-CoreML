import Accelerate
import CoreML
import Foundation

/// Audio frame helpers — Float32 samples to/from int16 PCM (little-endian).
public enum AudioStream {
    /// Convert `fp32[1, 1, 1920]` MLMultiArray to `Data` (int16 LE PCM).
    public static func frameToPCM16(_ array: MLMultiArray, clip: Bool = true) -> Data {
        let n = array.count
        var floats = [Float](repeating: 0, count: n)
        array.withUnsafeBufferPointer(ofType: Float.self) { src in
            memcpy(&floats, src.baseAddress!, n * MemoryLayout<Float>.stride)
        }
        if clip {
            // Clamp to [-1, 1] to avoid wraparound.
            var low: Float = -1
            var high: Float = 1
            vDSP_vclip(floats, 1, &low, &high, &floats, 1, vDSP_Length(n))
        }
        var scale: Float = 32767.0
        vDSP_vsmul(floats, 1, &scale, &floats, 1, vDSP_Length(n))
        var int16s = [Int16](repeating: 0, count: n)
        vDSP_vfixr16(floats, 1, &int16s, 1, vDSP_Length(n))
        return int16s.withUnsafeBufferPointer { buf in
            Data(buffer: buf)
        }
    }

    /// Write a 24 kHz mono WAV file from in-memory int16 PCM.
    public static func writeWAV(
        _ pcm: Data, sampleRate: Int = PocketTTSArch.sampleRate,
        to url: URL
    ) throws {
        var header = Data(capacity: 44)
        let byteRate = UInt32(sampleRate * 2)
        let subchunk2Size = UInt32(pcm.count)
        let chunkSize = 36 + subchunk2Size
        header.append("RIFF".data(using: .ascii)!)
        header.appendLE(UInt32(chunkSize))
        header.append("WAVE".data(using: .ascii)!)
        header.append("fmt ".data(using: .ascii)!)
        header.appendLE(UInt32(16))  // PCM fmt chunk
        header.appendLE(UInt16(1))  // PCM format
        header.appendLE(UInt16(1))  // mono
        header.appendLE(UInt32(sampleRate))
        header.appendLE(byteRate)
        header.appendLE(UInt16(2))  // block align
        header.appendLE(UInt16(16))  // bits/sample
        header.append("data".data(using: .ascii)!)
        header.appendLE(subchunk2Size)
        var out = header
        out.append(pcm)
        try out.write(to: url, options: .atomic)
    }

    /// Convenience: read a 24 kHz mono WAV back into `[Float]` samples.
    public static func readWAVAsFloat(_ url: URL) throws -> (samples: [Float], sampleRate: Int) {
        let data = try Data(contentsOf: url)
        // Minimal RIFF reader — assumes well-formed 16-bit mono file.
        guard data.count >= 44 else {
            throw NSError(
                domain: "AudioStream", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "WAV too small"])
        }
        let sr = Int(data.loadLE(offset: 24, as: UInt32.self))
        // Find "data" chunk.
        var cursor = 12
        var dataOffset = -1
        var dataLen = 0
        while cursor + 8 <= data.count {
            let id = data.subdata(in: cursor ..< (cursor + 4))
            let size = Int(data.loadLE(offset: cursor + 4, as: UInt32.self))
            if String(data: id, encoding: .ascii) == "data" {
                dataOffset = cursor + 8
                dataLen = size
                break
            }
            cursor += 8 + size
        }
        guard dataOffset >= 0 else {
            throw NSError(
                domain: "AudioStream", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "WAV missing data chunk"])
        }
        let pcm = data.subdata(in: dataOffset ..< (dataOffset + dataLen))
        let count = dataLen / 2
        var ints = [Int16](repeating: 0, count: count)
        pcm.withUnsafeBytes { raw in
            memcpy(&ints, raw.baseAddress!, count * 2)
        }
        var floats = [Float](repeating: 0, count: count)
        for i in 0 ..< count { floats[i] = Float(ints[i]) / 32768.0 }
        return (floats, sr)
    }
}

extension Data {
    mutating func appendLE<T: FixedWidthInteger>(_ value: T) {
        var v = value.littleEndian
        Swift.withUnsafeBytes(of: &v) { buf in
            self.append(buf.bindMemory(to: UInt8.self).baseAddress!, count: buf.count)
        }
    }

    func loadLE<T: FixedWidthInteger>(offset: Int, as: T.Type) -> T {
        return self.withUnsafeBytes { raw -> T in
            let v = raw.load(fromByteOffset: offset, as: T.self)
            return v.littleEndian
        }
    }
}
