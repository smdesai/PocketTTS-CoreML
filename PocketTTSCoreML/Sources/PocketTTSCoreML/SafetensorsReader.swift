import Foundation

/// Minimal safetensors reader.
///
/// Safetensors file layout:
///   | 8-byte little-endian uint64 header_size | JSON header | binary tensor blob |
///
/// Header JSON is a dict: `{name: {dtype, shape, data_offsets: [start, end]}}` plus
/// an optional `__metadata__` entry. Offsets are relative to the start of the blob.
public enum SafetensorsError: Error, CustomStringConvertible {
    case fileTooSmall
    case badHeader
    case missingKey(String)
    case unsupportedDType(String)
    case shapeMismatch(expected: [Int], actual: [Int])

    public var description: String {
        switch self {
        case .fileTooSmall: return "safetensors file is truncated"
        case .badHeader: return "safetensors header is not valid JSON"
        case .missingKey(let k): return "safetensors file missing key: \(k)"
        case .unsupportedDType(let s): return "unsupported safetensors dtype: \(s)"
        case .shapeMismatch(let e, let a):
            return "safetensors shape mismatch: expected \(e), got \(a)"
        }
    }
}

public enum SafetensorsDType: String, Sendable {
    case F16 = "F16"
    case F32 = "F32"
    case BF16 = "BF16"
    case I64 = "I64"
    case I32 = "I32"
    case I16 = "I16"
    case I8 = "I8"
    case U8 = "U8"
    case BOOL = "BOOL"

    var bytesPerElement: Int {
        switch self {
        case .F16, .BF16, .I16: return 2
        case .F32, .I32: return 4
        case .I64: return 8
        case .I8, .U8, .BOOL: return 1
        }
    }
}

public struct SafetensorsTensorInfo: Sendable {
    public let name: String
    public let dtype: SafetensorsDType
    public let shape: [Int]
    public let byteOffset: Int  // absolute offset in file
    public let byteLength: Int

    public var elementCount: Int { shape.reduce(1, *) }
}

/// Reader that mmaps the file and indexes the JSON header.
public final class SafetensorsReader: @unchecked Sendable {
    private let data: Data
    public let tensors: [String: SafetensorsTensorInfo]

    public init(url: URL) throws {
        // mmap the file. .alwaysMapped gives us a Data-backed memory view.
        self.data = try Data(contentsOf: url, options: .alwaysMapped)
        guard data.count >= 8 else { throw SafetensorsError.fileTooSmall }

        let headerSize = Int(
            data.withUnsafeBytes { raw -> UInt64 in
                raw.load(fromByteOffset: 0, as: UInt64.self).littleEndian
            })
        guard 8 + headerSize <= data.count else { throw SafetensorsError.fileTooSmall }
        let headerData = data.subdata(in: 8 ..< (8 + headerSize))

        guard
            let json = try? JSONSerialization.jsonObject(with: headerData, options: []),
            let dict = json as? [String: Any]
        else {
            throw SafetensorsError.badHeader
        }

        let blobOffset = 8 + headerSize
        var tensors: [String: SafetensorsTensorInfo] = [:]
        for (name, raw) in dict {
            if name == "__metadata__" { continue }
            guard let entry = raw as? [String: Any],
                let dtypeStr = entry["dtype"] as? String,
                let shapeRaw = entry["shape"] as? [Any],
                let offsets = entry["data_offsets"] as? [Any],
                offsets.count == 2,
                let start = (offsets[0] as? NSNumber)?.intValue,
                let end = (offsets[1] as? NSNumber)?.intValue
            else {
                throw SafetensorsError.badHeader
            }
            guard let dtype = SafetensorsDType(rawValue: dtypeStr) else {
                throw SafetensorsError.unsupportedDType(dtypeStr)
            }
            let shape = shapeRaw.compactMap { ($0 as? NSNumber)?.intValue }
            tensors[name] = SafetensorsTensorInfo(
                name: name,
                dtype: dtype,
                shape: shape,
                byteOffset: blobOffset + start,
                byteLength: end - start
            )
        }
        self.tensors = tensors
    }

    /// Raw bytes for tensor `name`. Zero-copy — backed by the same mmap.
    public func bytes(for name: String) throws -> Data {
        guard let info = tensors[name] else {
            throw SafetensorsError.missingKey(name)
        }
        return data.subdata(in: info.byteOffset ..< (info.byteOffset + info.byteLength))
    }

    /// Load as `[Float]` (fp32). Converts F16/BF16/F32 transparently.
    public func float32Array(for name: String) throws -> (values: [Float], shape: [Int]) {
        guard let info = tensors[name] else { throw SafetensorsError.missingKey(name) }
        let blob = try bytes(for: name)
        let count = info.elementCount
        var out = [Float](repeating: 0, count: count)

        switch info.dtype {
        case .F32:
            blob.withUnsafeBytes { raw in
                memcpy(&out, raw.baseAddress!, count * 4)
            }
        case .F16:
            // Use vDSP-style manual fp16->fp32 conversion.
            blob.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: UInt16.self)
                for i in 0 ..< count {
                    out[i] = Self.fp16ToFloat(src[i])
                }
            }
        case .BF16:
            blob.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: UInt16.self)
                for i in 0 ..< count {
                    // bf16 = upper 16 bits of fp32; zero-extend to fp32.
                    let bits: UInt32 = UInt32(src[i]) << 16
                    out[i] = Float(bitPattern: bits)
                }
            }
        default:
            throw SafetensorsError.unsupportedDType(info.dtype.rawValue)
        }
        return (out, info.shape)
    }

    /// Load as `[Int64]`. Accepts I64 or I32.
    public func int64Array(for name: String) throws -> (values: [Int64], shape: [Int]) {
        guard let info = tensors[name] else { throw SafetensorsError.missingKey(name) }
        let blob = try bytes(for: name)
        let count = info.elementCount
        var out = [Int64](repeating: 0, count: count)
        switch info.dtype {
        case .I64:
            blob.withUnsafeBytes { raw in
                memcpy(&out, raw.baseAddress!, count * 8)
            }
        case .I32:
            blob.withUnsafeBytes { raw in
                let src = raw.bindMemory(to: Int32.self)
                for i in 0 ..< count { out[i] = Int64(src[i]) }
            }
        default:
            throw SafetensorsError.unsupportedDType(info.dtype.rawValue)
        }
        return (out, info.shape)
    }

    /// Convert an IEEE 754 half-precision (fp16) bit pattern to Float.
    @inline(__always)
    static func fp16ToFloat(_ h: UInt16) -> Float {
        // Portable software fp16 -> fp32 conversion.
        let sign = UInt32(h & 0x8000) << 16
        let exp = UInt32((h >> 10) & 0x1F)
        let mant = UInt32(h & 0x3FF)
        var bits: UInt32 = 0
        if exp == 0 {
            if mant == 0 {
                bits = sign
            } else {
                // subnormal — renormalize
                var e: Int32 = -1
                var m = mant
                repeat {
                    e &-= 1
                    m <<= 1
                } while (m & 0x400) == 0
                let expBits = UInt32(127 + 15 + e) << 23
                let mantBits = (m & 0x3FF) << 13
                bits = sign | expBits | mantBits
            }
        } else if exp == 0x1F {
            bits = sign | 0x7F80_0000 | (mant << 13)
        } else {
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13)
        }
        return Float(bitPattern: bits)
    }
}

// MARK: - Writer

/// Minimal safetensors writer (F32 tensors only — enough for voice export).
public enum SafetensorsWriter {
    public struct Tensor {
        public let name: String
        public let shape: [Int]
        public let dtype: SafetensorsDType
        public let data: Data
        public init(name: String, shape: [Int], dtype: SafetensorsDType, data: Data) {
            self.name = name
            self.shape = shape
            self.dtype = dtype
            self.data = data
        }
    }

    public static func write(_ tensors: [Tensor], to url: URL) throws {
        // Build header JSON. Sort names deterministically.
        var header: [String: Any] = [:]
        var offset = 0
        for t in tensors.sorted(by: { $0.name < $1.name }) {
            let length = t.shape.reduce(1, *) * t.dtype.bytesPerElement
            header[t.name] = [
                "dtype": t.dtype.rawValue,
                "shape": t.shape,
                "data_offsets": [offset, offset + length],
            ]
            offset += length
        }
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        // Pad header to 8-byte alignment.
        var headerBytes = json
        let pad = (8 - headerBytes.count % 8) % 8
        if pad > 0 { headerBytes.append(Data(repeating: 0x20, count: pad)) }  // space pad
        // Serialize headerSize as 8-byte little-endian without relying on
        // `withUnsafePointer(to:)`-returning-a-pointer (that pointer is
        // only valid inside the closure body; using it afterwards is a
        // use-after-scope that surfaced on iPhone as garbage bytes in
        // the header-size prefix, corrupting the file).
        var headerSizeLE = UInt64(headerBytes.count).littleEndian
        let headerSizeBytes = Data(
            bytes: &headerSizeLE,
            count: MemoryLayout<UInt64>.size
        )

        var out = Data()
        out.append(headerSizeBytes)
        out.append(headerBytes)
        for t in tensors.sorted(by: { $0.name < $1.name }) {
            out.append(t.data)
        }
        try out.write(to: url, options: .atomic)
    }
}
