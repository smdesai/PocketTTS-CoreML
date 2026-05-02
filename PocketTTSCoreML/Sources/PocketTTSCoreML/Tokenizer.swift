import Foundation
import CSentencePieceBridge

/// Errors surfaced by `Tokenizer`.
public enum TokenizerError: Error, CustomStringConvertible {
    case modelLoadFailed(path: String)
    case encodeFailed
    case decodeFailed

    public var description: String {
        switch self {
        case .modelLoadFailed(let path): return "Failed to load SentencePiece model at \(path)"
        case .encodeFailed:              return "SentencePiece encode failed"
        case .decodeFailed:              return "SentencePiece decode failed"
        }
    }
}

/// Minimal Swift wrapper over the SentencePiece C++ processor.
///
/// Must produce identical int32 IDs to Python's
/// `sentencepiece.SentencePieceProcessor.EncodeAsIds(text)` — this is the
/// tokenizer parity contract (see `TokenizerParityTests`).
public final class Tokenizer: @unchecked Sendable {
    private let proc: SPBProcessor

    public let vocabSize: Int

    public init(modelURL: URL) throws {
        let path = modelURL.path
        guard let p = path.withCString({ spb_create($0) }) else {
            throw TokenizerError.modelLoadFailed(path: path)
        }
        self.proc = p
        self.vocabSize = Int(spb_vocab_size(p))
    }

    deinit {
        spb_destroy(proc)
    }

    /// Encode `text` to int32 token IDs.
    public func encode(_ text: String) -> [Int32] {
        var idsPtr: UnsafeMutablePointer<Int32>? = nil
        let n = withUnsafeMutablePointer(to: &idsPtr) { outer -> Int32 in
            text.withCString { cstr in
                spb_encode_as_ids(proc, cstr, outer)
            }
        }
        guard n > 0, let ids = idsPtr else { return [] }
        defer { spb_free_ids(ids) }
        return Array(UnsafeBufferPointer(start: ids, count: Int(n)))
    }

    /// Encode and return as `[Int]` for convenience.
    public func encodeAsInt(_ text: String) -> [Int] {
        encode(text).map { Int($0) }
    }

    public func decode(_ ids: [Int32]) -> String {
        guard !ids.isEmpty else { return "" }
        return ids.withUnsafeBufferPointer { buf -> String in
            guard let cstr = spb_decode_ids(proc, buf.baseAddress, Int32(buf.count)) else {
                return ""
            }
            defer { free(cstr) }
            return String(cString: cstr)
        }
    }

    public func pieces(for text: String) -> [String] {
        var out: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>? = nil
        let n = withUnsafeMutablePointer(to: &out) { outer -> Int32 in
            text.withCString { cstr in
                spb_encode_as_pieces(proc, cstr, outer)
            }
        }
        guard n > 0, let pieces = out else { return [] }
        defer { spb_free_pieces(pieces, n) }
        var result: [String] = []
        for i in 0..<Int(n) {
            if let p = pieces[i] { result.append(String(cString: p)) }
        }
        return result
    }
}
