import Foundation
import CoreML

/// Mimi decoder packed streaming state blob (fp16 flat tensor).
///
/// Reads its own layout from the sidecar JSON `mimi_decoder.state_layout.json`
/// produced alongside the `.mlpackage` at conversion time, so the Swift
/// runtime never has to track the offsets manually.
public struct MimiStateLayout: Sendable {
    public struct Slot: Sendable {
        public let name: String
        public let shape: [Int]
        public let offset: Int
        public let length: Int
    }

    public let totalElems: Int
    public let sCap: Int
    public let tStep: Int
    public let headDim: Int
    public let numHeads: Int
    public let numTxLayers: Int
    public let slots: [Slot]

    public static func load(from url: URL) throws -> MimiStateLayout {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let totalElems = json["total_elems"] as? Int,
              let sCap = json["s_cap"] as? Int,
              let tStep = json["t_step"] as? Int,
              let headDim = json["head_dim"] as? Int,
              let numHeads = json["num_heads"] as? Int,
              let numTxLayers = json["num_tx_layers"] as? Int,
              let slotsRaw = json["slots"] as? [[String: Any]]
        else {
            throw NSError(domain: "MimiStateLayout", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "invalid layout json"
            ])
        }
        let slots: [Slot] = slotsRaw.compactMap { entry in
            guard let name = entry["name"] as? String,
                  let shape = entry["shape"] as? [Int],
                  let offset = entry["offset"] as? Int,
                  let length = entry["length"] as? Int
            else { return nil }
            return Slot(name: name, shape: shape, offset: offset, length: length)
        }
        return MimiStateLayout(
            totalElems: totalElems, sCap: sCap, tStep: tStep,
            headDim: headDim, numHeads: numHeads, numTxLayers: numTxLayers,
            slots: slots
        )
    }
}

/// Two-buffer Mimi state ping-pong. The `mimi_decoder.mlpackage` is
/// converted at FP32 compute precision (see plan §Phase 3.5 and
/// docs/phase2_3_notes.md "mimi_decoder converted at FP32 compute precision"),
/// so its state I/O is fp32.
public final class MimiStateBuffer: @unchecked Sendable {
    public let layout: MimiStateLayout
    public private(set) var currentIn: MLMultiArray
    public private(set) var currentOut: MLMultiArray

    public init(layout: MimiStateLayout) throws {
        self.layout = layout
        self.currentIn = try MLMultiArray(shape: [NSNumber(value: layout.totalElems)],
                                          dataType: .float32)
        self.currentOut = try MLMultiArray(shape: [NSNumber(value: layout.totalElems)],
                                           dataType: .float32)
        zeroIn()
    }

    public func zeroIn() {
        let n = layout.totalElems * MemoryLayout<Float>.stride
        let ptr = currentIn.dataPointer
        memset(ptr, 0, n)
    }

    /// Swap so that the just-written `currentOut` becomes the next `currentIn`.
    public func swap() {
        let tmp = currentIn
        self.currentIn = currentOut
        self.currentOut = tmp
    }
}
