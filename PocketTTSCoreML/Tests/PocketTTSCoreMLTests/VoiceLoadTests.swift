import XCTest
@testable import PocketTTSCoreML

final class VoiceLoadTests: XCTestCase {

    func testVoiceOnlyLoad() throws {
        try XCTSkipUnless(FixturePaths.voiceAvailable,
                          "alba.safetensors not available at \(FixturePaths.voiceURL.path)")
        let handle = try VoiceLoader.load(url: FixturePaths.voiceURL)
        guard case let .voiceOnly(layers, _) = handle.kind else {
            XCTFail("Expected voice-only handle"); return
        }
        XCTAssertEqual(layers.count, PocketTTSArch.flowLayers, "expected 6 FlowLM layers")
        for (i, layer) in layers.enumerated() {
            XCTAssertEqual(layer.offset, 126, "layer \(i) offset should match voice prefix length")
            XCTAssertEqual(layer.tVoice, 126, "layer \(i) T_voice should be 126 slots")
            let expectedFloats = 2 * 1 * 126 * PocketTTSArch.flowHeads * PocketTTSArch.flowHeadDim
            XCTAssertEqual(layer.cache.count, expectedFloats,
                           "layer \(i) flat cache size mismatch")
            // Sanity: at least some non-zero values.
            XCTAssertTrue(layer.cache.contains(where: { $0 != 0 && !$0.isNaN }),
                          "layer \(i) cache is all-zero — voice prefix probably dropped")
        }
    }

    func testPrefilledLoadIfAvailable() throws {
        try XCTSkipUnless(FixturePaths.prefilledVoiceAvailable,
                          "alba_prefilled.safetensors not available — generate with `python -m pockettts_coreml.e2e.export_full_prefill`")
        let handle = try VoiceLoader.load(url: FixturePaths.prefilledVoiceURL)
        guard case let .prefilled(kvBox, offset, bos, _, _) = handle.kind else {
            XCTFail("Expected prefilled handle"); return
        }
        let expectedCount = (2 * PocketTTSArch.flowLayers) * 1 * PocketTTSArch.flowSCap
            * PocketTTSArch.flowHeads * PocketTTSArch.flowHeadDim
        XCTAssertEqual(kvBox.array.count, expectedCount)
        XCTAssertGreaterThan(offset, 0)
        XCTAssertEqual(bos.count, PocketTTSArch.latentDim)
    }

    func testSafetensorsRoundtrip() throws {
        // Write a tiny safetensors and read it back.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("pockettts_test_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tmp) }

        let floats: [Float] = [1.0, 2.0, 3.0, 4.0]
        let data = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        try SafetensorsWriter.write([
            .init(name: "foo", shape: [4], dtype: .F32, data: data)
        ], to: tmp)

        let reader = try SafetensorsReader(url: tmp)
        let (back, shape) = try reader.float32Array(for: "foo")
        XCTAssertEqual(shape, [4])
        XCTAssertEqual(back, floats)
    }
}
