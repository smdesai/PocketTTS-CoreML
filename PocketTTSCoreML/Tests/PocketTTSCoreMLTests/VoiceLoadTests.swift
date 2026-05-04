import XCTest

@testable import PocketTTSCoreML

final class VoiceLoadTests: XCTestCase {

    func testVoiceOnlyLoad() throws {
        try XCTSkipUnless(
            FixturePaths.voiceAvailable,
            "alba.safetensors not available at \(FixturePaths.voiceURL.path)")
        let handle = try VoiceLoader.load(url: FixturePaths.voiceURL)
        guard case .voiceOnly(let kvBox, let voiceOffset, _) = handle.kind else {
            XCTFail("Expected voice-only handle")
            return
        }
        // Rank-5 [12, 1, 256, 16, 64] fp16, packed from the 6 per-layer caches.
        let expected =
            (2 * PocketTTSArch.flowLayers) * 1 * PocketTTSArch.flowSCap
            * PocketTTSArch.flowHeads * PocketTTSArch.flowHeadDim
        XCTAssertEqual(kvBox.array.count, expected, "rank-5 KV size mismatch")
        XCTAssertEqual(
            voiceOffset, 126,
            "alba voice prefix length should be 126 after 4s audio @ 12.5 Hz")
        XCTAssertEqual(kvBox.array.dataType, .float16)
        // Sanity: voice slots [0, 126) carry signal; slots [126, 256) are zero.
        // Check by sampling layer 0's K row 0 and comparing to row 200.
        let ptr = kvBox.array.dataPointer.bindMemory(to: UInt16.self, capacity: kvBox.array.count)
        let rowStride = PocketTTSArch.flowSCap * PocketTTSArch.flowHeads * PocketTTSArch.flowHeadDim
        let perSlot = PocketTTSArch.flowHeads * PocketTTSArch.flowHeadDim
        // Layer 0 K = row 0 of rank-5.
        let slot0K = ptr.advanced(by: 0 * rowStride + 0 * perSlot)
        let slot200K = ptr.advanced(by: 0 * rowStride + 200 * perSlot)
        var any0 = false
        for i in 0 ..< perSlot where slot0K[i] != 0 {
            any0 = true
            break
        }
        XCTAssertTrue(any0, "layer 0 K slot 0 should be non-zero (voice KV)")
        for i in 0 ..< perSlot {
            XCTAssertEqual(slot200K[i], 0, "slot 200 (past voice_len=126) must be zero")
        }
    }

    func testPrefilledLoadIfAvailable() throws {
        try XCTSkipUnless(
            FixturePaths.prefilledVoiceAvailable,
            "alba_prefilled.safetensors not available — generate with `python -m pockettts_coreml.e2e.export_full_prefill`"
        )
        let handle = try VoiceLoader.load(url: FixturePaths.prefilledVoiceURL)
        guard case .prefilled(let kvBox, let offset, let bos, _, _) = handle.kind else {
            XCTFail("Expected prefilled handle")
            return
        }
        let expectedCount =
            (2 * PocketTTSArch.flowLayers) * 1 * PocketTTSArch.flowSCap
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
        try SafetensorsWriter.write(
            [
                .init(name: "foo", shape: [4], dtype: .F32, data: data)
            ], to: tmp)

        let reader = try SafetensorsReader(url: tmp)
        let (back, shape) = try reader.float32Array(for: "foo")
        XCTAssertEqual(shape, [4])
        XCTAssertEqual(back, floats)
    }
}
