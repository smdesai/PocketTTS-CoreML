import XCTest
import Foundation
@testable import PocketTTSCoreML

/// Phase-4B: end-to-end with in-Swift text prefill.
///
/// Loads the raw voice safetensors (not a pre-prefilled bundle) and
/// runs the full pipeline — text_conditioner + flow_lm_prefill +
/// flow_lm_main AR loop + flow_lm_flow + mimi_decoder — and gates on
/// PSNR ≥ 15 dB vs. the Phase-1 golden.
///
/// Skips if the prefill mlpackage or voice file is missing.
final class PrefillIntegrationTest: XCTestCase {

    func testVoiceOnlyGenerationMatchesGolden() async throws {
        try XCTSkipUnless(FixturePaths.artifactsAvailable,
                          "mlpackages not available at \(FixturePaths.artifactsDir.path)")
        try XCTSkipUnless(FixturePaths.tokenizerAvailable, "tokenizer missing")
        try XCTSkipUnless(FixturePaths.voiceAvailable,
                          "alba.safetensors missing at \(FixturePaths.voiceURL.path)")
        // Requires the new prefill .mlpackage + bos_emb sidecar.
        let prefillURL = FixturePaths.artifactsDir
            .appendingPathComponent("flow_lm_prefill.mlpackage")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: prefillURL.path),
            "flow_lm_prefill.mlpackage missing — run `python -m pockettts_coreml.convert --only flow_lm_prefill`"
        )

        let tts = try await PocketTTS(
            artifactsBundle: FixturePaths.artifactsDir,
            tokenizerPath: FixturePaths.tokenizerURL,
            computeUnits: .cpuAndNeuralEngine
        )
        // Load the ORIGINAL voice .safetensors — VoiceLoader auto-detects
        // (no `flow_kv_rank5` key => voice-only handle).
        let voice = try await tts.loadVoice(from: FixturePaths.voiceURL)
        guard case .voiceOnly = voice.kind else {
            XCTFail("Expected .voiceOnly after loading raw voice bundle"); return
        }

        var samples: [Int16] = []
        let startWall = Date()
        let stream = await tts.generate(
            text: "Pocket TTS is a lightweight text-to-speech model.",
            voice: voice
        )
        for try await pcm in stream {
            pcm.withUnsafeBytes { raw in
                let n = raw.count / 2
                let buf = raw.bindMemory(to: Int16.self)
                for i in 0..<n { samples.append(buf[i]) }
            }
        }
        let elapsed = Date().timeIntervalSince(startWall)
        let audioSec = Double(samples.count) / Double(PocketTTSArch.sampleRate)
        let coldRTF = elapsed / max(audioSec, 1e-9)

        XCTAssertGreaterThan(samples.count, 24_000, "expected ≥ 1 s of audio")
        let peak = samples.map { abs(Int($0)) }.max() ?? 0
        XCTAssertGreaterThan(peak, 1000, "audio appears silent / broken after prefill")

        if FixturePaths.goldenWavAvailable {
            let (goldenFloats, sr) = try AudioStream.readWAVAsFloat(FixturePaths.goldenWavURL)
            XCTAssertEqual(sr, PocketTTSArch.sampleRate)
            let swiftFloats = samples.map { Float($0) / 32768.0 }
            let psnr = psnrFull(swiftFloats, goldenFloats)
            print("[PrefillIntegration] RTF(cold,with-prefill)=\(coldRTF) | PSNR=\(psnr) dB | audio=\(audioSec)s")
            XCTAssertGreaterThanOrEqual(
                psnr, 15.0,
                "voice-only prefill path produced audio below PSNR floor (15 dB)"
            )
        }
    }

    private func psnrFull(_ a: [Float], _ b: [Float]) -> Float {
        let n = min(a.count, b.count)
        guard n > 0 else { return 0 }
        var mse: Double = 0
        for i in 0..<n {
            let d = Double(a[i] - b[i])
            mse += d * d
        }
        mse /= Double(n)
        if mse <= 0 { return .infinity }
        return Float(10 * log10(1.0 / mse))
    }
}
