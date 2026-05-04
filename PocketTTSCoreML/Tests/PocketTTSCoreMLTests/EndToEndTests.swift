import Foundation
import XCTest

@testable import PocketTTSCoreML

/// Full pipeline test: requires the Artifacts dir + tokenizer + a pre-
/// prefilled voice bundle. Skipped automatically if any are missing.
///
/// Phase 4A: text prefill is done in Python (see tools/export_full_prefill.py).
/// The Swift side consumes the prefilled bundle and runs the AR loop.
final class EndToEndTests: XCTestCase {

    func testCanonicalPromptGeneration() async throws {
        try XCTSkipUnless(
            FixturePaths.artifactsAvailable,
            "mlpackages not available at \(FixturePaths.artifactsDir.path)")
        try XCTSkipUnless(FixturePaths.tokenizerAvailable, "tokenizer missing")
        try XCTSkipUnless(
            FixturePaths.prefilledVoiceAvailable,
            "alba_prefilled.safetensors missing — regenerate via export_full_prefill.py")

        let tts = try await PocketTTS(
            artifactsBundle: FixturePaths.artifactsDir,
            tokenizerPath: FixturePaths.tokenizerURL,
            computeUnits: .cpuAndNeuralEngine
        )
        let voice = try await tts.loadVoice(from: FixturePaths.prefilledVoiceURL)

        var samples: [Int16] = []
        let stream = await tts.generate(
            text: "Pocket TTS is a lightweight text-to-speech model.",
            voice: voice
        )
        for try await pcm in stream {
            pcm.withUnsafeBytes { raw in
                let n = raw.count / 2
                let buf = raw.bindMemory(to: Int16.self)
                for i in 0 ..< n { samples.append(buf[i]) }
            }
        }

        // Basic sanity: non-empty, finite, non-silent.
        XCTAssertGreaterThan(samples.count, 24_000 * 1, "expected at least 1 second of audio")
        let peak = samples.map { abs(Int($0)) }.max() ?? 0
        XCTAssertGreaterThan(peak, 1000, "audio appears silent / broken")

        // Compare against golden — PSNR gate.
        if FixturePaths.goldenWavAvailable {
            let (goldenFloats, sr) = try AudioStream.readWAVAsFloat(FixturePaths.goldenWavURL)
            XCTAssertEqual(sr, PocketTTSArch.sampleRate)
            let swiftFloats = samples.map { Float($0) / 32768.0 }

            let (overallPSNR, bestEarly) = psnrAnalysis(swiftFloats, goldenFloats)
            print(
                "[EndToEnd] overall PSNR \(overallPSNR) dB | best early-frame PSNR \(bestEarly) dB")
            XCTAssertGreaterThanOrEqual(
                overallPSNR, 15,
                "overall PSNR below 15 dB floor")
            XCTAssertGreaterThanOrEqual(
                bestEarly, 25,
                "best early-frame PSNR below 25 dB floor")
        }
    }

    /// Compute (overall_PSNR_dB, best_early_frame_PSNR_dB). Early frames =
    /// first 5 frames of 1920 samples. Length-normalized by truncating to
    /// the shorter of the two signals.
    private func psnrAnalysis(_ a: [Float], _ b: [Float]) -> (Float, Float) {
        let n = min(a.count, b.count)
        guard n > 0 else { return (0, 0) }
        func psnr(_ start: Int, _ count: Int) -> Float {
            var mse: Double = 0
            for i in 0 ..< count {
                let d = Double(a[start + i] - b[start + i])
                mse += d * d
            }
            mse /= Double(count)
            if mse <= 0 { return .infinity }
            return Float(10 * log10(1.0 / mse))
        }
        let overall = psnr(0, n)
        let frame = PocketTTSArch.frameSize
        let framesAvailable = min(5, n / frame)
        var bestEarly: Float = -.infinity
        for f in 0 ..< framesAvailable {
            let v = psnr(f * frame, frame)
            if v > bestEarly { bestEarly = v }
        }
        return (overall, bestEarly)
    }
}
