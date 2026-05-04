import Foundation
import XCTest

@testable import PocketTTSCoreML

/// Measure RTF for the canonical prompt and assert ≤ 0.40 on Mac. Skipped
/// if artifacts aren't present (this is effectively a smoke bench).
final class BenchmarkTests: XCTestCase {

    func testMacRTF() async throws {
        try XCTSkipUnless(FixturePaths.artifactsAvailable, "mlpackages missing")
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
        await tts.warmup()

        let iterations = 3
        var best = Double.infinity
        for iter in 0 ..< iterations {
            let t0 = Date()
            var samples = 0
            let stream = await tts.generate(
                text: "Pocket TTS is a lightweight text-to-speech model.",
                voice: voice
            )
            for try await pcm in stream {
                samples += pcm.count / 2
            }
            let elapsed = Date().timeIntervalSince(t0)
            let audioSec = Double(samples) / Double(PocketTTSArch.sampleRate)
            let rtf = elapsed / max(audioSec, 1e-9)
            best = min(best, rtf)
            print("[Benchmark] iter=\(iter) audio=\(audioSec)s wall=\(elapsed)s rtf=\(rtf)")
        }
        print("[Benchmark] best RTF = \(best)")

        // Phase 4A gate: Mac RTF ≤ 0.40.
        XCTAssertLessThanOrEqual(
            best, 0.40,
            "Best Mac RTF \(best) exceeds the 0.40 gate")
    }
}
