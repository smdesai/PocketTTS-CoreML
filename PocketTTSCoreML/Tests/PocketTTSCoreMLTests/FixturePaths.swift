import Foundation

/// Resolves the repo root and paths to gated test fixtures (tokenizer,
/// voice safetensors, mlpackages). We walk upward from this source file's
/// location, which is robust for `swift test` runs regardless of cwd.
enum FixturePaths {
    static var repoRoot: URL {
        if let env = ProcessInfo.processInfo.environment["POCKETTTS_REPO_ROOT"] {
            return URL(fileURLWithPath: env)
        }
        // #filePath points at .../PocketTTSCoreML/Tests/PocketTTSCoreMLTests/FixturePaths.swift.
        // Walk up to the dir containing both `PocketTTSCoreML` and `Artifacts`.
        var u = URL(fileURLWithPath: #filePath)
        u.deleteLastPathComponent()  // Tests/PocketTTSCoreMLTests
        u.deleteLastPathComponent()  // Tests
        u.deleteLastPathComponent()  // PocketTTSCoreML
        u.deleteLastPathComponent()  // repo root
        return u
    }

    static var fixturesDir: URL {
        repoRoot
            .appendingPathComponent("pockettts_coreml")
            .appendingPathComponent("oracle")
            .appendingPathComponent("fixtures")
            .appendingPathComponent("english_alba_seed42")
    }

    static var tokenizerURL: URL { fixturesDir.appendingPathComponent("tokenizer.model") }
    static var voiceURL: URL { fixturesDir.appendingPathComponent("alba.safetensors") }
    static var prefilledVoiceURL: URL {
        fixturesDir.appendingPathComponent("alba_prefilled.safetensors")
    }
    static var goldenWavURL: URL { fixturesDir.appendingPathComponent("golden/output.wav") }

    static var artifactsDir: URL {
        repoRoot.appendingPathComponent("Artifacts").appendingPathComponent("en_fp16")
    }

    static var tokenizerAvailable: Bool {
        FileManager.default.fileExists(atPath: tokenizerURL.path)
    }
    static var voiceAvailable: Bool {
        FileManager.default.fileExists(atPath: voiceURL.path)
    }
    static var prefilledVoiceAvailable: Bool {
        FileManager.default.fileExists(atPath: prefilledVoiceURL.path)
    }
    static var goldenWavAvailable: Bool {
        FileManager.default.fileExists(atPath: goldenWavURL.path)
    }
    static var artifactsAvailable: Bool {
        let paths = [
            "flow_lm_main.mlpackage",
            "flow_lm_flow.mlpackage",
            "mimi_decoder.mlpackage",
            "mimi_decoder.state_layout.json",
        ]
        for p in paths {
            if !FileManager.default.fileExists(atPath: artifactsDir.appendingPathComponent(p).path)
            {
                return false
            }
        }
        return true
    }
}
