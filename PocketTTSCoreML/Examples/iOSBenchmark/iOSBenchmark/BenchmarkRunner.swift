//
// BenchmarkRunner.swift
//
// Drives PocketTTS for three MLComputeUnits configurations on the device
// and records cold / warm wall-clock timings + audio duration. Also
// collects a per-submodel ComputePlanReport via ComputePlanInspector.
//
// This file is intentionally UI-agnostic; ContentView observes the
// ObservableObject on the main actor.
//

import Foundation
import CoreML
import PocketTTSCoreML

// MARK: - Report types

/// A single RTF measurement for one compute-units configuration.
public struct BenchmarkRow: Codable, Identifiable, Sendable {
    public var id: String { mode }
    /// e.g. "cpuOnly", "cpuAndNeuralEngine", "all".
    public let mode: String
    /// Audio duration of the canonical prompt in seconds.
    public let audioSeconds: Double
    /// First predict (includes prefill + program caching).
    public let coldWallSeconds: Double
    /// Median of N warm runs.
    public let warmWallSecondsMedian: Double
    /// coldWallSeconds / audioSeconds.
    public let coldRTF: Double
    /// warmWallSecondsMedian / audioSeconds.
    public let warmRTF: Double
    /// Highest thermal state observed while this mode was running.
    public let peakThermalState: String
    /// True if peakThermalState > .nominal.
    public let throttled: Bool
    /// Per-submodel MLComputePlan aggregations (6 rows).
    public let computePlan: [ComputePlanReport]
    /// Ship-gate verdict for this row: "GREEN", "YELLOW", "RED".
    public let verdict: String
}

/// Snapshot of a full run (all 3 modes).
public struct BenchmarkReport: Codable, Sendable {
    public let timestamp: String
    public let deviceModel: String
    public let systemVersion: String
    public let rows: [BenchmarkRow]
    public let prompt: String
    public let voiceName: String
    public let notes: String
}

// MARK: - Runner

@MainActor
public final class BenchmarkRunner: ObservableObject {

    // The canonical prompt, identical to Python/Mac benchmarks.
    public static let canonicalPrompt =
        "Pocket TTS is a lightweight text-to-speech model."

    @Published public private(set) var isRunning = false
    @Published public private(set) var status = "Idle."
    @Published public private(set) var rows: [BenchmarkRow] = []
    @Published public private(set) var lastReport: BenchmarkReport?
    @Published public internal(set) var errorMessage: String?

    /// Number of warm iterations to run after the cold one.
    public var warmIterations: Int = 3

    /// Paths resolved from bundled resources.
    public let artifactsDir: URL
    public let voiceURL: URL
    public let tokenizerURL: URL

    /// Fallback constructor when resources are missing — builds a runner
    /// with `errorMessage` pre-populated so the UI has something to show.
    public static func failed(with message: String) -> BenchmarkRunner {
        let r = BenchmarkRunner(fallback: message)
        return r
    }

    private init(fallback message: String) {
        // Provide dummy URLs; `run()` will refuse to start if errorMessage
        // is populated at launch.
        let tmp = FileManager.default.temporaryDirectory
        self.artifactsDir = tmp
        self.voiceURL = tmp.appendingPathComponent("missing.safetensors")
        self.tokenizerURL = tmp.appendingPathComponent("missing.model")
        self.errorMessage = message
        self.status = "Initialization failed."
    }

    public init() throws {
        let bundle = Bundle.main
        guard let artifactsDir = bundle.url(
            forResource: "Artifacts", withExtension: nil
        ) ?? bundle.url(
            forResource: "en_alba_fp16", withExtension: nil
        ) else {
            throw BenchmarkError.missingResource(
                "Artifacts/ directory not bundled with app"
            )
        }
        // Prefer en_alba_fp16 sub-directory if the outer folder was copied.
        let inner = artifactsDir.appendingPathComponent("en_alba_fp16", isDirectory: true)
        if FileManager.default.fileExists(atPath: inner.path) {
            self.artifactsDir = inner
        } else {
            self.artifactsDir = artifactsDir
        }

        guard let voice = bundle.url(
            forResource: "alba", withExtension: "safetensors"
        ) else {
            throw BenchmarkError.missingResource("alba.safetensors not bundled")
        }
        self.voiceURL = voice

        guard let tokenizer = bundle.url(
            forResource: "tokenizer", withExtension: "model"
        ) else {
            throw BenchmarkError.missingResource("tokenizer.model not bundled")
        }
        self.tokenizerURL = tokenizer
    }

    public enum BenchmarkError: LocalizedError {
        case missingResource(String)
        case internalInconsistency(String)

        public var errorDescription: String? {
            switch self {
            case .missingResource(let m): return "Missing resource: \(m)"
            case .internalInconsistency(let m): return "Internal error: \(m)"
            }
        }
    }

    // MARK: - Entry point

    public func run() async {
        guard !isRunning else { return }
        // Refuse to start if we're in the failed-init fallback state —
        // the error message is already on-screen.
        if errorMessage != nil
            && !FileManager.default.fileExists(atPath: voiceURL.path) {
            return
        }
        isRunning = true
        errorMessage = nil
        rows = []
        lastReport = nil
        defer { isRunning = false }

        let modes: [(MLComputeUnits, String)] = [
            (.cpuOnly, "cpuOnly"),
            (.cpuAndNeuralEngine, "cpuAndNeuralEngine"),
            (.all, "all"),
        ]

        var collected: [BenchmarkRow] = []
        do {
            for (cu, label) in modes {
                status = "Running mode \(label)…"
                let row = try await runOne(mode: cu, label: label)
                collected.append(row)
                rows = collected
            }

            let report = BenchmarkReport(
                timestamp: ISO8601DateFormatter().string(from: Date()),
                deviceModel: Self.modelIdentifier(),
                systemVersion: ProcessInfo.processInfo.operatingSystemVersionString,
                rows: collected,
                prompt: Self.canonicalPrompt,
                voiceName: "alba",
                notes: "Warm=median of \(warmIterations); thermal state gated at .nominal."
            )
            lastReport = report
            status = "Done."
        } catch {
            errorMessage = error.localizedDescription
            status = "Failed: \(error.localizedDescription)"
        }
    }

    // MARK: - Per-mode execution

    private func runOne(mode: MLComputeUnits, label: String) async throws -> BenchmarkRow {
        var peakThermal = ProcessInfo.processInfo.thermalState

        let tts = try await PocketTTS(
            artifactsBundle: artifactsDir,
            tokenizerPath: tokenizerURL,
            computeUnits: mode
        )
        let voice = try await tts.loadVoice(from: voiceURL)

        // warmup — flow_lm_flow pre-pass, ignored from timing.
        await tts.warmup()
        peakThermal = Self.maxThermal(peakThermal, ProcessInfo.processInfo.thermalState)

        // Cold run — first generate(). Includes prefill.
        let (coldWall, coldAudio) = try await Self.timedGenerate(
            tts: tts, voice: voice
        )
        peakThermal = Self.maxThermal(peakThermal, ProcessInfo.processInfo.thermalState)

        // Warm runs — collect wall times, use median.
        var warmWalls: [Double] = []
        for i in 0..<warmIterations {
            status = "\(label): warm run \(i + 1)/\(warmIterations)…"
            let (wall, _) = try await Self.timedGenerate(tts: tts, voice: voice)
            warmWalls.append(wall)
            peakThermal = Self.maxThermal(peakThermal, ProcessInfo.processInfo.thermalState)
        }
        let warmMedian = Self.median(warmWalls)
        let audioSec = coldAudio  // audio length is deterministic across runs

        let coldRTF = coldWall / max(audioSec, 1e-9)
        let warmRTF = warmMedian / max(audioSec, 1e-9)

        // MLComputePlan per submodel. Available iOS 17.4+; meaningful only
        // on device (simulator reports CPU for everything).
        let computePlan: [ComputePlanReport]
        if #available(iOS 17.4, macOS 14.4, *) {
            computePlan = await ComputePlanInspector.inspectAll(
                artifactsBundle: artifactsDir, computeUnits: mode
            )
        } else {
            computePlan = []
        }

        let throttled = peakThermal.rawValue > ProcessInfo.ThermalState.nominal.rawValue
        let verdict = Self.verdict(
            warmRTF: warmRTF, computePlan: computePlan
        )

        return BenchmarkRow(
            mode: label,
            audioSeconds: audioSec,
            coldWallSeconds: coldWall,
            warmWallSecondsMedian: warmMedian,
            coldRTF: coldRTF,
            warmRTF: warmRTF,
            peakThermalState: Self.thermalString(peakThermal),
            throttled: throttled,
            computePlan: computePlan,
            verdict: verdict
        )
    }

    // MARK: - Generation helper

    /// Returns (wall seconds, audio seconds).
    private static func timedGenerate(
        tts: PocketTTS, voice: VoiceHandle
    ) async throws -> (Double, Double) {
        let t0 = Date()
        var samples = 0
        let stream = await tts.generate(
            text: canonicalPrompt, voice: voice
        )
        for try await pcm in stream {
            samples += pcm.count / 2  // int16 LE
        }
        let elapsed = Date().timeIntervalSince(t0)
        let audioSec = Double(samples) / Double(PocketTTSArch.sampleRate)
        return (elapsed, audioSec)
    }

    // MARK: - Utility

    private static func median(_ xs: [Double]) -> Double {
        guard !xs.isEmpty else { return Double.nan }
        let sorted = xs.sorted()
        if sorted.count % 2 == 1 {
            return sorted[sorted.count / 2]
        }
        let a = sorted[sorted.count / 2 - 1]
        let b = sorted[sorted.count / 2]
        return (a + b) * 0.5
    }

    private static func maxThermal(
        _ a: ProcessInfo.ThermalState, _ b: ProcessInfo.ThermalState
    ) -> ProcessInfo.ThermalState {
        a.rawValue >= b.rawValue ? a : b
    }

    static func thermalString(_ t: ProcessInfo.ThermalState) -> String {
        switch t {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private static func modelIdentifier() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        let id = withUnsafePointer(to: &systemInfo.machine) {
            $0.withMemoryRebound(
                to: CChar.self,
                capacity: Int(_SYS_NAMELEN)
            ) { String(cString: $0) }
        }
        return id
    }

    /// Ship-gate decision tree per plan / task description.
    private static func verdict(
        warmRTF: Double, computePlan: [ComputePlanReport]
    ) -> String {
        let main = computePlan.first(where: { $0.modelName == "flow_lm_main" })
        let ane = main?.anePct ?? 0.0
        if warmRTF <= 0.5, ane >= 80.0 {
            return "GREEN"
        }
        if warmRTF <= 0.8 || (ane >= 50.0 && ane < 80.0) {
            return "YELLOW"
        }
        return "RED"
    }
}

// MARK: - JSON + markdown export

public extension BenchmarkReport {

    /// Pretty-printed JSON.
    func jsonData() throws -> Data {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try enc.encode(self)
    }

    /// Tagged markdown summary tuned for pasting into a handoff.
    func markdownSummary() -> String {
        var out = ""
        out += "# PocketTTS CoreML — on-device RTF report\n\n"
        out += "- **Device model id:** `\(deviceModel)`\n"
        out += "- **System version:** \(systemVersion)\n"
        out += "- **Timestamp:** \(timestamp)\n"
        out += "- **Prompt:** \"\(prompt)\"\n"
        out += "- **Voice:** \(voiceName)\n\n"
        out += "## RTF by compute-units config\n\n"
        out += "| mode | audio s | cold s | warm s (median) | cold RTF | warm RTF | peak thermal | verdict |\n"
        out += "|---|---|---|---|---|---|---|---|\n"
        for r in rows {
            out += "| \(r.mode) "
            out += "| \(String(format: "%.3f", r.audioSeconds)) "
            out += "| \(String(format: "%.3f", r.coldWallSeconds)) "
            out += "| \(String(format: "%.3f", r.warmWallSecondsMedian)) "
            out += "| \(String(format: "%.3f", r.coldRTF)) "
            out += "| \(String(format: "%.3f", r.warmRTF)) "
            out += "| \(r.peakThermalState)\(r.throttled ? " (throttled)" : "") "
            out += "| \(r.verdict) |\n"
        }
        out += "\n## ANE residency (by op count)\n\n"
        out += "| mode | submodel | ops | ANE% | GPU% | CPU% |\n"
        out += "|---|---|---|---|---|---|\n"
        for r in rows {
            for p in r.computePlan {
                out += "| \(r.mode) | \(p.modelName) | \(p.opCount) "
                out += "| \(String(format: "%.1f", p.anePct)) "
                out += "| \(String(format: "%.1f", p.gpuPct)) "
                out += "| \(String(format: "%.1f", p.cpuPct)) |\n"
            }
        }
        out += "\n## Notes\n\n\(notes)\n"
        return out
    }
}
