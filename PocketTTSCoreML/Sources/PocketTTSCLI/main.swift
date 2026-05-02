import Foundation
import CoreML
import PocketTTSCoreML

@main
struct PocketTTSCLI {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            printUsage()
            exit(2)
        }
        do {
            switch args[1] {
            case "generate":  try await runGenerate(Array(args.dropFirst(2)))
            case "benchmark": try await runBenchmark(Array(args.dropFirst(2)))
            case "clone":     try await runClone(Array(args.dropFirst(2)))
            case "tokenize":  try runTokenize(Array(args.dropFirst(2)))
            case "--help", "-h", "help":
                printUsage()
            default:
                printUsage()
                exit(2)
            }
        } catch {
            FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
            exit(1)
        }
    }

    static func printUsage() {
        let text = """
        pocket-tts-cli  —  PocketTTS CoreML (Phase 4A) CLI

        Subcommands:
          generate  --artifacts DIR --tokenizer MODEL --voice VOICE.safetensors
                    --text TEXT --out OUT.wav
          benchmark --artifacts DIR --tokenizer MODEL --voice VOICE.safetensors
                    [--iterations N]
          clone     --artifacts DIR --audio INPUT.wav --out CLONED.safetensors
          tokenize  --tokenizer MODEL --text TEXT

        Notes:
          * `--voice` for `generate` / `benchmark` must be a PRE-PREFILLED
            safetensors produced by:
              python -m pockettts_coreml.e2e.export_full_prefill \\
                  --voice alba.safetensors --prompt "..." \\
                  --out alba_prefilled.safetensors

            Phase 4A cannot run the flow_lm_main prefill in Swift (see
            the package README for details).
          * Defaults: --artifacts=./Artifacts/en_alba_fp16
                      --tokenizer=./tokenizer.model
        """
        print(text)
    }

    // MARK: - Argument parsing

    struct Args {
        var map: [String: String] = [:]
        func get(_ key: String, default def: String? = nil) throws -> String {
            if let v = map[key] { return v }
            if let d = def { return d }
            throw CLIError.missingArg(key)
        }
        func maybe(_ key: String) -> String? { map[key] }
    }

    static func parseArgs(_ argv: [String]) -> Args {
        var a = Args()
        var i = 0
        while i < argv.count {
            let token = argv[i]
            if token.hasPrefix("--") {
                if i + 1 < argv.count && !argv[i+1].hasPrefix("--") {
                    a.map[token] = argv[i+1]
                    i += 2
                } else {
                    a.map[token] = "1"
                    i += 1
                }
            } else {
                i += 1
            }
        }
        return a
    }

    // MARK: - Commands

    static func runGenerate(_ argv: [String]) async throws {
        let a = parseArgs(argv)
        let artifactsURL = URL(fileURLWithPath: try a.get("--artifacts", default: "./Artifacts/en_alba_fp16"))
        let tokenizerURL = URL(fileURLWithPath: try a.get("--tokenizer", default: "./tokenizer.model"))
        let voiceURL = URL(fileURLWithPath: try a.get("--voice"))
        let outURL = URL(fileURLWithPath: try a.get("--out"))
        let text = try a.get("--text")

        fputs("Loading models from \(artifactsURL.path)...\n", stderr)
        let tts = try await PocketTTS(
            artifactsBundle: artifactsURL,
            tokenizerPath: tokenizerURL,
            computeUnits: .cpuAndNeuralEngine
        )
        fputs("Loading voice from \(voiceURL.lastPathComponent)...\n", stderr)
        let voice = try await tts.loadVoice(from: voiceURL)

        var pcm = Data()
        let startWall = Date()
        let stream = await tts.generate(text: text, voice: voice)
        for try await frame in stream {
            pcm.append(frame)
        }
        let elapsed = Date().timeIntervalSince(startWall)
        let audioSeconds = Double(pcm.count / 2) / Double(PocketTTSArch.sampleRate)
        let rtf = elapsed / max(audioSeconds, 1e-9)

        try AudioStream.writeWAV(pcm, to: outURL)
        fputs(String(format: "Wrote %@ | audio=%.3fs wall=%.3fs RTF=%.3f\n",
                     outURL.lastPathComponent, audioSeconds, elapsed, rtf), stderr)
    }

    static func runBenchmark(_ argv: [String]) async throws {
        let a = parseArgs(argv)
        let artifactsURL = URL(fileURLWithPath: try a.get("--artifacts", default: "./Artifacts/en_alba_fp16"))
        let tokenizerURL = URL(fileURLWithPath: try a.get("--tokenizer", default: "./tokenizer.model"))
        let voiceURL = URL(fileURLWithPath: try a.get("--voice"))
        let iterations = Int(a.maybe("--iterations") ?? "3") ?? 3

        let tts = try await PocketTTS(
            artifactsBundle: artifactsURL, tokenizerPath: tokenizerURL,
            computeUnits: .cpuAndNeuralEngine
        )
        let voice = try await tts.loadVoice(from: voiceURL)
        await tts.warmup()

        var rtfs: [Double] = []
        for i in 0..<iterations {
            var samples = 0
            let t0 = Date()
            let stream = await tts.generate(text: "Pocket TTS is a lightweight text-to-speech model.", voice: voice)
            for try await frame in stream {
                samples += frame.count / 2
            }
            let elapsed = Date().timeIntervalSince(t0)
            let audioSec = Double(samples) / Double(PocketTTSArch.sampleRate)
            let rtf = elapsed / max(audioSec, 1e-9)
            rtfs.append(rtf)
            print(String(format: "iter %d: audio=%.3fs wall=%.3fs RTF=%.3f",
                         i, audioSec, elapsed, rtf))
        }
        let best = rtfs.min() ?? 0
        let mean = rtfs.reduce(0, +) / Double(max(rtfs.count, 1))
        print(String(format: "best RTF=%.3f | mean RTF=%.3f | iterations=%d | compute=cpuAndNeuralEngine",
                     best, mean, iterations))
    }

    static func runClone(_ argv: [String]) async throws {
        let a = parseArgs(argv)
        let artifactsURL = URL(fileURLWithPath: try a.get("--artifacts", default: "./Artifacts/en_alba_fp16"))
        let tokenizerURL = URL(fileURLWithPath: try a.get("--tokenizer", default: "./tokenizer.model"))
        let audioURL = URL(fileURLWithPath: try a.get("--audio"))
        let outURL = URL(fileURLWithPath: try a.get("--out"))

        let tts = try await PocketTTS(
            artifactsBundle: artifactsURL, tokenizerPath: tokenizerURL,
            computeUnits: .cpuAndNeuralEngine
        )
        let handle = try await tts.cloneVoice(from: audioURL)
        // Voice-only handles cannot go through the standard saveVoice
        // (Phase 4A). Write a marker file with the latents so Python can
        // complete the prefill.
        switch handle.kind {
        case .prefilled:
            try await tts.saveVoice(handle, to: outURL)
            print("Wrote \(outURL.path)")
        case .voiceOnly(let layers, _):
            // Dump as a mimi_encoder latents bundle.
            let flat = layers.first?.cache ?? []
            let data = flat.withUnsafeBufferPointer { Data(buffer: $0) }
            try SafetensorsWriter.write([
                .init(name: "mimi_latents", shape: [flat.count], dtype: .F32, data: data)
            ], to: outURL)
            print("Wrote \(outURL.path) (mimi latents; run export_full_prefill.py to finish)")
        }
    }

    static func runTokenize(_ argv: [String]) throws {
        let a = parseArgs(argv)
        let tokenizerURL = URL(fileURLWithPath: try a.get("--tokenizer"))
        let text = try a.get("--text")
        let tok = try Tokenizer(modelURL: tokenizerURL)
        let ids = tok.encode(text)
        print(ids.map { String($0) }.joined(separator: ","))
    }
}

enum CLIError: Error, CustomStringConvertible {
    case missingArg(String)
    var description: String {
        switch self {
        case .missingArg(let a): return "missing required argument: \(a)"
        }
    }
}
