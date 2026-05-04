import CoreML
import Foundation
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
            case "generate": try await runGenerate(Array(args.dropFirst(2)))
            case "benchmark": try await runBenchmark(Array(args.dropFirst(2)))
            case "clone": try await runClone(Array(args.dropFirst(2)))
            case "tokenize": try runTokenize(Array(args.dropFirst(2)))
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
                        [--sample-text TEXT --sample-out WAV]
              tokenize  --tokenizer MODEL --text TEXT

            Notes:
              * `--voice` accepts EITHER:
                 - a raw voice safetensors (e.g. alba.safetensors from the
                   kyutai HF repo). The runtime tokenizes, runs
                   text_conditioner + flow_lm_prefill, and streams audio
                   without any Python helper.
                 - a pre-prefilled safetensors produced by
                   `python -m pockettts_coreml.e2e.export_full_prefill`
                   (back-compat path for apps that cache voice+prompt
                   pairs; the `--text` arg is ignored in that case).
              * Defaults: --artifacts=./Artifacts/en_fp16
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
                if i + 1 < argv.count && !argv[i + 1].hasPrefix("--") {
                    a.map[token] = argv[i + 1]
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
        let artifactsURL = URL(
            fileURLWithPath: try a.get("--artifacts", default: "./Artifacts/en_fp16"))
        let tokenizerURL = URL(
            fileURLWithPath: try a.get("--tokenizer", default: "./tokenizer.model"))
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
        fputs(
            String(
                format: "Wrote %@ | audio=%.3fs wall=%.3fs RTF=%.3f\n",
                outURL.lastPathComponent, audioSeconds, elapsed, rtf), stderr)
    }

    static func runBenchmark(_ argv: [String]) async throws {
        let a = parseArgs(argv)
        let artifactsURL = URL(
            fileURLWithPath: try a.get("--artifacts", default: "./Artifacts/en_fp16"))
        let tokenizerURL = URL(
            fileURLWithPath: try a.get("--tokenizer", default: "./tokenizer.model"))
        let voiceURL = URL(fileURLWithPath: try a.get("--voice"))
        let iterations = Int(a.maybe("--iterations") ?? "3") ?? 3

        let tts = try await PocketTTS(
            artifactsBundle: artifactsURL, tokenizerPath: tokenizerURL,
            computeUnits: .cpuAndNeuralEngine
        )
        let voice = try await tts.loadVoice(from: voiceURL)
        let kind: String
        switch voice.kind {
        case .prefilled: kind = "prefilled"
        case .voiceOnly: kind = "voiceOnly (runs in-Swift text prefill per call)"
        }
        fputs("Voice: \(kind)\n", stderr)
        await tts.warmup()

        var rtfs: [Double] = []
        for i in 0 ..< iterations {
            var samples = 0
            let t0 = Date()
            let stream = await tts.generate(
                text: "Pocket TTS is a lightweight text-to-speech model.", voice: voice)
            for try await frame in stream {
                samples += frame.count / 2
            }
            let elapsed = Date().timeIntervalSince(t0)
            let audioSec = Double(samples) / Double(PocketTTSArch.sampleRate)
            let rtf = elapsed / max(audioSec, 1e-9)
            rtfs.append(rtf)
            let label = (i == 0) ? "cold" : " warm"
            print(
                String(
                    format: "iter %d (%@): audio=%.3fs wall=%.3fs RTF=%.3f",
                    i, label as CVarArg, audioSec, elapsed, rtf))
        }
        let best = rtfs.min() ?? 0
        let mean = rtfs.reduce(0, +) / Double(max(rtfs.count, 1))
        print(
            String(
                format:
                    "best RTF=%.3f | mean RTF=%.3f | iterations=%d | compute=cpuAndNeuralEngine",
                best, mean, iterations))
    }

    static func runClone(_ argv: [String]) async throws {
        let a = parseArgs(argv)
        let artifactsURL = URL(
            fileURLWithPath: try a.get("--artifacts", default: "./Artifacts/en_fp16"))
        let tokenizerURL = URL(
            fileURLWithPath: try a.get("--tokenizer", default: "./tokenizer.model"))
        let audioURL = URL(fileURLWithPath: try a.get("--audio"))
        let outURL = URL(fileURLWithPath: try a.get("--out"))
        let sampleText = a.maybe("--sample-text")
        let sampleOutURL = a.maybe("--sample-out").map { URL(fileURLWithPath: $0) }

        fputs("Loading models from \(artifactsURL.path)...\n", stderr)
        let tts = try await PocketTTS(
            artifactsBundle: artifactsURL, tokenizerPath: tokenizerURL,
            computeUnits: .cpuAndNeuralEngine
        )
        fputs(
            "Cloning voice from \(audioURL.lastPathComponent) (stages: mimi_encoder + speaker_proj + flow_lm_prefill)...\n",
            stderr)
        let t0 = Date()
        let handle = try await tts.cloneVoice(from: audioURL)
        let cloneWall = Date().timeIntervalSince(t0)

        let voiceOff: Int
        switch handle.kind {
        case .voiceOnly(_, let off, _): voiceOff = off
        case .prefilled(_, let off, _, _, _): voiceOff = off
        }
        try await tts.saveVoice(handle, to: outURL)
        let size =
            (try? FileManager.default.attributesOfItem(atPath: outURL.path)[.size] as? Int) ?? 0
        fputs(
            String(
                format: "Cloned voice in %.2fs; wrote %@ (%.1f MB; voiceOffset=%d)\n",
                cloneWall, outURL.path, Double(size) / (1024 * 1024), voiceOff),
            stderr)

        // Optional: run a short sample generation with the cloned voice so
        // the caller can listen-check it.
        if let sampleText = sampleText {
            let outPath = sampleOutURL ?? URL(fileURLWithPath: "/tmp/cloned_test.wav")
            fputs("Generating sample audio with cloned voice -> \(outPath.path)...\n", stderr)
            var pcm = Data()
            let tg0 = Date()
            let stream = await tts.generate(text: sampleText, voice: handle)
            for try await frame in stream {
                pcm.append(frame)
            }
            let genWall = Date().timeIntervalSince(tg0)
            try AudioStream.writeWAV(pcm, to: outPath)
            let audioSec = Double(pcm.count / 2) / Double(PocketTTSArch.sampleRate)
            fputs(
                String(
                    format: "Wrote %@ (audio=%.2fs wall=%.2fs)\n",
                    outPath.lastPathComponent, audioSec, genWall), stderr)
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
