# PocketTTSCoreML — Swift runtime for the English PocketTTS CoreML port

This Swift Package loads the six `.mlpackage` bundles in
`Artifacts/en_fp16/` and produces 24 kHz mono int16 PCM, streaming
from Apple Neural Engine / GPU / CPU via CoreML.

**Status:** Phase 4B (macOS 14+) — text prefill runs in Swift; no
Python helper required for generation. iOS 18+ is declared in
`Package.swift` but not yet exercised.

## Requirements

- macOS 14 or newer.
- Xcode 16 / Swift toolchain ≥ 6.0 (the package pins
  `swift-tools-version:6.0` so it can declare `.iOS(.v18)`).
- The 6 `.mlpackage` bundles in `Artifacts/en_fp16/` plus the
  sidecar `mimi_decoder.state_layout.json` and
  `flow_lm_bos_emb.safetensors`. Regenerate via
  `python -m pockettts_coreml.convert --all`.
- A SentencePiece tokenizer model (`tokenizer.model` from the
  `kyutai/pocket-tts` HF repo).
- A voice `.safetensors` bundle — either raw (e.g. `alba.safetensors`
  from the kyutai HF repo) or a pre-prefilled one for cached voice+prompt
  pairs (see below).

## Build

`Package.swift` lives at the **repository root** (SwiftPM requires that for git
dependencies), so run SwiftPM commands from the repo root, not from this directory:

```bash
cd ..                      # repository root
swift build                # debug
swift build -c release     # release
swift test                 # runs all tests; uses fixtures under pockettts_coreml/oracle/fixtures/
```

## Use as a dependency

```swift
.package(url: "https://github.com/smdesai/PocketTTS-CoreML", from: "1.0.0")
// then in a target:
.product(name: "PocketTTSCoreML", package: "PocketTTS-CoreML")
```

In an XcodeGen `project.yml`:

```yaml
packages:
  PocketTTSCoreML:
    url: https://github.com/smdesai/PocketTTS-CoreML
    majorVersion: 1.0.0
```

## Public API

```swift
import PocketTTSCoreML

let tts = try await PocketTTS(
    artifactsBundle: URL(fileURLWithPath: "Artifacts/en_fp16"),
    tokenizerPath:   URL(fileURLWithPath: "tokenizer.model"),
    computeUnits:    .cpuAndNeuralEngine
)
// Load the raw voice file — Swift handles the text prefill.
let voice = try await tts.loadVoice(
    from: URL(fileURLWithPath: "alba.safetensors")
)
var pcm = Data()
for try await frame in await tts.generate(
    text: "Pocket TTS is a lightweight text-to-speech model.",
    voice: voice
) {
    pcm.append(frame)    // 1920 samples × 2 bytes = 3840 bytes per frame
}
try AudioStream.writeWAV(pcm, to: URL(fileURLWithPath: "out.wav"))
```

`loadVoice` auto-detects the bundle flavor:

- **Voice-only** (default for `alba.safetensors`-style files): the
  orchestrator runs `text_conditioner.mlpackage` +
  `flow_lm_prefill.mlpackage` on each `generate(text:voice:)` call to
  bake the text into the KV cache, then runs the AR loop. No Python
  helper required. The prefill itself is ~10–50 ms; the AR hot loop is
  identical to the pre-prefilled path.

- **Pre-prefilled** (optional back-compat): the KV cache already
  contains the text embedding for a specific prompt. Produced offline
  via `python -m pockettts_coreml.e2e.export_full_prefill`. Useful if
  an app caches canned (voice, prompt) pairs and wants to avoid
  per-call prefill entirely. The `text` argument is ignored in this
  case.

Voice cloning (runs `mimi_encoder.mlpackage` on a reference waveform;
the second stage that turns the latents into a FlowLM voice KV is still
offline — the CLI dumps latents that a follow-up step can consume):

```swift
let cloned = try await tts.cloneVoice(from: URL(fileURLWithPath: "sample.wav"))
```

## CLI

```bash
swift build -c release
./.build/release/pocket-tts-cli --help
./.build/release/pocket-tts-cli tokenize --tokenizer tokenizer.model --text "Hello world"
./.build/release/pocket-tts-cli generate  \
    --artifacts Artifacts/en_fp16    \
    --tokenizer tokenizer.model           \
    --voice    alba.safetensors           \
    --text "Pocket TTS is a lightweight text-to-speech model." \
    --out  out.wav
./.build/release/pocket-tts-cli benchmark \
    --artifacts Artifacts/en_fp16    \
    --tokenizer tokenizer.model           \
    --voice    alba.safetensors           \
    --iterations 5
./.build/release/pocket-tts-cli clone     \
    --artifacts Artifacts/en_fp16    \
    --tokenizer tokenizer.model           \
    --audio input.wav --out cloned.safetensors
```

## Tests

```bash
swift test
```

Covered:
- `TokenizerParityTests` — **BLOCKING**: Swift SP must match Python's
  `EncodeAsIds` byte-for-byte on 6 test strings including the canonical
  prompt.
- `VoiceLoadTests` — voice-only (`alba.safetensors`) + prefilled loader
  + safetensors roundtrip.
- `EndToEndTests` — canonical prompt generation from a pre-prefilled
  bundle, PSNR ≥ 15 dB overall and ≥ 25 dB best-early-frame.
- `PrefillIntegrationTest` — canonical prompt generation from the raw
  voice (in-Swift prefill), PSNR ≥ 15 dB.
- `BenchmarkTests` — RTF ≤ 0.40 on Mac (currently ~**0.09** with
  `.cpuAndNeuralEngine`, same with or without in-Swift prefill).

## Architecture

| Module                       | Role |
|------------------------------|------|
| `PocketTTS.swift`            | public `actor`; loads models, exposes `generate` |
| `Tokenizer.swift`            | Swift wrapper around `SentencePiece.xcframework` via a thin C bridge |
| `TextChunker.swift`          | port of `pocket_tts.models.tts_model.split_into_best_sentences` |
| `VoiceLoader.swift`          | reads voice-only + prefilled `.safetensors` bundles |
| `VoiceCloner.swift`          | runs `mimi_encoder.mlpackage` on a user waveform |
| `Orchestrator.swift`         | in-Swift text prefill + AR loop producer; fp16 flow_lm + fp32 mimi_decoder |
| `KVCacheBuffers.swift`       | FlowLM rank-5 KV cache constants |
| `MimiStateBuffer.swift`      | packed fp32 state blob + ping-pong buffers; reads `mimi_decoder.state_layout.json` |
| `RoPECache.swift`            | cos/sin tables precomputed once per AR direction |
| `NoiseSource.swift`          | seedable noise provider with precomputed injection |
| `Masks.swift`                | attention / offset / scatter mask builders (fp32 + fp16) |
| `Float16Ops.swift`           | vImage-backed fp32↔fp16 bulk conversion |
| `AudioStream.swift`          | fp32 → int16 PCM via `vDSP_vfixr16` + minimal WAV writer |
| `SafetensorsReader.swift`    | minimal reader + writer (no external dependency) |
| `Frameworks/SentencePiece.xcframework` | prebuilt SentencePiece with an extern-C `sentencepiece_*` API (`SentencePieceBridge.h`) |

## Data types at a glance

| Model              | I/O dtype |
|--------------------|-----------|
| `text_conditioner` | int32 in, fp16 out |
| `flow_lm_prefill`  | fp16 end-to-end (128-token prefill graph) |
| `flow_lm_main`     | fp16 end-to-end (AR step graph) |
| `flow_lm_flow`     | fp16 end-to-end |
| `mimi_encoder`     | fp16 end-to-end |
| `mimi_decoder`     | **fp32** end-to-end (converted at fp32 compute precision; see `docs/phase2_3_notes.md`) |

## Known gaps

1. **`cloneVoice`** runs only the first stage (Mimi encoder); the
   flow_lm pass over those latents to produce a usable voice KV still
   requires an offline step. Porting that is a follow-up to this cycle.
2. **No MLComputePlan / ANE residency** inspection.
3. **No iOS build.** Platforms list declares iOS 18+ but no
   `#if os(iOS)` code paths exist yet.

## Performance (reference numbers)

Measured on an Apple Silicon Mac (`.cpuAndNeuralEngine`, `swift run -c
release`), 3-iteration best-of, 3.6 s of audio per run:

| Configuration                              | RTF    |
|--------------------------------------------|--------|
| Python reference                           | 0.30   |
| Swift, pre-prefilled bundle                | 0.087  |
| Swift, raw voice (in-Swift text prefill)   | 0.087  |
| Swift, raw voice — first (cold) iteration  | ~0.10  |

The text prefill (128-token, single forward through the 6-layer
transformer) adds ~10–20 ms per utterance — negligible compared to the
~3 s of audio produced. Steady-state per-frame RTF is unchanged vs. the
pre-prefilled path because the AR loop executes the same
`flow_lm_main.mlpackage` graph either way.
