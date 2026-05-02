# PocketTTSCoreML — Swift runtime for the English PocketTTS CoreML port

This Swift Package loads the five `.mlpackage` bundles in
`Artifacts/en_alba_fp16/` and produces 24 kHz mono int16 PCM, streaming
from Apple Neural Engine / GPU / CPU via CoreML.

**Status:** Phase 4A (macOS 14+). iOS 18+ is declared in `Package.swift`
so Phase 4B can enable it without restructuring the manifest.

## Requirements

- macOS 14 or newer.
- Xcode 16 / Swift toolchain ≥ 6.0 (the package pins
  `swift-tools-version:6.0` so it can declare `.iOS(.v18)`).
- The 5 `.mlpackage` bundles in `Artifacts/en_alba_fp16/` plus the
  sidecar `mimi_decoder.state_layout.json`. Regenerate via
  `python -m pockettts_coreml.convert --all`.
- A SentencePiece tokenizer model (`tokenizer.model` from the
  `kyutai/pocket-tts` HF repo).
- A voice `.safetensors` bundle — either pre-exported by
  `pocket-tts export_voice` (voice-only) or **pre-prefilled** (see below).

## Build

From inside this directory:

```bash
swift build                # debug
swift build -c release     # release
swift test                 # runs all tests; uses fixtures under ../pockettts_coreml/oracle/fixtures/
```

## Phase 4A limitation — text prefill runs in Python

The currently-shipped `flow_lm_main.mlpackage` is AR-only: it takes a
single-step `sequence: fp16[1,1,32]` and has **no** `text_embeddings`
input. Text prefill in the Python reference driver
(`pockettts_coreml/e2e/generator.py`) is performed by the *reference
PyTorch* `StreamingTransformer`, not CoreML. Porting that prefill into
Swift means writing a 6-layer transformer in `MLMultiArray` ops by hand
(~800 lines) or adding a dedicated `flow_lm_prefill.mlpackage` to the
Phase-3 conversion pipeline.

Neither is in Phase 4A scope. Until then, the Swift side loads a
"pre-prefilled" voice+text bundle produced by a Python helper:

```bash
python -m pockettts_coreml.e2e.export_full_prefill \
    --voice pockettts_coreml/oracle/fixtures/english_alba_seed42/alba.safetensors \
    --prompt "Pocket TTS is a lightweight text-to-speech model." \
    --out   pockettts_coreml/oracle/fixtures/english_alba_seed42/alba_prefilled.safetensors \
    --seed 42
```

The resulting bundle contains:
- `flow_kv_rank5`: `fp32[12, 1, 256, 16, 64]` — rank-5 KV cache after
  voice + text prefill.
- `flow_offset`: `int64[1]` — write position after prefill.
- `bos_emb`: `fp32[32]` — latent for AR step 0.
- `prompt_utf8`: the prompt bytes (debug only).
- `noise_seq`: `fp32[MAX_STEPS, 32]` — precomputed per-step noise
  matching the reference's RNG trajectory (so Swift generation is
  bit-repro of Python output).

`PocketTTS.loadVoice` auto-detects voice-only vs pre-prefilled bundles.
See `Orchestrator.swift` for the full design rationale.

## Public API

```swift
import PocketTTSCoreML

let tts = try await PocketTTS(
    artifactsBundle: URL(fileURLWithPath: "Artifacts/en_alba_fp16"),
    tokenizerPath:   URL(fileURLWithPath: "tokenizer.model"),
    computeUnits:    .cpuAndNeuralEngine
)
let voice = try await tts.loadVoice(
    from: URL(fileURLWithPath: "alba_prefilled.safetensors")
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

Voice cloning (returns a voice-only handle; run the Python helper to
finish the prefill in Phase 4A):

```swift
let cloned = try await tts.cloneVoice(from: URL(fileURLWithPath: "sample.wav"))
```

## CLI

```bash
swift build -c release
./.build/release/pocket-tts-cli --help
./.build/release/pocket-tts-cli tokenize --tokenizer tokenizer.model --text "Hello world"
./.build/release/pocket-tts-cli generate  \
    --artifacts Artifacts/en_alba_fp16    \
    --tokenizer tokenizer.model           \
    --voice    alba_prefilled.safetensors \
    --text "Pocket TTS is a lightweight text-to-speech model." \
    --out  out.wav
./.build/release/pocket-tts-cli benchmark \
    --artifacts Artifacts/en_alba_fp16    \
    --tokenizer tokenizer.model           \
    --voice    alba_prefilled.safetensors \
    --iterations 5
./.build/release/pocket-tts-cli clone     \
    --artifacts Artifacts/en_alba_fp16    \
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
- `EndToEndTests` — canonical prompt generation, PSNR ≥ 15 dB overall
  and ≥ 25 dB best-early-frame (matches the Python e2e gate).
- `BenchmarkTests` — RTF ≤ 0.40 on Mac (currently ~**0.09** with
  `.cpuAndNeuralEngine`).

## Architecture

| Module                       | Role |
|------------------------------|------|
| `PocketTTS.swift`            | public `actor`; loads models, exposes `generate` |
| `Tokenizer.swift`            | Swift wrapper around `SentencePiece.xcframework` via a thin C bridge |
| `TextChunker.swift`          | port of `pocket_tts.models.tts_model.split_into_best_sentences` |
| `VoiceLoader.swift`          | reads voice-only + prefilled `.safetensors` bundles |
| `VoiceCloner.swift`          | runs `mimi_encoder.mlpackage` on a user waveform |
| `Orchestrator.swift`         | AR-loop producer; fp16 flow_lm + fp32 mimi_decoder |
| `KVCacheBuffers.swift`       | FlowLM rank-5 KV cache constants |
| `MimiStateBuffer.swift`      | packed fp32 state blob + ping-pong buffers; reads `mimi_decoder.state_layout.json` |
| `RoPECache.swift`            | cos/sin tables precomputed once per AR direction |
| `NoiseSource.swift`          | seedable noise provider with precomputed injection |
| `Masks.swift`                | attention / offset / scatter mask builders (fp32 + fp16) |
| `Float16Ops.swift`           | vImage-backed fp32↔fp16 bulk conversion |
| `AudioStream.swift`          | fp32 → int16 PCM via `vDSP_vfixr16` + minimal WAV writer |
| `SafetensorsReader.swift`    | minimal reader + writer (no external dependency) |
| `CSentencePieceBridge/`      | extern-C bridge over `sentencepiece::SentencePieceProcessor` |

## Data types at a glance

| Model              | I/O dtype |
|--------------------|-----------|
| `text_conditioner` | int32 in, fp16 out |
| `flow_lm_main`     | fp16 end-to-end |
| `flow_lm_flow`     | fp16 end-to-end |
| `mimi_encoder`     | fp16 end-to-end |
| `mimi_decoder`     | **fp32** end-to-end (converted at fp32 compute precision; see `docs/phase2_3_notes.md`) |

## Known gaps (intentional for Phase 4A)

1. **No in-Swift text prefill.** See limitation above.
2. **`cloneVoice`** returns latents, not a runnable handle — the prefill
   step (`flow_lm_main` prefill over the latents) needs Python too.
3. **No MLComputePlan / ANE residency** inspection. Phase 4B.
4. **No iOS build.** Platforms list declares iOS 18+ but no
   `#if os(iOS)` code paths exist yet (there's nothing iOS-specific to
   gate in 4A). AVAudioSession, background-mode entitlements, etc. come
   in Phase 4B.
5. **Tokenizer is not yet plugged into `generate`** at runtime — the
   pre-prefilled bundle encodes the text choice at export time. The
   tokenizer is exposed via `PocketTTS.tokenize(_:)` for tests and for
   Phase 4B's in-Swift prefill.

## Performance (reference numbers)

Measured on an Apple Silicon Mac (`.cpuAndNeuralEngine`, `swift run`
release), 3-iteration best-of:

| Stage            | RTF  |
|------------------|------|
| Python reference | 0.30 |
| Swift (this)     | **0.09** |

The Swift RTF is better than Python's because the Python driver
re-seeds / re-imports the reference PyTorch each call; the Swift side
only touches CoreML once per AR step.
