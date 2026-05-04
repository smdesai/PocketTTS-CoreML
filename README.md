# PocketTTS-CoreML

CoreML port of Kyutai's [PocketTTS](https://github.com/kyutai-labs/pocket-tts)
for Apple platforms. The project converts the PyTorch reference model into
CoreML bundles, provides a Swift runtime for streaming 24 kHz PCM, and includes
tools for validating CoreML output against the Python reference.

See [CONVERSION.md](CONVERSION.md) for the full conversion record, numerical
validation notes, and debugging history.

## Status

English, Spanish, German, Italian, Portuguese, and French run on device through
the Swift runtime. Voice cloning from a user `.wav` is wired end-to-end in Swift
via `PocketTTS.cloneVoice(from:)` and the CLI `clone` command.

The 6-layer languages (English, Spanish, German, Italian, Portuguese) share the
same 1024d architecture and produce roughly 356 MB per-language `.mlmodelc`
bundles. French ships as Kyutai's 24-layer `french_24l` variant, with a much
larger bundle and roughly 4x the transformer compute.

Measured on iPhone 17 Pro / A19 Pro for English:

- Warm generation RTF: about `0.09` (about `7.4x` realtime).
- Speaker similarity vs Python reference: `0.98`.
- `flow_lm_main` ANE residency: about `47%`.
- First cold install startup: about `60-80 s` while CoreML prepares ANE programs.
- Subsequent launches: about `3 s` with warm CoreML program cache.

## What This Repository Contains

- Python conversion scripts under `pockettts_coreml/convert/`.
- Trace-friendly patches under `pockettts_coreml/patches/`.
- Python parity/oracle tooling under `pockettts_coreml/oracle/` and `tests/`.
- Swift package runtime under `PocketTTSCoreML/`.
- Demo and benchmark app scaffolding under `PocketTTSCoreML/Examples/`.

The vendored Kyutai reference lives under `pockettts_coreml/reference/` and is
not modified directly. Local generated artifacts and downloaded voices are
gitignored.

## Requirements

- macOS with Xcode 16 / Swift 6.
- Python 3.11.
- `uv` or an equivalent virtual environment.
- `torch>=2.5,<2.8`.
- `coremltools==8.1`.
- Hugging Face access to gated `kyutai/pocket-tts`.

Do not bump to `coremltools` 9.x for the current pipeline; it raises the
deployment target to iOS 26.

## Setup

Accept the Kyutai license on Hugging Face, then authenticate:

```bash
hf auth login
```

Install Python dependencies:

```bash
uv sync
```

Download voice embeddings and tokenizers:

```bash
tools/download_voices.sh all
```

You can download a subset:

```bash
tools/download_voices.sh english spanish
```

This creates gitignored directories such as `voices_english/`,
`voices_spanish/`, and `voices_french/`.

## Convert Models

Convert a 6-layer language such as English:

```bash
uv run python -m pockettts_coreml.convert \
    --all \
    --language english \
    --out Artifacts/en_fp16
```

Precompile the models for app shipping:

```bash
mkdir -p Artifacts/en_fp16_mlmodelc
for model in text_conditioner flow_lm_main flow_lm_prefill flow_lm_flow \
             mimi_encoder mimi_decoder; do
    xcrun coremlcompiler compile \
        Artifacts/en_fp16/${model}.mlpackage \
        Artifacts/en_fp16_mlmodelc/
done

cp Artifacts/en_fp16/mimi_decoder.state_layout.json \
   Artifacts/en_fp16/flow_lm_bos_emb.safetensors \
   Artifacts/en_fp16/speaker_proj.safetensors \
   Artifacts/en_fp16_mlmodelc/
```

For other 6-layer languages, change `--language` and the output directory:

```bash
uv run python -m pockettts_coreml.convert \
    --all \
    --language spanish \
    --out Artifacts/es_fp16
```

French uses the 24-layer reference config:

```bash
uv run python -m pockettts_coreml.convert \
    --all \
    --language french_24l \
    --out Artifacts/fr_fp16
```

French is much larger. If the 24-layer `flow_lm_main` or
`flow_lm_prefill` models exceed ANE compile limits, use 6-bit
palettization for those models:

```bash
uv run python -m pockettts_coreml.convert.convert_flow_lm_main \
    --language french_24l \
    --palettize-bits 6 \
    --out Artifacts/fr_fp16_palette6/flow_lm_main.mlpackage

uv run python -m pockettts_coreml.convert.convert_flow_lm_prefill \
    --language french_24l \
    --palettize-bits 6 \
    --out Artifacts/fr_fp16_palette6/flow_lm_prefill.mlpackage
```

The 6-layer languages fit ANE as plain fp16. Avoid palettizing them unless
you specifically need the footprint reduction.

## Swift Runtime

Build the Swift package:

```bash
cd PocketTTSCoreML
swift build
swift test
```

Generate audio with the CLI:

```bash
swift run -c release pocket-tts-cli generate \
    --artifacts ../Artifacts/en_fp16_mlmodelc \
    --tokenizer ../voices_english/tokenizer.model \
    --voice ../voices_english/alba.safetensors \
    --text "Pocket TTS is a lightweight text-to-speech model." \
    --out out.wav
```

Benchmark:

```bash
swift run -c release pocket-tts-cli benchmark \
    --artifacts ../Artifacts/en_fp16_mlmodelc \
    --tokenizer ../voices_english/tokenizer.model \
    --voice ../voices_english/alba.safetensors \
    --iterations 5
```

Clone a voice from reference audio:

```bash
swift run -c release pocket-tts-cli clone \
    --artifacts ../Artifacts/en_fp16_mlmodelc \
    --tokenizer ../voices_english/tokenizer.model \
    --audio input.wav \
    --out cloned.safetensors \
    --sample-text "This is my cloned voice." \
    --sample-out cloned.wav
```

Minimal Swift usage:

```swift
import Foundation
import PocketTTSCoreML

let tts = try await PocketTTS(
    artifactsBundle: URL(fileURLWithPath: "Artifacts/en_fp16_mlmodelc"),
    tokenizerPath: URL(fileURLWithPath: "voices_english/tokenizer.model"),
    computeUnits: .cpuAndNeuralEngine
)

let voice = try await tts.loadVoice(
    from: URL(fileURLWithPath: "voices_english/alba.safetensors")
)

var pcm = Data()
let stream = await tts.generate(
    text: "Pocket TTS is a lightweight text-to-speech model.",
    voice: voice
)

for try await frame in stream {
    pcm.append(frame)
}

try AudioStream.writeWAV(pcm, to: URL(fileURLWithPath: "out.wav"))
```

## Runtime Architecture

PocketTTS is a continuous audio language model. It autoregressively predicts
32-dimensional continuous latents at 12.5 Hz, then decodes each latent into an
80 ms, 1920-sample, 24 kHz PCM frame.

Runtime path:

```text
text -> tokenizer -> text_conditioner -> flow_lm_prefill
voice.safetensors ---------------------> FlowLM KV cache

FlowLM KV + previous latent
    -> flow_lm_main -> ctx + eos_logit
    -> flow_lm_flow -> next latent
    -> mimi_decoder -> PCM frame
```

The Swift runtime loads two voice formats:

- Voice-only files from Kyutai-style `*.safetensors` voice embeddings. Swift
  runs text prefill for each prompt.
- Pre-prefilled files where voice and text are already baked into a KV cache.
  These are useful for cached canned phrases.

Voice cloning runs:

1. `mimi_encoder`: 4 seconds of 24 kHz mono audio to Mimi latents.
2. Accelerate `cblas_sgemm`: speaker projection from 32d latents to 1024d
   FlowLM conditioning.
3. `flow_lm_prefill`: writes the cloned voice KV cache.

`mimi_encoder` is intentionally loaded with `MLComputeUnits.cpuOnly`. On iOS,
loading it with ANE enabled can hang inside CoreML's ANE compiler because the
encoder contains unsupported SEANet transposed-convolution patterns. It runs
once per clone, not per generated audio frame.

## Artifacts

Each language bundle should contain:

- `text_conditioner.mlmodelc`
- `flow_lm_prefill.mlmodelc`
- `flow_lm_main.mlmodelc`
- `flow_lm_flow.mlmodelc`
- `mimi_encoder.mlmodelc`
- `mimi_decoder.mlmodelc`
- `flow_lm_bos_emb.safetensors`
- `speaker_proj.safetensors`
- `mimi_decoder.state_layout.json`

Development builds can load `.mlpackage` bundles, but iOS apps should ship
precompiled `.mlmodelc` directories. App bundles are read-only, so runtime
compilation cannot be cached back into the bundle.

## Validation

Regenerate Python golden fixtures:

```bash
uv run python -m pockettts_coreml.oracle.dump_golden \
    --voice pockettts_coreml/oracle/fixtures/english_alba_seed42/alba.safetensors
```

Run Python tests:

```bash
POCKETTTS_ORACLE_READY=1 .venv/bin/python -m pytest tests/
```

Run Swift tests:

```bash
cd PocketTTSCoreML
swift test
```

Many Swift tests skip when local model, tokenizer, or voice fixtures are not
present. The safetensors round-trip test does not require full model artifacts.

## Demo App

Stage generated resources into the SwiftUI demo app:

```bash
cd PocketTTSCoreML/Examples/DemoApp
./prepare_resources.sh
xcodegen generate
open DemoApp.xcodeproj
```

`prepare_resources.sh` copies language resources into the app bundle. Add new
languages to the source map near the top of that script.

## Directory Layout

```text
Artifacts/                         generated CoreML bundles, gitignored
voices_<language>/                 downloaded voices and tokenizers, gitignored
pockettts_coreml/reference/         Kyutai reference subtree
pockettts_coreml/patches/           trace-friendly PyTorch patches
pockettts_coreml/convert/           conversion scripts
pockettts_coreml/oracle/            golden fixture tooling
PocketTTSCoreML/                    Swift package and CLI
PocketTTSCoreML/Examples/           demo and benchmark app projects
docs/                               design notes and investigations
tests/                              Python parity tests
```

## Important Notes

- `flow_lm_main` and `flow_lm_prefill` pin softmax and layer normalization to
  fp32 compute after conversion. This fixes fp16 amplitude drift while keeping
  weights fp16.
- The conversion pipeline asserts zero `aten::Int` in traced graphs. Tensor to
  scalar integer conversions usually break CoreML conversion.
- fp16 attention masks use additive values `0.0` and `-65504.0`. Avoid boolean
  masks and `-inf` in fp16 softmax paths.
- CoreML program cache is tied to app identity. Rebuilds, clean installs, and
  app deletion can re-trigger long first-launch model preparation.
- Long text should be chunked at sentence boundaries. The FlowLM cache capacity
  is finite, and the Swift runtime stops cleanly if the cache is exhausted.
