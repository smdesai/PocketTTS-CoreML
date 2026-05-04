# PocketTTS → CoreML conversion guide

End-to-end record of porting Kyutai's PocketTTS model to Apple CoreML,
targeting ANE+CPU on iPhone (A19 Pro primary) for background-capable
real-time TTS. Covers the production conversion pipeline, every
significant bug encountered, and lessons learned that would bite anyone
porting a similar model.

**Status:** English, Spanish, German, Italian, Portuguese, and French
all ship on device, with **voice cloning from a user `.wav`** wired
end-to-end in the Swift runtime and surfaced in the demo app's Voice
picker ("+ Clone new voice" → record or pick audio). The first five
languages use the identical 6-layer 1024d architecture and share
~356 MB per-language bundles. French ships as the 24-layer
`french_24l` variant (the only variant Kyutai published for French)
and has ~4× the transformer footprint: per-language bundle ~1.05 GB,
voice `.safetensors` ~25 MB each (vs ~7 MB for 6L), per-step compute
~4× slower. iPhone 17 Pro A19 Pro (measured on English): warm RTF
0.09 (7.44× realtime), speaker similarity 0.98 vs Python reference,
thermal `.nominal`, 46.9% ANE residency on `flow_lm_main`. First
cold-install startup on A19 Pro takes ~60–80 s (ANE program compile
for 5 mlmodelc bundles); subsequent launches are ~3 s (warm cache).
French device RTF should be measured separately; theoretical upper
bound is ~1/4 of the 6L RTFx.

---

## 1. Result

| Submodel            | Size (fp16) | Purpose                                       | ANE residency |
|---------------------|------------:|------------------------------------------------|--------------:|
| `text_conditioner`  |     7.8 MB  | SentencePiece tokens → 1024-d embeddings       |   14%         |
| `flow_lm_prefill`   |   126 MB    | Full-text prefill over voice+text tokens       |   44.5%       |
| `flow_lm_main`      |   144 MB    | Per-step AR transformer (6 layers, 1024d)      |   46.9%       |
| `flow_lm_flow`      |    19 MB    | Per-step flow head (SimpleMLPAdaLN, N=1)       |    0%         |
| `mimi_encoder`      |    20 MB    | Audio → latents (voice cloning path)           |   33.8%       |
| `mimi_decoder`      |    39 MB    | 32-d latent → 24 kHz PCM frame (SEANet+MHA)    |    0% (CPU)   |
| **Total per language** | **~356 MB** |                                              |               |

Numbers from `Artifacts/en_fp16_mlmodelc/` measured via
`MLComputePlan` on iPhone 17 Pro (A19 Pro). Spanish is architecturally
identical (6L/1024d) and matches these numbers.

### Per-language validation results

From `pockettts_coreml/oracle/verify_<lang>.py` (Python reference vs
CoreML fp16 with fp32-softmax+LayerNorm fix applied, gates: cosine ≥
0.93, overall PSNR ≥ 15 dB, best-early ≥ 25 dB, drift slope ≤ 0
target):

| Language   | Layers | Cosine | Overall PSNR | Best-early | Drift slope   | Bundle |
|------------|:------:|-------:|-------------:|-----------:|--------------:|-------:|
| English    |   6    | 0.98   |          —   |        —   | +0.03 %/frame | 356 MB |
| Spanish    |   6    | 0.98   |          —   |        —   |           —   | 356 MB |
| German     |   6    | 0.99   |       24.97  |     49.15  | +0.32 %/frame | 485 MB |
| Italian    |   6    | 0.96   |       16.55  |     52.16  | +0.54 %/frame | 485 MB |
| Portuguese |   6    | 0.96   |       18.22  |     42.35  | +1.77 %/frame | 485 MB |
| French     |  24    | 0.99   |       21.56  |     60.68  | +0.01 %/frame | 1.22 GB |

All 6 pass. French's drift slope is actually the lowest despite having
4× the layers — the deeper LayerNorm chain appears to stabilize fp16
accumulation once the softmax + LN ops are pinned to fp32.
Portuguese has the highest slope; if you ship long-form Portuguese
audio, revisit the drift fix for that language specifically.

---

## 2. Running the conversion (command reference)

### 2.1 One-time setup

```bash
# 1. Accept Kyutai license at https://huggingface.co/kyutai/pocket-tts
# 2. Log in (token cached in ~/.cache/huggingface/token)
huggingface-cli login

# 3. Install Python deps
uv sync   # or: pip install -e .
```

Python 3.11, `torch>=2.5,<2.8`, `coremltools==8.1`.
**Do not bump to coremltools 9.x** — it hoists the deployment target to
iOS 26, which blocks shipping on iOS 18.

### 2.2 Convert English

```bash
cd /Users/sdesai/Tools/AI/pocketTTS-CoreML

# 1. Convert all 6 .mlpackage bundles (~15 minutes on M-series)
.venv/bin/python -m pockettts_coreml.convert \
    --all \
    --language english \
    --out Artifacts/en_fp16

# 2. Pre-compile to .mlmodelc for iOS ship (halves artifact size,
#    eliminates per-launch compile)
mkdir -p Artifacts/en_fp16_mlmodelc
for m in text_conditioner flow_lm_main flow_lm_prefill flow_lm_flow \
         mimi_encoder mimi_decoder; do
    xcrun coremlcompiler compile \
        Artifacts/en_fp16/${m}.mlpackage \
        Artifacts/en_fp16_mlmodelc/
done

# 3. Copy sidecars (the bos_emb + state layout are produced by the
#    conversion scripts and must travel with the .mlmodelc bundles)
cp Artifacts/en_fp16/mimi_decoder.state_layout.json \
   Artifacts/en_fp16/flow_lm_bos_emb.safetensors \
   Artifacts/en_fp16/speaker_proj.safetensors \
   Artifacts/en_fp16_mlmodelc/
```

### 2.3 Convert another language (e.g. Spanish)

```bash
# Same flow, different language + output dir. Voice/tokenizer files
# for Spanish live at kyutai/pocket-tts/languages/spanish/.
.venv/bin/python -m pockettts_coreml.convert \
    --all --language spanish \
    --out Artifacts/es_fp16

mkdir -p Artifacts/es_fp16_mlmodelc
for m in text_conditioner flow_lm_main flow_lm_prefill flow_lm_flow \
         mimi_encoder mimi_decoder; do
    xcrun coremlcompiler compile \
        Artifacts/es_fp16/${m}.mlpackage \
        Artifacts/es_fp16_mlmodelc/
done
cp Artifacts/es_fp16/mimi_decoder.state_layout.json \
   Artifacts/es_fp16/flow_lm_bos_emb.safetensors \
   Artifacts/es_fp16/speaker_proj.safetensors \
   Artifacts/es_fp16_mlmodelc/
```

German / Italian / Portuguese all have 6-layer variants with identical
architecture to English — same pipeline, just change `--language`.
**French** only ships as the 24-layer variant (`french_24l` in the
reference config). Use the same `--all` pipeline (`--language
french_24l --out Artifacts/fr_fp16`); the conversion code has no
hardcoded layer count (everything is driven from the YAML config).
Expect:

- Conversion wall time ~30-45 min (vs ~15 min for 6L) — 4× as many
  softmax/layernorm ops means the fp32-softmax MIL pass walks 4× as
  many targets.
- `flow_lm_main.mlmodelc` ~550 MB (vs 144 MB for 6L).
- `flow_lm_prefill.mlmodelc` ~500 MB (vs 126 MB).
- Voice `.safetensors` ~25 MB each (vs ~7 MB) because the rank-5 KV
  is `[48, 1, 256, 16, 64]` instead of `[12, 1, 256, 16, 64]`.
- Total per-language bundle ~1.05 GB (vs 356 MB for 6L).
- Python-side verify per forward ~30 s (vs ~3 s for 6L on Mac).
- Swift/ANE device RTFx drops proportionally; measure on device.

**6-bit palettization for French (ANE fit).** The 24L `flow_lm_main` /
`flow_lm_prefill` exceed the Mac ANE compile budget at fp16 (
`ANECCompile() FAILED`, CPU fallback at 33 ms/step). Both
converters accept `--palettize-bits N` (typical: 6 or 8) which runs
`coremltools.optimize.coreml.palettize_weights` (k-means,
`per_grouped_channel`, `group_size=16`) immediately after the
fp32-softmax MIL pass — palettization touches weight storage while
the compute-precision pass touches activation dtypes, so the two are
orthogonal. 6-bit cuts `flow_lm_main.mlpackage` 577 MB → ~220 MB and
`flow_lm_prefill.mlpackage` 559 MB → ~210 MB (2.6× reduction); the
palettization step adds ~5-10 min per model on top of the base
convert wall time. Example:

```bash
.venv/bin/python -m pockettts_coreml.convert.convert_flow_lm_main \
    --language french_24l --palettize-bits 6 \
    --out Artifacts/fr_fp16_palette6/flow_lm_main.mlpackage

.venv/bin/python -m pockettts_coreml.convert.convert_flow_lm_prefill \
    --language french_24l --palettize-bits 6 \
    --out Artifacts/fr_fp16_palette6/flow_lm_prefill.mlpackage
```

The 6L languages (English, Spanish, German, Italian, Portuguese) fit
ANE fine as plain fp16 — do NOT palettize them unless you need the
footprint win, since palettization costs ~2-3 dB of PSNR.

**Swift runtime layer resolution:** `PocketTTSArch.flowLayers` in
`PocketTTSCoreML/Sources/PocketTTSCoreML/KVCacheBuffers.swift` is
`nonisolated(unsafe) var` (defaults to 6). `PocketTTS.init` calls
`configureFlowLayers(from: flow_lm_main)` immediately after loading
the model, inspecting the `kv_cache_in` input's `multiArrayConstraint`
to read `2*L` and set `flowLayers = L`. 6L variants land at 6, French
24L lands at 24. Must be called once per PocketTTS instance before
any KV allocation or voice load — the init ordering enforces this.
Single-writer-before-reads ordering makes the `unsafe` mutation
safe in practice.

### 2.4 Stage resources into the iOS demo app

```bash
cd PocketTTSCoreML/Examples/DemoApp
./prepare_resources.sh   # wipes Resources/Languages/ and copies everything
xcodegen generate        # regenerate Xcode project; scheme is declared in project.yml
open DemoApp.xcodeproj
```

`prepare_resources.sh` reads its per-language source map from the
top of the script; add entries there when adding a new language.

### 2.5 Validate on command line

```bash
# Full fixture regen (bitwise-reproducible; gates under seed 42)
.venv/bin/python -m pockettts_coreml.oracle.dump_golden \
    --voice pockettts_coreml/oracle/fixtures/english_alba_seed42/alba.safetensors

# Full Python test suite
POCKETTTS_ORACLE_READY=1 .venv/bin/python -m pytest tests/

# Swift package tests (Mac)
cd PocketTTSCoreML && swift test
```

---

## 3. Architecture + what each component does

PocketTTS is **CALM** (Continuous Audio Language Model): autoregressive
over 32-d continuous audio latents at 12.5 Hz → 24 kHz mono PCM.
Not codec-token based; the "flow head" (diffusion MLP) samples the next
latent from the transformer's context vector.

```
text ─▶ tokenizer ─▶ text_conditioner ─▶ flow_lm_prefill ┐
                                                         │ KV cache
voice.safetensors ─▶ flow_kv_rank5 ─────────────────────┘
                                    (seeded AR loop)
                                         │
                                         ▼
                        ┌─▶ flow_lm_main ─(ctx)─▶ flow_lm_flow ─(latent)─┐
                        │          │                                     │
                        │     eos_logit                                  │
                        │          │                                     │
                        │     threshold check                            │
                        │          │                                     │
                        └──────────┴──── prev_latent ◀────────────────────┤
                                                                         │
                                                                         ▼
                                                                mimi_decoder ─▶ 1920-sample PCM
                                                                     │
                                                                     ▼
                                                          streaming state blob (in/out)
```

**Per-AR-step budget (80 ms at 12.5 Hz):**
one `flow_lm_main` + one `flow_lm_flow` + one `mimi_decoder`.
On A19 Pro: ~10 ms total → RTF 0.09 → 11× realtime.

**Voice = pre-prefilled KV cache,** not an embedding vector. A
`.safetensors` voice file is loaded and becomes the `flow_lm_main` KV
at offset 0; text prefill writes at offset `voice_len`; AR loop starts
at `voice_len + text_len`.

---

## 4. Conversion pipeline internals

### 4.1 Module boundaries

The plan follows Kyutai's own 5-bundle ONNX split. We add a 6th
(`flow_lm_prefill`) so Swift can run text prefill natively without a
Python helper.

Each `pockettts_coreml/convert/convert_*.py`:

1. `pockettts_coreml.patches.build_patched_submodules(language=...)`
   instantiates the reference model with weight-loading delegated to
   the reference (`pocket_tts.models.tts_model.TTSModel.load_model`).
2. Wraps the submodel in a `*_patched` class from
   `pockettts_coreml/patches/` that eliminates all 12+ traceability
   landmines (§5 below).
3. `torch.jit.trace` with fixed example inputs.
4. **Asserts zero `aten::Int` in the traced graph** — this is the
   canary that CoreML conversion will succeed. Any `.item()`,
   `int(tensor)`, `tensor.shape[k]` as scalar → generates `aten::Int`,
   breaks CoreML.
5. `coremltools.convert(..., compute_precision=FLOAT16, compute_units=CPU_AND_NE, minimum_deployment_target=iOS18)`.
6. Post-conversion MIL pass for `flow_lm_main` + `flow_lm_prefill`:
   pin `softmax` and `layer_norm` to fp32 via coremltools
   `FP16ComputePrecision` with a custom `op_selector`. Weights stay
   fp16; only compute precision of those two op types is elevated.
   This is the drift fix — see §6.6.

### 4.2 Patched modules (`pockettts_coreml/patches/`)

| Patch file                  | Replaces                         | What it fixes                                                                 |
|----------------------------|----------------------------------|--------------------------------------------------------------------------------|
| `transformer_patched.py`   | `modules/transformer.py`         | mask-based KV writes (no `.item()`), manual SDPA, fp16 additive attn mask     |
| `rope_patched.py`          | `modules/rope.py`                | Pre-computed cos/sin tables as inputs (no in-graph `arange`/`exp`)            |
| `mlp_patched.py`           | `modules/mlp.py`                 | `nn.LayerNorm` swap (ANE-fused), canonical RMSNorm                             |
| `flow_lm_patched.py`       | `models/flow_lm.py` fragments    | Explicit BOS + noise inputs (no `isnan`/`normal_` in graph); splits main/flow  |
| `mimi_patched.py`          | `modules/conv.py`, `resample.py` | `pure_forward(x, *state_in) -> (y, *state_out)` wrappers for streaming convs   |
| `mimi_model_patched.py`    | `models/mimi.py` assembly        | Wires patched pieces into `PatchedMimiEncoder` / `PatchedMimiDecoder`          |

The vendored reference at `pockettts_coreml/reference/` is
**never modified**. All patches consume the reference via hooks or
explicit wrappers.

### 4.3 Sidecar artifacts

Four small files travel with the 6 `.mlpackage` / `.mlmodelc` bundles:

- **`flow_lm_bos_emb.safetensors`** — `fp32[32]` BOS vector for AR step 0.
  Per-language (the BOS is a learned parameter, not a constant).
- **`speaker_proj.safetensors`** — two per-language parameters for
  voice cloning stage 2:
    - `speaker_proj`: `fp32[1024, 32]` learned linear that projects
      Mimi latents (ldim=32) to the flow_lm transformer's d_model
      (1024). Applied in Swift via `cblas_sgemm` after `mimi_encoder`
      in the clone pipeline.
    - `bos_before_voice` (optional): `fp32[1, 1, 1024]` prefix vector
      prepended to the projected voice conditioning before
      `flow_lm_prefill` is invoked. Present on configs with
      `insert_bos_before_voice: true` (all 6 shipped languages).
  Export via `python -m pockettts_coreml.convert.export_speaker_proj`.
- **`mimi_decoder.state_layout.json`** — describes how to pack the
  `mimi_decoder` streaming state into a single fp16 blob. See
  `docs/mimi_state_layout.md`. 10 slots: 1 upsample_partial,
  1 transformer KV rank-5, 4 SEANet `previous`, 3 `partial`, and 1
  intra-resblock state per resblock × 3.
- **`flow_kv_rank5`** is a key inside the voice `.safetensors`
  (not a file) — the rank-5 `fp16[12, 1, 256, 16, 64]` packed KV cache
  (6 layers × 2 for K/V). Rank-5 exists because CoreML caps tensor
  rank at 5; the natural rank-6 `[6, 2, 1, 256, 16, 64]` had to be
  collapsed.

### 4.4 Voice cloning pipeline (Swift runtime)

The `pocket-tts-cli clone` command — and `PocketTTS.cloneVoice(from:)`
in the Swift library — runs the two-stage clone entirely on device:

1. **Stage 1: `mimi_encoder.mlpackage`** — reference 4-second waveform
   (resampled to 24 kHz mono, padded/cropped to exactly 96 000 samples)
   → `fp16[1, 32, 50]` Mimi latents.
2. **Stage 2a: speaker projection** — `F.linear(latents^T, speaker_proj)`
   → `fp16[1, 50, 1024]` conditioning, executed in Swift via
   Accelerate's `cblas_sgemm` (no ML graph). Optionally prepend the
   per-language `bos_before_voice` vector → `fp16[1, 51, 1024]`.
3. **Stage 2b: `flow_lm_prefill.mlpackage`** — conditioning is padded
   to `[1, 128, 1024]` and fed to the same prefill bundle text
   prefill uses (with `startOffset=0`). Writes the voice KV cache at
   slots `[0, 51)`.

The resulting `VoiceHandle.voiceOnly(flowKVRank5, voiceOffset=51)` is
then ready for `generate(text:voice:)`: the orchestrator runs its
normal text-prefill path over the voice KV (writing to slots
`[51, 51 + S_text)`) and then the AR loop.

**Implementation notes.**

- `mimi_encoder` output `MLMultiArray` uses padded/aligned strides
  (observed strides `[2048, 64, 1]` on a `[1, 32, 50]` tensor — the
  channel dim is padded from 50 to 64 for ANE alignment). Swift must
  gather the latents respecting `strides`, not assume tight
  row-major packing. `VoiceProjection.applySpeakerProjection` does
  this correctly; callers of `MLMultiArray.dataPointer` on any
  encoder output should follow the same pattern.
- The flow_lm_prefill bundle is reused unchanged: voice conditioning
  and text conditioning have the same shape
  (`fp16[1, 128, 1024]` after padding) and the prefill bundle has
  no special casing for which kind of conditioning it sees. Masks and
  RoPE are constructed in Swift per-call based on `startOffset`.
- **`mimi_encoder` must be loaded with `MLComputeUnits.cpuOnly`**.
  When loaded with `.cpuAndNeuralEngine`, iOS CoreML's ANE program
  compiler attempts to split the graph around the SEANet
  stride 6/5/4 inverse + depthwise `ConvTrUpsample1d` ops (which
  aren't ANE-supported) and **hangs indefinitely** inside
  `MLModel(contentsOf:)`. The encoder ships at ~70 ms/predict on CPU
  and runs once per clone (not per AR frame), so forcing CPU has no
  user-visible cost. `PocketTTS.ensureVoiceCloner` hard-codes
  `.cpuOnly` for this reason; don't "fix" it back to the runtime's
  computeUnits.
- **`VoiceCloner` is lazy + cached.** `PocketTTS` builds the cloner
  on the first `cloneVoice` call (paying the mimi_encoder init +
  CPU-path compile cost once) and keeps it alive for all subsequent
  clones in the same session. Preloading at `PocketTTS.init` instead
  would add ~5–10 s to app startup even for users who never clone.

---

## 5. Landmines closed during conversion

### 5.1 PyTorch → CoreML (the "zero `aten::Int`" rule)

These all emit `aten::Int` from TorchScript, which `ct.convert` rejects.
Every one was removed from the patched modules:

1. **`.item()` / `int(tensor)`** — anywhere. Most common in state-offset
   code. Fix: pass one-hot masks instead of scalar indices.
2. **`tensor.shape[k]`** used as an int inside an op — even when the
   shape is static at trace time. Fix: use `tensor.size(k)` → still int,
   same problem; real fix is to make shapes flow through tensors via
   `unflatten` / `reshape(..., -1)` idioms.
3. **`torch.where(isnan(x), bos, x)`** — we had this as a step-0
   convention. Fix: caller passes `bos` explicitly on step 0.
4. **In-graph `torch.arange(T)` / `torch.exp(...)`** — RoPE tables.
   Fix: precompute on host, pass as inputs.
5. **`scores.view(..., 3, H, D)` + `slice`** for QKV split — CoreML's
   QKV-fusion pass mis-applied RoPE to V. Fix: `torch.chunk(3, dim=-1)`.

### 5.2 Coremltools 8.1 specific issues

Each of these silently produced wrong output (not a build-time error).
Found via per-stage numerical parity tests or by listening to generated
audio.

6. **`F.scaled_dot_product_attention` with fp16 additive -65500 mask
   produces NaN** on the CoreML compiled graph (even when the eager
   PyTorch output is fine). Fix: manual `matmul / softmax / matmul`
   chain with explicit casts. Affects `mimi_decoder` and every patched
   MHA.
7. **`einsum` for KV scatter at fp16 NaNs.** Fix: replace with
   `matmul + unflatten`.
8. **Boolean attention masks produce NaN after softmax at fp16.**
   Fix: fp16 additive mask with values `0.0` (visible) and
   `-65500.0` (masked). Never `-inf` (becomes NaN), never boolean.
9. **`MLState` with dual-state variables fails on ANE.** (External
   research warning; we never tripped this because we chose explicit
   KV I/O from the start. If the KV-memcpy cost ever becomes the
   bottleneck, switch to `MLState` but pack all 12×K+V into a single
   rank-5 state tensor.)

### 5.3 ANE placement constraints

These submodels partially or fully fall back to CPU on device:

10. **`mimi_decoder` SEANet** — `ConvTranspose1d` with strides 6, 5, 4
    is not ANE-supported. Depthwise `ConvTrUpsample1d` (groups=512,
    stride=16) also CPU-only. Runs on CPU; 3 ms/frame on Mac, fine.
11. **Dilated convs** in SEANet — concern from external research, but
    verified `n_residual_layers=1` means only `dilation=1` fires in
    English. False alarm per `docs/research/reference_codebase_map.md`.

### 5.4 Stateful iOS constraints

12. **iOS app bundles are read-only.** The original Swift package
    tried to compile `.mlpackage` at runtime and cache `.mlmodelc` back
    into the bundle; the cache write silently failed, and every
    launch re-paid the 1-3 second ANE compile cost per model × 6
    models. Fix: pre-compile on Mac via `xcrun coremlcompiler compile`;
    ship `.mlmodelc` directly. This also **halves ship size**
    (713 MB → 356 MB) because `.mlmodelc` drops the MIL protobuf
    source.
13. **Xcode folder references don't follow symlinks reliably for
    bundled resources.** Physically `cp -R` the mlmodelc dirs into
    the app's Resources/. `prepare_resources.sh` does this.

---

## 6. Problems encountered & resolutions

### 6.1 The reference is PyTorch, not MLX

The reference repo lives at `/Users/sdesai/Tools/MLX/pocket-tts/` but
is **entirely PyTorch** (verified in its `pyproject.toml`). The path
name is misleading. No MLX→PyTorch bridge needed; we trace directly.

### 6.2 Voice cloning requires gated weights

The ungated HuggingFace checkpoint `kyutai/pocket-tts-without-voice-cloning`
sets `has_voice_cloning = False` (tts_model.py:206) and raises
`ValueError(VOICE_CLONING_UNSUPPORTED)` when a user `.wav` is passed.
We use the gated `kyutai/pocket-tts` checkpoint throughout. Contributors
need an HF license acceptance + token (`huggingface-cli login` works).

### 6.3 KV cache is a single-use handle

`PocketTTS.generate` mutates the `VoiceHandle`'s KV cache in place
(text prefill + AR writes go directly into the handle's buffer). A
cached handle can be used **once**. The first naive cache in
`TTSViewModel.voiceHandle(for:)` caused chunk 2+ to inherit chunk 1's
poisoned KV → `flowSCap=256` overflow → `continuation.finish()` with
no PCM → stream 1/3 stuck indicator.

**Fix:** reload the voice from disk per chunk. Cost is ~10 ms parsing
the safetensors — negligible next to generation.

### 6.4 Flow_lm_main was AR-only; needed a prefill variant

The initial Phase 3 conversion baked `flow_lm_main.mlpackage` without
`text_embeddings` as input, so it could only do per-AR-step forward
calls. The Python driver used reference PyTorch for text prefill.
This blocked the Swift package from running end-to-end on arbitrary
user text without a Python helper.

**Fix:** add `flow_lm_prefill.mlpackage` — same 6-layer transformer
weights, but input signature accepts `text_embeddings[1, 128, 1024]`
and runs the whole text in one forward pass. Swift now does prefill
natively.

### 6.5 Mimi transformer's `inverse(int32)` trace error

The reference `StreamingMultiheadAttention` hits `inverse(int32)` under
torch trace — some attention-mask arithmetic uses `1 / attn_mask`
where the mask is booleanish.

**Fix:** route the encoder + decoder transformers through the same
`PatchedStreamingMultiheadAttention` class used by FlowLM. Shared
implementation, no inverse.

### 6.6 fp16 amplitude decay on long generations (the big one)

Detected after Phase 4A shipped. On any single sentence longer than
~40 AR frames (~3 s audio), amplitude audibly decayed toward silence.
The reference at fp32 is stable for 20+ seconds; the CoreML fp16 port
drifted at +0.50% per frame.

**Localization:** `docs/investigations/fp16_drift_localization.md`.
Per-submodel fp32 swap experiment showed **`flow_lm_main` alone**
accounted for the drift. Swapping only it to fp32 raised envelope
correlation from 0.45 → 0.95.

**Fix applied in commit `214209b`:** post-conversion MIL pass using
`coremltools.optimize.coreml.FP16ComputePrecision` with an
`op_selector` that pins `softmax` and `layer_norm` to fp32. Weights
stay fp16. Applied to `flow_lm_main` + `flow_lm_prefill`. **Initial
attempt (PyTorch-side `scores.float()` cast in the manual SDPA) was
fused away by coremltools' optimizer.** The post-conversion MIL pass
is more robust because it sets op compute_precision directly.

**Cost:** RTFx dropped from ~10× to **7.44×** realtime on A19 Pro.
Artifact sizes unchanged. Drift slope now +0.03% per frame (unaudible).

### 6.7 Spanish port: tokenizer mismatch in the Python CLI driver

When testing Spanish end-to-end on command line, CoreML generation
triggered EOS on step 2. Cause: my test script constructed
`CoreMLGenerator(artifacts_dir="Artifacts/es_fp16")` without passing
`language="spanish"`. The generator defaulted to `language="english"`,
loading the English tokenizer + BOS path but the Spanish mlpackages.
English tokenizer on Spanish text → byte-fallback garbage tokens →
spurious logits → immediate EOS.

**No code fix needed** — the iOS app threads language correctly. Just
pass `language=` explicitly to `CoreMLGenerator` in any CLI test
script.

### 6.8 SwiftUI gotchas in the demo app

- **StreamingPlayer completion race.** AVAudioPlayerNode's
  `scheduleBuffer(..., completion:)` handlers all captured the same
  `currentChunk` at schedule time (all buffers queued before any
  completed), so `currentChunk` never advanced past 1. Fix: capture
  each chunk's own `finishIndex = ++scheduledCount` in the closure.
- **Menu/Picker inside custom card = flaky hit test.** Replaced with
  a full `.sheet(isPresented:)` + `List` picker. Bulletproof for 21+
  voices.
- **`@Observable` struct-nested mutation.** Mutating
  `stats.audioSeconds = x` directly sometimes didn't publish. Fix:
  replace the whole `stats` struct (`var s = stats; s.x = …; stats = s`)
  so one change notification fires per update.
- **`Color.primary` vs `.orange` in a ternary** — Swift 6 strictness
  rejected the mixed types. Explicit `Color.primary : Color.orange`.
- **`xcodegen generate` wipes `xcshareddata/xcschemes/`** unless the
  scheme is explicitly declared in `project.yml`. Added a `schemes:`
  block; without it, Xcode shows "no scheme" after regen.

### 6.9 Pytest env confusion

`uv run pytest` resolved pytest from a global environment that didn't
include the project's site-packages. **Use `.venv/bin/python -m pytest`
directly** — the project venv has the right deps.

### 6.10 SafetensorsWriter use-after-scope corruption (cloned voices)

Symptom: cloned voice files written by `VoiceLoader.saveVoiceOnly`
failed to load with `SafetensorsError.unsupportedDType` when the
runtime tried to use them. Mac often worked; iPhone always failed.
Inspection showed the 8-byte header-size prefix held garbage bytes
(observed `6_812_757_168` instead of the correct `1272`), and the
JSON header was effectively unreadable past position 1273.

Root cause: the writer was serializing `UInt64(headerBytes.count)
.littleEndian` into `Data` with this pattern:

    out.append(Data(
        bytes: withUnsafePointer(to: headerSize) { UnsafeRawPointer($0) },
        count: 8
    ))

`withUnsafePointer(to:)` hands the closure a pointer to the
stack-local `headerSize`. Returning that pointer from the closure and
using it in `Data(bytes:count:)` is a **use-after-scope**: the
compiler is free to reuse the stack slot once the closure returns,
and by the time `Data` reads the 8 bytes those bytes may have been
overwritten by the next tensor payload. iPhone was reliably
clobbering them; Mac occasionally got lucky.

Fix: serialize via a local `var` + inout pointer, which guarantees
the variable is still alive while `Data` reads it:

    var headerSizeLE = UInt64(headerBytes.count).littleEndian
    let headerSizeBytes = Data(
        bytes: &headerSizeLE,
        count: MemoryLayout<UInt64>.size
    )

**General rule for Swift `UnsafePointer` patterns:** any pointer
returned from a `withUnsafePointer`-style closure is only valid
inside the closure body. Using it after the closure returns is
undefined behavior — even if Mac seems to tolerate it.

### 6.11 mimi_encoder ANE compiler hang on iPhone

Symptom: loading `mimi_encoder.mlmodelc` on iPhone via
`MLModel(contentsOf:configuration:)` with `computeUnits=
.cpuAndNeuralEngine` never returned. Load time was "still running
after 30+ seconds" with the app appearing hung during first
voice-clone. Mac did not reproduce. Other models (flow_lm_main,
flow_lm_prefill, mimi_decoder, flow_lm_flow, text_conditioner)
loaded fine with the same compute-units setting.

Root cause: mimi_encoder has SEANet stride-6/5/4 inverse convolutions
and a depthwise `ConvTrUpsample1d` with `stride=16, groups=512` —
none of which are ANE-supported. iOS 26's ANE program compiler tries
to split the graph around these ops; for this particular encoder
shape the splitter enters a state from which it never returns
(confirmed by pause-in-Xcode landing inside an ANECompilerService
call stack for minutes).

Fix: force `computeUnits = .cpuOnly` when loading `mimi_encoder`
specifically. Hard-coded in `PocketTTS.ensureVoiceCloner`. The
encoder was measured at ~70 ms/predict on CPU anyway (it's off the
hot path — runs once per voice clone, not per AR frame), so there's
no perf cost.

**General rule:** for CoreML submodels that are mostly CPU-fallback
anyway, don't include ANE in their `computeUnits`. ANE's graph
splitter has known loop/hang modes when too many unsupported ops
force it to fragment the graph too aggressively.

### 6.12 ANE program cache is per-app-identity

Spent an hour chasing a phantom "startup went from 3s → 30s" regression.
The 3s number we'd measured before was **warm ANE program cache** from
previous app launches. True cold-install startup is 60–80 s for 5
mlmodelc bundles on A19 Pro (each model pays a multi-second ANE
program compile). After that first launch, subsequent launches of
the same app binary are ~3 s (cache hit).

Any action that changes the app framework binary identity invalidates
the cache. Examples we observed:
- Xcode rebuild that touches the SwiftPM package binary
- `Product → Clean Build Folder`
- Delete + reinstall app

**Diagnosis rule:** when "load got slower" after an edit, **close the
app from the switcher and relaunch without rebuilding**. If the second
launch is fast, you're just seeing cold-cache cost on reinstalls — no
real regression.

**Practical mitigations for production apps:**
- Show a splash screen "Preparing voices (first launch only)…" that
  runs the load in a background task.
- Or do it via a Background Task + local notification so the user
  doesn't sit watching a spinner.
- Or use App Clips / App Extensions to amortize the cost.

---

## 7. Directory layout

```
pocketTTS-CoreML/
├── Artifacts/                              (gitignored: regenerate locally)
│   ├── en_fp16/                       6 .mlpackage + 2 sidecars (dev-time)
│   ├── en_fp16_mlmodelc/              6 .mlmodelc + 2 sidecars (ship)
│   └── es_fp16/, es_fp16_mlmodelc/         Spanish counterparts
├── voices/                                 (gitignored) 21 English voice safetensors
├── voices_spanish/                         (gitignored) Spanish voices + tokenizer
├── pockettts_coreml/
│   ├── reference/                          git subtree @ kyutai-labs/pocket-tts v2.0.0
│   ├── patches/                            our forked, trace-friendly modules
│   ├── convert/                            per-submodel conversion scripts + __main__
│   ├── e2e/                                Python CoreML orchestrator (CoreMLGenerator)
│   └── oracle/                             golden-output fixture + dump_golden
├── PocketTTSCoreML/                        Swift package
│   ├── Package.swift
│   ├── Frameworks/SentencePiece.xcframework
│   ├── Sources/PocketTTSCoreML/
│   │   ├── PocketTTS.swift                 actor: public API
│   │   ├── Orchestrator.swift              AR loop; AsyncThrowingStream producer
│   │   ├── KVCacheBuffers.swift            rank-5 KV preallocation
│   │   ├── MimiStateBuffer.swift           pack/unpack decoder state blob
│   │   ├── VoiceLoader.swift               safetensors → KV
│   │   ├── Tokenizer.swift                 SentencePiece wrapper
│   │   └── …
│   ├── Sources/PocketTTSCLI/               macOS CLI (generate/benchmark/clone)
│   └── Examples/
│       ├── iOSBenchmark/                   Phase 4B: RTF + ANE residency on device
│       └── DemoApp/                        Phase 7: SwiftUI demo (Generate/Stream/Play/Share)
├── docs/
│   ├── research/                           external research + reference code map
│   ├── investigations/                     drift localization write-up
│   ├── phase2_3_notes.md                   conversion details (5 submodels)
│   └── mimi_state_layout.md                Mimi state blob format
├── tests/                                  Python pytest suite (34 tests)
├── thoughts/shared/plans/                  original phased plan
└── thoughts/shared/handoffs/               session handoffs
```

---

## 8. Lessons learned (condensed)

### 8.1 On CoreML conversion

1. **The `aten::Int` rule is the #1 gate.** Anything that produces a
   scalar int at trace time from a tensor will block `ct.convert`.
   Plan the model's forward signature around pure tensor ops — one-hot
   masks, scatter_mask, precomputed lookup tables — from day one, not
   as retrofits.

2. **Fp16 + mask + softmax is a landmine.** Never use `-inf` or
   boolean masks with fp16 softmax. Always use additive fp16 masks
   with `-65500` (fp16 near-min). **And even then, pin softmax to
   fp32** if the AR loop iterates more than ~20 times — the residual
   drift compounds.

3. **`F.scaled_dot_product_attention` is not a free win on CoreML.**
   Under some shape + mask combinations the fused op silently produces
   NaN. Manual `matmul / softmax / matmul` is safer when you're
   already patching the module.

4. **Post-conversion MIL passes beat PyTorch-side casts.** We tried
   inline `scores.float()` casts first; coremltools' optimizer fused
   them away. The `FP16ComputePrecision` pass with a custom
   `op_selector` is the only reliable way to pin specific ops to fp32
   while keeping weights fp16.

5. **coremltools version pin is load-bearing.** We pin `==8.1`. 9.x
   bumps deployment target to iOS 26 which is unshippable today.

### 8.2 On the model

6. **Autoregressive + fp16 + long context = compounding drift.** The
   port functions correctly for short sentences at fp16 but amplitude
   decays audibly past ~40 AR steps. Localizing the drift to a single
   submodel (attention softmax + layer norm in `flow_lm_main`) took
   ~1 hour of per-submodel fp32 swap experiments. **Do this early for
   any AR model** — you will find the bug eventually, and it's cheaper
   to find before shipping.

7. **"Speaker embedding" models (Resemblyzer cosine) are robust to
   fp16 drift.** Our waveform PSNR vs Python reference was 15 dB
   (terrible if you're looking at samples), but speaker similarity
   was 0.98 (excellent). The waveform diverges due to AR phase drift;
   the *speaker identity* is preserved. Use the right metric for the
   right question.

8. **Voice = prefilled KV, not an embedding.** PocketTTS voices are
   `.safetensors` files containing a serialized KV cache after running
   a voice `.wav` through mimi_encoder + flow_lm prefill. This shapes
   everything downstream:
   - Voice switching is free (copy the KV into the preallocated
     buffer).
   - Voice cloning needs `mimi_encoder + flow_lm_main` to run
     together, not just the encoder.
   - Each language has its own voice set — they don't interchange.

### 8.3 On shipping to iOS

9. **Pre-compile `.mlmodelc` on Mac, ship it, skip runtime compile.**
   The default pattern (ship `.mlpackage`, compile on first run, cache
   result) does not work on iOS because the app bundle is read-only.
   Every launch re-pays the ANE compile cost (1-3 s × 6 models).

10. **Ship size: .mlmodelc is half the size of .mlpackage.** The MIL
    protobuf source gets dropped.

11. **ANE residency doesn't dominate perf on A19 Pro.** We measured
    46.9% ANE placement on `flow_lm_main` but wall time was
    indistinguishable from `cpuOnly` mode. A19's CPU is fast enough
    that the ANE dispatch overhead breaks even with the ANE compute
    speedup. This may not hold on older SoCs; re-measure if
    targeting A15 or earlier.

12. **Thermal state stays `.nominal` under sustained gen.** Neither
    the 6-layer transformer nor the CPU-heavy mimi_decoder generate
    enough heat to throttle even on iPhone 17 Pro in sustained use.

### 8.4 On the Swift runtime

13. **Keep a clear actor boundary.** `PocketTTS` is an actor; voice
    handles are single-use; the orchestrator runs inside. Trying to
    cache voice handles in a `@Observable` class burned hours of
    debugging because KV mutation crossed actor boundaries silently.

14. **Explicit KV I/O beats `MLState` when perf is good enough.**
    We went with explicit rank-5 KV tensors in/out of every
    `.predict()` call. The copy cost is ~157 MB/s which sounds awful
    but is fine on unified memory. `MLState` is a potential future
    optimization if we need to claw back ~5% RTF.

15. **`AVAudioPlayerNode` completion handlers capture closures, not
    references.** If you use `currentChunk` to track progress, capture
    a per-buffer `finishIndex` explicitly — don't read the mutable
    state from inside the closure.

16. **Never use `withUnsafePointer(to:) { UnsafeRawPointer($0) }`
    and use the result after the closure returns.** The pointer
    references a stack-local that goes out of scope when the closure
    exits. Swift is free to reuse the slot; you'll read whatever
    overwrote it. Mac often gets away with it; iPhone reliably does
    not. Use `Data(bytes: &localVar, count: …)` instead — it reads
    the bytes while `localVar` is still in scope. SafetensorsWriter
    had this bug for ~3 weeks before surfacing on iPhone (§6.10).

17. **`.cpuAndNeuralEngine` can hang iOS CoreML** on models where
    most ops are ANE-unsupported. The ANE graph splitter enters a
    non-terminating state trying to fragment the graph. If a
    submodel is mostly CPU-fallback anyway, force `.cpuOnly`. We
    had to do this for `mimi_encoder` (§6.11).

18. **ANE program cache is keyed to app-binary identity.** Rebuild
    or reinstall → cold compile. "Startup got slower" after an edit
    is usually just cold-cache cost; relaunch without rebuilding to
    confirm. Real first-install cost for 5 mlmodelc bundles on A19
    Pro is 60–80 s (§6.12), not the 3 s warm-cache number most
    measurements were picking up.

### 8.5 On process

16. **Chunk planning paid for itself many times over.** The original
    plan (`thoughts/shared/plans/2026-05-01_...md`) set up gates for
    each phase. When drift surfaced in Phase 4, the "ship fp16
    anyway, fix with selective fp32" decision was easy because the
    tradeoffs had already been reasoned about.

17. **Compress only when you understand what you're skipping.** Phases
    2+3 were compressed to a single kraken cycle (skipped per-stage
    parity harness, used end-to-end audio only). This saved ~2 days
    but we had to rediscover some specifics later. For a greenfield
    port it would have cost more than it saved.

18. **Premortem-before-build caught the "gated weights disable voice
    cloning" tension.** The ungated checkpoint can't clone voices —
    which is why the plan got upgraded to use gated weights with
    HF_TOKEN during the premortem pass. Saved a week of rework.

---

## 9. Known limitations + future work

### 9.1 Known limitations

- **Commercial licensing:** Kyutai weights are **CC-BY-NC-SA**. Any
  app shipping these weights to the App Store needs separate
  commercial terms from Kyutai. Your MIT-licensed code is fine; the
  weights aren't.
- **Adding a new 6L language is ~15 min** conversion + ~2 min
  mlmodelc compile + 1 line in `LANG_SPECS` + 1 entry in
  `VoiceCatalog.Language.all` + a verify script run. The pipeline
  is fully parameterized by `--language`.
- **Adding a 24L language costs ~60 min** (longer conversion due to
  4× as many ops to trace + fp32-softmax MIL pass), ~4× the bundle
  size per language (~1.22 GB vs 356 MB), and ~4× lower per-step
  compute. The Swift runtime resolves layer count at init from the
  loaded mlpackage — no code changes needed per language after the
  initial French 24L work.
- **Voice cloning from user `.wav` in Swift:** shipped. The
  `pocket-tts-cli clone` command and `PocketTTS.cloneVoice(from:)` API
  run the full two-stage pipeline (mimi_encoder → speaker projection →
  flow_lm_prefill) entirely on device. See §4.4 for the pipeline and
  §4.3 for the `speaker_proj.safetensors` sidecar. Speaker cosine
  similarity vs the reference Python path: ~0.88 on 4 s of Alba
  (Resemblyzer gate 0.85) — reference fp32 path on the same 4 s scores
  ~0.89 vs the source wav. Voice quality is gated by fp16 roundoff in
  `mimi_encoder` + the padded `[1, 128, 1024]` prefill input, not by
  the Swift math.

### 9.2 Optional follow-ups (in priority order)

1. **Stream → Play/Share pipeline** in the demo app. Currently
   Play/Share disable after Stream because streamed chunks aren't
   accumulated into `generatedPCM`. ~20 LOC fix.
2. **Quantization (6-bit palettize)** to shrink ~356 MB → ~100 MB.
   Only worth it if ship size matters; perf is already solved. Test
   required per-language (quantization may not hold across languages
   equally).
3. **Re-export `mimi_encoder` with a longer static `T_audio`** (say
   10 s → T_latent=126) to match the duration Kyutai's shipped
   voices use. Would boost cloned-voice similarity closer to the
   pre-exported ceiling without sacrificing real-time speed
   (cloning is a one-shot op).
4. **Background audio testing.** `UIBackgroundModes: [audio]` is
   declared; actual background-playback behavior under iOS 18.5's
   background scheduling hasn't been measured.
5. **Other language ports.** Each new language is ~15 min convert
   time + ~5 min mlmodelc compile, plus any per-language tokenizer
   + voice verification.

---

## 10. References

- Plan: `thoughts/shared/plans/2026-05-01_pockettts_coreml_english_port.md`
- Research: `docs/research/external_research.md` (ANE constraints + CoreML-LLM patterns)
- Reference code map: `docs/research/reference_codebase_map.md`
- Conversion notes: `docs/phase2_3_notes.md`
- Mimi state layout: `docs/mimi_state_layout.md`
- Drift investigation: `docs/investigations/fp16_drift_localization.md`
- Session handoffs:
  - `thoughts/shared/handoffs/pockettts-coreml/2026-05-01_21-38_demo-app-shipped.yaml`
  - `thoughts/shared/handoffs/pockettts-coreml/2026-05-03_14-34_6-languages-and-drift-fix.yaml`
- Kyutai PocketTTS paper: [arxiv 2509.06926](https://arxiv.org/abs/2509.06926)
- Upstream repo: [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts)
