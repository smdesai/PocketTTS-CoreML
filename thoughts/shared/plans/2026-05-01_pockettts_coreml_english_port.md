# Implementation Plan: PocketTTS (English) → CoreML / ANE+CPU

Generated: 2026-05-01
Author: plan-agent (Opus)
Target for review: premortem → kraken (phase-by-phase implementation)

Source research (mandatory reading, not duplicated here):
- `docs/research/external_research.md` — CoreML/ANE patterns + PocketTTS architecture
- `docs/research/reference_codebase_map.md` — file:line map of the Python reference at `/Users/sdesai/Tools/MLX/pocket-tts`

---

## TL;DR

- Split into **5 CoreML packages** (`text_conditioner`, `flow_lm_main`, `flow_lm_flow`, `mimi_encoder`, `mimi_decoder`) mirroring Kyutai's own ONNX export, convert from PyTorch (not MLX — the reference at `/Users/sdesai/Tools/MLX/pocket-tts` is PyTorch despite the path name, verified in its `pyproject.toml`), target `.cpuAndNeuralEngine` with explicit KV I/O (not `MLState`) to dodge the dual-state ANE failure documented in CoreML-LLM's LFM2 findings.
- **Per-frame budget (80 ms at 12.5 Hz)** with `lsd_decode_steps=1` (verified default): 1 transformer fwd + 1 flow MLP fwd + 1 mimi decode + KV state copy. RTF<1 is achievable with fp16 only if we eliminate the 5 portability landmines in a PyTorch fork *before* tracing — verified by the research as the single biggest risk.
- Gate each phase on **numerical parity vs. reference Python** (golden dumps per-stage), then on **ANE-residency from `MLComputePlan`** per submodel, then on **RTF on target hardware (M4 laptop as primary, iPhone 15 Pro / A17 as stretch)**. No phase ships until its gate passes; quantization below fp16 is deferred until fp16 meets RTF.

---

## Assumptions (explicit — flag in premortem if wrong)

1. **Reference impl is PyTorch, not MLX.** Verified `/Users/sdesai/Tools/MLX/pocket-tts/pyproject.toml` line 8 (`torch>=2.5.0`, CPU-only index) and `pocket_tts/models/flow_lm.py` (`import torch`). The parent directory name `MLX/` is a filesystem quirk; there is no MLX code in the reference. We can therefore:
   - Trace directly from the reference after landmine patches.
   - Skip any MLX→PyTorch bridge.
   - Use the reference as the numerical oracle without conversion.

2. **Primary target hardware: MacBook Air/Pro M4 (dev + RTF baseline).** Stretch: iPhone 15 Pro (A17 Pro) for mobile background execution. Rationale: M4 matches the reference's advertised ~6× RTF on CPU, so we have a fair comparison point; A17 Pro has the ANE generation (16-core, 35 TOPS) that the CoreML-LLM reference work validated with similar-shape transformers. iPhone 17 Pro (A19) and M-series newer SoCs should be superset-compatible.

3. **Target OS floor: iOS 18 / macOS 15.** Required for SDPA fusion, `ct.StateType`, and `MLComputePlan`. We will ship with `minimum_deployment_target=ct.target.iOS18`. iOS 26 / macOS 26 features (coremltools 9.x) are out of scope.

4. **coremltools 8.x pin.** Specifically `coremltools==8.1` (or latest 8.x at repo-setup time). 9.0b1 is a hard no; it bumps deployment target to iOS 26.

5. **Voice embeddings are pre-serialized KV-cache prefills** (opaque tensor bundles), loaded by Swift and fed as the initial KV state. Runtime never touches `mimi_encoder` in the hot path. This matches `export_model_state()` in `tts_model.py:1047`.

6. **`lsd_decode_steps=1` is the production default** (verified `default_parameters.py:4`). Higher N is a quality knob exposed at the Swift API surface; we pre-compile enumerated-shape variants for N ∈ {1, 2, 4} if/when needed.

7. **Mimi decoder will partially fall back to CPU.** SEANet `ConvTranspose1d` strides 6/5/4 (`english.yaml:38`) and `ConvTrUpsample1d` depthwise-with-stride-16 (`resample.py:34`) are both ANE-unsupported. We accept this and budget for it.

8. **English only, v1.** Per-language weights (~219 MB fp32 → ~110 MB fp16) shippable in-app. Other languages deferred — `_24l` variants (24 layers) are 4× the FlowLM weight footprint and require re-validation.

9. **Weights source: HuggingFace `kyutai/pocket-tts` (gated, requires `HF_TOKEN`).** Per user decision (post-premortem). Rationale: the ungated checkpoint `kyutai/pocket-tts-without-voice-cloning` *disables* runtime voice cloning — the reference sets `has_voice_cloning = False` at `tts_model.py:206` and raises `ValueError(VOICE_CLONING_UNSUPPORTED)` at `tts_model.py:872` when a user wav is supplied. Since voice cloning is IN scope for v1 (Decision 5), we must use the gated checkpoint. Implications:
   - **CI:** requires `HF_TOKEN` as a GitHub Actions secret (user-provided). If unavailable, CI falls back to unit-test-only paths (no weight loading), with a nightly cron on a self-hosted runner for full integration tests.
   - **Local dev:** contributors need HF account + license acceptance on the Kyutai repo + `huggingface-cli login` once.
   - **Phase 1 fixture:** still uses `alba` (pre-exported voice) but loaded via the gated weights; parity between ungated and gated checkpoints for non-cloning paths is NOT assumed — we validate against gated-weight reference outputs only.

---

## Package layout

Single Python package (conversion) + single Swift package (runtime), both in this repo.

```
pocketTTS-CoreML/
├── pockettts_coreml/              # Python: conversion + verification
│   ├── __init__.py
│   ├── reference/                 # git-subtree (or submodule) of /Users/sdesai/Tools/MLX/pocket-tts
│   ├── patches/                   # our forked-module overrides that fix the 5 landmines
│   │   ├── transformer_patched.py
│   │   ├── rope_patched.py
│   │   ├── mlp_patched.py
│   │   └── flow_lm_patched.py
│   ├── oracle/                    # golden-output generation + diffing
│   │   ├── dump_golden.py
│   │   └── compare.py
│   ├── convert/                   # one script per submodel
│   │   ├── convert_text_conditioner.py
│   │   ├── convert_flow_lm_main.py
│   │   ├── convert_flow_lm_flow.py
│   │   ├── convert_mimi_encoder.py
│   │   └── convert_mimi_decoder.py
│   ├── quantize/                  # phase 6
│   └── bench/                     # phase 5 harness
├── PocketTTSCoreML/               # Swift package (runtime)
│   ├── Package.swift
│   ├── Sources/PocketTTSCoreML/
│   │   ├── PocketTTS.swift        # public API
│   │   ├── TextConditioner.swift
│   │   ├── FlowLMOrchestrator.swift
│   │   ├── MimiDecoder.swift
│   │   ├── KVCacheBuffers.swift
│   │   ├── RoPECache.swift
│   │   ├── NoiseSource.swift
│   │   ├── VoiceLoader.swift
│   │   └── AudioStream.swift
│   └── Tests/PocketTTSCoreMLTests/
├── Artifacts/                     # .mlpackage output (gitignored; see "Distribution" below)
├── thoughts/shared/plans/         # this doc lives here
├── docs/research/                 # existing research docs
└── pyproject.toml                 # coremltools + torch (matching reference) + pytest
```

### Distribution of `.mlpackage` artifacts

- **Not checked into git.** `.mlpackage` is a bundle (directory); git LFS is possible but awkward for bundles and bloats clones.
- **GitHub Releases** as the primary distribution channel: each release tag (`v0.1.0-english-fp16`, `v0.2.0-english-fp16-palettized6bit`, …) publishes a `.zip` of all 5 `.mlpackage` files plus `voices/*.safetensors`.
- **Download helper**: `pockettts_coreml.download.fetch_artifacts(tag, dest)` pulls from the Release; Swift side reads from app bundle or `Documents/`.
- **CI builds + publishes artifacts** on every `main` push that modifies `pockettts_coreml/convert/`. Dev-local builds write to `Artifacts/` (gitignored).

---

## Phase 1 — Repo scaffold + verification harness (greenfield → first golden dump)

### Goal
- A fully-wired Python project that can (a) load reference PocketTTS, (b) run the English model on a fixed prompt + fixed voice, (c) dump a per-stage "golden output" bundle to disk for later comparison. No CoreML yet.

### Scope in
- Python package layout (`pockettts_coreml/`).
- Reference repo ingested as **`git subtree`** under `pockettts_coreml/reference/` (not submodule — avoids a second clone step for contributors; we can `subtree pull` periodically). Pinned to the current commit on `main` (tag v2.0.0 per map doc).
- `pyproject.toml` with `torch>=2.5.0,<2.8` (matches reference), `coremltools==8.1`, `pytest`, `safetensors`, `numpy>=2`, `sentencepiece`.
- One ungated weight + one voice + one canonical English prompt checked into `pockettts_coreml/oracle/fixtures/`:
  - Prompt: "Pocket TTS is a lightweight text-to-speech model." (50-ish tokens, single chunk, fits in one `MAX_TOKEN_PER_CHUNK=50` segment — avoids sentence splitting in the oracle path).
  - Voice: `alba` (first in README catalog, ungated, small file).
  - Fixed seed: `torch.manual_seed(42)` for noise sampling.
- `oracle/dump_golden.py`: runs the reference end-to-end and captures **per-boundary tensors** matching the 5-submodel split:
  1. `text_conditioner_out` — `[1, S_text, 1024]` after `LUTConditioner`.
  2. `flow_lm_main_step_{0..N-1}` — per-AR-step dump of `(sequence_in, text_embeddings, kv_cache_in, noise_in)` and `(ctx_out, eos_out, kv_cache_out)` — structured as tensor dict.
  3. `flow_lm_flow_step_{0..N-1}` — `(c, s, t, x_in, u_out)`.
  4. `mimi_encoder_out` — voice-embedding path, 1 tensor. (Verify the voice file is a `.wav` path, not a pre-serialized safetensors — we want to exercise the encoder at least once.)
  5. `mimi_decoder_step_{0..N-1}` — `(latent_in, conv_state_in, transformer_kv_in, audio_out, conv_state_out, transformer_kv_out)`.
- `oracle/compare.py`: a stage-agnostic `assert_close(actual, golden, rtol, atol, stage_name)` with per-stage tolerances (defined in Phase 3 gates).
- Pytest smoke test `tests/test_oracle_roundtrip.py`: load golden → re-run reference → golden ≡ self with `atol=0`. Proves determinism of the reference under our fixed seed + `torch.set_num_threads(1)`.

### Scope out
- No CoreML conversion.
- No landmine patches.
- No Swift code.
- No voice-cloning beyond loading the pre-existing embedding.

### Concrete deliverables
- `pyproject.toml`, `uv.lock` (we follow the reference's choice of `uv`).
- `pockettts_coreml/reference/` subtree.
- `pockettts_coreml/oracle/{dump_golden.py,compare.py,fixtures/…}`.
- `pockettts_coreml/oracle/fixtures/english_alba_seed42/` containing:
  - `prompt.txt`
  - `voice_embedding.safetensors` (pre-exported via reference's `export_voice` CLI)
  - `golden/*.safetensors` (one file per stage/step)
  - `metadata.json` (tokenizer version hash, torch version, seed, LSD steps = 1, temperature)
- `tests/test_oracle_roundtrip.py`.
- `README.md` (repo-level) with a 10-line "how to regenerate the oracle" recipe.

### Verification criteria
- `pytest tests/test_oracle_roundtrip.py` passes: reference vs. golden is bitwise-identical (`atol=0, rtol=0`) when re-run under the same seed.
- Golden bundle size sanity check: ≤ 200 MB total (KV cache dominates; one AR step × 6 layers × 2 × 1 × ~150 × 16 × 64 × 4 bytes ≈ 1.5 MB per step; 50 steps ≈ 75 MB; mimi streaming state another ~20 MB).
- Manual listen-test: the generated wav from the oracle sounds like intelligible English speech with the expected voice.
- Reference's own tests (`tests/test_cli_generate.py`, `tests/test_documentation_examples.py`) pass under our environment — smoke check that our subtree ingestion didn't break anything.

### Risk flags
- **Reference determinism under fp32 CPU threading.** The reference sets `torch.set_num_threads(1)` at import time. We must match this in the oracle or golden comparisons will drift. Mitigation: `pytest` conftest forces threads=1.
- **Voice embedding format stability.** `export_model_state` serializes torch state dicts to safetensors. If Kyutai changes the keys, our golden breaks. Mitigation: pin to reference v2.0.0; store schema hash in `metadata.json`.
- **HuggingFace gated weights.** Per Assumption-9 update, we use the gated `kyutai/pocket-tts` checkpoint. CI requires `HF_TOKEN` secret. If CI token is absent or license changes revoke access, (a) cache the weights in GitHub Releases as an encrypted artifact (if license permits) or (b) allow CI to skip weight-loading tests with a `PYTEST_SKIP_WEIGHTS` marker and run nightly on a self-hosted runner with a persistent token.
- **`tokenizer.model` availability.** SentencePiece tokenizer ships with the weights; verify it's present in the ungated repo.

---

## Phase 2 — Portability patches in PyTorch (parity with reference, no CoreML yet)

### Goal
- Fork the 5 modules that hit traceability landmines (per `reference_codebase_map.md §11`) into `pockettts_coreml/patches/`. Prove that **patched_forward(x) ≡ reference_forward(x)** to within fp32-determinism tolerance on every stage boundary. The patched fork becomes the conversion-source-of-truth.

### Scope in

The 5 patches (all verified blockers in research doc):

1. **`transformer_patched.py`** — replacement for `pocket_tts/modules/transformer.py`:
   - Rewrite `complete_kv` (reference line 9-19) to eliminate `int(offset.view(-1)[0].item())` (produces `aten::Int`). Two strategies; we pick (a) and fall back to (b) if Phase 3 shows ANE rejection:
     - **(a) KV-as-I/O with mask-based write.** `complete_kv(cache, offset_onehot, k, v) -> new_cache` where `offset_onehot: [1, S_capacity]` is a one-hot indicator produced in Swift. Write becomes `cache = cache * (1 - offset_mask) + scatter(k, offset_mask)` — fully tensor-valued, zero `aten::Int`. The Apple `SliceUpdateKeyValueCache` pattern (vendored knowledge) is the reference.
     - **(b) Register cache as `register_buffer` + `ct.StateType`** following Apple's Llama pattern. Requires the write to compile into `coreml.slice_update` op. Only attempt this if (a) is too slow in Phase 3 AND the state is packed into a **single** state tensor. Per external_research.md the dual-state MLState pattern fails on ANE; we pack all 6×K+6×V into one `fp16[12, 1, S_cap, 16, 64]` state buffer to stay single-state.
   - Rewrite `_build_attention_mask` (reference line 23) to take a pre-computed **fp16 additive mask** of shape `[1, 1, T_q, S_capacity]` with values `0.0` (visible) or `-65500.0` (masked) — NOT a bool mask. fp16 arithmetic turns `-inf` into NaN in some code paths on ANE; `-65500` is the canonical "effectively zero after softmax" value documented in CoreML-LLM optimization notes. Drop all `offset + arange(T)` arithmetic.
   - Use `F.scaled_dot_product_attention(..., is_causal=False, attn_mask=bool_mask)` — the research confirms the reference already uses SDPA which fuses on iOS 18+.

2. **`rope_patched.py`** — replacement for `pocket_tts/modules/rope.py`:
   - Delete in-graph `torch.arange(T)`, `torch.exp`, `freqs = ...`. Forward becomes `apply_rope(q, k, cos, sin)` where `cos, sin: [1, T, 1, head_dim]` are passed as module inputs.
   - Swift side generates `cos/sin` tables from `(offset, T)` using a pre-computed once-per-load lookup table of shape `[MAX_CONTEXT, head_dim]`. This pushes the `arange` and integer math entirely outside the graph.

3. **`mlp_patched.py`** — replacement for `pocket_tts/modules/mlp.py`:
   - Swap custom `LayerNorm` (reference `mlp.py:39`, uses `x.var(unbiased=False)`) for `torch.nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)` inside `ResBlock`. The reference comment says the custom one was to support `jvp`; we don't need jvp for inference.
   - Swap custom `RMSNorm` (reference `mlp.py:20`, `x.var(dim=-1)` + `rsqrt`) for the canonical `x * rsqrt(mean(x**2, dim=-1, keepdim=True) + eps) * alpha`. Equivalent up to eps-shift; confirm parity in Phase 2 gates.
   - Keep the custom ops as Python fallbacks behind a `USE_ANE_FUSED_NORMS` env flag so we can A/B the accuracy delta. Note: RMSNorm still won't fuse to ANE (the `cat([x,-x])→LN→slice` trick from CoreML-LLM is LayerNorm-targeted); but the normalized form has better chance of being recognized by coremltools.

4. **`flow_lm_patched.py`** — replacement for parts of `pocket_tts/models/flow_lm.py`:
   - **Eliminate `torch.where(isnan(sequence), bos_emb, sequence)` (reference line 121).** New signature takes `sequence: float32[1, 1, 32]` only — no NaN convention. Caller (Swift) is responsible for passing `bos_emb` broadcast as the step-0 latent. First-step path uses `bos_emb` explicitly.
   - **Move noise sampling out of the graph** (reference lines 133-137, `torch.nn.init.normal_`/`trunc_normal_`). New inputs: `noise: float32[1, 32]` (Gaussian) and `noise_clamp: float` (passed as a tensor constant or hardcoded). Swift samples noise using Accelerate's `vDSP_vgauss`; for deterministic tests we pass a fixed tensor.
   - Refactor `FlowLMModel.forward` into two pure-functional paths: `flow_lm_main_forward(sequence, text_embeddings, kv_caches_in, rope_cos, rope_sin, attn_mask) -> (ctx, eos, kv_caches_out)` and let `flow_lm_flow_forward(ctx, s, t, x, noise) -> u` live separately. This is the seam for the 2-package split in Phase 3.

5. **(Implicit 5th patch) Mimi streaming state as explicit I/O.** Not a separate file; inside `patches/mimi_patched.py` wrap `StreamingConv1d`/`StreamingConvTranspose1d` so their `previous`/`partial` state dicts become explicit function inputs/outputs. This makes the graph pure-functional for tracing. The reference's `StatefulModule` dispatcher (`stateful_module.py`) stays intact for pytest parity runs; the patched wrappers expose a `pure_forward(x, *state_tensors_in) -> (y, *state_tensors_out)` shim.

### Patched-module parity harness
- `tests/test_patch_parity.py` loads the reference model + patched model with identical weights, runs each submodel on 10 random inputs + 1 oracle input, and asserts:
  - `text_conditioner`: `atol=0` (pure table lookup, no float math)
  - `flow_lm_main`: `atol=1e-5, rtol=1e-5` (matmul associativity noise only)
  - `flow_lm_flow`: `atol=1e-5, rtol=1e-5`
  - `mimi_encoder`: `atol=1e-4, rtol=1e-4` (longer conv chain, more accumulation)
  - `mimi_decoder`: `atol=1e-4, rtol=1e-4`
- End-to-end: generate the same oracle prompt+voice with the patched model; audio MSE vs. golden < `1e-6` (fp32 should be nearly exact with identical seed).

### Scope out
- No CoreML tracing yet.
- No Swift.
- No performance work — patches may be slower than the reference (we accept this).
- No SEANet architecture change (stride 6/5/4 → stride-2 cascade). That's a Phase 6-or-later optimization; Phase 2 keeps the arch identical.

### Concrete deliverables
- `pockettts_coreml/patches/{transformer,rope,mlp,flow_lm,mimi}_patched.py`.
- `pockettts_coreml/patches/__init__.py` with a `build_patched_model(config)` helper that instantiates the patched module graph using the reference's config loader + weight loader (we reuse `utils/config.py` and `utils/weights_loading.py` from the reference — no reason to re-implement).
- `tests/test_patch_parity.py`.
- A short doc `docs/phase2_patches.md` listing every landmine and the patch applied, with reference file:line citations (sourced from `reference_codebase_map.md`).

### Verification criteria
- All `test_patch_parity.py` cases pass with tolerances above.
- `torch.jit.trace`-ability smoke: for each patched submodel, `traced = torch.jit.trace(patched, example_inputs)` runs without error **and** `assert "aten::Int" not in str(traced.graph)` (CoreML-LLM's blessed check from `external_research.md §B.1 item 11`).
- `traced(example_inputs)` output matches `patched(example_inputs)` with `atol=1e-6`.
- End-to-end audio MSE vs golden < `1e-6`.

### Risk flags
- **Custom-LayerNorm eps drift.** The reference's `mlp.LayerNorm` uses `x.var(unbiased=False)` and `nn.LayerNorm` uses the same formula but with `eps` inside the sqrt in a potentially-different position. Parity to `rtol=1e-5` may fail. Mitigation: if it fails, add an `eps_shift` compensation or keep the custom LN but mark it as a known CPU-fallback op (flow_net is small; CPU cost is negligible).
- **RMSNorm formula switch.** Reference is `x * (alpha * rsqrt(var(x)+eps))`; canonical is `x * rsqrt(mean(x**2)+eps) * alpha`. These differ numerically — `var` subtracts the mean. For a zero-mean signal (which the pre-LayerNorm output is expected to be) they coincide; for non-zero-mean they don't. The timestep-embedder output before the RMSNorm may *not* be zero-mean. We must verify parity before committing to the swap; if it fails, keep the custom RMSNorm.
- **Mimi streaming-state refactor correctness.** The `StreamingConv1d` rolling buffer is fiddly (see `reference_codebase_map.md §4` — the `pad_mode="replicate"` branch is dead for English but still in the code). Risk: we silently drop a buffer update and audio quality degrades subtly over long generations. Mitigation: the parity test generates ≥ 10 seconds of audio (125 frames) and asserts per-frame PSNR > 80 dB vs. golden.
- **Scope creep: 10 attention layers × custom rewrite.** The FlowLM has 6 + Mimi has 2×2 = 10 total attention modules needing the patch. If we make per-layer subclass differences we multiply the work. Mitigation: single `PatchedStreamingMultiheadAttention` class used by both; differences (no context limit for FlowLM, context=250 for Mimi) are config params.

---

## Phase 2.5 — iPhone A17 Pro feasibility spike (3-day time-boxed; GO/NO-GO gate)

### Goal
- Answer "can a 6-layer 1024d autoregressive transformer at fp16 actually hit RTF ≤ 0.5 on iPhone A17 Pro?" before committing to the full Phase 3-7 effort. This is a response to pre-mortem Tiger-3: the iPhone-primary + 0.5 RTF target has no cited precedent for a model of this exact shape.

### Scope in (bounded — 3 calendar days MAX)
- Take Phase-2 patched `flow_lm_main` (just this one submodel; don't bother with Phase-2 patches for the others yet).
- Trace + convert to fp16 CoreML with explicit KV I/O, target iOS18, compute_units `.cpuAndNeuralEngine`.
- Deploy as a minimal Xcode project (single button, runs 100 predictions with a pre-computed input bundle, reports p50/p95 latency + ANE residency from `MLComputePlan`).
- Run on a physical iPhone 15 Pro or newer.
- Measure: p95 forward latency per step, ANE-residency %.

### Decision gate (strict — blocks Phase 3 until resolved)

| Measurement | Outcome | Action |
|-------------|---------|--------|
| p95 ≤ 30 ms AND ANE ≥ 80% | GREEN | Proceed to Phase 3 as planned; RTF≤0.5 is plausible. |
| 30 ms < p95 ≤ 50 ms OR 50% ≤ ANE < 80% | YELLOW | Pause; evaluate (a) whether palettization from Phase 6 can claw back the needed 40%, (b) whether RTF target should relax to 0.7. Re-plan before Phase 3. |
| p95 > 50 ms OR ANE < 50% | RED | Architecture likely cannot meet 0.5 RTF on iPhone at fp16. Options: drop iPhone primary (return to Mac-primary + iPhone stretch), relax RTF target to 0.9, or choose a different TTS model. STOP the port until direction is reconfirmed. |

### Scope out
- No other submodels.
- No production-grade code (the spike's code is throw-away; Phase 3 rebuilds cleanly).
- No mimi_decoder investigation (if flow_lm_main doesn't fit, mimi doesn't matter).
- No quantization, no palettization.

### Concrete deliverables
- `pockettts_coreml/spike/flow_lm_main_spike.py` — one-off conversion script, keep the commit around as documentation but not as part of the shipping codebase.
- `pockettts_coreml/spike/ANESpikeApp/` — single-screen SwiftUI iOS project that runs the bench.
- `docs/phase2_5_feasibility_report.md` — the numbers + GO/NO-GO call.

### Verification criteria
- Bench runs on iPhone 15 Pro without crash.
- Measurement includes warm-up (≥ 10 iterations discarded) before timed run.
- Report includes: p50, p95, p99 latency; ANE residency by op count AND by FLOP weight; thermal state before and after the 100-iter run.

### Risk flags
- **Spike overruns 3 days.** Strict time-box: if we can't get a measurement in 3 days, that's itself a NO-GO signal (the conversion pipeline is not tractable in reasonable time). Fail fast.
- **Spike uses unpatched landmines and fails to trace.** Acceptable — the spike's purpose is feasibility, not correctness. Apply *just enough* patches (eliminate `aten::Int`; leave RMSNorm custom) to get a traceable graph. Accept that the numerical output may be wrong.

---

## Phase 3 — Per-submodel trace → CoreML conversion (fp16)

### Goal
- Five `.mlpackage` files generated from the Phase-2 patched modules, each individually validated vs. the Phase-1 golden outputs within stage-specific tolerances. Everything in fp16, `minimum_deployment_target=iOS18`, `compute_units=.cpuAndNeuralEngine`. No Swift yet — we drive each `.mlpackage` from Python via the coremltools `.predict()` API for testing.

### Scope in

For each submodel, one conversion script under `pockettts_coreml/convert/`. Common recipe per script:

1. Instantiate patched model; set `eval()`; load weights.
2. Build `example_inputs` (static shapes where possible; `ct.RangeDim` where truly variable).
3. `torch.jit.trace(patched, example_inputs)` with `strict=False`.
4. Assert zero `aten::Int` in `str(traced.graph)`.
5. `ct.convert(traced, inputs=[...], outputs=[...], minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE)`.
6. Save to `Artifacts/en_alba_fp16/<name>.mlpackage`.
7. Run `mlmodel.predict(golden_inputs)` → compare to `golden_outputs`.

Per-submodel specifics and shapes (derived from `reference_codebase_map.md`):

### 3.1 `text_conditioner.mlpackage`
- **Input:** `tokens: int32[1, S_text]`, `S_text = RangeDim(1, MAX_CONTEXT=256, default=50)`.
- **Output:** `embeddings: fp16[1, S_text, 1024]`.
- **Graph:** just `nn.Embedding(4001, 1024)`. Single `gather` op. Expected CPU placement (ANE doesn't do gather well) — acceptable, runs once per chunk.
- **Size:** 4001 × 1024 × 2 bytes ≈ 8 MB.
- **Tolerance vs golden:** fp16 cast of golden `atol=1e-3, rtol=1e-3` (literally the fp32→fp16 roundoff of the embedding table).

### 3.2 `flow_lm_main.mlpackage`
- **Inputs:**
  - `sequence: fp16[1, 1, 32]` (the previous latent; step 0 = `bos_emb`)
  - `text_embeddings: fp16[1, S_prefill, 1024]` (padded; S_prefill static at `MAX_TEXT=128` with zero-mask). **This is the big question** — see Risk flags.
  - `kv_cache_in: fp16[6, 2, 1, S_cap, 16, 64]` (6 layers × 2 for K/V × batch × S_capacity × heads × head_dim)
  - `kv_offset_onehot: fp16[1, S_cap]` (one-hot position for this step's write)
  - `rope_cos, rope_sin: fp16[1, 1, 1, 64]` (one-step at a time in AR; S=1)
  - `attn_mask: fp16[1, 1, 1, S_cap]` (additive mask: 0.0 visible, -65500.0 masked; NOT boolean, see Phase-2 landmine-1 mitigation)
- **Outputs:**
  - `ctx: fp16[1, 1024]` (last-token context vector for flow MLP)
  - `eos_logit: fp16[1, 1]`
  - `kv_cache_out: fp16[6, 2, 1, S_cap, 16, 64]`
- **`S_cap` choice:** 256. Reference uses dynamic pre-alloc (`_expand_kv_cache`, `tts_model.py:390`); for CoreML we pick the largest typical prompt+generation window. 256 covers 50 text tokens + 206 audio frames ≈ 16 seconds of audio. Generations longer than that force a "new utterance" reinitialization (which the Swift layer does anyway for `split_into_best_sentences` chunks of 50 tokens).
- **fp16 concerns:** The 6-layer stack with 256-long KV will have attention scores magnitudes up to O(head_dim) = 64; softmax is safe. fp16 activation range per the vendored `16-bit.md` is fine. But: pre-normalized Q·K products could hit fp16 overflow at large offsets if RoPE amplification × matmul accumulation exceeds 65504. Mitigation: run Phase 2's parity test in fp16 before converting (i.e. `patched_model.half()`, compare to fp32); if PSNR drops below 40 dB, we need manual fp32 rescales on the softmax path (per CoreML-LLM's "Manual softmax with explicit fp16 casts" tip).
- **Tolerance vs golden:** `atol=5e-3, rtol=5e-3` on `ctx`. This is loose (fp16-on-ANE propagates through 6 layers × LN → MHA → FFN); based on CoreML-LLM's acceptance of "fp16 baseline 226× RTF" without WER regression, stage-level fp16 drift of this magnitude is expected and audio-inaudible.
- **Size:** 6 layers × (4 × 1024 × 1024 in_proj + 1024 × 1024 out_proj + 2 × 1024 × 4096 FFN) × 2 bytes ≈ 90 MB. Plus embedding tables: `bos_emb` 32×2, `input_linear` 32×1024×2, `out_norm` 1024×4, `out_eos` 1024×2 — rounding to ~91 MB.

### 3.3 `flow_lm_flow.mlpackage`
- **Inputs:** `c: fp16[1, 1024]`, `s: fp16[1, 1]`, `t: fp16[1, 1]`, `x: fp16[1, 32]`, `noise: fp16[1, 32]`.
- **Output:** `next_latent: fp16[1, 32]` (the final `x + u/num_steps` at num_steps=1 reduces to `x + u`, but we keep the full formula inlined for N>1 compatibility).
- **`lsd_decode_steps` variants:** compile N=1 as the default. Optionally compile N=2 and N=4 as separate `.mlpackage` files named `flow_lm_flow_n1.mlpackage`, `flow_lm_flow_n2.mlpackage`, `flow_lm_flow_n4.mlpackage`. For N>1 we **unroll** the loop into the graph (N `SimpleMLPAdaLN` copies chained) — rationale in Risk #3 of external research: avoids N-1 extra ANE dispatches per frame.
- **Size:** `SimpleMLPAdaLN` = `input_proj(32,512) + 2 × TimestepEmbedder + cond_embed(1024,512) + 6 × ResBlock(512) + FinalLayer(512,32)` ≈ 3 MB per copy. N=1 package ≈ 3 MB, N=4 unrolled ≈ 12 MB.
- **Tolerance vs golden:** `atol=1e-3, rtol=1e-3` on the output latent. 32-d output, 6-block MLP — fp16 drift bounded.

### 3.4 `mimi_encoder.mlpackage` (Phase-4 voice-cloning dependency — runs on-device each time a user clones a voice)
- **Inputs:** `waveform: fp16[1, 1, T_audio]` (pad to multiple of 1920).
- **Output:** `latents: fp16[1, 32, T_audio/1920]`.
- **Shape handling:** variable `T_audio` via `ct.RangeDim(1920, 24000*30, default=24000*3)` (up to 30 s of reference audio).
- **ANE placement:** **expected to fall back to CPU** entirely (stride 6/5/4 inverse path + ConvTrUpsample1d depthwise). Acceptable: cloning is a one-time op per voice (not a per-frame hot path), and result is cached on disk. Target: ≤ 500 ms for 10 s of reference audio on A17 Pro. Longer is acceptable with a "preparing voice" UX indicator.
- **Tolerance vs golden:** `atol=1e-3, rtol=1e-3` on the latent; ultimately validated end-to-end via the resulting KV-cache prefill producing voice-identity-equivalent audio (MFCC cosine ≥ 0.85 gate from Phase 4).
- **Important:** mimi_encoder alone is NOT a voice embedding. The voice embedding is the KV-cache state after a flow_lm_main prefill over these latents. See §Phase 4 `VoiceCloner.swift` for the two-stage pipeline.

### 3.5 `mimi_decoder.mlpackage`
- **Inputs:**
  - `latent: fp16[1, 32, 1]` (one AR frame)
  - Conv state tensors for every streaming conv layer (many — count them in Phase 2 when wrapping).
  - KV cache tensors for the 2 transformer layers (context=250).
  - Upsample `partial` state for `ConvTrUpsample1d`.
- **Output:**
  - `audio: fp16[1, 1, 1920]`
  - Updated conv/transformer/upsample state tensors.
- **ANE placement:** **expected mixed CPU+ANE.** Research flags stride-6/5/4 `ConvTranspose1d` and depthwise stride-16 `ConvTrUpsample1d` as CPU-fallback ops. The DummyQuantizer (Conv1d k=1), ELU, and residual pointwise convs should stay on ANE. The 2-layer mimi transformer will be on ANE if it's wired the same way as FlowLM (post-Phase-2 patches, it is).
- **Tolerance vs golden:** `atol=5e-3, rtol=5e-3` on audio samples. This is the loosest stage; more accumulation, and partial CPU fallback introduces additional fp16↔fp32 boundary rounding.
- **Size:** SEANet + mimi transformer + quantizer + upsample ≈ 20 MB fp16.

### Integration test (5-way composition in Python)
- `tests/test_coreml_end_to_end_python.py`: drive all 5 `.mlpackage` files from Python (coremltools `.predict()`, no Swift yet), reproduce the oracle prompt, compare final audio to golden. **Audio-level tolerance: PSNR ≥ 40 dB, MSE ≤ 1e-3.** This is the fp16 end-to-end gate.

### Scope out
- No Swift.
- No RTF measurement.
- No `MLComputePlan` inspection yet (deferred to Phase 5 to keep Phase 3's scope tight).
- No quantization below fp16.
- No stateful `ct.StateType` — we commit to explicit KV I/O in Phase 3. `MLState` experimentation is Phase 6-or-later optimization.

### Concrete deliverables
- 5 conversion scripts: `pockettts_coreml/convert/convert_*.py`.
- 5 `.mlpackage` bundles in `Artifacts/en_alba_fp16/`.
- `tests/test_coreml_end_to_end_python.py`.
- `docs/phase3_conversion_notes.md` — per-submodel: final input/output signature, ANE/CPU placement observed (from `mlmodel.compute_unit_used_info` if available in ct 8.1, else deferred to Phase 5), any surprises.

### Verification criteria
- All 5 per-submodel parity tests pass at the stated tolerances.
- End-to-end audio: PSNR ≥ 40 dB vs Phase-1 golden WAV, no audible artifacts on manual listen-test.
- Zero `aten::Int` in any of the 5 traced graphs (asserted in each conversion script).
- `coremltools.convert` returns with no warnings of the form "Op X not supported, falling back to CPU" for `text_conditioner`, `flow_lm_main`, `flow_lm_flow`. (Warnings for `mimi_decoder` and `mimi_encoder` are expected per research.)

### Risk flags
- **Text prefill shape.** `text_embeddings` has variable length up to 50 tokens per chunk (plus optional voice-embedding prefix). The research says `S_text` varies; do we pad to `MAX_TEXT=128` (wastes ANE compute) or use `ct.RangeDim` (risks CPU fallback on dynamic-shape ops)? **Decision:** static pad to 128 with fp16 additive attention mask (0.0 / -65500.0); the reference caps at 50 tokens per chunk (`MAX_TOKEN_PER_CHUNK=50`) so 128 covers it with margin. Wasted compute is acceptable; text prefill is a once-per-chunk cost, not per-frame. **Verification (new):** Phase 3 gate compares generation with padded-128 text to unpadded reference; PSNR on output audio ≥ 50 dB required.
- **KV cache size 256.** If the Swift runtime wants to generate >16 seconds of audio without resetting, we need a larger `S_cap` or multi-chunk stitching. Decision: enforce 256 in v1; stitching logic in Swift uses the existing 50-token chunk boundaries as natural reset points (the reference already does this via `split_into_best_sentences`).
- **SDPA fusion regression in coremltools 8.1.** Research (`external_research.md §B.3`) documents SDPA fusion at `ct.target.macOS15`/`.iOS18`. If it doesn't fire we get attention op-by-op, which is slow but correct. Mitigation: check `mlmodel.input_description` + inspect converted mlprogram for presence of `mb.scaled_dot_product_attention` op (visible in `MIL` IR dump).
- **Mimi conv-state tensor count explosion.** Every `StreamingConv1d`/`StreamingConvTranspose1d` needs `previous` (or `partial`) as an input and output. SEANet decoder has ~10 conv modules, plus the upsample. 10+ state tensors per call × 2 (input+output) = 20+ signature entries. CoreML handles this but the Swift glue code gets verbose. Mitigation: pack all conv states into a single `fp16[TOTAL_STATE_ELEMS]` tensor with a fixed layout, slice in/out in Swift using `vDSP_mmov`.
- **fp16 quality regression on flow MLP's sinusoidal timestep embedding.** `TimestepEmbedder` computes `torch.cat([cos(args), sin(args)])` with `args = t * freqs` where `freqs = exp(log(max_period) * arange(half))` — at `half=128, max_period=10000` the highest frequency is `~10000` rad/step which in fp16 at large `t` can lose precision. At `t∈{0,1}` it's fine; this only matters for `lsd_decode_steps > 10` which no one uses. Accept the risk for N ≤ 4.

---

## Phase 4 — Swift runtime glue (end-to-end from text to PCM)

### Goal
- A Swift Package (`PocketTTSCoreML`) that loads the 5 `.mlpackage` files, orchestrates a full generation, and streams 24 kHz mono PCM to an `AsyncSequence<Data>`. Numerical parity with Phase 3 Python-driven E2E test.

### Scope in
- `PocketTTS.swift` — public API:
  ```
  public actor PocketTTS {
    public init(modelBundle: URL, voicesBundle: URL) async throws
    public func generate(text: String, voice: String, lsdSteps: Int = 1)
      -> AsyncThrowingStream<Data /* int16 PCM */, Error>
  }
  ```
- `TextConditioner.swift` — loads SentencePiece tokenizer via a minimal Swift port (we use the `SentencePiece` C++ library via a Swift wrapper; alt is swift-sentencepiece package). Runs `text_conditioner.mlpackage`.
- `FlowLMOrchestrator.swift` — owns the KV cache `MLMultiArray`s; runs `flow_lm_main.mlpackage` once per AR step; runs `flow_lm_flow.mlpackage` once (or N times if unrolled not available); maintains `offset` counter.
- `MimiDecoder.swift` — owns conv/transformer streaming-state `MLMultiArray`s; runs `mimi_decoder.mlpackage` per AR step.
- `KVCacheBuffers.swift` — pre-allocated `MLMultiArray` pool (the `FeatureBag` pattern from CoreML-LLM / parakeet).
- `RoPECache.swift` — once-per-load cosine/sine table, sliced per step by `offset`.
- `NoiseSource.swift` — `vDSP_vgauss`-backed `MLMultiArray` provider; seedable for tests.
- `VoiceLoader.swift` — parses the `safetensors` voice bundle, copies prefilled KV into `KVCacheBuffers` (no encoder run at inference).
- `AudioStream.swift` — 1920-sample `MLMultiArray` → `Data` (int16 LE PCM) via `vDSP_vfixr16`.
- **Threading model:** two `Task`s with a capacity-2 bounded `AsyncChannel`:
  - Producer task: text-prefill → AR loop → `latentsChannel.send(next_latent)`.
  - Consumer task: `latentsChannel.receive()` → mimi_decoder → audio stream yield.
  - Bounded at 2 prevents IOSurface exhaustion (per parakeet OPTIMIZATIONS.md).
- **`autoreleasepool` around every `model.prediction()`** call.
- **Compute units:** `.cpuAndNeuralEngine` default. Per-model override via `MLModelConfiguration` if Phase 5 finds one submodel degrades on ANE.

### Tokenizer detail
- SwiftPM has `swift-sentencepiece` (Apple-aligned fork); fallback is vendored `sentencepiece` C++ via a local CocoaPod/SwiftPM binary. Decision: use `swift-sentencepiece` as primary; if it's missing required features (like `decode(ids_list)` for sentence splitting), vendor the C++ lib.
- `split_into_best_sentences` (reference `tts_model.py:978`) is Python-side string manipulation. Port to Swift as `TextChunker.swift` — straightforward; no tensor ops.

### Scope out
- No `MLComputePlan` inspection (Phase 5).
- No RTF measurements (Phase 5).
- No fancy background-execution hooks (`AVAudioSession` setup) — Phase 7.
- No voice *cloning* from a wav (i.e. running `mimi_encoder` at runtime) — v1 loads pre-exported voice safetensors only. Cloning-at-runtime deferred or done via a separate Swift tool.

### Concrete deliverables
- `PocketTTSCoreML/` SwiftPM package.
- `PocketTTSCoreML/Tests/PocketTTSCoreMLTests/EndToEndTests.swift` — loads the Phase-3 oracle fixtures, drives end-to-end, compares resulting int16 PCM to Python-driven E2E output. Same MSE threshold.
- `PocketTTSCoreML/Benchmarks/` — CLI binary (via SwiftPM `executableTarget`) that generates audio and dumps PCM to stdout; used by Phase 5 bench harness.

### Verification criteria
- `swift test` passes; Swift E2E output matches Python E2E output with `atol=1e-3` on int16 PCM (≈ 0.003% of full scale; inaudible).
- No memory leaks: Instruments "Leaks" template over 60 s of continuous generation shows stable `phys_footprint`.
- No `IOSurface` exhaustion over 10 minutes of continuous generation.
- Swift E2E audio PSNR ≥ 35 dB vs Phase-1 golden WAV (slightly looser than Phase 3's 40 dB to allow for int16 quantization).

### Risk flags
- **SentencePiece on iOS.** The C++ library is ~5 MB statically linked. Must target arm64; avoid x86_64 dependencies. Confirm `swift-sentencepiece` (if we pick it) supports `pod_mode=byte_fallback` that Kyutai's tokenizer uses (verify from `tokenizer.model` metadata).
- **MLMultiArray layout mismatches.** Reference uses NCHW-like `[B, C, H, W]` but PocketTTS transformer is `[B, S, C]` (channel-last). CoreML `MLMultiArray` can represent either; but operations like `vDSP_vfixr16` assume contiguous float arrays. Any non-contiguous view is a landmine. Mitigation: explicit `.copy()` on any reshaped tensor at the Swift/CoreML boundary.
- **KV cache copy cost.** Explicit KV I/O means 6 layers × 2 × 256 × 16 × 64 × 2 bytes = **6.3 MB copied in and 6.3 MB copied out, per AR step**. At 12.5 Hz = 157 MB/s memory traffic just for KV copies. Apple's unified memory handles this but it's not free; may eat 1-2 ms/frame. If this pushes RTF over, we switch to `ct.StateType` in Phase 5 or 6.
- **AsyncChannel vs AsyncStream back-pressure.** Swift Concurrency's `AsyncStream` has unbounded buffering by default; for back-pressure we need `AsyncChannel` (from `swift-async-algorithms`) or manual semaphore. The research prescribes capacity-2. Confirm `swift-async-algorithms` is API-stable at iOS 18; if not, roll our own with `DispatchSemaphore`.
- **Noise determinism across ANE.** `vDSP_vgauss` is CPU-only (fine), but CoreML conversion of fp16 noise addition may produce different results on ANE vs CPU due to ANE's fp16-only compute (vs Python oracle's fp32). Mitigation: relax E2E audio tolerance (already 35 dB not 60 dB); accept that ANE-converted fp16 will not be sample-bitwise-identical to the fp32 oracle — this is CoreML-LLM's accepted practice.

---

## Phase 5 — ANE verification + RTF measurement on target hardware

### Goal
- Answer: "Does this actually run on ANE, and is RTF<1 achievable for streaming TTS on the target hardware?" with quantitative evidence.

### Scope in
- **Per-submodel `MLComputePlan` inspection** (iOS 18 API — the *only* reliable way per vendored `is-model-using-ane.md`). Script: `pockettts_coreml/bench/inspect_compute_plan.py` (calls into a tiny Swift helper via `swift run` that returns JSON: `{op_name, dispatched_device: [cpuAndGPU|ane|cpuOnly]}` per op). Report aggregates:
  - `ANE-residency % by op count`
  - `ANE-residency % by FLOPs (weighted)`
- **Powermetrics correlation:** during a 30-second continuous generation, sample `powermetrics --samplers ane_power,cpu_power,gpu_power -i 100` at 10 Hz; plot. ANE power > 0 during transformer forward confirms hardware ANE usage, independent of Apple's reporting API.
- **Per-submodel latency microbench:** `pockettts_coreml/bench/bench_per_submodel.py` runs each `.mlpackage` 1000 times with warm-start (the 231 ms cold-start fix from CoreML-LLM), reports p50 / p95 / p99.
- **End-to-end RTF:** run Swift benchmark on 30 seconds of generated audio; `RTF = wall_time_seconds / audio_seconds_generated`. **SHIP TARGET: RTF ≤ 0.5 on iPhone A17 Pro or newer.** Mac (M-series) is expected ≤ 0.3 as dev-sanity but does not gate ship.
- **Primary hardware target: iPhone A17 Pro (or newer).** Per user decision. Mac M-series is dev-only / sanity-check. Implications:
  - ANE-inspector Xcode project at `PocketTTSCoreML/Examples/ANEInspector/` runs `MLComputePlan` on device.
  - Xcode Instruments Energy Log (category: "ANE") replaces macOS `powermetrics` for hardware ANE verification.
  - Thermal state tracking via `ProcessInfo.thermalState` during the RTF run; a thermal state above `.fair` invalidates the run.

### Decision framework for Phase 5 outcome
- **RTF ≤ 0.5 on M4, any ANE residency reported:** ship. Proceed to Phase 7 (streaming API); Phase 6 (quantization) becomes optional for size reduction not perf.
- **RTF 0.5-1.0 on M4:** partial success. Investigate hot spots; if a submodel is < 50% ANE-resident, consider Phase 6 palettization targeted at that submodel or arch changes for Mimi decoder.
- **RTF > 1.0 on M4:** fail. Root-cause investigation:
  - Is `flow_lm_main` on ANE? If not, the 6-layer transformer is CPU-bound → investigate KV-cache copy cost → try `ct.StateType` path.
  - Is `mimi_decoder` > 40 ms/frame? Then the SEANet CPU fallback is blowing the 80 ms budget → rewrite stride-6 as stride-3+stride-2 cascade, re-train? (Not feasible — no training; instead, exploit that the 24 kHz output can be downsampled to 16 kHz via a smaller output head — but that's an arch change.)
  - Is `lsd_decode_steps=1` still the default? (Should be. If Swift defaulted to higher, fix.)

### Scope out
- No fixes beyond identifying hot spots. Phase 5 is diagnostic; fixes are Phase 6.
- No multi-device sweep beyond primary (M4) + stretch (iPhone 15 Pro). Other M-series / A-series hardware tested opportunistically but not blocking.

### Concrete deliverables
- `pockettts_coreml/bench/inspect_compute_plan.py` + `pockettts_coreml/bench/inspect_compute_plan_helper.swift`.
- `pockettts_coreml/bench/bench_per_submodel.py`.
- `PocketTTSCoreML/Benchmarks/e2e_rtf.swift`.
- `docs/phase5_ane_report.md` — the actual measurements, tables, graphs (powermetrics trace images).

### Verification criteria
- `docs/phase5_ane_report.md` contains numerical tables with the specific numbers:
  - `text_conditioner`: ANE% by op count, p95 latency (µs).
  - `flow_lm_main`: ANE% by op count, p95 latency (ms). **Target: ≥ 80% ANE, ≤ 30 ms p95.**
  - `flow_lm_flow`: ANE% by op count, p95 latency (ms). **Target: ≥ 80% ANE, ≤ 5 ms p95.**
  - `mimi_decoder`: ANE% by op count (expected < 80%), p95 latency (ms). **Target: ≤ 40 ms p95** (irrespective of where it runs — within 80 ms frame budget).
  - `mimi_encoder`: latency per second of reference audio. **Target: ≤ 500 ms** (voice loading not a hot path; 2× real time is fine).
- End-to-end RTF ≤ 0.5 on M4 laptop.
- Phase-5 report includes powermetrics screenshots showing ANE power > 0 during generation.
- No thermal throttling during 5 minutes of continuous generation (`pmset -g therm` reports `CPU_Scheduler_Limit == 100` and `CPU_Speed_Limit == 100`).

### Risk flags
- **MLComputePlan API churn.** It's iOS 18-only, first-gen; if Apple changes the JSON shape in a 18.x dot release, our helper breaks. Mitigation: gate on `@available(iOS 18.0, *)` and have a manual "look at the Xcode ML profiler screenshot" fallback documented.
- **Cold-start dominates short benchmarks.** First prediction per `.mlpackage` takes ~230 ms (ANE program compile). Benchmarks must exclude cold-start. Mitigation: 10-call warmup before timed run; document.
- **iPhone-vs-Mac ANE residency differs.** Apple's compiler may place ops differently per SoC — e.g. A17 Pro ANE may accept ops that M2 ANE rejects. Mitigation: rerun MLComputePlan on both; report separately. Don't merge the numbers.
- **KV cache explicit-I/O is a perf floor.** If RTF is bad, we don't know whether ANE itself is slow or whether the 157 MB/s KV memcpy is the killer. Mitigation: add an "idle throughput" bench that runs `flow_lm_main` with zero-sized KV to isolate compute from memcpy.

---

## Phase 6 — Quantization / size reduction (optional, conditional on Phase 5)

### Goal
- Reduce weight size (fp16 ~110 MB → ideally < 60 MB) without audible quality loss. **Only executed if Phase 5 shows RTF headroom (RTF < 0.6 on M4)** — otherwise quantization often regresses perf on ANE due to mixed-precision boundary cost (per parakeet research).

### Scope in (ordered from safest to most aggressive; stop at first acceptable result)

1. **Palettize FlowLM transformer weights to 6-bit per_grouped_channel, group_size=16.** CoreML API: `ct.optimize.coreml.palettize_weights(mlmodel, PalettizationConfig(mode="kmeans", nbits=6, granularity="per_grouped_channel", group_size=16))`. This is the "sweet spot" from parakeet OPTIMIZATIONS.md. Target: 90 MB → ~35 MB for the main transformer.
2. **Palettize mimi_decoder to 8-bit** (keep Mimi conservative per parakeet's "decoder at fp16 only" guidance — but TTS decoder != ASR decoder; we experiment). 20 MB → ~10 MB.
3. **Keep `flow_lm_flow` and `text_conditioner` at fp16.** They're small; not worth the risk.
4. **Try 4-bit PGC on FlowLM** (per_grouped_channel with group_size=16, *not* enable_per_channel_scale). Only if 6-bit degrades too little — we want to hit 25 MB. Quality gate: perceptual listen-test + WER via openai-whisper on 100 generations of the oracle fixture; WER delta ≤ 1 pp.
5. **Never quantize below int8 on mimi_decoder.** SEANet+continuous-latent decoder is the most quality-sensitive part; parakeet's finding that "decoder at fp16 only" holds a fortiori here.

### Scope out
- Int8 activation quantization (W8A8). Research (`external_research.md §B.2`) flags 2-7pp WER cost without huge calibration corpora — not worth it.
- `.mlpackage` file merging (combining all 5 into one). Not beneficial; the split is deliberate.
- Arch changes (stride-6 decomposition, re-training). Out of scope for v1; revisit if v2 needs iPhone SE / older-SoC support.

### Verification criteria
- **Quality gate:** a 100-sample held-out prompt set, WER via whisper-large-v3 vs fp16 baseline, delta ≤ 1 pp. Plus a manual listen test on 10 samples scoring ≥ 4/5 MOS relative to fp16 (A/B blind).
- **Size gate:** target total bundle ≤ 60 MB (from ~130 MB fp16, assuming embeddings and voice safetensors excluded).
- **Perf gate:** RTF does not regress by more than 10% vs fp16 (quantization may slow down if scheduler can't fuse the dequant path).

### Risk flags
- **Palettized weights silently CPU-fall-back on certain group sizes.** The `enable_per_channel_scale` flag combined with palettization is a CoreML-LLM-documented trap: 492 ops go to CPU. We explicitly avoid it (use PGC alone). Verify via MLComputePlan post-palettization.
- **Dequantization fusing on ANE is compiler-version-sensitive.** coremltools 8.1 may differ from 8.2; pin exactly.
- **Mimi decoder palettization may introduce audible artifacts.** Speech synthesis is more ear-sensitive than ASR. Mitigation: ABX listen tests; keep fp16 Mimi as the fallback.

---

## Phase 7 — Streaming API + background-execution wrapper

### Goal
- The public Swift API surface a consuming iOS app uses. Background-audio-session plumbing, chunked generation, cancellation, multi-utterance queueing.

### Scope in
- **Background audio session configuration** documented: app must declare `UIBackgroundModes: [audio]` in Info.plist, set `AVAudioSession.Category = .playback, mode = .spokenAudio`, and activate before first generation. Our package provides a convenience helper:
  ```
  public extension PocketTTS {
    func configureBackgroundAudio() throws  // sets AVAudioSession for TTS
  }
  ```
- **Chunked generation:** `generate(text:)` already streams internally via AsyncChannel; the public API now exposes a continuation that can be `cancel()`-ed mid-generation. Cancellation resets all `.mlpackage` states before the next utterance.
- **Multi-utterance queue:** a serial executor inside `PocketTTS` actor ensures utterances are processed FIFO. Each utterance resets KV cache (new `MLMultiArray`s reused from the pool).
- **Warmup:** `await tts.warmup()` pre-runs dummy predictions for each of the 5 submodels to prime the ANE compiler cache. Call this at app start.
- **Progress callback:** `generate(text:, onProgress: ((Double) -> Void)?)` where progress is `tokens_generated / estimated_total_tokens`.
- **Voice switching:** `await tts.setVoice(name)` swaps voice embeddings; cheap (state copy into pre-alloc buffers).
- **Error taxonomy:** `enum PocketTTSError { case modelLoadFailed, tokenizationFailed, generationTimeout(afterFrames:Int), voiceNotFound, backgroundAudioUnavailable }`.

### Scope out
- (Previously deferred, now IN SCOPE per user decision) Voice cloning at runtime from a user-provided `.wav`. Implemented in Phase 4 (`VoiceCloner.swift`) and exposed in Phase 7 via `cloneVoice(from: URL) async throws -> VoiceHandle`.
- Long-form text (>10 min). The chunked 50-token architecture handles this but we haven't validated it end-to-end; explicit scope cutoff.
- Multi-language runtime switch. v1 ships English only; other languages require different `.mlpackage` files and embedding files.

### Concrete deliverables
- `PocketTTSCoreML/Sources/PocketTTSCoreML/PocketTTS.swift` — final public API.
- `PocketTTSCoreML/Examples/iOS/` — a minimal SwiftUI demo app demonstrating background audio playback.
- `docs/phase7_integration_guide.md` — how a consuming app hooks up TTS including Info.plist entries, AVAudioSession setup, typical error handling, memory footprint expectations.

### Verification criteria
- SwiftUI demo app runs on iPhone simulator + physical device (target iPhone 15 Pro).
- Background audio test: play a 60-second generation, background the app (home button), audio continues; `phys_footprint` stable.
- Cancellation test: start a 60-second generation, cancel at 10 seconds; no crash, no hang, subsequent generation works.
- Voice switch test: generate with voice A, switch to voice B mid-stream (in a new utterance); voice changes audibly.

### Risk flags
- **AVAudioSession activation fails in background-launch scenarios.** iOS restricts audio-session activation for apps launched directly into the background. Mitigation: document that first-use must be in foreground; fail gracefully with `.backgroundAudioUnavailable`.
- **Actor reentrancy under cancellation.** `await`-ing an async cancellation inside the actor while the AR loop task is running can deadlock. Mitigation: prefer `Task.cancel()` with cooperative cancellation checks every 10 frames in the AR loop.
- **ANE power budget in background.** Apple may throttle ANE more aggressively when the app is backgrounded. Monitor RTF in background via a telemetry hook; if it drops below 1.0, have a fallback that routes to `.cpuOnly` for background and re-routes to `.cpuAndNeuralEngine` on foreground.

---

## Cross-phase risks (not tied to a specific phase)

1. **Apple changes coremltools or iOS 18 runtime behavior mid-implementation.** We pin versions but a point release can still shift placement decisions. Mitigation: CI runs the full Phase-3 + Phase-5 suite on every coremltools minor version bump before upgrading.

2. **`split_into_best_sentences` (reference `tts_model.py:978`) port to Swift has subtle bugs.** Sentence boundary detection uses SentencePiece token-level lookahead that's easy to mis-port. Mitigation: golden dataset of 100 long-text inputs with expected chunk splits from the Python reference, validated in Swift.

3. **Distribution size for App Store.** With all voice embeddings (21 voices × ~5 MB each = ~100 MB) plus 5 `.mlpackage` files (~130 MB fp16) we're at ~230 MB per-language. Mitigation: ship with 3-5 core voices bundled; remainder downloaded on-demand from a CDN / GitHub Releases.

4. **Reference repo drift.** Our `git subtree` pin to v2.0.0; Kyutai ships v2.1 with arch changes (e.g. adds `layer_scale` to FlowLM). We'd have to re-run all golden dumps, re-patch, re-convert. Mitigation: treat subtree pulls as deliberate "upgrade" tasks, not routine.

5. **Licensing.** PocketTTS repo is MIT; paper weights under CC-BY-NC-SA (non-commercial). Our work is MIT for the conversion code but any bundled weights inherit CC-BY-NC-SA. **Explicit in README**: this is for research/open-source/non-commercial use. Commercial users must contact Kyutai for licensing.

6. **iPhone-primary RTF≤0.5 is aggressive.** A17 Pro ANE has ~35 TOPS (sustained), but thermal and background throttling can cut that to ~60% under sustained load. Our per-frame budget on iPhone is effectively ~50 ms (0.5 × 80 ms frame duration, allowing 1 frame lookahead). This is tight: if `flow_lm_main` at fp16 takes 30 ms and `mimi_decoder` takes 25 ms, we are already over. Mitigations in priority order:
   a. Confirm SDPA fusion on iOS18+ (Phase 3 gate).
   b. If fp16 misses the gate, Phase 6 palettization becomes critical path, not optional.
   c. If still missing, consider offloading `mimi_decoder` entirely to GPU (`.cpuAndGPU`) — still meets background-execution constraint if AsyncChannel back-pressure is correct.
   d. Last resort: reduce to `lsd_decode_steps=1` only (already default), drop N>1 variants to save ANE program cache pressure.

7. **Voice-cloning adds a cold-start cost on first generation per new voice.** User-recorded wav → mimi_encoder → voice KV prefill takes ~500 ms on iPhone (estimate; verify Phase 5). The Swift UX must show a "preparing voice..." indicator and the clone result should be cached on-disk so repeat uses of the same cloned voice skip the encoder.

---

## User decisions (resolved 2026-05-01; premortem to operate on these as constraints)

1. **Git subtree** for the reference repo. (default accepted)
2. **CI:** GitHub Actions with macOS runners for Phase 2/3; **Phase 5 iPhone benchmarking is manual-only.** (default accepted)
3. **Distribution:** GitHub Releases. (default accepted)
4. **Non-English:** deferred; no runtime language-switch wiring in Phase 7. (default accepted)
5. **Voice cloning at runtime: INCLUDED IN v1.** `mimi_encoder.mlpackage` is now a hot-path artifact, runs on a user-provided `.wav` to produce per-voice KV-cache prefill at runtime. Phase 4 adds a `cloneVoice(from: URL)` API; Phase 7 adds UX + documentation. Add ~1 week to Phase 4 and ~0.5 week to Phase 7.
6. **PRIMARY target hardware: iPhone (A17 Pro or newer).** This is a significant shift from the draft plan. Implications:
   - All Phase 5 gates are measured on iPhone as the ship-decision hardware. Mac is dev-only.
   - `MLComputePlan` inspection via a small iOS test app (Xcode project under `PocketTTSCoreML/Examples/ANEInspector/`), not macOS-only.
   - Powermetrics replaced by Xcode Instruments Energy Log for the iPhone ANE-power verification step.
   - ANE residency requirements tightened: A17 Pro ANE is more constrained than M-series on activation memory; any submodel < 80% ANE-resident is a Phase-5 blocker, not just a flag.
7. **RTF target: ≤ 0.5 on iPhone A17 Pro (primary).** Mac M-series is expected to be ≤ 0.3 as a sanity check but is not the ship gate. This is aggressive; see Risk-6 below.

### Consequences of decisions 5, 6, 7 on the plan

- **Phase 4 scope now includes:** `VoiceCloner.swift` that runs the **two-stage voice-cloning pipeline** matching reference `tts_model.py:886-899`:
  1. `mimi_encoder.mlpackage` on user audio (resampled to 24 kHz mono via `AVAudioConverter`) → continuous latents `[1, 32, T_latent]` where `T_latent = audio_samples / 1920`.
  2. Prepend `bos_before_voice` token, then drive `flow_lm_main.mlpackage` in a prefill loop for `T_latent + 1` steps, capturing the final KV cache state as the voice embedding.
  3. The voice bundle written to disk is the **KV cache snapshot**, not the encoder output. Format matches Kyutai's pre-exported `.safetensors` (same key schema — verified via Phase-1 golden fixture).
  **This is a single cohesive operation, not two separate models called independently.** The Swift `VoiceCloner.clone(from:) -> VoiceHandle` API hides the two-stage orchestration.
- **Phase 5 primary-target swap:** all success criteria in Phase 5 now reference iPhone A17 Pro numbers. `flow_lm_main` target remains ≥ 80% ANE / ≤ 30 ms p95; `mimi_decoder` target tightened to ≤ 25 ms p95 (not 40 ms) because the 80 ms frame budget on iPhone has less headroom than Mac due to thermal/background throttling.
- **Phase 6 may move onto the critical path.** If fp16 iPhone RTF lands in the 0.5-0.8 band, palettization becomes required not optional. We'll know after Phase 5.
- **Voice cloning quality gate in Phase 4:** clone-a-voice → generate → compare voice timbre vs Kyutai's pre-exported embedding of the same source voice. Use MFCC cosine similarity ≥ 0.85 as the numerical check; manual listen confirms identity.

### Remaining open (low-priority; can defer)

- `_24l` 24-layer English variant: out of scope for v1; revisit only if quality is insufficient.
- Voice-clone recording length: recommend 10-30s clips; enforce a 60s cap.

---

## Contradictions and caveats vs the research docs

One thing to surface — NOT a contradiction but a refinement of an external_research claim:

- `external_research.md §C.5` says "PocketTTS at fp16 is ~200 MB of weights." This is the paper-level figure including all non-English embeddings. The **per-language** English weights per HF API are ~219 MB fp32 → ~110 MB fp16. The Mimi shared portion is a subset. Actual English-only `.mlpackage` total (all 5 submodels, fp16, pre-palettization) is closer to **~130 MB** based on my per-submodel size estimates in Phase 3. Both numbers are "correct" at different granularities; I flagged this to avoid confusion in the premortem.

- `reference_codebase_map.md §4` correctly notes that `n_residual_layers=1` means **no dilated convs are actually used in the English SEANet decoder**, contradicting `external_research.md §C.1`'s concern about `dilation_base=2` forcing CPU fallback. The map doc is right; the external research was an inferred risk that turned out not to apply for English. Stride 6/5/4 remains the real CPU-fallback driver for `mimi_decoder` — the risk is narrower than the external research suggested.

- The reference repo name `/Users/sdesai/Tools/MLX/pocket-tts` is misleading — the code is PyTorch-only, per its own `pyproject.toml` (line 8 `torch>=2.5.0`, line 41-43 the `pytorch-cpu` uv index). No MLX code exists. Conversion pipeline is PyTorch → `torch.jit.trace` → `coremltools.convert`, no MLX intermediate step. This resolves the "conversion pipeline" open question in the brief.

---

## Estimated complexity (for scheduling, not a commitment)

| Phase | Effort (engineer-weeks) | Blocker for next phase? |
|-------|-------------------------|-------------------------|
| 1 | 0.5 | Yes (golden outputs needed for all later phases) |
| 2 | 1.5 | Yes (patched model is conversion-source-of-truth) |
| 3 | 2.0 | Yes (artifacts needed for Swift) |
| 4 | 1.5 | Yes (Swift runtime needed for RTF) |
| 5 | 1.0 | Partially (ship/fail decision; Phase 6 gated on result) |
| 6 | 0.5 — 1.5 (conditional) | No |
| 7 | 1.0 | No |
| **Total** | **~7.5 — 9 weeks** | |

Critical path is Phase 1 → 2 → **2.5** → 3 → 4 → 5. Phase 2.5 is a GO/NO-GO gate; RED outcome may terminate the project or force target renegotiation. Phases 6 and 7 are not blockers for "v1 works on a developer's laptop"; they are blockers for "v1 ships to App Store".

---

## Risk Mitigations (Pre-Mortem)

Pre-mortem run: 2026-05-01, deep mode, against this plan's pre-mitigation state.

### Tigers addressed (HIGH severity)
1. **Voice cloning requires gated weights, ungated checkpoint disables it** (tts_model.py:206, 872)
   - Mitigation: Assumption-9 rewritten to commit to `kyutai/pocket-tts` gated weights throughout. User provides `HF_TOKEN`. CI secret or nightly self-hosted runner.
   - Applied in: Assumption-9, Phase 1 risks.

2. **mimi_encoder alone is not a voice embedding; cloning needs encoder + flow_lm prefill**
   - Mitigation: Phase 4 `VoiceCloner.swift` scope rewritten as a two-stage pipeline matching tts_model.py:886-899. Phase 3.4 mimi_encoder description explicitly notes this.
   - Applied in: Phase 4 consequences block, Phase 3.4.

3. **iPhone A17 Pro RTF≤0.5 target has no empirical baseline**
   - Mitigation: Inserted **Phase 2.5 feasibility spike** — 3-day time-boxed GO/NO-GO gate. Measures actual `flow_lm_main` fp16 latency on iPhone before committing to Phase 3+.
   - Applied in: Phase 2.5 (new), critical path updated.

### Tigers addressed (MEDIUM severity)
4. **MLState fallback contradicts the "no MLState" TL;DR commitment**
   - Mitigation: Phase-2 landmine-1 strategy (b) now specifies **single**-state packing (`fp16[12, 1, S_cap, 16, 64]` combining all K+V layers) to avoid the dual-state ANE failure mode.
   - Applied in: Phase 2 landmine-1(b).

5. **fp16 softmax over boolean-masked padded positions can produce NaN**
   - Mitigation: Phase-2 mask rewrite now specifies **fp16 additive mask** with values 0.0/-65500.0. Phase 3.2 input signature updated. New Phase 3 verification gate: padded-128 vs. unpadded PSNR ≥ 50 dB.
   - Applied in: Phase 2 landmine-1, Phase 3.2 inputs, Phase 3 risk mitigation.

### Elephants noted
6. **Commercial/App Store licensing (CC-BY-NC-SA)**
   - User decision: **unknown — decide later.** Explicit deferral. Phase 7 distribution docs will block until resolved; Phases 1-6 proceed.
   - Action item parked: before Phase 7 gate, resolve (a) contact Kyutai for commercial terms, OR (b) commit to non-commercial scope in README.

### Paper tigers (noted, already mitigated)
- 5× mlpackage cold-start — `warmup()` in Phase 7 (line 478).
- KV=256 for long utterances — already handled by reference's own chunking (line 314).

### Accepted risks
- Phase 2.5 RED outcome (architecture-unviable-on-iPhone) may require scope renegotiation. Explicitly surfaced; user acknowledges.
- Gated-weights CI dependency on user-provided `HF_TOKEN` — if revoked/changed, CI breaks. Fallback: skip-marker + self-hosted nightly.

### Pre-mortem run summary
- Date: 2026-05-01
- Mode: deep
- Tigers found: 5 (3 HIGH, 2 MEDIUM)
- Elephants: 2 (1 unresolved, 1 mitigated)
- All HIGH tigers addressed via plan amendments above.
- Ready for kraken implementation, starting at Phase 1.
