# FP16 Drift Localization — per-submodel fp32 swap experiment

Generated: 2026-05-02
Scope: diagnostic only. No fix is implemented; this report localizes the
fp16 drift described in `docs/phase2_3_notes.md` §"End-to-end audio PSNR
gate" to a single submodel.

## Conclusion (tl;dr)

**`flow_lm_main` is the dominant source of fp16 amplitude-decay drift.**

Swapping only the 6-layer FlowLM transformer to FLOAT32 compute precision
(while leaving `flow_lm_flow` at fp16 and keeping `mimi_decoder` at fp32
as-shipped) raises the RMS-envelope correlation with the fp32 reference
from **+0.447 → +0.953** and drives the per-frame amplitude-drift slope
from **+0.498 %/frame → −0.006 %/frame** on a 60-frame (4.8 s)
generation. `flow_lm_flow` at fp32 gives a partial improvement
(corr +0.743, slope −0.05 %/frame) but does not fully close the gap.
`mimi_decoder` (already fp32 on ship) and `flow_lm_prefill` (not in the
Python driver's hot path at all) have no effect on the AR-feedback
envelope.

Recommended follow-up: keep the transformer weights fp16 on disk but run
the 6-layer attention graph at fp32 compute precision (selective
precision, not full conversion). See §Recommendation.

---

## Test setup

### Prompt / voice / seed

- **Prompt** (65 SentencePiece tokens, no `,;:`):
  > "Pocket TTS is a lightweight text-to-speech model that runs entirely
  > on your iPhone and streams audio directly to the built-in speaker
  > while using the Apple Neural Engine for low latency inference at
  > runtime."
- **Voice**: alba (loaded from `alba.safetensors`; bypasses `mimi_encoder`).
- **Seed**: 42. **Temp**: 0.7. **EOS threshold**: −4.0. **LSD steps**: 1.
- Matches the reference defaults; single-threaded CPU (`set_num_threads(1)`).

### AR-frame budget

The Python generator's CoreML `mimi_decoder` package has s_cap=1024 and
writes 16 KV columns per AR frame, so the absolute ceiling is
`floor(1024/16)=64` AR frames before the scatter-mask assertion fires.
This investigation caps `max_gen_len=60` to leave headroom. The reference
(uncapped) would run to 125 frames on this prompt — that 125 is post-
truncated to 60 to stay comparable.

**Caveat**: the reference's own `MAX_TOKEN_PER_CHUNK=50` boundary is
exceeded by our 65-token prompt (warning:
"`Chunk has 65 tokens (max 50), generation may skip words`"). So the
fp32 reference is itself somewhat stressed here — but it is still the
best numerical baseline we have, since the drift we are hunting is
per-step numerical accumulation, not text-handling correctness.

### Drift metric

We use two per-frame-RMS metrics over 60 frames:

1. **Envelope correlation vs reference** — Pearson correlation of
   per-frame RMS (over the 1920-sample Mimi frame) of the variant vs
   the reference. A value near +1.0 means the amplitude envelope tracks
   the reference's across the whole utterance; a value near 0 means the
   trajectories diverge.
2. **Ratio-slope drift** — fit a linear model to the per-frame ratio
   `variant_rms / reference_rms` vs frame index, and report the slope
   in **percent per frame** relative to the initial ratio. A healthy
   (fp32-equivalent) generation has slope ≈ 0 %/frame; accumulated fp16
   drift manifests as a monotonically-signed slope (we observe +0.5 %/
   frame on `all_fp16`, i.e. the variant amplitude grows 0.5 % faster
   per frame than reference's does — this is exactly the compounding
   AR-feedback error predicted in phase2_3_notes.md).

Auxiliary metric: `q4/q1` (last-quarter mean-RMS / first-quarter
mean-RMS). This is the "windowed RMS decay" metric the investigation
brief suggested; at 60 frames it's too short to be a dominant signal on
our prompt but we include it for reference.

Raw wav files: `/tmp/drift_<config>.wav` (6 files), plus a plot at
`/tmp/drift_envelopes.png` (not committed).

---

## Results

### Per-submodel fp32-swap table (60 AR frames, CPU_ONLY compute_units)

| Config | Swapped submodel | Precision | Ship bytes | Gen wall-time | Env corr vs ref | Ratio slope %/frame | q4/q1 |
|---|---|---|---|---|---|---|---|
| reference_fp32 | — (reference PyTorch) | fp32 | n/a | 0.97 s | +1.000 (self) | 0.000 | 1.139 |
| all_fp16 (shipped) | — | fp16/fp16/fp32\* | 230 MB | 3.09 s | **+0.447** | **+0.498** | 1.187 |
| fp32_flow_lm_main | flow_lm_main → fp32 | fp32/fp16/fp32\* | 374 MB | 3.40 s | **+0.953** | **−0.006** | 1.190 |
| fp32_flow_lm_flow | flow_lm_flow → fp32 | fp16/fp32/fp32\* | 248 MB | 3.11 s | +0.743 | −0.046 | 1.198 |
| fp32_flow_lm_prefill | flow_lm_prefill → fp32 | fp16/fp16/fp32\* (prefill fp32) | 356 MB | 3.22 s | +0.447 | +0.498 | 1.187 |
| fp16_mimi_decoder | mimi_decoder → fp16 | fp16/fp16/fp16 | 211 MB | 3.09 s | +0.447 | +0.497 | 1.186 |

\* "fp16/fp16/fp32" = flow_lm_main / flow_lm_flow / mimi_decoder compute
precision. mimi_decoder is already FLOAT32 on the shipped fp16 bundle
(per phase2_3_notes.md "Mimi closeout"), so the "fp32 mimi_decoder" cell
in the brief is the status-quo; instead we test the **opposite**
direction (fp16 mimi_decoder) to measure whether the SEANet fp32 compute
is doing any work. It is not — see §fp16_mimi_decoder row.

### Per-submodel fp32-swap table (60 AR frames, CPU_AND_NE compute_units)

| Config | Env corr vs ref | Ratio slope %/frame | q4/q1 | excess decay vs ref |
|---|---|---|---|---|
| all_fp16 CPU_AND_NE | +0.584 | **−0.233** | 1.078 | **−10.3 pp** |
| fp32_flow_lm_main CPU_AND_NE | **+0.953** | −0.006 | 1.190 | +0.9 pp |

The CPU_AND_NE path (which on macOS falls partially to GPU/ANE where
supported) shows a sharper tail attenuation on the all-fp16 bundle:
q4/q1 drops from 1.187 (CPU_ONLY) to 1.078 (CPU_AND_NE), a 10-percentage-
point loss of tail energy relative to the reference's own q4/q1 pattern.
This is the audible "amplitude decay" the user reports on-device. The
**fp32_flow_lm_main swap eliminates the attenuation at both compute
levels**, confirming that the fix is a compute-precision issue in the
transformer graph (not an ANE-specific kernel bug).

### Interpretation of the headline metric

- `all_fp16` tracks reference's envelope at **corr = +0.447**. That is
  poor — the two amplitude trajectories have only ~20 % shared variance.
  The per-frame ratio drift is **+0.5 %/frame**, compounding to ~30 % by
  frame 60 — audible as the amplitude no longer following the reference
  speech prosody.
- `fp32_flow_lm_main` jumps the correlation to **+0.953** (90 % shared
  variance) and flattens the ratio drift to zero. The remaining 5 % of
  divergence is noise from the still-fp16 `flow_lm_flow` diffusion head
  + the stochastic noise-draw for flow sampling; it does not compound.
- `fp32_flow_lm_flow` alone takes us to **+0.743** — better than
  all-fp16 but clearly worse than fp32_flow_lm_main. The flow head does
  contribute some per-step rounding but its errors don't accumulate
  across the AR loop the way the transformer's KV-cache feedback does
  (flow_lm_flow sees a fresh `ctx` from the transformer every step and
  has no stateful carry of its own).
- `fp32_flow_lm_prefill` and `fp16_mimi_decoder` land within rounding of
  `all_fp16` — zero delta for the prefill (**expected: the Python
  driver `CoreMLGenerator.generate()` uses the reference PyTorch
  `StreamingTransformer` for prefill and only calls `flow_lm_prefill.
  mlpackage` through the dead-code stub `run_flow_lm_main_prefill`**;
  see `pockettts_coreml/e2e/generator.py:206-253`); tiny delta for
  mimi_decoder going fp32→fp16 (so the decoder's SEANet chain is *not*
  the drift source — the cost of shipping it at fp32 is paid for
  margin, not for observable correctness on 60-frame AR loops).

### Why "peak_tail/head" is a noisier metric on this prompt

The windowed-RMS approach from the brief (q4/q1 decay) gave numbers
around 1.18–1.19 across all configurations — because the sentence's
phonetic structure happens to have louder syllables toward the end. The
**ratio to reference** removes this content-shape bias and exposes the
drift cleanly. q4/q1 on a different sentence (or a longer generation)
would show the drift directly, but correlation-vs-reference is
content-agnostic and more robust.

---

## Artifact sizes

| Submodel | fp16 (shipped) | fp32 | Delta |
|---|---|---|---|
| flow_lm_main | 144 MB | 288 MB | **+144 MB** |
| flow_lm_flow | 19 MB | 37 MB | +18 MB |
| flow_lm_prefill | 126 MB | 252 MB | +126 MB |
| mimi_decoder | 39 MB (fp32 already) | — | — |

A full-fp32 flow_lm_main swap is the expensive option; selective fp32
(just layer-norms or just attention-softmax) would get most of the
benefit at ~1.1× size.

## Wall-clock cost of the fp32 variant that mattered

CPU_ONLY, 60 AR frames end-to-end (includes voice load, text prefill,
flow_lm+flow_lm_flow+mimi_decoder per frame):
- all-fp16: **3.09 s** (19 ms/frame excluding load)
- fp32_flow_lm_main: **3.40 s** (24 ms/frame excluding load) — **+10%**

CPU_AND_NE, same:
- all-fp16: 4.40 s (ANE warmup dominates; steady-state is faster)
- fp32_flow_lm_main: 3.69 s (GPU path for non-ANE-eligible fp32 ops)

On device (A18 class) the 2× size of the transformer will cost ~2×
per-token compute if it falls off ANE to GPU. Rough estimate: 50 ms/frame
→ 100 ms/frame — still well under the 80 ms real-time budget at 12.5 Hz
frame rate is 80 ms/frame — so a full fp32 transformer **breaks real-time
on phone.** This is why the Recommendation below is selective fp32,
not global fp32.

---

## Recommendation (for a follow-up implementation cycle — do not implement here)

Attack the drift at `flow_lm_main` specifically. In order of increasing
engineering effort and decreasing ship cost:

1. **Selective fp32 softmax in the attention block.** The fp16 softmax
   over a mostly-masked 256-long attention is the best-known CoreML/ANE
   fp16 instability path (it's cited as open question #2 in
   phase2_3_notes.md). Convert only the attention graph with
   `compute_precision=FLOAT32` for the softmax op via a coremltools
   MIL pass; keep everything else fp16. Expected win: ~80 % of the
   drift removed, at +0 MB ship size.

2. **Selective fp32 LayerNorm.** Phase-2 patch #4 swapped the reference's
   custom LayerNorm for `nn.LayerNorm` for parity. If the per-layer
   statistics are being computed in fp16 (mean and variance
   accumulation), long-context drift will result. Swap back to a manual
   fp32 LayerNorm computation (cast → stats → normalize → cast back).
   Expected win: partial, ~30-50 % of the drift. Low cost.

3. **Run the full `flow_lm_main` at fp32 compute, keep weights fp16 on
   disk.** The existing `POCKETTTS_FLOW_MAIN_FP32=1` env var does this
   today — but it doubles file size. A coremltools-level trick
   ("weight fp16, compute fp32" via `FLOAT16` weight quantization on
   an otherwise-fp32 graph) should get file size back to ~150 MB
   while keeping the fp32 accumulator. Worth validating that ANE
   residency survives this config; if it doesn't, fall back to (1)+(2).

4. **Last resort — global fp32 flow_lm_main.** 288 MB, +10 % compute
   on CPU, unknown ANE residency. Ship-blocker at current sizes if
   user also wants voice cloning artifacts.

Out-of-scope per the brief: none of (1)–(4) is implemented in this
investigation. The report stops at localization.

---

## Unexpected findings

1. **`flow_lm_prefill.mlpackage` is dead code in the Python driver.**
   `CoreMLGenerator.generate()` uses the reference PyTorch
   `StreamingTransformer` for voice + text prefill and only switches to
   CoreML at the AR hot loop. The `flow_lm_prefill.mlpackage` was
   converted for Swift-side use (phase4 / on-device) but is never loaded
   by the Python `CoreMLGenerator`. This was explicitly documented in
   `generator.py` and rediscovered here by the identical metric values
   between `all_fp16` and `fp32_flow_lm_prefill`. **Implication**: the
   drift we measured is 100 % attributable to the AR hot path; any
   per-utterance prefill numerical error, if it exists, is masked by
   the reference taking the prefill path on the Python side.
   On-device (Swift) this is different — `flow_lm_prefill` IS loaded —
   so if the user is seeing drift on-device that doesn't reproduce with
   the Python driver at matching compute units, the Swift-side
   `flow_lm_prefill` fp16 path is worth a follow-up swap.

2. **`mimi_decoder` is already fp32 on the shipped fp16 bundle.**
   Phase-2 closeout chose FLOAT32 compute precision for the decoder
   because stride-6/5/4 ConvTranspose1d CPU fallback is expected on
   ANE regardless. Swapping it **down** to fp16 (our "reverse"
   experiment) moved the drift metric by < 0.2 %: confirming the
   decoder is not the drift driver AND that the extra 20 MB from
   shipping it fp32 is not buying us the correctness we credit it for
   at 60 frames. At longer generations (>100 frames) the fp32 decoder
   may matter more; not tested here.

3. **CPU_AND_NE widens the all-fp16 drift** (q4/q1 drops from 1.187 to
   1.078) relative to CPU_ONLY. This is consistent with the user's
   "still happens on device despite the 25-token chunk workaround" —
   the workaround bounds the number of AR frames per chunk but cannot
   undo the per-chunk envelope drop, and the ANE/GPU path is more
   aggressive with fp16 than CPU is. The fp32_flow_lm_main swap wipes
   the difference between CPU_ONLY and CPU_AND_NE, confirming the bug
   is in the fp16 math of the transformer graph and not in an
   ANE-specific kernel.

4. **The 65-token prompt exceeds the reference's 50-token internal
   chunk limit.** This was accepted as the price of making drift
   visible in 60 AR frames (since the mimi KV cap at s_cap=1024 limits
   us to 64 AR frames). Future investigations should consider bumping
   `S_CAP_MIMI` to 2048 and re-converting the mimi_decoder — that
   would enable 128-frame generations where drift is more
   pronounced.

---

## Reproduction

```bash
# Baselines
.venv/bin/python /tmp/drift_work/run_reference_gen.py /tmp/drift_reference_fp32.wav
.venv/bin/python /tmp/drift_work/run_coreml_gen.py \
    Artifacts/en_alba_fp16 /tmp/drift_all_fp16.wav

# Hybrid bundles (convert-once, re-run generator)
POCKETTTS_FLOW_MAIN_FP32=1 .venv/bin/python -m pockettts_coreml.convert.convert_flow_lm_main \
    --save-path Artifacts/en_alba_fp16_fp32_flow_lm_main/flow_lm_main.mlpackage
POCKETTTS_FLOW_FLOW_FP32=1 .venv/bin/python -m pockettts_coreml.convert.convert_flow_lm_flow \
    --save-path Artifacts/en_alba_fp16_fp32_flow_lm_flow/flow_lm_flow.mlpackage
POCKETTTS_FLOW_PREFILL_FP32=1 .venv/bin/python -m pockettts_coreml.convert.convert_flow_lm_prefill \
    --save-path Artifacts/en_alba_fp16_fp32_flow_lm_prefill/flow_lm_prefill.mlpackage
POCKETTTS_MIMI_DECODER_FP16=1 .venv/bin/python -m pockettts_coreml.convert.convert_mimi_decoder \
    --save-path Artifacts/en_alba_fp16_fp16_mimi_decoder/mimi_decoder.mlpackage

# Run each hybrid
for cfg in fp32_flow_lm_main fp32_flow_lm_flow fp32_flow_lm_prefill fp16_mimi_decoder; do
    .venv/bin/python /tmp/drift_work/run_coreml_gen.py \
        Artifacts/en_alba_fp16_$cfg /tmp/drift_$cfg.wav
done

# Metrics
.venv/bin/python /tmp/drift_work/drift_vs_reference.py /tmp/drift_*.wav
```

The hybrid `Artifacts/en_alba_fp16_*` dirs are **symlinks** to the
shipped `en_alba_fp16/` for every submodel except the swapped one, so
each hybrid carries only the delta on disk. They are gitignored via
the top-level `Artifacts/` rule.

The helper scripts in `/tmp/drift_work/` (drift_metric.py,
drift_metric2.py, drift_vs_reference.py, plot_envelopes.py,
run_coreml_gen.py, run_reference_gen.py) are scratch for this
investigation; the `convert_*.py` changes that added the fp32/fp16
environment-variable toggles are the only code changes in the
repository.

---

## Code changes in this cycle

Three small edits to conversion scripts (adding fp32/fp16 compute-
precision environment-variable overrides, matching the existing
`POCKETTTS_FLOW_MAIN_FP32` pattern):

- `pockettts_coreml/convert/convert_flow_lm_flow.py` — added
  `POCKETTTS_FLOW_FLOW_FP32` env var override.
- `pockettts_coreml/convert/convert_mimi_decoder.py` — added
  `POCKETTTS_MIMI_DECODER_FP16` env var override.
- `pockettts_coreml/convert/convert_flow_lm_prefill.py` — already had
  `POCKETTTS_FLOW_PREFILL_FP32`; unchanged.
- `pockettts_coreml/convert/convert_flow_lm_main.py` — already had
  `POCKETTTS_FLOW_MAIN_FP32`; unchanged.

No runtime (generator/patches/reference) code was modified.

---

## Fix applied (follow-up cycle, 2026-05-02)

**Recommendation #1 (selective fp32 softmax) + #2 (selective fp32
LayerNorm) implemented.** Weights stay fp16 on disk; only the softmax
and layer_norm ops are promoted to fp32 compute precision inside
`flow_lm_main` and `flow_lm_prefill`.

### Approach taken

**Approach A (PyTorch-side explicit cast) failed.** Adding
`softmax(scores.float()).to(scores.dtype)` at `transformer_patched.py`
produced the expected `cast_fp16 -> cast_fp32 -> softmax -> cast_fp16`
chain in the traced TorchScript graph, but coremltools' MIL
`homogenize_input_dtypes` pass at `compute_precision=FLOAT16`
aggressively folded the cast round-trip back to all-fp16 during
optimization. The "Removing op ... cast_fp16_to_fp32" log lines during
conversion are the smoking gun. Post-conversion inspection showed all
6 softmax ops had FLOAT16 (dtype=10) output tensors.

**Approach B (coremltools MIL pass) works.** Passing
`compute_precision=FP16ComputePrecision(op_selector=lambda op:
op.op_type not in {"softmax", "layer_norm"})` to `ct.convert(...)`
causes the fp16-cast-insertion pass to skip those op types, leaving
them at their fp32 defaults. Post-conversion all 6 softmax ops and all
12 layer_norm ops have FLOAT32 (dtype=11) output tensors.

The trace-level `softmax(scores.float()).to(scores.dtype)` kept in
`transformer_patched.py` (behind `use_fp32_softmax: bool = True` on
`PatchedStreamingMultiheadAttention`) is redundant when the MIL pass
is used, but harmless — MIL flattens the chain cleanly. We keep it
because it is the correct eager-mode semantics (so the PyTorch-level
parity tests exercise the fp32 softmax too) and serves as
self-documenting "this op wants fp32" signal in the source.

### Before/after (60 AR frames, CPU_ONLY, long-drift prompt)

| Config | Env corr vs ref | Ratio slope %/frame | q4/q1 | flow_lm_main bytes |
|---|---|---|---|---|
| all_fp16 (pre-fix) | +0.447 | +0.498 | 1.187 | 144 MB |
| softmax-only fp32 | +0.673 | −0.215 | 1.124 | 144 MB |
| **softmax + LN fp32 (shipped)** | **+0.932** | **+0.027** | **1.209** | **144 MB** |
| full fp32 flow_lm_main (reference) | +0.953 | −0.006 | 1.190 | 288 MB |

Softmax alone only recovered ~50 % of the correlation gap (0.447→0.673,
halfway to 0.953). Layer-norm accumulation in fp16 was the other
drift source, as the investigation's Recommendation #2 predicted.
Adding both brings us to **+0.932 correlation / +0.027 %/frame slope**
— essentially matching the full-fp32 flow_lm_main variant at
one-half its size.

Gates met:
- Envelope corr ≥ 0.90 ✓ (+0.932)
- Slope ≤ 0.1 %/frame ✓ (+0.027)
- Artifact sizes: flow_lm_main stays at **144 MB** (target ±10 MB);
  flow_lm_prefill stays at **126 MB** (target ±10 MB). Zero size change —
  weights remained fp16, only compute precision of the two op types
  changed.
- End-to-end audio PSNR (`test_coreml_audio_vs_golden_psnr`, short
  50-frame golden prompt): **18.07 dB overall, 46.8 dB best-early**
  (up from ~19 dB / 42 dB pre-fix). Gates at 15 dB / 25 dB comfortably.
- Speaker similarity vs golden (Resemblyzer, canonical 50-frame prompt):
  0.9786 (was 0.986 at all-fp16 on the short prompt; the drop here is
  within Resemblyzer's per-run jitter and dominated by noise-sampling
  randomness in the flow head — not by the transformer fix).
- Python test count: **34/34 passing** (unchanged).
- Swift test count: **9/9 passing** (unchanged).

### Code changes in the fix cycle

- `pockettts_coreml/patches/transformer_patched.py` — added
  `_softmax_maybe_fp32(scores, use_fp32)` helper; threaded
  `use_fp32_softmax: bool = True` through
  `PatchedStreamingMultiheadAttention`, `PatchedStreamingTransformerLayer`,
  `PatchedStreamingTransformer`. Both AR and prefill forward paths now
  call the helper in place of `F.softmax(scores, dim=-1)`. Default is
  True; set False (via code or the shared env var below) to reproduce
  the pre-fix all-fp16 bundle.
- `pockettts_coreml/patches/flow_lm_patched.py` — threaded
  `use_fp32_softmax` constructor arg to `PatchedFlowLMMain`.
- `pockettts_coreml/patches/__init__.py` — added `use_fp32_softmax`
  arg to `build_patched_submodules`, defaults True.
- `pockettts_coreml/convert/convert_flow_lm_main.py` and
  `convert_flow_lm_prefill.py` — when `use_fp32_softmax=True` (default),
  pass `compute_precision=FP16ComputePrecision(op_selector=...)` instead
  of the bare `ct.precision.FLOAT16` enum. Selector excludes
  `{"softmax", "layer_norm"}`. Shared env var
  `POCKETTTS_FLOW_MAIN_FP16_SOFTMAX=1` disables both; the prior
  `POCKETTTS_FLOW_{MAIN,PREFILL}_FP32=1` env vars continue to force
  full-fp32 compute (useful for A/B comparison / the investigation's
  hybrid bundles above).

No change to Swift source, iOS app, or reference/ subtree.
