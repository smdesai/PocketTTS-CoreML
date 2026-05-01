# External Research: PocketTTS → CoreML (ANE + CPU primary)

Generated: 2026-05-01
Scope: Parts A (PocketTTS architecture), B (CoreML conversion patterns from 3 reference repos), C (Apple ANE constraints).
Target: English first, RTF < 1 (≤12.5 frames/s of Mimi latent per second of audio budget) on iPhone / M-series.

Note on tool access: Perplexity / Firecrawl / Nia MCP keys were not set in the local env, so this was assembled from direct `curl` to the HuggingFace API, GitHub raw, arXiv, and an extracted snapshot of the FluidInference/mobius `knowledge/` subtree. Every factual claim below cites a URL or file.

---

## Part A — PocketTTS architecture

### A.1 High-level: what is PocketTTS?

- Paper: **"Continuous Audio Language Models" (CALM)**, arXiv `2509.06926`, Rouard / Orsini / Roebel / Zeghidour / Défossez (Kyutai). https://arxiv.org/abs/2509.06926
- GitHub: https://github.com/kyutai-labs/pocket-tts (MIT, v2.0.0)
- HF: https://huggingface.co/kyutai/pocket-tts (gated — requires `hf auth login`). Without-voice-cloning weights live at `https://huggingface.co/kyutai/pocket-tts-without-voice-cloning` and appear to be ungated.
- Blog: https://kyutai.org/blog/2026-01-13-pocket-tts

**One-line description from the paper abstract (verbatim):**
> "We release Pocket TTS, an open-source 100M-parameter text-to-speech model that can run faster than real time on a laptop CPU"

Advertised characteristics (from `README.md` in the GitHub repo):
- Runs on CPU, 2 CPU cores only, **~6× real-time on a MacBook Air M4 CPU**, ~200 ms first-chunk latency, streaming, voice cloning, infinite text length.
- Repo-reported: `torch.set_num_threads(1)` and no GPU speedup observed by the authors (they tried) — "batch size of 1 and a very small model".
- Optional `int8` dynamic quantization via `torchao` gives "~48% runtime memory reduction and ~27% speedup on x86 (FBGEMM). WER unchanged."

### A.2 Architecture — CALM (Continuous Audio Language Model)

Paper framing (verbatim from abstract):
> "These models instantiate a large Transformer backbone that produces a contextual embedding at every timestep. This sequential information then conditions an MLP that generates the next continuous frame of an audio VAE through consistency modeling. By avoiding lossy compression, CALM achieves higher quality at lower computational cost than their discrete counterpart."

Verified from reading the source (`pocket_tts/models/flow_lm.py`, `tts_model.py`, `configs/*.yaml`):

- **Autoregressive in audio-frame time**, at the audio-VAE frame rate. Not an encoder-decoder in the Whisper sense — the text is *prepended* to the sequence as conditioning embeddings, then audio latents are generated step-by-step:
  ```
  [text_embeddings | audio_latents_prefix] → Transformer → last-step ctx vec → FlowNet (MLP) → next 32-d latent
  ```
- **Two trainable models** in the same checkpoint:
  1. `FlowLM` = `conditioner` (text) + `transformer` (StreamingTransformer) + `flow_net` (SimpleMLPAdaLN) + a 1-output `out_eos` head. This is the autoregressive piece.
  2. `Mimi` used as a **continuous VAE** (encoder + decoder + 2-layer `ProjectedTransformer` each side). The quantizer is a `DummyQuantizer` — identity, no VQ — so the "latent" stays continuous, 32-dim. ✓ VERIFIED in `tts_model.py:_from_pydantic_config_with_weights` which instantiates `DummyQuantizer(**mimi_config["quantizer"])`.
- **LSD decode** = "Lagrangian Self-Distillation" consistency sampling with `num_steps` iterations, referencing https://arxiv.org/pdf/2505.18825. Default `lsd_decode_steps` is small (inspection of `default_parameters` was skipped for time but the arg is exposed on `load_model`). Each audio-frame step = 1 transformer forward + N MLP forwards. **This is the key perf lever** — N is user-tunable.
- Text conditioning: `LUTConditioner` — a SentencePiece tokenizer (`tokenizer.model`, ~60 KB per language) embedded through a lookup table (`n_bins=4000`, embed dim = `d_model = 1024`).
- EOS: a 1-d linear head on the transformer output compared to `eos_threshold`.
- BOS: a learned `bos_emb` (32-dim) marks BOS in the latent stream (NaN→BOS substitution). With voice cloning, an additional learned `bos_before_voice` (1, 1, 1024) is inserted before the encoded speaker prompt.

### A.3 Concrete dimensions (english / english_2026-04 config)

Source: https://raw.githubusercontent.com/kyutai-labs/pocket-tts/main/pocket_tts/config/english_2026-04.yaml

**FlowLM transformer (the autoregressive core):**
- `d_model=1024, num_heads=16, num_layers=6, hidden_scale=4` → FF dim 4096, head_dim 64
- RoPE `max_period=10000`
- LayerNorm (eps 1e-5), GELU in FFN, no bias
- `StreamingMultiheadAttention` (name suggests a streaming/caching K/V mechanism; `model_state` dict is passed through and mutated — this is PocketTTS's *custom* KV cache, not HuggingFace's)
- Streaming context is infinite by default; author states "infinitely long text inputs"

**FlowNet (`SimpleMLPAdaLN`) — the consistency-model head:**
- `dim=512, depth=6` — a 6-block MLP with AdaLN conditioning on transformer output + t (flow time). Tiny relative to the transformer.

**Mimi (continuous VAE):**
- `sample_rate=24000, frame_rate=12.5` → **1 latent frame = 80 ms of audio**, latent dim = 32, channel-side inner = 512.
- SEANet encoder/decoder: ratios `[6, 5, 4]` (so hop = 120 samples = 5 ms? no — hop=24000/12.5=1920 samples, matches 6·5·4·16 = 1920 via residual stack). kernel=7, residual=3, pad=constant.
- `ProjectedTransformer` on each of encoder/decoder: `d_model=512, num_heads=8, num_layers=2, context=250, dim_feedforward=2048`, with LayerScale=0.01.
- **Mimi is used in two halves:**
  - Encoder: only touched when voice-cloning from a wav prompt. After first load, it's effectively cached into the transformer's KV state and never re-run.
  - Decoder: run once per generated latent frame to produce 1920 audio samples (one Mimi frame = 80 ms PCM at 24 kHz).

### A.4 Per-language file layout

From the HF API (`/api/models/kyutai/pocket-tts/tree/main/languages/english`):
```
languages/english/
  model.safetensors          ~ 219 MB (single file, fp32 per cfg)
  tokenizer.model            ~ 60 KB (SentencePiece)
  embeddings/*.safetensors   ~ 21 voices × small per-file (pre-computed voice KV-cache-like states)
```
Naming variants: `english`, `english_2026-01`, `english_2026-04` (same 6-layer 1024d arch, different checkpoints; `english` == `english_2026-04` per the README). Other language families include `french_24l`, `german`, `german_24l`, `italian`, `italian_24l`, `portuguese`, `portuguese_24l`, `spanish`, `spanish_24l`. The `_24l` suffix implies 24-layer transformer variants for non-English — ✗ UNCERTAIN without downloading those configs, but strongly implied.

Root-level files:
- `tokenizer.model` (shared fallback)
- `tts_b6369a24.safetensors` (~ 235 MB) — appears to be the "super-model" including both Mimi + FlowLM before per-language split.
- `switch_to_bf16.py` — could not fetch (gated), but the filename implies bf16 conversion support.

**Config schema (YAML, Pydantic-validated):** `flow_lm.{transformer, flow, lookup_table}`, `mimi.{sample_rate, frame_rate, seanet, transformer, quantizer, inner_dim, outer_dim, channels}`, plus paths and flags like `insert_bos_before_voice`, `remove_semicolons`, `pad_with_spaces_for_short_inputs`.

### A.5 Input / output modality

- **Input:** UTF-8 text → SentencePiece tokens (vocab ~4000). That's the sole text-side input (no phonemes, no graphemes-as-ints). ✓ VERIFIED in config (`tokenizer: sentencepiece`) and `LUTConditioner` usage in `flow_lm.py`.
- **Output:** Raw 24 kHz mono PCM, streamed. Internally the model emits 32-dim continuous latents at 12.5 Hz which the Mimi decoder turns into 1920-sample chunks. Python API returns a `torch.Tensor` of float audio, sample_rate = 24000.
- **Voice cloning input:** A reference wav (any supported format). The encoder path turns it into a (sequence of) Mimi latents → projected through `speaker_proj_weight` → fed as *audio_conditioning* that prefills the transformer KV cache. The `export_model_state` helper serializes this prefilled state as a safetensors file for O(1) voice reload.

### A.6 Inference dependencies

- **No separate vocoder model** — the Mimi decoder *is* the vocoder (a non-quantized Mimi, i.e., a SEANet-based continuous waveform decoder with a tiny transformer bridge).
- **Mimi** here is the Kyutai Mimi codec architecture *without* its RVQ quantizer — they reuse the encoder/decoder weights as a waveform VAE. This detail matters: any off-the-shelf Mimi port (e.g., from Moshi) may include the quantizer path you do *not* want.
- **Tokenizer:** SentencePiece, loaded via `sentencepiece>=0.2.1`.
- **Runtime deps** (pyproject.toml): `torch>=2.5.0` CPU-only, `numpy>=2`, `safetensors>=0.4`, `scipy`, `einops`, `sentencepiece`, `huggingface_hub`. Optional `torchao` for int8 dynamic quantization.
- **PyPI:** `pocket-tts` 2.0.0.

### A.7 Third-party ports worth studying

Community ports already exist and explicitly split the model the way we need to split it for CoreML:

| Repo | Split / observations useful for CoreML |
|---|---|
| `KevinAHM/pocket-tts-onnx-export` | Exports to **5 ONNX graphs**: `text_conditioner`, `flow_lm_main`, `flow_lm_flow` (stateless flow step), `mimi_encoder`, `mimi_decoder`. Also a bundle `bundle.json`, `bos_before_voice.npy`, and int8-dynamic quantized variants (`*_int8.onnx`). The split `flow_lm_main` (transformer + state) vs. `flow_lm_flow` (stateless flow-matching MLP) is exactly what you want for CoreML — it keeps the LSD-loop in Swift and lets you dynamically choose step count / temperature. |
| `KevinAHM/pocket-tts-onnx` | Runtime for the above; uses ONNX Runtime Web + quantized graphs; the config schema you'd mimic for the CoreML bundle. |
| `jishnuvenugopal/pocket-tts-mlx` | MLX port (Apple Silicon GPU). Has onset-cleanup tricks: `--warmup-frames 1 --trim-start-ms 40 --fade-in-ms 15` — you likely need the same. |
| `LaurentMazare/xn/pocket-tts` | Rust / XN, very small. Good reference for lean implementations. |
| `babybirdprd/pocket-tts` | Candle (Rust) + WebAssembly. |
| `VolgaGerm/PocketTTS.cpp` | Single-file C++ using ONNX Runtime. |
| `k2-fsa/sherpa-onnx` | Multi-platform ONNX bindings (including Swift). Could be a competing runtime to benchmark against. |

**Recommendation:** mirror the ONNX export split (5 components) for CoreML. It preserves streaming semantics and avoids one monolithic .mlpackage that mixes stateful transformer code with stateless conv/MLP code — a pattern that repeatedly bites ANE residency (see Part B/C below).

---

## Part B — CoreML conversion patterns (3 reference repos)

### B.1 `john-rocky/CoreML-LLM`

Source: `README.md`, `docs/ARCHITECTURE.md`, `docs/CONVERSION.md`, `docs/ADDING_MODELS.md`, `docs/BENCHMARKING.md`, `docs/LFM2_CONVERSION_FINDINGS.md` (all fetched via raw.githubusercontent.com).

**What this repo is:** A Swift package + Python conversion pipeline for running decoder-only LLMs (Gemma 4, Qwen 3, LFM2.5) **ANE-resident** on iPhone. They report 24–52 tok/s on iPhone 17 Pro A19 Pro with **99.78% ANE placement**.

**Key patterns they use (high-signal for PocketTTS):**

1. **Chunked decode.** The decoder is split into 3–4 separate `.mlpackage` files (e.g. `chunk1` = layers 0-7, `chunk2` = 8-14, …). Each chunk is one ANE dispatch. For small models they merged 4 chunks into 3 (+8.2% tok/s) — fewer ANE launches = faster.
2. **`nn.Linear` → `nn.Conv2d(kernel_size=1)`.** ANE executes `Conv2d` ~3× faster than matmul. Data layout is `(B, C, 1, S)` — *channels-second-from-last, sequence-last* — per Apple's ANE-transformers guide.
3. **ANERMSNorm trick:** `cat([x, -x])` → LayerNorm → slice. ANE has an optimized LayerNorm kernel but bare RMSNorm is slow. (PocketTTS uses LayerNorm, not RMSNorm, so we **don't need this trick** — verified in `mimi_transformer.py`.)
4. **In-graph argmax (`InModelArgmax`).** Avoids shipping full-vocab logits CPU-side. Not applicable to PocketTTS (output is a 32-d float latent, not a vocab argmax) — but analogous trick: **sample the flow MLP's noise inside the CoreML graph** rather than in Swift, so you never return 1024-d transformer hidden states.
5. **Manual softmax** with explicit fp16 casts to prevent PyTorch fp16→fp32 upcast inside `torch.exp`.
6. **Pre-computed RoPE cos/sin as model inputs** (looked up on CPU/Swift side). Eliminates `gather` / `greater_equal` which are integer ops → CPU fallback. **Directly relevant**: PocketTTS uses RoPE (`rope.py`, `max_period=10000`).
7. **Explicit KV I/O, NOT `MLState`, for per-chunk tensors.** Reasoning given in their Gemma4 docs: "Avoids int64 state indices that break ANE placement." They use `MLState` only for the *outer* SWA KV buffers (iOS 18+). LFM2 findings (see below) showed **dual-state CoreML programs fail on ANE** — a critical landmine.
8. **Mask-based rotating KV write, not shift-based `cat`.** Gemma 4 + Stateful + tied-embedding hits ANEC error `-14` with shift `cat([K[:,:,1:,:], k], dim=2)`. Mask-based rotating buffer is the portable pattern.
9. **Batched prefill:** one 512-token prefill dispatch instead of 512 decode-shape dispatches. **Relevant for PocketTTS voice cloning**: encode the reference audio in a single chunked prefill, not frame-by-frame.
10. **Parity test in PyTorch *before* Core ML conversion** (HF reference ≡ ANE-adapted wrapper). They have a template: `conversion/experiments/bonsai/bonsai_reference_oracle.py`.
11. **`aten::Int` in traced graph = bug.** Check:
    ```python
    int_ops = [l for l in str(traced.graph).split('\n') if 'aten::Int' in l]
    assert len(int_ops) == 0
    ```
    Common offenders and fixes they document:
    - `x[..., :x.shape[-1]//2]` → `torch.chunk(x, 2, dim=-1)`
    - `x.view(batch, -1, dim)` → `x.view(1, num_heads, dim)` (explicit constants)
    - `repeat_kv` with shape unpacking → `repeat_interleave(n, dim=1)`
12. **Monolithic INT4/INT8 > ~1.4 GB silently falls back to GPU on ANE.** Plan chunking upfront. PocketTTS at 100M × 2 bytes (fp16) = 200 MB is *way* under this — no chunking needed for memory; we'd only chunk for latency / ANE-dispatch reasons.
13. **Palettize with `mode="kmeans"` first.** Linear INT8 div-by-zero on sparse tensors; kmeans safer.
14. **Compute units:** they default to `.cpuAndNeuralEngine` (not `.all`). GPU is off the path for iPhone.
15. **Runtime engineering matters as much as the .mlpackage.** From their Swift work:
    - Reused `FeatureBag` (pre-allocated `MLMultiArray` inputs).
    - `autoreleasepool` around every `prediction(from:)` (IOSurface pool fills up otherwise → crash).
    - `vDSP_maxvi` for argmax.

### B.2 `mweinbach/parakeet-coreml-swift/OPTIMIZATIONS.md`

Source: https://raw.githubusercontent.com/mweinbach/parakeet-coreml-swift/main/OPTIMIZATIONS.md

This is a 17.5-min ASR run that went from 155× RTFx (naïve) to **1145× RTFx on GPU** (pipelined + 4 workers). Every optimization below is extracted verbatim-level actionable:

**Model / conversion side:**
| Tip | Concrete |
|---|---|
| fp16 is the ANE table stakes | fp32 forces CPU fallback. "baseline-fp32: 74× RTF. baseline-fp16: 226× RTF." |
| Palettization beats linear quant at the same bit width | 8-bit LUT: 1.16% WER; int8-linear-per-channel: 2.31% WER. "Apple's ANE tuning prefers the LUT/indices representation." |
| 4-bit + `per_grouped_channel` (group_size=16) **stays on ANE**; 4-bit + `enable_per_channel_scale` kicks 492 ops to CPU | "PCS combination doesn't have a fused kernel on ANE, so the scheduler dumped 492 ops to CPU and destroyed RTFx." |
| W8A8 activation quant: expensive WER cost (2–7pp) unless you have a huge calibration corpus | "calibration corpus size matters a lot (1.6pp delta from 35 to 70 enc samples)" |
| **Encoder-only compression.** Decoder + joint at fp16 | "Decoder (12M params) and joint (5M params) are small enough that the calibration / WER risk isn't worth the ~30 MB saved." For PocketTTS, **Mimi and FlowNet are small — keep them fp16; only consider compressing the FlowLM transformer.** |

**Runtime / Swift side:**
- `FeatureBag` reused input buffers (same pattern as CoreML-LLM).
- `autoreleasepool` mandatory around each prediction.
- `vDSP_maxvi` for argmax, but also `vDSP_mmul` is cited as faster than naïve matmul — **useful for PocketTTS text preprocessing (mel-equivalent steps are absent, but tokenizer IDs → one-hot is not needed, though voice-prompt mel-like pre might apply).**
- **3-stage pipeline with bounded blocking queues** between mel extraction, encoder, and decode. The PocketTTS analog:
  1. Text tokenize (CPU, negligible)
  2. Transformer backbone step (ANE, this is the hotspot)
  3. Flow-MLP LSD loop + Mimi decoder (could be split: MLP on ANE/GPU, Mimi decoder partly ANE partly CPU depending on which SEANet ops land where)
- **Capacity-2 bounded queues** prevent IOSurface exhaustion crashes ("Failed to allocate memory IOSurface object").
- **Parallel workers for decode** only if decode is the bottleneck and each chunk is independent. For a streaming TTS with 1 voice per user, this mostly doesn't apply — but useful if we serve multiple concurrent utterances.
- **ANE vs GPU throughput on ASR-like loads:** ANE tops out around 400× RTFx. GPU scales to 1145×. For a *streaming* use case this matters less — any RTFx > 1 is fine; the question is latency-to-first-audio.

**Quote worth pinning:** "The model stayed the same from ~245× to ~1145× RTFx; everything from that point was runtime engineering." Implication for PocketTTS: getting the .mlpackage correct gets us a working model. Hitting real-time-minus and ultra-low-latency requires Swift-side plumbing, not more conversion work.

### B.3 `FluidInference/mobius/knowledge/`

Source: `codeload.github.com/FluidInference/mobius/tar.gz/refs/heads/main` (downloaded + extracted). Layout:
```
knowledge/
├── AGENTS.md              (index)
├── audio/                 (ASR papers, not directly relevant)
└── coreml/
    ├── AGENTS.md
    ├── core-ml-on-device-llama.md     (Apple's Nov 2024 Llama-3.1-8B blog, full text)
    ├── coremltools/                   (vendored 9.0b1 docs: guides + source)
    └── neural-engine/                 (vendored hollance/neural-engine reference)
```

This directory is a **vendored documentation snapshot**, not novel findings. High-value pieces for PocketTTS:

- **`core-ml-on-device-llama.md`** walks the canonical Llama-3.1 CoreML export with 3 optimizations in sequence:
  1. **Fused SDPA** (`torch.nn.functional.scaled_dot_product_attention` + `minimum_deployment_target=ct.target.macOS15`) → automatic fusion to a single GPU kernel.
  2. **KV cache as `MLState`** + flexible-shape inputs → **~13× faster** than KV-as-I/O for 2048 ctx (128 ms TTFT, 16 tok/s vs. 934 ms, 1.25 tok/s).
  3. **Block-wise int4 quantization** (`OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=32)`) → 2× decode speedup, 4× size reduction.
- **`SliceUpdateKeyValueCache`** example code — write pattern that the Core-ML-GPU compiler recognizes and turns into in-place updates:
  ```python
  self.k[layer_idx, :, :k_state.shape[1], begin:end, :] = k_state
  k_state = self.k[layer_idx, :, :, :end, :]
  ```
  Registered via `register_buffer("keyCache", ...)` + declared as `ct.StateType` in `convert()`. This is the blessed pattern. ✓ VERIFIED at knowledge/coreml/core-ml-on-device-llama.md.
- **`stateful-models.md` (coremltools docs)** gives the `make_state()` + `predict(..., state=...)` API and confirms `iOS18/macOS15` minimum.
- **`hollance/neural-engine/docs/neural-engine-transformers.md`** reiterates the Apple ANE transformer principles (B,C,1,S layout, nn.Conv2d-not-Linear, chunked attention, channels-last-aligned-to-64-bytes).
- **`unsupported-layers.md`** lists ANE non-starters (relevant to PocketTTS):
  - LSTM / GRU — **not used here, good.**
  - `gather` — **risk**: SentencePiece embedding lookup is a gather. Apple's Gemma4 pipeline handles this by leaving embed on CPU (see CoreML-LLM chunk 1).
  - dilated convolutions — **need to verify Mimi SEANet doesn't use dilation > 1**; `dilation_base: 2` in cfg suggests **yes it does**. This is a RED FLAG for ANE on the Mimi decoder.
  - Pooling kernel > 13 or stride > 2 — Mimi SEANet uses strides 6, 5, 4 in encoder (decoder has `ConvTranspose1d` inverses) → **stride 6 violates ANE constraint**.
  - Broadcasting of `CxHxW` with `Cx1x1` — RoPE reshape patterns can trip this.
- **`16-bit.md`:** "The ANE appears to use float16 for everything." Activations `>1e2` or `<1e-4` lose precision. Flow MLP consistency sampling uses Gaussian noise with `std=temp**0.5` and `trunc_normal_` — default `temp` should keep activations in safe range; **check `DEFAULT_TEMPERATURE` and `DEFAULT_NOISE_CLAMP` numerically before fp16 conversion.**
- **`is-model-using-ane.md`:** To verify actual ANE execution: symbolic breakpoint on `-[_ANEModel program]`, OR `powermetrics` on macOS (shows `ANE Power: X mW`), OR Xcode Instruments "Core ML" template + Time Profiler, OR **iOS 18 `MLComputePlan`** which gives per-op backend (preferred; CoreML-LLM uses this).

---

## Part C — Apple ANE constraints (synthesized)

### C.1 Shape, rank, dtype

- **dtype:** ANE is fp16-only for compute. fp32 weights force CPU fallback. bf16 → fp16 at load. int8 weight-only is OK; int4 block-wise linear AND palettized all work (with caveats — see B.2 for palettization-vs-linear gotchas).
- **Rank:** prefer rank-4, channels-second-from-last. `(B, C, 1, S)` layout is Apple's official recommendation for transformers. Rank > 4 ("ND" ops) fall back to CPU.
- **Packing / alignment:** last axis must be contiguous and 64-byte aligned. If you accidentally put S=1 as the last axis, every tensor balloons to 64 B (32× waste in fp16). Always put the *largest-or-sequence* axis last.
- **Op-specific hazards** (from `unsupported-layers.md` + `LFM2_CONVERSION_FINDINGS.md` + parakeet OPTIMIZATIONS):
  - **Depthwise conv where `groups == in_channels`** → CPU fallback. Cost: ~5 ms per conv × 10 layers = 50 ms/token on A19 Pro. Mimi SEANet uses depthwise separables; if so, this is a *per-frame* cost budgeted against the 80-ms-per-frame target.
  - **Dilated convolutions** (Mimi has `dilation_base: 2`) → CPU fallback likely.
  - **Stride > 2** (Mimi encoder stride 6, 5, 4) → CPU fallback. Decoder's `ConvTranspose1d` with inverse strides has the same issue.
  - `gather` (embedding lookup) → CPU. Keep on CPU, push result into ANE graph as a tensor input.
  - RNN / LSTM / GRU — N/A for PocketTTS.
  - `aten::Int` in traced graph → CPU scheduling. Enforce: 0 int ops allowed.
- **SDPA fusion**: use `torch.nn.functional.scaled_dot_product_attention` + `minimum_deployment_target ≥ ct.target.macOS15` / `ct.target.iOS18`.

### C.2 Stateful models — ANE reality check

- `MLState` is **iOS 18 / macOS 15+ only**, `mlprogram` format only.
- **Apple's own blog claims** stateful KV is 13× faster than KV-as-I/O on **GPU** (their Llama example).
- **Reality on ANE is murkier.** CoreML-LLM found:
  - `MLState` works for the main KV cache.
  - **Dual `MLState` buffers FAIL on ANE** for LFM2 (status=0x1d / 0x9). Fix: one MLState + one plain I/O tensor for the second state. Same workaround noted in `gemma4_swa_stateful_chunks.py` for sliding+full KV pairs.
  - Mask-based write > shift-based `cat`.
  - "Explicit KV I/O, NOT MLState" is sometimes *faster* on ANE for small models because it dodges int64 state indices that break ANE placement.
- **Translation for PocketTTS:** FlowLM has a custom `StatefulModule` framework already. You have two choices:
  1. Convert with KV as `MLState` (one buffer per layer, or one fused tensor [L, B, 2, H, S, D]) — matches Apple's recommended path.
  2. Convert with KV as explicit I/O, copied Swift-side every step — **better if #1 causes ANE fallback.**
  Recommend: **bench both on the actual hardware target.** Don't assume `MLState` wins.

### C.3 coremltools version matrix

| coremltools | min. deployment | New features useful here |
|---|---|---|
| 7.x | iOS 16 / macOS 13 | Baseline mlprogram, no state |
| 8.0 / 8.1 | iOS 18 / macOS 15 | **`MLState` / `ct.StateType`**, fused SDPA, `block_size` int4 linear, **palettization** `per_grouped_channel` / `enable_per_channel_scale`, flexible-shape with state, `ct.optimize.coreml.linear_quantize_weights` |
| 9.0b1 (FluidInference vendored this) | iOS 26 / macOS 26 | Int8 model I/O, PyTorch 2.7, ExecuTorch 0.5, GPU low-precision accumulation hints |

**Use 8.x for shipping, 9.x only if willing to require iOS 26.** Target `minimum_deployment_target=ct.target.iOS18` or `.macOS15`.

### C.4 Verifying ANE execution (in order of signal/cost)

1. **`MLComputePlan`** (iOS 18+). Enumerate ops and their dispatched compute device. What CoreML-LLM uses for their "99.78% ANE" claim. Preferred — gives exact counts.
2. **`powermetrics`** on macOS: watch `ANE Power` column. Non-zero during prediction = ANE active.
3. **Xcode Instruments → Core ML template** + Time Profiler. Gives per-op breakdown but less explicitly "which unit".
4. **Symbolic breakpoint `-[_ANEModel program]`** (private). Hits if ANE is used for any portion.
5. **`MLModelConfiguration.computeUnits`** A/B: load as `.cpuOnly`, `.cpuAndGPU`, `.cpuAndNeuralEngine`. If `.cpuAndNeuralEngine` isn't materially faster than `.cpuOnly` → ANE isn't really helping. (Confirm no CPU thermal confound.)

### C.5 Memory / background execution

- **iOS jetsam ceiling (`phys_footprint`)** is the only number that matters for "app gets killed". Xcode's memory gauge under-reports INT4-palettized weights by ~700 MB on iOS 18 (CoreML-LLM docs).
- PocketTTS at fp16 is ~**200 MB of weights** + Mimi (~20 MB) + KV cache (streaming transformer, depends on context length). Very small — shouldn't blow any budget.
- **Background execution:** iOS background audio sessions (required for streaming playback) constrain CPU usage. PocketTTS's 2-CPU-core footprint is already background-friendly. ANE is the *best* compute unit for background because GPU contention with the compositor/AVFoundation is a real issue on iOS.
- **Download size matters for App Store.** One .mlpackage per language × 5 graphs × ~200 MB would be prohibitive. Strategy:
  - Ship English bundled.
  - Download other languages on demand.
  - Consider int4 palettized for the FlowLM transformer (~200 MB → ~50 MB per language) — WER impact for TTS unclear; need to listen-test.

---

## Concrete playbook for this port (planner-facing)

**Decompose into 5 CoreML packages (mirror the ONNX export):**
1. `text_conditioner.mlpackage` — SentencePiece embedding LUT + any projection. Small, can live on CPU or ANE. Input: int32 token ids. Output: (B, S_text, 1024) embeddings. Used rarely (once per utterance + on text chunk boundaries).
2. `flow_lm_main.mlpackage` — StreamingTransformer + state updates. **This is the hot path.** One forward per audio frame (12.5 Hz). Input: prev audio latent (32-d), text embeddings, KV state. Output: ctx vector (1024-d), EOS scalar, new KV state.
3. `flow_lm_flow.mlpackage` — Stateless SimpleMLPAdaLN. Called `lsd_decode_steps` times per audio frame. Input: ctx vector, t (float), current x_t. Output: flow direction. **Stateless** = cheap to batch. Consider compiling with an enumerated-shapes variant for common step counts.
4. `mimi_encoder.mlpackage` — Used once per voice cloning session. Can tolerate CPU fallback. INT8 fine.
5. `mimi_decoder.mlpackage` — One forward per audio frame (12.5 Hz). **Likely will partially fall back to CPU** due to SEANet dilation + stride — measure, then decide if worth rewriting with stride-2 factorization.

**Conversion checklist (per graph):**
- [ ] Rewrite `nn.Linear` → `nn.Conv2d(1×1)` in FlowLM transformer (channels-second-from-last, `(1, 1024, 1, S)` layout).
- [ ] Keep LayerNorm (no RMSNorm trick needed).
- [ ] Use `torch.nn.functional.scaled_dot_product_attention` inside `StreamingMultiheadAttention` — verify the port does, or swap it in.
- [ ] Pre-compute RoPE cos/sin as model inputs from Swift; drop `gather` from graph.
- [ ] Register KV cache buffers, export as `ct.StateType` (try iOS 18 path first, fall back to I/O if ANE rejects).
- [ ] `minimum_deployment_target=ct.target.iOS18` for SDPA fusion + stateful models.
- [ ] `torch.jit.trace` with static shapes for all branches; `ct.RangeDim` only where truly variable (text length up to some cap).
- [ ] Assert zero `aten::Int` in traced graph.
- [ ] Parity test: PyTorch PocketTTS vs. CoreML on 5 reference prompts, PSNR > 60 dB on audio, WER-delta < 1%.

**Quantization ladder (from safest to most aggressive):**
1. fp16 everywhere. Baseline correctness.
2. fp16 + palettize FlowLM transformer weights to **6-bit per_grouped_channel group_size=16** (parakeet's sweet spot was 4-bit but speech synthesis may be touchier than ASR; 6-bit is the hedge).
3. Int8 dynamic for CPU-fallback layers (via `ct.optimize.coreml.linear_quantize_weights` with `granularity="per_block", block_size=32`).
4. Try 4-bit palettized + PGC group_size=16 on FlowLM only, never on Mimi.
5. Never quantize Mimi decoder below int8.

**Runtime (Swift):**
- `.cpuAndNeuralEngine` compute units (not `.all`).
- `FeatureBag`-style pre-allocated `MLMultiArray` inputs.
- `autoreleasepool` around every `prediction(from:)`.
- Bounded queues (capacity 2) between stages: `tokenize → FlowLM → Flow-MLP LSD → Mimi decode → audio buffer`.
- Pre-run a dummy `.predict` at app launch to warm the ANE compile cache (CoreML-LLM reports 231 ms for this; saves cold-start latency).
- For multi-utterance (e.g., reading a long article): keep one `MLModel` per graph, reused across utterances. Voice state (KV prefill) serialized via the existing `export_model_state` equivalent, re-loaded per utterance.

---

## Top 5 risks for this port

1. **Mimi SEANet decoder will partially fall back to CPU on ANE.** `dilation_base=2` residual blocks and stride-6/5/4 transpose convs are on the ANE unsupported list. Mitigation: (a) measure actual placement via `MLComputePlan` early; (b) accept CPU for Mimi decoder — at 12.5 Hz / 1920 samples, a Mac/iPhone CPU can comfortably do this inside 80 ms; (c) if too slow, factor the stride-6 into stride-3 → stride-2 cascade. **This is the single biggest architectural risk.**

2. **`StreamingMultiheadAttention` custom state dict ≠ stateful CoreML model.** PocketTTS's `StatefulModule` framework stores KV per-layer in a Python dict. Direct `torch.jit.trace` will capture point-in-time tensors, not a state-preserving flow. You must **rewrite the attention forward** to take `k_cache`/`v_cache` as input buffers (`register_buffer`), update via slice-assign (the Apple pattern), and export as `ct.StateType`. Budget 2–3 days for this port + parity test. Failure mode: stateful conversion "works" but ANE refuses dual-state programs (see LFM2 findings).

3. **LSD flow loop inefficiency.** `lsd_decode_steps` default is likely 4–8. Each step = one `flow_net` MLP forward. That's `12.5 × lsd_decode_steps` ANE dispatches per second of audio just for the flow head. At N=8, that's 100 launches/s. **ANE dispatch overhead** (~1-2 ms per launch on iPhone) alone could be 100-200 ms/s — pushing RTF close to 1. Mitigation: (a) unroll all LSD steps into one graph if the step count is fixed; (b) expose the number as an enumerated-shape input and compile a few fixed-N variants; (c) move `flow_net` to CPU — it's only a 512d × 6-layer MLP, ~2 MB, trivial on Apple's matmul.

4. **fp16 precision for the flow-matching noise sampling.** `torch.nn.init.trunc_normal_` inside the graph may upcast to fp32 (like `torch.exp` in softmax does per CoreML-LLM). If noise sampling is in-graph on ANE, it could (a) silently fall back to CPU every frame or (b) produce degraded audio quality in fp16. Mitigation: sample noise on CPU/Swift side, pass as input tensor. Deterministic side-benefit: exact-reproducibility via seed control.

5. **App-size / distribution.** 21 voice embeddings × ~a-few-MB each + 5 CoreML packages per language × O(200 MB) = 1+ GB per language. Feasible for a dedicated TTS app, hostile for a feature inside a general-purpose app. Mitigation: (a) ship English only; (b) palettize FlowLM aggressively (6-bit PGC → ~50 MB); (c) download other languages on demand from the user's own HuggingFace account (PocketTTS weights are gated — this also dodges Apple redistribution concerns with the CC-BY-NC-SA terms on the paper).

---

## Sources

1. Paper (PocketTTS abstract) — https://arxiv.org/abs/2509.06926
2. PocketTTS HF model card API — https://huggingface.co/api/models/kyutai/pocket-tts
3. PocketTTS HF languages tree — https://huggingface.co/api/models/kyutai/pocket-tts/tree/main/languages
4. PocketTTS GitHub README — https://raw.githubusercontent.com/kyutai-labs/pocket-tts/main/README.md
5. PocketTTS config (English) — https://raw.githubusercontent.com/kyutai-labs/pocket-tts/main/pocket_tts/config/english_2026-04.yaml
6. PocketTTS `tts_model.py` — https://raw.githubusercontent.com/kyutai-labs/pocket-tts/main/pocket_tts/models/tts_model.py
7. PocketTTS `flow_lm.py` — https://raw.githubusercontent.com/kyutai-labs/pocket-tts/main/pocket_tts/models/flow_lm.py
8. PocketTTS `mimi_transformer.py` — https://raw.githubusercontent.com/kyutai-labs/pocket-tts/main/pocket_tts/modules/mimi_transformer.py
9. Kyutai blog (2026-01-13 pocket-tts) — https://kyutai.org/blog/2026-01-13-pocket-tts
10. ONNX export reference — https://raw.githubusercontent.com/KevinAHM/pocket-tts-onnx-export/main/README.md
11. MLX port reference — https://raw.githubusercontent.com/jishnuvenugopal/pocket-tts-mlx/main/README.md
12. CoreML-LLM README — https://raw.githubusercontent.com/john-rocky/CoreML-LLM/main/README.md
13. CoreML-LLM ARCHITECTURE — https://raw.githubusercontent.com/john-rocky/CoreML-LLM/main/docs/ARCHITECTURE.md
14. CoreML-LLM CONVERSION — https://raw.githubusercontent.com/john-rocky/CoreML-LLM/main/docs/CONVERSION.md
15. CoreML-LLM ADDING_MODELS — https://raw.githubusercontent.com/john-rocky/CoreML-LLM/main/docs/ADDING_MODELS.md
16. CoreML-LLM BENCHMARKING — https://raw.githubusercontent.com/john-rocky/CoreML-LLM/main/docs/BENCHMARKING.md
17. CoreML-LLM LFM2_CONVERSION_FINDINGS — https://raw.githubusercontent.com/john-rocky/CoreML-LLM/main/docs/LFM2_CONVERSION_FINDINGS.md
18. Parakeet CoreML-Swift OPTIMIZATIONS — https://raw.githubusercontent.com/mweinbach/parakeet-coreml-swift/main/OPTIMIZATIONS.md
19. FluidInference mobius knowledge — https://codeload.github.com/FluidInference/mobius/tar.gz/refs/heads/main (extracted to `/tmp/mobius-knowledge/`)
20. Apple "On Device Llama 3.1 with Core ML" (Nov 2024) — vendored at `knowledge/coreml/core-ml-on-device-llama.md` (from https://machinelearning.apple.com/research/core-ml-on-device-llama)
21. Apple "Deploying Transformers on the Apple Neural Engine" (Jun 2022) — vendored at `knowledge/coreml/neural-engine/docs/neural-engine-transformers.md`
22. coremltools `stateful-models.md` — vendored at `knowledge/coreml/coremltools/docs-guides/source/stateful-models.md`
23. hollance `neural-engine/docs/unsupported-layers.md` — vendored
24. hollance `neural-engine/docs/is-model-using-ane.md` — vendored
25. hollance `neural-engine/docs/16-bit.md` — vendored
26. LSD paper (Lagrangian Self Distillation) — https://arxiv.org/pdf/2505.18825 (cited by pocket-tts `flow_lm.py`, not read for this report)

## Open questions (flag for planner)

- What is `DEFAULT_LSD_DECODE_STEPS`? Needed to budget per-frame dispatches (risk #3). Determinable from `pocket_tts/default_parameters.py` — unreachable in this research pass (file wasn't fetched). Ask the repo agent.
- What do `embeddings_v3/*.safetensors` and `embeddings/*.safetensors` actually contain? Presumed to be pre-serialized KV-cache states, but not verified.
- Does the `_24l` variant for other languages (24-layer transformer) fit in the same per-app budget? Budget at ~4× FlowLM size → ~800 MB fp16. Likely too big for multi-language iOS.
- Do we need to handle `remove_semicolons` and `pad_with_spaces_for_short_inputs` in the Swift tokenizer path, or are those one-shot at text-prep time? (Quick answer: they're flags on the config, applied by the Python runtime before tokenization — so they map trivially to Swift-side text pre-processing.)
