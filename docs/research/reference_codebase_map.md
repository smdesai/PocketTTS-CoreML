# Reference Codebase Map: pocket-tts (MLX/Python reference implementation)

Generated: 2026-05-01  
Source: `/Users/sdesai/Tools/MLX/pocket-tts`  
Version: v2.0.0 (tag), commit on `main`  
All claims VERIFIED by reading the actual source files.

---

## 0. Quick orientation

The repo is a single Python package `pocket_tts/` with this layout:

```
pocket_tts/
  __main__.py             → CLI entry (delegates to main.py:cli_app)
  main.py                 → CLI commands: generate, serve, export_voice
  default_parameters.py   → All numeric defaults (LSD steps, temperature, etc.)
  quantization.py         → torchao / torch.ao int8 dynamic quant helpers
  conditioners/
    base.py               → BaseConditioner, TokenizedText namedtuple
    text.py               → SentencePieceTokenizer, LUTConditioner
  data/
    audio.py              → audio_read, StreamingWAVWriter, stream_audio_chunks
    audio_utils.py        → convert_audio (scipy.signal.resample_poly)
  models/
    flow_lm.py            → FlowLMModel (conditioner + transformer + flow_net + EOS head)
    mimi.py               → MimiModel (encoder + decoder + quantizer + resamplers)
    tts_model.py          → TTSModel (top-level: load, generate, voice-clone)
  modules/
    conv.py               → StreamingConv1d, StreamingConvTranspose1d
    dummy_quantizer.py    → DummyQuantizer (identity projection, no VQ)
    layer_scale.py        → LayerScale
    mimi_transformer.py   → StreamingTransformerLayer, StreamingTransformer, ProjectedTransformer
    mlp.py                → SimpleMLPAdaLN, ResBlock, FinalLayer, TimestepEmbedder, RMSNorm
    resample.py           → ConvDownsample1d, ConvTrUpsample1d
    rope.py               → RotaryEmbedding, apply_rope
    seanet.py             → SEANetEncoder, SEANetDecoder, SEANetResnetBlock
    stateful_module.py    → StatefulModule ABC, init_states(), increment_steps()
    transformer.py        → StreamingMultiheadAttention, _LinearKVCacheBackend
  utils/
    config.py             → Pydantic Config / FlowLMConfig / MimiConfig + load_config()
    weights_loading.py    → get_flow_lm_state_dict, get_mimi_state_dict
  config/
    english.yaml          → English model config (same weights as english_2026-04)
    english_2026-04.yaml  → Same architecture, different HF hash
    english_2026-01.yaml  → Earlier checkpoint, same arch
    french_24l.yaml, german_24l.yaml, ... → 24-layer variants (different arch)
```

---

## 1. Text Conditioner

### Location

- `pocket_tts/conditioners/text.py` — `SentencePieceTokenizer`, `LUTConditioner`
- `pocket_tts/conditioners/base.py` — `BaseConditioner`, `TokenizedText`
- Config fields: `flow_lm.lookup_table.*` in the YAML

### Forward-pass signature

**SentencePieceTokenizer.__call__**
```
text: str → TokenizedText(tokens: Tensor[1, S_text], dtype=int64)
```
- Calls `sentencepiece.SentencePieceProcessor.encode(text, out_type=int)` then wraps in a 1-dim batch.
- Pure Python / CPU. Not a `nn.Module`.

**LUTConditioner.forward** (via `BaseConditioner.forward`)
```
inputs: TokenizedText(tokens: int64[1, S_text])
  → text_embeddings: float32[1, S_text, 1024]
```
- `self.embed = nn.Embedding(4001, 1024)` — vocab = 4000 + 1 padding token.
- Note: `output_dim == dim == 1024`, so there is NO output projection (`force_linear=True` is the base-class default but the embed dim already matches d_model). The embedded output goes directly into the transformer backbone.
- English tokenizer path: `hf://kyutai/pocket-tts-without-voice-cloning/languages/english/tokenizer.model` (ungated).

**Text pre-processing before tokenization** (`tts_model.py:prepare_text_prompt` line 913)
- Strip, collapse whitespace/newlines.
- Optionally remove semicolons (`remove_semicolons` flag from config — `False` for English).
- Uppercase first character.
- Append `.` if last char is alphanumeric.
- Optionally prefix 8 spaces if word-count < 5 (`pad_with_spaces_for_short_inputs` — `False` for English 2026-04).
- Long text is split into sentence-boundary chunks of ≤ `MAX_TOKEN_PER_CHUNK = 50` tokens (`split_into_best_sentences` at line 978), splitting first on `.!?` boundaries, then on `,;:` for oversized sentences.

### State management

None. The conditioner is stateless. `nn.Embedding` weight is a parameter, not a buffer.

### ANE hazards

- `nn.Embedding` is a `gather` op → **CPU fallback on ANE**. This is expected and acceptable; embed once per chunk and pass the resulting `float32[1, S_text, 1024]` tensor into the transformer graph as a fixed input.
- No dynamic shapes within the op itself, but S_text varies per chunk (max 50 tokens).

### Non-tensor logic

- `prepare_text_prompt` normalises text in Python (no tensor ops).
- `split_into_best_sentences` tokenizes the whole string first, finds boundary tokens by scanning a Python list, then re-decodes segments with `sp.decode`. This is all CPU/Python; call it once before the CoreML pipeline starts.

---

## 2. FlowLM Main (StreamingTransformer)

### Location

- `pocket_tts/models/flow_lm.py` — `FlowLMModel` (the shell that ties conditioner + transformer + flow_net + EOS head)
- `pocket_tts/modules/mimi_transformer.py` — `StreamingTransformerLayer`, `StreamingTransformer`
- `pocket_tts/modules/transformer.py` — `StreamingMultiheadAttention`, `_LinearKVCacheBackend`
- `pocket_tts/modules/rope.py` — `RotaryEmbedding`, `apply_rope`

### Concrete dimensions (English, from english.yaml / english_2026-04.yaml)

| Parameter | Value |
|-----------|-------|
| `d_model` | 1024 |
| `num_heads` | 16 |
| `num_layers` | 6 |
| `dim_feedforward` | 4096 (= 1024 × hidden_scale 4) |
| `head_dim` | 64 (= 1024 / 16) |
| `max_period` (RoPE) | 10000 |
| `context` | None (full causal attention, no sliding window) |
| `dtype` | float32 |
| Layer norm eps | 1e-5 |
| Activation | GELU |
| Bias | False on Linear1/Linear2; False on in_proj/out_proj |

### Forward-pass signature

**`FlowLMModel.forward`** (`flow_lm.py:95`)
```
sequence:         float32[1, 1, 32]     ← previous latent (or NaN for BOS)
text_embeddings:  float32[1, S_t+S_a, 1024]  ← text tokens + optional audio conditioning
model_state:      dict[str, dict]       ← mutable KV cache state
lsd_decode_steps: int
temp:             float
noise_clamp:      float | None
eos_threshold:    float
→ (next_latent: float32[1, 32], is_eos: bool[1, 1])
```

The internal call chain:
1. `torch.where(isnan(sequence), bos_emb, sequence)` — replaces NaN with learned BOS embedding.
2. `input_linear`: `Linear(32, 1024)` — project latent to model dim.
3. `backbone`: cat `[text_embeddings, input_]` along dim=1, then `StreamingTransformer(x, model_state)` → `out_norm(LayerNorm)` → slice last `sequence.shape[1]` tokens → `transformer_out: float32[1, 1, 1024]`.
4. Cast to float32. Take `[:, -1]` → `float32[1, 1024]`.
5. `out_eos: Linear(1024, 1)` → compare to `eos_threshold`.
6. Sample Gaussian/TruncNormal noise `float32[1, 32]`.
7. Dispatch to `lsd_decode` (Python loop over `num_steps`).

**`StreamingTransformerLayer.forward`** (`mimi_transformer.py:51`)
```
x: float32[1, 1, 1024], model_state → float32[1, 1, 1024]
```
Pre-norm: `norm1(x)` → MHA → residual + LayerScale. Pre-norm: `norm2(x)` → FFN (linear1 → GELU → linear2) → residual + LayerScale.

**`StreamingMultiheadAttention.forward`** (`transformer.py:128`)
```
query: float32[1, T, 1024], model_state → float32[1, T, 1024]
```
- `in_proj: Linear(1024, 3×1024)` → unpack Q/K/V each `[1, T, 16, 64]`.
- RoPE on Q, K.
- `_LinearKVCacheBackend.append_and_get`: writes K/V into `cache[2, 1, S_total, 16, 64]` at `offset`, returns full K/V up to offset+T.
- Build boolean attention mask from positions.
- `F.scaled_dot_product_attention(q, k_cache, v_cache, mask, dropout=0)`.
- `out_proj: Linear(1024, 1024)`.

### State management — KV cache

This is the most important detail for the CoreML port.

**`_LinearKVCacheBackend.init_state`** (`transformer.py:47`):
```python
dict(
  offset = torch.zeros(batch_size, dtype=torch.long),          # shape [B]
  cache  = torch.full((2, B, S_capacity, H, D), NaN, ...)      # shape [2, 1, S, 16, 64]
)
```
- `cache[0]` = K values, `cache[1]` = V values.
- `offset` is the **absolute time index** of the next write position.

**`complete_kv`** (`transformer.py:9`):
```python
cache[0, :, offset:offset+T] = k   # in-place write
cache[1, :, offset:offset+T] = v
return cache[:, :, :offset+T]       # slice valid portion
```
This is a **direct in-place mutation** plus a dynamic slice. Both operations fight CoreML:
- The in-place write `cache[0, :, offset:offset+T] = k` uses a Python integer extracted from the tensor via `.item()` (line 14: `offset_value = int(offset.view(-1)[0].item())`). This produces `aten::Int` in the traced graph.
- The slice `cache[:, :, :offset+T]` is also dynamic-length.

**`increment_step`** (`transformer.py:59`): `state["offset"] += increment` — called after every forward in `increment_steps()`.

**The capacity is pre-allocated** in `_expand_kv_cache` (`tts_model.py:390`). KV cache is expanded to `required_len = current_end + token_count + max_gen_len` before generation starts.

**KV cache shape at inference**: for a typical 50-token text + ~50 audio frames at 12.5 Hz: `[2, 1, ~120, 16, 64]` fp32 ≈ 240 KB per layer × 6 layers ≈ 1.4 MB total. Negligible.

**`StatefulModule.get_state`** (`stateful_module.py:42`): uses `self._module_absolute_name` (set post-load by iterating `named_modules`) to look up its slice from the global `model_state` dict. **This is the dispatch mechanism** — every `StreamingMultiheadAttention` fetches its own sub-dict by name.

### ANE hazards

1. **`aten::Int` from `int(offset.item())`** in `complete_kv` (line 14). This is the canonical blocker for tracing. Must be rewritten as a mask-based or slice-assign pattern before `torch.jit.trace`.
2. **Boolean attention mask** (`_build_attention_mask`, line 23): `pos_q[:, :, None] - pos_k[:, None, :]` + boolean `&` ops. The mask is `bool[1, 1, T_q, T_k]`. `F.scaled_dot_product_attention` accepts a boolean mask; under the hood it adds `-inf` where False. This should fuse on ANE with `minimum_deployment_target=iOS18` but the dynamic shape of `T_k` is a concern.
3. **RoPE**: `apply_rope` (`rope.py:6`) computes `torch.arange(T)`, `freqs = torch.exp(...)`, two multiplies, sin/cos, stack — **entirely within the forward pass with no pre-computed cache**. This generates `aten::arange` → integer control flow that can produce `aten::Int`. Must be pre-computed and passed as input.
4. **`torch.where` on NaN** (`flow_lm.py:121`): `torch.where(isnan(sequence), bos_emb, sequence)`. The `isnan` check is a boolean mask operation. At inference (after the first frame), sequence is never NaN, but the trace captures this branch. For CoreML export, replace with a static BOS path: on the first frame, pass `bos_emb.unsqueeze(0).unsqueeze(0)` directly as `sequence`; never pass NaN.
5. **Python loop** in `StreamingTransformer.forward` (6 `layer(x, model_state)` calls): fine for tracing — the loop unrolls.
6. **Python loop** in `lsd_decode` (`flow_lm.py:33`): `for i in range(num_steps)` — **must be unrolled** into the trace or run separately per step. See Flow Head section.
7. **Noise sampling in-graph**: `torch.nn.init.normal_` / `trunc_normal_` (`flow_lm.py:135-137`). These are in-place ops on a freshly allocated tensor and will be captured by trace as non-reproducible. For CoreML, sample noise on the Swift side and pass as an input tensor.
8. **LayerScale** (`layer_scale.py`): `scale * x` — a simple per-channel multiply. ANE-friendly.
9. **No `layer_scale` for the FlowLM transformer** (config has no `layer_scale` field in `FlowLMTransformerConfig`). The `StreamingTransformerLayer` passes `layer_scale=None` → `nn.Identity()` for both scale ops. No issue.

---

## 3. FlowLM Flow Head (SimpleMLPAdaLN)

### Location

- `pocket_tts/modules/mlp.py` — `SimpleMLPAdaLN`, `ResBlock`, `FinalLayer`, `TimestepEmbedder`, `RMSNorm`, `LayerNorm`, `modulate`

### Concrete dimensions (English)

| Parameter | Value |
|-----------|-------|
| `in_channels` | 32 (latent dim) |
| `model_channels` | 512 |
| `out_channels` | 32 |
| `cond_channels` | 1024 (transformer d_model) |
| `num_res_blocks` | 6 |
| `num_time_conds` | 2 (s and t, hardcoded in `from_pydantic_config` at mlp.py:183) |

### Forward-pass signature

**`SimpleMLPAdaLN.forward`** (`mlp.py:188`)
```
c: float32[1, 1024]   ← conditioning vector from transformer (last token output)
s: float32[1, 1]      ← start time (scalar per batch)
t: float32[1, 1]      ← end time (scalar per batch)
x: float32[1, 32]     ← current noisy latent x_t
→ float32[1, 32]      ← predicted flow direction u_t
```
Internal flow:
1. `input_proj: Linear(32, 512)` → `x: [1, 512]`.
2. Two `TimestepEmbedder` calls on `s` and `t` → each `[1, 512]` → average → `t_combined: [1, 512]`.
3. `cond_embed: Linear(1024, 512)` → `c: [1, 512]`.
4. `y = t_combined + c`: `[1, 512]` conditioning.
5. 6 × `ResBlock(x, y)`: LayerNorm → AdaLN modulate (shift/scale/gate from 3-way split of `adaLN_modulation(y)`) → Linear(512,512) → SiLU → Linear(512,512) → gate × h + x.
6. `FinalLayer(x, y)`: LayerNorm (no affine) → AdaLN modulate → `Linear(512, 32)`.

**`TimestepEmbedder.forward`** (`mlp.py:78`):
```
t: float32[1, 1]
  → sinusoidal encoding [1, 256]
  → Linear(256, 512) → SiLU → Linear(512, 512) → RMSNorm
  → float32[1, 512]
```
- Uses `self.freqs` (a pre-computed buffer of shape `[128]`).
- `torch.cat([torch.cos(args), torch.sin(args)], dim=-1)` — standard sinusoidal embedding.

**`RMSNorm.forward`** (`mlp.py:35`):
```python
var = eps + x.var(dim=-1, keepdim=True)
y = (x * (alpha * torch.rsqrt(var))).to(x_dtype)
```
Custom RMSNorm. Notably uses `x.var()` not `(x**2).mean()` — this is slightly non-standard but equivalent to the standard RMS formula when `eps` accounts for the Bessel correction difference. **ANE does not have a fused RMSNorm kernel.** Options: (a) replace with `x / x.norm(dim=-1, keepdim=True) * alpha` pattern; (b) use the `cat([x, -x]) → LayerNorm → slice` trick from CoreML-LLM (but that trick was designed for LayerNorm-as-RMSNorm and may not apply cleanly here); (c) just accept CPU fallback for this tiny op inside the flow net (it's tiny: 1 × 512-dim vec).

**`LayerNorm` in ResBlock** (`mlp.py:39`):
```python
mean = x.mean(dim=-1, keepdim=True)
var  = x.var(dim=-1, unbiased=False, keepdim=True)
x    = (x - mean) / sqrt(var + eps)
```
A custom re-implementation ("because the default one doesn't support jvp"). This will NOT benefit from ANE's optimized LayerNorm kernel unless replaced with `nn.LayerNorm`. Swap to `nn.LayerNorm` before export.

### Called from `lsd_decode` (`flow_lm.py:19-40`)

```python
def lsd_decode(v_t, x_0, num_steps):
    current = x_0
    for i in range(num_steps):
        s = i / num_steps
        t = (i + 1) / num_steps
        flow_dir = v_t(s * ones_like(x_0[..., :1]),
                       t * ones_like(x_0[..., :1]),
                       current)
        current += flow_dir / num_steps
    return current
```
- `v_t = partial(self.flow_net, transformer_out)` — the conditioning vector `c` is fixed across all LSD steps.
- `num_steps` = `DEFAULT_LSD_DECODE_STEPS = 1` (from `default_parameters.py:4`). **The default is 1 step.** This means normally the loop runs exactly once, making it effectively: `x_1 = x_0 + flow_net(c, 0.0, 1.0, x_0)`.
- At `num_steps=1`: `s=0.0, t=1.0` — constant scalars.
- For higher quality, users set `--lsd-decode-steps N` (tests use 2 and 5).

### State management

None — `SimpleMLPAdaLN` is stateless. No `StatefulModule`. Pure function: `(c, s, t, x) → u`.

### ANE hazards

1. **Custom `LayerNorm` in ResBlock**: will not fuse with ANE LayerNorm kernel. Replace with `nn.LayerNorm(channels, eps=1e-6)`.
2. **Custom `RMSNorm`**: `x.var()` → ANE CPU fallback. Replace or fuse manually.
3. **`chunk(3, dim=-1)`** in `ResBlock.adaLN_modulation` → split into 3 tensors. CoreML handles this as a slice; should be fine, but verify no `aten::Int` leakage.
4. **`sum(... for i in ...)` loop** in `TimestepEmbedder` combination: Python-level sum over 2 tensors → fine (constant loop, unrolls to one `add`).
5. At `num_steps=1`, the entire LSD loop collapses to one `flow_net` call — very clean for unrolling. For `N>1`, consider exporting N copies of the MLP (unrolled graph) vs. looping in Swift.
6. **`ones_like(x_0[..., :1])`**: produces a scalar-fill tensor of shape `[1, 1]`. If `s` and `t` are constant at `num_steps=1` (0.0 and 1.0), just pass literal constants.

---

## 4. Mimi Encoder + Decoder

### Location

- `pocket_tts/models/mimi.py` — `MimiModel`
- `pocket_tts/modules/seanet.py` — `SEANetEncoder`, `SEANetDecoder`, `SEANetResnetBlock`
- `pocket_tts/modules/conv.py` — `StreamingConv1d`, `StreamingConvTranspose1d`
- `pocket_tts/modules/mimi_transformer.py` — `ProjectedTransformer` (2 instances: encoder-side and decoder-side)
- `pocket_tts/modules/dummy_quantizer.py` — `DummyQuantizer`
- `pocket_tts/modules/resample.py` — `ConvDownsample1d`, `ConvTrUpsample1d`

### Concrete dimensions (English)

| Component | Detail |
|-----------|--------|
| SEANet ratios | `[6, 5, 4]` (stored reversed internally → `[4, 5, 6]` for encoder) |
| hop_length | 4 × 5 × 6 = 120 → encoder native frame rate = 24000/120 = 200 Hz |
| frame_rate | 12.5 Hz (the AR rate) |
| encoder_frame_rate | 24000 / hop_length (200 Hz) → downsample factor = 200/12.5 = 16× |
| n_filters | 64 |
| SEANet inner dim | 512 |
| dilation_base | 2 |
| n_residual_layers | 1 |
| residual_kernel_size | 3 |
| `inner_dim` | 32 (outer latent dim for FlowLM) |
| `outer_dim` | 512 |
| quantizer | `DummyQuantizer(dimension=32, output_dimension=512)` — a single `Conv1d(32, 512, 1)` |
| Mimi transformer | `d_model=512, num_heads=8, num_layers=2, context=250, dim_ff=2048, layer_scale=0.01` |

### Forward-pass signatures

**`MimiModel.decode_from_latent`** (`mimi.py:89`) — called once per generated audio frame:
```
latent: float32[1, 32, 1]    ← one latent frame (transposed from FlowLM output)
mimi_state: dict              ← streaming state for conv layers + upsample + transformer
→ audio: float32[1, 1, 1920] ← one frame of 24kHz PCM (= 80 ms)
```
Call chain:
1. `DummyQuantizer(latent)`: `Conv1d(32, 512, 1)` → `[1, 512, 1]`.
2. `_to_encoder_framerate`: `ConvTrUpsample1d(stride=16)` → `[1, 512, 16]`. This is a `StreamingConvTranspose1d(512, 512, kernel_size=32, stride=16, groups=512, bias=False)` — **depthwise** transposed conv (groups == in_channels == 512).
3. `decoder_transformer(emb, mimi_state)`: `ProjectedTransformer` → input `[1, 512, 16]`, transpose to `[1, 16, 512]`, 2-layer `StreamingTransformer` (d=512, h=8, context=250), output back to `[1, 512, 16]`.
4. `SEANetDecoder(z, mimi_state)`: upsample `[1, 512, 16]` → `[1, 1, 1920]` through transposed convs with ratios `[6, 5, 4]` and residual blocks.

**`MimiModel.encode_to_latent`** (`mimi.py:96`) — called only for voice cloning:
```
x: float32[1, 1, T]  ← mono 24kHz waveform (padded to frame_size multiple)
→ latent: float32[1, 32, T/120/16]  ← latents at 12.5 Hz
```
Stateless (passes `model_state=None` to SEANetEncoder and ProjectedTransformer).

### SEANet decoder architecture detail

`SEANetDecoder` with `ratios=[6,5,4], n_filters=64, n_residual_layers=1`:

```
StreamingConv1d(32→512, k=7)       [1, 512, 1]  → [1, 512, 1]
ELU
StreamingConvTranspose1d(512→256, k=12, stride=6)  → [1, 256, 6]
  SEANetResnetBlock(256): dilations=[1,1], k=[3,1]
ELU
StreamingConvTranspose1d(256→128, k=10, stride=5)  → [1, 128, 30]
  SEANetResnetBlock(128): dilations=[1,1], k=[3,1]
ELU
StreamingConvTranspose1d(128→64, k=8, stride=4)   → [1, 64, 120]
  SEANetResnetBlock(64): dilations=[1,1], k=[3,1]
ELU
StreamingConv1d(64→1, k=3)         → [1, 1, 120]
```
Note: `n_residual_layers=1` means ONE `SEANetResnetBlock` per ratio stage. `dilation_base=2` means residual dilations are `2^j` for j in `range(n_residual_layers)` = `[1]`. So **dilation=1 only** — the `dilation_base` only matters for `j>0`. With `n_residual_layers=1`, **no dilated convolutions are actually used in the English model**. This is a significant relief compared to the external research concern.

BUT: the `ratios` for `ConvTranspose1d` are 6, 5, 4 — **strides 6, 5, 4 violate the ANE stride ≤ 2 constraint.** These will fall back to CPU.

**Also critical**: `ConvTrUpsample1d` (the 12.5 Hz → 200 Hz bridge):
- `groups=dimension` → **depthwise transposed conv** (groups == in_channels == 512) → ANE CPU fallback for depthwise.

### State management — Mimi streaming

`StreamingConv1d` and `StreamingConvTranspose1d` are `StatefulModule` with:
- `StreamingConv1d.init_state`: `{previous: float32[B, C, kernel-stride], first: bool[B]}` — a left-context buffer.
- `StreamingConvTranspose1d.init_state`: `{partial: float32[B, C_out, K-S]}` — an overlap-add accumulation buffer.

All conv layers in the SEANet decoder maintain these rolling buffers in `mimi_state`. For the streaming path (one latent frame at a time), the input to `SEANetDecoder` is always `T=1` at the latent rate before upsampling.

`ProjectedTransformer` inside Mimi also has KV cache (same `_LinearKVCacheBackend` as the FlowLM transformer but with `context=250`). This limits attention to the 250 most recent frames.

### ANE hazards

1. **`ConvTranspose1d` with stride 6, 5, 4**: ANE constraint is stride ≤ 2. All three upsample stages fall to CPU. This is the largest expected CPU-fallback zone.
2. **Depthwise `ConvTrUpsample1d`** (stride=16, groups=512): double-hit — depthwise + large stride.
3. **`StreamingConv1d.forward`**: uses in-place state writes (`state["previous"][:] = ...`) and `torch.cat([state["previous"], x])`. The `torch.where(state["first"].view(-1,1,1), init, state["previous"])` is a conditional broadcast that needs verification. The `first` boolean flag is only relevant for `pad_mode="replicate"` — English uses `pad_mode="constant"` (`padmode: constant` in yaml), so the `replicate` branch is dead code.
4. **`StreamingConvTranspose1d.forward`**: `y[..., :PT] += layer_state` and `layer_state[:] = y[..., -PT:]` — in-place ops on state tensors. Standard overlap-add.
5. **Mimi transformer context=250**: unlike the FlowLM transformer (infinite context), the Mimi transformer has `context=250`, so `_build_attention_mask` uses `mask & (delta < context)`. This is an additional boolean op in the mask but otherwise the same structure.
6. **`DummyQuantizer`**: `Conv1d(32, 512, kernel_size=1)` — ANE-friendly pointwise conv.

---

## 5. Streaming MHA with KV Cache (cross-cutting)

The same `StreamingMultiheadAttention` / `_LinearKVCacheBackend` infrastructure is used by both the FlowLM transformer (6 layers, no context limit) and the Mimi `ProjectedTransformer` (2 layers, context=250 per transformer).

This section captures the details critical for CoreML state handling.

### State schema (per attention layer)

```
{
  "offset": int64[B],               # absolute write position / RoPE offset
  "cache":  float32[2, B, S, H, D]  # S=capacity, H=num_heads, D=dim_per_head
}
```

The model_state dict keyed by `_module_absolute_name` (set after model load in `tts_model.py:224`). Example keys for FlowLM:
```
"transformer.layers.0.self_attn" → {offset, cache}
"transformer.layers.1.self_attn" → {offset, cache}
... (×6 layers)
```
And for mimi decode:
```
"upsample.convtr"         → {partial}
"decoder_transformer.transformer.layers.0.self_attn" → {offset, cache}
"decoder_transformer.transformer.layers.1.self_attn" → {offset, cache}
"decoder.model.1"         → {previous, first}
... (many conv layers)
```

### Rotation / position logic

There is NO rotation/circular-buffer logic. The `cache` is a linear pre-allocated buffer written left-to-right. `offset` monotonically increases. The entire valid cache from position 0 to offset+T is read every step. For the FlowLM with typical inputs (100 tokens + 50 frames), the total context is ~150 tokens which is small; no performance concern.

For Mimi with `context=250`, the mask clips attention to the 250 most recent positions, but the buffer still grows linearly. Over a long sentence (many frames), this buffer could grow large. In practice, `init_states` is called fresh for each utterance.

### Critical for CoreML export

The `complete_kv` function (`transformer.py:9-19`) is the write path:
```python
offset_value = int(offset.view(-1)[0].item())          # ← produces aten::Int
cache[0, :, offset_value : offset_value + k.shape[1]] = k  # ← dynamic index write
```
This must be rewritten as:
- **Option A (MLState / SliceUpdate)**: Register cache as `ct.StateType`. Use a mask-based write pattern that the CoreML compiler recognizes as an in-place update. Follow the Apple `SliceUpdateKeyValueCache` pattern from the vendored knowledge.
- **Option B (explicit KV I/O)**: Pass `cache` as an input tensor, return updated cache as output. Swift copies it back before the next step. Avoids `aten::Int` entirely but doubles the memory traffic.

---

## 6. Top-level generation entrypoint

### CLI

`pocket_tts/main.py:222` — `generate()` command. Invoked as `pocket-tts generate --text "..." --voice <path>`.

Call graph:
```
generate() [main.py:222]
  TTSModel.load_model() [tts_model.py:232]
  TTSModel.get_state_for_audio_prompt(voice) [tts_model.py:787]
    ↳ (if .wav) _encode_audio() → mimi.encode_to_latent() + F.linear(speaker_proj_weight)
    ↳ (if .safetensors) _import_model_state()
    ↳ init_states(flow_lm) + _run_flow_lm_and_increment_step(audio_conditioning=prompt)
  TTSModel.generate_audio_stream(model_state, text) [tts_model.py:544]
    ↳ split_into_best_sentences()
    ↳ _generate_audio_stream_short_text() [tts_model.py:633]
        ↳ (thread) _generate() [tts_model.py:707]
            ↳ _run_flow_lm_and_increment_step(text_tokens=...)  ← text prefill
            ↳ (thread) _autoregressive_generation() [tts_model.py:745]
                ↳ for step in range(max_gen_len):
                    _run_flow_lm_and_increment_step(backbone_input=prev_latent)
                    → next_latent [float32[1, 1, 32]]
                    latents_queue.put(next_latent)
        ↳ (thread) _decode_audio_worker() [tts_model.py:433]
            ↳ for latent in latents_queue:
                latent * emb_std + emb_mean   ← un-normalize
                mimi.quantizer(transposed)    ← DummyQuantizer: Conv1d(32,512,1)
                mimi.decode_from_latent(quantized, mimi_state)
                    ↳ upsample → decoder_transformer → SEANetDecoder
```

### Python API

`TTSModel.generate_audio(model_state, text)` (`tts_model.py:476`) — collects the stream, returns `float32[samples]` (shape `[S]`, mono, 24kHz).

### Streaming implementation

YES, the model supports **chunk-wise streaming** at the audio frame level. `generate_audio_stream` yields `float32[1920]` chunks (80 ms each) as they are decoded. The AR generation loop and Mimi decoding run on **two separate threads** with a `queue.Queue` between them (line 647). There is no minimum chunk size forced by the code (beyond the 1920-sample Mimi frame).

---

## 7. Sampling loop structure

### Per-step (inside the AR loop) — runs 12.5 times per second of audio:

1. `_run_flow_lm_and_increment_step` → `_run_flow_lm` → `FlowLMModel._sample_next_latent` → `FlowLMModel.forward`:
   - NaN-to-BOS replacement
   - `input_linear` (Linear 32→1024)
   - `backbone`: cat + 6-layer `StreamingTransformer` + `out_norm` + slice
   - Cast to float32, take `[:, -1]`
   - EOS linear + threshold
   - Noise sample
   - `lsd_decode` loop × `lsd_decode_steps` = **1** (default):
     - `flow_net(c, s=0.0, t=1.0, x_noise)` → one `SimpleMLPAdaLN` forward

2. `increment_steps(flow_lm, model_state)` — increments `offset` for all 6 attention layers by 1.

3. `latents_queue.put(next_latent)` — hands off to Mimi thread.

### Once per utterance (prefill phase):

1. `_run_flow_lm_and_increment_step(text_tokens=tokens)` — runs transformer over all text tokens at once, writes them into KV cache, increments offset by S_text.
2. Voice conditioning: before text prefill, the voice audio runs through Mimi encoder + projection + another `_run_flow_lm_and_increment_step(audio_conditioning=prompt)`.

### LSD decode steps

`DEFAULT_LSD_DECODE_STEPS = 1` (`default_parameters.py:4`). This is the **normal runtime default**. At N=1:
- s=0, t=1 (constant, known at trace time)
- One `flow_net` call per AR step
- Total ANE dispatches per second: 12.5 (transformer) + 12.5 (flow net) + 12.5 (Mimi decode) ≈ 37.5/s

At N=4 (quality mode): 12.5 + 50 + 12.5 = 75 dispatches/s.

---

## 8. Export scripts and existing ports

### In the reference repo

- **No ONNX / CoreML / MLX export scripts exist** in the reference repository. The only "export" is `export_voice` CLI / `export_model_state()` function which serializes a pre-filled KV-cache state to safetensors for fast voice reload.
- `scripts/generate_default_voices.py` — generates voice `.safetensors` files from `.wav` inputs. Not a model export.
- `scripts/evaluate_quantization.py` — evaluates int8 quantization quality (WER via stt_model). Not a model export.
- `pocket_tts/quantization.py` — torchao / torch.ao int8 dynamic quantization for inference acceleration. The quantized layers are `{"attention", "ffn"}` (the `RECOMMENDED_CONFIG`). `flow_net` and `mimi` are explicitly excluded.

### External ports (from external_research.md, not duplicated here)

See external_research.md §A.7 for the ONNX export split (`KevinAHM/pocket-tts-onnx-export`). The 5-component split it uses is the right model for CoreML too.

---

## 9. Test files for end-to-end English verification

All tests are in `tests/`. The end-to-end ones that exercise English generation:

| File | Test | What it covers |
|------|------|----------------|
| `tests/test_cli_generate.py:19` | `test_generate_basic_usage` | CLI generate → WAV, asserts shape + 24 kHz |
| `tests/test_cli_generate.py:116` | `test_generate_default_text` | Default English text |
| `tests/test_cli_generate.py:128` | `test_generate_long_text` | Long text splitting, asserts ≥10s audio |
| `tests/test_documentation_examples.py:7` | `test_readme_example` | Python API: load → voice → generate → wavwrite |
| `tests/test_documentation_examples.py:19` | `test_quick_start` | Same, different voice |
| `tests/test_documentation_examples.py:96` | `test_generate_audio` | `frames_after_eos=2` |
| `tests/test_documentation_examples.py:110` | `test_generate_audio_stream` | Streaming, yields chunks |
| `demo.py` | N/A | Batch generation script, 6 voices × 6 phrases → WAV files |

`conftest.py` sets `POCKET_TTS_ERROR_WITHOUT_EOS=1` so tests fail loudly if generation maxes out without EOS (catches silent failures).

---

## 10. Audio post-processing

### Output normalization

Before passing to the Mimi decoder, the raw latent is un-normalized:
```python
mimi_decoding_input = latent * self.flow_lm.emb_std + self.flow_lm.emb_mean
```
(`tts_model.py:449`). `emb_std` and `emb_mean` are registered buffers of shape `[32]` in `FlowLMModel`. At inference they encode the training-set statistics of the latent distribution. This is a pure elementwise broadcast — ANE-friendly.

### Audio output from Mimi decoder

- Mimi decoder outputs `float32[1, 1, 1920]`.
- The `StreamingWAVWriter` (`data/audio.py:55`) converts to int16 PCM: `(chunk.clamp(-1,1) * 32767).short()`.
- A 200 ms silence tail is appended at end of utterance.
- No dithering, no loudness normalization.

### Voice prompt resampling

`convert_audio` (`data/audio_utils.py:8`) uses `scipy.signal.resample_poly` with `gcd`-reduced up/down ratios. This is CPU-only, runs once before inference. No issue for CoreML port.

---

## 11. Things that will be annoying to port (ranked by effort)

1. **KV cache write = `aten::Int` + dynamic slice** (`transformer.py:9-19`). The `int(offset.item())` extraction and `cache[0, :, offset:offset+T]` write are both traceable blockers. Every attention layer (6 in FlowLM + 2 in Mimi transformer × 2 = 10 total) needs rewriting. This is the single most invasive change — budget 2-3 days for rewrite + parity test. Options: MLState with mask-write pattern (Apple's blessed path) or KV-as-I/O (safer but doubles memory traffic per step).

2. **RoPE computed fully inside forward** (`rope.py:26-35`). `torch.arange(T)`, `torch.exp(...)`, `ts += offset` (where offset is a tensor) — all produce `aten::Int` or `aten::Float` from tensor arithmetic on scalars. Must pre-compute `cos_cached / sin_cached` buffers (indexed by absolute position) and pass them as model inputs from Swift. This affects every attention layer's forward signature.

3. **ConvTranspose1d strides 6, 5, 4 in SEANet decoder** will fall to CPU on ANE. Mitigation: accept CPU for the Mimi decoder (80 ms budget, the decoder runs in parallel with transformer so only the critical path matters) OR decompose each stride-6 transpose conv into stride-2 + stride-3 stages using two stacked transposed convs. The latter is a non-trivial architecture change that must be weight-initialized correctly.

4. **`lsd_decode` Python loop** (`flow_lm.py:33`). At `DEFAULT_LSD_DECODE_STEPS=1` this is trivially inlined. For N>1, the loop needs to be either unrolled into a single trace (fixed N, multiple flow_net calls concatenated) or called from Swift N times per AR step. The latter adds N-1 extra ANE dispatches per frame. Recommendation: unroll for `N ∈ {1, 2, 4}` and export enumerated-shape variants.

5. **Custom `LayerNorm` and `RMSNorm`** in `mlp.py`. `mlp.LayerNorm` (`x.var(unbiased=False)` path) and `RMSNorm` (`x.var()` path) are both non-standard. They will not benefit from ANE's fused LayerNorm kernel and will fall to CPU element-wise. For the flow net (tiny 512-dim vectors), this may be acceptable — measure before optimizing.

6. **`torch.where(isnan(sequence), ...)` in the AR loop** (`flow_lm.py:121`). The `isnan` op is an integer-class boolean check; ANE may not have a fused path. Fix: on step 0, pass `bos_emb[None, None, :]` as sequence directly; on subsequent steps, pass the previous latent. Eliminate the NaN convention entirely from the CoreML graph.

7. **Noise sampling in-graph** (`flow_lm.py:133-137`). `torch.nn.init.normal_` and `trunc_normal_` are in-place random ops that are non-deterministic and will not trace correctly. Move noise generation to Swift, pass as an additional input tensor `float32[1, 32]`.

8. **`_build_attention_mask` position arithmetic** (`transformer.py:23-29`). `pos_q = offset + arange(T)` and the `delta` computation involve integer tensors derived from `offset`. These will produce `aten::Int`. Replace with a pre-computed causal mask (upper-triangular, shape `[1, 1, T_q, T_k]`) that CoreML can handle, OR use the SDPA's built-in causal masking (`is_causal=True`) which avoids explicit mask construction.

9. **Voice conditioning `speaker_proj_weight`** (`tts_model.py:142`). This is a `torch.nn.Parameter` added after `_from_pydantic_config` (dynamically appended to `flow_lm`, not a module). `F.linear(latents, self.flow_lm.speaker_proj_weight)` at line 388. The speaker projection can be pre-computed entirely on CPU (it's only run once per voice cloning session, not in the AR loop).

10. **Chunked text splitting logic** (`split_into_best_sentences` at `tts_model.py:978`). This runs entirely in Python/SentencePiece before any tensor ops. Max 50 tokens per chunk (`MAX_TOKEN_PER_CHUNK=50`). This constraint exists because the model degrades on longer inputs — it must be respected in the Swift tokenizer path. There is a comment `# TODO: add the teacher forcing method` suggesting this limitation is acknowledged.

---

## Appendix: Key file:line index

| Topic | File | Line |
|-------|------|------|
| Default LSD steps = 1 | `pocket_tts/default_parameters.py` | 4 |
| Default temperature = 0.7 | `pocket_tts/default_parameters.py` | 3 |
| MAX_TOKEN_PER_CHUNK = 50 | `pocket_tts/default_parameters.py` | 7 |
| TTSModel.load_model() | `pocket_tts/models/tts_model.py` | 232 |
| TTSModel.generate_audio() | `pocket_tts/models/tts_model.py` | 476 |
| TTSModel.generate_audio_stream() | `pocket_tts/models/tts_model.py` | 544 |
| AR loop | `pocket_tts/models/tts_model.py` | 756 |
| NaN→BOS in forward | `pocket_tts/models/flow_lm.py` | 121 |
| lsd_decode() | `pocket_tts/models/flow_lm.py` | 19 |
| KV cache write (complete_kv) | `pocket_tts/modules/transformer.py` | 9 |
| aten::Int source (offset.item()) | `pocket_tts/modules/transformer.py` | 14 |
| RoPE forward | `pocket_tts/modules/rope.py` | 6 |
| Custom LayerNorm (mlp) | `pocket_tts/modules/mlp.py` | 39 |
| Custom RMSNorm | `pocket_tts/modules/mlp.py` | 20 |
| Noise sampling in forward | `pocket_tts/models/flow_lm.py` | 133 |
| SEANet dilation_base=2 | `pocket_tts/config/english.yaml` | 46 |
| SEANet ratios [6,5,4] | `pocket_tts/config/english.yaml` | 38 |
| DummyQuantizer (Conv1d identity) | `pocket_tts/modules/dummy_quantizer.py` | 16 |
| ConvTrUpsample1d (depthwise, stride=16) | `pocket_tts/modules/resample.py` | 34 |
| emb_std / emb_mean un-normalize | `pocket_tts/models/tts_model.py` | 449 |
| export_model_state() | `pocket_tts/models/tts_model.py` | 1047 |
| CLI generate entrypoint | `pocket_tts/main.py` | 222 |
| English config (2026-04) | `pocket_tts/config/english_2026-04.yaml` | — |
