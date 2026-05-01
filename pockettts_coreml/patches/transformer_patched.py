"""Traceable streaming multihead attention + transformer layer/stack.

Replaces `pocket_tts/modules/transformer.py` and the attention path in
`pocket_tts/modules/mimi_transformer.py`. The single
`PatchedStreamingMultiheadAttention` class serves both the FlowLM
transformer (6 layers, no context window) and the Mimi transformer (2
layers, context=250); the layers differ only by constructor params.

### Landmines eliminated

  1. `complete_kv` (ref `modules/transformer.py:9-19`) used
     `int(offset.view(-1)[0].item())` to produce a Python scalar offset,
     then did an in-place dynamic slice write into a `cache[..., offset:offset+T]`
     slot. Both operations produce `aten::Int` in `torch.jit.trace`.
     Replaced with a mask-based write:

         cache_new = cache * (1 - offset_mask) + broadcast(k) * offset_mask

     The caller computes `offset_mask` (a one-hot indicator along the
     capacity axis) in Swift and passes it as an input. Fully tensor-valued.

  2. `_build_attention_mask` (ref `modules/transformer.py:22-29`) used
     `offset + arange(T)` and boolean arithmetic. Replaced with an
     additive fp32 attention mask passed as input, values 0.0 (visible)
     and -65500.0 (masked). Bool masks + fp16 can NaN in softmax on some
     CoreML paths; additive fp16-safe mask avoids this.

  3. RoPE in-graph computation (ref `modules/rope.py:7-58`) replaced with
     pre-computed cos/sin tables passed as inputs (see `rope_patched.py`).

### Forward signature

    PatchedStreamingMultiheadAttention.forward(
        x, kv_cache, offset_mask, attn_mask, rope_cos, rope_sin
    ) -> (x_out, kv_cache_out)

All shapes are documented in the class docstring.
"""
# ref: modules/transformer.py:9-19 — mask-based KV write replaces in-place slice.
# ref: modules/transformer.py:22-29 — additive attn mask replaces bool mask.
# ref: modules/rope.py:7-58 — cos/sin passed in, not recomputed.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pockettts_coreml.patches.rope_patched import apply_rope

# Canonical "effectively-zero after softmax" value in fp16. Documented in
# the CoreML-LLM optimization notes. Using -inf would produce NaN on some
# fp16 code paths.
ATTN_MASK_NEG = -6.5504e4


class PatchedStreamingMultiheadAttention(nn.Module):
    """Streaming MHA with pure-functional KV I/O.

    Input shapes (static, traceable):
        x:           fp32[B, T_q, d_model]         — query (and self-attn source).
        kv_cache:    fp32[2, B, S_capacity, H, D]  — [K, V] along dim 0.
        offset_mask: fp32[B, S_capacity]           — one-hot indicator for
                                                     the slot being written
                                                     this step (T_q=1 case);
                                                     for prefill T_q>1 it's
                                                     a contiguous-block mask
                                                     of exactly T_q ones.
        attn_mask:   fp32[1, 1, T_q, S_capacity]   — additive, 0.0 visible,
                                                     -65500.0 masked.
        rope_cos:    fp32[1, T_q, 1, D // 2]       — RoPE cosine, pre-sliced
                                                     to this step's offset.
        rope_sin:    fp32[1, T_q, 1, D // 2]

    Output shapes:
        x_out:        fp32[B, T_q, d_model]
        kv_cache_out: fp32[2, B, S_capacity, H, D]

    Weight compatibility: matches reference
    `StreamingMultiheadAttention.in_proj` and `.out_proj` exactly (same
    shapes, same ordering of Q/K/V slices when reshaped to
    `[B, T, 3, H, D]`).
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // num_heads

        # Reference packs Q, K, V into a single linear projection of size
        # 3 * embed_dim. The reshape order is [B, T, 3, H, D] with
        # torch.unbind(dim=2) -> (Q, K, V). We match that exactly so the
        # same state_dict loads unchanged.
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    @torch.no_grad()
    def load_reference_weights(self, ref_attn: nn.Module) -> None:
        """Copy weights from a reference `StreamingMultiheadAttention`.

        Reference has the SAME layout (`in_proj.weight` shape
        `[3*embed_dim, embed_dim]`, `out_proj.weight` shape
        `[embed_dim, embed_dim]`). Direct copy works.
        """
        self.in_proj.weight.copy_(ref_attn.in_proj.weight)
        self.out_proj.weight.copy_(ref_attn.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: torch.Tensor,
        offset_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H = self.num_heads
        D = self.dim_per_head

        # IMPORTANT trace-compat: do NOT extract B/T from x.shape as
        # Python ints (that produces `aten::Int`).
        #
        # IMPORTANT CoreML bug: we use `torch.chunk(3, dim=-1)` rather than
        # `unflatten(-1, (3, H, D)); [:, :, 0|1|2]`. CoreML's converter
        # contains an optimization pass that detects the
        # "unflatten + slice into 3 along middle dim" pattern as a fused
        # QKV projection and mistakenly applies apply_rope's cos/sin
        # multiply to V as well as Q/K. The bug manifests as V drifting
        # dim-wise (verified 2026-05-01 on coremltools 8.1). `chunk`
        # breaks the pattern and lowers to three separate slices.
        projected = self.in_proj(x)  # [B, T, 3*H*D]
        q_flat, k_flat_, v_flat = projected.chunk(3, dim=-1)  # each [B, T, H*D]
        q = q_flat.unflatten(-1, (H, D))  # [B, T, H, D]
        k = k_flat_.unflatten(-1, (H, D))
        v = v_flat.unflatten(-1, (H, D))

        # RoPE on Q and K, using pre-computed tables.
        q, k = apply_rope(q, k, rope_cos, rope_sin)

        # --- mask-based KV write (T_q=1 AR path) -----------------------
        # offset_mask: [B, S] -> [B, S, 1, 1] to broadcast against cache.
        mask_bsh11 = offset_mask.unsqueeze(-1).unsqueeze(-1)  # [B, S, 1, 1]
        keep = 1.0 - mask_bsh11

        # cache: [2, B, S, H, D]. Slice K/V along dim 0 (static size 2).
        cache_k = kv_cache[0]  # [B, S, H, D]
        cache_v = kv_cache[1]

        # k/v are [B, 1, H, D]. Broadcasting directly: [B, 1, H, D] *
        # [B, S, 1, 1] => [B, S, H, D]. No explicit `.expand` needed —
        # PyTorch broadcast semantics handle it at the elementwise
        # multiply. This avoids `aten::Int` from reading cache.shape[1].
        cache_k_new = cache_k * keep + k * mask_bsh11
        cache_v_new = cache_v * keep + v * mask_bsh11
        kv_cache_out = torch.stack([cache_k_new, cache_v_new], dim=0)

        # Permute to [B, H, S, D] for SDPA.
        k_attn = cache_k_new.permute(0, 2, 1, 3)
        v_attn = cache_v_new.permute(0, 2, 1, 3)
        q_attn = q.permute(0, 2, 1, 3)  # [B, H, T, D]

        # Manual SDPA (see forward_prefill for rationale).
        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) * scale
        scores = scores + attn_mask
        probs = F.softmax(scores, dim=-1)
        x_attn = torch.matmul(probs, v_attn)
        # [B, H, T, D] -> [B, T, H, D] -> [B, T, H*D].  `flatten(-2)`
        # merges the last two axes without reading shape as Python ints.
        x_out = x_attn.permute(0, 2, 1, 3).flatten(-2)
        x_out = self.out_proj(x_out)
        return x_out, kv_cache_out

    def forward_prefill(
        self,
        x: torch.Tensor,
        kv_cache: torch.Tensor,
        scatter_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prefill path for T_q > 1 steps written into an empty cache region.

        `scatter_mask`: fp32[B, S, T_q] — each column j is a one-hot along S
            pointing at the target slot for new k/v entry j. For a
            contiguous prefill starting at position P of length T_q:
            scatter_mask[:, P+j, j] = 1.0, zeros elsewhere.

        Numerically equivalent to the single-step path executed T_q times.
        """
        H = self.num_heads
        D = self.dim_per_head

        projected = self.in_proj(x)
        # See forward() above for rationale on chunk vs unflatten+slice.
        q_flat, k_flat_, v_flat = projected.chunk(3, dim=-1)
        q = q_flat.unflatten(-1, (H, D))
        k = k_flat_.unflatten(-1, (H, D))
        v = v_flat.unflatten(-1, (H, D))
        q, k = apply_rope(q, k, rope_cos, rope_sin)

        # scatter_mask: [B, S, T_q]. For each position s, place a
        # weighted sum of new k rows. We use matmul instead of einsum —
        # einsum compiles to an unstable path under CoreML FP16 for this
        # batched reduction (produces NaN on CPU_ONLY). `bsj @ bj(h*d) ->
        # bs(h*d)` is numerically identical but lowers to a standard
        # batched matmul that CoreML handles cleanly.
        #
        # Use `flatten(-2)` / `unflatten(-1, (H, D))` to avoid any Python-
        # int shape arithmetic (traceable with no aten::Int).
        k_flat = k.flatten(-2)  # [B, T_q, H*D]
        v_flat = v.flatten(-2)
        new_k = torch.matmul(scatter_mask, k_flat).unflatten(-1, (H, D))  # [B, S, H, D]
        new_v = torch.matmul(scatter_mask, v_flat).unflatten(-1, (H, D))

        # "keep" mask = 1 everywhere except slots that had any j=1 written.
        written = scatter_mask.sum(dim=-1, keepdim=False)  # [B, S]
        keep = 1.0 - written.unsqueeze(-1).unsqueeze(-1)  # [B, S, 1, 1]

        cache_k = kv_cache[0]
        cache_v = kv_cache[1]
        cache_k_new = cache_k * keep + new_k
        cache_v_new = cache_v * keep + new_v
        kv_cache_out = torch.stack([cache_k_new, cache_v_new], dim=0)

        k_attn = cache_k_new.permute(0, 2, 1, 3)
        v_attn = cache_v_new.permute(0, 2, 1, 3)
        q_attn = q.permute(0, 2, 1, 3)

        # Manual SDPA. `F.scaled_dot_product_attention` compiles to a
        # CoreML fp16 codepath that produces NaN on additive masks
        # containing large-magnitude values (-65500) — verified on CPU_ONLY
        # with coremltools 8.1 / iOS18 target. The manual form lowers to
        # matmul+add+softmax+matmul and is numerically stable.
        # `D` is a Python-int class attribute (= self.dim_per_head) so
        # the scale is baked into the trace as a constant.
        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) * scale
        scores = scores + attn_mask
        probs = F.softmax(scores, dim=-1)
        x_attn = torch.matmul(probs, v_attn)
        x_out = x_attn.permute(0, 2, 1, 3).flatten(-2)
        x_out = self.out_proj(x_out)
        return x_out, kv_cache_out


class PatchedStreamingTransformerLayer(nn.Module):
    """One layer = norm1 -> MHA -> residual -> norm2 -> FFN -> residual.

    Layout matches reference `StreamingTransformerLayer`:
        - `norm1`, `norm2`: nn.LayerNorm(d_model, eps=1e-5)
        - `linear1`: Linear(d_model, dim_feedforward, bias=False)
        - `linear2`: Linear(dim_feedforward, d_model, bias=False)
        - Activation: GELU
        - `layer_scale` (optional per-channel scale).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        layer_scale: float | None = None,
    ):
        super().__init__()
        self.self_attn = PatchedStreamingMultiheadAttention(
            embed_dim=d_model, num_heads=num_heads
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)

        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            # Reference LayerScale is a per-channel learnable scalar
            # multiply. Mimic by storing a parameter vector and applying
            # it in forward (CoreML-friendly elementwise).
            self.layer_scale_1 = _PatchedLayerScale(d_model, init=layer_scale)
            self.layer_scale_2 = _PatchedLayerScale(d_model, init=layer_scale)

    @torch.no_grad()
    def load_reference_weights(self, ref_layer: nn.Module) -> None:
        self.self_attn.load_reference_weights(ref_layer.self_attn)
        self.norm1.weight.copy_(ref_layer.norm1.weight)
        self.norm1.bias.copy_(ref_layer.norm1.bias)
        self.norm2.weight.copy_(ref_layer.norm2.weight)
        self.norm2.bias.copy_(ref_layer.norm2.bias)
        self.linear1.weight.copy_(ref_layer.linear1.weight)
        self.linear2.weight.copy_(ref_layer.linear2.weight)
        if isinstance(self.layer_scale_1, _PatchedLayerScale):
            # Reference stores `LayerScale.scale` as a Parameter.
            self.layer_scale_1.scale.copy_(ref_layer.layer_scale_1.scale)
            self.layer_scale_2.scale.copy_(ref_layer.layer_scale_2.scale)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: torch.Tensor,
        offset_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self-attention block.
        x_norm = self.norm1(x)
        update, kv_cache_out = self.self_attn(
            x_norm, kv_cache, offset_mask, attn_mask, rope_cos, rope_sin
        )
        x = x + self.layer_scale_1(update)

        # FFN block.
        x_norm = self.norm2(x)
        ff = self.linear2(F.gelu(self.linear1(x_norm)))
        x = x + self.layer_scale_2(ff)
        return x, kv_cache_out

    def forward_prefill(
        self,
        x: torch.Tensor,
        kv_cache: torch.Tensor,
        scatter_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_norm = self.norm1(x)
        update, kv_cache_out = self.self_attn.forward_prefill(
            x_norm, kv_cache, scatter_mask, attn_mask, rope_cos, rope_sin
        )
        x = x + self.layer_scale_1(update)
        x_norm = self.norm2(x)
        ff = self.linear2(F.gelu(self.linear1(x_norm)))
        x = x + self.layer_scale_2(ff)
        return x, kv_cache_out


class _PatchedLayerScale(nn.Module):
    """Per-channel learnable scalar multiply, traceable elementwise op.

    Reference `modules/layer_scale.py` is equivalent; we re-implement
    to avoid the reference import at trace time.
    """

    def __init__(self, channels: int, init: float = 1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class PatchedStreamingTransformer(nn.Module):
    """Stack of `num_layers` patched transformer layers.

    Pure-functional KV I/O: input `kv_caches` is a single tensor of shape
    `[num_layers, 2, B, S, H, D]`; output has the same shape. RoPE tables
    are shared across layers (one cos/sin pair).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        layer_scale: float | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = d_model // num_heads
        self.layers = nn.ModuleList(
            [
                PatchedStreamingTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    layer_scale=layer_scale,
                )
                for _ in range(num_layers)
            ]
        )

    @torch.no_grad()
    def load_reference_weights(self, ref_transformer: nn.Module) -> None:
        """Copy weights from reference `StreamingTransformer`."""
        assert len(self.layers) == len(ref_transformer.layers)
        for patched_l, ref_l in zip(self.layers, ref_transformer.layers):
            patched_l.load_reference_weights(ref_l)

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: torch.Tensor,
        offset_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step AR forward.

        Accepts either:
          - rank-6  [num_layers, 2, B, S, H, D]  (eager/tests; easy indexing)
          - rank-5  [2*num_layers, B, S, H, D]   (CoreML; Core ML caps at rank 5)

        We detect by `dim()` and fold the layer index accordingly. Both
        shapes are numerically identical; the rank-5 layout is simply
        the rank-6 layout with dims 0 and 1 flattened into `[2L]`.
        """
        if kv_caches.dim() == 6:
            new_caches = []
            for i, layer in enumerate(self.layers):
                x, kv_out = layer(
                    x, kv_caches[i], offset_mask, attn_mask, rope_cos, rope_sin,
                )
                new_caches.append(kv_out)
            return x, torch.stack(new_caches, dim=0)
        else:
            # rank-5: layer i occupies rows [2*i : 2*i+2]
            new_caches = []
            for i, layer in enumerate(self.layers):
                layer_cache = kv_caches[2 * i : 2 * i + 2]  # [2, B, S, H, D]
                x, kv_out = layer(
                    x, layer_cache, offset_mask, attn_mask, rope_cos, rope_sin,
                )
                new_caches.append(kv_out)
            # Concat along dim 0 to preserve [2L, B, S, H, D].
            return x, torch.cat(new_caches, dim=0)

    def forward_prefill(
        self,
        x: torch.Tensor,
        kv_caches: torch.Tensor,
        scatter_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kv_caches.dim() == 6:
            new_caches = []
            for i, layer in enumerate(self.layers):
                x, kv_out = layer.forward_prefill(
                    x, kv_caches[i], scatter_mask, attn_mask, rope_cos, rope_sin
                )
                new_caches.append(kv_out)
            return x, torch.stack(new_caches, dim=0)
        else:
            new_caches = []
            for i, layer in enumerate(self.layers):
                layer_cache = kv_caches[2 * i : 2 * i + 2]
                x, kv_out = layer.forward_prefill(
                    x, layer_cache, scatter_mask, attn_mask, rope_cos, rope_sin
                )
                new_caches.append(kv_out)
            return x, torch.cat(new_caches, dim=0)


# ------------------------------------------------------------------
# Helper builders used by tests + Phase 3 conversion scripts
# ------------------------------------------------------------------


def build_additive_attention_mask_step(
    offset: int, s_capacity: int, context: int | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Pre-trace helper: additive attention mask for a single AR step at
    absolute position `offset` with a cache of capacity `s_capacity`.

    Returns shape `[1, 1, 1, s_capacity]`. Values 0.0 at positions the
    query can attend to, ATTN_MASK_NEG elsewhere.

    Causality: visible <=> pos_k in [0, offset].  (Inclusive: the
    just-written slot at `offset` IS visible.)
    Context window: if `context` is set, additionally require
        (offset - pos_k) < context.

    Called OUTSIDE the traced graph; passed as an input.
    """
    assert 0 <= offset < s_capacity
    mask = torch.full((1, 1, 1, s_capacity), ATTN_MASK_NEG, dtype=dtype)
    lo = 0 if context is None else max(0, offset - context + 1)
    mask[0, 0, 0, lo : offset + 1] = 0.0
    return mask


def build_one_hot_offset_mask(
    offset: int, s_capacity: int, batch_size: int = 1, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Pre-trace helper: one-hot `[B, S_capacity]` indicator for the AR
    write slot. Called OUTSIDE the traced graph.
    """
    mask = torch.zeros((batch_size, s_capacity), dtype=dtype)
    mask[:, offset] = 1.0
    return mask


def build_scatter_prefill_mask(
    start_offset: int,
    prefill_len: int,
    s_capacity: int,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pre-trace helper: `[B, S_capacity, prefill_len]` scatter map for a
    contiguous prefill of `prefill_len` tokens starting at absolute
    position `start_offset`.
    """
    assert 0 <= start_offset
    assert start_offset + prefill_len <= s_capacity
    mask = torch.zeros((batch_size, s_capacity, prefill_len), dtype=dtype)
    for j in range(prefill_len):
        mask[:, start_offset + j, j] = 1.0
    return mask


def build_additive_attention_mask_prefill(
    start_offset: int,
    prefill_len: int,
    s_capacity: int,
    context: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """`[1, 1, prefill_len, s_capacity]` additive causal mask for prefill.

    Row `i` (i = 0..prefill_len-1) corresponds to absolute position
    `start_offset + i`. It can attend to positions
    `[lo, start_offset + i]` where `lo` is
    `max(0, start_offset + i - context + 1)`.
    """
    mask = torch.full(
        (1, 1, prefill_len, s_capacity), ATTN_MASK_NEG, dtype=dtype
    )
    for i in range(prefill_len):
        pos = start_offset + i
        lo = 0 if context is None else max(0, pos - context + 1)
        mask[0, 0, i, lo : pos + 1] = 0.0
    return mask


def init_empty_kv_cache(
    num_layers: int, batch_size: int, s_capacity: int, num_heads: int, head_dim: int,
    dtype: torch.dtype = torch.float32,
    rank: int = 6,
) -> torch.Tensor:
    """Zero-initialized KV cache.

    rank=6 (default, used by tests): shape [L, 2, B, S, H, D].
    rank=5 (used by CoreML graphs):  shape [2*L, B, S, H, D].

    The rank-5 form is numerically identical to rank-6 with the first two
    dimensions flattened. Core ML only supports tensors up to rank 5
    (`Core ML only supports tensors with rank <= 5`), so the CoreML
    conversion and runtime paths use rank 5.
    """
    if rank == 6:
        return torch.zeros(
            (num_layers, 2, batch_size, s_capacity, num_heads, head_dim), dtype=dtype
        )
    if rank == 5:
        return torch.zeros(
            (2 * num_layers, batch_size, s_capacity, num_heads, head_dim), dtype=dtype
        )
    raise ValueError(f"init_empty_kv_cache supports rank 5 or 6, got {rank}")
