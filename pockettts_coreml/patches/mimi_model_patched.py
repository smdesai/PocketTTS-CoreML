"""Patched Mimi encoder + decoder assemblies for CoreML tracing.

This module builds two pure-functional wrappers:

  - PatchedMimiEncoder (non-streaming):
      waveform[1,1,T] -> latents[1,32,T/1920]
      Runs the SEANet encoder (zero conv state), encoder_transformer (prefill
      with causal+context mask), and the 16x depthwise downsample.

  - PatchedMimiDecoder (streaming, per-frame):
      latent[1,32,1] + packed_state[fp16[N]]
        -> audio[1,1,1920] + packed_state_out[fp16[N]]
      Runs the 16x depthwise upsample, decoder_transformer (one AR step,
      context=250), and the SEANet decoder with all streaming state
      threaded through explicitly.

Shared: `PatchedMimiTransformer` wraps a `ProjectedTransformer`-equivalent
with our traceable `PatchedStreamingMultiheadAttention` — this is what
replaces the reference's `inverse(int32)` landmine in the encoder and
decoder transformer stacks.

State blob layout
-----------------
The decoder packs all streaming states into a single `fp16[N]` blob so
Swift sees a single input/output tensor. The layout (produced at init)
is a list of (name, shape, offset_in_blob, length). See
`docs/mimi_state_layout.md` for the authoritative documentation.
"""
# ref: modules/mimi_transformer.py:57-101 — StreamingTransformer replaced
# ref: modules/mimi_transformer.py:104-150 — ProjectedTransformer replaced
# ref: modules/seanet.py:7-180 — SEANet encoder + decoder replaced
# ref: modules/resample.py:7-51 — ConvDownsample1d + ConvTrUpsample1d replaced
# ref: models/mimi.py:89-120 — decode_from_latent + encode_to_latent replaced

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pockettts_coreml.patches.mimi_patched import (
    PatchedStreamingConv1d,
    PatchedStreamingConvTranspose1d,
)
from pockettts_coreml.patches.rope_patched import apply_rope, build_rope_tables
from pockettts_coreml.patches.transformer_patched import (
    ATTN_MASK_NEG,
    PatchedStreamingMultiheadAttention,
    _PatchedLayerScale,
)


# --------------------------------------------------------------------
# Mimi transformer layer/stack (reuses our patched MHA)
# --------------------------------------------------------------------


class PatchedMimiTransformerLayer(nn.Module):
    """One Mimi-transformer layer: norm1 -> MHA -> residual -> norm2 -> FFN.

    Mirrors `pocket_tts/modules/mimi_transformer.py:StreamingTransformerLayer`
    but routes attention through the traceable MHA. Supports layer_scale
    (Mimi sets it to 0.01 via config).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        layer_scale: float | None,
    ):
        super().__init__()
        self.self_attn = PatchedStreamingMultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
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
            self.layer_scale_1.scale.copy_(ref_layer.layer_scale_1.scale)
            self.layer_scale_2.scale.copy_(ref_layer.layer_scale_2.scale)

    def forward_step(
        self,
        x: torch.Tensor,
        kv_cache: torch.Tensor,
        offset_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single AR step: x is [B, 1, d_model]."""
        x_norm = self.norm1(x)
        update, kv_out = self.self_attn(
            x_norm, kv_cache, offset_mask, attn_mask, rope_cos, rope_sin,
        )
        x = x + self.layer_scale_1(update)
        x_norm = self.norm2(x)
        ff = self.linear2(F.gelu(self.linear1(x_norm)))
        x = x + self.layer_scale_2(ff)
        return x, kv_out

    def forward_prefill_like(
        self,
        x: torch.Tensor,               # [B, T_q, d_model]
        kv_cache: torch.Tensor,        # [2, B, S_cap, H, D]
        scatter_mask: torch.Tensor,    # [B, S_cap, T_q]
        attn_mask: torch.Tensor,       # [1, 1, T_q, S_cap]
        rope_cos: torch.Tensor,        # [1, T_q, 1, D//2]
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prefill-style call: T_q > 1 with scatter_mask KV write."""
        x_norm = self.norm1(x)
        update, kv_out = self.self_attn.forward_prefill(
            x_norm, kv_cache, scatter_mask, attn_mask, rope_cos, rope_sin,
        )
        x = x + self.layer_scale_1(update)
        x_norm = self.norm2(x)
        ff = self.linear2(F.gelu(self.linear1(x_norm)))
        x = x + self.layer_scale_2(ff)
        return x, kv_out

    def forward_nonstreaming(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Non-streaming full-sequence forward (used by the encoder).

        Equivalent to reference StreamingMultiheadAttention.forward with
        model_state=None: all Q, K, V are derived from the full prefill
        tensor; no KV cache carry-over. Numerically identical to the
        reference's non-streaming path.
        """
        # --- SA block
        x_norm = self.norm1(x)
        B = x_norm.shape[0]
        H = self.self_attn.num_heads
        D = self.self_attn.dim_per_head
        projected = self.self_attn.in_proj(x_norm)
        # Use chunk (not unflatten+slice) to avoid CoreML's QKV-fusion
        # mis-optimization (see transformer_patched.py for details).
        q_flat, k_flat_, v_flat = projected.chunk(3, dim=-1)
        q = q_flat.unflatten(-1, (H, D))
        k = k_flat_.unflatten(-1, (H, D))
        v = v_flat.unflatten(-1, (H, D))
        q, k = apply_rope(q, k, rope_cos, rope_sin)
        q_attn = q.permute(0, 2, 1, 3)
        k_attn = k.permute(0, 2, 1, 3)
        v_attn = v.permute(0, 2, 1, 3)
        # Manual SDPA — F.scaled_dot_product_attention produces NaN on
        # fp16 CoreML with additive mask values of -65500. Lowering to
        # explicit matmul + softmax keeps the compute in a stable path.
        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) * scale
        scores = scores + attn_mask
        probs = F.softmax(scores, dim=-1)
        sa = torch.matmul(probs, v_attn)
        sa = sa.permute(0, 2, 1, 3).flatten(-2)
        sa = self.self_attn.out_proj(sa)
        x = x + self.layer_scale_1(sa)
        # --- FFN block
        x_norm = self.norm2(x)
        ff = self.linear2(F.gelu(self.linear1(x_norm)))
        x = x + self.layer_scale_2(ff)
        return x


class PatchedMimiTransformer(nn.Module):
    """2-layer Mimi transformer stack (context=250).

    Two paths:
      - forward_step: streaming per-frame (used by decoder), KV cache I/O.
      - forward_nonstreaming: full prefill (used by encoder).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        context: int,
        layer_scale: float | None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.context = context
        self.head_dim = d_model // num_heads
        self.layers = nn.ModuleList(
            [
                PatchedMimiTransformerLayer(
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
        """Copy from `ProjectedTransformer.transformer` (StreamingTransformer)."""
        assert len(self.layers) == len(ref_transformer.layers)
        for p, r in zip(self.layers, ref_transformer.layers):
            p.load_reference_weights(r)

    def forward_step(
        self,
        x: torch.Tensor,               # [B, 1, d_model]
        kv_caches: torch.Tensor,       # [2*L, B, S_cap, H, D]  (rank-5 packed)
        offset_mask: torch.Tensor,     # [B, S_cap]
        attn_mask: torch.Tensor,       # [1, 1, 1, S_cap]
        rope_cos: torch.Tensor,        # [1, 1, 1, D//2]
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[2 * i : 2 * i + 2]  # [2, B, S, H, D]
            x, kv_out = layer.forward_step(
                x, layer_cache, offset_mask, attn_mask, rope_cos, rope_sin,
            )
            new_caches.append(kv_out)
        return x, torch.cat(new_caches, dim=0)

    def forward_step_multi(
        self,
        x: torch.Tensor,               # [B, T_q, d_model]
        kv_caches: torch.Tensor,       # [2*L, B, S_cap, H, D]
        scatter_mask: torch.Tensor,    # [B, S_cap, T_q] -- one-hot per col
        attn_mask: torch.Tensor,       # [1, 1, T_q, S_cap]
        rope_cos: torch.Tensor,        # [1, T_q, 1, D//2]
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-step path (T_q > 1) with scatter-mask write into KV cache.

        Equivalent to reference's single call with T=T_q. Used by the
        decoder where each AR frame runs T_q=16 through the transformer
        (16 encoder-framerate slots per frame-rate latent).
        """
        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[2 * i : 2 * i + 2]
            x, kv_out = layer.forward_prefill_like(
                x, layer_cache, scatter_mask, attn_mask, rope_cos, rope_sin,
            )
            new_caches.append(kv_out)
        return x, torch.cat(new_caches, dim=0)

    def forward_nonstreaming(
        self,
        x: torch.Tensor,               # [B, T, d_model]
        attn_mask: torch.Tensor,       # [1, 1, T, T]
        rope_cos: torch.Tensor,        # [1, T, 1, D//2]
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer.forward_nonstreaming(x, attn_mask, rope_cos, rope_sin)
        return x


class PatchedProjectedTransformer(nn.Module):
    """Mimi's `ProjectedTransformer` with patched internals.

    Matches reference wiring:
        x_in (shape [B, C, T]) -> transpose(1,2) -> optional input_proj ->
        transformer -> [output_projs...] -> transpose(1,2) -> outputs

    English config has `input_proj=None` and one `output_proj=Identity()`.
    We support both code paths.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimensions: Sequence[int],
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        context: int,
        layer_scale: float | None,
    ):
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimensions = tuple(output_dimensions)
        self.transformer = PatchedMimiTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            context=context,
            layer_scale=layer_scale,
        )
        if d_model != input_dimension:
            self.input_proj = nn.Linear(input_dimension, d_model, bias=False)
        else:
            self.input_proj = None
        self.output_projs = nn.ModuleList()
        for od in output_dimensions:
            if d_model == od:
                self.output_projs.append(nn.Identity())
            else:
                self.output_projs.append(nn.Linear(d_model, od, bias=False))

    @torch.no_grad()
    def load_reference_weights(self, ref_projected: nn.Module) -> None:
        self.transformer.load_reference_weights(ref_projected.transformer)
        if self.input_proj is not None:
            assert ref_projected.input_proj is not None
            self.input_proj.weight.copy_(ref_projected.input_proj.weight)
        else:
            assert ref_projected.input_proj is None
        for p_out, r_out in zip(self.output_projs, ref_projected.output_projs):
            if isinstance(p_out, nn.Linear):
                p_out.weight.copy_(r_out.weight)

    def _project_out(self, z: torch.Tensor) -> torch.Tensor:
        """English case: single identity output proj. Returns `[B, C, T]`.

        For multiple output projections, returns the first (English use
        has exactly one).
        """
        y = self.output_projs[0](z)
        y = y.transpose(1, 2)
        return y

    def forward_step(
        self,
        x_in: torch.Tensor,             # [B, C, 1]
        kv_caches: torch.Tensor,
        offset_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x_in.transpose(1, 2)  # [B, 1, C]
        if self.input_proj is not None:
            x = self.input_proj(x)
        z, kv_out = self.transformer.forward_step(
            x, kv_caches, offset_mask, attn_mask, rope_cos, rope_sin,
        )
        return self._project_out(z), kv_out

    def forward_step_multi(
        self,
        x_in: torch.Tensor,             # [B, C, T_q]
        kv_caches: torch.Tensor,
        scatter_mask: torch.Tensor,     # [B, S_cap, T_q]
        attn_mask: torch.Tensor,        # [1, 1, T_q, S_cap]
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x_in.transpose(1, 2)  # [B, T_q, C]
        if self.input_proj is not None:
            x = self.input_proj(x)
        z, kv_out = self.transformer.forward_step_multi(
            x, kv_caches, scatter_mask, attn_mask, rope_cos, rope_sin,
        )
        return self._project_out(z), kv_out

    def forward_nonstreaming(
        self,
        x_in: torch.Tensor,             # [B, C, T]
        attn_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x_in.transpose(1, 2)
        if self.input_proj is not None:
            x = self.input_proj(x)
        z = self.transformer.forward_nonstreaming(x, attn_mask, rope_cos, rope_sin)
        return self._project_out(z)


# --------------------------------------------------------------------
# SEANet encoder (non-streaming) + decoder (streaming, packed state)
# --------------------------------------------------------------------


def _new_conv1d(ref_conv: nn.Conv1d) -> PatchedStreamingConv1d:
    """Create a PatchedStreamingConv1d that matches a reference Conv1d shape
    and copies its weights.
    """
    c = ref_conv
    p = PatchedStreamingConv1d(
        in_channels=c.in_channels,
        out_channels=c.out_channels,
        kernel_size=c.kernel_size[0],
        stride=c.stride[0],
        dilation=c.dilation[0],
        groups=c.groups,
        bias=c.bias is not None,
    )
    p.conv.weight.data.copy_(c.weight.data)
    if c.bias is not None:
        p.conv.bias.data.copy_(c.bias.data)
    return p


def _new_convtr1d(ref_convtr: nn.ConvTranspose1d) -> PatchedStreamingConvTranspose1d:
    c = ref_convtr
    p = PatchedStreamingConvTranspose1d(
        in_channels=c.in_channels,
        out_channels=c.out_channels,
        kernel_size=c.kernel_size[0],
        stride=c.stride[0],
        groups=c.groups,
        bias=c.bias is not None,
    )
    p.convtr.weight.data.copy_(c.weight.data)
    if c.bias is not None:
        p.convtr.bias.data.copy_(c.bias.data)
    return p


class PatchedSEANetResnetBlockEncoder(nn.Module):
    """Non-streaming SEANet residual block for the encoder.

    Matches `modules/seanet.py:SEANetResnetBlock`. Non-streaming: left-pad
    with zeros, no state carry-over.
    """

    def __init__(self, ref_block: nn.Module, batch_size: int = 1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.order: list[str] = []  # 'elu' or 'conv'
        prev_zeros: list[torch.Tensor] = []
        for sub in ref_block.block:
            if isinstance(sub, nn.ELU):
                self.order.append("elu")
            else:
                patched = _new_conv1d(sub.conv)
                self.convs.append(patched)
                self.order.append("conv")
                prev_zeros.append(
                    torch.zeros(batch_size, patched.in_channels, patched.state_length)
                )
        for i, buf in enumerate(prev_zeros):
            self.register_buffer(f"_prev_zero_{i}", buf)
        self._prev_zero_list = [
            getattr(self, f"_prev_zero_{i}") for i in range(len(prev_zeros))
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x
        ci = 0
        for tag in self.order:
            if tag == "elu":
                v = F.elu(v, alpha=1.0)
            else:
                conv = self.convs[ci]
                prev_buf = self._prev_zero_list[ci]
                ci += 1
                if conv.state_length > 0:
                    vp = torch.cat([prev_buf, v], dim=-1)
                else:
                    vp = v
                v = conv.conv(vp)
        return x + v


class PatchedSEANetEncoder(nn.Module):
    """Traceable SEANet encoder, non-streaming path.

    Walks the reference `SEANetEncoder.model` ModuleList and replaces each
    `StreamingConv1d` with a zero-state wrapper and each `SEANetResnetBlock`
    with `PatchedSEANetResnetBlockEncoder`.

    Input:  x fp32[B, 1, T_audio]   (T_audio = multiple of frame_size)
    Output: x fp32[B, 512, T_enc]   (T_enc = T_audio / 1920 * 16 = T_audio / 120)
    """

    def __init__(self, ref_encoder: nn.Module, batch_size: int = 1):
        super().__init__()
        from pocket_tts.modules.conv import StreamingConv1d
        from pocket_tts.modules.seanet import SEANetResnetBlock
        self.modules_list = nn.ModuleList()
        self.kinds: list[str] = []
        # Pre-register zero left-context buffers (one per conv) so trace
        # doesn't need to read dynamic shapes.
        prev_zero_buffers: list[torch.Tensor] = []
        for layer in ref_encoder.model:
            if isinstance(layer, StreamingConv1d):
                patched = _new_conv1d(layer.conv)
                self.modules_list.append(patched)
                self.kinds.append("conv")
                prev_zero_buffers.append(
                    torch.zeros(batch_size, patched.in_channels, patched.state_length)
                )
            elif isinstance(layer, SEANetResnetBlock):
                self.modules_list.append(PatchedSEANetResnetBlockEncoder(layer, batch_size))
                self.kinds.append("resblock")
                prev_zero_buffers.append(torch.zeros(0))  # placeholder, unused
            elif isinstance(layer, nn.ELU):
                self.modules_list.append(nn.ELU(alpha=1.0))
                self.kinds.append("elu")
                prev_zero_buffers.append(torch.zeros(0))  # placeholder, unused
            else:
                raise RuntimeError(f"Unexpected encoder layer: {type(layer)}")
        # Register as buffers so they ride along in state_dict / to(device).
        for i, buf in enumerate(prev_zero_buffers):
            self.register_buffer(f"_prev_zero_{i}", buf)
        # Accessor list (not a ParameterList — just plain tensor refs).
        self._prev_zero_buffers = [
            getattr(self, f"_prev_zero_{i}") for i in range(len(prev_zero_buffers))
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mod, kind, prev_buf in zip(
            self.modules_list, self.kinds, self._prev_zero_buffers
        ):
            if kind == "conv":
                if mod.state_length > 0:
                    xp = torch.cat([prev_buf, x], dim=-1)
                else:
                    xp = x
                x = mod.conv(xp)
            elif kind == "resblock":
                x = mod(x)
            else:  # elu
                x = mod(x)
        return x


# --------------------------------------------------------------------
# Decoder state blob layout
# --------------------------------------------------------------------


@dataclass
class StateSlot:
    name: str
    shape: tuple[int, ...]
    offset: int
    length: int  # numel

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "offset": self.offset,
            "length": self.length,
        }


def compute_decoder_state_layout_from_ref(
    ref_mimi: nn.Module, state_s_cap: int, batch_size: int = 1,
) -> "DecoderStateLayout":
    """Compute the packed state-blob layout directly from a reference
    MimiModel WITHOUT instantiating a PatchedMimiDecoder (which would
    draw RNG for its nn.Linear/Conv1d default init).

    Used by the CoreML e2e generator to pre-allocate the state blob
    without disturbing the torch RNG trajectory vs the oracle.
    """
    from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
    from pocket_tts.modules.seanet import SEANetResnetBlock

    up_conv = ref_mimi.upsample.convtr.convtr
    up_k = up_conv.kernel_size[0]
    up_s = up_conv.stride[0]
    outer_dim = up_conv.out_channels

    ref_layer0 = ref_mimi.decoder_transformer.transformer.layers[0]
    tx_num_heads = ref_layer0.self_attn.num_heads
    tx_head_dim = ref_layer0.self_attn.dim_per_head
    tx_num_layers = len(ref_mimi.decoder_transformer.transformer.layers)

    # Walk the decoder to collect per-conv state descriptors.
    state_descs: list[dict] = []
    for i, layer in enumerate(ref_mimi.decoder.model):
        if isinstance(layer, StreamingConv1d):
            c = layer.conv
            K = c.kernel_size[0]; S = c.stride[0]; DIL = c.dilation[0]
            TP = (K - 1) * DIL + 1 - S
            if TP > 0:
                state_descs.append(dict(
                    name=f"dec.idx{i}.conv",
                    shape=(batch_size, c.in_channels, TP),
                ))
        elif isinstance(layer, StreamingConvTranspose1d):
            c = layer.convtr
            K = c.kernel_size[0]; S = c.stride[0]
            PT = K - S
            if PT > 0:
                state_descs.append(dict(
                    name=f"dec.idx{i}.convtr",
                    shape=(batch_size, c.out_channels, PT),
                ))
        elif isinstance(layer, SEANetResnetBlock):
            prefix = f"dec.idx{i}.resblock"
            ci = 0
            for sub in layer.block:
                if isinstance(sub, StreamingConv1d):
                    K = sub.conv.kernel_size[0]
                    S = sub.conv.stride[0]
                    DIL = sub.conv.dilation[0]
                    ke = (K - 1) * DIL + 1
                    TP = ke - S
                    if TP > 0:
                        state_descs.append(dict(
                            name=f"{prefix}.conv{ci}",
                            shape=(batch_size, sub.conv.in_channels, TP),
                        ))
                    ci += 1
    return DecoderStateLayout(
        seanet_decoder_layers=state_descs,
        outer_dim=outer_dim,
        upsample_kernel=up_k,
        upsample_stride=up_s,
        tx_num_layers=tx_num_layers,
        tx_s_cap=state_s_cap,
        tx_num_heads=tx_num_heads,
        tx_head_dim=tx_head_dim,
        batch_size=batch_size,
    )


class DecoderStateLayout:
    """Authoritative layout of the decoder's packed state blob.

    Layout order (slot, shape, purpose):
      - upsample_partial:     [1, outer_dim=512, 16]         (ConvTrUpsample1d)
      - tx_kv:                [2*L=4, 1, S_cap=256, H=8, D=64]
      - conv previous/partial slots in layer-walk order, skipping zero-length.

    On a 1-layer resnet block (k=3 + k=1), only the k=3 conv's previous is
    non-zero (TP=2); the k=1 one has TP=0 and is omitted.
    """

    def __init__(
        self,
        seanet_decoder_layers: list[dict],
        outer_dim: int,
        upsample_kernel: int,
        upsample_stride: int,
        tx_num_layers: int,
        tx_s_cap: int,
        tx_num_heads: int,
        tx_head_dim: int,
        batch_size: int = 1,
    ):
        self.slots: list[StateSlot] = []
        offset = 0

        # 1) Upsample partial first (so Swift initializes it as the first slice).
        up_pt = upsample_kernel - upsample_stride
        shp = (batch_size, outer_dim, up_pt)
        n = math.prod(shp)
        self.slots.append(StateSlot("upsample_partial", shp, offset, n))
        offset += n

        # 2) Transformer KV cache packed rank-5.
        shp = (2 * tx_num_layers, batch_size, tx_s_cap, tx_num_heads, tx_head_dim)
        n = math.prod(shp)
        self.slots.append(StateSlot("tx_kv", shp, offset, n))
        offset += n

        # 3) SEANet decoder conv/convtr state slots in layer-walk order.
        for desc in seanet_decoder_layers:
            name = desc["name"]
            shp = tuple(desc["shape"])
            n = math.prod(shp)
            if n == 0:
                # skip zero-length slots entirely (k=1 convs, stride==K convtrs)
                continue
            self.slots.append(StateSlot(name, shp, offset, n))
            offset += n

        self.total_elems = offset

    def serialize(self) -> list[dict]:
        return [s.as_dict() for s in self.slots]


# --------------------------------------------------------------------
# Pure-functional SEANet decoder
# --------------------------------------------------------------------


class PatchedSEANetResnetBlockDecoder(nn.Module):
    """Streaming SEANet residual block for the decoder.

    Matches `modules/seanet.py:SEANetResnetBlock`. Carries an ordered list
    of (kind, name) items and a ModuleList of patched convs. Each
    non-zero-state conv has a state entry keyed by the unique name assigned
    at build time.
    """

    def __init__(self, ref_block: nn.Module, name_prefix: str):
        super().__init__()
        self.convs = nn.ModuleList()
        self.order: list[tuple[str, str]] = []
        # unique state names for this block's convs
        self.state_names: list[str] = []
        ci = 0
        for sub in ref_block.block:
            if isinstance(sub, nn.ELU):
                self.order.append(("elu", ""))
            else:
                patched = _new_conv1d(sub.conv)
                self.convs.append(patched)
                sname = f"{name_prefix}.conv{ci}"
                self.order.append(("conv", sname))
                self.state_names.append(sname)
                ci += 1

    def forward(
        self, x: torch.Tensor, state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        v = x
        ci = 0
        out_state = dict(state)
        for kind, sname in self.order:
            if kind == "elu":
                v = F.elu(v, alpha=1.0)
            else:
                conv = self.convs[ci]
                ci += 1
                if conv.state_length > 0:
                    prev = state[sname]
                    v, new_prev = conv.pure_forward(v, prev)
                    out_state[sname] = new_prev
                else:
                    # zero-length state: call the conv directly, no state
                    v = conv.conv(v)
        return x + v, out_state


class PatchedSEANetDecoder(nn.Module):
    """Traceable streaming SEANet decoder.

    Walks `SEANetDecoder.model` and produces a parallel list of patched
    equivalents with named state slots. The forward consumes the state
    blob as a dict (name -> tensor) and returns a new dict.

    State keys (produced at __init__ from layer-walk):
      - "dec.idx0.conv"         for first StreamingConv1d (k=7, previous)
      - "dec.idx2.convtr"       for first StreamingConvTranspose1d (partial)
      - "dec.idx3.resblock.conv0"  for residual block's k=3 conv
      - ... in walk order.

    Layers marked `zero_state` are still applied but consume / produce no
    state entry.
    """

    def __init__(self, ref_decoder: nn.Module):
        super().__init__()
        from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
        from pocket_tts.modules.seanet import SEANetResnetBlock

        self.modules_list = nn.ModuleList()
        self.kinds: list[tuple[str, str]] = []  # (kind, state_key)
        # Descriptor list used by DecoderStateLayout to compute offsets.
        self._state_descs: list[dict] = []

        for i, layer in enumerate(ref_decoder.model):
            if isinstance(layer, StreamingConv1d):
                patched = _new_conv1d(layer.conv)
                sname = f"dec.idx{i}.conv"
                self.modules_list.append(patched)
                self.kinds.append(("conv", sname))
                if patched.state_length > 0:
                    self._state_descs.append(dict(
                        name=sname,
                        shape=(1, patched.in_channels, patched.state_length),
                    ))
            elif isinstance(layer, StreamingConvTranspose1d):
                patched = _new_convtr1d(layer.convtr)
                sname = f"dec.idx{i}.convtr"
                self.modules_list.append(patched)
                self.kinds.append(("convtr", sname))
                if patched.state_length > 0:
                    self._state_descs.append(dict(
                        name=sname,
                        shape=(1, patched.out_channels, patched.state_length),
                    ))
            elif isinstance(layer, SEANetResnetBlock):
                prefix = f"dec.idx{i}.resblock"
                patched = PatchedSEANetResnetBlockDecoder(layer, prefix)
                self.modules_list.append(patched)
                self.kinds.append(("resblock", prefix))
                # Extract state descs for its non-zero convs.
                ci = 0
                for sub in layer.block:
                    if isinstance(sub, StreamingConv1d):
                        K = sub.conv.kernel_size[0]
                        S = sub.conv.stride[0]
                        DIL = sub.conv.dilation[0]
                        ke = (K - 1) * DIL + 1
                        TP = ke - S
                        if TP > 0:
                            self._state_descs.append(dict(
                                name=f"{prefix}.conv{ci}",
                                shape=(1, sub.conv.in_channels, TP),
                            ))
                        ci += 1
            elif isinstance(layer, nn.ELU):
                self.modules_list.append(nn.ELU(alpha=1.0))
                self.kinds.append(("elu", ""))
            else:
                raise RuntimeError(f"Unexpected decoder layer: {type(layer)}")

    def state_descriptors(self) -> list[dict]:
        return list(self._state_descs)

    def forward(
        self, x: torch.Tensor, state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out_state = dict(state)
        for mod, (kind, sname) in zip(self.modules_list, self.kinds):
            if kind == "conv":
                if mod.state_length > 0:
                    x, new_prev = mod.pure_forward(x, state[sname])
                    out_state[sname] = new_prev
                else:
                    x = mod.conv(x)
            elif kind == "convtr":
                if mod.state_length > 0:
                    x, new_partial = mod.pure_forward(x, state[sname])
                    out_state[sname] = new_partial
                else:
                    x = mod.convtr(x)
            elif kind == "resblock":
                x, out_state = mod(x, out_state)
            else:  # elu
                x = mod(x)
        return x, out_state


# --------------------------------------------------------------------
# Full Mimi decoder assembly: latent -> audio + state I/O
# --------------------------------------------------------------------


class PatchedConvTrUpsample1d(nn.Module):
    """Non-streaming / streaming depthwise ConvTranspose1d (stride 16, k 32).

    Reference `modules/resample.py:ConvTrUpsample1d` with groups=dimension.
    """

    def __init__(self, ref_upsample: nn.Module):
        super().__init__()
        self.convtr = _new_convtr1d(ref_upsample.convtr.convtr)

    def pure_forward(
        self, x: torch.Tensor, partial_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.convtr.pure_forward(x, partial_in)


class PatchedMimiDecoder(nn.Module):
    """End-to-end patched Mimi decoder, one AR frame per call.

    Inputs:
      latent:       fp32[1, inner_dim=32, 1]   -- one AR frame latent
      state_blob:   fp32[total_elems]          -- packed streaming state
      offset_mask:  fp32[1, S_cap]             -- one-hot for transformer write
      attn_mask:    fp32[1, 1, 1, S_cap]       -- additive fp16-safe
      rope_cos:     fp32[1, 1, 1, head_dim//2] -- this step's RoPE cos
      rope_sin:     fp32[1, 1, 1, head_dim//2]

    Outputs:
      audio:        fp32[1, 1, frame_size=1920]
      state_blob_out: fp32[total_elems]
    """

    def __init__(
        self,
        ref_mimi: nn.Module,
        state_s_cap: int = 256,
        emb_std: torch.Tensor | None = None,
        emb_mean: torch.Tensor | None = None,
    ):
        super().__init__()
        # Capture configuration we need.
        # inner_dim = FlowLM latent dim (32), outer_dim = Mimi transformer
        # / SEANet channel count (512). Reference MimiModel exposes these
        # via the quantizer's (dimension, output_dimension). The upsample's
        # convtr runs at outer_dim -> outer_dim (depthwise, groups=outer).
        self.inner_dim = ref_mimi.quantizer.dimension  # 32
        self.outer_dim = ref_mimi.quantizer.output_dimension  # 512
        self.frame_size = ref_mimi.frame_size  # 1920

        # Unnormalization buffers (FlowLM emits latents in normalized space).
        if emb_std is None:
            emb_std = torch.ones(self.inner_dim)
        if emb_mean is None:
            emb_mean = torch.zeros(self.inner_dim)
        self.register_buffer("emb_std", emb_std.clone().view(1, -1, 1))
        self.register_buffer("emb_mean", emb_mean.clone().view(1, -1, 1))

        # Upsample (stride-16 depthwise).
        self.upsample = PatchedConvTrUpsample1d(ref_mimi.upsample)

        # Quantizer is DummyQuantizer = pointwise Conv1d(inner=32, outer=512).
        # Copy weights into a plain nn.Conv1d for traceability.
        q = ref_mimi.quantizer
        assert q.dimension == self.inner_dim and q.output_dimension == self.outer_dim, (
            f"quantizer dims mismatch: {q.dimension}->{q.output_dimension} "
            f"vs inner={self.inner_dim} outer={self.outer_dim}"
        )
        self.quantizer = nn.Conv1d(
            self.inner_dim, self.outer_dim, kernel_size=1, bias=False,
        )
        with torch.no_grad():
            self.quantizer.weight.copy_(q.output_proj.weight)

        # Decoder transformer (2 layers, context=250).
        mt_config = ref_mimi.decoder_transformer
        # ref_projected_transformer has .transformer (StreamingTransformer) and
        # input_proj/output_projs.
        # Pull config from the reference's transformer layers themselves.
        ref_layer0 = mt_config.transformer.layers[0]
        # LayerScale is a parameter; detect presence.
        ls = getattr(ref_layer0, "layer_scale_1", None)
        layer_scale = None
        if hasattr(ls, "scale") and isinstance(ls.scale, torch.nn.Parameter):
            # Use the initial value from the reference's parameter (it's the
            # learned scale; we'll copy the weights later).
            layer_scale = float(ls.scale.detach().abs().mean().item()) or 1e-2
        d_model = ref_layer0.self_attn.embed_dim
        num_heads = ref_layer0.self_attn.num_heads
        num_layers = len(mt_config.transformer.layers)
        dim_feedforward = ref_layer0.linear1.out_features
        context = ref_layer0.self_attn.context
        self.decoder_transformer = PatchedProjectedTransformer(
            input_dimension=mt_config.input_dimension,
            output_dimensions=mt_config.output_dimensions,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            context=context,
            layer_scale=layer_scale,
        )
        self.decoder_transformer.load_reference_weights(mt_config)

        # SEANet decoder.
        self.seanet = PatchedSEANetDecoder(ref_mimi.decoder)

        # Build the state layout.
        self.state_s_cap = state_s_cap
        self.num_tx_layers = num_layers
        self.tx_num_heads = num_heads
        self.tx_head_dim = d_model // num_heads
        self.layout = DecoderStateLayout(
            seanet_decoder_layers=self.seanet.state_descriptors(),
            outer_dim=self.outer_dim,
            upsample_kernel=self.upsample.convtr.convtr.kernel_size[0],
            upsample_stride=self.upsample.convtr.convtr.stride[0],
            tx_num_layers=num_layers,
            tx_s_cap=state_s_cap,
            tx_num_heads=num_heads,
            tx_head_dim=d_model // num_heads,
        )

    # ---- state (un)packing -----------------------------------------

    def unpack_state(self, blob: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split the flat blob into named state tensors.

        Uses static `narrow` + `view` calls (the offsets/shapes are
        Python ints at trace time) so the traced graph has no Python-int
        extraction from tensors.
        """
        out: dict[str, torch.Tensor] = {}
        for slot in self.layout.slots:
            flat = blob.narrow(0, slot.offset, slot.length)
            out[slot.name] = flat.reshape(slot.shape)
        return out

    def pack_state(self, named: dict[str, torch.Tensor]) -> torch.Tensor:
        """Reverse of unpack_state. Same key set, same order."""
        flats = []
        for slot in self.layout.slots:
            flats.append(named[slot.name].reshape(slot.length))
        return torch.cat(flats, dim=0)

    # ---- forward ---------------------------------------------------

    def forward(
        self,
        latent: torch.Tensor,
        state_in: torch.Tensor,
        scatter_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One AR frame: `latent` → `audio` + updated `state_out`.

        Implements the reference `MimiModel.decode_from_latent(latent, state)`:
          emb = _to_encoder_framerate(latent, state)    # upsample partial state
          (emb,) = decoder_transformer(emb, state)      # KV cache state
          out = decoder(emb, state)                     # SEANet conv states

        The reference also expects the CALLER to feed an unnormalized
        `latent` — the FlowLM loop multiplies by `emb_std` and adds
        `emb_mean` before calling decode_from_latent (`tts_model.py:449`).
        To keep this CoreML graph "just drive it with the FlowLM output",
        we bake emb_std/emb_mean INTO this graph.
        """
        state_named = self.unpack_state(state_in)

        # Un-normalize (from FlowLM's normalized latent space to mimi's).
        x = latent * self.emb_std + self.emb_mean  # [1, 32, 1]

        # Quantizer: pointwise Conv1d(inner=32 -> outer=512).
        x = self.quantizer(x)  # [1, 512, 1]

        # Upsample 16x with partial-state.
        up_partial = state_named["upsample_partial"]
        emb, up_partial_new = self.upsample.pure_forward(x, up_partial)
        # emb: [1, 512, 16]
        state_named["upsample_partial"] = up_partial_new

        # Decoder transformer: process T_q=16 encoder-framerate tokens in
        # one shot using scatter_mask-based KV write.
        emb, kv_out = self.decoder_transformer.forward_step_multi(
            emb, state_named["tx_kv"], scatter_mask, attn_mask, rope_cos, rope_sin,
        )
        state_named["tx_kv"] = kv_out

        # SEANet decoder.
        audio, state_named = self.seanet(emb, state_named)

        state_out = self.pack_state(state_named)
        return audio, state_out
