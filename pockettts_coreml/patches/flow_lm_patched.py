"""Traceable FlowLM main + flow head split.

Replaces the in-graph logic of `pocket_tts/models/flow_lm.py:FlowLMModel.forward`
and `lsd_decode`. The refactor cleaves the model along the natural
Phase-3 package boundary:

  - `FlowLMMain` wraps: input_linear -> [text+audio concat] -> 6-layer
    patched transformer -> out_norm -> slice-last -> out_eos. Its output
    is `(ctx: [B, d_model], eos_logit: [B, 1], kv_caches_out)` — the
    "context vector and EOS indicator" consumed by the flow head.

  - `FlowLMFlow` wraps: the single lsd_decode step at `num_steps=1`
    (the production default): `flow_net(ctx, s=0, t=1, noise) -> u`;
    returning `x_1 = noise + u * (1/1) = noise + u` (identical to the
    reference `lsd_decode` at N=1).

### Landmines eliminated

  1. `torch.where(isnan(sequence), bos_emb, sequence)`
     (ref `models/flow_lm.py:121`) is gone. Caller MUST pass `sequence`
     explicitly populated with `bos_emb` on step 0 (no NaN convention).
     The `build_initial_sequence` helper (Python-side) creates the
     correct tensor for step 0.

  2. In-graph noise sampling
     (ref `models/flow_lm.py:131-137`, `torch.nn.init.normal_` /
     `trunc_normal_`) is gone. Caller passes `noise: fp32[B, ldim]` as
     an input. Pre-sampled outside the graph (Swift's `vDSP_vgauss` at
     runtime; deterministic tensor from test seed in parity tests).

  3. KV cache state as explicit I/O — see transformer_patched.py.

### Forward signatures

    flow_lm_main_forward(
        sequence,                # fp32[B, 1, ldim]           - prev latent (or bos_emb on step 0)
        text_embeddings,         # fp32[B, S_prefix, d_model] - prefix (text/audio/empty)
        kv_caches_in,            # fp32[L, 2, B, S_cap, H, D]
        offset_mask,             # fp32[B, S_cap]             - one-hot for AR step write slot
        attn_mask,               # fp32[1, 1, T_total, S_cap] - additive, T_total = S_prefix + 1
        rope_cos,                # fp32[1, T_total, 1, D//2]
        rope_sin,                # fp32[1, T_total, 1, D//2]
    ) -> (
        ctx,                     # fp32[B, d_model]           - last-token context vec
        eos_logit,               # fp32[B, 1]                 - out_eos(ctx); threshold in Swift
        kv_caches_out,           # fp32[L, 2, B, S_cap, H, D]
    )

NOTE: This `forward` is the AR-step path. Prefill (text/audio prefix) is
modeled separately via `flow_lm_main_prefill_forward` using
`scatter_mask` rather than `offset_mask`. The two paths are both pure,
traceable, and numerically match the reference's unified `forward`.

    flow_lm_flow_forward(
        ctx,    # fp32[B, d_model]
        s,     # fp32[B, 1]       - LSD start time (0.0 at N=1)
        t,     # fp32[B, 1]       - LSD end time   (1.0 at N=1)
        x,     # fp32[B, ldim]    - the "noise" input (pre-sampled)
        num_steps_inv,  # Python scalar, 1.0 at N=1
    ) -> next_latent  # fp32[B, ldim] = x + u * num_steps_inv, matching lsd_decode.
"""
# ref: models/flow_lm.py:95-139 — split into main + flow + eliminate NaN/noise.
# ref: models/flow_lm.py:19-40 — lsd_decode inlined for N=1 default.

from __future__ import annotations

import torch
import torch.nn as nn

from pockettts_coreml.patches.mlp_patched import PatchedSimpleMLPAdaLN
from pockettts_coreml.patches.transformer_patched import (
    PatchedStreamingTransformer,
)


class PatchedFlowLMMain(nn.Module):
    """Main-transformer path of FlowLMModel, without NaN/noise landmines.

    Holds:
      - `bos_emb`: ldim-d learned BOS parameter.
      - `input_linear`: Linear(ldim, d_model, bias=False).
      - `transformer`: patched 6-layer streaming transformer.
      - `out_norm`: nn.LayerNorm(d_model, eps=1e-5).
      - `out_eos`: Linear(d_model, 1).
      - `emb_std`, `emb_mean`: ldim buffers used by Swift for
        un-normalization before mimi decode (exposed here for weight
        transfer but not used in the forward path).
    """

    def __init__(
        self,
        ldim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dtype: torch.dtype = torch.float32,
        use_fp32_softmax: bool = True,
    ):
        super().__init__()
        self.ldim = ldim
        self.d_model = d_model
        self.bos_emb = nn.Parameter(torch.zeros(ldim, dtype=dtype))
        self.input_linear = nn.Linear(ldim, d_model, bias=False, dtype=dtype)
        self.transformer = PatchedStreamingTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            layer_scale=None,  # FlowLM transformer has no layer_scale.
            use_fp32_softmax=use_fp32_softmax,
        )
        self.out_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.out_eos = nn.Linear(d_model, 1, dtype=dtype)

        # Carry un-normalization buffers through for Phase-4 Swift use.
        self.register_buffer("emb_std", torch.ones(ldim, dtype=dtype))
        self.register_buffer("emb_mean", torch.zeros(ldim, dtype=dtype))

    @torch.no_grad()
    def load_reference_weights(self, ref_flow_lm: nn.Module) -> None:
        """Copy weights from reference FlowLMModel (just the main-path
        slice — bos_emb, input_linear, transformer, out_norm, out_eos,
        emb_std/mean). Does NOT copy flow_net or conditioner.
        """
        self.bos_emb.copy_(ref_flow_lm.bos_emb)
        self.input_linear.weight.copy_(ref_flow_lm.input_linear.weight)
        self.transformer.load_reference_weights(ref_flow_lm.transformer)
        self.out_norm.weight.copy_(ref_flow_lm.out_norm.weight)
        self.out_norm.bias.copy_(ref_flow_lm.out_norm.bias)
        self.out_eos.weight.copy_(ref_flow_lm.out_eos.weight)
        self.out_eos.bias.copy_(ref_flow_lm.out_eos.bias)
        self.emb_std.copy_(ref_flow_lm.emb_std)
        self.emb_mean.copy_(ref_flow_lm.emb_mean)

    def forward(
        self,
        sequence: torch.Tensor,
        text_embeddings: torch.Tensor,
        kv_caches_in: torch.Tensor,
        offset_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """AR-step forward.

        Assumes `sequence.shape[1] == 1` (one new latent per step). The
        caller sets `sequence = bos_emb[None, None, :]` on step 0.

        `text_embeddings.shape[1]` is typically 0 in the AR hot path (no
        new text to mix in), but can be >0 on the step that transitions
        out of prefill. The concat `[text_embeddings, input_]` along
        dim=1 matches the reference `backbone` (flow_lm.py:150).
        """
        # NOTE: no isnan/torch.where. Caller is responsible for passing
        # bos_emb on step 0 (see build_initial_sequence helper).
        input_ = self.input_linear(sequence)
        x = torch.cat([text_embeddings, input_], dim=1)  # [B, T_total, d_model]

        # Run the transformer. offset_mask applies to the LAST (new) token;
        # attn_mask is sized for T_total rows = text_embeddings.shape[1] + 1.
        # For the hot AR path, text_embeddings.shape[1] == 0.
        x, kv_caches_out = self.transformer(
            x, kv_caches_in, offset_mask, attn_mask, rope_cos, rope_sin
        )
        x = self.out_norm(x)
        # Slice last `sequence.shape[1]` tokens (= 1). Static size slice.
        ctx_full = x[:, -1, :]  # [B, d_model]
        eos_logit = self.out_eos(ctx_full)  # [B, 1]
        return ctx_full, eos_logit, kv_caches_out

    def forward_prefill(
        self,
        sequence: torch.Tensor,       # fp32[B, 1, ldim] or [B, S_audio, ldim]
        text_embeddings: torch.Tensor,  # fp32[B, S_text, d_model]
        kv_caches_in: torch.Tensor,
        scatter_mask: torch.Tensor,    # fp32[B, S_cap, T_total]
        attn_mask: torch.Tensor,       # fp32[1, 1, T_total, S_cap]
        rope_cos: torch.Tensor,        # fp32[1, T_total, 1, D//2]
        rope_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prefill path: `T_total = text_embeddings.shape[1] + sequence.shape[1]`.

        Used when Swift is filling the cache with text (or text+audio
        conditioning) tokens before the AR hot loop begins. Uses
        scatter_mask semantics (a [B, S_cap, T_total] one-hot matrix).
        """
        input_ = self.input_linear(sequence)
        x = torch.cat([text_embeddings, input_], dim=1)
        x, kv_caches_out = self.transformer.forward_prefill(
            x, kv_caches_in, scatter_mask, attn_mask, rope_cos, rope_sin
        )
        x = self.out_norm(x)
        ctx_full = x[:, -1, :]
        eos_logit = self.out_eos(ctx_full)
        return ctx_full, eos_logit, kv_caches_out


class PatchedFlowLMFlow(nn.Module):
    """Flow-head path of FlowLMModel at num_steps=1.

    This is a thin wrapper around `PatchedSimpleMLPAdaLN` that performs
    ONE lsd_decode step and returns `noise + u * (1/num_steps)`.

    At `num_steps=1` (production default), s=0.0, t=1.0 are known
    constants; the caller passes them as `[B, 1]` tensors so they remain
    fp32 traceable inputs (rather than being baked into the graph —
    slightly more flexible for Phase-3 N>1 variants).
    """

    def __init__(
        self,
        ldim: int,
        d_model: int,
        flow_dim: int,
        flow_depth: int,
    ):
        super().__init__()
        self.flow_net = PatchedSimpleMLPAdaLN(
            in_channels=ldim,
            model_channels=flow_dim,
            out_channels=ldim,
            cond_channels=d_model,
            num_res_blocks=flow_depth,
            num_time_conds=2,
        )

    @torch.no_grad()
    def load_reference_weights(self, ref_flow_lm: nn.Module) -> None:
        self.flow_net.load_reference_weights(ref_flow_lm.flow_net)

    def forward(
        self,
        ctx: torch.Tensor,     # [B, d_model]
        s: torch.Tensor,       # [B, 1]  (0.0 at num_steps=1)
        t: torch.Tensor,       # [B, 1]  (1.0 at num_steps=1)
        x: torch.Tensor,       # [B, ldim]  (the sampled noise)
        num_steps_inv: float,  # 1.0 at num_steps=1
    ) -> torch.Tensor:
        """Single lsd_decode step. Returns `x_1 = x + u * num_steps_inv`.

        For `num_steps=1` the reference loop runs exactly once with
        `s=0/1=0.0`, `t=1/1=1.0`, and `current += flow_dir / 1`. This
        matches our formula.
        """
        u = self.flow_net(ctx, s, t, x)
        return x + u * num_steps_inv


# ------------------------------------------------------------------
# Pre-trace helpers (Python-side)
# ------------------------------------------------------------------


def build_initial_sequence(bos_emb: torch.Tensor) -> torch.Tensor:
    """Step-0 `sequence` input: the BOS embedding broadcast to [1, 1, ldim]."""
    assert bos_emb.dim() == 1
    return bos_emb.unsqueeze(0).unsqueeze(0).contiguous()


def sample_noise(ldim: int, std: float, clamp: float | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """Pre-trace noise sampler matching reference's in-graph distribution.

    Reference `flow_lm.py:131-137` samples fp32 `[1, ldim]` gaussian with
    stddev = sqrt(temp). If `noise_clamp` is not None, uses
    `trunc_normal_(a=-clamp, b=+clamp)`. This helper matches that exactly
    so fixed-seed tests can reproduce the reference RNG draw.
    """
    noise = torch.empty((1, ldim), dtype=torch.float32)
    if clamp is None:
        torch.nn.init.normal_(noise, mean=0.0, std=std, generator=generator)
    else:
        torch.nn.init.trunc_normal_(noise, mean=0.0, std=std, a=-clamp, b=clamp, generator=generator)
    return noise
