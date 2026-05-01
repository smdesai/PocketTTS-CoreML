"""Patched SimpleMLPAdaLN (the flow head) with traceable norms.

Reference: `pocket_tts/modules/mlp.py`.

### Landmines

  1. Reference `LayerNorm` (ref `modules/mlp.py:39-55`) is a custom
     reimplementation "because the default one doesn't support jvp"
     (ref comment `mlp.py:40`). Uses `x.mean`, `x.var(unbiased=False)`,
     `(x - mean)/sqrt(var + eps)`. ANE fuses `torch.nn.LayerNorm`; the
     custom one won't fuse. Replaced with `nn.LayerNorm(eps=1e-6)`.
     Parity: `nn.LayerNorm` with `unbiased=False`-equivalent math.

  2. Reference `RMSNorm` (ref `modules/mlp.py:20-36`) uses
        var = eps + x.var(dim=-1, keepdim=True)
        y   = x * (alpha * rsqrt(var))
     with `x.var` defaulting to `unbiased=True` (Bessel correction). The
     canonical RMSNorm formula
        y = x * rsqrt(mean(x**2, dim=-1, keepdim=True) + eps) * alpha
     differs numerically even for zero-mean inputs (Bessel term `N/(N-1)`).
     PARITY DECISION (see docs/phase2_patches.md): we mirror the
     reference formula exactly using `torch.var` with default
     `unbiased=True`. This is traceable and numerically identical to
     the reference. Any ANE fallback is accepted (flow head RMSNorm is
     a 512-dim op run 12.5 times/s — negligible CPU cost).

### Forward signature

    PatchedSimpleMLPAdaLN.forward(c, s, t, x) -> u

Same as reference; stateless.
"""
# ref: modules/mlp.py:39-55 — custom LayerNorm swapped for nn.LayerNorm.
# ref: modules/mlp.py:20-36 — RMSNorm formula kept identical to reference.
# ref: modules/mlp.py:58-83 — TimestepEmbedder.
# ref: modules/mlp.py:86-111 — ResBlock.
# ref: modules/mlp.py:114-131 — FinalLayer.
# ref: modules/mlp.py:134-215 — SimpleMLPAdaLN.

from __future__ import annotations

import math

import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Matches reference `modulate` in mlp.py:16.
    return x * (1 + scale) + shift


class PatchedRMSNorm(nn.Module):
    """RMS normalization with reference-identical formula.

    Reference `_rms_norm` does `var = eps + x.var(dim=-1, keepdim=True)`
    where `torch.var` defaults to unbiased=True (Bessel-corrected). We
    replicate bit-for-bit.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(dim))
        self._dim = int(dim)
        # Bessel correction: `var = sum(sq_diff) / (N - 1)`, so pre-compute
        # `1 / (N - 1)` as a graph constant. N=1 would be degenerate; the
        # reference only applies RMSNorm over hidden dims >= 256.
        assert self._dim > 1, "PatchedRMSNorm requires dim > 1 (Bessel correction)."
        self._bessel_scale = 1.0 / float(self._dim - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        # Manual unbiased variance: `torch.var(dim=-1, unbiased=True)` is
        # not implemented by coremltools 8.1 (RuntimeError: op 'var' not
        # implemented). We replicate it explicitly:
        #   var = sum((x - mean)**2) / (N - 1)
        # The `N / (N - 1)` Bessel factor matches the reference exactly.
        # `N` is the last-dim size, a trace-time integer baked into the
        # graph via `unflatten`/shape tracking without producing aten::Int.
        x32 = x.to(torch.float32)
        mean = x32.mean(dim=-1, keepdim=True)
        diff = x32 - mean
        # `.sum(dim=-1).div(N-1)` would require a Python-int N, which we
        # obtain from the LAST dim at construction time (self._dim).
        sse = (diff * diff).sum(dim=-1, keepdim=True)
        var = sse * self._bessel_scale + self.eps
        y = (x32 * (self.alpha.to(var) * torch.rsqrt(var))).to(x_dtype)
        return y


class PatchedLayerNorm(nn.Module):
    """nn.LayerNorm-backed wrapper mirroring the reference's shape/dtype
    contract. Uses biased variance by default (matching reference
    `x.var(unbiased=False)`).

    Reference's custom LayerNorm formula:
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, unbiased=False, keepdim=True)
        x    = (x - mean) / sqrt(var + eps)
        if weight: x = x * weight + bias

    nn.LayerNorm uses biased variance internally and the same formula,
    differing only in that it lumps (x - mean)/sqrt(var+eps) together
    (potentially better numerics). Tolerances cover any sub-1e-6 drift.
    """

    def __init__(self, channels: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(
            channels, eps=eps, elementwise_affine=elementwise_affine
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class PatchedTimestepEmbedder(nn.Module):
    """Mirrors reference TimestepEmbedder (mlp.py:58-83)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, max_period: int = 10000):
        super().__init__()
        self.linear1 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.norm = PatchedRMSNorm(hidden_size)
        self.frequency_embedding_size = frequency_embedding_size
        half = frequency_embedding_size // 2
        # Pre-computed buffer, same as reference.
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t * self.freqs.to(t.dtype)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        y = self.linear1(embedding)
        y = self.act(y)
        y = self.linear2(y)
        y = self.norm(y)
        return y


class PatchedResBlock(nn.Module):
    """Mirrors reference ResBlock (mlp.py:86-111)."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.in_ln = PatchedLayerNorm(channels, eps=1e-6)
        self.mlp_l1 = nn.Linear(channels, channels, bias=True)
        self.mlp_act = nn.SiLU()
        self.mlp_l2 = nn.Linear(channels, channels, bias=True)
        self.ada_act = nn.SiLU()
        self.ada_linear = nn.Linear(channels, 3 * channels, bias=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gated = self.ada_linear(self.ada_act(y))
        # Static chunk along dim=-1 with size 3; slices, not aten::Int.
        shift_mlp, scale_mlp, gate_mlp = gated.chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp_l2(self.mlp_act(self.mlp_l1(h)))
        return x + gate_mlp * h


class PatchedFinalLayer(nn.Module):
    """Mirrors reference FinalLayer (mlp.py:114-131)."""

    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = PatchedLayerNorm(model_channels, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.ada_act = nn.SiLU()
        self.ada_linear = nn.Linear(model_channels, 2 * model_channels, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gated = self.ada_linear(self.ada_act(c))
        shift, scale = gated.chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PatchedSimpleMLPAdaLN(nn.Module):
    """Patched flow head. Stateless. Signature unchanged from reference
    (c, s, t, x) -> u.

    Weight-compat: exposes `load_reference_weights(ref_mlp)` which copies
    weights from a reference `SimpleMLPAdaLN` into the patched submodules
    (mapping the reference's `nn.Sequential` container into the flattened
    layout here).
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        cond_channels: int,
        num_res_blocks: int,
        num_time_conds: int = 2,
    ):
        super().__init__()
        assert num_time_conds != 1, "Reference asserts this at mlp.py:162."
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.num_time_conds = num_time_conds

        self.time_embed = nn.ModuleList(
            [PatchedTimestepEmbedder(model_channels) for _ in range(num_time_conds)]
        )
        self.cond_embed = nn.Linear(cond_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList(
            [PatchedResBlock(model_channels) for _ in range(num_res_blocks)]
        )
        self.final_layer = PatchedFinalLayer(model_channels, out_channels)

    @torch.no_grad()
    def load_reference_weights(self, ref_mlp: nn.Module) -> None:
        """Copy weights from a reference SimpleMLPAdaLN.

        Reference layout mapping:
            ref.input_proj.weight/bias              -> self.input_proj
            ref.cond_embed.weight/bias              -> self.cond_embed
            ref.time_embed[i].mlp[0]  (Linear)      -> self.time_embed[i].linear1
            ref.time_embed[i].mlp[2]  (Linear)      -> self.time_embed[i].linear2
            ref.time_embed[i].mlp[3]  (RMSNorm.alpha) -> self.time_embed[i].norm.alpha
            ref.time_embed[i].freqs (buffer)        -> self.time_embed[i].freqs
            ref.res_blocks[j].in_ln.weight/bias     -> self.res_blocks[j].in_ln.norm.weight/bias
            ref.res_blocks[j].mlp[0]                -> self.res_blocks[j].mlp_l1
            ref.res_blocks[j].mlp[2]                -> self.res_blocks[j].mlp_l2
            ref.res_blocks[j].adaLN_modulation[1]   -> self.res_blocks[j].ada_linear
            ref.final_layer.norm_final (no affine)  -> self.final_layer.norm_final
            ref.final_layer.linear                  -> self.final_layer.linear
            ref.final_layer.adaLN_modulation[1]     -> self.final_layer.ada_linear
        """
        self.input_proj.weight.copy_(ref_mlp.input_proj.weight)
        self.input_proj.bias.copy_(ref_mlp.input_proj.bias)
        self.cond_embed.weight.copy_(ref_mlp.cond_embed.weight)
        self.cond_embed.bias.copy_(ref_mlp.cond_embed.bias)

        for i, ref_te in enumerate(ref_mlp.time_embed):
            # ref_te.mlp is Sequential(Linear, SiLU, Linear, RMSNorm).
            ref_l1 = ref_te.mlp[0]
            ref_l2 = ref_te.mlp[2]
            ref_rms = ref_te.mlp[3]
            self.time_embed[i].linear1.weight.copy_(ref_l1.weight)
            self.time_embed[i].linear1.bias.copy_(ref_l1.bias)
            self.time_embed[i].linear2.weight.copy_(ref_l2.weight)
            self.time_embed[i].linear2.bias.copy_(ref_l2.bias)
            self.time_embed[i].norm.alpha.copy_(ref_rms.alpha)
            self.time_embed[i].freqs.copy_(ref_te.freqs)

        for j, ref_rb in enumerate(ref_mlp.res_blocks):
            # ref_rb.in_ln has weight/bias.
            self.res_blocks[j].in_ln.norm.weight.copy_(ref_rb.in_ln.weight)
            self.res_blocks[j].in_ln.norm.bias.copy_(ref_rb.in_ln.bias)
            self.res_blocks[j].mlp_l1.weight.copy_(ref_rb.mlp[0].weight)
            self.res_blocks[j].mlp_l1.bias.copy_(ref_rb.mlp[0].bias)
            self.res_blocks[j].mlp_l2.weight.copy_(ref_rb.mlp[2].weight)
            self.res_blocks[j].mlp_l2.bias.copy_(ref_rb.mlp[2].bias)
            # adaLN_modulation is Sequential(SiLU, Linear).
            self.res_blocks[j].ada_linear.weight.copy_(
                ref_rb.adaLN_modulation[1].weight
            )
            self.res_blocks[j].ada_linear.bias.copy_(ref_rb.adaLN_modulation[1].bias)

        # final_layer.norm_final: elementwise_affine=False (no weight/bias).
        # Nothing to copy for the norm itself. Linear + adaLN_modulation only.
        self.final_layer.linear.weight.copy_(ref_mlp.final_layer.linear.weight)
        self.final_layer.linear.bias.copy_(ref_mlp.final_layer.linear.bias)
        self.final_layer.ada_linear.weight.copy_(
            ref_mlp.final_layer.adaLN_modulation[1].weight
        )
        self.final_layer.ada_linear.bias.copy_(
            ref_mlp.final_layer.adaLN_modulation[1].bias
        )

    def forward(
        self,
        c: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        ts = [s, t]
        x = self.input_proj(x)
        # Average the two timestep embeddings, matching reference.
        t_combined = (self.time_embed[0](ts[0]) + self.time_embed[1](ts[1])) / self.num_time_conds
        c_emb = self.cond_embed(c)
        y = t_combined + c_emb
        for block in self.res_blocks:
            x = block(x, y)
        return self.final_layer(x, y)
