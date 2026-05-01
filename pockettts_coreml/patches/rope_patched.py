"""RoPE with pre-computed cos/sin tables.

The reference `pocket_tts/modules/rope.py:apply_rope` computes the cos/sin
tables inline via `torch.arange(T)`, `torch.exp(...)`, `ts += offset` —
integer-valued tensor arithmetic that materializes as `aten::Int` in a
traced graph and is a known CoreML blocker.

This patch splits RoPE into two halves:

  - `build_rope_tables(max_context, head_dim, max_period)`:
        Python-side, runs once at module load. Produces
        `cos_table, sin_table` each shaped `[max_context, head_dim // 2]`.
        NOT called inside any traced graph.

  - `apply_rope(q, k, cos, sin)`:
        Pure tensor op. `cos`/`sin` are inputs already sliced/broadcast
        to the right shape. Fully traceable, zero integer ops.

At inference, Swift keeps `cos_table`/`sin_table` as pre-loaded buffers
and slices `[offset : offset+T, :]` before each call.

Parity vs reference: with identical `offset`, `T`, `max_period`, and
identical input `q, k`, the output must match to atol=1e-6 in fp32.
"""
# ref: modules/rope.py:6-58 — relocate arange/exp/ts math out of the traced graph.
from __future__ import annotations

import math

import torch


def build_rope_tables(
    max_context: int,
    head_dim: int,
    max_period: float = 10_000.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin lookup tables covering positions [0, max_context).

    Returns:
        cos, sin: shape [max_context, head_dim // 2] in `dtype` on `device`.
    """
    assert head_dim > 0 and head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    assert max_context > 0
    assert max_period > 0

    half = head_dim // 2
    # Use fp32 internally for table construction; cast at the end. This
    # matches the reference which does its `freqs` math in fp32 regardless
    # of the attention dtype.
    ds = torch.arange(half, device=device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / head_dim))  # [half]
    ts = torch.arange(max_context, device=device, dtype=torch.float32).view(-1, 1)  # [T, 1]
    args = ts * freqs  # [T, half]
    cos = torch.cos(args).to(dtype)
    sin = torch.sin(args).to(dtype)
    return cos, sin


def slice_rope_tables(
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    offset: int,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-trace helper: extract [offset : offset+length] rows and reshape
    to the `[1, T, 1, head_dim // 2]` broadcast shape `apply_rope` expects.

    `offset` and `length` are Python ints — called OUTSIDE the traced graph.
    """
    assert offset >= 0
    assert length > 0
    cos = cos_table[offset : offset + length].unsqueeze(0).unsqueeze(2)
    sin = sin_table[offset : offset + length].unsqueeze(0).unsqueeze(2)
    return cos, sin


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE rotation to `q` and `k` using pre-computed `cos`/`sin`.

    Args:
        q: shape [B, T, H, D]
        k: shape [B, T, Hk, D]  (Hk may equal H or differ for grouped query)
        cos: shape [1, T, 1, D // 2]  (broadcastable across heads + batch)
        sin: shape [1, T, 1, D // 2]

    Returns:
        q_rot, k_rot: same shapes as inputs.

    IMPORTANT trace-compat note: we avoid `torch.reshape(B, T, H, D//2, 2)`
    because pulling `B, T, H, D` out of `q.shape` and passing them back
    as Python ints produces `aten::Int` ops in the traced graph (one per
    extracted dim). Instead we use `unflatten(-1, (-1, 2))` which takes
    a negative-stride arg for "infer the rest" and passes a literal 2 as
    the fixed size — no tensor-derived integer plumbing.
    """
    # Split the last dim into (D//2, 2) without reading `q.shape` as
    # Python ints. `unflatten(-1, (-1, 2))` produces shape [..., D//2, 2].
    q_pairs = q.unflatten(-1, (-1, 2))
    k_pairs = k.unflatten(-1, (-1, 2))

    qr = q_pairs[..., 0]
    qi = q_pairs[..., 1]
    kr = k_pairs[..., 0]
    ki = k_pairs[..., 1]

    orig_dtype = q.dtype
    qr_f = qr.to(torch.float32)
    qi_f = qi.to(torch.float32)
    kr_f = kr.to(torch.float32)
    ki_f = ki.to(torch.float32)
    cos_f = cos.to(torch.float32)
    sin_f = sin.to(torch.float32)

    qor = qr_f * cos_f - qi_f * sin_f
    qoi = qr_f * sin_f + qi_f * cos_f
    kor = kr_f * cos_f - ki_f * sin_f
    koi = kr_f * sin_f + ki_f * cos_f

    # Restack and re-flatten the last two dims back to D. `flatten(-2)`
    # collapses the last two axes without Python int extraction.
    qo = torch.stack([qor.to(orig_dtype), qoi.to(orig_dtype)], dim=-1).flatten(-2)
    ko = torch.stack([kor.to(orig_dtype), koi.to(orig_dtype)], dim=-1).flatten(-2)
    return qo, ko
