"""Phase 2 gate: each patched submodel traces cleanly with zero `aten::Int`.

`aten::Int` in a traced graph is the canonical CoreML-LLM blocker (per
external_research.md §B.1 item 11). It comes from any `int(tensor.item())`
or integer arithmetic on tensor-derived scalars inside a traced forward.

For each patched module we:
  1. Build a tiny instance with random weights.
  2. Construct `example_inputs` (fixed shapes).
  3. `torch.jit.trace(module, example_inputs, strict=False)`.
  4. Assert `"aten::Int" not in str(traced.graph)`.
  5. Assert traced output matches eager output at atol=1e-6.

These tests are CPU-only, deterministic, and run in <5 s total.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_REFERENCE_DIR = Path(__file__).resolve().parent.parent / "pockettts_coreml" / "reference"
if str(_REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_REFERENCE_DIR))


def _assert_no_aten_int(graph_str: str, label: str) -> None:
    if "aten::Int" in graph_str:
        # Extract a snippet for diagnostics.
        lines = [ln for ln in graph_str.splitlines() if "aten::Int" in ln]
        raise AssertionError(
            f"[{label}] traced graph contains {len(lines)} aten::Int op(s):\n"
            + "\n".join(lines[:5])
        )


def _assert_traced_matches_eager(
    traced: torch.jit.ScriptModule,
    eager: nn.Module,
    inputs: tuple,
    label: str,
    atol: float = 1e-6,
) -> None:
    with torch.no_grad():
        eager_out = eager(*inputs)
        traced_out = traced(*inputs)
    # Normalize to tuple for iteration.
    if not isinstance(eager_out, tuple):
        eager_out = (eager_out,)
    if not isinstance(traced_out, tuple):
        traced_out = (traced_out,)
    assert len(eager_out) == len(traced_out), f"[{label}] output-count mismatch"
    for i, (e, t) in enumerate(zip(eager_out, traced_out)):
        diff = (e.float() - t.float()).abs().max().item()
        if not torch.allclose(e, t, atol=atol, rtol=atol):
            raise AssertionError(
                f"[{label}/out{i}] traced-vs-eager mismatch: max_abs={diff:.3e}"
            )


def _seed(s: int = 42) -> None:
    torch.manual_seed(s)
    torch.set_num_threads(1)


# ------------------------------------------------------------------
# 1. RoPE apply_rope (as a plain function-module wrapper)
# ------------------------------------------------------------------


def test_trace_rope_patched() -> None:
    _seed(42)
    from pockettts_coreml.patches.rope_patched import apply_rope, build_rope_tables

    B, T, H, D = 1, 1, 16, 64

    class _RopeModule(nn.Module):
        def forward(self, q, k, cos, sin):
            q_out, k_out = apply_rope(q, k, cos, sin)
            return q_out, k_out

    mod = _RopeModule()
    mod.eval()

    cos_table, sin_table = build_rope_tables(max_context=256, head_dim=D)
    cos = cos_table[5 : 5 + T].unsqueeze(0).unsqueeze(2)
    sin = sin_table[5 : 5 + T].unsqueeze(0).unsqueeze(2)
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, H, D)

    traced = torch.jit.trace(mod, (q, k, cos, sin), strict=False)
    _assert_no_aten_int(str(traced.graph), "rope")
    _assert_traced_matches_eager(traced, mod, (q, k, cos, sin), "rope")


# ------------------------------------------------------------------
# 2. Patched MHA (AR-step path)
# ------------------------------------------------------------------


def test_trace_patched_mha_ar_step() -> None:
    _seed(43)
    from pockettts_coreml.patches.rope_patched import (
        build_rope_tables, slice_rope_tables,
    )
    from pockettts_coreml.patches.transformer_patched import (
        PatchedStreamingMultiheadAttention,
        build_additive_attention_mask_step,
        build_one_hot_offset_mask,
    )

    embed_dim, num_heads = 256, 8
    head_dim = embed_dim // num_heads
    S_cap = 16

    mha = PatchedStreamingMultiheadAttention(embed_dim, num_heads)
    mha.eval()

    x = torch.randn(1, 1, embed_dim)
    kv_cache = torch.zeros((2, 1, S_cap, num_heads, head_dim))
    offset_mask = build_one_hot_offset_mask(offset=3, s_capacity=S_cap)
    attn_mask = build_additive_attention_mask_step(offset=3, s_capacity=S_cap)
    cos_table, sin_table = build_rope_tables(max_context=S_cap, head_dim=head_dim)
    cos, sin = slice_rope_tables(cos_table, sin_table, offset=3, length=1)

    inputs = (x, kv_cache, offset_mask, attn_mask, cos, sin)
    traced = torch.jit.trace(mha, inputs, strict=False)
    _assert_no_aten_int(str(traced.graph), "patched_mha")
    _assert_traced_matches_eager(traced, mha, inputs, "patched_mha")


# ------------------------------------------------------------------
# 3. Patched SimpleMLPAdaLN (flow head)
# ------------------------------------------------------------------


def test_trace_patched_flow_mlp() -> None:
    _seed(44)
    from pockettts_coreml.patches.mlp_patched import PatchedSimpleMLPAdaLN

    mlp = PatchedSimpleMLPAdaLN(
        in_channels=32, model_channels=128, out_channels=32,
        cond_channels=256, num_res_blocks=2, num_time_conds=2,
    )
    mlp.eval()
    c = torch.randn(1, 256)
    s = torch.zeros(1, 1)
    t = torch.ones(1, 1)
    x = torch.randn(1, 32)
    inputs = (c, s, t, x)
    traced = torch.jit.trace(mlp, inputs, strict=False)
    _assert_no_aten_int(str(traced.graph), "flow_mlp")
    _assert_traced_matches_eager(traced, mlp, inputs, "flow_mlp")


# ------------------------------------------------------------------
# 4. Patched FlowLM main path (AR step)
# ------------------------------------------------------------------


def test_trace_patched_flow_lm_main_ar_step() -> None:
    _seed(45)
    from pockettts_coreml.patches.flow_lm_patched import PatchedFlowLMMain
    from pockettts_coreml.patches.rope_patched import (
        build_rope_tables, slice_rope_tables,
    )
    from pockettts_coreml.patches.transformer_patched import (
        build_additive_attention_mask_step,
        build_one_hot_offset_mask,
    )

    # Tiny dims for trace-speed.
    d_model, num_heads, num_layers = 128, 8, 2
    head_dim = d_model // num_heads
    S_cap = 16
    ldim = 32

    main = PatchedFlowLMMain(
        ldim=ldim, d_model=d_model, num_heads=num_heads,
        num_layers=num_layers, dim_feedforward=4 * d_model,
    )
    main.eval()

    sequence = torch.randn(1, 1, ldim)
    text_embeddings = torch.zeros(1, 0, d_model)  # empty text in hot AR path
    kv_caches = torch.zeros((num_layers, 2, 1, S_cap, num_heads, head_dim))
    offset_mask = build_one_hot_offset_mask(offset=2, s_capacity=S_cap)
    attn_mask = build_additive_attention_mask_step(offset=2, s_capacity=S_cap)
    cos_table, sin_table = build_rope_tables(max_context=S_cap, head_dim=head_dim)
    cos, sin = slice_rope_tables(cos_table, sin_table, offset=2, length=1)

    inputs = (sequence, text_embeddings, kv_caches, offset_mask, attn_mask, cos, sin)
    traced = torch.jit.trace(main, inputs, strict=False)
    _assert_no_aten_int(str(traced.graph), "flow_lm_main")
    _assert_traced_matches_eager(traced, main, inputs, "flow_lm_main")


# ------------------------------------------------------------------
# 5. Patched FlowLM flow head
# ------------------------------------------------------------------


def test_trace_patched_flow_lm_flow() -> None:
    _seed(46)
    from pockettts_coreml.patches.flow_lm_patched import PatchedFlowLMFlow

    flow = PatchedFlowLMFlow(ldim=32, d_model=128, flow_dim=128, flow_depth=2)
    flow.eval()

    ctx = torch.randn(1, 128)
    s = torch.zeros(1, 1)
    t = torch.ones(1, 1)
    x = torch.randn(1, 32)

    # `num_steps_inv` is a Python float constant, baked into the trace.
    # Wrap in a tiny module to fix it at trace time.
    class _FlowWrap(nn.Module):
        def __init__(self, f):
            super().__init__()
            self.f = f
        def forward(self, c, s_, t_, x_):
            return self.f(c, s_, t_, x_, num_steps_inv=1.0)

    wrapped = _FlowWrap(flow)
    wrapped.eval()
    inputs = (ctx, s, t, x)
    traced = torch.jit.trace(wrapped, inputs, strict=False)
    _assert_no_aten_int(str(traced.graph), "flow_lm_flow")
    _assert_traced_matches_eager(traced, wrapped, inputs, "flow_lm_flow")


# ------------------------------------------------------------------
# 6. Mimi streaming conv pure_forward
# ------------------------------------------------------------------


def test_trace_patched_streaming_conv1d() -> None:
    _seed(47)
    from pockettts_coreml.patches.mimi_patched import PatchedStreamingConv1d

    conv = PatchedStreamingConv1d(
        in_channels=8, out_channels=16, kernel_size=7, stride=1,
        dilation=1, groups=1, bias=True,
    )
    conv.eval()

    class _Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x, prev):
            return self.m.pure_forward(x, prev)

    wrapped = _Wrap(conv)
    wrapped.eval()

    x = torch.randn(1, 8, 2)
    prev = conv.init_state(batch_size=1)
    inputs = (x, prev)
    traced = torch.jit.trace(wrapped, inputs, strict=False)
    _assert_no_aten_int(str(traced.graph), "sconv1d")
    _assert_traced_matches_eager(traced, wrapped, inputs, "sconv1d")


def test_trace_patched_streaming_convtranspose1d() -> None:
    _seed(48)
    from pockettts_coreml.patches.mimi_patched import PatchedStreamingConvTranspose1d

    convtr = PatchedStreamingConvTranspose1d(
        in_channels=16, out_channels=8, kernel_size=12, stride=6,
        groups=1, bias=True,
    )
    convtr.eval()

    class _Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x, partial):
            return self.m.pure_forward(x, partial)

    wrapped = _Wrap(convtr)
    wrapped.eval()

    x = torch.randn(1, 16, 3)
    partial = convtr.init_state(batch_size=1)
    inputs = (x, partial)
    traced = torch.jit.trace(wrapped, inputs, strict=False)
    _assert_no_aten_int(str(traced.graph), "sconvtr")
    _assert_traced_matches_eager(traced, wrapped, inputs, "sconvtr")
