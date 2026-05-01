"""Patched-module parity harness (Phase 2).

For each patched submodel:
 1. Instantiate patched module.
 2. Instantiate the corresponding reference module.
 3. Copy weights from reference into patched.
 4. Run both on random inputs + 1 oracle-derived input.
 5. Assert outputs match to plan-mandated tolerances.

Tolerances (plan §Phase 2):
    flow_lm_main:   atol=1e-5, rtol=1e-5
    flow_lm_flow:   atol=1e-5, rtol=1e-5
    mimi_encoder:   atol=1e-4, rtol=1e-4   (skipped: golden empty, Phase 2 doesn't need it)
    mimi_decoder:   atol=1e-4, rtol=1e-4
    text_conditioner: atol=0  (not patched; sanity check only)

All tests are CPU-only and seeded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Make the vendored reference importable as `pocket_tts`.
_REFERENCE_DIR = Path(__file__).resolve().parent.parent / "pockettts_coreml" / "reference"
if str(_REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_REFERENCE_DIR))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _seed_all(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _tight_allclose(
    actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float, label: str = ""
) -> None:
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = expected.float().abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    ok = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    if not ok:
        raise AssertionError(
            f"[{label}] parity mismatch: max_abs={max_abs:.3e}, "
            f"max_rel={max_rel:.3e}, atol={atol}, rtol={rtol}, "
            f"shape={tuple(actual.shape)}"
        )


# ------------------------------------------------------------------
# Patch 2: RoPE parity vs reference apply_rope
# ------------------------------------------------------------------


@pytest.mark.parametrize("offset,T", [(0, 1), (5, 1), (20, 3), (0, 7)])
def test_rope_patched_parity(offset: int, T: int) -> None:
    """apply_rope(q,k,cos,sin) with pre-computed tables must match the
    reference `apply_rope(q, k, offset, max_period)`.
    """
    _seed_all(42)
    from pocket_tts.modules.rope import apply_rope as ref_apply_rope

    from pockettts_coreml.patches.rope_patched import (
        apply_rope as patched_apply_rope,
        build_rope_tables,
        slice_rope_tables,
    )

    B, H, D = 1, 16, 64
    max_period = 10_000.0
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, H, D)

    # Reference: accepts offset as int.
    q_ref, k_ref = ref_apply_rope(q, k, offset=offset, max_period=max_period)

    # Patched: build tables once, slice, apply.
    max_context = 256
    cos_table, sin_table = build_rope_tables(max_context, D, max_period)
    cos, sin = slice_rope_tables(cos_table, sin_table, offset=offset, length=T)
    q_pt, k_pt = patched_apply_rope(q, k, cos, sin)

    _tight_allclose(q_pt, q_ref, atol=1e-6, rtol=1e-6, label="rope/q")
    _tight_allclose(k_pt, k_ref, atol=1e-6, rtol=1e-6, label="rope/k")


# ------------------------------------------------------------------
# Patch 1: StreamingMultiheadAttention parity
# ------------------------------------------------------------------


def _build_ref_mha(embed_dim: int, num_heads: int, seed: int = 0):
    from pocket_tts.modules.rope import RotaryEmbedding
    from pocket_tts.modules.stateful_module import init_states
    from pocket_tts.modules.transformer import StreamingMultiheadAttention

    torch.manual_seed(seed)
    rope = RotaryEmbedding(max_period=10_000.0)
    mha = StreamingMultiheadAttention(
        embed_dim=embed_dim, num_heads=num_heads, rope=rope, context=None
    )
    mha.eval()
    # Must set _module_absolute_name for .get_state.
    mha._module_absolute_name = "root"
    return mha


def test_patched_mha_parity_ar_step() -> None:
    """T_q=1 AR step: patched MHA with mask-based KV write matches reference."""
    _seed_all(7)
    embed_dim, num_heads = 1024, 16
    head_dim = embed_dim // num_heads
    S_cap = 32

    ref_mha = _build_ref_mha(embed_dim, num_heads)

    from pockettts_coreml.patches.rope_patched import (
        build_rope_tables,
        slice_rope_tables,
    )
    from pockettts_coreml.patches.transformer_patched import (
        PatchedStreamingMultiheadAttention,
        build_additive_attention_mask_step,
        build_one_hot_offset_mask,
    )

    patched = PatchedStreamingMultiheadAttention(embed_dim, num_heads)
    patched.eval()
    patched.load_reference_weights(ref_mha)

    # Initialize reference state + prefill up to offset=5 with random
    # K/V values, so we're testing write INTO a partially-full cache.
    from pocket_tts.modules.stateful_module import init_states, increment_steps  # noqa: F401

    # Build state manually matching the reference schema.
    ref_state = {
        "root": ref_mha.init_state(batch_size=1, sequence_length=S_cap)
    }
    # Prepopulate cache positions [0..5) with random K/V, set offset=5.
    torch.manual_seed(11)
    K_prev = torch.randn(1, 5, num_heads, head_dim)
    V_prev = torch.randn(1, 5, num_heads, head_dim)
    ref_state["root"]["cache"][0, :, :5] = K_prev
    ref_state["root"]["cache"][1, :, :5] = V_prev
    ref_state["root"]["offset"][:] = 5

    # Mirror the state for the patched path: build explicit kv_cache of
    # shape [2, B, S_cap, H, D] populated identically.
    patched_cache = torch.zeros((2, 1, S_cap, num_heads, head_dim))
    patched_cache[0, :, :5] = K_prev
    patched_cache[1, :, :5] = V_prev

    # Single AR step at offset=5.
    x = torch.randn(1, 1, embed_dim)

    # Reference forward.
    y_ref = ref_mha(x, ref_state)

    # Patched forward — build inputs.
    cos_table, sin_table = build_rope_tables(max_context=256, head_dim=head_dim)
    cos, sin = slice_rope_tables(cos_table, sin_table, offset=5, length=1)
    offset_mask = build_one_hot_offset_mask(offset=5, s_capacity=S_cap, batch_size=1)
    attn_mask = build_additive_attention_mask_step(
        offset=5, s_capacity=S_cap, context=None
    )

    y_pt, cache_out = patched(x, patched_cache, offset_mask, attn_mask, cos, sin)

    _tight_allclose(y_pt, y_ref, atol=1e-5, rtol=1e-5, label="patched_mha/ar_step/y")

    # The patched cache at positions [0..5) must be unchanged; position 5
    # must contain the freshly-written k/v; positions [6..] must stay 0.
    assert torch.allclose(cache_out[0, :, :5], K_prev, atol=1e-6)
    assert torch.allclose(cache_out[1, :, :5], V_prev, atol=1e-6)
    assert torch.all(cache_out[0, :, 6:] == 0.0)
    assert torch.all(cache_out[1, :, 6:] == 0.0)
    # The reference cache row at offset=5 should match the patched one.
    _tight_allclose(
        cache_out[0, :, 5], ref_state["root"]["cache"][0, :, 5],
        atol=1e-6, rtol=1e-6, label="patched_mha/ar_step/k_written"
    )
    _tight_allclose(
        cache_out[1, :, 5], ref_state["root"]["cache"][1, :, 5],
        atol=1e-6, rtol=1e-6, label="patched_mha/ar_step/v_written"
    )


def test_patched_mha_parity_prefill() -> None:
    """T_q>1 prefill into an empty cache slot range must match reference."""
    _seed_all(13)
    embed_dim, num_heads = 1024, 16
    head_dim = embed_dim // num_heads
    S_cap = 64
    prefill_len = 7

    ref_mha = _build_ref_mha(embed_dim, num_heads)

    from pockettts_coreml.patches.rope_patched import (
        build_rope_tables,
        slice_rope_tables,
    )
    from pockettts_coreml.patches.transformer_patched import (
        PatchedStreamingMultiheadAttention,
        build_additive_attention_mask_prefill,
        build_scatter_prefill_mask,
    )

    patched = PatchedStreamingMultiheadAttention(embed_dim, num_heads)
    patched.eval()
    patched.load_reference_weights(ref_mha)

    ref_state = {"root": ref_mha.init_state(batch_size=1, sequence_length=S_cap)}
    # Empty state, offset=0.
    x = torch.randn(1, prefill_len, embed_dim)

    y_ref = ref_mha(x, ref_state)

    # Patched prefill starting at offset=0 for T_q=prefill_len.
    patched_cache = torch.zeros((2, 1, S_cap, num_heads, head_dim))
    cos_table, sin_table = build_rope_tables(max_context=256, head_dim=head_dim)
    cos, sin = slice_rope_tables(cos_table, sin_table, offset=0, length=prefill_len)
    scatter_mask = build_scatter_prefill_mask(
        start_offset=0, prefill_len=prefill_len, s_capacity=S_cap, batch_size=1
    )
    attn_mask = build_additive_attention_mask_prefill(
        start_offset=0, prefill_len=prefill_len, s_capacity=S_cap, context=None
    )

    y_pt, cache_out = patched.forward_prefill(
        x, patched_cache, scatter_mask, attn_mask, cos, sin
    )

    _tight_allclose(y_pt, y_ref, atol=1e-5, rtol=1e-5, label="patched_mha/prefill/y")

    # Check cache entries at positions [0..prefill_len) match reference.
    _tight_allclose(
        cache_out[0, :, :prefill_len], ref_state["root"]["cache"][0, :, :prefill_len],
        atol=1e-6, rtol=1e-6, label="patched_mha/prefill/k_cache"
    )
    _tight_allclose(
        cache_out[1, :, :prefill_len], ref_state["root"]["cache"][1, :, :prefill_len],
        atol=1e-6, rtol=1e-6, label="patched_mha/prefill/v_cache"
    )


def test_patched_mha_prefill_with_context_window() -> None:
    """Context-window masking (used by the Mimi transformer, context=250)
    must be respected by the additive mask builder.
    """
    _seed_all(17)
    # Small case so context=3 actually clips something.
    embed_dim, num_heads, context = 64, 4, 3
    head_dim = embed_dim // num_heads
    S_cap = 16
    prefill_len = 6

    from pocket_tts.modules.rope import RotaryEmbedding
    from pocket_tts.modules.transformer import StreamingMultiheadAttention

    torch.manual_seed(0)
    rope = RotaryEmbedding(max_period=10_000.0)
    ref_mha = StreamingMultiheadAttention(
        embed_dim=embed_dim, num_heads=num_heads, rope=rope, context=context
    )
    ref_mha.eval()
    ref_mha._module_absolute_name = "root"

    from pockettts_coreml.patches.rope_patched import (
        build_rope_tables,
        slice_rope_tables,
    )
    from pockettts_coreml.patches.transformer_patched import (
        PatchedStreamingMultiheadAttention,
        build_additive_attention_mask_prefill,
        build_scatter_prefill_mask,
    )

    patched = PatchedStreamingMultiheadAttention(embed_dim, num_heads)
    patched.eval()
    patched.load_reference_weights(ref_mha)

    ref_state = {"root": ref_mha.init_state(batch_size=1, sequence_length=S_cap)}
    x = torch.randn(1, prefill_len, embed_dim)

    y_ref = ref_mha(x, ref_state)

    patched_cache = torch.zeros((2, 1, S_cap, num_heads, head_dim))
    cos_table, sin_table = build_rope_tables(max_context=S_cap, head_dim=head_dim)
    cos, sin = slice_rope_tables(cos_table, sin_table, offset=0, length=prefill_len)
    scatter_mask = build_scatter_prefill_mask(
        start_offset=0, prefill_len=prefill_len, s_capacity=S_cap
    )
    attn_mask = build_additive_attention_mask_prefill(
        start_offset=0, prefill_len=prefill_len, s_capacity=S_cap, context=context
    )

    y_pt, _ = patched.forward_prefill(x, patched_cache, scatter_mask, attn_mask, cos, sin)

    _tight_allclose(y_pt, y_ref, atol=1e-5, rtol=1e-5, label="patched_mha/prefill+ctx")


# ------------------------------------------------------------------
# Patch 3: SimpleMLPAdaLN (flow head) parity
# ------------------------------------------------------------------


def _build_ref_flow_mlp(seed: int = 0):
    from pocket_tts.modules.mlp import SimpleMLPAdaLN

    torch.manual_seed(seed)
    mlp = SimpleMLPAdaLN(
        in_channels=32,
        model_channels=512,
        out_channels=32,
        cond_channels=1024,
        num_res_blocks=6,
        num_time_conds=2,
    )
    mlp.train(False)
    return mlp


def test_patched_flow_mlp_parity_random() -> None:
    """Patched flow MLP must match reference at atol=1e-5."""
    _seed_all(101)
    from pockettts_coreml.patches.mlp_patched import PatchedSimpleMLPAdaLN

    ref = _build_ref_flow_mlp(seed=101)
    patched = PatchedSimpleMLPAdaLN(
        in_channels=32,
        model_channels=512,
        out_channels=32,
        cond_channels=1024,
        num_res_blocks=6,
        num_time_conds=2,
    )
    patched.train(False)
    patched.load_reference_weights(ref)

    for i in range(10):
        torch.manual_seed(200 + i)
        c = torch.randn(1, 1024)
        s = torch.randn(1, 1)
        t = torch.randn(1, 1)
        x = torch.randn(1, 32)
        with torch.no_grad():
            y_ref = ref(c, s, t, x)
            y_pt = patched(c, s, t, x)
        _tight_allclose(y_pt, y_ref, atol=1e-5, rtol=1e-5, label=f"flow_mlp/rand_{i}")


def test_patched_flow_mlp_parity_lsd_step1() -> None:
    """At lsd_decode_steps=1 (production default), s=0.0 and t=1.0 are
    the constants actually used by the reference's `lsd_decode` loop.
    """
    _seed_all(103)
    from pockettts_coreml.patches.mlp_patched import PatchedSimpleMLPAdaLN

    ref = _build_ref_flow_mlp(seed=103)
    patched = PatchedSimpleMLPAdaLN(
        in_channels=32, model_channels=512, out_channels=32,
        cond_channels=1024, num_res_blocks=6, num_time_conds=2,
    )
    patched.train(False)
    patched.load_reference_weights(ref)

    torch.manual_seed(55)
    c = torch.randn(1, 1024)
    x = torch.randn(1, 32)
    s = torch.zeros(1, 1)
    t = torch.ones(1, 1)
    with torch.no_grad():
        y_ref = ref(c, s, t, x)
        y_pt = patched(c, s, t, x)
    _tight_allclose(y_pt, y_ref, atol=1e-5, rtol=1e-5, label="flow_mlp/lsd_step1")


# ------------------------------------------------------------------
# Patch 4: FlowLM main + flow split parity
# ------------------------------------------------------------------


class _RefFlowLMShim(nn.Module):
    """Test-only shim: a plain nn.Module that holds the pieces of the
    reference FlowLMModel needed for parity testing, without going
    through FlowLMModel's constructor (which requires a real
    SentencePiece-backed LUTConditioner, needing gated HF weights).

    Exposes the same attribute names (`transformer`, `flow_net`,
    `bos_emb`, `input_linear`, `out_norm`, `out_eos`, `emb_std`,
    `emb_mean`) so `PatchedFlowLMMain.load_reference_weights` works
    unchanged.
    """

    def __init__(self, transformer, flow_net, ldim: int, d_model: int, dtype: torch.dtype):
        super().__init__()
        self.transformer = transformer
        self.flow_net = flow_net
        self.bos_emb = nn.Parameter(torch.randn(ldim, dtype=dtype))
        self.input_linear = nn.Linear(ldim, d_model, bias=False, dtype=dtype)
        self.out_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.out_eos = nn.Linear(d_model, 1, dtype=dtype)
        self.register_buffer("emb_std", torch.rand(ldim, dtype=dtype) + 0.5)
        self.register_buffer("emb_mean", torch.randn(ldim, dtype=dtype) * 0.1)


def _build_ref_flow_lm(seed: int = 0):
    """Build a test shim matching the English FlowLM shape."""
    from pocket_tts.modules.mimi_transformer import StreamingTransformer
    from pocket_tts.modules.mlp import SimpleMLPAdaLN

    torch.manual_seed(seed)

    transformer = StreamingTransformer(
        d_model=1024, num_heads=16, num_layers=6,
        dim_feedforward=4096, layer_scale=None, context=None, max_period=10_000.0,
    )
    flow_net = SimpleMLPAdaLN(
        in_channels=32, model_channels=512, out_channels=32,
        cond_channels=1024, num_res_blocks=6, num_time_conds=2,
    )
    shim = _RefFlowLMShim(
        transformer=transformer, flow_net=flow_net,
        ldim=32, d_model=1024, dtype=torch.float32,
    )
    shim.train(False)
    # Required for StatefulModule.get_state lookups — every StatefulModule
    # inside transformer needs a qualified name.
    for module_name, module in shim.named_modules():
        if hasattr(module, "_module_absolute_name"):
            module._module_absolute_name = module_name
    return shim


def _ref_main_path(ref_shim, sequence, text_embeddings, ref_state):
    """Run just the main path of reference FlowLMModel manually.

    Mirrors `FlowLMModel.forward` up to `ctx` + `eos_logit`, WITHOUT
    noise sampling or flow_net. Matches the real reference's logic for
    the main-path slice (ref `models/flow_lm.py:120-130`).
    """
    sequence_filled = torch.where(torch.isnan(sequence), ref_shim.bos_emb, sequence)
    input_ = ref_shim.input_linear(sequence_filled)
    backbone_in = torch.cat([text_embeddings, input_], dim=1)
    trans_out = ref_shim.transformer(backbone_in, ref_state)
    trans_out = ref_shim.out_norm(trans_out)
    ctx = trans_out[:, -1, :]
    eos_logit = ref_shim.out_eos(ctx)
    return ctx, eos_logit


def test_patched_flow_lm_main_parity_ar_step() -> None:
    """Patched main-path AR step must match reference's transformer+norm+eos
    at atol=1e-5.
    """
    _seed_all(301)
    from pockettts_coreml.patches.flow_lm_patched import PatchedFlowLMMain
    from pockettts_coreml.patches.transformer_patched import (
        build_additive_attention_mask_step,
        build_one_hot_offset_mask,
    )
    from pocket_tts.modules.stateful_module import init_states, increment_steps

    ref = _build_ref_flow_lm(seed=301)
    patched = PatchedFlowLMMain(
        ldim=32, d_model=1024, num_heads=16, num_layers=6, dim_feedforward=4096,
    )
    patched.train(False)
    patched.load_reference_weights(ref)

    S_cap = 24
    # Step 1: drive the reference through a 5-token prefill so offsets
    # in both paths start from the same "mid-cache" configuration.
    ref_state = init_states(ref, batch_size=1, sequence_length=S_cap)
    torch.manual_seed(400)
    text_prefix = torch.randn(1, 4, 1024)
    seq_step0 = torch.full((1, 1, 32), float("nan"))
    with torch.no_grad():
        _ = _ref_main_path(ref, seq_step0, text_prefix, ref_state)
    increment_steps(ref, ref_state, increment=4 + 1)

    # Snapshot caches BEFORE the AR step under test. Copy ONLY the
    # valid-written region [0..5); rest stays zero. The reference's
    # cache is NaN-initialized (`transformer.py:51-57`), and NaN propagates
    # through matmul regardless of post-matmul additive masking — so the
    # patched path (which presents the full cache to SDPA) needs
    # well-defined (zero) values at unwritten positions.
    kv_caches_in = torch.zeros((6, 2, 1, S_cap, 16, 64))
    prior_len = 4 + 1  # 4 text + 1 seq tokens written during prefill
    for li in range(6):
        ref_cache = ref_state[f"transformer.layers.{li}.self_attn"]["cache"]
        kv_caches_in[li, :, :, :prior_len] = ref_cache[:, :, :prior_len].clone()

    # AR step at offset=5.
    torch.manual_seed(500)
    seq = torch.randn(1, 1, 32)
    empty_text = torch.zeros(1, 0, 1024)

    with torch.no_grad():
        ctx_ref, eos_logit_ref = _ref_main_path(ref, seq, empty_text, ref_state)

    # Patched path.
    from pockettts_coreml.patches.rope_patched import (
        build_rope_tables, slice_rope_tables,
    )
    cos_table, sin_table = build_rope_tables(max_context=256, head_dim=64)
    cos, sin = slice_rope_tables(cos_table, sin_table, offset=5, length=1)
    offset_mask = build_one_hot_offset_mask(offset=5, s_capacity=S_cap, batch_size=1)
    attn_mask = build_additive_attention_mask_step(
        offset=5, s_capacity=S_cap, context=None
    )
    with torch.no_grad():
        ctx_pt, eos_logit_pt, kv_caches_out_pt = patched(
            sequence=seq, text_embeddings=empty_text,
            kv_caches_in=kv_caches_in,
            offset_mask=offset_mask, attn_mask=attn_mask,
            rope_cos=cos, rope_sin=sin,
        )

    _tight_allclose(ctx_pt, ctx_ref, atol=1e-5, rtol=1e-5, label="flow_lm_main/ctx")
    _tight_allclose(eos_logit_pt, eos_logit_ref, atol=1e-5, rtol=1e-5, label="flow_lm_main/eos")


def test_patched_flow_lm_main_parity_prefill() -> None:
    """Prefill path: start from empty cache, push text_prefix (S=4) plus
    one BOS-sequence latent. Compare the last-token ctx + eos logit.
    """
    _seed_all(302)
    from pockettts_coreml.patches.flow_lm_patched import PatchedFlowLMMain
    from pockettts_coreml.patches.transformer_patched import (
        build_additive_attention_mask_prefill,
        build_scatter_prefill_mask,
    )
    from pocket_tts.modules.stateful_module import init_states

    ref = _build_ref_flow_lm(seed=302)
    patched = PatchedFlowLMMain(
        ldim=32, d_model=1024, num_heads=16, num_layers=6, dim_feedforward=4096,
    )
    patched.train(False)
    patched.load_reference_weights(ref)

    S_cap = 24
    S_text = 4
    T_total = S_text + 1

    ref_state = init_states(ref, batch_size=1, sequence_length=S_cap)

    torch.manual_seed(770)
    text_prefix = torch.randn(1, S_text, 1024)
    seq_step0 = torch.full((1, 1, 32), float("nan"))

    with torch.no_grad():
        ctx_ref, eos_logit_ref = _ref_main_path(ref, seq_step0, text_prefix, ref_state)

    # Patched prefill: sequence at step 0 = bos_emb (no NaN).
    seq_step0_bos = patched.bos_emb.view(1, 1, -1).contiguous()

    from pockettts_coreml.patches.rope_patched import (
        build_rope_tables, slice_rope_tables,
    )
    cos_table, sin_table = build_rope_tables(max_context=256, head_dim=64)
    cos, sin = slice_rope_tables(cos_table, sin_table, offset=0, length=T_total)
    scatter_mask = build_scatter_prefill_mask(
        start_offset=0, prefill_len=T_total, s_capacity=S_cap, batch_size=1
    )
    attn_mask = build_additive_attention_mask_prefill(
        start_offset=0, prefill_len=T_total, s_capacity=S_cap, context=None
    )

    kv_caches_in = torch.zeros((6, 2, 1, S_cap, 16, 64))
    with torch.no_grad():
        ctx_pt, eos_logit_pt, _ = patched.forward_prefill(
            sequence=seq_step0_bos, text_embeddings=text_prefix,
            kv_caches_in=kv_caches_in,
            scatter_mask=scatter_mask, attn_mask=attn_mask,
            rope_cos=cos, rope_sin=sin,
        )

    _tight_allclose(ctx_pt, ctx_ref, atol=1e-5, rtol=1e-5, label="flow_lm_main_prefill/ctx")
    _tight_allclose(eos_logit_pt, eos_logit_ref, atol=1e-5, rtol=1e-5, label="flow_lm_main_prefill/eos")


def test_patched_flow_lm_flow_parity() -> None:
    """Patched flow head (single-step lsd_decode at N=1) must match
    reference `lsd_decode` at num_steps=1.
    """
    _seed_all(601)
    from pockettts_coreml.patches.flow_lm_patched import PatchedFlowLMFlow
    from pocket_tts.models.flow_lm import lsd_decode
    from pocket_tts.modules.mlp import SimpleMLPAdaLN
    from functools import partial

    torch.manual_seed(601)
    ref_flow_net = SimpleMLPAdaLN(
        in_channels=32, model_channels=512, out_channels=32,
        cond_channels=1024, num_res_blocks=6, num_time_conds=2,
    )
    ref_flow_net.train(False)

    # Wrap into a tiny shim matching PatchedFlowLMFlow's `load_reference_weights` contract.
    class _ShimFlow:
        pass
    shim = _ShimFlow()
    shim.flow_net = ref_flow_net

    patched = PatchedFlowLMFlow(ldim=32, d_model=1024, flow_dim=512, flow_depth=6)
    patched.train(False)
    patched.load_reference_weights(shim)

    torch.manual_seed(602)
    c = torch.randn(1, 1024)
    x_noise = torch.randn(1, 32)

    # Reference: lsd_decode with num_steps=1.
    with torch.no_grad():
        v_t = partial(ref_flow_net, c)
        ref_out = lsd_decode(v_t, x_noise.clone(), num_steps=1)

        s = torch.zeros(1, 1)
        t = torch.ones(1, 1)
        pt_out = patched(c, s, t, x_noise, num_steps_inv=1.0)

    _tight_allclose(pt_out, ref_out, atol=1e-5, rtol=1e-5, label="flow_lm_flow/lsd_N1")


# ------------------------------------------------------------------
# Patch 5: Mimi streaming conv pure-functional wrappers
# ------------------------------------------------------------------


def test_patched_streaming_conv1d_parity() -> None:
    """StreamingConv1d (pad_mode=constant) must match reference over a
    streaming sequence of calls with identical inputs + propagated state.
    """
    _seed_all(801)
    from pocket_tts.modules.conv import StreamingConv1d as RefConv1d
    from pocket_tts.modules.stateful_module import StatefulModule
    from pockettts_coreml.patches.mimi_patched import PatchedStreamingConv1d

    # Pick a non-trivial config: kernel=7, stride=1 -> state_length=6 (the
    # SEANet input conv at decoder entry, ref `modules/seanet.py`).
    torch.manual_seed(801)
    ref = RefConv1d(
        in_channels=32, out_channels=64, kernel_size=7, stride=1,
        dilation=1, groups=1, bias=True, pad_mode="constant",
    )
    ref.train(False)
    ref._module_absolute_name = "root"
    patched = PatchedStreamingConv1d(
        in_channels=32, out_channels=64, kernel_size=7, stride=1,
        dilation=1, groups=1, bias=True,
    )
    patched.train(False)
    patched.load_reference_weights(ref)

    B = 1
    # Stream 5 calls with T=2 each (stride=1 so T%stride=0 trivially).
    # Reference state is a dict; patched state is a tensor.
    ref_state = {"root": ref.init_state(batch_size=B, sequence_length=0)}
    pt_state = patched.init_state(batch_size=B)

    for call_i in range(5):
        torch.manual_seed(900 + call_i)
        x = torch.randn(B, 32, 2)

        with torch.no_grad():
            y_ref = ref(x.clone(), ref_state)
            y_pt, pt_state = patched.pure_forward(x.clone(), pt_state)

        _tight_allclose(y_pt, y_ref, atol=1e-5, rtol=1e-5, label=f"sconv1d/call{call_i}")

    # Also verify post-streaming state matches the reference's rolling buffer.
    _tight_allclose(
        pt_state, ref_state["root"]["previous"],
        atol=1e-5, rtol=1e-5, label="sconv1d/state",
    )


def test_patched_streaming_convtranspose1d_parity() -> None:
    """StreamingConvTranspose1d with stride=6 (matches SEANet decoder's
    first upsample stage, ref `modules/seanet.py`).
    """
    _seed_all(821)
    from pocket_tts.modules.conv import StreamingConvTranspose1d as RefConvTr
    from pockettts_coreml.patches.mimi_patched import (
        PatchedStreamingConvTranspose1d,
    )

    torch.manual_seed(821)
    ref = RefConvTr(
        in_channels=64, out_channels=32, kernel_size=12, stride=6,
        groups=1, bias=True,
    )
    ref.train(False)
    ref._module_absolute_name = "root"
    patched = PatchedStreamingConvTranspose1d(
        in_channels=64, out_channels=32, kernel_size=12, stride=6,
        groups=1, bias=True,
    )
    patched.train(False)
    patched.load_reference_weights(ref)

    B = 1
    ref_state = {"root": ref.init_state(batch_size=B, sequence_length=0)}
    pt_state = patched.init_state(batch_size=B)

    for call_i in range(4):
        torch.manual_seed(820 + call_i)
        x = torch.randn(B, 64, 3)  # 3 input frames per call.
        with torch.no_grad():
            y_ref = ref(x.clone(), ref_state)
            y_pt, pt_state = patched.pure_forward(x.clone(), pt_state)

        _tight_allclose(y_pt, y_ref, atol=1e-5, rtol=1e-5, label=f"sconvtr/call{call_i}")

    _tight_allclose(
        pt_state, ref_state["root"]["partial"],
        atol=1e-5, rtol=1e-5, label="sconvtr/state",
    )


def test_patched_streaming_convtranspose1d_depthwise_parity() -> None:
    """ConvTrUpsample1d (stride=16, depthwise/groups=dimension) — the
    12.5->200Hz bridge. Plan accepts CPU fallback; we still need parity.
    """
    _seed_all(831)
    from pocket_tts.modules.conv import StreamingConvTranspose1d as RefConvTr
    from pockettts_coreml.patches.mimi_patched import (
        PatchedStreamingConvTranspose1d,
    )

    dim = 64  # smaller than reference (512) for test speed.
    torch.manual_seed(831)
    ref = RefConvTr(
        in_channels=dim, out_channels=dim, kernel_size=32, stride=16,
        groups=dim, bias=False,
    )
    ref.train(False)
    ref._module_absolute_name = "root"
    patched = PatchedStreamingConvTranspose1d(
        in_channels=dim, out_channels=dim, kernel_size=32, stride=16,
        groups=dim, bias=False,
    )
    patched.train(False)
    patched.load_reference_weights(ref)

    B = 1
    ref_state = {"root": ref.init_state(batch_size=B, sequence_length=0)}
    pt_state = patched.init_state(batch_size=B)

    for call_i in range(3):
        torch.manual_seed(830 + call_i)
        x = torch.randn(B, dim, 1)
        with torch.no_grad():
            y_ref = ref(x.clone(), ref_state)
            y_pt, pt_state = patched.pure_forward(x.clone(), pt_state)

        _tight_allclose(y_pt, y_ref, atol=1e-5, rtol=1e-5, label=f"sconvtr_dw/call{call_i}")

    _tight_allclose(
        pt_state, ref_state["root"]["partial"],
        atol=1e-5, rtol=1e-5, label="sconvtr_dw/state",
    )


def test_patched_streaming_conv1d_stride_gt_1() -> None:
    """StreamingConv1d with stride=1 has kernel_eff-stride=6 state; at
    stride=k (where k == kernel_size) state_length=0 (no streaming carry).
    Verify the state_length=0 code path by constructing such a conv.
    """
    _seed_all(841)
    from pocket_tts.modules.conv import StreamingConv1d as RefConv1d
    from pockettts_coreml.patches.mimi_patched import PatchedStreamingConv1d

    torch.manual_seed(841)
    # kernel=4, stride=4 -> kernel_eff - stride = 0. No rolling buffer.
    ref = RefConv1d(
        in_channels=8, out_channels=16, kernel_size=4, stride=4,
        dilation=1, groups=1, bias=True, pad_mode="constant",
    )
    ref.train(False)
    ref._module_absolute_name = "root"
    patched = PatchedStreamingConv1d(
        in_channels=8, out_channels=16, kernel_size=4, stride=4,
        dilation=1, groups=1, bias=True,
    )
    patched.train(False)
    patched.load_reference_weights(ref)

    assert patched.state_length == 0

    B = 1
    ref_state = {"root": ref.init_state(batch_size=B, sequence_length=0)}
    pt_state = patched.init_state(batch_size=B)

    for call_i in range(3):
        torch.manual_seed(840 + call_i)
        x = torch.randn(B, 8, 8)  # T=8, multiple of stride=4
        with torch.no_grad():
            y_ref = ref(x.clone(), ref_state)
            y_pt, pt_state = patched.pure_forward(x.clone(), pt_state)
        _tight_allclose(y_pt, y_ref, atol=1e-5, rtol=1e-5, label=f"sconv1d_sgt1/call{call_i}")
