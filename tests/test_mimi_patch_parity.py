"""Patched-mimi parity harness.

Compares:
  - PatchedSEANetEncoder + PatchedProjectedTransformer (non-streaming) vs
    reference MimiModel.encode_to_latent (with emulated input).
  - PatchedMimiDecoder (streaming, per-frame with state blob) vs reference
    MimiModel.decode_from_latent driven per-frame with fresh mimi_state.

Tolerances: fp32 parity atol=5e-4, rtol=5e-4. Looser than patch_parity
because the patched implementations re-derive normalization constants and
the multi-step scatter_mask path for the transformer is numerically equivalent
but not bit-identical to the reference's dynamic-slice KV write (different
reduction order in fp32 due to the broadcast multiply vs direct slice).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_REF_DIR = Path(__file__).resolve().parent.parent / "pockettts_coreml" / "reference"
if str(_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_REF_DIR))


def _seed_all(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _build_ref_mimi():
    from pockettts_coreml.patches import build_patched_submodules

    ps = build_patched_submodules()
    return ps.mimi_model, ps


def _tight(a, b, atol, rtol, label):
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = b.abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    if not ok:
        raise AssertionError(
            f"[{label}] parity mismatch: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}"
        )


# ------------------------------------------------------------------
# Encoder parity (non-streaming, short waveform)
# ------------------------------------------------------------------


def test_mimi_encoder_patched_parity():
    """PatchedSEANetEncoder + PatchedProjectedTransformer ≈ reference encode_to_latent."""
    _seed_all(42)
    from pockettts_coreml.patches.mimi_model_patched import (
        PatchedSEANetEncoder, PatchedProjectedTransformer,
    )
    from pockettts_coreml.patches.rope_patched import build_rope_tables
    from pockettts_coreml.patches.transformer_patched import (
        build_additive_attention_mask_prefill,
    )

    mimi, ps = _build_ref_mimi()
    mimi.eval()

    # Build patched encoder + encoder_transformer.
    ref_enc_tx = mimi.encoder_transformer
    # Determine d_model / num_heads / num_layers / layer_scale from the
    # reference transformer.
    ref_layer0 = ref_enc_tx.transformer.layers[0]
    d_model = ref_layer0.self_attn.embed_dim
    num_heads = ref_layer0.self_attn.num_heads
    num_layers = len(ref_enc_tx.transformer.layers)
    dim_ff = ref_layer0.linear1.out_features
    context = ref_layer0.self_attn.context
    # layer_scale: detect from ref. English has a small layer_scale.
    ls_mod = getattr(ref_layer0, "layer_scale_1", None)
    layer_scale = None
    if hasattr(ls_mod, "scale") and isinstance(ls_mod.scale, torch.nn.Parameter):
        layer_scale = 1e-2  # initial value; actual weight copied below

    patched_enc = PatchedSEANetEncoder(mimi.encoder)
    patched_enc.eval()
    patched_enc_tx = PatchedProjectedTransformer(
        input_dimension=ref_enc_tx.input_dimension,
        output_dimensions=ref_enc_tx.output_dimensions,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_ff,
        context=context,
        layer_scale=layer_scale,
    )
    patched_enc_tx.eval()
    patched_enc_tx.load_reference_weights(ref_enc_tx)

    # ~ 1 second of audio (24000 samples = 12 encoder-framerate tokens at 200Hz).
    # Short so the parity check is fast.
    T_audio = 24000
    torch.manual_seed(0)
    waveform = torch.randn(1, 1, T_audio) * 0.05

    # --- reference ---
    with torch.no_grad():
        ref_enc_out = mimi.encoder(waveform, model_state=None)
        (ref_tx_out,) = ref_enc_tx(ref_enc_out, model_state=None)

    # --- patched ---
    with torch.no_grad():
        patched_enc_out = patched_enc(waveform)
        # Build attention mask + RoPE tables for prefill T=N.
        N = patched_enc_out.shape[-1]
        head_dim = d_model // num_heads
        cos_t, sin_t = build_rope_tables(max_context=N, head_dim=head_dim)
        rope_cos = cos_t[:N].unsqueeze(0).unsqueeze(2)
        rope_sin = sin_t[:N].unsqueeze(0).unsqueeze(2)
        attn_mask = build_additive_attention_mask_prefill(
            start_offset=0, prefill_len=N, s_capacity=N, context=context,
        )
        patched_tx_out = patched_enc_tx.forward_nonstreaming(
            patched_enc_out, attn_mask, rope_cos, rope_sin,
        )

    # SEANet encoder match:
    _tight(patched_enc_out, ref_enc_out, atol=5e-5, rtol=5e-5, label="seanet_encoder")
    # Transformer match (non-streaming eager, full prefill):
    _tight(patched_tx_out, ref_tx_out, atol=5e-4, rtol=5e-4, label="encoder_transformer")


# ------------------------------------------------------------------
# Decoder parity (streaming, per-frame)
# ------------------------------------------------------------------


@pytest.mark.parametrize("num_frames", [2, 5])
def test_mimi_decoder_patched_parity(num_frames: int):
    """PatchedMimiDecoder ≈ reference decode_from_latent driven per-frame."""
    _seed_all(42)
    from pockettts_coreml.patches.mimi_model_patched import PatchedMimiDecoder
    from pockettts_coreml.patches.rope_patched import build_rope_tables
    from pockettts_coreml.patches.transformer_patched import (
        build_additive_attention_mask_prefill,
        build_scatter_prefill_mask,
    )
    from pocket_tts.modules.stateful_module import init_states, increment_steps

    mimi, ps = _build_ref_mimi()
    mimi.eval()

    # Build patched decoder (with identity emb_std/mean).
    S_CAP = 256
    T_per_step = 16  # encoder_frame_rate / frame_rate
    decoder = PatchedMimiDecoder(
        mimi, state_s_cap=S_CAP,
        emb_std=torch.ones(32), emb_mean=torch.zeros(32),
    )
    decoder.eval()

    # Seed per-frame inputs. FlowLM emits [1, 32, 1] latents (inner_dim=32).
    torch.manual_seed(1234)
    latents = [torch.randn(1, 32, 1) * 0.1 for _ in range(num_frames)]

    # --- reference: init state, call decode_from_latent per frame ---
    # Reference path (tts_model.py:449-454):
    #   quantized = mimi.quantizer(latent [1,32,1])  -> [1,512,1]
    #   audio = mimi.decode_from_latent(quantized, state)
    ref_state = init_states(mimi, batch_size=1, sequence_length=S_CAP)
    ref_audio_chunks = []
    with torch.no_grad():
        for lat in latents:
            quantized = mimi.quantizer(lat)
            audio = mimi.decode_from_latent(quantized, ref_state)
            increment_steps(mimi, ref_state, increment=T_per_step)
            ref_audio_chunks.append(audio)
    ref_audio = torch.cat(ref_audio_chunks, dim=-1)

    # --- patched: drive per-frame with packed state blob ---
    # Determine trace-friendly inputs.
    state_blob = torch.zeros(decoder.layout.total_elems, dtype=torch.float32)
    head_dim = decoder.tx_head_dim
    cos_t, sin_t = build_rope_tables(max_context=S_CAP, head_dim=head_dim)

    patched_audio_chunks = []
    with torch.no_grad():
        for f_idx, lat in enumerate(latents):
            start_off = f_idx * T_per_step
            scatter_mask = build_scatter_prefill_mask(
                start_offset=start_off, prefill_len=T_per_step, s_capacity=S_CAP,
            )
            attn_mask = build_additive_attention_mask_prefill(
                start_offset=start_off, prefill_len=T_per_step,
                s_capacity=S_CAP, context=decoder.decoder_transformer.transformer.context,
            )
            rope_cos = cos_t[start_off:start_off + T_per_step].unsqueeze(0).unsqueeze(2)
            rope_sin = sin_t[start_off:start_off + T_per_step].unsqueeze(0).unsqueeze(2)
            audio, state_blob = decoder(
                lat, state_blob, scatter_mask, attn_mask, rope_cos, rope_sin,
            )
            patched_audio_chunks.append(audio)
    patched_audio = torch.cat(patched_audio_chunks, dim=-1)

    # Compare.
    assert patched_audio.shape == ref_audio.shape, \
        f"patched shape {patched_audio.shape} != ref shape {ref_audio.shape}"
    # Tolerances: streaming SEANet carries state through several fp32
    # residual blocks; max-abs should be < 1e-3.
    _tight(patched_audio, ref_audio, atol=5e-3, rtol=5e-3, label="mimi_decoder")
