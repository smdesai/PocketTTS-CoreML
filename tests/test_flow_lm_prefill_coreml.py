"""Parity test: `flow_lm_prefill.mlpackage` vs the reference prefill path.

Validates that running the CoreML prefill bundle on the canonical
(voice, prompt) fixture produces a KV cache that matches the one
obtained by running `TTSModel._run_flow_lm_and_increment_step` directly.

Gate: post-prefill KV fp16 `atol=5e-3` averaged over written slots.

Skips cleanly if the .mlpackage isn't built yet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "Artifacts" / "en_alba_fp16"
VOICE = (
    REPO_ROOT / "pockettts_coreml" / "oracle" / "fixtures"
    / "english_alba_seed42" / "alba.safetensors"
)
PROMPT = "Pocket TTS is a lightweight text-to-speech model."

_REF = REPO_ROOT / "pockettts_coreml" / "reference"
if str(_REF) not in sys.path:
    sys.path.insert(0, str(_REF))


def _require(name: str) -> Path:
    p = ARTIFACTS_DIR / f"{name}.mlpackage"
    if not p.exists():
        pytest.skip(f"{p} missing; run `python -m pockettts_coreml.convert --only {name}`.")
    return p


def test_prefill_kv_matches_reference():
    """CoreML prefill KV output parity with reference.

    Reference path: `_run_flow_lm_and_increment_step(text_tokens=...)`
    over a voice-conditioned state. We compare the rank-5 KV cache
    after prefill, restricting to the `[0, voice_len + S_text)` slots.
    Slots beyond that are untouched (zero) in both paths.
    """
    import coremltools as ct
    from coremltools.models import MLModel

    if not VOICE.exists():
        pytest.skip(f"{VOICE} missing")
    _require("flow_lm_prefill")

    from pockettts_coreml.patches import (
        build_additive_attention_mask_prefill,
        build_patched_submodules,
        build_rope_tables,
        build_scatter_prefill_mask,
    )

    S_CAP = 256
    S_TEXT_PAD = 128
    L, H, D = 6, 16, 64

    ps = build_patched_submodules()
    tts = ps.tts_model
    tts.eval()

    prepared = tts.flow_lm.conditioner.prepare(PROMPT)
    text_tokens = prepared.tokens  # [1, S_text]
    S_text = int(text_tokens.shape[-1])
    assert 0 < S_text <= S_TEXT_PAD

    # Reference prefill on voice-conditioned state.
    torch.manual_seed(0)
    voice_state = tts.get_state_for_audio_prompt(str(VOICE))
    tts._expand_kv_cache(voice_state, sequence_length=S_CAP)
    voice_len = int(voice_state[f"transformer.layers.0.self_attn"]["offset"].item())
    assert voice_len > 0

    # Build rank-5 from voice state (voice slots only).
    def _kv_from_state(state):
        kv = torch.zeros((2 * L, 1, S_CAP, H, D), dtype=torch.float32)
        for layer in range(L):
            name = f"transformer.layers.{layer}.self_attn"
            cache = state[name]["cache"]
            off = int(state[name]["offset"].item())
            k = cache[0].clone()
            v = cache[1].clone()
            k[torch.isnan(k)] = 0.0
            v[torch.isnan(v)] = 0.0
            if off > 0:
                kv[2 * layer, :, :off] = k[:, :off]
                kv[2 * layer + 1, :, :off] = v[:, :off]
        return kv

    kv_pre = _kv_from_state(voice_state)

    # Run reference prefill (mutates voice_state in-place).
    with torch.no_grad():
        tts._run_flow_lm_and_increment_step(
            model_state=voice_state, text_tokens=text_tokens,
        )
    kv_ref = _kv_from_state(voice_state)
    ref_off = int(voice_state[f"transformer.layers.0.self_attn"]["offset"].item())
    assert ref_off == voice_len + S_text

    # Build CoreML inputs.
    with torch.no_grad():
        text_emb_real = tts.flow_lm.conditioner(prepared).to(torch.float32).detach()
    assert text_emb_real.shape == (1, S_text, ps.d_model)
    text_embeddings = torch.zeros((1, S_TEXT_PAD, ps.d_model), dtype=torch.float32)
    text_embeddings[:, :S_text] = text_emb_real

    full_scatter = build_scatter_prefill_mask(
        start_offset=voice_len, prefill_len=S_text, s_capacity=S_CAP,
    )
    scatter_mask = torch.zeros((1, S_CAP, S_TEXT_PAD), dtype=torch.float32)
    scatter_mask[:, :, :S_text] = full_scatter

    attn_real = build_additive_attention_mask_prefill(
        start_offset=voice_len, prefill_len=S_text, s_capacity=S_CAP,
    )
    attn_mask = torch.full((1, 1, S_TEXT_PAD, S_CAP), -6.5504e4, dtype=torch.float32)
    attn_mask[:, :, :S_text, :] = attn_real
    attn_mask[:, :, S_text:, :] = attn_real[:, :, -1:, :]

    cos_t, sin_t = build_rope_tables(max_context=S_CAP, head_dim=D)
    rope_cos = cos_t[voice_len : voice_len + S_TEXT_PAD].unsqueeze(0).unsqueeze(2)
    rope_sin = sin_t[voice_len : voice_len + S_TEXT_PAD].unsqueeze(0).unsqueeze(2)

    mlmodel = MLModel(str(_require("flow_lm_prefill")), compute_units=ct.ComputeUnit.CPU_ONLY)
    pred = mlmodel.predict({
        "text_embeddings": text_embeddings.numpy().astype(np.float32),
        "kv_cache_in": kv_pre.numpy().astype(np.float32),
        "scatter_mask": scatter_mask.numpy().astype(np.float32),
        "attn_mask": attn_mask.numpy().astype(np.float32),
        "rope_cos": rope_cos.numpy().astype(np.float32),
        "rope_sin": rope_sin.numpy().astype(np.float32),
    })
    kv_ml = torch.as_tensor(pred["kv_cache_out"]).float()
    assert kv_ml.shape == kv_ref.shape

    # Compare over the written region [0, voice_len + S_text).
    end = voice_len + S_text
    diff_written = (kv_ml[:, :, :end] - kv_ref[:, :, :end]).abs()
    diff_untouched = (kv_ml[:, :, end:] - kv_ref[:, :, end:]).abs()

    max_w = float(diff_written.max().item())
    mean_w = float(diff_written.mean().item())
    max_u = float(diff_untouched.max().item())
    print(
        f"[prefill parity] written: max={max_w:.3e} mean={mean_w:.3e} | "
        f"untouched slots: max={max_u:.3e}"
    )
    # Untouched slots must be essentially zero (both sides).
    assert max_u < 1e-4, f"prefill bled past voice_len+S_text: max_u={max_u}"
    # Written slots: fp16 drift on a 6-layer transformer over 128 tokens
    # accumulates. Mean drift should still be small — it's the signal
    # that matters for the downstream softmax, not the per-element max.
    # The audio-PSNR gate at the end-to-end Swift level is the real
    # quality signal (see the AR flow_lm_main converter's own note).
    assert mean_w < 2e-3, (
        f"flow_lm_prefill KV mean drift too large: mean_w={mean_w:.3e} (max=2e-3)"
    )
    # Sanity: max drift should be bounded (5e-2 observed on canonical
    # fixture; allow 1e-1 for CI noise headroom).
    assert max_w < 1e-1, (
        f"flow_lm_prefill KV max drift unreasonable: max_w={max_w:.3e}"
    )
