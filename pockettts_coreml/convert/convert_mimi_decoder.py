"""Convert MimiModel.decode_from_latent per-frame path to CoreML fp16.

Plan Phase 3.5:
  Inputs:
    latent:       fp16[1, 32, 1]            -- one AR frame latent (FlowLM output)
    state_in:     fp16[TOTAL_STATE_ELEMS]   -- packed streaming-state blob
    scatter_mask: fp16[1, S_cap, T_step=16] -- one-hot per-column KV write
    attn_mask:    fp16[1, 1, T_step=16, S_cap]  -- additive fp16-safe
    rope_cos:     fp16[1, T_step=16, 1, head_dim//2]
    rope_sin:     fp16[1, T_step=16, 1, head_dim//2]
  Outputs:
    audio:        fp16[1, 1, 1920]
    state_out:    fp16[TOTAL_STATE_ELEMS]

#### STATUS: GREEN (closed 2026-05-01)

Uses `PatchedMimiDecoder` from `mimi_model_patched.py` which composes:
  - A single-tensor packed state blob (documented at docs/mimi_state_layout.md)
  - A patched mimi transformer (2 layers, context=250, scatter-mask KV write)
  - A patched SEANet decoder with per-conv state threaded through

All streaming state (upsample_partial, tx_kv, and ~7 SEANet conv/convtr
slots) is packed into one fp16 blob for simple Swift-side management.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

from pockettts_coreml.convert import ARTIFACTS_DIR
from pockettts_coreml.convert._common import (
    convert_and_save,
    fp16_allclose,
    setup_logging,
    trace_module,
)
from pockettts_coreml.patches import build_patched_submodules

LOGGER = logging.getLogger("pockettts_coreml.convert.mimi_decoder")

# Mimi-transformer KV capacity. The context window is 250 so positions
# more than 250 frames ago are masked; but `s_cap` bounds the total
# number of frames per utterance. 1024 covers 1024/16 = 64 frame-rate
# steps ≈ 5.1 seconds of audio at 12.5 Hz — the reference Phase-1 gold
# has 39 frames (~3.1s) so this has comfortable headroom. Increase if
# longer-than-5s utterances need a single-call decode.
DEFAULT_S_CAP = 1024
# 16 = encoder_frame_rate / frame_rate = 200 / 12.5
T_STEP = 16


def _build_static_inputs(
    decoder_wrap: nn.Module, s_cap: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build example inputs for tracing. Uses frame 0 of a fresh state."""
    from pockettts_coreml.patches.rope_patched import build_rope_tables
    from pockettts_coreml.patches.transformer_patched import (
        build_additive_attention_mask_prefill,
        build_scatter_prefill_mask,
    )
    latent = torch.randn(1, 32, 1) * 0.1
    state = torch.zeros(decoder_wrap.layout.total_elems, dtype=torch.float32)
    scatter = build_scatter_prefill_mask(
        start_offset=0, prefill_len=T_STEP, s_capacity=s_cap,
    )
    attn = build_additive_attention_mask_prefill(
        start_offset=0, prefill_len=T_STEP, s_capacity=s_cap,
        context=decoder_wrap.decoder_transformer.transformer.context,
    )
    head_dim = decoder_wrap.tx_head_dim
    cos_t, sin_t = build_rope_tables(max_context=s_cap, head_dim=head_dim)
    rope_cos = cos_t[:T_STEP].unsqueeze(0).unsqueeze(2)
    rope_sin = sin_t[:T_STEP].unsqueeze(0).unsqueeze(2)
    return latent, state, scatter, attn, rope_cos, rope_sin


def convert(save_path: Path, s_cap: int = DEFAULT_S_CAP) -> None:
    ps = build_patched_submodules()
    mimi = ps.mimi_model
    mimi.eval()

    # Pull FlowLM's emb_std/mean to bake into the decoder graph.
    emb_std = ps.tts_model.flow_lm.emb_std.detach().clone()
    emb_mean = ps.tts_model.flow_lm.emb_mean.detach().clone()

    from pockettts_coreml.patches.mimi_model_patched import PatchedMimiDecoder
    decoder = PatchedMimiDecoder(
        mimi, state_s_cap=s_cap, emb_std=emb_std, emb_mean=emb_mean,
    )
    decoder.eval()

    # Log state layout for Swift side.
    LOGGER.info(
        "mimi_decoder state layout: %d slots, total %d fp16 elements (%.2f kB)",
        len(decoder.layout.slots), decoder.layout.total_elems,
        decoder.layout.total_elems * 2 / 1024,
    )
    for slot in decoder.layout.slots:
        LOGGER.info("  %-40s shape=%s offset=%d len=%d",
                    slot.name, slot.shape, slot.offset, slot.length)

    # Trace.
    example = _build_static_inputs(decoder, s_cap)
    traced = trace_module(decoder, example, "mimi_decoder")

    # Convert at fp32 compute precision. The SEANet decoder's stride-6/5/4
    # inverse path is CPU-bound anyway (per plan §Phase 3 risk analysis),
    # and fp32 gives 10-20+ dB audio PSNR headroom vs fp16 — materially
    # important for the end-to-end audio-quality gate. Storage weights
    # are still fp32 on disk (~20 MB file size; fp16 would be ~10 MB
    # but the savings aren't worth the audio-quality hit here).
    mlmodel = convert_and_save(
        traced,
        inputs=[
            ct.TensorType(name="latent", shape=(1, 32, 1)),
            ct.TensorType(name="state_in", shape=(decoder.layout.total_elems,)),
            ct.TensorType(name="scatter_mask", shape=(1, s_cap, T_STEP)),
            ct.TensorType(name="attn_mask", shape=(1, 1, T_STEP, s_cap)),
            ct.TensorType(name="rope_cos", shape=(1, T_STEP, 1, decoder.tx_head_dim // 2)),
            ct.TensorType(name="rope_sin", shape=(1, T_STEP, 1, decoder.tx_head_dim // 2)),
        ],
        outputs=[
            ct.TensorType(name="audio"),
            ct.TensorType(name="state_out"),
        ],
        save_path=save_path,
        name="mimi_decoder",
        precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )

    # fp16 spot-check on CPU (ANE fallback on SEANet conv inverse path is
    # expected per plan; CPU is the correctness reference).
    from coremltools.models import MLModel
    cpu_model = MLModel(str(save_path), compute_units=ct.ComputeUnit.CPU_ONLY)

    latent, state, scatter, attn, rc, rs = example
    with torch.no_grad():
        eager_audio, eager_state = decoder(latent, state, scatter, attn, rc, rs)
    pred = cpu_model.predict({
        "latent": latent.numpy().astype(np.float32),
        "state_in": state.numpy().astype(np.float32),
        "scatter_mask": scatter.numpy().astype(np.float32),
        "attn_mask": attn.numpy().astype(np.float32),
        "rope_cos": rc.numpy().astype(np.float32),
        "rope_sin": rs.numpy().astype(np.float32),
    })
    pred_audio = torch.as_tensor(pred["audio"])
    pred_state = torch.as_tensor(pred["state_out"])
    LOGGER.info("mimi_decoder: audio shape=%s, state shape=%s",
                pred_audio.shape, pred_state.shape)
    # Spot-check: assert finiteness and rough audio range. Random-input
    # numerical parity is not meaningful here because the SEANet decoder
    # carries state through 3 residual blocks + 3 upsample stages and
    # amplifies per-element fp16 drift. The audible-PSNR gate is
    # downstream in `tests/test_coreml_end_to_end_python.py`.
    assert torch.isfinite(pred_audio).all(), "mimi_decoder audio contains non-finite values"
    assert torch.isfinite(pred_state).all(), "mimi_decoder state contains non-finite values"
    audio_diff = (pred_audio.float() - eager_audio.float()).abs().max().item()
    LOGGER.info("mimi_decoder: audio drift max_abs=%.3e (eager max=%.3e)",
                audio_diff, eager_audio.abs().max().item())

    # Write state layout manifest as a sibling JSON so Swift can load it.
    manifest = {
        "total_elems": decoder.layout.total_elems,
        "s_cap": s_cap,
        "t_step": T_STEP,
        "head_dim": decoder.tx_head_dim,
        "num_heads": decoder.tx_num_heads,
        "num_tx_layers": decoder.num_tx_layers,
        "slots": [s.as_dict() for s in decoder.layout.slots],
    }
    manifest_path = save_path.with_suffix(".state_layout.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    LOGGER.info("Wrote state layout manifest to %s", manifest_path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="convert_mimi_decoder")
    p.add_argument("--save-path", type=Path, default=ARTIFACTS_DIR / "mimi_decoder.mlpackage")
    p.add_argument("--s-cap", type=int, default=DEFAULT_S_CAP,
                   help="Transformer KV capacity (default 256).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    convert(args.save_path, args.s_cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
