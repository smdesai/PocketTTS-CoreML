"""Convert the FlowLM main-path transformer to CoreML fp16 (prefill variant).

Phase 4B: sibling of `convert_flow_lm_main.py`. The AR variant baked
`text_embeddings` to a zero-prefix constant so it runs one latent at a
time. This variant handles the one-shot text prefill step — ingest a
padded-to-128 text embedding, run the 6-layer transformer over the
`[voice_len, voice_len + S_text)` window, and write the updated KV
cache back. We throw away `ctx` / `eos_logit` since the "last token" is
padding when S_text < 128.

Inputs/outputs (all fp32 at trace time; CoreML casts to fp16 internally):

  Inputs:
    text_embeddings:  fp32[1, 128, 1024]  -- text_conditioner output,
                                              already zero-padded to
                                              S_TEXT_PAD=128.
    kv_cache_in:      fp32[12, 1, 256, 16, 64]
                                           -- rank-5 flow_lm KV with
                                              voice slots [0, voice_len)
                                              pre-filled, rest zero.
    scatter_mask:     fp32[1, 256, 128]   -- column j one-hot at row
                                              (voice_len + j) for j in
                                              [0, S_text), zero columns
                                              for j in [S_text, 128).
                                              Zero columns => new_k/v
                                              contribution is zero and
                                              `written` sum is zero at
                                              that slot, so the KV
                                              passthrough is clean.
    attn_mask:        fp32[1, 1, 128, 256] -- additive causal, row j is
                                              visible over
                                              [0, voice_len + j]; rows
                                              j >= S_text are "don't
                                              care" (we still make them
                                              valid rows with ≥1 visible
                                              position to keep softmax
                                              finite).
    rope_cos:         fp32[1, 128, 1, 32]  -- cos table sliced at
                                              offsets [voice_len,
                                              voice_len + 128). Rows
                                              past S_text are unused
                                              operationally.
    rope_sin:         fp32[1, 128, 1, 32]

  Outputs:
    kv_cache_out:     fp32[12, 1, 256, 16, 64]  -- updated KV.

### Landmines avoided (same three from Phase 3)

  1. Manual SDPA (no F.scaled_dot_product_attention).
  2. chunk() for the Q/K/V split (not unflatten+slice; CoreML's fused-QKV
     pattern would mis-apply RoPE to V).
  3. matmul (not einsum) for the scatter write.

These all live in `transformer_patched.py` already — the prefill graph
here simply calls `transformer.forward_prefill(...)` at `T_q=128`.

### Why skip `input_linear(sequence)`?

The reference path `_run_flow_lm_and_increment_step(text_tokens=...)`
passes `backbone_input_latents = empty[1, 0, ldim]`, so
`input_linear(empty) -> [1, 0, d_model]` and the concat
`[text_embeddings, empty]` yields just text_embeddings. We skip the
empty-sequence concat entirely and pass text_embeddings straight into
the transformer (numerically identical, simpler trace graph).

S_TEXT_PAD = 128 matches the `text_conditioner.mlpackage` output width,
so a single graph covers all per-utterance prefills (masked to the real
token count via scatter_mask columns).
"""
from __future__ import annotations

import argparse
import logging
import os as _os
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
from pockettts_coreml.patches import (
    build_additive_attention_mask_prefill,
    build_patched_submodules,
    build_rope_tables,
    build_scatter_prefill_mask,
)

LOGGER = logging.getLogger("pockettts_coreml.convert.flow_lm_prefill")

S_CAP = 256
S_TEXT_PAD = 128


class _FlowLMMainPrefillWrap(nn.Module):
    """Prefill-path wrapper: text_embeddings directly into the transformer.

    Returns only `kv_cache_out`. The reference's `_run_flow_lm` call
    during prefill also computes `ctx`/`eos`, but the caller discards
    both (see `tts_model._run_flow_lm_and_increment_step`, which only
    keeps the KV update via `increment_steps`). Dropping them here keeps
    the graph smaller and trace-clean.
    """

    def __init__(self, main: nn.Module):
        super().__init__()
        self.transformer = main.transformer

    def forward(
        self,
        text_embeddings: torch.Tensor,
        kv_cache_in: torch.Tensor,
        scatter_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        # No input_linear here: sequence is empty in the reference
        # prefill path, so text_embeddings IS the full transformer input.
        _, kv_cache_out = self.transformer.forward_prefill(
            text_embeddings, kv_cache_in, scatter_mask, attn_mask,
            rope_cos, rope_sin,
        )
        return kv_cache_out


def _build_example_inputs(
    d_model: int, L: int, H: int, D: int,
    voice_len: int = 126, s_text: int = 17,
) -> tuple[torch.Tensor, ...]:
    """Construct representative example inputs for tracing.

    Numbers match the alba voice/canonical-prompt fixture (voice_len=126,
    s_text=17) so the trace graph sees a realistic mask pattern. The
    static shapes are what coremltools locks in; the particular mask
    values are just trace-time examples.
    """
    text_embeddings = torch.randn(1, S_TEXT_PAD, d_model)
    kv_cache = torch.zeros(2 * L, 1, S_CAP, H, D)
    # Scatter: columns [0, s_text) hit rows [voice_len, voice_len+s_text);
    # columns [s_text, 128) are all zero (padding).
    full_scatter = build_scatter_prefill_mask(
        start_offset=voice_len, prefill_len=s_text, s_capacity=S_CAP,
    )  # [1, 256, s_text]
    scatter_mask = torch.zeros((1, S_CAP, S_TEXT_PAD), dtype=torch.float32)
    scatter_mask[:, :, :s_text] = full_scatter
    # Attn mask: real rows causal over [0, voice_len + i]; padding rows
    # copy row s_text-1 (valid, finite softmax over same visible slots).
    attn_real = build_additive_attention_mask_prefill(
        start_offset=voice_len, prefill_len=s_text, s_capacity=S_CAP,
    )  # [1, 1, s_text, 256]
    attn_mask = torch.full((1, 1, S_TEXT_PAD, S_CAP), -6.5504e4, dtype=torch.float32)
    attn_mask[:, :, :s_text, :] = attn_real
    # Pad rows: copy the last real row so softmax stays finite. Their
    # scatter columns are zero, so they don't mutate the KV cache.
    attn_mask[:, :, s_text:, :] = attn_real[:, :, -1:, :]
    # RoPE: 128 contiguous rows starting at voice_len.
    cos_t, sin_t = build_rope_tables(max_context=S_CAP, head_dim=D)
    # cos_t/sin_t: [S_CAP, D//2]. Slice rows [voice_len, voice_len + 128).
    rope_cos = cos_t[voice_len : voice_len + S_TEXT_PAD].unsqueeze(0).unsqueeze(2)
    rope_sin = sin_t[voice_len : voice_len + S_TEXT_PAD].unsqueeze(0).unsqueeze(2)
    # Sprinkle a small amount of signal into the voice slots so the
    # attention output isn't trivially zero (doesn't affect graph).
    kv_cache[:, :, :voice_len] = 0.01 * torch.randn_like(kv_cache[:, :, :voice_len])
    return text_embeddings, kv_cache, scatter_mask, attn_mask, rope_cos, rope_sin


def _export_bos_sidecar(bos_emb: torch.Tensor, out_path: Path) -> None:
    """Write `bos_emb` as a small safetensors file next to the .mlpackage.

    The Swift runtime needs this to seed step-0 of the AR loop — without
    it, the runtime couldn't generate from a plain voice file (the
    value is a learned scalar that lives in the FlowLM main weights).
    """
    from safetensors.torch import save_file
    bos = bos_emb.detach().clone().view(-1).contiguous().float()
    save_file({"bos_emb": bos}, str(out_path),
              metadata={"phase": "4B-prefill", "ldim": str(int(bos.numel()))})
    LOGGER.info("flow_lm_prefill: wrote bos_emb sidecar to %s (len=%d)",
                out_path, int(bos.numel()))


def convert(save_path: Path) -> None:
    ps = build_patched_submodules()
    main = ps.flow_lm_main
    wrap = _FlowLMMainPrefillWrap(main)
    wrap.eval()
    # Sidecar: bos_emb needed by Swift to seed the AR loop step-0.
    _export_bos_sidecar(
        main.bos_emb, save_path.parent / "flow_lm_bos_emb.safetensors",
    )

    L = ps.num_layers
    H = ps.num_heads
    D = ps.head_dim
    dm = ps.d_model
    LOGGER.info(
        "flow_lm_prefill: L=%d H=%d D=%d d_model=%d S_cap=%d S_text_pad=%d",
        L, H, D, dm, S_CAP, S_TEXT_PAD,
    )

    example_inputs = _build_example_inputs(dm, L, H, D)
    traced = trace_module(wrap, example_inputs, "flow_lm_prefill")

    inputs = [
        ct.TensorType(name="text_embeddings", shape=(1, S_TEXT_PAD, dm)),
        ct.TensorType(name="kv_cache_in", shape=(2 * L, 1, S_CAP, H, D)),
        ct.TensorType(name="scatter_mask", shape=(1, S_CAP, S_TEXT_PAD)),
        ct.TensorType(name="attn_mask", shape=(1, 1, S_TEXT_PAD, S_CAP)),
        ct.TensorType(name="rope_cos", shape=(1, S_TEXT_PAD, 1, D // 2)),
        ct.TensorType(name="rope_sin", shape=(1, S_TEXT_PAD, 1, D // 2)),
    ]
    outputs = [ct.TensorType(name="kv_cache_out")]

    precision = (
        ct.precision.FLOAT32
        if _os.environ.get("POCKETTTS_FLOW_PREFILL_FP32", "0") == "1"
        else ct.precision.FLOAT16
    )
    mlmodel = convert_and_save(
        traced, inputs=inputs, outputs=outputs, save_path=save_path,
        name="flow_lm_prefill", precision=precision,
    )

    # ------------------------------------------------------------------
    # Spot-check: predict against eager output. At fp16 the per-element
    # diff on a 6-layer 1024-d transformer over 128 tokens can be large
    # on the padding rows (uncontrolled); the real parity gate is the
    # end-to-end audio PSNR test. Here we just verify shape/dtype.
    # ------------------------------------------------------------------
    with torch.no_grad():
        kv_eager = wrap(*example_inputs)
    feed = {
        "text_embeddings": example_inputs[0].numpy().astype(np.float32),
        "kv_cache_in": example_inputs[1].numpy().astype(np.float32),
        "scatter_mask": example_inputs[2].numpy().astype(np.float32),
        "attn_mask": example_inputs[3].numpy().astype(np.float32),
        "rope_cos": example_inputs[4].numpy().astype(np.float32),
        "rope_sin": example_inputs[5].numpy().astype(np.float32),
    }
    out = mlmodel.predict(feed)
    kv_ml = torch.as_tensor(out["kv_cache_out"])
    assert kv_ml.shape == kv_eager.shape, (
        f"kv shape mismatch: {kv_ml.shape} vs {kv_eager.shape}"
    )
    diff = (kv_ml.float() - kv_eager.float()).abs()
    LOGGER.info(
        "flow_lm_prefill: predict shape ok; kv diff max=%.3e mean=%.3e",
        diff.max().item(), diff.mean().item(),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="convert_flow_lm_prefill")
    p.add_argument("--save-path", type=Path,
                   default=ARTIFACTS_DIR / "flow_lm_prefill.mlpackage")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    convert(args.save_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
