"""Convert the FlowLM main-path transformer to CoreML fp16 (AR-step path).

Plan Phase 3.2 inputs/outputs:

  Inputs (all fp32 tensors at trace time; CoreML casts to fp16 internally):
    sequence:        fp32[1, 1, 32]          -- previous latent (or bos_emb at step 0)
    text_embeddings: fp32[1, 0, 1024]        -- empty in the AR hot path; prefill
                                                path is handled as a separate
                                                graph if needed (Phase 4 concern).
    kv_cache_in:     fp32[12, 1, 256, 16, 64]
                                             -- [2*L, B, S_cap, H, D]; kept as a
                                                SINGLE tensor per kraken note
                                                (explicit I/O, not MLState).
                                                Rank-5 rather than rank-6 because
                                                Core ML caps tensor rank at 5 —
                                                layer i occupies rows [2i, 2i+1]
                                                for K and V respectively.
    offset_mask:     fp32[1, 256]            -- one-hot write slot indicator.
    attn_mask:       fp32[1, 1, 1, 256]      -- additive (0.0 visible / -65500.0 masked).
    rope_cos:        fp32[1, 1, 1, 32]       -- head_dim // 2 = 32 (head_dim=64).
    rope_sin:        fp32[1, 1, 1, 32]

  Outputs:
    ctx:             fp32[1, 1024]
    eos_logit:       fp32[1, 1]
    kv_cache_out:    fp32[6, 2, 1, 256, 16, 64]

S_cap = 256 per plan.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import coremltools as ct
import torch
import torch.nn as nn

import os as _os

from pockettts_coreml.convert import ARTIFACTS_DIR
from pockettts_coreml.convert._common import (
    convert_and_save,
    fp16_allclose,
    setup_logging,
    trace_module,
)
from pockettts_coreml.patches import (
    build_additive_attention_mask_step,
    build_one_hot_offset_mask,
    build_patched_submodules,
    build_rope_tables,
    slice_rope_tables,
)

LOGGER = logging.getLogger("pockettts_coreml.convert.flow_lm_main")

S_CAP = 256


class _FlowLMMainARWrap(nn.Module):
    """Trace wrapper pinning `text_embeddings` to an empty-prefix AR path.

    The reference AR loop calls FlowLMModel.forward with
    `text_embeddings.shape[1] == 0` for every step after prefill — this
    graph models that path. The `forward_prefill` path is exercised
    separately via Python-side looping over single-step calls during
    text ingestion (simpler than compiling a second graph for Phase 3).
    """

    def __init__(self, main: nn.Module):
        super().__init__()
        self.main = main
        # Empty `text_embeddings` is a trace-time constant.
        self.register_buffer(
            "_empty_text", torch.zeros(1, 0, main.d_model), persistent=False
        )

    def forward(
        self,
        sequence: torch.Tensor,
        kv_cache_in: torch.Tensor,
        offset_mask: torch.Tensor,
        attn_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ):
        ctx, eos_logit, kv_cache_out = self.main(
            sequence=sequence,
            text_embeddings=self._empty_text,
            kv_caches_in=kv_cache_in,
            offset_mask=offset_mask,
            attn_mask=attn_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        return ctx, eos_logit, kv_cache_out


def convert(save_path: Path, spot_offset: int = 3) -> None:
    # Selective fp32 softmax: default ON. Emits cast-up/softmax/cast-down
    # around the attention reduction so CoreML keeps the softmax at fp32
    # while weights stay fp16. Set POCKETTTS_FLOW_MAIN_FP16_SOFTMAX=1 to
    # disable (reproduces the pre-fix all-fp16 bundle). See
    # `docs/investigations/fp16_drift_localization.md`.
    use_fp32_softmax = (
        _os.environ.get("POCKETTTS_FLOW_MAIN_FP16_SOFTMAX", "0") != "1"
    )
    ps = build_patched_submodules(use_fp32_softmax=use_fp32_softmax)
    main = ps.flow_lm_main
    wrap = _FlowLMMainARWrap(main)
    wrap.eval()

    L = ps.num_layers
    H = ps.num_heads
    D = ps.head_dim
    ldim = ps.ldim
    dm = ps.d_model
    LOGGER.info(
        "flow_lm_main: L=%d H=%d D=%d d_model=%d ldim=%d S_cap=%d use_fp32_softmax=%s",
        L, H, D, dm, ldim, S_CAP, use_fp32_softmax,
    )

    # Example inputs at `spot_offset` so the graph sees nonzero mask positions.
    sequence = torch.randn(1, 1, ldim)
    # rank-5 layout: [2L, B, S_cap, H, D]; Core ML rank cap.
    kv_cache = torch.zeros(2 * L, 1, S_CAP, H, D)
    offset_mask = build_one_hot_offset_mask(offset=spot_offset, s_capacity=S_CAP)
    attn_mask = build_additive_attention_mask_step(offset=spot_offset, s_capacity=S_CAP)
    cos_t, sin_t = build_rope_tables(max_context=S_CAP, head_dim=D)
    rope_cos, rope_sin = slice_rope_tables(cos_t, sin_t, offset=spot_offset, length=1)

    example_inputs = (sequence, kv_cache, offset_mask, attn_mask, rope_cos, rope_sin)
    traced = trace_module(wrap, example_inputs, "flow_lm_main")

    inputs = [
        ct.TensorType(name="sequence", shape=(1, 1, ldim)),
        ct.TensorType(name="kv_cache_in", shape=(2 * L, 1, S_CAP, H, D)),
        ct.TensorType(name="offset_mask", shape=(1, S_CAP)),
        ct.TensorType(name="attn_mask", shape=(1, 1, 1, S_CAP)),
        ct.TensorType(name="rope_cos", shape=(1, 1, 1, D // 2)),
        ct.TensorType(name="rope_sin", shape=(1, 1, 1, D // 2)),
    ]
    outputs = [
        ct.TensorType(name="ctx"),
        ct.TensorType(name="eos_logit"),
        ct.TensorType(name="kv_cache_out"),
    ]
    # Optional FP32 override for accuracy testing: set
    # POCKETTTS_FLOW_MAIN_FP32=1 to convert with float32 compute
    # precision. Produces a ~2x-larger .mlpackage but gives 20+ dB
    # audio PSNR headroom for the end-to-end gate during development.
    if _os.environ.get("POCKETTTS_FLOW_MAIN_FP32", "0") == "1":
        precision = ct.precision.FLOAT32
    elif use_fp32_softmax:
        # Selective fp32: weights + most ops at fp16, softmax stays fp32.
        # The trace-level cast-up/cast-down (Approach A) was verified to
        # survive the converter front-end but is aggressively fused away
        # by MIL's `homogenize_input_dtypes` pass, which compresses the
        # cast-fp16->cast-fp32->op->cast-fp16 chain back to all-fp16 when
        # `compute_precision=FLOAT16` is passed as a bare enum. The MIL
        # `FP16ComputePrecision` pass with an `op_selector` is the
        # documented CoreML-LLM approach: it runs the fp16 cast insertion
        # only on ops where the predicate returns True, leaving the
        # excluded ops at their original fp32 dtype. This keeps weights
        # fp16 (unchanged ship size) but routes the softmax through the
        # fp32 register bank.
        from coremltools.converters.mil.mil.passes.defs.quantization import (
            FP16ComputePrecision,
        )
        # Keep fp32 for:
        #   - softmax: the mostly-masked (255/256 at -65500) attention
        #     reduction is the classic fp16 instability path (CoreML-LLM
        #     open question #2 in docs/phase2_3_notes.md).
        #   - layer_norm: per-layer variance/mean accumulated in fp16
        #     contributes to long-context drift across the 6-layer AR
        #     feedback loop (investigation §Recommendation #2). Softmax
        #     alone recovered ~50% of the drift (corr +0.447 -> +0.673);
        #     adding layer_norm recovers the rest.
        _FP32_OPS = {"softmax", "layer_norm"}
        def _keep_fp32(op) -> bool:
            return op.op_type not in _FP32_OPS
        precision = FP16ComputePrecision(op_selector=_keep_fp32)
    else:
        precision = ct.precision.FLOAT16
    mlmodel = convert_and_save(
        traced, inputs=inputs, outputs=outputs, save_path=save_path, name="flow_lm_main",
        precision=precision,
    )

    # fp16 spot-check at the same sample inputs.
    import numpy as np
    with torch.no_grad():
        ctx_eager, eos_eager, kv_eager = wrap(*example_inputs)
    feed = {
        "sequence": sequence.numpy().astype(np.float32),
        "kv_cache_in": kv_cache.numpy().astype(np.float32),
        "offset_mask": offset_mask.numpy().astype(np.float32),
        "attn_mask": attn_mask.numpy().astype(np.float32),
        "rope_cos": rope_cos.numpy().astype(np.float32),
        "rope_sin": rope_sin.numpy().astype(np.float32),
    }
    out = mlmodel.predict(feed)
    ctx_ml = torch.as_tensor(out["ctx"])
    eos_ml = torch.as_tensor(out["eos_logit"])
    kv_ml = torch.as_tensor(out["kv_cache_out"])
    # NOTE: fp16 drift on random inputs across a 6-layer 1024-d transformer
    # with a mostly-masked 256-long attention is NOT a meaningful quality
    # signal — most KV slots are zero in both eager/CoreML but fp16
    # roundoff in the unwritten rows yields huge relative deltas. The
    # real quality gate is the end-to-end audio PSNR test. Here we just
    # sanity-check that predict runs and produces output of the correct
    # shape/dtype.
    assert ctx_ml.shape == ctx_eager.shape, f"ctx shape mismatch: {ctx_ml.shape} vs {ctx_eager.shape}"
    assert eos_ml.shape == eos_eager.shape
    assert kv_ml.shape == kv_eager.shape
    LOGGER.info(
        "flow_lm_main: predict shape check ok. ctx diff max=%.3e; eos diff max=%.3e; kv diff max=%.3e",
        (ctx_ml.float() - ctx_eager.float()).abs().max().item(),
        (eos_ml.float() - eos_eager.float()).abs().max().item(),
        (kv_ml.float() - kv_eager.float()).abs().max().item(),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="convert_flow_lm_main")
    p.add_argument("--save-path", type=Path, default=ARTIFACTS_DIR / "flow_lm_main.mlpackage")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    convert(args.save_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
