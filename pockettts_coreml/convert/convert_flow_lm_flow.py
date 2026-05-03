"""Convert the FlowLM flow-head (SimpleMLPAdaLN + 1 LSD step) to CoreML fp16.

Plan Phase 3.3:
  Inputs:
    c:     fp32[1, 1024]
    s:     fp32[1, 1]
    t:     fp32[1, 1]
    x:     fp32[1, 32]         -- sampled noise (Swift pre-samples via vDSP_vgauss).
    noise: fp32[1, 32]         -- provided for API symmetry; same as `x` for N=1.
           (plan note: x IS the noise at N=1; we expose both names for
           clarity and future N>1 variants)

  Output:
    next_latent: fp32[1, 32]   -- x + u * num_steps_inv   (num_steps_inv=1.0 for N=1)

N=1 is the production default. Higher-N variants are a Phase-4+ concern.
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
from pockettts_coreml.patches import build_patched_submodules

LOGGER = logging.getLogger("pockettts_coreml.convert.flow_lm_flow")


class _FlowLMFlowWrap(nn.Module):
    """Pin `num_steps_inv=1.0` so it's a trace-time constant."""

    def __init__(self, flow: nn.Module):
        super().__init__()
        self.flow = flow

    def forward(
        self,
        c: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.flow(c, s, t, x, num_steps_inv=1.0)


def convert(save_path: Path, language: str = "english") -> None:
    ps = build_patched_submodules(language=language)
    flow = ps.flow_lm_flow
    wrap = _FlowLMFlowWrap(flow)
    wrap.eval()

    d_model = ps.d_model
    ldim = ps.ldim
    LOGGER.info("flow_lm_flow: d_model=%d ldim=%d", d_model, ldim)

    c = torch.randn(1, d_model)
    s = torch.zeros(1, 1)
    t = torch.ones(1, 1)
    x = torch.randn(1, ldim)
    example_inputs = (c, s, t, x)
    traced = trace_module(wrap, example_inputs, "flow_lm_flow")

    # Optional FP32 override (diagnostic; matches the main/prefill convert
    # scripts' POCKETTTS_FLOW_*_FP32 flags for drift-localization work).
    precision = (
        ct.precision.FLOAT32
        if _os.environ.get("POCKETTTS_FLOW_FLOW_FP32", "0") == "1"
        else ct.precision.FLOAT16
    )
    mlmodel = convert_and_save(
        traced,
        inputs=[
            ct.TensorType(name="c", shape=(1, d_model)),
            ct.TensorType(name="s", shape=(1, 1)),
            ct.TensorType(name="t", shape=(1, 1)),
            ct.TensorType(name="x", shape=(1, ldim)),
        ],
        outputs=[ct.TensorType(name="next_latent")],
        save_path=save_path,
        name="flow_lm_flow",
        precision=precision,
    )

    # fp16 spot-check (atol=1e-3 per plan; small 512-wide 6-block MLP).
    with torch.no_grad():
        eager_out = wrap(c, s, t, x)
    feed = {
        "c": c.numpy().astype(np.float32),
        "s": s.numpy().astype(np.float32),
        "t": t.numpy().astype(np.float32),
        "x": x.numpy().astype(np.float32),
    }
    out = mlmodel.predict(feed)
    pred = torch.as_tensor(out.get("next_latent", next(iter(out.values()))))
    fp16_allclose(pred, eager_out, atol=5e-2, rtol=5e-2, label="flow_lm_flow")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="convert_flow_lm_flow")
    p.add_argument("--save-path", type=Path, default=ARTIFACTS_DIR / "flow_lm_flow.mlpackage")
    p.add_argument("--language", default="english",
                   help="Reference language config (english/spanish/...).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    convert(args.save_path, language=args.language)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
