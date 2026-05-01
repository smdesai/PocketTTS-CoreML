"""Phase-3 per-submodel CoreML conversion scripts.

One script per submodel:
  - convert_text_conditioner.py   (plan Phase 3.1)
  - convert_flow_lm_main.py       (plan Phase 3.2)
  - convert_flow_lm_flow.py       (plan Phase 3.3)
  - convert_mimi_encoder.py       (plan Phase 3.4)
  - convert_mimi_decoder.py       (plan Phase 3.5)

Common recipe (plan Phase 3):
    1. Instantiate patched module; set eval(); load reference weights
       (via `pockettts_coreml.patches.build_patched_submodules`).
    2. Build example inputs with static shapes (RangeDim only where
       truly variable; the mimi encoder is the one such case).
    3. torch.jit.trace(module, example_inputs, strict=False).
    4. Assert `"aten::Int" not in str(traced.graph)`.
    5. ct.convert(traced,
           minimum_deployment_target=ct.target.iOS18,
           compute_precision=ct.precision.FLOAT16,
           compute_units=ct.ComputeUnit.CPU_AND_NE,
       ).
    6. Save to Artifacts/en_alba_fp16/<name>.mlpackage.
    7. Spot-check: predict on a known input; compare to eager-fp32
       output with atol=5e-3, rtol=5e-3 (fp16 roundoff tolerance).

Run via `python -m pockettts_coreml.convert --all` or individual
`--only <submodel>` commands. See `convert.__main__`.
"""
from __future__ import annotations

from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "Artifacts" / "en_alba_fp16"

__all__ = ["ARTIFACTS_DIR"]
