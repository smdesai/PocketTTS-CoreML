"""Patched forks of reference modules that hit CoreML traceability landmines.

Each submodule is a drop-in pure-functional replacement for the
correspondingly-named reference file. The reference subtree at
`pockettts_coreml/reference/` is NEVER modified; parity is verified by
loading the same weights into patched modules and comparing outputs.

Public API:
    - PatchedStreamingMultiheadAttention  (transformer_patched)
    - build_rope_tables, apply_rope        (rope_patched)
    - PatchedSimpleMLPAdaLN                (mlp_patched)
    - PatchedFlowLMMain, PatchedFlowLMFlow (flow_lm_patched)
    - PatchedStreamingConv1d, PatchedStreamingConvTranspose1d  (mimi_patched)
    - build_patched_submodules             (this file)

### Phase-2 quality gate

Per-module parity is validated by `tests/test_patch_parity.py` at fp32
with `atol=1e-5, rtol=1e-5`, and traceability (zero `aten::Int`) by
`tests/test_trace_no_aten_int.py`. The end-to-end audio-vs-golden gate
that was originally specified (MSE<1e-6) is REPLACED in this cycle by
the CoreML-composed end-to-end test (`tests/test_coreml_end_to_end_python.py`)
per kraken's compressed-phase instructions. Rationale: per-module parity
at atol=1e-5 fp32 is numerically stronger evidence than end-to-end MSE
would be, because it catches drift at every boundary rather than the
aggregate; and the compressed schedule prioritizes getting to fp16
CoreML artifacts.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pockettts_coreml.patches.flow_lm_patched import (
    PatchedFlowLMFlow,
    PatchedFlowLMMain,
)
from pockettts_coreml.patches.mimi_patched import (
    PatchedStreamingConv1d,
    PatchedStreamingConvTranspose1d,
)
from pockettts_coreml.patches.mlp_patched import PatchedSimpleMLPAdaLN
from pockettts_coreml.patches.rope_patched import (
    apply_rope,
    build_rope_tables,
    slice_rope_tables,
)
from pockettts_coreml.patches.transformer_patched import (
    PatchedStreamingMultiheadAttention,
    PatchedStreamingTransformer,
    PatchedStreamingTransformerLayer,
    build_additive_attention_mask_prefill,
    build_additive_attention_mask_step,
    build_one_hot_offset_mask,
    build_scatter_prefill_mask,
    init_empty_kv_cache,
)

__all__ = [
    # Patched classes
    "PatchedStreamingMultiheadAttention",
    "PatchedStreamingTransformer",
    "PatchedStreamingTransformerLayer",
    "PatchedSimpleMLPAdaLN",
    "PatchedFlowLMMain",
    "PatchedFlowLMFlow",
    "PatchedStreamingConv1d",
    "PatchedStreamingConvTranspose1d",
    # RoPE
    "build_rope_tables",
    "slice_rope_tables",
    "apply_rope",
    # Pre-trace helpers
    "build_additive_attention_mask_step",
    "build_additive_attention_mask_prefill",
    "build_one_hot_offset_mask",
    "build_scatter_prefill_mask",
    "init_empty_kv_cache",
    # Model assembly
    "build_patched_submodules",
    "PatchedSubmodules",
    "ensure_reference_on_path",
]

_REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"


def ensure_reference_on_path(disable_beartype: bool = False) -> None:
    """Add the vendored reference subtree to sys.path so `import pocket_tts`
    resolves. Idempotent. Same pattern as `oracle.dump_golden`.

    If `disable_beartype=True`, additionally monkey-patches
    `beartype.claw.beartype_this_package` to a no-op BEFORE `pocket_tts`
    is imported. This is required for CoreML conversion of reference
    submodules (e.g. Mimi encoder/decoder) because beartype's runtime
    int-type checks fire under `torch.jit.trace`'s `_slow_forward`
    (tensor-wrapped `batch_size` violates `<class 'int'>`).

    The disable is only safe BEFORE the reference has been imported —
    otherwise the decorators have already run. We check via
    `"pocket_tts" in sys.modules` and warn/raise if it's too late.
    """
    ref = str(_REFERENCE_DIR)
    if ref not in sys.path:
        sys.path.insert(0, ref)
    if disable_beartype:
        if "pocket_tts" in sys.modules:
            raise RuntimeError(
                "ensure_reference_on_path(disable_beartype=True) must be called "
                "BEFORE `pocket_tts` is first imported. Restart the Python process."
            )
        import beartype.claw
        # no-op: prevent package-wide beartype decoration. Module-level
        # `from beartype.typing import ...` imports still work (those are
        # just type aliases, not decorators).
        beartype.claw.beartype_this_package = lambda *a, **k: None  # type: ignore[assignment]


@dataclass
class PatchedSubmodules:
    """Bag of patched submodules with reference weights loaded.

    Each field is a fully-initialized patched nn.Module. The originating
    reference `TTSModel` is kept (`tts_model`) so conversion scripts can
    pull config/derived metadata (sample rate, ldim, etc.) without
    re-parsing the YAML.
    """

    tts_model: Any  # pocket_tts.models.tts_model.TTSModel
    flow_lm_main: PatchedFlowLMMain
    flow_lm_flow: PatchedFlowLMFlow
    # The reference text conditioner is a plain `nn.Embedding` wrapped in
    # `LUTConditioner`; no patching is required. We expose it directly.
    text_conditioner: torch.nn.Module  # LUTConditioner
    # Architecture constants (resolved from config for easy downstream use).
    d_model: int
    num_heads: int
    num_layers: int
    dim_feedforward: int
    head_dim: int
    ldim: int
    # Mimi architecture exposure (CoreML decoder composition needs these).
    mimi_model: Any  # pocket_tts.models.mimi.MimiModel
    mimi_d_model: int
    mimi_num_heads: int
    mimi_num_layers: int
    mimi_context: int
    mimi_inner_dim: int  # = ldim at the FlowLM boundary (32)
    mimi_outer_dim: int
    mimi_frame_rate: float
    mimi_sample_rate: int


def build_patched_submodules(
    language: str = "english",
    temp: float = 0.7,
    lsd_decode_steps: int = 1,
    noise_clamp: float | None = None,
    eos_threshold: float = -4.0,
) -> PatchedSubmodules:
    """Load the reference TTSModel, then build patched equivalents for
    the two FlowLM submodules (main + flow) with the reference weights
    copied in.

    The Mimi submodel patches (`PatchedStreamingConv1d`,
    `PatchedStreamingConvTranspose1d`) are per-layer wrappers; the
    Phase-3 `convert_mimi_encoder.py` and `convert_mimi_decoder.py`
    scripts build their own traced graphs directly from the reference
    `MimiModel` + the patched conv wrappers (because the full SEANet
    decoder is a heavier assembly than a single pair of submodules).

    Args/return: see the arg docs in `oracle/dump_golden.py` for the
    fixed fixture values. Defaults here match the Phase-1 golden.
    """
    # Disable beartype before the first reference import so CoreML tracing
    # of reference Mimi/SEANet modules (where int-typed params meet tensor
    # inputs under `torch.jit.trace._slow_forward`) doesn't throw
    # BeartypeCallHintParamViolation. No-op if already disabled.
    already_imported = "pocket_tts" in sys.modules
    ensure_reference_on_path(disable_beartype=not already_imported)
    # These imports only resolve after the reference is on sys.path.
    from pocket_tts.models.tts_model import TTSModel  # noqa: E402

    torch.set_num_threads(1)
    tts_model = TTSModel.load_model(
        language=language,
        temp=temp,
        lsd_decode_steps=lsd_decode_steps,
        noise_clamp=noise_clamp,
        eos_threshold=eos_threshold,
        quantize=False,
    )
    tts_model.eval()

    flow_cfg = tts_model.config.flow_lm
    tcfg = flow_cfg.transformer
    d_model = int(tcfg.d_model)
    num_heads = int(tcfg.num_heads)
    num_layers = int(tcfg.num_layers)
    dim_feedforward = int(d_model * tcfg.hidden_scale)
    head_dim = d_model // num_heads
    # `flow_lm.ldim` is the continuous latent dim passed into the flow head.
    ldim = int(tts_model.flow_lm.ldim)
    flow_dim = int(flow_cfg.flow.dim)
    flow_depth = int(flow_cfg.flow.depth)

    # --- Build patched FlowLM main (transformer + prefix/eos head) ------
    main = PatchedFlowLMMain(
        ldim=ldim,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
    )
    main.eval()
    main.load_reference_weights(tts_model.flow_lm)

    # --- Build patched FlowLM flow head ---------------------------------
    flow = PatchedFlowLMFlow(
        ldim=ldim, d_model=d_model, flow_dim=flow_dim, flow_depth=flow_depth,
    )
    flow.eval()
    flow.load_reference_weights(tts_model.flow_lm)

    # --- Mimi transformer params ----------------------------------------
    mcfg = tts_model.config.mimi
    mt = mcfg.transformer
    return PatchedSubmodules(
        tts_model=tts_model,
        flow_lm_main=main,
        flow_lm_flow=flow,
        text_conditioner=tts_model.flow_lm.conditioner,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        head_dim=head_dim,
        ldim=ldim,
        mimi_model=tts_model.mimi,
        mimi_d_model=int(mt.d_model),
        mimi_num_heads=int(mt.num_heads),
        mimi_num_layers=int(mt.num_layers),
        mimi_context=int(mt.context),
        mimi_inner_dim=int(mcfg.inner_dim),
        mimi_outer_dim=int(mcfg.outer_dim),
        mimi_frame_rate=float(mcfg.frame_rate),
        mimi_sample_rate=int(mcfg.sample_rate),
    )
