"""Shared helpers for Phase-3 conversion scripts.

Design: keep conversion scripts short and uniform. Each one imports
these helpers, builds its input-spec list, traces, and calls
`convert_and_save(...)` with a human-readable name.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import coremltools as ct
import torch
import torch.nn as nn

LOGGER = logging.getLogger("pockettts_coreml.convert")

_DEFAULT_MIN_TARGET = ct.target.iOS18
_DEFAULT_PRECISION = ct.precision.FLOAT16
_DEFAULT_COMPUTE_UNITS = ct.ComputeUnit.CPU_AND_NE


def assert_no_aten_int(traced: torch.jit.ScriptModule, label: str) -> None:
    """Assert the traced graph has zero `aten::Int` ops.

    `aten::Int` is the canonical CoreML-LLM blocker (external_research.md
    §B.1 item 11); it arises from any `int(tensor.item())` or integer
    arithmetic on tensor-derived scalars. All five patched submodels
    must trace cleanly.
    """
    graph_s = str(traced.graph)
    if "aten::Int" in graph_s:
        bad_lines = [ln for ln in graph_s.splitlines() if "aten::Int" in ln]
        raise AssertionError(
            f"[{label}] traced graph contains {len(bad_lines)} aten::Int op(s):\n"
            + "\n".join(bad_lines[:5])
        )


def trace_module(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    label: str,
) -> torch.jit.ScriptModule:
    """Eval-mode trace with `strict=False`. Asserts graph has no aten::Int."""
    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, example_inputs, strict=False)
    assert_no_aten_int(traced, label)
    return traced


def convert_and_save(
    traced: torch.jit.ScriptModule,
    *,
    inputs: Sequence[ct.TensorType],
    outputs: Sequence[ct.TensorType] | None,
    save_path: Path,
    name: str,
    compute_units: ct.ComputeUnit = _DEFAULT_COMPUTE_UNITS,
    precision: ct.precision = _DEFAULT_PRECISION,
    min_target: ct.target = _DEFAULT_MIN_TARGET,
) -> ct.models.MLModel:
    """Convert a traced module and write the .mlpackage bundle to disk.

    Returns the converted model instance so callers can immediately
    run a `.predict()` spot-check without reloading from disk.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("[%s] coremltools converting (precision=%s, compute_units=%s) ...",
                name, precision, compute_units)
    t0 = time.time()
    kwargs: dict[str, Any] = dict(
        inputs=list(inputs),
        minimum_deployment_target=min_target,
        compute_precision=precision,
        compute_units=compute_units,
        convert_to="mlprogram",
    )
    if outputs is not None:
        kwargs["outputs"] = list(outputs)
    mlmodel = ct.convert(traced, **kwargs)
    dt = time.time() - t0
    LOGGER.info("[%s] convert done in %.1fs; saving to %s", name, dt, save_path)
    mlmodel.save(str(save_path))
    return mlmodel


def fp16_allclose(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 5e-3,
    rtol: float = 5e-3,
    label: str = "",
) -> None:
    """fp16-tolerant comparison used for the per-submodel predict spot-check."""
    diff = (actual.float() - expected.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = expected.float().abs().clamp_min(1e-6)
    max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
    ok = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    if not ok:
        raise AssertionError(
            f"[{label}] fp16 mismatch vs eager: max_abs={max_abs:.3e}, "
            f"max_rel={max_rel:.3e}, atol={atol}, rtol={rtol}, "
            f"shape_a={tuple(actual.shape)}, shape_e={tuple(expected.shape)}"
        )
    LOGGER.info("[%s] fp16 spot-check ok: max_abs=%.3e, max_rel=%.3e",
                label, max_abs, max_rel)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
