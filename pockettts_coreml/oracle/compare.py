"""Stage-agnostic comparison helpers for the per-stage golden bundles.

Phase 1 defines the interface and the tightest (`atol=0`) re-run check.
Later phases (3) will call `assert_close` with the per-stage tolerances
listed in the plan (§3.1-3.5):

    text_conditioner:  atol=1e-3,  rtol=1e-3   (fp16 cast of embedding table)
    flow_lm_main:      atol=5e-3,  rtol=5e-3   (fp16 6-layer drift)
    flow_lm_flow:      atol=1e-3,  rtol=1e-3
    mimi_encoder:      atol=1e-3,  rtol=1e-3
    mimi_decoder:      atol=5e-3,  rtol=5e-3
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


@dataclass(frozen=True)
class CompareResult:
    stage_name: str
    tensor_key: str
    max_abs_diff: float
    max_rel_diff: float
    shape: tuple[int, ...]
    passed: bool


def assert_close(
    actual: torch.Tensor | np.ndarray,
    golden: torch.Tensor | np.ndarray,
    *,
    rtol: float,
    atol: float,
    stage_name: str,
    tensor_key: str = "<unnamed>",
) -> CompareResult:
    """Compare `actual` against `golden` for a single tensor.

    Raises AssertionError on mismatch with a diagnostic message that names
    the stage and tensor. Returns a CompareResult for programmatic use
    (e.g. aggregation across many tensors in a bundle).
    """
    a = _to_torch(actual).detach().cpu()
    g = _to_torch(golden).detach().cpu()

    if a.shape != g.shape:
        raise AssertionError(
            f"[{stage_name}/{tensor_key}] shape mismatch: "
            f"actual {tuple(a.shape)} vs golden {tuple(g.shape)}"
        )
    if a.dtype != g.dtype:
        # Upcast to the wider dtype for comparison; many stages cross
        # fp16/fp32 boundaries deliberately.
        common = torch.promote_types(a.dtype, g.dtype)
        a = a.to(common)
        g = g.to(common)

    diff = (a.float() - g.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() > 0 else 0.0
    denom = g.float().abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item()) if diff.numel() > 0 else 0.0

    passed = bool(torch.allclose(a, g, rtol=rtol, atol=atol, equal_nan=False))
    if not passed:
        raise AssertionError(
            f"[{stage_name}/{tensor_key}] numerical mismatch: "
            f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, "
            f"rtol={rtol:.0e}, atol={atol:.0e}, shape={tuple(a.shape)}"
        )

    return CompareResult(
        stage_name=stage_name,
        tensor_key=tensor_key,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        shape=tuple(a.shape),
        passed=True,
    )


def assert_bundle_close(
    actual_path: Path | str,
    golden_path: Path | str,
    *,
    rtol: float,
    atol: float,
    stage_name: str,
) -> list[CompareResult]:
    """Compare two safetensors bundles key-by-key.

    Used by `tests/test_oracle_roundtrip.py` to verify that a fresh run
    reproduces the golden bundle bit-for-bit (rtol=atol=0).
    """
    actual = load_file(str(actual_path))
    golden = load_file(str(golden_path))

    missing_in_actual = sorted(set(golden) - set(actual))
    missing_in_golden = sorted(set(actual) - set(golden))
    if missing_in_actual or missing_in_golden:
        raise AssertionError(
            f"[{stage_name}] bundle keyset mismatch: "
            f"missing_in_actual={missing_in_actual!r}, "
            f"missing_in_golden={missing_in_golden!r}"
        )

    results: list[CompareResult] = []
    for key in sorted(golden.keys()):
        results.append(
            assert_close(
                actual[key],
                golden[key],
                rtol=rtol,
                atol=atol,
                stage_name=stage_name,
                tensor_key=key,
            )
        )
    return results


def save_bundle(tensors: dict[str, torch.Tensor], path: Path | str) -> None:
    """Write a stage bundle to disk as safetensors. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # safetensors requires contiguous tensors on cpu.
    cleaned = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    save_file(cleaned, str(path))


def _to_torch(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.asarray(x))
