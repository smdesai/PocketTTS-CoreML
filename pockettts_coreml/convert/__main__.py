"""Drive per-submodel CoreML conversions.

Usage:
    python -m pockettts_coreml.convert --all
    python -m pockettts_coreml.convert --only flow_lm_main
    python -m pockettts_coreml.convert --only text_conditioner --only flow_lm_flow

Status in this cycle (see individual convert_*.py module docstrings
for details):
    text_conditioner  GREEN  (runs, converts, fp16 spot-check passes)
    flow_lm_main      GREEN  (runs, converts, shape spot-check passes)
    flow_lm_flow      GREEN  (runs, converts, fp16 spot-check passes)
    mimi_encoder      YELLOW (conversion blocked on StreamingMultiheadAttention
                              int32->inverse; docstring has the plan)
    mimi_decoder      YELLOW (deferred; docstring has the plan)
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from pockettts_coreml.convert import ARTIFACTS_DIR
from pockettts_coreml.convert._common import setup_logging

LOGGER = logging.getLogger("pockettts_coreml.convert.__main__")

_GREEN = ("text_conditioner", "flow_lm_main", "flow_lm_flow")
_YELLOW = ("mimi_encoder", "mimi_decoder")
_ALL = _GREEN + _YELLOW


def _run_one(name: str, include_yellow: bool) -> tuple[bool, str]:
    """Invoke a specific convert script. Returns (ok, message)."""
    t0 = time.time()
    try:
        if name == "text_conditioner":
            from pockettts_coreml.convert.convert_text_conditioner import convert
            convert(ARTIFACTS_DIR / "text_conditioner.mlpackage")
        elif name == "flow_lm_main":
            from pockettts_coreml.convert.convert_flow_lm_main import convert
            convert(ARTIFACTS_DIR / "flow_lm_main.mlpackage")
        elif name == "flow_lm_flow":
            from pockettts_coreml.convert.convert_flow_lm_flow import convert
            convert(ARTIFACTS_DIR / "flow_lm_flow.mlpackage")
        elif name == "mimi_encoder":
            if not include_yellow:
                return True, "SKIPPED (yellow; pass --include-yellow to attempt)"
            from pockettts_coreml.convert.convert_mimi_encoder import convert
            convert(ARTIFACTS_DIR / "mimi_encoder.mlpackage")
        elif name == "mimi_decoder":
            if not include_yellow:
                return True, "SKIPPED (yellow; pass --include-yellow to attempt)"
            from pockettts_coreml.convert.convert_mimi_decoder import convert
            convert(ARTIFACTS_DIR / "mimi_decoder.mlpackage")
        else:
            return False, f"unknown submodel '{name}'"
    except Exception as exc:
        return False, f"FAILED after {time.time() - t0:.1f}s: {exc!r}"
    return True, f"ok ({time.time() - t0:.1f}s)"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m pockettts_coreml.convert",
        description="Drive all or a subset of per-submodel CoreML conversions.",
    )
    p.add_argument("--all", action="store_true", help=f"Run all 5 submodels: {_ALL}")
    p.add_argument(
        "--only", action="append", default=[],
        choices=list(_ALL),
        help="Run only specific submodel(s). Repeatable.",
    )
    p.add_argument(
        "--include-yellow", action="store_true",
        help="Include YELLOW-status submodels (will fail with documented errors).",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    if args.all:
        targets = list(_ALL)
    elif args.only:
        targets = args.only
    else:
        p.error("Must specify --all or --only <submodel>.")

    results: list[tuple[str, bool, str]] = []
    for name in targets:
        LOGGER.info("=== Converting %s ===", name)
        ok, msg = _run_one(name, include_yellow=args.include_yellow)
        results.append((name, ok, msg))
        LOGGER.info("=== %s: %s ===", name, msg)

    LOGGER.info("")
    LOGGER.info("Summary:")
    any_fail = False
    for name, ok, msg in results:
        tag = "PASS" if ok else "FAIL"
        LOGGER.info("  %s  %-18s  %s", tag, name, msg)
        if not ok:
            any_fail = True
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
