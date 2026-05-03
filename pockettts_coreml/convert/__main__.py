"""Drive per-submodel CoreML conversions.

Usage:
    python -m pockettts_coreml.convert --all
    python -m pockettts_coreml.convert --only flow_lm_main
    python -m pockettts_coreml.convert --only text_conditioner --only flow_lm_flow

Status (all GREEN as of 2026-05-01, see individual convert_*.py
module docstrings for per-submodel details):
    text_conditioner  GREEN  (fp16; 8 MB)
    flow_lm_main      GREEN  (fp16; 144 MB; rank-5 KV I/O)
    flow_lm_flow      GREEN  (fp16; 19 MB; num_steps=1)
    mimi_encoder      GREEN  (fp16; 20 MB; static T_audio=96000)
    mimi_decoder      GREEN  (fp32; 39 MB; single packed-state blob,
                              s_cap=1024 transformer KV)

The `--include-yellow` flag is retained for historical symmetry but
has no effect — both Mimi packages are part of the default `--all`.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from pockettts_coreml.convert import ARTIFACTS_DIR
from pockettts_coreml.convert._common import setup_logging

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

LOGGER = logging.getLogger("pockettts_coreml.convert.__main__")

_GREEN = ("text_conditioner", "flow_lm_main", "flow_lm_prefill", "flow_lm_flow")
_YELLOW = ("mimi_encoder", "mimi_decoder")
_ALL = _GREEN + _YELLOW


def _run_one(name: str, include_yellow: bool, out_dir: Path, language: str) -> tuple[bool, str]:
    """Invoke a specific convert script. Returns (ok, message)."""
    t0 = time.time()
    try:
        if name == "text_conditioner":
            from pockettts_coreml.convert.convert_text_conditioner import convert
            convert(out_dir / "text_conditioner.mlpackage", language=language)
        elif name == "flow_lm_main":
            from pockettts_coreml.convert.convert_flow_lm_main import convert
            convert(out_dir / "flow_lm_main.mlpackage", language=language)
        elif name == "flow_lm_prefill":
            from pockettts_coreml.convert.convert_flow_lm_prefill import convert
            convert(out_dir / "flow_lm_prefill.mlpackage", language=language)
        elif name == "flow_lm_flow":
            from pockettts_coreml.convert.convert_flow_lm_flow import convert
            convert(out_dir / "flow_lm_flow.mlpackage", language=language)
        elif name == "mimi_encoder":
            if not include_yellow:
                return True, "SKIPPED (yellow; pass --include-yellow to attempt)"
            from pockettts_coreml.convert.convert_mimi_encoder import convert
            convert(out_dir / "mimi_encoder.mlpackage", language=language)
        elif name == "mimi_decoder":
            if not include_yellow:
                return True, "SKIPPED (yellow; pass --include-yellow to attempt)"
            from pockettts_coreml.convert.convert_mimi_decoder import convert
            convert(out_dir / "mimi_decoder.mlpackage", language=language)
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
    p.add_argument(
        "--language", default="english",
        help="Reference language config (english/spanish/...). "
             "Selects `pockettts_coreml/reference/pocket_tts/config/<language>.yaml`.",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output directory for .mlpackage artifacts. Defaults to "
             "`Artifacts/en_alba_fp16/` (for backwards compat with the "
             "English bundle).",
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

    out_dir = args.out if args.out is not None else ARTIFACTS_DIR
    if not out_dir.is_absolute():
        out_dir = (_REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Output directory: %s (language=%s)", out_dir, args.language)

    results: list[tuple[str, bool, str]] = []
    for name in targets:
        LOGGER.info("=== Converting %s ===", name)
        ok, msg = _run_one(
            name, include_yellow=args.include_yellow,
            out_dir=out_dir, language=args.language,
        )
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
