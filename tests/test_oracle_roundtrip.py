"""Verify that the reference model, run twice under the same seed, produces
bitwise-identical per-stage outputs.

This is Phase 1's gate: if this test passes we've proven the reference is
deterministic under our (seed + `torch.set_num_threads(1)`) discipline,
which is a prerequisite for every later phase asserting parity against
the golden bundle.

The test SKIPS cleanly when the golden fixtures or gated weights are
unavailable. Set `POCKETTTS_ORACLE_READY=1` in addition to `HF_TOKEN` to
force the full roundtrip.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from pockettts_coreml.oracle.compare import assert_bundle_close

FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent
    / "pockettts_coreml" / "oracle" / "fixtures" / "english_alba_seed42"
)


def _fixture_ready() -> bool:
    """A 'ready' fixture has a voice file (.wav or .safetensors) plus at
    least one golden tensor bundle."""
    golden_dir = FIXTURE_DIR / "golden"
    if not golden_dir.exists():
        return False
    has_voice = any(FIXTURE_DIR.glob("*.safetensors")) or any(
        FIXTURE_DIR.glob("*.wav")
    )
    return has_voice and any(golden_dir.glob("*.safetensors"))


def _hf_creds_available() -> bool:
    """HF_TOKEN env var OR cached `huggingface-cli login` token."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import HfFolder
        return bool(HfFolder.get_token())
    except Exception:
        return False


needs_oracle = pytest.mark.skipif(
    not (_fixture_ready() and os.environ.get("POCKETTTS_ORACLE_READY") == "1"
         and _hf_creds_available()),
    reason=(
        "Oracle fixture or HF credentials not available. Run "
        "`python -m pockettts_coreml.oracle.dump_golden` after "
        "`huggingface-cli login`, then set POCKETTTS_ORACLE_READY=1."
    ),
)


def test_package_importable():
    """Phase-1 gate #1: the package must import cleanly with no side
    effects that require the reference subtree.
    """
    import pockettts_coreml  # noqa: F401

    assert pockettts_coreml.__version__


def test_dump_golden_help_runs():
    """Phase-1 gate #3: `python -m pockettts_coreml.oracle.dump_golden --help`
    must print a usable help string without importing the reference subtree.

    We invoke the argument parser directly rather than spawning a subprocess
    so the test stays fast and portable. The parser must be constructable
    without the reference being present.
    """
    from pockettts_coreml.oracle.dump_golden import _build_parser

    parser = _build_parser()
    help_text = parser.format_help()
    assert "dump_golden" in help_text
    assert "--voice" in help_text
    assert "--seed" in help_text


@needs_oracle
def test_oracle_roundtrip_bitwise_identical(tmp_path: Path):
    """Phase-1 gate #5: re-running dump_golden under the same seed must
    produce outputs bitwise-identical to the shipped golden bundle.

    We run `dump_golden` into a scratch directory and diff every
    safetensors file in `golden/` against the fixture.
    """
    from pockettts_coreml.oracle.dump_golden import dump_golden

    # Re-run into a temp directory so we don't clobber the committed
    # fixture on accidental drift.
    scratch = tmp_path / "english_alba_seed42"
    scratch.mkdir(parents=True)

    # Find the voice file used for the committed fixture (either a .wav
    # or a pre-exported .safetensors; metadata.json is authoritative).
    golden_meta = json.loads((FIXTURE_DIR / "metadata.json").read_text())
    voice = Path(golden_meta["voice_path"])
    if not voice.is_absolute():
        voice = FIXTURE_DIR / voice.name
    assert voice.exists(), f"voice file referenced in metadata not found: {voice}"

    dump_golden(output_dir=scratch, voice_path=voice)

    # Compare every stage bundle at atol=rtol=0. fp32 deterministic
    # reference + seeded RNG + 1 thread should be bitwise identical.
    for stage_file in sorted((FIXTURE_DIR / "golden").glob("*.safetensors")):
        scratch_file = scratch / "golden" / stage_file.name
        assert scratch_file.exists(), (
            f"Expected {scratch_file} to be produced by dump_golden"
        )
        assert_bundle_close(
            actual_path=scratch_file,
            golden_path=stage_file,
            rtol=0.0,
            atol=0.0,
            stage_name=stage_file.stem,
        )

    # metadata.json should also match in key fields (not full-file; the
    # generated_at_utc timestamp will differ, obviously).
    fresh_meta = json.loads((scratch / "metadata.json").read_text())
    # `golden_meta` was read above for voice resolution; reuse it.
    for key in ("prompt", "voice_name", "seed", "lsd_decode_steps",
                "temperature", "eos_threshold", "audio_samples",
                "tokenizer_sha256"):
        assert fresh_meta.get(key) == golden_meta.get(key), (
            f"metadata.json field {key!r} drifted: "
            f"{fresh_meta.get(key)!r} vs {golden_meta.get(key)!r}"
        )
