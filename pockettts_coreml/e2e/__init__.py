"""End-to-end CoreML-driven generation helpers.

This subpackage composes all 5 converted `.mlpackage` bundles to generate
audio from text, mirroring the reference's `TTSModel.generate_audio`
flow but replacing every forward pass with a CoreML `.predict()` call.

Used by `tests/test_coreml_end_to_end_python.py` to drive the audio-PSNR
gate vs `golden/output.wav`.
"""
from __future__ import annotations

__all__ = ["CoreMLGenerator"]

from pockettts_coreml.e2e.generator import CoreMLGenerator  # noqa: E402
