"""Convert MimiModel.encode_to_latent to CoreML fp16.

Plan Phase 3.4:
  Input:  waveform: fp32[1, 1, T_audio]  (T_audio multiple of 1920)
  Output: latents:  fp32[1, 32, T_audio/1920]

#### STATUS: YELLOW (flagged)

The reference encoder_transformer uses `StreamingMultiheadAttention` with
a `_LinearKVCacheBackend.rope_offset` path that computes an int32-typed
`ts = offset + arange(T)` and feeds it through trigonometric ops. Under
trace this lands as an int32 op into CoreML's `inverse` (1/x), which is
fp-only:

    ValueError: Op "253" (op_type: inverse) Input x="D.1" expects
    tensor or scalar of dtype from type domain ['fp16', 'fp32'] but
    got tensor[1,int32]

To clear this, the encoder_transformer needs the same patch surgery as
the FlowLM transformer (replace the reference MHA + RoPE with the
patched pure-functional versions). That surgery is beyond the compressed
Phase-2+3 scope — the encoder runs ONCE per voice clone (not per audio
frame), so it's a voice-loading prerequisite but not a hot-path blocker.

#### Workaround for this cycle
The oracle fixture loads voice from a pre-exported `.safetensors` (see
`metadata.json`: voice_path_kind="safetensors"), so the reference voice
path works without the encoder. Swift-side voice cloning from raw wav
is deferred to a follow-up cycle that builds a `PatchedSEANetEncoder`
wrapping patched StreamingConv1d + patched MHA.

This conversion script is left in place so a future cycle can finish
it; the unit-test gate and `__main__` driver skip it unless
`--include-yellow` is passed.
"""
from __future__ import annotations

import argparse
import logging
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
from pockettts_coreml.patches import build_patched_submodules, ensure_reference_on_path

LOGGER = logging.getLogger("pockettts_coreml.convert.mimi_encoder")

# Default voice-reference length in seconds. The actual voice_reference.wav
# is variable; we fix a single static size to simplify the first port.
FRAME_SIZE = 1920
DEFAULT_T_AUDIO_SECONDS = 4
# Concrete T_audio: 24000 * 4 = 96000 (exactly 50 frames at 12.5 Hz).
DEFAULT_T_AUDIO = 24000 * DEFAULT_T_AUDIO_SECONDS


def _install_trace_friendly_forwards() -> None:
    """Monkey-patch reference StreamingConv*/SEANet forwards with trace-friendly
    versions that accept tensor `B` (avoid beartype int-check failures) and
    use pure-functional zero-state logic.

    Only the non-streaming (`model_state=None`) path is patched — the
    streaming path remains untouched for Python reference use elsewhere.
    Idempotent.
    """
    ensure_reference_on_path()
    import torch
    from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d

    if getattr(StreamingConv1d.forward, "_trace_patched", False):
        return

    def _sc1d_forward(self, x, model_state):  # type: ignore[no-redef]
        # Zero-state path only; matches reference with model_state=None.
        # Effective kernel size = (K-1)*dilation + 1; left-context (TP) is
        # (kernel_eff - stride). For the non-streaming init_state call the
        # reference makes `previous` of shape [B, C, TP] but TP is fixed by
        # the module config, not by input — we can pre-compute it in Python.
        dilation = self.conv.dilation[0]
        K = self.conv.kernel_size[0]
        S = self.conv.stride[0]
        kernel_eff = (K - 1) * dilation + 1
        TP = kernel_eff - S
        if model_state is None:
            # Zero-init previous; no `first`-flag branch needed for the
            # 'constant' pad mode used by English.
            if TP > 0:
                zeros = torch.zeros(
                    (x.shape[0], self.conv.in_channels, TP),
                    device=x.device, dtype=x.dtype,
                )
                x = torch.cat([zeros, x], dim=-1)
            return self.conv(x)
        # Fallback: defer to reference (won't happen in our trace path).
        raise NotImplementedError("streaming StreamingConv1d not trace-patched")

    _sc1d_forward._trace_patched = True  # type: ignore[attr-defined]
    StreamingConv1d.forward = _sc1d_forward  # type: ignore[method-assign]

    def _sct1d_forward(self, x, mimi_state):  # type: ignore[no-redef]
        K = self.convtr.kernel_size[0]
        S = self.convtr.stride[0]
        PT = K - S
        y = self.convtr(x)
        if mimi_state is None:
            # Zero partial: just drop last PT samples (pure conv_transpose
            # with no overlap-add carry).
            if PT > 0:
                y = y[..., :-PT]
            return y
        raise NotImplementedError("streaming StreamingConvTranspose1d not trace-patched")

    _sct1d_forward._trace_patched = True  # type: ignore[attr-defined]
    StreamingConvTranspose1d.forward = _sct1d_forward  # type: ignore[method-assign]

    # Also patch resample.ConvDownsample1d / ConvTrUpsample1d if they share
    # the same beartype landmine. Inspect as needed.
    from pocket_tts.modules import resample as _resample
    for cls_name in ("ConvDownsample1d", "ConvTrUpsample1d"):
        cls = getattr(_resample, cls_name, None)
        if cls is None:
            continue
        if getattr(cls.forward, "_trace_patched", False):
            continue
        _orig = cls.forward

        def _make_wrap(_orig_fn, _cls_name=cls_name):
            def _wrapped(self, x, mimi_state=None):
                # Resample uses an internal StreamingConv1d or ConvTranspose1d
                # that we've already patched; call original by bypassing
                # beartype wrapper where possible. Since `_orig_fn` is the
                # beartype-decorated version, a direct call should be safe
                # as we've already ensured inputs are tensors.
                return _orig_fn(self, x, mimi_state)

            _wrapped._trace_patched = True
            return _wrapped

        # We leave resample unpatched unless explicitly needed; the
        # underlying conv patch is usually sufficient.



class _MimiEncoderWrap(nn.Module):
    """Wrap `MimiModel.encode_to_latent` for trace.

    `encode_to_latent` internally calls:
      1. pad_for_conv1d
      2. self.encoder(x, model_state=None)
      3. self.encoder_transformer(emb, model_state=None)
      4. self._to_framerate(emb) -> self.downsample(emb, model_state=None)

    With `model_state=None`, all StreamingConv1d/StreamingConvTranspose1d
    instances use their `init_state(B, 0)` fresh-zero path — fully
    traceable with no dict-mutation landmines.
    """

    def __init__(self, mimi_model: nn.Module):
        super().__init__()
        self.mimi = mimi_model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Call the encoder's internals directly, skipping `pad_for_conv1d`
        # which uses Python-int arithmetic on `x.shape[-1]` and fails
        # beartype under trace. Caller must pass a waveform whose length
        # is already a multiple of `frame_size` (enforced in `convert()`).
        emb = self.mimi.encoder(waveform, model_state=None)
        (emb,) = self.mimi.encoder_transformer(emb, model_state=None)
        # _to_framerate: downsample if encoder_frame_rate != frame_rate.
        if self.mimi.encoder_frame_rate != self.mimi.frame_rate:
            emb = self.mimi.downsample(emb, model_state=None)
        return emb


def convert(save_path: Path, t_audio: int = DEFAULT_T_AUDIO) -> None:
    # `build_patched_submodules` disables beartype on first import (needed
    # for CoreML tracing of reference Mimi modules).
    ps = build_patched_submodules()
    mimi = ps.mimi_model
    mimi.eval()

    # Sanity: T_audio must be divisible by the encoder's frame_size.
    frame_size = mimi.frame_size
    assert t_audio % frame_size == 0, f"t_audio {t_audio} must be multiple of {frame_size}"
    T_lat = t_audio // frame_size
    LOGGER.info("mimi_encoder: T_audio=%d samples (%.1fs), T_latent=%d",
                t_audio, t_audio / mimi.sample_rate, T_lat)

    wrap = _MimiEncoderWrap(mimi)
    wrap.eval()

    example_wave = torch.randn(1, 1, t_audio) * 0.1
    traced = trace_module(wrap, (example_wave,), "mimi_encoder")

    mlmodel = convert_and_save(
        traced,
        inputs=[ct.TensorType(name="waveform", shape=(1, 1, t_audio))],
        outputs=[ct.TensorType(name="latents")],
        save_path=save_path,
        name="mimi_encoder",
    )

    # Spot-check.
    with torch.no_grad():
        eager = wrap(example_wave)
    pred = mlmodel.predict({"waveform": example_wave.numpy().astype(np.float32)})
    pred_t = torch.as_tensor(pred.get("latents", next(iter(pred.values()))))
    LOGGER.info("mimi_encoder: eager shape=%s, pred shape=%s", eager.shape, pred_t.shape)
    fp16_allclose(pred_t, eager, atol=5e-2, rtol=5e-2, label="mimi_encoder")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="convert_mimi_encoder")
    p.add_argument("--save-path", type=Path, default=ARTIFACTS_DIR / "mimi_encoder.mlpackage")
    p.add_argument("--t-audio", type=int, default=DEFAULT_T_AUDIO,
                   help=f"Audio length in samples (default: {DEFAULT_T_AUDIO}).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    convert(args.save_path, args.t_audio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
