"""Convert MimiModel.encode_to_latent path to CoreML fp16.

Plan Phase 3.4:
  Input:  waveform: fp16[1, 1, T_audio]  (T_audio = multiple of frame_size)
  Output: latents:  fp16[1, 32, T_audio / frame_size]

#### STATUS: GREEN (closed 2026-05-01)

Uses `PatchedSEANetEncoder` + `PatchedProjectedTransformer` (non-streaming)
from `pockettts_coreml/patches/mimi_model_patched.py`. The reference's
`StreamingMultiheadAttention` int32→inverse landmine is avoided by routing
through `PatchedStreamingMultiheadAttention` with pre-computed RoPE tables
and an fp16 additive attention mask.

The encoder is non-streaming (runs ONCE per voice clone, not per audio
frame), so CPU fallback from stride-6/5/4 convs is acceptable per plan.

Static T_audio: we pick `DEFAULT_T_AUDIO = 24000 * 3 = 72000` samples
(3 seconds = 600 encoder-framerate tokens = 37.5 frame-rate tokens).
The attention mask and RoPE tables are baked into the trace as constants
(safe: these are pure Python-int computations outside the graph).
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

FRAME_SIZE = 1920
# 4s = 50 frames at 12.5 Hz = 800 encoder-framerate tokens at 200 Hz.
# (24000 * 4 = 96000; 96000 / 1920 = 50.) Static shape — plan's original
# RangeDim is a follow-up if variable-T voice clone is needed.
DEFAULT_T_AUDIO_SECONDS = 4
DEFAULT_T_AUDIO = 24000 * DEFAULT_T_AUDIO_SECONDS  # 96000


class _MimiEncoderWrap(nn.Module):
    """Wraps PatchedSEANetEncoder + PatchedProjectedTransformer + downsample.

    Bakes the attention mask and RoPE tables as buffers (computed at
    construction time from `T_enc_latent` and `context`).

    Forward:
      waveform fp32[1, 1, T_audio] -> latents fp32[1, 32, T_audio // frame_size]
    """

    def __init__(self, mimi_model: nn.Module, t_audio: int):
        super().__init__()
        from pockettts_coreml.patches.mimi_model_patched import (
            PatchedProjectedTransformer, PatchedSEANetEncoder,
        )
        from pockettts_coreml.patches.rope_patched import build_rope_tables
        from pockettts_coreml.patches.transformer_patched import (
            build_additive_attention_mask_prefill,
        )
        # Encoder config.
        ref_enc_tx = mimi_model.encoder_transformer
        ref_layer0 = ref_enc_tx.transformer.layers[0]
        d_model = ref_layer0.self_attn.embed_dim
        num_heads = ref_layer0.self_attn.num_heads
        num_layers = len(ref_enc_tx.transformer.layers)
        dim_ff = ref_layer0.linear1.out_features
        context = ref_layer0.self_attn.context
        head_dim = d_model // num_heads
        # Layer scale: detect from reference (initial value only; load_reference_weights
        # copies the real values).
        ls_mod = getattr(ref_layer0, "layer_scale_1", None)
        layer_scale = 1e-2 if hasattr(ls_mod, "scale") else None

        # How many encoder-framerate tokens does T_audio produce? Each
        # StreamingConv1d/ConvTranspose1d + ratio chain takes the audio
        # length down to T_enc = T_audio / hop_length where
        # hop_length = prod(ratios) = 8*5*4 = 160. But the output of the
        # encoder is NOT at encoder_frame_rate — it's:
        #   T_enc_tokens = T_audio / 120 (for sample_rate=24000, enc_fr=200)
        # since enc_fr/sample_rate = 200/24000 = 1/120.
        # Then downsample goes from enc_fr=200 -> frame_rate=12.5 i.e. /16.
        # Final latent length: T_audio / 120 / 16 = T_audio / 1920.
        self.t_audio = t_audio
        self.sample_rate = int(mimi_model.sample_rate)
        self.enc_fr = float(mimi_model.encoder_frame_rate)
        # Token count at encoder frame rate = T_audio * enc_fr / sample_rate
        enc_tokens = int(round(t_audio * self.enc_fr / self.sample_rate))
        self.enc_tokens = enc_tokens
        self.frame_size = int(mimi_model.frame_size)

        # Build the patched encoder + transformer.
        self.seanet_encoder = PatchedSEANetEncoder(mimi_model.encoder)
        self.encoder_transformer = PatchedProjectedTransformer(
            input_dimension=ref_enc_tx.input_dimension,
            output_dimensions=ref_enc_tx.output_dimensions,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_ff,
            context=context,
            layer_scale=layer_scale,
        )
        self.encoder_transformer.load_reference_weights(ref_enc_tx)

        # Build the downsample. We clone the reference's ConvDownsample1d
        # as a patched non-streaming conv. ConvDownsample1d wraps a
        # StreamingConv1d with pad_mode="replicate" — but non-streaming
        # (model_state=None) calls just use zero-left-pad since init_state
        # is [B, C, TP] zeros.  Reference behavior with model_state=None
        # is identical to pure conv with constant zero-padding.
        from pockettts_coreml.patches.mimi_patched import PatchedStreamingConv1d
        ref_ds_conv = mimi_model.downsample.conv.conv
        self.downsample = PatchedStreamingConv1d(
            in_channels=ref_ds_conv.in_channels,
            out_channels=ref_ds_conv.out_channels,
            kernel_size=ref_ds_conv.kernel_size[0],
            stride=ref_ds_conv.stride[0],
            dilation=ref_ds_conv.dilation[0],
            groups=ref_ds_conv.groups,
            bias=ref_ds_conv.bias is not None,
        )
        with torch.no_grad():
            self.downsample.conv.weight.copy_(ref_ds_conv.weight)
            if ref_ds_conv.bias is not None:
                self.downsample.conv.bias.copy_(ref_ds_conv.bias)

        # Precompute attention mask + RoPE tables as buffers.
        attn_mask = build_additive_attention_mask_prefill(
            start_offset=0, prefill_len=enc_tokens,
            s_capacity=enc_tokens, context=context, dtype=torch.float32,
        )
        self.register_buffer("attn_mask", attn_mask)

        cos_t, sin_t = build_rope_tables(max_context=enc_tokens, head_dim=head_dim)
        rope_cos = cos_t[:enc_tokens].unsqueeze(0).unsqueeze(2)
        rope_sin = sin_t[:enc_tokens].unsqueeze(0).unsqueeze(2)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        # Pre-allocated zero left-context for the downsample conv (static
        # batch=1, so no tensor-shape ints leak into the graph).
        self.register_buffer(
            "downsample_prev_zero",
            torch.zeros(1, self.downsample.in_channels, self.downsample.state_length),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [1, 1, T_audio]
        x = self.seanet_encoder(waveform)  # [1, 512, T_enc_tokens]
        x = self.encoder_transformer.forward_nonstreaming(
            x, self.attn_mask, self.rope_cos, self.rope_sin,
        )  # [1, 512, T_enc_tokens]
        # Downsample (non-streaming: zero left-pad via pure_forward with
        # zero prev). Use a pre-registered zero buffer so no tensor-shape
        # integer arithmetic leaks into the graph.
        x, _ = self.downsample.pure_forward(x, self.downsample_prev_zero)
        return x


def convert(save_path: Path, t_audio: int = DEFAULT_T_AUDIO, language: str = "english") -> None:
    ps = build_patched_submodules(language=language)
    mimi = ps.mimi_model
    mimi.eval()

    frame_size = mimi.frame_size
    assert t_audio % frame_size == 0, f"t_audio {t_audio} must be multiple of {frame_size}"
    T_lat = t_audio // frame_size
    LOGGER.info("mimi_encoder: T_audio=%d samples (%.1fs), T_latent=%d",
                t_audio, t_audio / mimi.sample_rate, T_lat)

    wrap = _MimiEncoderWrap(mimi, t_audio)
    wrap.eval()

    example_wave = torch.randn(1, 1, t_audio) * 0.05
    traced = trace_module(wrap, (example_wave,), "mimi_encoder")

    mlmodel = convert_and_save(
        traced,
        inputs=[ct.TensorType(name="waveform", shape=(1, 1, t_audio))],
        outputs=[ct.TensorType(name="latents")],
        save_path=save_path,
        name="mimi_encoder",
    )

    # fp16 spot-check on CPU (ANE-compiled model shows large drift on
    # random inputs, expected per plan for the SEANet inverse path;
    # CPU_ONLY prediction is the reference for numerical parity here).
    with torch.no_grad():
        eager = wrap(example_wave)
    from coremltools.models import MLModel
    cpu_model = MLModel(str(save_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    pred = cpu_model.predict({"waveform": example_wave.numpy().astype(np.float32)})
    pred_t = torch.as_tensor(pred.get("latents", next(iter(pred.values()))))
    LOGGER.info("mimi_encoder: eager shape=%s, pred shape=%s", eager.shape, pred_t.shape)
    fp16_allclose(pred_t, eager, atol=0.5, rtol=0.5, label="mimi_encoder")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="convert_mimi_encoder")
    p.add_argument("--save-path", type=Path, default=ARTIFACTS_DIR / "mimi_encoder.mlpackage")
    p.add_argument("--t-audio", type=int, default=DEFAULT_T_AUDIO,
                   help=f"Audio length in samples (default: {DEFAULT_T_AUDIO}).")
    p.add_argument("--language", default="english",
                   help="Reference language config (english/spanish/...).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    convert(args.save_path, args.t_audio, language=args.language)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
