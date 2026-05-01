"""Convert MimiModel.decode_from_latent per-frame path to CoreML fp16.

Plan Phase 3.5:
  Inputs:
    latent: fp32[1, 32, 1]   -- one AR frame
    Plus streaming-state tensors (conv previous / partial buffers and
    the 2-layer mimi transformer KV cache) packed into a single blob.
  Outputs:
    audio: fp32[1, 1, 1920]
    plus updated state.

#### STATUS: YELLOW (flagged — conversion deferred)

The reference decoder path has two non-trivial landmines that are each
bigger than this cycle's 30-minute per-submodel budget:

 1. `decoder_transformer` (`ProjectedTransformer` wrapping a 2-layer
    `StreamingTransformer` with context=250) reuses the same
    `StreamingMultiheadAttention` + `_LinearKVCacheBackend` that fails
    CoreML conversion in the mimi_encoder (int32 feeding `inverse`).
    Converting it requires building a
    `PatchedMimiDecoderTransformer` that wires our patched MHA in at
    the `ProjectedTransformer` scope, loads reference weights, and
    maintains KV cache as explicit I/O with context=250.

 2. The SEANet decoder has ~12 streaming `StreamingConv1d` /
    `StreamingConvTranspose1d` instances plus a `ConvTrUpsample1d`;
    each needs its `previous`/`partial` buffer threaded through as
    explicit input/output. The patched
    `PatchedStreamingConv1d.pure_forward` /
    `PatchedStreamingConvTranspose1d.pure_forward` wrappers are ready
    for this, but assembling them into a `PatchedSEANetDecoder` that
    mirrors the reference's exact weight layout is a dedicated
    multi-hour task.

Blocking scope: the entire `.mlpackage` for mimi_decoder.

#### Workaround for this cycle

The Python-driven end-to-end test (`tests/test_coreml_end_to_end_python.py`)
composes the three converted FlowLM-side submodels
(`text_conditioner`, `flow_lm_main`, `flow_lm_flow`) with CoreML
`.predict()` but drives the reference `MimiModel.decode_from_latent`
directly in PyTorch. That still validates the FlowLM-side CoreML port
end-to-end on audio PSNR while leaving the mimi_decoder
CoreML conversion as a follow-up.

If a downstream phase needs Swift-side mimi_decoder dispatch, this
script is the hook point: replace the reference calls with patched
equivalents, thread state as explicit I/O, and trace.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pockettts_coreml.convert import ARTIFACTS_DIR
from pockettts_coreml.convert._common import setup_logging

LOGGER = logging.getLogger("pockettts_coreml.convert.mimi_decoder")


def convert(save_path: Path) -> None:
    raise NotImplementedError(
        "mimi_decoder CoreML conversion is DEFERRED (see module docstring). "
        "The Python end-to-end test drives the reference mimi_decoder directly; "
        "a follow-up cycle adds a PatchedSEANetDecoder + "
        "PatchedMimiDecoderTransformer assembly to close this."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="convert_mimi_decoder")
    p.add_argument("--save-path", type=Path, default=ARTIFACTS_DIR / "mimi_decoder.mlpackage")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    LOGGER.error("mimi_decoder conversion is DEFERRED in this cycle (see docstring).")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
