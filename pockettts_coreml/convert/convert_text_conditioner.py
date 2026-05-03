"""Convert the LUTConditioner (text -> embeddings) to CoreML fp16.

Plan Phase 3.1:
  Input:  tokens: int32[1, S_text], S_text static = 128 (plan decision).
  Output: embeddings: fp16[1, 128, 1024].
  Graph:  one `nn.Embedding(4001, 1024)` (n_bins + 1 for padding).
  Size:   4001 x 1024 x 2 bytes ~= 8 MB.

Shape decision: per plan §Phase 3 Risk flags, we static-pad text to
128 rather than using `ct.RangeDim`. The reference caps per-chunk text
tokens at 50 (default_parameters.MAX_TOKEN_PER_CHUNK), so 128 covers
every chunk with margin and avoids dynamic-shape ANE fallback risk.
Padding convention: token id 4000 (= n_bins), which maps to the
Embedding's "padding index" (nn.Embedding(4001, ...)).

Compute-units note: the embedding is a single gather and is likely to
run on CPU; that's fine (plan 3.1). Size dominated by the 4001x1024
weight matrix.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import coremltools as ct
import torch
import torch.nn as nn

from pockettts_coreml.convert import ARTIFACTS_DIR
from pockettts_coreml.convert._common import (
    convert_and_save,
    fp16_allclose,
    setup_logging,
    trace_module,
)
from pockettts_coreml.patches import build_patched_submodules

LOGGER = logging.getLogger("pockettts_coreml.convert.text_conditioner")

S_TEXT_STATIC = 128


class _TextConditionerGraph(nn.Module):
    """Thin wrapper around nn.Embedding for stable trace semantics.

    `LUTConditioner._get_condition` expects a TokenizedText wrapper;
    here we take a plain int32 token tensor and run `self.embed(tokens)`
    directly. Weight tying is preserved via a shared Parameter.
    """

    def __init__(self, n_bins_plus_one: int, dim: int):
        super().__init__()
        self.embed = nn.Embedding(n_bins_plus_one, dim)

    @torch.no_grad()
    def load_reference_weights(self, ref_lut: nn.Module) -> None:
        self.embed.weight.copy_(ref_lut.embed.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens)


def convert(save_path: Path, language: str = "english") -> None:
    ps = build_patched_submodules(language=language)
    ref_lut = ps.text_conditioner  # LUTConditioner
    n_bins_plus_one = int(ref_lut.embed.num_embeddings)  # 4001
    dim = int(ref_lut.embed.embedding_dim)              # 1024
    LOGGER.info("text_conditioner: vocab=%d, dim=%d", n_bins_plus_one, dim)

    mod = _TextConditionerGraph(n_bins_plus_one, dim)
    mod.load_reference_weights(ref_lut)
    mod.eval()

    # Example input: zeros (padding id). Any int32 tensor traces identically.
    tokens_example = torch.zeros((1, S_TEXT_STATIC), dtype=torch.int32)
    traced = trace_module(mod, (tokens_example,), "text_conditioner")

    mlmodel = convert_and_save(
        traced,
        inputs=[
            ct.TensorType(
                name="tokens",
                shape=(1, S_TEXT_STATIC),
                dtype=ct.converters.mil.mil.types.int32,
            ),
        ],
        outputs=[ct.TensorType(name="embeddings")],
        save_path=save_path,
        name="text_conditioner",
    )

    # Spot check with the actual fixture prompt tokens if available.
    fixture_tokens = _build_fixture_tokens(ref_lut)
    eager_out = mod(fixture_tokens).detach()
    mlmodel_out = _predict_embeddings(mlmodel, fixture_tokens)
    fp16_allclose(
        torch.as_tensor(mlmodel_out), eager_out,
        atol=5e-3, rtol=5e-3, label="text_conditioner",
    )


def _build_fixture_tokens(ref_lut) -> torch.Tensor:
    """Tokenize the Phase-1 fixture prompt, pad/truncate to S_TEXT_STATIC."""
    prompt = "Pocket TTS is a lightweight text-to-speech model."
    tokenized = ref_lut.tokenizer(prompt)  # TokenizedText(namedtuple-ish)
    tokens = tokenized[0] if not torch.is_tensor(tokenized) else tokenized
    if hasattr(tokens, "tokens"):
        tokens = tokens.tokens
    tokens = tokens.to(torch.int32)
    T = tokens.shape[-1]
    padded = torch.zeros((1, S_TEXT_STATIC), dtype=torch.int32)
    copy_len = min(T, S_TEXT_STATIC)
    padded[0, :copy_len] = tokens[0, :copy_len]
    return padded


def _predict_embeddings(mlmodel, tokens: torch.Tensor):
    import numpy as np

    out = mlmodel.predict({"tokens": tokens.cpu().numpy().astype(np.int32)})
    # Keys depend on how we named the output; fall back to first value.
    if "embeddings" in out:
        return out["embeddings"]
    return next(iter(out.values()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="convert_text_conditioner")
    p.add_argument("--save-path", type=Path, default=ARTIFACTS_DIR / "text_conditioner.mlpackage")
    p.add_argument("--language", default="english",
                   help="Reference language config (english/spanish/...).")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    convert(args.save_path, language=args.language)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
