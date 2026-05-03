"""Export speaker_proj_weight (+ bos_before_voice if present) as sidecar.

Voice cloning is a two-stage pipeline:

1. `mimi_encoder.mlpackage`  →  latents  fp16[1, 32, T_latent]
2. Project to d_model via a learned linear, then run `flow_lm_prefill.mlpackage`
   over the projected conditioning to populate the voice KV cache.

Step 2 uses `flow_lm.speaker_proj_weight` (shape `[d_model=1024, ldim=32]`),
a plain `nn.Parameter` on the reference `flow_lm`. It's not part of any
existing exported mlpackage nor of the `flow_lm_bos_emb.safetensors`
sidecar, so we dump it out here so the Swift runtime can run stage 2
without re-tracing any CoreML bundle.

Some languages also set `flow_lm.insert_bos_before_voice = True`, in which
case the reference concatenates a learned `bos_before_voice` vector
(shape `[1, 1, d_model]`) onto the front of the projected conditioning.
If that parameter exists we emit it under key `bos_before_voice` so the
runtime can prepend it before calling flow_lm_prefill.

Output: `<out_dir>/speaker_proj.safetensors` — always contains
    - `speaker_proj`: fp32 [d_model, ldim]  (typically [1024, 32])
    - optional `bos_before_voice`: fp32 [1, 1, d_model]  (only for
      configs with `insert_bos_before_voice: true`)

Usage:
    python -m pockettts_coreml.convert.export_speaker_proj \\
        --language english --out Artifacts/en_alba_fp16
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from pockettts_coreml.convert import ARTIFACTS_DIR
from pockettts_coreml.convert._common import setup_logging
from pockettts_coreml.patches import build_patched_submodules

LOGGER = logging.getLogger("pockettts_coreml.convert.export_speaker_proj")


def export(save_path: Path, language: str = "english") -> None:
    """Extract speaker_proj_weight (+ bos_before_voice) from the reference
    flow_lm and write both to a single safetensors sidecar at `save_path`.
    """
    ps = build_patched_submodules(language=language)
    flow_lm = ps.tts_model.flow_lm

    if not hasattr(flow_lm, "speaker_proj_weight"):
        raise RuntimeError(
            f"[{language}] flow_lm has no `speaker_proj_weight` attribute. "
            f"Voice cloning is unsupported for this config."
        )
    w = flow_lm.speaker_proj_weight
    # `speaker_proj_weight` is declared as nn.Parameter in
    # TTSModel._from_pydantic_config_with_weights. Unwrap defensively in
    # case a future config change promotes it to a Module.
    if isinstance(w, torch.nn.Parameter):
        weight = w.detach().clone()
    elif isinstance(w, torch.Tensor):
        weight = w.detach().clone()
    elif hasattr(w, "weight"):
        weight = w.weight.detach().clone()
    else:
        raise RuntimeError(
            f"[{language}] speaker_proj_weight is of unexpected type "
            f"{type(w).__name__}; can't extract a raw tensor."
        )
    weight = weight.to(torch.float32).contiguous()
    # Shape check: expected [d_model, ldim] = [1024, 32] (6L), [2048, 32] (24L).
    # Ldim=32 for mimi.inner_dim=32 (all shipped configs). d_model varies:
    # 1024 for 6L configs, 2048 for french_24l.
    d_model = ps.d_model
    ldim = ps.ldim
    if tuple(weight.shape) != (d_model, ldim):
        raise RuntimeError(
            f"[{language}] speaker_proj_weight shape {tuple(weight.shape)} "
            f"!= expected [{d_model}, {ldim}]"
        )

    from safetensors.torch import save_file
    tensors: dict[str, torch.Tensor] = {"speaker_proj": weight}

    # bos_before_voice is optional — only present when
    # insert_bos_before_voice is True for this language's config.
    if getattr(flow_lm, "insert_bos_before_voice", False):
        b = flow_lm.bos_before_voice  # nn.Parameter [1, 1, d_model]
        bos = b.detach().clone().to(torch.float32).contiguous()
        if tuple(bos.shape) != (1, 1, d_model):
            raise RuntimeError(
                f"[{language}] bos_before_voice shape {tuple(bos.shape)} "
                f"!= expected [1, 1, {d_model}]"
            )
        tensors["bos_before_voice"] = bos
        LOGGER.info(
            "[%s] bos_before_voice: shape=%s", language, tuple(bos.shape),
        )
    else:
        LOGGER.info(
            "[%s] insert_bos_before_voice=False; no bos_before_voice sidecar entry",
            language,
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "language": language,
        "d_model": str(d_model),
        "ldim": str(ldim),
        "phase": "4B-voice-clone-stage2",
    }
    save_file(tensors, str(save_path), metadata=meta)
    size_kb = save_path.stat().st_size / 1024.0
    LOGGER.info(
        "[%s] wrote speaker_proj sidecar to %s (%.1f KB; keys=%s)",
        language, save_path, size_kb, sorted(tensors.keys()),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m pockettts_coreml.convert.export_speaker_proj",
        description="Export flow_lm.speaker_proj_weight (+ bos_before_voice "
                    "if present) as a Swift-loadable safetensors sidecar.",
    )
    p.add_argument(
        "--language", default="english",
        help="Reference language config (english/spanish/german/italian/"
             "portuguese/french_24l). Selects the matching YAML under "
             "pockettts_coreml/reference/pocket_tts/config/.",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output directory for `speaker_proj.safetensors`. Defaults to "
             "Artifacts/en_alba_fp16/ (the convert-step default).",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)

    out_dir = args.out if args.out is not None else ARTIFACTS_DIR
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        _repo_root = Path(__file__).resolve().parent.parent.parent
        out_dir = (_repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "speaker_proj.safetensors"
    export(save_path, language=args.language)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
