"""Generate per-stage golden outputs from the unmodified reference model.

This is Phase 1's central script. It:

1. Loads the reference TTSModel (English, gated `kyutai/pocket-tts` weights).
2. Loads the pre-exported `alba` voice from `voice_embedding.safetensors`
   (produced out-of-band via the reference's `pocket-tts export_voice` CLI).
3. Runs a single fixed prompt under a fixed seed with `torch.set_num_threads(1)`.
4. Registers `register_forward_hook` callbacks at the 5 submodel boundaries
   (text_conditioner, flow_lm_main, flow_lm_flow, mimi_encoder, mimi_decoder)
   so we capture inputs/outputs of every per-boundary call WITHOUT editing
   the reference. Hook placements are derived from
   `docs/research/reference_codebase_map.md`:

     - text_conditioner   → LUTConditioner          (conditioners/text.py)
     - flow_lm_main       → StreamingTransformer    (modules/mimi_transformer.py)
                            inside flow_lm.backbone (flow_lm.py:95)
     - flow_lm_flow       → SimpleMLPAdaLN          (mlp.py:188)
     - mimi_encoder       → MimiModel.encode_to_latent path
                            (mimi.py:96; only hit once, during voice prep)
     - mimi_decoder       → MimiModel.decode_from_latent path
                            (mimi.py:89; per-audio-frame)

5. Writes the captured bundles to
   `fixtures/english_alba_seed42/golden/<stage>_step_<i>.safetensors`.
6. Writes `metadata.json` with torch version / seed / lsd steps /
   tokenizer hash so later phases can detect drift.
7. Also writes the final PCM waveform to `golden/output.wav` for manual
   listen-test.

HF_TOKEN handling: if the env var is absent we abort with a helpful
message BEFORE pulling weights. Never embed a token, never attempt
anonymous download.

Usage:
    export HF_TOKEN=hf_...
    python -m pockettts_coreml.oracle.dump_golden \
        --voice-safetensors path/to/alba.safetensors \
        [--output-dir fixtures/english_alba_seed42]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from pockettts_coreml.oracle.compare import save_bundle

LOGGER = logging.getLogger("pockettts_coreml.oracle.dump_golden")

# Fixed fixture constants. The plan pins these in §Phase-1 (prompt,
# voice, seed). Changing any of them invalidates the golden bundle.
FIXTURE_PROMPT = "Pocket TTS is a lightweight text-to-speech model."
FIXTURE_VOICE_NAME = "alba"
FIXTURE_SEED = 42
FIXTURE_LSD_DECODE_STEPS = 1  # reference default; see default_parameters.py:4
FIXTURE_TEMPERATURE = 0.7  # reference default; see default_parameters.py:3
FIXTURE_EOS_THRESHOLD = -4.0  # reference default; see default_parameters.py:6
FIXTURE_NOISE_CLAMP = None  # reference default

GATED_REPO = "kyutai/pocket-tts"
GATED_LICENSE_URL = f"https://huggingface.co/{GATED_REPO}"


@dataclass
class HookBundle:
    """Accumulated per-stage tensors captured via forward hooks."""

    # `per_step` is a list of (step_index, key->tensor) dicts. For stages
    # that run once (text_conditioner, mimi_encoder) it has length 1.
    per_step: list[dict[str, torch.Tensor]] = field(default_factory=list)

    def record(self, tensors: dict[str, torch.Tensor]) -> None:
        self.per_step.append({k: v.detach().cpu().clone() for k, v in tensors.items()})


def _abort(msg: str, code: int = 2) -> None:
    LOGGER.error(msg)
    print(msg, file=sys.stderr)
    sys.exit(code)


def _check_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        _abort(
            "ERROR: HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is not set.\n"
            f"The English PocketTTS weights live behind a gated repo at\n"
            f"  {GATED_LICENSE_URL}\n"
            "You must (a) accept the license on that page while logged in\n"
            "to HuggingFace and (b) provide a token via:\n"
            "  export HF_TOKEN=hf_xxxxxxxxxxxxxxxx\n"
            "See `README.md` for the full regeneration recipe."
        )
    return token  # type: ignore[return-value]


def _set_determinism(seed: int) -> None:
    """Match the reference's own threading + seed discipline.

    pocket_tts calls `torch.set_num_threads(1)` at import time (see the
    reference's `pocket_tts/__init__.py`). We replicate that here AND set
    the torch RNG so noise sampling inside FlowLMModel.forward
    (`flow_lm.py:133-137`) is reproducible.
    """
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    # `use_deterministic_algorithms` will raise if a non-deterministic op
    # appears. We pass `warn_only=True` so the script still completes
    # under the unlikely case that some op in the reference flips on a
    # non-deterministic backend — a clean error from torch here is a
    # drift-detection signal.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older torch without `warn_only` kwarg: best effort.
        torch.use_deterministic_algorithms(True)
    # CPU-only: we never want accidental CUDA/MPS drift.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _install_hooks(tts_model: Any) -> dict[str, HookBundle]:
    """Register forward hooks on the 5 submodel boundaries.

    Returns a dict of stage_name -> HookBundle. Callers MUST remove the
    hooks after use (we return the handles via `tts_model._hook_handles`
    for convenience; see `_uninstall_hooks`).
    """
    # Late import so `import pockettts_coreml` (without the reference
    # subtree) still succeeds. Phase 1's package-importability gate is
    # purely `import pockettts_coreml`; dump_golden is only callable once
    # the subtree lands.
    from pockettts_coreml.reference.pocket_tts.conditioners.text import LUTConditioner
    from pockettts_coreml.reference.pocket_tts.models.mimi import MimiModel
    from pockettts_coreml.reference.pocket_tts.modules.mimi_transformer import (
        StreamingTransformer,
    )
    from pockettts_coreml.reference.pocket_tts.modules.mlp import SimpleMLPAdaLN

    bundles: dict[str, HookBundle] = {
        "text_conditioner": HookBundle(),
        "flow_lm_main": HookBundle(),
        "flow_lm_flow": HookBundle(),
        "mimi_encoder": HookBundle(),
        "mimi_decoder": HookBundle(),
    }
    handles: list[torch.utils.hooks.RemovableHandle] = []

    # --- text_conditioner ---------------------------------------------
    # LUTConditioner.forward returns a float tensor [1, S_text, 1024].
    # We hook on the LUT conditioner instance attached to the flow_lm
    # sub-model (FlowLMModel holds a `.conditioner`).
    def _tc_hook(_mod, inputs, output):
        # inputs is a tuple; the first positional is the TokenizedText
        # (namedtuple). The actual int64 token tensor is `.tokens`.
        token_tensor = getattr(inputs[0], "tokens", None)
        if token_tensor is None and isinstance(inputs[0], torch.Tensor):
            token_tensor = inputs[0]
        rec = {"embeddings_out": output.detach().cpu().clone()}
        if token_tensor is not None:
            rec["tokens_in"] = token_tensor.detach().cpu().clone()
        bundles["text_conditioner"].record(rec)

    for m in tts_model.flow_lm.modules():
        if isinstance(m, LUTConditioner):
            handles.append(m.register_forward_hook(_tc_hook))

    # --- flow_lm_main -------------------------------------------------
    # StreamingTransformer.forward(x, model_state) -> x_out. This runs
    # once per AR step (with T=1 at the sequence dim) AND once per
    # prefill call (with T=S_text or T=S_audio_cond). We capture all of
    # them indiscriminately; the bundle metadata records the call index.
    def _flow_main_hook(_mod, inputs, output):
        x_in = inputs[0]
        rec = {
            "x_in": x_in.detach().cpu().clone(),
            "x_out": output.detach().cpu().clone(),
        }
        bundles["flow_lm_main"].record(rec)

    for m in tts_model.flow_lm.modules():
        if isinstance(m, StreamingTransformer):
            handles.append(m.register_forward_hook(_flow_main_hook))

    # --- flow_lm_flow -------------------------------------------------
    # SimpleMLPAdaLN.forward(c, s, t, x) -> u. Called once per LSD step
    # per AR step; at lsd_decode_steps=1 it's once per AR step.
    def _flow_flow_hook(_mod, inputs, output):
        c, s, t, x_noise = inputs
        rec = {
            "c_in": c.detach().cpu().clone(),
            "s_in": s.detach().cpu().clone(),
            "t_in": t.detach().cpu().clone(),
            "x_in": x_noise.detach().cpu().clone(),
            "u_out": output.detach().cpu().clone(),
        }
        bundles["flow_lm_flow"].record(rec)

    for m in tts_model.flow_lm.modules():
        if isinstance(m, SimpleMLPAdaLN):
            handles.append(m.register_forward_hook(_flow_flow_hook))

    # --- mimi_encoder / mimi_decoder ----------------------------------
    # The `MimiModel.encode_to_latent` and `.decode_from_latent` are
    # plain methods, not `nn.Module.forward`, so we can't use a single
    # hook on the MimiModel. Instead we monkey-wrap those two methods
    # on the *instance only* (not the class), preserving the rest of
    # the reference untouched.
    mimi: MimiModel = tts_model.mimi

    if hasattr(mimi, "encode_to_latent"):
        orig_encode = mimi.encode_to_latent

        def _wrapped_encode(x, *args, **kwargs):
            out = orig_encode(x, *args, **kwargs)
            bundles["mimi_encoder"].record({
                "waveform_in": x.detach().cpu().clone(),
                "latent_out": out.detach().cpu().clone(),
            })
            return out

        mimi.encode_to_latent = _wrapped_encode  # type: ignore[assignment]
        # Stash for removal.
        handles.append(_MethodRestore(mimi, "encode_to_latent", orig_encode))

    if hasattr(mimi, "decode_from_latent"):
        orig_decode = mimi.decode_from_latent

        def _wrapped_decode(latent, state, *args, **kwargs):
            out = orig_decode(latent, state, *args, **kwargs)
            bundles["mimi_decoder"].record({
                "latent_in": latent.detach().cpu().clone(),
                "audio_out": out.detach().cpu().clone(),
            })
            return out

        mimi.decode_from_latent = _wrapped_decode  # type: ignore[assignment]
        handles.append(_MethodRestore(mimi, "decode_from_latent", orig_decode))

    tts_model._hook_handles = handles  # type: ignore[attr-defined]
    return bundles


class _MethodRestore:
    """Tiny shim that mimics torch hook handle API for method wrapping."""

    def __init__(self, obj: Any, attr: str, original: Any):
        self._obj = obj
        self._attr = attr
        self._original = original
        self._removed = False

    def remove(self) -> None:
        if not self._removed:
            setattr(self._obj, self._attr, self._original)
            self._removed = True


def _uninstall_hooks(tts_model: Any) -> None:
    for h in getattr(tts_model, "_hook_handles", []):
        try:
            h.remove()
        except Exception:
            pass
    tts_model._hook_handles = []


def _bundle_to_flat_dict(bundle: HookBundle) -> dict[str, torch.Tensor]:
    """Flatten per-step records into a single safetensors-compatible dict."""
    flat: dict[str, torch.Tensor] = {}
    for i, rec in enumerate(bundle.per_step):
        for key, val in rec.items():
            flat[f"step_{i:04d}/{key}"] = val
    return flat


def _write_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    import scipy.io.wavfile  # lazy: scipy is in pyproject deps

    pcm = (audio.clamp(-1.0, 1.0) * 32767).to(torch.int16).numpy()
    scipy.io.wavfile.write(str(path), sample_rate, pcm)


def _tokenizer_hash(tts_model: Any) -> str | None:
    """Best-effort: hash the SentencePieceProcessor model bytes, if we can
    get at them. Stored in metadata.json for drift detection.
    """
    try:
        tok = tts_model.flow_lm.conditioner.tokenizer
        sp = getattr(tok, "sp", None) or getattr(tok, "_sp", None)
        if sp is None:
            return None
        # sentencepiece doesn't expose the raw bytes trivially; serialize.
        raw = sp.serialized_model_proto()
        return hashlib.sha256(raw).hexdigest()
    except Exception:
        return None


def dump_golden(
    output_dir: Path,
    voice_safetensors: Path,
    prompt: str = FIXTURE_PROMPT,
    seed: int = FIXTURE_SEED,
    lsd_decode_steps: int = FIXTURE_LSD_DECODE_STEPS,
) -> None:
    """End-to-end driver. See module docstring."""
    _check_hf_token()
    _set_determinism(seed)

    # Late import. The reference subtree ships its own `torch.set_num_threads(1)`
    # on import, which harmlessly duplicates our call above.
    try:
        from pockettts_coreml.reference.pocket_tts.models.tts_model import TTSModel
    except ImportError as exc:
        _abort(
            "ERROR: cannot import the reference TTSModel. The git subtree under\n"
            "  pockettts_coreml/reference/\n"
            "is probably missing. Follow README.md step 'Ingest the reference repo'.\n"
            f"Underlying ImportError: {exc}"
        )

    LOGGER.info("Loading reference TTSModel (English)...")
    tts_model = TTSModel.load_model(
        language="english",
        temp=FIXTURE_TEMPERATURE,
        lsd_decode_steps=lsd_decode_steps,
        noise_clamp=FIXTURE_NOISE_CLAMP,
        eos_threshold=FIXTURE_EOS_THRESHOLD,
        quantize=False,
    )
    tts_model.eval()

    # Install forward hooks BEFORE voice loading so we capture the
    # mimi_encoder call that happens inside get_state_for_audio_prompt.
    # When the voice is loaded from a `.safetensors`, _import_model_state
    # bypasses the encoder — that's fine; mimi_encoder bundle will just
    # be empty and we note that in metadata.
    bundles = _install_hooks(tts_model)

    try:
        # Re-seed immediately before the hot path so noise sampling
        # (flow_lm.py:133-137) is reproducible independent of any torch
        # RNG draws during model load.
        torch.manual_seed(seed)

        LOGGER.info("Loading voice embedding from %s ...", voice_safetensors)
        model_state = tts_model.get_state_for_audio_prompt(
            audio_conditioning=str(voice_safetensors),
        )

        LOGGER.info("Generating audio for fixture prompt...")
        torch.manual_seed(seed)  # re-seed for the AR loop itself
        audio = tts_model.generate_audio(
            model_state=model_state,
            text_to_generate=prompt,
            copy_state=True,
        )
        LOGGER.info("Generated %d audio samples.", audio.shape[-1])
    finally:
        _uninstall_hooks(tts_model)

    # --- dump bundles to disk ----------------------------------------
    golden_dir = output_dir / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)

    for stage, bundle in bundles.items():
        flat = _bundle_to_flat_dict(bundle)
        if not flat:
            LOGGER.info("Stage %s produced no records (expected if voice was "
                        "loaded from safetensors; mimi_encoder is skipped).",
                        stage)
            continue
        out_path = golden_dir / f"{stage}.safetensors"
        save_bundle(flat, out_path)
        LOGGER.info("Wrote %s (%d records, %d tensors)",
                    out_path, len(bundle.per_step), len(flat))

    # Write the PCM wav for manual listen-test.
    sample_rate = int(tts_model.config.mimi.sample_rate)
    _write_wav(golden_dir / "output.wav", audio, sample_rate)

    # Metadata sidecar.
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "voice_name": FIXTURE_VOICE_NAME,
        "voice_safetensors": str(voice_safetensors),
        "seed": seed,
        "lsd_decode_steps": lsd_decode_steps,
        "temperature": FIXTURE_TEMPERATURE,
        "eos_threshold": FIXTURE_EOS_THRESHOLD,
        "noise_clamp": FIXTURE_NOISE_CLAMP,
        "torch_version": torch.__version__,
        "sample_rate": sample_rate,
        "audio_samples": int(audio.shape[-1]),
        "tokenizer_sha256": _tokenizer_hash(tts_model),
        "stage_record_counts": {
            name: len(b.per_step) for name, b in bundles.items()
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    LOGGER.info("Wrote metadata.json")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pockettts_coreml.oracle.dump_golden",
        description=(
            "Regenerate the per-stage golden bundle for the English PocketTTS "
            "port. Requires HF_TOKEN in env and a pre-exported voice .safetensors. "
            "Outputs go under --output-dir (default: the fixture path shipped with "
            "this repo)."
        ),
    )
    default_out = Path(__file__).parent / "fixtures" / "english_alba_seed42"
    default_voice = default_out / "voice_embedding.safetensors"
    p.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help=f"Fixture directory (default: {default_out}).",
    )
    p.add_argument(
        "--voice-safetensors",
        type=Path,
        default=default_voice,
        help=(
            "Path to a pre-exported voice .safetensors (produced by the "
            f"reference's `pocket-tts export_voice` CLI). Default: {default_voice}."
        ),
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=FIXTURE_PROMPT,
        help=f"Prompt to generate (default pinned: {FIXTURE_PROMPT!r}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=FIXTURE_SEED,
        help=f"torch.manual_seed value (default: {FIXTURE_SEED}).",
    )
    p.add_argument(
        "--lsd-decode-steps",
        type=int,
        default=FIXTURE_LSD_DECODE_STEPS,
        help=f"LSD decode steps (default: {FIXTURE_LSD_DECODE_STEPS}).",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.voice_safetensors.exists():
        _abort(
            f"ERROR: voice safetensors not found at {args.voice_safetensors}.\n"
            "Export it with the reference CLI (after HF_TOKEN is set and the\n"
            f"license at {GATED_LICENSE_URL} is accepted):\n"
            "  pocket-tts export_voice hf://kyutai/tts-voices/alba-mackenna/casual.wav "
            f"{args.voice_safetensors}\n"
        )

    dump_golden(
        output_dir=args.output_dir,
        voice_safetensors=args.voice_safetensors,
        prompt=args.prompt,
        seed=args.seed,
        lsd_decode_steps=args.lsd_decode_steps,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
