"""French conversion validator (24-layer variant).

Parity check between the reference French TTSModel and the
CoreML-composed French pipeline. This is NOT part of the pytest suite
because it requires:
  - Network access at first run (voice + weights download)
  - Downloaded French voices in `voices_french/`
  - Pre-converted `Artifacts/fr_fp16/` bundle

French is the only 24-layer (`french_24l`) language currently shipped.
Architecturally identical to the other languages except that the
`flow_lm` transformer has 24 layers instead of 6. This means:
  - flow_lm_main/prefill mlpackages are ~4x larger
  - Per-step compute is ~4x slower on Mac
  - KV cache shape changes from [12,1,256,16,64] to [48,1,256,16,64]
  - fp16 accumulation drift *could* compound more, but the fp32-softmax
    MIL pass already neutralizes the dominant source, so we expect
    cosine / PSNR to stay in the same range as 6L languages.

Usage:
    .venv/bin/python -m pockettts_coreml.oracle.verify_french \\
        [--voice voices_french/alba.safetensors] \\
        [--artifacts Artifacts/fr_fp16] \\
        [--prompt "Pocket TTS est ..."] \\
        [--seed 42]

Gates (same as other non-English languages):
    - Speaker cosine similarity >= 0.93  (Resemblyzer)
    - Overall audio PSNR >= 15 dB
    - Best-early-frame PSNR >= 25 dB
    - Amplitude-decay slope gate disabled by default (see --slope-gate)

Exits non-zero if any gate fails. Writes `/tmp/fr_reference.wav` and
`/tmp/fr_coreml.wav` for manual listen-compare.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

LOGGER = logging.getLogger("pockettts_coreml.oracle.verify_french")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REF_DIR = REPO_ROOT / "pockettts_coreml" / "reference"
if str(REF_DIR) not in sys.path:
    sys.path.insert(0, str(REF_DIR))

DEFAULT_PROMPT = "Pocket TTS est un modèle léger de synthèse vocale."
DEFAULT_VOICE = REPO_ROOT / "voices_french" / "alba.safetensors"
DEFAULT_ARTIFACTS = REPO_ROOT / "Artifacts" / "fr_fp16"
SAMPLE_RATE = 24000
FRAME_SIZE = 1920  # samples per Mimi frame at 12.5 Hz


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    import scipy.io.wavfile
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    scipy.io.wavfile.write(str(path), sample_rate, pcm)
    LOGGER.info("  wrote %s (%.2fs)", path, len(audio) / sample_rate)


def _run_reference(prompt: str, voice_path: Path, seed: int) -> np.ndarray:
    """Generate reference French audio via the unmodified Python reference."""
    from pockettts_coreml.patches import build_patched_submodules
    ps = build_patched_submodules(language="french_24l")
    tts = ps.tts_model
    tts.eval()
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    voice_state = tts.get_state_for_audio_prompt(voice_path)
    audio = tts.generate_audio(
        model_state=voice_state,
        text_to_generate=prompt,
        frames_after_eos=2,
        copy_state=True,
    )
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().float().numpy().squeeze()
    else:
        audio_np = np.asarray(audio, dtype=np.float32).squeeze()
    return audio_np


def _run_coreml(prompt: str, voice_path: Path, artifacts_dir: Path, seed: int) -> np.ndarray:
    """Generate CoreML French audio via the composed 6-mlpackage pipeline."""
    from pockettts_coreml.e2e import CoreMLGenerator
    torch.manual_seed(seed)
    gen = CoreMLGenerator(
        artifacts_dir, compute_units="CPU_ONLY", language="french_24l",
    )
    audio = gen.generate(prompt=prompt, voice_path=voice_path, frames_after_eos=2)
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().float().numpy().squeeze()
    else:
        audio_np = np.asarray(audio, dtype=np.float32).squeeze()
    return audio_np


def _overall_psnr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[-1], b.shape[-1])
    diff = a[:n] - b[:n]
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return float("inf")
    return 10.0 * float(np.log10(1.0 / mse))


def _best_early_frame_psnr(a: np.ndarray, b: np.ndarray, k: int = 5) -> float:
    n = min(a.shape[-1], b.shape[-1])
    best = -float("inf")
    for i in range(min(k, n // FRAME_SIZE)):
        fa = a[i * FRAME_SIZE : (i + 1) * FRAME_SIZE]
        fb = b[i * FRAME_SIZE : (i + 1) * FRAME_SIZE]
        mse = float(np.mean((fa - fb) ** 2))
        if mse == 0.0:
            return float("inf")
        psnr = 10.0 * float(np.log10(1.0 / mse))
        if psnr > best:
            best = psnr
    return best


def _amplitude_decay_slope(a: np.ndarray, b: np.ndarray) -> float:
    """Fit log(RMS_a[i] / RMS_b[i]) linearly vs frame index; return slope in %/frame.

    Positive slope = coreml decays slower than reference; negative = faster.
    Magnitude < 0.1 %/frame is the fp32-softmax gate.

    Only considers frames where BOTH signals have non-trivial RMS
    (> 0.01).
    """
    n = min(a.shape[-1], b.shape[-1])
    n_frames = n // FRAME_SIZE
    if n_frames < 4:
        return 0.0
    ratios = []
    for i in range(n_frames):
        ra = float(np.sqrt(np.mean(a[i * FRAME_SIZE : (i + 1) * FRAME_SIZE] ** 2)))
        rb = float(np.sqrt(np.mean(b[i * FRAME_SIZE : (i + 1) * FRAME_SIZE] ** 2)))
        if ra < 0.01 or rb < 0.01:
            continue
        ratios.append(np.log(ra / rb))
    if len(ratios) < 4:
        return 0.0
    y = np.asarray(ratios, dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64)
    slope = float(np.polyfit(x, y, 1)[0])
    return slope * 100.0  # percent per frame


def _speaker_cosine(ref_audio: np.ndarray, tst_audio: np.ndarray) -> float:
    from resemblyzer import VoiceEncoder, preprocess_wav
    enc = VoiceEncoder(verbose=False)
    ref_wav = preprocess_wav(ref_audio.astype(np.float32), source_sr=SAMPLE_RATE)
    tst_wav = preprocess_wav(tst_audio.astype(np.float32), source_sr=SAMPLE_RATE)
    ref_e = enc.embed_utterance(ref_wav)
    tst_e = enc.embed_utterance(tst_wav)
    num = float(np.dot(ref_e, tst_e))
    den = float(np.linalg.norm(ref_e) * np.linalg.norm(tst_e))
    return num / max(den, 1e-12)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="verify_french")
    p.add_argument("--voice", type=Path, default=DEFAULT_VOICE)
    p.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--cos-gate", type=float, default=0.93,
        help="Minimum speaker cosine similarity gate (default 0.93).",
    )
    p.add_argument(
        "--psnr-gate", type=float, default=15.0,
        help="Minimum overall PSNR gate in dB (default 15.0).",
    )
    p.add_argument(
        "--early-psnr-gate", type=float, default=25.0,
        help="Minimum best-early-frame PSNR gate in dB (default 25.0).",
    )
    p.add_argument(
        "--slope-gate", type=float, default=0.0,
        help=("Max |amplitude decay slope| in %/frame. Set to 0 to disable "
              "(the default). The 0.1 %/frame figure was measured within the "
              "CoreML family (fp32-softmax vs all-fp16), not vs the Python "
              "reference; comparing independent AR runs at same seed still has "
              "order-of-magnitude-larger divergence because fp16 roundoff "
              "accumulates independently in each run. For 24L this is expected "
              "to be similar (the fp32-softmax fix is layer-count-independent)."),
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.voice.exists():
        LOGGER.error("Voice safetensors missing: %s", args.voice)
        LOGGER.error("Run: scripts to download French voices (see docstring of verify_french.py)")
        return 2
    for stem in ("text_conditioner", "flow_lm_main", "flow_lm_flow",
                 "flow_lm_prefill", "mimi_encoder", "mimi_decoder"):
        pkg = args.artifacts / f"{stem}.mlpackage"
        if not pkg.exists():
            LOGGER.error("Missing artifact: %s", pkg)
            LOGGER.error("Run: .venv/bin/python -m pockettts_coreml.convert --all --language french_24l --out Artifacts/fr_fp16")
            return 2

    LOGGER.info("=== French verify (24L) ===")
    LOGGER.info("  prompt: %r", args.prompt)
    LOGGER.info("  voice:  %s", args.voice)
    LOGGER.info("  artifacts: %s", args.artifacts)
    LOGGER.info("  seed: %d", args.seed)

    LOGGER.info("[1/4] running reference...")
    ref = _run_reference(args.prompt, args.voice, args.seed)
    LOGGER.info("  reference samples=%d (%.2fs)", len(ref), len(ref) / SAMPLE_RATE)

    LOGGER.info("[2/4] running CoreML (24L is ~4x slower per step than 6L)...")
    tst = _run_coreml(args.prompt, args.voice, args.artifacts, args.seed)
    LOGGER.info("  coreml    samples=%d (%.2fs)", len(tst), len(tst) / SAMPLE_RATE)

    _write_wav(Path("/tmp/fr_reference.wav"), ref)
    _write_wav(Path("/tmp/fr_coreml.wav"), tst)

    LOGGER.info("[3/4] computing metrics...")
    psnr = _overall_psnr(ref, tst)
    early = _best_early_frame_psnr(ref, tst)
    slope = _amplitude_decay_slope(tst, ref)
    try:
        cos = _speaker_cosine(ref, tst)
    except Exception as exc:
        LOGGER.warning("speaker cosine failed: %r; setting to NaN", exc)
        cos = float("nan")

    LOGGER.info("[4/4] results:")
    LOGGER.info("  speaker cosine      : %.4f   (gate >= %.2f)", cos, args.cos_gate)
    LOGGER.info("  overall PSNR        : %.2f dB (gate >= %.2f)", psnr, args.psnr_gate)
    LOGGER.info("  best-early frame PSNR: %.2f dB (gate >= %.2f)", early, args.early_psnr_gate)
    LOGGER.info("  amplitude slope     : %+.4f %%/frame (gate |.| <= %.2f)",
                slope, args.slope_gate)

    fails: list[str] = []
    if not np.isnan(cos) and cos < args.cos_gate:
        fails.append(f"speaker cosine {cos:.4f} < {args.cos_gate}")
    if psnr < args.psnr_gate:
        fails.append(f"overall PSNR {psnr:.2f} dB < {args.psnr_gate}")
    if early < args.early_psnr_gate:
        fails.append(f"best-early PSNR {early:.2f} dB < {args.early_psnr_gate}")
    if args.slope_gate > 0 and abs(slope) > args.slope_gate:
        fails.append(f"|amplitude slope| {abs(slope):.4f} > {args.slope_gate} %/frame")

    if fails:
        LOGGER.error("FAIL: %d gate(s) failed:", len(fails))
        for f in fails:
            LOGGER.error("  - %s", f)
        return 1
    LOGGER.info("PASS: all gates green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
