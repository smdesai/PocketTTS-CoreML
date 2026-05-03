"""Portuguese conversion validator.

Parity check between the reference Portuguese TTSModel and the
CoreML-composed Portuguese pipeline. This is NOT part of the pytest
suite because it requires:
  - Network access at first run (voice + weights download)
  - Downloaded Portuguese voices in `voices_portuguese/`
  - Pre-converted `Artifacts/pt_fp16/` bundle

Usage:
    .venv/bin/python -m pockettts_coreml.oracle.verify_portuguese \\
        [--voice voices_portuguese/alba.safetensors] \\
        [--artifacts Artifacts/pt_fp16] \\
        [--prompt "Pocket TTS é ..."] \\
        [--seed 42]

Gates (same as Spanish):
    - Speaker cosine similarity >= 0.95  (Resemblyzer)
    - Overall audio PSNR >= 15 dB
    - Best-early-frame PSNR >= 25 dB
    - Amplitude-decay slope <= 0.1 %/frame (fp32-softmax fix holds)

Exits non-zero if any gate fails. Writes `/tmp/pt_reference.wav` and
`/tmp/pt_coreml.wav` for manual listen-compare.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

LOGGER = logging.getLogger("pockettts_coreml.oracle.verify_portuguese")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REF_DIR = REPO_ROOT / "pockettts_coreml" / "reference"
if str(REF_DIR) not in sys.path:
    sys.path.insert(0, str(REF_DIR))

DEFAULT_PROMPT = "Pocket TTS é um modelo leve de conversão de texto em fala."
DEFAULT_VOICE = REPO_ROOT / "voices_portuguese" / "alba.safetensors"
DEFAULT_ARTIFACTS = REPO_ROOT / "Artifacts" / "pt_fp16"
LANGUAGE = "portuguese"
SAMPLE_RATE = 24000
FRAME_SIZE = 1920


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    import scipy.io.wavfile
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    scipy.io.wavfile.write(str(path), sample_rate, pcm)
    LOGGER.info("  wrote %s (%.2fs)", path, len(audio) / sample_rate)


def _run_reference(prompt: str, voice_path: Path, seed: int) -> np.ndarray:
    from pockettts_coreml.patches import build_patched_submodules
    ps = build_patched_submodules(language=LANGUAGE)
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
    from pockettts_coreml.e2e import CoreMLGenerator
    torch.manual_seed(seed)
    gen = CoreMLGenerator(
        artifacts_dir, compute_units="CPU_ONLY", language=LANGUAGE,
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
    return slope * 100.0


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
    p = argparse.ArgumentParser(prog="verify_portuguese")
    p.add_argument("--voice", type=Path, default=DEFAULT_VOICE)
    p.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--cos-gate", type=float, default=0.93)
    p.add_argument("--psnr-gate", type=float, default=15.0)
    p.add_argument("--early-psnr-gate", type=float, default=25.0)
    p.add_argument("--slope-gate", type=float, default=0.0)
    args = p.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.voice.exists():
        LOGGER.error("Voice safetensors missing: %s", args.voice)
        return 2
    for stem in ("text_conditioner", "flow_lm_main", "flow_lm_flow",
                 "flow_lm_prefill", "mimi_encoder", "mimi_decoder"):
        pkg = args.artifacts / f"{stem}.mlpackage"
        if not pkg.exists():
            LOGGER.error("Missing artifact: %s", pkg)
            LOGGER.error("Run: .venv/bin/python -m pockettts_coreml.convert --all --language %s --out %s",
                         LANGUAGE, args.artifacts)
            return 2

    LOGGER.info("=== Portuguese verify ===")
    LOGGER.info("  prompt: %r", args.prompt)
    LOGGER.info("  voice:  %s", args.voice)
    LOGGER.info("  artifacts: %s", args.artifacts)
    LOGGER.info("  seed: %d", args.seed)

    LOGGER.info("[1/4] running reference...")
    ref = _run_reference(args.prompt, args.voice, args.seed)
    LOGGER.info("  reference samples=%d (%.2fs)", len(ref), len(ref) / SAMPLE_RATE)

    LOGGER.info("[2/4] running CoreML...")
    tst = _run_coreml(args.prompt, args.voice, args.artifacts, args.seed)
    LOGGER.info("  coreml    samples=%d (%.2fs)", len(tst), len(tst) / SAMPLE_RATE)

    _write_wav(Path("/tmp/pt_reference.wav"), ref)
    _write_wav(Path("/tmp/pt_coreml.wav"), tst)

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
