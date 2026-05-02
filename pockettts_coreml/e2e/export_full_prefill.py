"""Export a voice + text prefill bundle for the Swift runtime.

Phase 4A limitation: `flow_lm_main.mlpackage` only exposes the AR
per-step path — it does not accept `text_embeddings` as input, so the
text-prefill step cannot be run inside CoreML with the currently-exported
bundle. Until a dedicated prefill `.mlpackage` is added (Phase 4B), the
Swift runtime loads a pre-prefilled KV cache produced by this helper.

Usage:
    python -m pockettts_coreml.e2e.export_full_prefill \\
        --voice alba.safetensors \\
        --prompt "Pocket TTS is a lightweight text-to-speech model." \\
        --out alba_prefilled.safetensors \\
        --seed 42

The output safetensors contains:
  - `flow_kv_rank5`: fp32 [12, 1, 256, 16, 64] — the rank-5 KV cache after
     the voice + text prefill, zero-padded to s_cap=256.
  - `flow_offset`:   int64 [1] — absolute write position after prefill.
  - `bos_emb`:       fp32 [32] — the sequence latent for AR step 0.
  - `prompt_utf8`:   uint8 [N] — the prompt text (for debugging).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_REF_DIR = _REPO_ROOT / "pockettts_coreml" / "reference"
if str(_REF_DIR) not in sys.path:
    sys.path.insert(0, str(_REF_DIR))


def export_prefill(voice_path: Path, prompt: str, out_path: Path, seed: int = 42) -> None:
    from pockettts_coreml.patches import build_patched_submodules

    S_CAP = 256
    L, H, D = 6, 16, 64

    ps = build_patched_submodules()
    tts = ps.tts_model
    tts.eval()

    prepared = tts.flow_lm.conditioner.prepare(prompt)
    text_tokens = prepared.tokens  # [1, S_text]

    voice_state = tts.get_state_for_audio_prompt(str(voice_path))
    tts._expand_kv_cache(voice_state, sequence_length=S_CAP)

    torch.manual_seed(seed)

    with torch.no_grad():
        tts._run_flow_lm_and_increment_step(
            model_state=voice_state, text_tokens=text_tokens,
        )

    kv_rank5 = torch.zeros((2 * L, 1, S_CAP, H, D), dtype=torch.float32)
    offsets = []
    for layer in range(L):
        name = f"transformer.layers.{layer}.self_attn"
        cache = voice_state[name]["cache"]
        off = int(voice_state[name]["offset"].item())
        offsets.append(off)
        k = cache[0].clone()
        v = cache[1].clone()
        k[torch.isnan(k)] = 0.0
        v[torch.isnan(v)] = 0.0
        if off > 0:
            kv_rank5[2 * layer, :, :off] = k[:, :off]
            kv_rank5[2 * layer + 1, :, :off] = v[:, :off]
    assert len(set(offsets)) == 1, f"layer offset mismatch: {offsets}"
    offset = offsets[0]

    bos_emb = tts.flow_lm.bos_emb.detach().clone().view(32).contiguous().float()
    prompt_bytes = torch.tensor(list(prompt.encode("utf-8")), dtype=torch.uint8)

    # Sample noise sequence matching the reference AR loop RNG trajectory.
    # Reseed to match generator.py: the oracle reseeds after prefill before
    # drawing the first per-step noise.
    torch.manual_seed(seed)
    TEMP = 0.7
    MAX_STEPS = 64  # ≥ observed frame count (~39 for canonical prompt)
    noise_seq = torch.empty((MAX_STEPS, 32), dtype=torch.float32)
    for i in range(MAX_STEPS):
        n = torch.empty((1, 32), dtype=torch.float32)
        torch.nn.init.normal_(n, mean=0.0, std=TEMP ** 0.5)
        noise_seq[i] = n[0]

    save_file(
        {
            "flow_kv_rank5": kv_rank5,
            "flow_offset": torch.tensor([offset], dtype=torch.int64),
            "bos_emb": bos_emb,
            "prompt_utf8": prompt_bytes,
            "noise_seq": noise_seq,
        },
        str(out_path),
        metadata={
            "phase": "4A-prefill",
            "s_cap": str(S_CAP),
            "prompt": prompt,
            "seed": str(seed),
        },
    )
    print(f"Wrote {out_path} | offset={offset} | text_tokens={int(text_tokens.shape[-1])}")


def _main():
    p = argparse.ArgumentParser()
    p.add_argument("--voice", type=Path, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    export_prefill(args.voice, args.prompt, args.out, args.seed)


if __name__ == "__main__":
    _main()
