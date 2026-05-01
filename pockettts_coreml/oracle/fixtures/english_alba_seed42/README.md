# Fixture: english_alba_seed42

Per-stage golden outputs for the English PocketTTS model under:

- Prompt: `prompt.txt` (fixed: "Pocket TTS is a lightweight text-to-speech model.")
- Voice: `alba` (see `voice_embedding.safetensors` — must be pre-exported; see below)
- Seed: `42`
- LSD decode steps: `1`
- Temperature: `0.7`

## Files

| Path | Created by | Status |
|------|------------|--------|
| `prompt.txt` | checked in | present |
| `voice_embedding.safetensors` | user (reference `export_voice` CLI) | **missing — generate first** |
| `golden/*.safetensors` | `python -m pockettts_coreml.oracle.dump_golden` | **missing — generate after voice** |
| `golden/output.wav` | `python -m pockettts_coreml.oracle.dump_golden` | **missing — generate after voice** |
| `metadata.json` | `python -m pockettts_coreml.oracle.dump_golden` | **missing — generate after voice** |

## Regeneration recipe

```bash
# 1. Accept the license for the gated weights:
#    https://huggingface.co/kyutai/pocket-tts
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx

# 2. Export the alba voice to safetensors once:
pocket-tts export_voice \
    hf://kyutai/tts-voices/alba-mackenna/casual.wav \
    pockettts_coreml/oracle/fixtures/english_alba_seed42/voice_embedding.safetensors

# 3. Regenerate the golden bundle:
python -m pockettts_coreml.oracle.dump_golden
```

## Why this fixture is committed with placeholders

The `kyutai/pocket-tts` weights are gated behind a HuggingFace license
acceptance; we intentionally do not ship them (or derived tensors) from
this repo. After step 3 above, the generated files can be committed
alongside your work, or left uncommitted if your use case demands that
every contributor regenerate locally (safer for license compliance).
