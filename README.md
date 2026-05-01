# pocketTTS-CoreML

CoreML port of Kyutai's [PocketTTS](https://github.com/kyutai-labs/pocket-tts)
(English) targeting Apple Neural Engine + CPU. See
`thoughts/shared/plans/2026-05-01_pockettts_coreml_english_port.md` for
the full phased plan.

**Status:** Phase 1 (oracle harness) only. No CoreML artifacts yet.

## Regenerate the Phase-1 golden bundle

```bash
# 0. Install (uv recommended; matches the reference repo)
uv sync

# 1. Accept the PocketTTS license on HuggingFace, then:
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx

# 2. Ingest the reference repo as a git subtree (see "Repo setup" below).

# 3. Pre-export the `alba` voice using the reference CLI:
uv run pocket-tts export_voice \
    hf://kyutai/tts-voices/alba-mackenna/casual.wav \
    pockettts_coreml/oracle/fixtures/english_alba_seed42/voice_embedding.safetensors

# 4. Generate per-stage golden outputs + output.wav + metadata.json:
uv run python -m pockettts_coreml.oracle.dump_golden

# 5. Verify determinism (bitwise-identical rerun):
POCKETTTS_ORACLE_READY=1 uv run pytest tests/test_oracle_roundtrip.py
```

## Repo setup

The reference repo lives under `pockettts_coreml/reference/` as a **git
subtree** (not a submodule). The subtree has not been ingested yet; the
user must run the commands below once, after `git init`.
