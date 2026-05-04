# Test fixtures for PocketTTSCoreML

The blocking integration fixtures live outside the test bundle at the
repository root:

- `../../pockettts_coreml/oracle/fixtures/english_alba_seed42/tokenizer.model`
- `../../pockettts_coreml/oracle/fixtures/english_alba_seed42/alba.safetensors`
- `../../pockettts_coreml/oracle/fixtures/english_alba_seed42/golden/output.wav`
- `../../Artifacts/en_fp16/*.mlpackage`
- `../../Artifacts/en_fp16/mimi_decoder.state_layout.json`

Tests locate them via the `POCKETTTS_REPO_ROOT` environment variable,
falling back to the compiled `#file` path. This avoids embedding gated
weights into the test target.

Prefilled bundle used by the end-to-end test:

    python -m pockettts_coreml.e2e.export_full_prefill \
        --voice pockettts_coreml/oracle/fixtures/english_alba_seed42/alba.safetensors \
        --prompt "Pocket TTS is a lightweight text-to-speech model." \
        --out pockettts_coreml/oracle/fixtures/english_alba_seed42/alba_prefilled.safetensors
