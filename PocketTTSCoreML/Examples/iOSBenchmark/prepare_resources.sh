#!/usr/bin/env bash
#
# Seed iOSBenchmark/Resources/ with symlinks to the real model artifacts +
# voice + tokenizer. Called once after cloning; re-runnable.
#
# Xcode resolves symlinks when copying bundle resources, so the resulting
# .app bundle contains real files (no dangling symlink risk in the .ipa).
#
# Run from the iOSBenchmark/ directory.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
RES="$HERE/iOSBenchmark/Resources"

# Sanity-check source paths.
ARTIFACTS="$REPO/Artifacts/en_alba_fp16"
VOICE="$REPO/pockettts_coreml/oracle/fixtures/english_alba_seed42/alba.safetensors"
TOKENIZER="$REPO/pockettts_coreml/oracle/fixtures/english_alba_seed42/tokenizer.model"

for required in \
    "$ARTIFACTS/flow_lm_main.mlpackage" \
    "$ARTIFACTS/flow_lm_flow.mlpackage" \
    "$ARTIFACTS/flow_lm_prefill.mlpackage" \
    "$ARTIFACTS/mimi_decoder.mlpackage" \
    "$ARTIFACTS/mimi_encoder.mlpackage" \
    "$ARTIFACTS/text_conditioner.mlpackage" \
    "$ARTIFACTS/mimi_decoder.state_layout.json" \
    "$ARTIFACTS/flow_lm_bos_emb.safetensors" \
    "$VOICE" \
    "$TOKENIZER"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: missing source file: $required" >&2
        exit 1
    fi
done

mkdir -p "$RES/Artifacts"

link() {
    local src="$1" dst="$2"
    ln -sfn "$src" "$dst"
    echo "  linked: $(basename "$dst") -> $src"
}

echo "Seeding $RES ..."
for pkg in flow_lm_main flow_lm_flow flow_lm_prefill mimi_decoder mimi_encoder text_conditioner; do
    link "$ARTIFACTS/${pkg}.mlpackage" "$RES/Artifacts/${pkg}.mlpackage"
done
link "$ARTIFACTS/mimi_decoder.state_layout.json"     "$RES/Artifacts/mimi_decoder.state_layout.json"
link "$ARTIFACTS/flow_lm_bos_emb.safetensors"        "$RES/Artifacts/flow_lm_bos_emb.safetensors"
link "$VOICE"                                         "$RES/alba.safetensors"
link "$TOKENIZER"                                     "$RES/tokenizer.model"

echo "Done. Now run: xcodegen generate (from $HERE)"
