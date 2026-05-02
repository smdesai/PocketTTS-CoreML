#!/usr/bin/env bash
#
# Physically copy model artifacts + 21 voices + tokenizer into
# DemoApp/DemoApp/Resources/. Unlike iOSBenchmark's symlink approach,
# Xcode folder references for .mlmodelc don't always follow symlinks
# reliably during device builds — so we use real copies here. Re-runnable;
# destination trees are wiped then re-populated.
#
# Run from DemoApp/ (this directory). Total payload is ~490 MB.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
RES="$HERE/DemoApp/Resources"

ARTIFACTS_SRC="$REPO/Artifacts/en_alba_fp16_mlmodelc"
VOICES_SRC="$REPO/voices"
TOKENIZER_SRC="$REPO/pockettts_coreml/oracle/fixtures/english_alba_seed42/tokenizer.model"

# Sanity-check all sources before touching anything.
required_artifacts=(
    "flow_lm_main.mlmodelc"
    "flow_lm_flow.mlmodelc"
    "flow_lm_prefill.mlmodelc"
    "mimi_decoder.mlmodelc"
    "mimi_encoder.mlmodelc"
    "text_conditioner.mlmodelc"
    "mimi_decoder.state_layout.json"
    "flow_lm_bos_emb.safetensors"
)
for item in "${required_artifacts[@]}"; do
    if [[ ! -e "$ARTIFACTS_SRC/$item" ]]; then
        echo "ERROR: missing artifact: $ARTIFACTS_SRC/$item" >&2
        exit 1
    fi
done
if [[ ! -d "$VOICES_SRC" ]]; then
    echo "ERROR: missing voices dir: $VOICES_SRC" >&2
    exit 1
fi
if [[ ! -f "$TOKENIZER_SRC" ]]; then
    echo "ERROR: missing tokenizer: $TOKENIZER_SRC" >&2
    exit 1
fi

echo "Seeding $RES ..."
# Wipe and rebuild the resources dirs so stale bundles don't leak in.
rm -rf "$RES/Artifacts" "$RES/Voices"
mkdir -p "$RES/Artifacts" "$RES/Voices"

echo "  copying 6 .mlmodelc bundles + sidecars..."
for item in "${required_artifacts[@]}"; do
    cp -R "$ARTIFACTS_SRC/$item" "$RES/Artifacts/$item"
done

echo "  copying tokenizer.model..."
cp "$TOKENIZER_SRC" "$RES/tokenizer.model"

echo "  copying voice safetensors..."
voice_count=0
shopt -s nullglob
for v in "$VOICES_SRC"/*.safetensors; do
    cp "$v" "$RES/Voices/$(basename "$v")"
    voice_count=$((voice_count + 1))
done
shopt -u nullglob

if [[ $voice_count -eq 0 ]]; then
    echo "ERROR: no voice .safetensors files found in $VOICES_SRC" >&2
    exit 1
fi

size_mb=$(du -sm "$RES" | awk '{print $1}')
echo "Done. Copied $voice_count voices. Resources total: ${size_mb} MB."
echo "Next: xcodegen generate  (from $HERE)"
