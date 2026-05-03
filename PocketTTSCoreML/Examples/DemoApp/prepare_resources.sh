#!/usr/bin/env bash
#
# Physically copy model artifacts + voices + tokenizer for every supported
# language into DemoApp/DemoApp/Resources/Languages/<id>/. Unlike
# iOSBenchmark's symlink approach, Xcode folder references for .mlmodelc
# don't always follow symlinks reliably during device builds — so we use
# real copies here. Re-runnable; destination trees are wiped then
# re-populated.
#
# Layout produced:
#
#   Resources/
#   └── Languages/
#       ├── en/
#       │   ├── Artifacts/    (6 .mlmodelc + 2 sidecars)
#       │   ├── Voices/       (21 .safetensors)
#       │   └── tokenizer.model
#       └── es/
#           ├── Artifacts/
#           ├── Voices/
#           └── tokenizer.model
#
# Run from DemoApp/ (this directory). Total payload scales linearly with
# language count (~490 MB per 6-layer language: 356 MB artifacts + ~130 MB
# voices). French is the only 24-layer language currently shipped and
# adds ~1.55 GB on its own (~1.05 GB artifacts + ~500 MB voices).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
RES="$HERE/DemoApp/Resources"

# Required artifact items (names are shared across languages).
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

# Sidecars that travel with the bundle when present but aren't strictly
# required for generation (voice cloning stage 2 only).
optional_artifacts=(
    "speaker_proj.safetensors"
)

# Per-language source paths. Extend this map to add more languages.
#   <id>|<artifacts_src>|<voices_src>|<tokenizer_src>
LANG_SPECS=(
    "en|$REPO/Artifacts/en_alba_fp16_mlmodelc|$REPO/voices|$REPO/pockettts_coreml/oracle/fixtures/english_alba_seed42/tokenizer.model"
    "es|$REPO/Artifacts/es_fp16_mlmodelc|$REPO/voices_spanish|$REPO/voices_spanish/tokenizer.model"
    "de|$REPO/Artifacts/de_fp16_mlmodelc|$REPO/voices_german|$REPO/voices_german/tokenizer.model"
    "it|$REPO/Artifacts/it_fp16_mlmodelc|$REPO/voices_italian|$REPO/voices_italian/tokenizer.model"
    "pt|$REPO/Artifacts/pt_fp16_mlmodelc|$REPO/voices_portuguese|$REPO/voices_portuguese/tokenizer.model"
    "fr|$REPO/Artifacts/fr_fp16_mlmodelc|$REPO/voices_french|$REPO/voices_french/tokenizer.model"
)

# ------------------------------------------------------------------
# 1) Sanity-check every source for every language before touching the
#    destination. Fail fast with a specific message.
# ------------------------------------------------------------------
for spec in "${LANG_SPECS[@]}"; do
    IFS='|' read -r LANG_ID ART_SRC VOICES_SRC TOKENIZER_SRC <<< "$spec"
    echo "checking sources for $LANG_ID ..."
    for item in "${required_artifacts[@]}"; do
        if [[ ! -e "$ART_SRC/$item" ]]; then
            echo "ERROR [$LANG_ID]: missing artifact: $ART_SRC/$item" >&2
            exit 1
        fi
    done
    if [[ ! -d "$VOICES_SRC" ]]; then
        echo "ERROR [$LANG_ID]: missing voices dir: $VOICES_SRC" >&2
        exit 1
    fi
    if [[ ! -f "$TOKENIZER_SRC" ]]; then
        echo "ERROR [$LANG_ID]: missing tokenizer: $TOKENIZER_SRC" >&2
        exit 1
    fi
done

# ------------------------------------------------------------------
# 2) Wipe Resources/Languages entirely and rebuild from sources.
#    Also wipe the legacy flat Artifacts/Voices/tokenizer.model if they
#    exist from a prior single-language layout.
# ------------------------------------------------------------------
echo "seeding $RES ..."
rm -rf "$RES/Languages"
rm -rf "$RES/Artifacts" "$RES/Voices" "$RES/tokenizer.model"
mkdir -p "$RES/Languages"

for spec in "${LANG_SPECS[@]}"; do
    IFS='|' read -r LANG_ID ART_SRC VOICES_SRC TOKENIZER_SRC <<< "$spec"

    LANG_DIR="$RES/Languages/$LANG_ID"
    echo "  [$LANG_ID] populating $LANG_DIR"
    mkdir -p "$LANG_DIR/Artifacts" "$LANG_DIR/Voices"

    # Artifacts + sidecars (6 mlmodelc + 2 sidecars + optional clone sidecar).
    for item in "${required_artifacts[@]}"; do
        cp -R "$ART_SRC/$item" "$LANG_DIR/Artifacts/$item"
    done
    for item in "${optional_artifacts[@]}"; do
        if [[ -e "$ART_SRC/$item" ]]; then
            cp -R "$ART_SRC/$item" "$LANG_DIR/Artifacts/$item"
        else
            echo "  [$LANG_ID] note: optional artifact $item missing — " \
                 "voice cloning will be unavailable for this language"
        fi
    done

    # Tokenizer.
    cp "$TOKENIZER_SRC" "$LANG_DIR/tokenizer.model"

    # Voices.
    voice_count=0
    shopt -s nullglob
    for v in "$VOICES_SRC"/*.safetensors; do
        # Skip any accidentally-placed tokenizer.model or other files;
        # only .safetensors voice embeddings are copied.
        cp "$v" "$LANG_DIR/Voices/$(basename "$v")"
        voice_count=$((voice_count + 1))
    done
    shopt -u nullglob

    if [[ $voice_count -eq 0 ]]; then
        echo "ERROR [$LANG_ID]: no voice .safetensors files in $VOICES_SRC" >&2
        exit 1
    fi
    echo "    -> $voice_count voices copied"
done

size_mb=$(du -sm "$RES" | awk '{print $1}')
echo "done. Resources total: ${size_mb} MB."
echo "next: xcodegen generate  (from $HERE)"
