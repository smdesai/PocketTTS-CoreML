#!/usr/bin/env bash
#
# download_voices.sh — populate voices_<lang>/ from kyutai/pocket-tts.
#
# The voices_<lang>/ directories are gitignored and hold the per-language
# voice embeddings + SentencePiece tokenizer that the Python reference
# (verify_<lang>.py, dump_golden.py, etc.) and prepare_resources.sh read
# when building the iOS demo bundle. The repo on the Hugging Face side is
# `kyutai/pocket-tts` and its layout is:
#
#     kyutai/pocket-tts/
#       languages/
#         english/
#           embeddings/*.safetensors        (voice embeddings)
#           tokenizer.model                 (SentencePiece)
#           model.safetensors               (reference weights — not needed here)
#         french_24l/ ...                   (French ships only as 24L)
#         german/, italian/, portuguese/, spanish/
#
# This script downloads just the embeddings + tokenizer into voices_<lang>/,
# flattens the nested `languages/<subdir>/embeddings/` layout that the HF
# CLI produces, and removes the HF-cache scaffolding.
#
# Usage:
#   tools/download_voices.sh <language> [<language>...]
#   tools/download_voices.sh all
#
# Examples:
#   tools/download_voices.sh french
#   tools/download_voices.sh english spanish german italian portuguese french
#   tools/download_voices.sh all
#
# Prereqs:
#   - Accept Kyutai license at https://huggingface.co/kyutai/pocket-tts
#   - `huggingface-cli login` once so the token lives in ~/.cache/huggingface/token
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_REPO="kyutai/pocket-tts"

ALL_LANGS=(english spanish german italian portuguese french)

# Local dir -> HF subdir under languages/. French is only available as 24L.
# Keep this as a case statement for compatibility with macOS's Bash 3.2.
lang_subdir() {
    case "$1" in
        english) echo "english" ;;
        spanish) echo "spanish" ;;
        german) echo "german" ;;
        italian) echo "italian" ;;
        portuguese) echo "portuguese" ;;
        french) echo "french_24l" ;;
        *) return 1 ;;
    esac
}

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <language> [<language>...]  |  $0 all" >&2
    echo "languages: ${ALL_LANGS[*]}" >&2
    exit 2
fi

if [[ "$1" == "all" ]]; then
    targets=("${ALL_LANGS[@]}")
else
    targets=("$@")
fi

if command -v hf >/dev/null 2>&1; then
    HF_DOWNLOAD=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_DOWNLOAD=(huggingface-cli download)
else
    echo "error: Hugging Face CLI not on PATH." >&2
    echo "  Install huggingface_hub, then run: hf auth login" >&2
    exit 1
fi

for lang in "${targets[@]}"; do
    if ! sub="$(lang_subdir "$lang")"; then
        echo "error: unknown language '$lang' (known: ${ALL_LANGS[*]})" >&2
        exit 2
    fi

    dest="$REPO_ROOT/voices_${lang}"
    echo "=== $lang → $dest (hf://$HF_REPO/languages/$sub) ==="
    mkdir -p "$dest"

    # huggingface-cli download lands files under the cache-style layout
    # `<local-dir>/languages/<sub>/{embeddings,*}` even with --include
    # filtering. We move them up and strip the nesting after the fact.
    "${HF_DOWNLOAD[@]}" "$HF_REPO" \
        --include "languages/$sub/embeddings/*.safetensors" \
        --include "languages/$sub/tokenizer.model" \
        --local-dir "$dest"

    src="$dest/languages/$sub"
    if [[ ! -d "$src" ]]; then
        echo "error: expected HF output at $src but it is missing." >&2
        exit 1
    fi

    # Flatten: voices_<lang>/languages/<sub>/embeddings/*.safetensors
    #      -> voices_<lang>/*.safetensors
    mv "$src/embeddings/"*.safetensors "$dest/"
    if [[ -f "$src/tokenizer.model" ]]; then
        mv "$src/tokenizer.model" "$dest/tokenizer.model"
    fi

    # Remove empty HF scaffolding dirs. We only clean what we created —
    # leave the .cache/ dir alone since it is useful for later re-runs.
    rmdir "$src/embeddings" 2>/dev/null || true
    rmdir "$src" 2>/dev/null || true
    rmdir "$dest/languages" 2>/dev/null || true

    n=$(ls "$dest"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
    tok_ok="no"
    [[ -f "$dest/tokenizer.model" ]] && tok_ok="yes"
    echo "  done: $n .safetensors, tokenizer.model=$tok_ok"
done

echo ""
echo "All requested languages populated. Sanity check:"
for lang in "${targets[@]}"; do
    d="$REPO_ROOT/voices_${lang}"
    n=$(ls "$d"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
    echo "  voices_${lang}/: $n voices"
done
