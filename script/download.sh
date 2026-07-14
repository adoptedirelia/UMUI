#!/bin/bash
# Download the released UNLI model and the UMUI dataset from Hugging Face.
#
# Usage:
#   bash script/download.sh [target_dir]        # default: ./UMUI-data
#
# The model and the small label files are always fetched; the media tarballs
# (wikivideo.tar.gz ~17 GB, audio_entailment.tar.gz ~9 GB) are only needed for
# the wikivideo / clotho evals — skip them with SKIP_MEDIA=1.
# Afterwards point the eval scripts at the target dir via UMUI_ROOT.
set -euo pipefail

ROOT=${1:-${UMUI_ROOT:-./UMUI-data}}
mkdir -p "$ROOT"

# model (base + LoRA adapter); lands in the HF cache, run_eval.py loads it by repo id
hf download AdoptedIrelia/UNLI

# label files
hf download AdoptedIrelia/UMUI --repo-type dataset --local-dir "$ROOT" \
    --include "UMUI-annotation/*" "training-data/unli/*" "UMUI-binary/clotho/*"

if [ -z "${SKIP_MEDIA:-}" ]; then
    hf download AdoptedIrelia/UMUI --repo-type dataset --local-dir "$ROOT" \
        --include "wikivideo.tar.gz" "audio_entailment.tar.gz"
    tar -xzf "$ROOT/wikivideo.tar.gz" -C "$ROOT"          # -> $ROOT/wikivideo_
    tar -xzf "$ROOT/audio_entailment.tar.gz" -C "$ROOT"   # -> $ROOT/audio_entailment
fi

echo "done. run the evals with: UMUI_ROOT=$ROOT bash script/eval_<dataset>.sh"
