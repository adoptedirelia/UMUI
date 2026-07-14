#!/bin/bash
# Clotho entailment (audio). Needs the extracted audio_entailment.tar.gz
# under $UMUI_ROOT (see script/download.sh). Extra args pass through.
#
#   UMUI_ROOT=/path/to/UMUI-data bash script/eval_clotho.sh [--batch-size 8 ...]
set -euo pipefail

ROOT=${UMUI_ROOT:-./UMUI-data}

python -m src.eval.run_eval --dataset clotho --modality audio \
    --media-root "$ROOT/audio_entailment" \
    "$@"
