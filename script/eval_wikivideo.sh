#!/bin/bash
# WikiVideo (video / audio / omni). Needs the extracted wikivideo.tar.gz
# under $UMUI_ROOT (see script/download.sh). Extra args pass through.
#
#   UMUI_ROOT=/path/to/UMUI-data bash script/eval_wikivideo.sh [modality] [--batch-size 8 ...]
set -euo pipefail

ROOT=${UMUI_ROOT:-./UMUI-data}
MODALITY=video
if [ $# -gt 0 ] && [[ $1 != -* ]]; then
    MODALITY=$1
    shift
fi

python -m src.eval.run_eval --dataset wikivideo --modality "$MODALITY" \
    --media-root "$ROOT/wikivideo_" \
    "$@"
