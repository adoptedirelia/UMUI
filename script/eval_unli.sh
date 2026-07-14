#!/bin/bash
# UNLI validation split (text). Run from the repo root with the qwen25omni
# env active; extra args are passed through to run_eval.py.
#
#   UMUI_ROOT=/path/to/UMUI-data bash script/eval_unli.sh [--batch-size 8 ...]
#
# The labels download from the hub automatically; set UMUI_ROOT to use the
# local copy instead.
set -euo pipefail

ROOT=${UMUI_ROOT:-./UMUI-data}
DATA_FILE="$ROOT/training-data/unli/validation.jsonl"

ARGS=()
if [ -f "$DATA_FILE" ]; then
    ARGS+=(--data-file "$DATA_FILE")
fi

python -m src.eval.run_eval --dataset unli --modality text "${ARGS[@]}" "$@"
