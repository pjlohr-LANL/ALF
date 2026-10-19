#!/usr/bin/env bash
# Run inside tmux or an appropriate driver allocation after configuring Parsl.
set -euo pipefail
RUN_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ALF_PYTHON:-python}"
cd "$RUN_DIRECTORY"
export PYTHONPATH="$RUN_DIRECTORY:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
CHECK_OPTIONS=()
if [[ ! -f h5store/data-0000.h5 ]]; then
    CHECK_OPTIONS+=(--require-seeds)
fi
"$PYTHON_BIN" prepare.py check --master master_config.json "${CHECK_OPTIONS[@]}"
exec "$PYTHON_BIN" -u -m alframework --master master_config.json
