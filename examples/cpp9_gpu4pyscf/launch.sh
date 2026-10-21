#!/usr/bin/env bash
set -euo pipefail
EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd -- "$EXAMPLE_DIR/../.." && pwd)"
if [[ -d "$REPOSITORY/alframework" ]]; then
    export PYTHONPATH="$REPOSITORY:${PYTHONPATH:-}"
fi
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${ALF_PYTHON:-python}" -u "$EXAMPLE_DIR/run_trial.py" "$@"
