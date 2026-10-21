#!/usr/bin/env bash
#SBATCH --job-name=CPP9_DRIVER
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=cpp9-driver-%j.out
#SBATCH --error=cpp9-driver-%j.err
#SBATCH --signal=B:TERM@300
#SBATCH --export=ALL

# Supply site-specific --partition, --account and --qos to sbatch.
# Submit from the example directory, or export CPP_EXAMPLE_DIR explicitly.
set -euo pipefail
EXAMPLE_DIR="${CPP_EXAMPLE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -f "$EXAMPLE_DIR/launch.sh" ]]; then
    echo 'Set CPP_EXAMPLE_DIR to the copied cpp9_gpu4pyscf example directory.' >&2
    exit 2
fi
exec bash "$EXAMPLE_DIR/launch.sh" "$@"
