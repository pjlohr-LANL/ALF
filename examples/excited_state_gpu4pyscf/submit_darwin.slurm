#!/bin/bash
#SBATCH --partition=general
#SBATCH --account=y2020-bf
#SBATCH --qos=long
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=ALF_GPU4PYSCF_KETO
#SBATCH --output=ALF_GPU4PYSCF_KETO_%j.log
#SBATCH --error=ALF_GPU4PYSCF_KETO_%j.err
#SBATCH --export=ALL

set -euo pipefail

EXAMPLE_DIR="${ALF_GPU4PYSCF_EXAMPLE_DIR:-/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF/examples/excited_state_gpu4pyscf}"
exec bash "${EXAMPLE_DIR}/launch_headnode.sh"
