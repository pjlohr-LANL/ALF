#!/bin/bash
#SBATCH --partition=general
#SBATCH --account=y2020-bf
#SBATCH --qos=long
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=ALF_6STATE_KETO
#SBATCH --output=ALF_6STATE_KETO_%j.log
#SBATCH --error=ALF_6STATE_KETO_%j.err
#SBATCH --export=ALL

set -euo pipefail

EXAMPLE_DIR="/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF/examples/excited_state_pyseqm"
REPOSITORY_ROOT="/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF"
PYTHON_BIN="/vast/home/pjlohr/.conda/envs/atomistic/bin/python"
CONDA_SH="/projects/opt/centos8/x86_64/miniconda3/py312_24.11.1/etc/profile.d/conda.sh"
CONDA_ENV="/vast/home/pjlohr/.conda/envs/atomistic"

export ALF_DARWIN_ACCOUNT="y2020-bf"
export ALF_DARWIN_ML_QOS="long"
export ALF_DARWIN_CUDA_MODULE="module load cuda/12.2.2 2>/dev/null || true"
export ALF_DARWIN_ENV_ACTIVATION='source /projects/opt/centos8/x86_64/miniconda3/py312_24.11.1/etc/profile.d/conda.sh; conda activate /vast/home/pjlohr/.conda/envs/atomistic'
export ALF_DARWIN_ML_SCHEDULER_OPTIONS=""
export ALF_DARWIN_SAMPLER_SCHEDULER_OPTIONS=""
export ALF_DARWIN_QM_SCHEDULER_OPTIONS=""
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPOSITORY_ROOT}:${PYTHONPATH:-}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${EXAMPLE_DIR}"

echo "Starting six-state keto ALF production replica"
echo "Host: $(hostname)"
echo "Start time: $(date --iso-8601=seconds)"
echo "Python: ${PYTHON_BIN}"
echo "Master: ${EXAMPLE_DIR}/master_config.json"

"${PYTHON_BIN}" -u -m alframework --master master_config.json
