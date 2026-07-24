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

EXAMPLE_DIR="${ALF_GPU4PYSCF_EXAMPLE_DIR:-/path/to/ALF/examples/excited_state_gpu4pyscf}"
REPOSITORY_ROOT="${ALF_REPOSITORY_ROOT:-/path/to/ALF}"
CONDA_SH="${ALF_CONDA_SH:-/path/to/conda.sh}"
CONDA_ENV="${ALF_DRIVER_CONDA_ENV:-/path/to/atomistic}"
PYTHON_BIN="${ALF_DRIVER_PYTHON:-${CONDA_ENV}/bin/python}"
MASTER_CONFIG="${ALF_GPU4PYSCF_MASTER:-master_config.json}"

export ALF_DARWIN_ACCOUNT="${ALF_DARWIN_ACCOUNT:-y2020-bf}"
export ALF_DARWIN_ML_QOS="${ALF_DARWIN_ML_QOS:-long}"
export ALF_DARWIN_CUDA_MODULE="${ALF_DARWIN_CUDA_MODULE:-module load cuda/12.2.2 2>/dev/null || true}"
export ALF_DARWIN_ENV_ACTIVATION="${ALF_DARWIN_ENV_ACTIVATION:-source ${CONDA_SH}; conda activate ${CONDA_ENV}}"
export ALF_DARWIN_GPU4PYSCF_ENV_ACTIVATION="${ALF_DARWIN_GPU4PYSCF_ENV_ACTIVATION:-${ALF_DARWIN_ENV_ACTIVATION}}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPOSITORY_ROOT}:${PYTHONPATH:-}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${EXAMPLE_DIR}"

echo "Starting six-state keto GPU4PySCF ALF workflow"
echo "Host: $(hostname)"
echo "Start time: $(date --iso-8601=seconds)"
echo "Python: ${PYTHON_BIN}"
echo "Master: ${EXAMPLE_DIR}/${MASTER_CONFIG}"

"${PYTHON_BIN}" -u -m alframework --master "${MASTER_CONFIG}"
