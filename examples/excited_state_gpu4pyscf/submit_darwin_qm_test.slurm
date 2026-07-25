#!/bin/bash
#SBATCH --partition=general
#SBATCH --account=y2020-bf
#SBATCH --qos=long
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=GPU4PYSCF_QM_TEST
#SBATCH --output=GPU4PYSCF_QM_TEST_%j.log
#SBATCH --error=GPU4PYSCF_QM_TEST_%j.err
#SBATCH --export=ALL

set -euo pipefail

EXAMPLE_DIR=/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF/examples/excited_state_gpu4pyscf
REPOSITORY_ROOT=/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF
CONDA_SH=/projects/opt/centos8/x86_64/miniconda3/py312_24.11.1/etc/profile.d/conda.sh
CONDA_ENV=/vast/home/pjlohr/.conda/envs/alf_env

export ALF_DARWIN_ACCOUNT=y2020-bf
export ALF_DARWIN_CUDA_MODULE="module load cuda/12.2.2 2>/dev/null || true"
export ALF_DARWIN_ENV_ACTIVATION="source ${CONDA_SH}; conda activate ${CONDA_ENV}"
export ALF_DARWIN_GPU4PYSCF_ENV_ACTIVATION="${ALF_DARWIN_ENV_ACTIVATION}"
export ALF_DARWIN_SAMPLER_SCHEDULER_OPTIONS="#SBATCH --constraint=gpu_count:4"
export ALF_DARWIN_GPU4PYSCF_SCHEDULER_OPTIONS="#SBATCH --constraint=gpu_count:4"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPOSITORY_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/alf-gpu4pyscf-qm-test-mpl"
mkdir -p "${MPLCONFIGDIR}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${EXAMPLE_DIR}"

"${CONDA_ENV}/bin/python" -u gpu4pyscf_qm_smoke.py
