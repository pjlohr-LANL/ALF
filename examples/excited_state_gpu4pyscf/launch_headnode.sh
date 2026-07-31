#!/bin/bash

# Run this driver directly inside a persistent tmux session on a Darwin head
# node. Training, sampling, and QM calculations are submitted to Slurm by the
# independent Parsl executors configured in parsl_configs.py.

set -euo pipefail

EXAMPLE_DIR="${ALF_GPU4PYSCF_EXAMPLE_DIR:-/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF/examples/excited_state_gpu4pyscf}"
REPOSITORY_ROOT="${ALF_REPOSITORY_ROOT:-/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF}"
CONDA_SH="${ALF_CONDA_SH:-/projects/opt/centos8/x86_64/miniconda3/py312_24.11.1/etc/profile.d/conda.sh}"
CONDA_ENV="${ALF_DRIVER_CONDA_ENV:-/vast/home/pjlohr/.conda/envs/alf_env}"
PYTHON_BIN="${ALF_DRIVER_PYTHON:-${CONDA_ENV}/bin/python}"

export ALF_DARWIN_ACCOUNT="${ALF_DARWIN_ACCOUNT:-y2020-bf}"
export ALF_DARWIN_ML_QOS="${ALF_DARWIN_ML_QOS:-long}"
export ALF_DARWIN_SAMPLER_QOS="${ALF_DARWIN_SAMPLER_QOS:-long}"
export ALF_DARWIN_GPU4PYSCF_QOS="${ALF_DARWIN_GPU4PYSCF_QOS:-long}"
export ALF_DARWIN_CUDA_MODULE="${ALF_DARWIN_CUDA_MODULE:-module load cuda/12.2.2 2>/dev/null || true}"
export ALF_DARWIN_ENV_ACTIVATION="${ALF_DARWIN_ENV_ACTIVATION:-source ${CONDA_SH}; conda activate ${CONDA_ENV}}"
export ALF_DARWIN_GPU4PYSCF_ENV_ACTIVATION="${ALF_DARWIN_GPU4PYSCF_ENV_ACTIVATION:-${ALF_DARWIN_ENV_ACTIVATION}}"
export ALF_DARWIN_SAMPLER_SCHEDULER_OPTIONS="${ALF_DARWIN_SAMPLER_SCHEDULER_OPTIONS:-#SBATCH --constraint=gpu_count:4}"
export ALF_DARWIN_GPU4PYSCF_SCHEDULER_OPTIONS="${ALF_DARWIN_GPU4PYSCF_SCHEDULER_OPTIONS:-#SBATCH --constraint=gpu_count:4}"
export ALF_DARWIN_SAMPLER_WALLTIME="${ALF_DARWIN_SAMPLER_WALLTIME:-12:00:00}"
export ALF_DARWIN_GPU4PYSCF_WALLTIME="${ALF_DARWIN_GPU4PYSCF_WALLTIME:-08:00:00}"
export ALF_DARWIN_GPU4PYSCF_DRAIN_PERIOD="${ALF_DARWIN_GPU4PYSCF_DRAIN_PERIOD:-28200}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPOSITORY_ROOT}:${PYTHONPATH:-}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${EXAMPLE_DIR}"

echo "Starting seeded excited-state GPU4PySCF ALF workflow"
echo "Driver host: $(hostname)"
echo "Start time: $(date --iso-8601=seconds)"
echo "Python: ${PYTHON_BIN}"
echo "Master: ${EXAMPLE_DIR}/master_config.json"
echo "Sampling constraint: ${ALF_DARWIN_SAMPLER_SCHEDULER_OPTIONS}"
echo "Sampling QoS/walltime: ${ALF_DARWIN_SAMPLER_QOS} ${ALF_DARWIN_SAMPLER_WALLTIME}"
echo "QM constraint: ${ALF_DARWIN_GPU4PYSCF_SCHEDULER_OPTIONS}"
echo "QM QoS/walltime: ${ALF_DARWIN_GPU4PYSCF_QOS} ${ALF_DARWIN_GPU4PYSCF_WALLTIME}"
echo "QM drain period: ${ALF_DARWIN_GPU4PYSCF_DRAIN_PERIOD} seconds"

exec "${PYTHON_BIN}" -u -m alframework --master master_config.json
