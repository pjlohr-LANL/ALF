"""Dynamic Darwin resources for the GPU4PySCF excited-state example.

The ALF driver runs on a head node. Training, sampling, and QM labeling use
independent Slurm allocations. GPU4PySCF remains nonbatched: every QM Parsl
task receives one accelerator and evaluates all configured electronic states
for one molecule on that accelerator.
"""

from __future__ import annotations

import os
from pathlib import Path

from parsl.config import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.launchers import SingleNodeLauncher
from parsl.providers import SlurmProvider


ML_PARTITION = "ml4chem"
SAMPLER_PARTITION = "shared-gpu-ampere"
QM_PARTITION = "shared-gpu-ampere"

DARWIN_ACCOUNT = os.environ.get("ALF_DARWIN_ACCOUNT", "y2020-bf") or None
ML_QOS = os.environ.get("ALF_DARWIN_ML_QOS", "long") or None
SAMPLER_QOS = os.environ.get("ALF_DARWIN_SAMPLER_QOS", "long") or None
QM_QOS = os.environ.get("ALF_DARWIN_GPU4PYSCF_QOS", "long") or None

ML_SCHEDULER_OPTIONS = os.environ.get(
    "ALF_DARWIN_ML_SCHEDULER_OPTIONS",
    "",
)
SAMPLER_SCHEDULER_OPTIONS = os.environ.get(
    "ALF_DARWIN_SAMPLER_SCHEDULER_OPTIONS",
    "",
)
# Set this to Darwin's site-approved A100 request when the partition requires
# an explicit GPU directive.
QM_SCHEDULER_OPTIONS = os.environ.get(
    "ALF_DARWIN_GPU4PYSCF_SCHEDULER_OPTIONS",
    "",
)

ML_WALLTIME = os.environ.get("ALF_DARWIN_ML_WALLTIME", "16:00:00")
SAMPLER_WALLTIME = os.environ.get(
    "ALF_DARWIN_SAMPLER_WALLTIME",
    "12:00:00",
)
QM_WALLTIME = os.environ.get(
    "ALF_DARWIN_GPU4PYSCF_WALLTIME",
    "08:00:00",
)
QM_DRAIN_PERIOD = int(
    os.environ.get("ALF_DARWIN_GPU4PYSCF_DRAIN_PERIOD", "28200")
)

ML_MAX_BLOCKS = int(os.environ.get("ALF_DARWIN_ML_MAX_BLOCKS", "1"))
SAMPLER_MAX_BLOCKS = int(
    os.environ.get("ALF_DARWIN_SAMPLER_MAX_BLOCKS", "2")
)
QM_MAX_BLOCKS = int(
    os.environ.get("ALF_DARWIN_GPU4PYSCF_MAX_BLOCKS", "2")
)
QM_CORES_PER_WORKER = float(
    os.environ.get("ALF_DARWIN_GPU4PYSCF_CORES_PER_WORKER", "8")
)

CUDA_MODULE_COMMAND = os.environ.get(
    "ALF_DARWIN_CUDA_MODULE",
    "module load cuda/12.2.2 2>/dev/null || true",
)
ATOMISTIC_ENV_ACTIVATION = os.environ.get(
    "ALF_DARWIN_ENV_ACTIVATION",
    (
        "source /projects/opt/centos8/x86_64/miniconda3/"
        "py312_24.11.1/etc/profile.d/conda.sh; "
        "conda activate /vast/home/pjlohr/.conda/envs/alf_env"
    ),
)
GPU4PYSCF_ENV_ACTIVATION = os.environ.get(
    "ALF_DARWIN_GPU4PYSCF_ENV_ACTIVATION",
    ATOMISTIC_ENV_ACTIVATION,
)
GPU4PYSCF_PYTHONPATH = os.environ.get(
    "ALF_DARWIN_GPU4PYSCF_PYTHONPATH",
    "",
)
DARWIN_CACHE_ROOT = os.environ.get("ALF_DARWIN_CACHE_ROOT", "")
GPU4PYSCF_CACHE_ROOT = os.environ.get(
    "ALF_DARWIN_GPU4PYSCF_CACHE_ROOT",
    DARWIN_CACHE_ROOT,
)
REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])


def _worker_init(
    *,
    environment_activation: str,
    cache_root: str,
    extra_pythonpath: str = "",
) -> str:
    pythonpath_entries = [REPOSITORY_ROOT]
    if extra_pythonpath:
        pythonpath_entries.insert(0, extra_pythonpath)
    pythonpath_prefix = ":".join(pythonpath_entries)
    commands = [
        CUDA_MODULE_COMMAND,
        environment_activation,
        "export PYTHONNOUSERSITE=1",
        f'export PYTHONPATH="{pythonpath_prefix}:${{PYTHONPATH:-}}"',
        (
            f'export ALF_WORKER_CACHE_ROOT="{cache_root}"'
            if cache_root
            else (
                'export ALF_WORKER_CACHE_ROOT="'
                '${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/'
                'alf-${SLURM_JOB_ID:-local}"'
            )
        ),
        'export WARP_CACHE_PATH="$ALF_WORKER_CACHE_ROOT/warp"',
        'export MPLCONFIGDIR="$ALF_WORKER_CACHE_ROOT/matplotlib"',
        'export CUPY_CACHE_DIR="$ALF_WORKER_CACHE_ROOT/cupy"',
        (
            'mkdir -p "$WARP_CACHE_PATH" "$MPLCONFIGDIR" '
            '"$CUPY_CACHE_DIR"'
        ),
        "export PYTORCH_ALLOC_CONF=expandable_segments:True",
    ]
    return "; ".join(command for command in commands if command)


ATOMISTIC_WORKER_INIT = _worker_init(
    environment_activation=ATOMISTIC_ENV_ACTIVATION,
    cache_root=DARWIN_CACHE_ROOT,
)
GPU4PYSCF_WORKER_INIT = _worker_init(
    environment_activation=GPU4PYSCF_ENV_ACTIVATION,
    cache_root=GPU4PYSCF_CACHE_ROOT,
    extra_pythonpath=GPU4PYSCF_PYTHONPATH,
)


def _executor(
    *,
    label: str,
    partition: str,
    qos: str | None,
    scheduler_options: str,
    walltime: str,
    max_blocks: int,
    max_workers_per_node: int,
    worker_init: str,
    cores_per_worker: float = 1.0,
    available_accelerators: int | None = None,
    drain_period: int | None = None,
) -> HighThroughputExecutor:
    accelerator_options = (
        {}
        if available_accelerators is None
        else {"available_accelerators": available_accelerators}
    )
    return HighThroughputExecutor(
        label=label,
        max_workers_per_node=max_workers_per_node,
        **accelerator_options,
        cores_per_worker=cores_per_worker,
        cpu_affinity="none",
        prefetch_capacity=0,
        drain_period=drain_period,
        provider=SlurmProvider(
            partition=partition,
            account=DARWIN_ACCOUNT,
            qos=qos,
            init_blocks=0,
            min_blocks=0,
            max_blocks=max_blocks,
            nodes_per_block=1,
            scheduler_options=scheduler_options,
            worker_init=worker_init,
            launcher=SingleNodeLauncher(),
            walltime=walltime,
            cmd_timeout=120,
        ),
    )


def _darwin_config(*, debug: bool) -> Config:
    return Config(
        executors=[
            ThreadPoolExecutor(
                label="alf_builder_executor",
                max_threads=4,
            ),
            _executor(
                label="alf_ML_executor",
                partition=ML_PARTITION,
                qos=ML_QOS,
                scheduler_options=ML_SCHEDULER_OPTIONS,
                walltime="01:00:00" if debug else ML_WALLTIME,
                max_blocks=1 if debug else ML_MAX_BLOCKS,
                max_workers_per_node=1,
                worker_init=ATOMISTIC_WORKER_INIT,
            ),
            _executor(
                label="alf_sampler_executor",
                partition=SAMPLER_PARTITION,
                qos=SAMPLER_QOS,
                scheduler_options=SAMPLER_SCHEDULER_OPTIONS,
                walltime="01:00:00" if debug else SAMPLER_WALLTIME,
                max_blocks=1 if debug else SAMPLER_MAX_BLOCKS,
                max_workers_per_node=4,
                available_accelerators=4,
                worker_init=ATOMISTIC_WORKER_INIT,
            ),
            _executor(
                label="alf_QM_executor",
                partition=QM_PARTITION,
                qos=QM_QOS,
                scheduler_options=QM_SCHEDULER_OPTIONS,
                walltime="01:00:00" if debug else QM_WALLTIME,
                max_blocks=1 if debug else QM_MAX_BLOCKS,
                max_workers_per_node=4,
                available_accelerators=4,
                cores_per_worker=QM_CORES_PER_WORKER,
                drain_period=QM_DRAIN_PERIOD,
                worker_init=GPU4PYSCF_WORKER_INIT,
            ),
        ]
    )


config_darwin = _darwin_config(debug=False)
config_darwin_debug = _darwin_config(debug=True)
