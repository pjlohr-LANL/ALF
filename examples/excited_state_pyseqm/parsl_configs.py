"""Dynamic Darwin Slurm resources for the excited-state PySEQM example.

The ALF driver runs on a head node. Parsl requests independent allocations for
multi-state training, ALCHEMI sampling, and PySEQM labeling only when those
queues contain work. Cluster accounts, QoS names, GPU request syntax, and
environment activation are intentionally configurable below.
"""

from __future__ import annotations

import os
from pathlib import Path

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SingleNodeLauncher
from parsl.providers import SlurmProvider


ML_PARTITION = "ml4chem"
SAMPLER_PARTITION = "shared-gpu-ampere"
QM_PARTITION = "shared-gpu-ampere"

# Production-first Darwin defaults; environment variables remain authoritative.
DARWIN_ACCOUNT = os.environ.get("ALF_DARWIN_ACCOUNT", "y2020-bf") or None
ML_QOS = os.environ.get("ALF_DARWIN_ML_QOS", "long") or None
SAMPLER_QOS = os.environ.get("ALF_DARWIN_SAMPLER_QOS") or None
QM_QOS = os.environ.get("ALF_DARWIN_QM_QOS") or None

# These Darwin partitions consist of GPU nodes and normally allocate the full
# node without an explicit GPU TRES request. A site-specific directive remains
# available as an override.
ML_SCHEDULER_OPTIONS = os.environ.get(
    "ALF_DARWIN_ML_SCHEDULER_OPTIONS",
    "",
)
SAMPLER_SCHEDULER_OPTIONS = os.environ.get(
    "ALF_DARWIN_SAMPLER_SCHEDULER_OPTIONS",
    "",
)
QM_SCHEDULER_OPTIONS = os.environ.get(
    "ALF_DARWIN_QM_SCHEDULER_OPTIONS",
    "",
)

ML_WALLTIME = os.environ.get("ALF_DARWIN_ML_WALLTIME", "16:00:00")
SAMPLER_WALLTIME = os.environ.get(
    "ALF_DARWIN_SAMPLER_WALLTIME",
    "04:00:00",
)
QM_WALLTIME = os.environ.get("ALF_DARWIN_QM_WALLTIME", "04:00:00")

ML_MAX_BLOCKS = int(os.environ.get("ALF_DARWIN_ML_MAX_BLOCKS", "1"))
SAMPLER_MAX_BLOCKS = int(
    os.environ.get("ALF_DARWIN_SAMPLER_MAX_BLOCKS", "2")
)
QM_MAX_BLOCKS = int(os.environ.get("ALF_DARWIN_QM_MAX_BLOCKS", "2"))

CUDA_MODULE_COMMAND = os.environ.get(
    "ALF_DARWIN_CUDA_MODULE",
    "module load cuda/12.2.2 2>/dev/null || true",
)
# Example:
# export ALF_DARWIN_ENV_ACTIVATION='source /path/to/conda.sh; conda activate /path/to/atomistic'
ENV_ACTIVATION_COMMAND = os.environ.get(
    "ALF_DARWIN_ENV_ACTIVATION",
    (
        "source /projects/opt/centos8/x86_64/miniconda3/"
        "py312_24.11.1/etc/profile.d/conda.sh; "
        "conda activate /vast/home/pjlohr/.conda/envs/atomistic"
    ),
)
# Leave empty to use Slurm-local temporary storage. Set this to a shared or
# node-local writable path if Darwin's worker environment does not provide
# SLURM_TMPDIR or TMPDIR.
DARWIN_CACHE_ROOT = os.environ.get("ALF_DARWIN_CACHE_ROOT", "")
REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])


def _worker_init() -> str:
    commands = [
        CUDA_MODULE_COMMAND,
        ENV_ACTIVATION_COMMAND,
        f'export PYTHONPATH="{REPOSITORY_ROOT}:${{PYTHONPATH:-}}"',
        (
            f'export ALF_WORKER_CACHE_ROOT="{DARWIN_CACHE_ROOT}"'
            if DARWIN_CACHE_ROOT
            else (
                'export ALF_WORKER_CACHE_ROOT="'
                '${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/'
                'alf-${SLURM_JOB_ID:-local}"'
            )
        ),
        'export WARP_CACHE_PATH="$ALF_WORKER_CACHE_ROOT/warp"',
        'export MPLCONFIGDIR="$ALF_WORKER_CACHE_ROOT/matplotlib"',
        'mkdir -p "$WARP_CACHE_PATH" "$MPLCONFIGDIR"',
        "export PYTORCH_ALLOC_CONF=expandable_segments:True",
    ]
    return "; ".join(command for command in commands if command)


WORKER_INIT = _worker_init()


def _executor(
    *,
    label: str,
    partition: str,
    qos: str | None,
    scheduler_options: str,
    walltime: str,
    max_blocks: int,
    max_workers_per_node: int,
    available_accelerators: int | None = None,
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
        cores_per_worker=1.0,
        cpu_affinity="none",
        prefetch_capacity=0,
        provider=SlurmProvider(
            partition=partition,
            account=DARWIN_ACCOUNT,
            qos=qos,
            init_blocks=0,
            min_blocks=0,
            max_blocks=max_blocks,
            nodes_per_block=1,
            scheduler_options=scheduler_options,
            worker_init=WORKER_INIT,
            launcher=SingleNodeLauncher(),
            walltime=walltime,
            cmd_timeout=120,
        ),
    )


config_darwin = Config(
    executors=[
        _executor(
            label="alf_ML_executor",
            partition=ML_PARTITION,
            qos=ML_QOS,
            scheduler_options=ML_SCHEDULER_OPTIONS,
            walltime=ML_WALLTIME,
            max_blocks=ML_MAX_BLOCKS,
            max_workers_per_node=1,
        ),
        _executor(
            label="alf_sampler_executor",
            partition=SAMPLER_PARTITION,
            qos=SAMPLER_QOS,
            scheduler_options=SAMPLER_SCHEDULER_OPTIONS,
            walltime=SAMPLER_WALLTIME,
            max_blocks=SAMPLER_MAX_BLOCKS,
            max_workers_per_node=4,
            available_accelerators=4,
        ),
        _executor(
            label="alf_QM_executor",
            partition=QM_PARTITION,
            qos=QM_QOS,
            scheduler_options=QM_SCHEDULER_OPTIONS,
            walltime=QM_WALLTIME,
            max_blocks=QM_MAX_BLOCKS,
            max_workers_per_node=4,
            available_accelerators=4,
        ),
    ]
)


config_darwin_debug = Config(
    executors=[
        _executor(
            label="alf_ML_executor",
            partition=ML_PARTITION,
            qos=ML_QOS,
            scheduler_options=ML_SCHEDULER_OPTIONS,
            walltime="01:00:00",
            max_blocks=1,
            max_workers_per_node=1,
        ),
        _executor(
            label="alf_sampler_executor",
            partition=SAMPLER_PARTITION,
            qos=SAMPLER_QOS,
            scheduler_options=SAMPLER_SCHEDULER_OPTIONS,
            walltime="01:00:00",
            max_blocks=1,
            max_workers_per_node=4,
            available_accelerators=4,
        ),
        _executor(
            label="alf_QM_executor",
            partition=QM_PARTITION,
            qos=QM_QOS,
            scheduler_options=QM_SCHEDULER_OPTIONS,
            walltime="01:00:00",
            max_blocks=1,
            max_workers_per_node=4,
            available_accelerators=4,
        ),
    ]
)
