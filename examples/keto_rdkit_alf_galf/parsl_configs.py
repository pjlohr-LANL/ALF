"""Editable Slurm resources; importing this file submits no allocations.

Run the driver in its prepared run directory. Configure the account, partitions,
GPU directives, and worker environment for your site before launch.
"""
import os
from pathlib import Path
import shlex

import alframework
from parsl.config import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.launchers import SingleNodeLauncher
from parsl.providers import SlurmProvider

ROOT = Path(__file__).resolve().parent
REPOSITORY = Path(alframework.__file__).resolve().parents[1]
GPUS_PER_NODE = int(os.environ.get('ALF_GPUS_PER_NODE', '4'))  # Match master_config.json.
WORKER_INIT = '\n'.join([
    os.environ.get('ALF_ENV_ACTIVATION', ''),
    f'export PYTHONPATH={shlex.quote(str(REPOSITORY))}:"${{PYTHONPATH:-}}"',
    'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1',
    'export ALF_WORKER_CACHE_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/alf-${SLURM_JOB_ID:-local}"',
    'export WARP_CACHE_PATH="$ALF_WORKER_CACHE_ROOT/warp" MPLCONFIGDIR="$ALF_WORKER_CACHE_ROOT/matplotlib"',
    'mkdir -p "$WARP_CACHE_PATH" "$MPLCONFIGDIR"',
])


def executor(role, *, workers, accelerators=False):
    return HighThroughputExecutor(
        label=f'alf_{role}_executor', max_workers_per_node=workers,
        **({'available_accelerators': GPUS_PER_NODE} if accelerators else {}),
        cores_per_worker=1., cpu_affinity='none', prefetch_capacity=0,
        provider=SlurmProvider(
            account=os.environ.get('ALF_ACCOUNT') or None,
            partition=os.environ.get(f'ALF_{role.upper()}_PARTITION', 'ml4chem'),
            qos=os.environ.get(f'ALF_{role.upper()}_QOS') or None,
            scheduler_options=os.environ.get(f'ALF_{role.upper()}_SCHEDULER_OPTIONS', ''),
            worker_init=WORKER_INIT, nodes_per_block=1,
            init_blocks=0, min_blocks=0,
            max_blocks=int(os.environ.get(f'ALF_{role.upper()}_MAX_BLOCKS', '1')),
            walltime=os.environ.get(f'ALF_{role.upper()}_WALLTIME', '04:00:00'),
            launcher=SingleNodeLauncher(), cmd_timeout=120,
        ),
    )


config = Config(
    executors=[ThreadPoolExecutor(label='alf_builder_executor', max_threads=2),
               executor('ML', workers=1),
               executor('sampler', workers=GPUS_PER_NODE, accelerators=True),
               executor('QM', workers=GPUS_PER_NODE, accelerators=True)],
    run_dir=str(ROOT / 'runinfo'),
)
