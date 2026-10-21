"""Editable local/Slurm resources; constructing configurations submits no work."""
from pathlib import Path
import shlex

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SingleNodeLauncher
from parsl.providers import LocalProvider, SlurmProvider


def validate_resources(settings):
    if settings.get('backend') not in {'local','slurm'}:
        raise ValueError('resources.backend must be local or slurm.')
    for key in ('gpus_per_node','qm_workers_per_node','qm_cores_per_worker','ml_cores_per_worker'):
        value=settings.get(key)
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(f'resources.{key} must be a positive integer.')
    if settings['qm_workers_per_node']>settings['gpus_per_node']:
        raise ValueError('Use at most one QM worker per configured GPU.')
    if not isinstance(settings.get('worker_init',''),str):
        raise ValueError('resources.worker_init must be shell text.')
    for role in ('QM','ML'):
        site=settings[role]
        if isinstance(site.get('max_blocks'),bool) or not isinstance(site.get('max_blocks'),int) or site['max_blocks']<1:
            raise ValueError(f'{role}.max_blocks must be positive.')
        hours,minutes,seconds=map(int,site['walltime'].split(':'))
        if not (hours>=0 and 0<=minutes<60 and 0<=seconds<60 and hours*3600+minutes*60+seconds>300):
            raise ValueError(f'{role}.walltime must exceed five minutes.')
        if settings['backend']=='slurm' and not site.get('partition'):
            raise ValueError(f'Set resources.{role}.partition for this Slurm site.')
    return settings


def worker_init(example_root,settings,role):
    import alframework
    package_root=Path(alframework.__file__).resolve().parent.parent
    threads=(settings['qm_cores_per_worker'] if role=='QM' else
             max(1,settings['ml_cores_per_worker']//min(4,settings['gpus_per_node'])))
    paths=shlex.quote(str(Path(example_root).resolve())+':'+str(package_root))
    # Respect the scheduler/Parsl GPU assignment; never manufacture visible devices.
    return '\n'.join([
        'set -e',settings.get('worker_init',''),
        'export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1',
        f'export PYTHONPATH={paths}:"${{PYTHONPATH:-}}"',
        f'export OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads}',
        'export CPP_WORKER_CACHE="$(mktemp -d "${TMPDIR:-/tmp}/cpp9-worker-XXXXXX")"',
        'export CUPY_CACHE_DIR="$CPP_WORKER_CACHE/cupy"',
        'export MPLCONFIGDIR="$CPP_WORKER_CACHE/matplotlib"',
        'export WARP_CACHE_PATH="$CPP_WORKER_CACHE/warp"',
        'mkdir -p "$CUPY_CACHE_DIR" "$MPLCONFIGDIR" "$WARP_CACHE_PATH"',
        'export PYTORCH_ALLOC_CONF=expandable_segments:True','',
    ])


def build_config(example_root,run_dir,settings, *, roles=('QM','ML')):
    validate_resources(settings)
    executors=[]
    for role in roles:
        if role not in {'QM','ML'}:
            raise ValueError('Unsupported CPP9 executor role.')
        site=settings[role]
        initialization=worker_init(example_root,settings,role)
        if settings['backend']=='local':
            provider=LocalProvider(init_blocks=0,min_blocks=0,max_blocks=1,worker_init=initialization)
            drain={}
        else:
            provider=SlurmProvider(account=settings.get('account'),qos=settings.get('qos'),
                partition=site['partition'],nodes_per_block=1,init_blocks=0,min_blocks=0,
                max_blocks=site['max_blocks'],walltime=site['walltime'],
                scheduler_options=site.get('scheduler_options',''),
                cores_per_node=site['cores_per_node'],mem_per_node=site['mem_per_node_gb'],
                parallelism=1.,cmd_timeout=120,launcher=SingleNodeLauncher(),worker_init=initialization)
            hours,minutes,seconds=map(int,site['walltime'].split(':'))
            drain={'drain_period':hours*3600+minutes*60+seconds-300}
        gpu_assignment={'available_accelerators':settings['gpus_per_node']} if role=='QM' else {}
        executors.append(HighThroughputExecutor(label=f'alf_{role}_executor',
            max_workers_per_node=settings['qm_workers_per_node'] if role=='QM' else 1,
            cores_per_worker=settings['qm_cores_per_worker'] if role=='QM' else settings['ml_cores_per_worker'],
            cpu_affinity='none',prefetch_capacity=0,provider=provider,**gpu_assignment,**drain))
    return Config(executors=executors,run_dir=str(Path(run_dir)/'runinfo'),
                  strategy='simple',max_idletime=120,retries=0)
