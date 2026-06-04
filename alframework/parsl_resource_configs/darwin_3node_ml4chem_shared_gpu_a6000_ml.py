from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider


def _allocation_nodes() -> list[str]:
    nodelist = os.environ.get("SLURM_NODELIST")
    if not nodelist:
        host = socket.gethostname()
        return [host, host, host]

    output = subprocess.check_output(
        ["scontrol", "show", "hostnames", nodelist],
        text=True,
    )
    nodes = [line.strip() for line in output.splitlines() if line.strip()]
    if len(nodes) < 3:
        raise RuntimeError(
            "darwin_3node_ml4chem_shared_gpu_a6000_ml requires at least 3 allocated nodes. "
            f"Found {len(nodes)} node(s): {nodes}"
        )
    return nodes


def _node_features(node_name: str) -> str:
    output = subprocess.check_output(
        ["scontrol", "show", "node", node_name],
        text=True,
    )
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("AvailableFeatures="):
            return stripped.split("=", 1)[1]
    return ""


def _select_nodes(nodes: list[str]) -> tuple[list[str], str]:
    a6000_nodes = [
        node
        for node in nodes
        if "gpu1_model:NVIDIA_RTX_A6000" in _node_features(node)
    ]
    if not a6000_nodes:
        raise RuntimeError(
            "No allocated ml4chem node has NVIDIA RTX A6000 GPUs. "
            f"Allocated nodes: {nodes}. Add an A6000 Slurm constraint/preference or resubmit later."
        )
    node_ml = a6000_nodes[-1]
    shared_gpu_nodes = [node for node in nodes if node != node_ml][:2]
    if len(shared_gpu_nodes) < 2:
        raise RuntimeError(
            "Need two non-ML nodes for shared GPU sampling/QM plus one A6000 ML node. "
            f"Allocated nodes: {nodes}, selected ML node: {node_ml}."
        )
    return shared_gpu_nodes, node_ml


def _worker_init() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return (
        "module load cuda/12.2.2 2>/dev/null || true; "
        "source /projects/opt/centos8/x86_64/miniconda3/py312_24.11.1/etc/profile.d/conda.sh; "
        "conda activate /vast/home/pjlohr/.conda/envs/atomistic; "
        f"export PYTHONPATH={repo_root}:${{PYTHONPATH:-}}; "
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    )


def _srun_overrides(node_names: list[str]) -> str:
    return (
        f"--nodes={len(node_names)} --ntasks={len(node_names)} --ntasks-per-node=1 "
        f"--nodelist={','.join(node_names)} --cpu-bind=none"
    )


class DarwinInAllocationSrunLauncher:
    """Srun launcher that writes one command file per Parsl executor block."""

    def __init__(self, debug: bool = True, overrides: str = ""):
        self.debug = debug
        self.overrides = overrides

    def __call__(self, command: str, tasks_per_node: int, nodes_per_block: int) -> str:
        task_blocks = tasks_per_node * nodes_per_block
        debug_num = int(self.debug)
        return """set -e
export CORES=$SLURM_CPUS_ON_NODE
export NODES=$SLURM_JOB_NUM_NODES
export PARSL_CMD_FILE="cmd_${{JOBNAME:-$SLURM_JOB_NAME}}.sh"

[[ "{debug}" == "1" ]] && echo "Found cores : $CORES"
[[ "{debug}" == "1" ]] && echo "Found nodes : $NODES"
[[ "{debug}" == "1" ]] && echo "Parsl command file : $PARSL_CMD_FILE"
WORKERCOUNT={task_blocks}

cat << SLURM_EOF > "$PARSL_CMD_FILE"
{command}
SLURM_EOF
chmod a+x "$PARSL_CMD_FILE"

srun --ntasks {task_blocks} -l {overrides} bash "$PARSL_CMD_FILE"

[[ "{debug}" == "1" ]] && echo "Done"
""".format(
            command=command,
            task_blocks=task_blocks,
            overrides=self.overrides,
            debug=debug_num,
        )


_nodes = _allocation_nodes()
_shared_gpu_nodes, _node_ml = _select_nodes(_nodes)
_worker_init_string = _worker_init()


config = Config(
    executors=[
        HighThroughputExecutor(
            label="alf_gpu_executor",
            provider=LocalProvider(
                nodes_per_block=2,
                init_blocks=1,
                min_blocks=1,
                max_blocks=1,
                worker_init=_worker_init_string,
                launcher=DarwinInAllocationSrunLauncher(overrides=_srun_overrides(_shared_gpu_nodes)),
            ),
            max_workers_per_node=4,
            available_accelerators=4,
            cores_per_worker=1.0,
            cpu_affinity="none",
            prefetch_capacity=0,
        ),
        HighThroughputExecutor(
            label="alf_ML_executor",
            provider=LocalProvider(
                nodes_per_block=1,
                init_blocks=1,
                min_blocks=1,
                max_blocks=1,
                worker_init=_worker_init_string,
                launcher=DarwinInAllocationSrunLauncher(overrides=_srun_overrides([_node_ml])),
            ),
            max_workers_per_node=1,
            available_accelerators=4,
            cores_per_worker=1.0,
            cpu_affinity="none",
            prefetch_capacity=0,
        ),
    ]
)
