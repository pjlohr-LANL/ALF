# Six-state keto GPU4PySCF active learning

This example labels keto acetylacetone with GPU4PySCF and trains the same
six-surface HIPPYNN/ALCHEMI workflow used by the excited-state PySEQM example.
The examples are intentionally independent: they do not share QM
configurations, HDF5 shards, models, status files, scratch directories, or
worker environments.

## Electronic-structure contract

Every nonbatched QM Parsl task receives one molecule and one A100. That task
runs one density-fitted RKS/TDA calculation and returns:

```text
sE0, F0, sE1, F1, ..., sE5, F5
```

All six surfaces remain on the same worker and GPU. Electronic states are not
assigned to different GPUs. When four A100s are available, Parsl may run four
different molecules concurrently, one molecule per GPU.

The validated settings in `QM_config.json` are:

- CAM-B3LYP/6-31G*
- neutral singlet
- density-fitted RKS
- grid level 3
- TDA with five excited roots
- TDA convergence tolerance `1.0e-6` and at most 300 cycles
- eight CPU threads per GPU worker

The task requires SCF and every requested root to converge before it stores
any label. Transition dipoles, NACVs, state tracking, full TDDFT, PBC, CPU
fallback, and QM batching are not part of this example.

## Startup modes

### Self-contained bootstrap

`master_config.json` follows:

```text
keto CFG
  -> 50 serially generated GPU4PySCF labels
  -> data-0000.h5
  -> six-state HIPPYNN
  -> state-cycled ALCHEMI
  -> additional GPU4PySCF labels and retraining
```

The CFG loader applies a `0.03 Å` coordinate shake so the bootstrap structures
are distinct. Fifty structures are enough to exercise the complete workflow,
but they are not a scientifically converged production training set.

### Existing GPU4PySCF HDF5

Place a compatible shard at `h5store/data-0000.h5` and use:

```bash
python -m alframework --master master_config_existing_h5.json
```

With no status file, ALF detects the shard, skips bootstrap labeling, and
trains the first model. The shard must contain finite `sE0/F0` through
`sE5/F5` labels using the configured database names, scales, 15-atom
composition, and stable atomic ordering.

Do not copy the AM1 PySEQM seed shard from `excited_state_pyseqm`. The legacy
ALF HDF5 schema does not record enough electronic-structure provenance to
detect a manually copied, method-incompatible shard. Keeping the two example
directories separate prevents accidental automatic reuse, but operators must
verify the provenance of imported HDF5 data.

## Darwin execution

`parsl_configs.py` creates independent, dynamically scaling executors:

| Stage | Partition | Workers/node | GPUs per worker | Maximum blocks |
| --- | --- | ---: | ---: | ---: |
| HIPPYNN training | `ml4chem` | 1 | ensemble-managed | 1 |
| ALCHEMI sampling | `shared-gpu-ampere` | 4 | 1 | 2 |
| GPU4PySCF QM | `shared-gpu-ampere` | 4 | 1 | 2 |

The QM executor advertises four accelerators, so Parsl pins each worker to one
A100. `target_queued_QM: 6` keeps the four workers occupied while allowing a
small queue. Each QM task still calculates all six surfaces for only one
molecule.

The master `gpus_per_node` remains four because ALF also passes it to the
four-GPU sampler and trainer. GPU4PySCF selects the worker's visible local
device; it does not use four GPUs for one molecule.

The checked-in launchers default to account `y2020-bf`, CUDA `12.2.2`, the
four-GPU-node constraint, and the self-contained environment at
`/vast/home/pjlohr/.conda/envs/alf_env`. The settings remain configurable
through these environment variables:

```bash
export ALF_DARWIN_ACCOUNT="..."
export ALF_DARWIN_GPU4PYSCF_QOS="..."
export ALF_DARWIN_GPU4PYSCF_SCHEDULER_OPTIONS="..."
export ALF_DARWIN_GPU4PYSCF_WALLTIME="12:00:00"
export ALF_DARWIN_GPU4PYSCF_MAX_BLOCKS="2"
export ALF_DARWIN_GPU4PYSCF_CORES_PER_WORKER="8"
export ALF_DARWIN_ENV_ACTIVATION="source .../conda.sh; conda activate .../alf_env"
export ALF_DARWIN_GPU4PYSCF_ENV_ACTIVATION="${ALF_DARWIN_ENV_ACTIVATION}"
```

If GPU4PySCF is used from a source checkout, also set:

```bash
export ALF_DARWIN_GPU4PYSCF_PYTHONPATH="/path/containing/gpu4pyscf"
```

Install the distribution matching Darwin's CUDA runtime in the QM worker
environment, for example `gpu4pyscf-cuda12x`. The worker environment must
also contain ALF's ordinary runtime dependencies.

## Stage checks

Start with the exact, unshaken CFG geometry:

```bash
cd examples/excited_state_gpu4pyscf
export PYTHONPATH=/path/to/ALF:${PYTHONPATH:-}

python -m alframework --master master_config_debug.json --test_builder
python -m alframework --master master_config_debug.json --test_qm
```

The QM result must contain six finite energies and six finite `(15, 3)` force
arrays. Its metadata must report `qm_backend: gpu4pyscf`, one selected CUDA
device, successful SCF/TDA convergence, and five excited roots.

Training and sampling checks require a compatible HDF5 shard/model:

```bash
python -m alframework --master master_config_debug.json --test_ml
python -m alframework --master master_config_debug.json --test_sampler
```

For Darwin acceptance, compare one `--test_qm` geometry against
`dataset_workflow/scripts/07_gpu4pyscf_backend.py` using five roots and the
same method, basis, charge, and grid. Then submit several molecules and verify
from worker metadata/logs that different tasks use different A100s while all
S0-S5 results for a molecule share one device.

## Production launch from a head node

Start the ALF driver directly in a persistent `tmux` session. The driver stays
on the head node; the three Parsl executors dynamically submit all training,
sampling, and QM compute work to Slurm.

```bash
tmux new -s alf_gpu4pyscf
cd /vast/home/pjlohr/ALF_LANL/ALF_fork/ALF/examples/excited_state_gpu4pyscf
bash launch_headnode.sh
```

Detach without stopping the driver with `Ctrl-b d`. Reattach later with:

```bash
tmux attach -t alf_gpu4pyscf
```

If the driver exits and must be restarted, reattach to the session (or create
another one), return to this directory, and run `bash launch_headnode.sh`
again. ALF resumes from `status.txt`; use the same master configuration for
the restart.

For a fresh run, verify that `status.txt`, `h5store/data-*.h5`, and
`models/model-*` do not exist before launching. The default
`master_config.json` generates 50 GPU4PySCF bootstrap labels. Set
`ALF_GPU4PYSCF_MASTER=master_config_existing_h5.json` only when starting from
a compatible GPU4PySCF shard.

To keep the driver itself in a persistent `general`-partition allocation
instead, submit the alternate wrapper:

```bash
sbatch submit_darwin.slurm
```

Generated data, models, caches, scratch files, logs, sampling outputs, and
Parsl run information are ignored by Git.
