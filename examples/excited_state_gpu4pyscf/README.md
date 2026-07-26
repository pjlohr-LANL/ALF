# Six-state keto GPU4PySCF active learning

This example labels keto acetylacetone with GPU4PySCF and trains the same
six-surface HIPPYNN/ALCHEMI workflow used by the excited-state PySEQM example.
The GPU4PySCF bootstrap reads geometries from the PySEQM example, but it does
not reuse any PySEQM labels. The examples have independent QM configurations,
output HDF5 shards, models, status files, scratch directories, and worker
environments.

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
- energy offset `-9405.0 eV`, so stored energies are
  `E_raw - (-9405.0 eV)`

The task requires SCF and every requested root to converge before it stores
any label. Transition dipoles, NACVs, state tracking, full TDDFT, PBC, CPU
fallback, and QM batching are not part of this example.

## Startup modes

### Exhaustive geometry-relabel bootstrap

`master_config.json` follows:

```text
4,990 geometries from ../excited_state_pyseqm/h5store/data-0000.h5
  -> discard all PySEQM labels
  -> 4,990 independent GPU4PySCF labeling attempts
  -> data-0000.h5
  -> six-state HIPPYNN
  -> state-cycled ALCHEMI
  -> additional GPU4PySCF labels and retraining
```

The local replay builder reads source frames `0` through `4989` sequentially
in batches of at most 50. It reads geometry, species, topology atom IDs, and
source identifiers only; it never returns the source `sE*` or `F*` datasets.
GPU4PySCF stores only converged, finite, topology-valid results in this
example's own `h5store/data-0000.h5`, so the final training count can be below
4,990 if any labeling attempts are rejected.

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
export ALF_DARWIN_SAMPLER_QOS="..."
export ALF_DARWIN_SAMPLER_WALLTIME="12:00:00"
export ALF_DARWIN_GPU4PYSCF_QOS="..."
export ALF_DARWIN_GPU4PYSCF_SCHEDULER_OPTIONS="..."
export ALF_DARWIN_GPU4PYSCF_WALLTIME="08:00:00"
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

## Production driver on Darwin

The preferred production launch is the persistent Slurm wrapper:

```bash
cd /vast/home/pjlohr/ALF_LANL/ALF_fork/ALF/examples/excited_state_gpu4pyscf
sbatch submit_darwin.slurm
```

It requests one CPU-only driver task on `ml4chem` with account `y2020-bf`,
`qos=long`, and a two-day walltime. The driver allocation does not request a
GPU. Parsl independently requests the training, sampling, and QM allocations
described above, so those workers retain their separate partitions,
constraints, and walltimes.

For an interactive alternative, run the same driver in `tmux` on a Darwin
frontend:

```bash
hostname
tmux new -s alf_gpu4pyscf
cd /vast/home/pjlohr/ALF_LANL/ALF_fork/ALF/examples/excited_state_gpu4pyscf
bash launch_headnode.sh
```

Detach without stopping the driver with `Ctrl-b d`. Darwin has multiple
frontend nodes, and each frontend has its own local tmux server. Record the
hostname printed above. To find and reattach the session, reconnect to that
exact frontend, not merely whichever frontend a new login selects:

```bash
ssh darwin-fe1
hostname
tmux ls
tmux attach -t alf_gpu4pyscf
```

Replace `darwin-fe1` with the frontend recorded when the session was created.
A session on `darwin-fe1` is not visible to `tmux ls` on another frontend.

### Restart after a driver crash

Only one ALF driver may control an example directory. Before restarting,
confirm that the old driver is gone and identify any Parsl allocations it
left behind:

```bash
squeue -u "$USER" \
  -o "%.18i %.16P %.40j %.8T %.10M %.20R"
```

Cancel only the explicit job IDs belonging to orphaned `parsl.alf_*`
allocations from the failed driver. Never use a user-wide cancellation:

```bash
scancel JOBID1 JOBID2
```

Live builder, sampler, QM, and ML queues exist only in the driver process and
are not reconstructed from `runinfo`. Results from sampling or QM tasks that
were still in flight are therefore lost when the driver crashes. Preserve
`status.txt`, every HDF5 shard, promoted models, the failed `runinfo/NNN`,
and its scheduler logs; these are restart evidence and scientific state, not
cleanup targets.

Validate the persisted checkpoint before resubmitting:

```bash
cd /vast/home/pjlohr/ALF_LANL/ALF_fork/ALF/examples/excited_state_gpu4pyscf
python -m json.tool status.txt >/dev/null
python - <<'PY'
import glob
import json

with open("status.txt", encoding="utf-8") as handle:
    status = json.load(handle)
print(json.dumps(status, indent=2))
print("HDF5 shards:", sorted(glob.glob("h5store/data-*.h5")))
print("model paths:", sorted(glob.glob("models/model-*")))
PY
```

`current_h5_id` must identify the next shard to be written, and
`current_model_id` must identify the newest fully promoted four-member model.
`current_training_id` is advanced when training is submitted, not when it
finishes. Consequently, a crash during model training requires manual
inspection of the status IDs and the possibly partial `models/model-NNNN`
directory before restart. Do not delete, renumber, or treat that directory as
promoted without reconciling it.

Once the checkpoint and directories agree, restart with the same master
configuration:

```bash
sbatch submit_darwin.slurm
```

The new driver creates the next `runinfo` directory and resumes from
`status.txt`. Do not simultaneously run `launch_headnode.sh` in tmux.

For a genuinely fresh run, in contrast, `status.txt`, output
`h5store/data-*.h5`, and `models/model-*` must not exist. The default
`master_config.json` exhaustively relabels the 4,990 source geometries.
Set `ALF_GPU4PYSCF_MASTER=master_config_existing_h5.json` only when starting
from a compatible, already labeled GPU4PySCF shard.

Generated data, models, caches, scratch files, logs, sampling outputs, and
Parsl run information are ignored by Git.
