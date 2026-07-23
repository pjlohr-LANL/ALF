# Excited-state PySEQM active learning

This example connects ALF's standard CFG loader, multi-state HIPPYNN trainer,
batched ALCHEMI sampler, and convergence-safe PySEQM interface into one
head-node active-learning process. Parsl dynamically requests separate Darwin
Slurm allocations for training, sampling, and QM labeling.

Two startup modes are included:

```text
CFG -> PySEQM bootstrap -> HDF5 -> HIPPYNN -> ALCHEMI sampling
```

```text
Existing HDF5 -> HIPPYNN
CFG -> ALCHEMI sampling -> PySEQM -> additional HDF5 shards
```

The included water system and two electronic states make this an integration
example. The thresholds, network size, data volume, and trajectory length are
starting values, not a scientifically converged water model.

## What CFG contributes

`simple_cfg_loader_task` reads atomic identity, order, coordinates, and cell
data from `fragment_library/water.cfg`. ASE's AtomEye CFG reader always marks
CFG structures periodic, so both builder configurations explicitly set
`"pbc": false` after loading.

CFG auxiliary arrays are not used as multi-state labels. Energies and forces
enter ALF through PySEQM or through HDF5 datasets selected by
`properties_list`:

```json
{
  "sE0": ["state_0_energy", "system", 1.0],
  "F0": ["state_0_forces", "atomic", 1.0],
  "sE1": ["state_1_energy", "system", 1.0],
  "F1": ["state_1_forces", "atomic", 1.0],
  "dE01": ["gap_01", "system", 1.0]
}
```

PySEQM labels both states and derives `dE01 = sE1 - sE0`. The HIPPYNN trainer
uses one shared trunk with state-specific energy/force heads and a derived gap
loss.

## Files

| File | Purpose |
| --- | --- |
| `master_config.json` | Self-contained PySEQM bootstrap workflow |
| `master_config_existing_h5.json` | Start from existing labeled HDF5 shards |
| `master_config_debug.json` | Batch-size-one stage checks |
| `builder_config_bootstrap.json` | CFG loading with a small geometry shake |
| `builder_config_existing_h5.json` | Exact CFG reuse with `shake: 0.0` |
| `sampler_config.json` | Two-state strict batches of 50 |
| `sampler_config_debug.json` | One-replica debug sampler |
| `QM_config.json` | Single-molecule PySEQM settings |
| `hippynn_config.json` | Multi-state HIPPYNN settings |
| `parsl_configs.py` | Dynamically scaling Darwin Slurm executors |

The XYZ file is the fixed-topology reference. It has the same `O, H, H` atom
order as the CFG file.

## Darwin resource setup

`parsl_configs.config_darwin` defines three independent executor pools:

| ALF stage | Executor | Darwin partition | Workers per node |
| --- | --- | --- | --- |
| HIPPYNN training | `alf_ML_executor` | `ml4chem` | 1 |
| ALCHEMI sampling and CFG loading | `alf_sampler_executor` | `shared-gpu-ampere` | 4 |
| PySEQM labeling | `alf_QM_executor` | `shared-gpu-ampere` | 4 |

All providers use `init_blocks=0` and `min_blocks=0`. The main ALF process can
therefore remain on a head node while Parsl requests allocations only when a
queue contains work. Sampling and QM use separate allocations even though
they target the same partition.

Before a real run, review the constants at the top of `parsl_configs.py` or
set the corresponding environment variables:

```bash
export ALF_DARWIN_ACCOUNT="your_account"
export ALF_DARWIN_ENV_ACTIVATION='source /path/to/conda.sh; conda activate /path/to/atomistic'
export ALF_DARWIN_ML_SCHEDULER_OPTIONS='#SBATCH --gpus-per-node=4'
export ALF_DARWIN_SAMPLER_SCHEDULER_OPTIONS='#SBATCH --gpus-per-node=4'
export ALF_DARWIN_QM_SCHEDULER_OPTIONS='#SBATCH --gpus-per-node=4'
# Optional; otherwise each allocation uses SLURM_TMPDIR or TMPDIR.
export ALF_DARWIN_CACHE_ROOT='/path/to/writable/node-local/cache'
```

Account, QoS, walltime, block limits, CUDA module command, and GPU request
syntax are deliberately configurable. Worker initialization creates writable
Warp and Matplotlib caches in Slurm or process-local temporary storage.

From the repository root, install the ALCHEMI and topology extras, and install
LANL PySEQM in the same worker environment:

```bash
python -m pip install -e ".[gpu_dynamics,topology]"
```

## Mode 1: bootstrap from CFG

Run from this directory:

```bash
cd examples/excited_state_pyseqm
python -m alframework --master master_config.json
```

With no existing `status_bootstrap.txt` or bootstrap HDF5 shards, ALF:

1. Loads and perturbs water CFG geometries.
2. Queues 500 single-molecule PySEQM calculations.
3. Writes `h5store_bootstrap/data-0000.h5`.
4. Trains `models_bootstrap/model-0000`.
5. Starts strict two-state ALCHEMI sampling.
6. Labels uncertain candidates and retrains after the configured threshold.

The small shake applies only to the bootstrap and sampler starting geometry.
It does not import or invent labels.

## Mode 2: start from existing HDF5

Place compatible shards at:

```text
h5store/data-0000.h5
h5store/data-0001.h5
...
```

Do not create `status.txt` for the first launch. Then run:

```bash
python -m alframework --master master_config_existing_h5.json
```

ALF discovers the first unused HDF5 index. Because labeled data already
exists, it skips the PySEQM bootstrap stage, trains the initial model from the
shards, and uses exact CFG geometries (`shake: 0.0`) as sampling seeds.

Existing shards must contain finite coordinates, an identical atomic-number
sequence, and the configured database names:

```text
state_0_energy
state_0_forces
state_1_energy
state_1_forces
gap_01
```

The gap values must equal `state_1_energy - state_0_energy`, and all three
system properties must use compatible scaling.

## Stage checks

The debug master selects the short Darwin executor configuration and the
batch-size-one sampler required by `--test_sampler`:

```bash
python -m alframework --master master_config_debug.json --test_builder
python -m alframework --master master_config_debug.json --test_qm
```

`--test_ml` requires an HDF5 shard, and `--test_sampler` requires a completed
model. Run them after the bootstrap has produced the corresponding artifacts:

```bash
python -m alframework --master master_config_debug.json --test_ml
python -m alframework --master master_config_debug.json --test_sampler
```

The test commands use the same bootstrap HDF5 and model paths as
`master_config.json`.

## Sampling behavior

The production sampler forms full batches of 50. Consecutive molecule-ID
blocks alternate between state 0 and state 1, so `parallel_samplers: 200` can
fill four state-specific batches. Parsl assigns those tasks to arbitrary
sampler GPUs; electronic state is not tied to GPU number.

The baseline uses production-compatible stop-on-uncertainty, temperature
scheduling, friction, and spherical-well behavior. Gap diagnostics are
recorded, while LCM gap seeking is configured but disabled. After the direct
dynamics workflow is validated, set `gap_seeking.enabled` to `true` to test
the per-replica stay-fixed LCM path.

Topology is checked before dynamics and at every uncertainty synchronization.
The water-specific `distcut` is below the reference O-H bond length. Invalid
topologies and close-contact frames are discarded before they can consume
candidate or QM slots.

PySEQM remains one molecule per Parsl task. SCF nonconvergence, malformed
convergence flags, timeouts, and nonfinite results mark the molecule
unconverged and prevent partial labels from entering HDF5.

## Run outputs and restarts

Generated HDF5 stores, model directories, status files, PySEQM scratch, logs,
and Parsl `runinfo` directories are run artifacts and should not be committed.
ALF resumes from the configured status file. To intentionally start a new run,
use new output paths or archive the previous artifacts rather than mixing
independent histories.
