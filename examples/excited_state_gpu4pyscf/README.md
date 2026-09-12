# Seeded excited-state GPU4PySCF active learning

This example starts from an existing, labeled HDF5 dataset. ALF trains an
excited-state HIPPYNN ensemble from that seed, samples new structures with
ALCHEMI, labels them with GPU4PySCF, and retrains as new HDF5 shards are
accepted.

The checked-in files provide a concrete keto acetylacetone calculation with
five excited roots. The workflow is not limited to six total states: if
`nroots` is `N`, GPU4PySCF returns the ground state plus `N` excited states,
numbered `0` through `N`.

## Workflow

```text
user-provided h5store/data-0000.h5
  -> initial HIPPYNN ensemble in models/model-0000
  -> deterministic replay of seed geometries
  -> state-selected ALCHEMI sampling
  -> one GPU4PySCF calculation per molecule and GPU
  -> screened labels in h5store/data-0001.h5
  -> models/model-0001
  -> repeat
```

Every GPU4PySCF task calculates all requested states for one molecule on one
GPU. States are not distributed across GPUs. On a four-GPU node, Parsl can
run four molecules concurrently.

## Checked-in calculation

The supplied files use:

- keto acetylacetone with 15 atoms and no periodic boundary conditions;
- density-fitted CAM-B3LYP/6-31G* RKS;
- neutral singlet charge and multiplicity;
- grid level 3;
- `nroots: 5`, giving states 0 through 5;
- energies `sE0` through `sE5` and forces `F0` through `F5`;
- an energy offset of `-9405.0 eV`, meaning stored energies are
  `E_raw - (-9405.0 eV)`; and
- four HIPPYNN ensemble members.

SCF and every requested TDA root must converge before a molecule is accepted.
The interface does not store partial state results. Full TDDFT, state
tracking, transition properties, NACVs, periodic systems, CPU fallback, and
multi-molecule QM batches are outside this example.

## Before you begin

You need:

1. An ALF environment containing the normal ALF, HIPPYNN, ALCHEMI, ASE,
   Parsl, NumPy, and HDF5 dependencies.
2. A GPU4PySCF installation compatible with the CUDA runtime on the QM
   workers. For the supplied CUDA 12 profile, this is typically a
   `gpu4pyscf-cuda12x` distribution.
3. Access to Darwin Slurm partitions, or a replacement `parsl_configs.py`
   for your machine.
4. A labeled seed HDF5 that matches the molecule, electronic-structure
   method, state count, units, and energy offset configured here.

Do not use a seed labeled with a different QM method merely because its
dataset names match. Legacy ALF HDF5 does not record enough method provenance
to detect that mistake automatically.

## Prepare the seed HDF5

From this directory, copy your seed into the first ALF shard:

```bash
mkdir -p h5store
cp /path/to/your/gpu4pyscf_seed.h5 h5store/data-0000.h5
sha256sum h5store/data-0000.h5
```

Record the source path, checksum, electronic-structure settings, and any data
conversion used to create the file in your run notes. This example does not
provide or prescribe a universal seed checksum.

### Required HDF5 data

For `nroots: N`, every molecular group in the seed must contain:

| Dataset | Required shape | Units or meaning |
| --- | --- | --- |
| `coordinates` | `[structures, atoms, 3]` | Angstrom |
| `species` | `[atoms]` or `[structures, atoms]` | Symbols or positive atomic numbers |
| `sE0` ... `sEN` | `[structures]` | eV after the configured common offset |
| `F0` ... `FN` | `[structures, atoms, 3]` | eV/Angstrom |
| `_id` | `[structures]` | Optional source identifiers |
| `topology_atom_ids` | `[atoms]` or `[structures, atoms]` | Optional canonical atom-index mapping |

All required arrays must be finite and have the same number of structures.
Every structure must have the same atom count and atomic-number sequence.
ALF-written files store atoms in stable atomic-number order. A foreign seed
must either use that order or provide valid `topology_atom_ids` so replay can
restore the order in the topology reference.

The configured 10% validation and 10% test splits require at least ten usable
structures after filtering. A practical seed should be substantially larger
and representative of the intended sampling region.

You can inspect dataset names, shapes, data types, and finiteness without
starting a GPU job:

```bash
python - <<'PY'
import h5py
import numpy as np

path = "h5store/data-0000.h5"
with h5py.File(path, "r") as handle:
    def report(name, obj):
        if isinstance(obj, h5py.Dataset):
            finite = "n/a"
            if np.issubdtype(obj.dtype, np.number):
                finite = bool(np.isfinite(obj[...]).all())
            print(name, obj.shape, obj.dtype, "finite:", finite)
    handle.visititems(report)
PY
```

Before launch, confirm that the seed energies use the same offset convention
as `QM_config.json`. New GPU4PySCF labels and existing seed labels must be on
one consistent energy scale.

## Choose the number of states

The state count is controlled by several files that must agree. To use `N`
excited roots:

1. Set `QM_config.json:nroots` to `N`. The current interface requires
   `N >= 1` and produces `N + 1` total surfaces.
2. In `master_config.json:properties_list`, define every contiguous pair
   `sE0/F0` through `sEN/FN`. Energies use scope `system`; forces use scope
   `atomic`.
3. Supply the same `sE0/F0` through `sEN/FN` datasets in every seed group.
4. Set `sampler_config.json:state_selection.states` to the states that should
   drive sampling. To cycle over every surface, use `0` through `N`.

The excited-state HIPPYNN trainer constructs its output heads from
`properties_list`; there is no separate `n_states` setting in
`hippynn_config.json`.

## Optional: summed candidate scoring

`sampler_config.json` uses the default scoring, which ranks candidates by the
normalized worst violation `max(Es/Escut, Fs/Fscut, Fsmax/(3*Fscut))`. Because
it also uses `uncertainty_policy: "stop"`, each replica freezes at its first
uncertain frame, so the score is recorded but never actually selects anything.

`sampler_config_summed_score.json` is an alternative that ranks by a weighted
sum instead, letting you state how much energy, force, and state-gap
disagreement each matter:

```json
"score": {
  "mode": "sum",
  "w_energy": 1.0,
  "w_force": 1.0,
  "w_gap_uncertainty": 5.0,
  "gap_std_cut_eV": 0.01,
  "gap_pairs": "selected_adjacent"
}
```

Every term is divided by its own cutoff, so the weights carry no units and mean
the same thing if you change molecule or thresholds. The gap term needs no
change to `properties_list`, the HDF5 contract, or the QM interface: the gap
deviation is derived by subtracting per-member state energies the ensemble
already predicts.

The remaining settings change together with it, and the change is only meaningful
as a group. `max_candidates_per_replica` is new rather than changed:

| Setting | Default config | Summed-score config | Reason |
| --- | --- | --- | --- |
| `uncertainty_policy` | `stop` | `continue` | A score can only rank when replicas keep running and produce competing frames. |
| `Ncheck` | 10 | 10 | Unchanged. `continue` at `dt: 0.1` fs therefore checks every 1 fs, 2000 times per replica. Redundancy is controlled by the per-replica cap below rather than by check spacing. |
| `return_top_k` | 50 | 100 | Per-task QM budget. Candidates are cheaper and more numerous under `continue`. |
| `max_candidates_per_replica` | unset | 10 | Without a cap, one diverging trajectory can fill every returned slot. |
| `save_h5_threshold` (master) | 50 | 500 | Larger, less frequent HDF5 shards to match the per-task budget. |
| `ML_config_path` (master) | `hippynn_config.json` | `hippynn_config_summed_score.json` | Separate network settings, so this example does not alter the default one. |

The cap and `return_top_k` interact: with 10 and 100, filling the budget needs a
minimum of ten distinct trajectories, out of the 50 in a batch. Lower the cap for
a stronger diversity floor — 5 forces at least 20 trajectories, and 2 forces all
50 to contribute their two best frames each.

### Relationship to the PySEQM production runs

These values are aligned with the validated prototype campaign in
`ALF/production/Enol_BALF_4_states_EF_300k_topology`, so results are roughly
comparable. Most parameters already agreed: `dt`, `maxt`, `min_time`, friction,
temperatures, `min_distance_cutoff`, `max_force_cutoff`, the topology bond scales,
screening flags, and cyclic state selection.

Two things worth understanding:

- **The energy and force weights already match production.** That run sums raw
  units (`1.0*uE + 0.1*uF`, eV and eV/Angstrom) while this one sums
  cutoff-normalized ratios. Since `Escut` and `Fscut` already encode a 10x ratio,
  production's extra `0.1` on force cancels exactly, and `w_energy = w_force = 1.0`
  gives a ranking proportional to production's. Its uncertainty gate
  (`min_uE` 0.001, `min_uF` 0.01) is likewise identical to `Escut` and `Fscut`.
- **Two deliberate divergences.** Production sets `w_gap_uncertainty` to `0.0`;
  this config keeps `5.0` so the gap channel is actually exercised. And production
  uses `return_top_n: 500` with a 2000-structure shard threshold, which is
  affordable for semiempirical PySEQM but not for CAM-B3LYP TDA, so both numbers
  are scaled down by five here.

### Cost

`Ncheck: 10` under `continue` means no replica freezes early, so every one runs all
2000 checks. Measured on one A100, 300 checks of a 50-replica batch took 18
minutes, which puts a full sampler task near **two hours**. The per-step cost is
inherent to the excited-state calculator: it evaluates all six states for all four
ensemble members at every MD step regardless of which surface is selected. The
default `stop` configuration never pays this, because replicas freeze at their
first uncertain frame.

`return_top_k: 100` also queues up to 100 CAM-B3LYP TDA calculations per sampler
task. For reference, a four-GPU node labels roughly 500 structures per hour, so a
500-structure shard is about an hour of QM.

Because `hippynn_config_summed_score.json` adopts the production network
(`n_features` 145, `n_sensitivities` 20), ensembles trained here are not
architecturally comparable to any trained with the default config's 75 and 63.
Ensemble spread, and therefore score magnitudes, will differ between the two.

Run it with the matching master config:

```bash
python -m alframework --master master_config_summed_score.json
```

Use a separate run directory from the default configuration. The two produce
different datasets and must not share `status.txt`, `h5store/`, or `models/`.

## Adapt the example to another molecule

Change all coupled inputs before reusing the workflow:

- Replace `keto_form_coords.xyz` with an ASE-readable reference geometry in
  the canonical atom order. Update `reference_conformer_path` and
  `reference_charge` in `sampler_config.json` if its name or charge changes.
- Set `hippynn_config.json:n_atoms` to the fixed atom count. Update
  `network_params.possible_species`; it must start with padding species `0`
  and include every atomic number in the seed.
- Review HIPPYNN distance cutoffs and sampling temperature/time parameters for
  the new chemistry.
- Set the QM functional, basis, charge, multiplicity, roots, convergence
  controls, and energy offset in `QM_config.json`. This interface supports
  singlet RKS/TDA only.
- Update topology, minimum-distance, and maximum-force screening thresholds
  in `sampler_config.json`.
- Ensure the seed atom ordering, labels, units, and provenance match all of
  those choices.

This is a fixed-composition workflow: do not combine groups with different
atom counts or atomic-number sequences in one run.

## Configure Darwin

`parsl_configs.py` creates independent, dynamically scaling executors:

| Stage | Partition | Workers/node | GPUs/worker | Maximum blocks |
| --- | --- | ---: | ---: | ---: |
| HIPPYNN training | `ml4chem` | 1 | ensemble-managed | 1 |
| ALCHEMI sampling | `shared-gpu-ampere` | 4 | 1 | 2 |
| GPU4PySCF labeling | `shared-gpu-ampere` | 4 | 1 | 2 |

The checked-in defaults use account `y2020-bf`, CUDA `12.2.2`, the four-GPU
node constraint, and `/vast/home/pjlohr/.conda/envs/alf_env`. At minimum,
review the paths and account in `submit_darwin.slurm` and set environment
overrides before submitting:

```bash
export ALF_GPU4PYSCF_EXAMPLE_DIR=/absolute/path/to/ALF/examples/excited_state_gpu4pyscf
export ALF_REPOSITORY_ROOT=/absolute/path/to/ALF
export ALF_DRIVER_CONDA_ENV=/absolute/path/to/your/alf/environment
export ALF_DARWIN_ACCOUNT=your_account
```

Additional `ALF_DARWIN_*` variables control QoS, scheduler directives,
walltimes, block limits, worker environments, and cache roots. If GPU4PySCF
is imported from a source checkout, set:

```bash
export ALF_DARWIN_GPU4PYSCF_PYTHONPATH=/path/containing/gpu4pyscf
```

For another cluster, replace `parsl_configs.py` and the Slurm wrapper with
site-appropriate executors and launch settings while retaining the executor
labels used by the master tasks.

## Check the setup

Run the component tests from the repository root:

```bash
export PYTHONPATH=/absolute/path/to/ALF:${PYTHONPATH:-}
python -m pytest -q \
  tests/test_gpu4pyscf_interface.py \
  tests/test_h5_replay_builder.py \
  tests/test_excited_state_hippynn_interface.py
```

After placing the seed, check replay from the example directory:

```bash
python -m alframework --master master_config.json --test_builder
```

Then run one real GPU4PySCF label through the configured Darwin executors:

```bash
python -m alframework --master master_config.json --test_qm
```

The QM check writes `qm_test.h5`. Confirm that all configured `sEi` and `Fi`
arrays are present, finite, and have the expected shapes. Compare at least one
geometry with an independent calculation using identical QM settings before
committing significant compute time.

The ALF test modes also create or update `status.txt`. A status created by
these checks in this clean directory is safe to continue from; never copy a
status file from another run.

`--test_ml` performs actual ensemble training rather than a quick syntax
check. The normal seeded launch performs that training as its first stage.

## Start the run

For a fresh run, the directory must contain:

```text
h5store/data-0000.h5
```

It must not contain an old `status.txt` or `models/model-*`. Choose exactly
one of the following driver launch methods.

### Slurm driver (recommended)

Submit the persistent CPU-only ALF driver:

```bash
sbatch submit_darwin.slurm
```

The wrapper requests one driver task on `ml4chem` and writes
`ALF_GPU4PYSCF_KETO_JOBID.log` and `.err`. The driver allocation does not
request a GPU. Parsl independently submits the training, sampling, and QM
worker allocations configured in `parsl_configs.py`. The driver continues
after you disconnect from Darwin.

### Headnode driver in tmux or screen

As an interactive alternative, run `launch_headnode.sh` inside a persistent
terminal session on a Darwin frontend. With `tmux`:

```bash
hostname
tmux new -s alf_gpu4pyscf
cd /absolute/path/to/ALF/examples/excited_state_gpu4pyscf
bash launch_headnode.sh
```

Detach without stopping ALF with `Ctrl-b d`. Record the hostname printed
before starting the session. Each Darwin frontend has its own local tmux
server, so reconnect to that exact frontend before reattaching:

```bash
ssh darwin-fe1
tmux ls
tmux attach -t alf_gpu4pyscf
```

Replace `darwin-fe1` with the recorded hostname. A session created on one
frontend will not appear in `tmux ls` on another frontend.

GNU `screen` can be used instead:

```bash
hostname
screen -S alf_gpu4pyscf
cd /absolute/path/to/ALF/examples/excited_state_gpu4pyscf
bash launch_headnode.sh
```

Detach with `Ctrl-a d`. After reconnecting to the same frontend, use
`screen -ls` and `screen -r alf_gpu4pyscf` to reattach.

In both interactive cases, the headnode process is only the ALF driver;
training, sampling, and GPU4PySCF calculations are still submitted to Slurm
by Parsl. Do not also submit `submit_darwin.slurm` for the same run directory.

The driver discovers that HDF5 index 0 already exists, sets the next HDF5 id
to 1, and skips QM bootstrap labeling. It first trains
`models/model-0000`, then begins replay, sampling, GPU4PySCF labeling, and
retraining. With the checked-in `save_h5_threshold: 50`, the first accepted
batch is written to `h5store/data-0001.h5` before `model-0001` is trained.

The first useful acceptance milestones are:

1. Every member of `models/model-0000` completes.
2. Sampling visits the configured state list.
3. QM worker metadata reports the GPU4PySCF backend and one CUDA device per
   molecule.
4. Screening accepts candidates into `h5store/data-0001.h5`.
5. Training begins for `models/model-0001`.

## Monitor and restart

Use Slurm to identify the driver and its independent Parsl allocations:

```bash
squeue -u "$USER" -o "%.18i %.16P %.40j %.8T %.10M %.20R"
```

Only one ALF driver may control this directory. If the driver stops, preserve
`status.txt`, every HDF5 shard, completed models, and `runinfo`. Cancel only
explicit orphaned Parsl job IDs from that driver, inspect the persisted state,
and resubmit the same command:

```bash
sbatch submit_darwin.slurm
```

If the driver was launched in `tmux` or `screen`, restart it with
`bash launch_headnode.sh` inside a new persistent session instead. Do not use
both launch methods simultaneously.

ALF reads `status.txt` and resumes. Live queues from a failed driver are not
reconstructed, so sampling or QM tasks still in flight at failure time must be
submitted again by the resumed workflow.

For a genuinely new scientific run, use a separate clean directory containing
only its own `h5store/data-0000.h5`. Never mix seeds, later shards, models, or
status files from runs with different molecules, state contracts, methods, or
energy offsets.
