# CPP9: portable finite GPU4PySCF labeling and ensemble training

This example packages the existing `cpp9_trial_001` calculation as tracked
inputs and code. It labels a fixed geometry pool, trains four HIP-HOP models
on S₀–S₃ energies and forces, and exits. The supplied pool contains **1,000
unlabeled C54H36 geometries**, shuffled with seed 42: 500 narrow perturbations
(phenyl rotations ±15°, rattle RMS 0–0.015 Å) and 500 broad perturbations
(±30°, 0–0.030 Å). The optimized reference is separate from that pool.

The 2.1 MB H5 input is included directly in Git. No sampling regeneration,
benchmark directory, source production directory, previous labels, or trained
models are needed. Copy this directory to another machine with the production
excited-state ALF fork installed, or clone the fork and use this example there.
Run outputs default to the ignored `runs/` directory; `--run-dir` can point
elsewhere. Moving an example enables fresh runs. Existing trial checkpoints
are not imported, and moving an active run across environments is not a
supported checkpoint migration.

## Software and hardware

Use the `production_excited_states` revision that includes this example.
Activate the desired environment before launch. It must provide this ALF fork,
NumPy, H5py, ASE, Parsl, Torch, a HIPPYNN build supporting HIP-HOP, PySCF,
GPU4PySCF, and a compatible CuPy/CUDA stack. GPU4PySCF must expose TDA gradients
and the excited–excited NAC solver used by this fork. RDKit and PySEQM are not
needed for this example. The source trial used Python 3.12, GPU4PySCF 1.8.0,
PySCF 2.14.0, CuPy 13.4.1, Torch 2.9.0, and Parsl 2026.7.20; these record the
reference installation, rather than promising compatibility with every build
bearing those version numbers.

The original driver assumed four GPUs and four QM workers per node. The
portable defaults retain those quantities and use **local workers**. On a
single-GPU computer, set `gpus_per_node` and `qm_workers_per_node` to 1 in
`resources.json`; four models will still train. Adjust CPU thread counts to
available hardware. Training CPU threads are divided across concurrent model
members (eight threads each with the reference 32-core/four-GPU setting).
Separate labeling and training Parsl sessions release QM
workers before model training starts. A login node is not an appropriate place
to run the local GPU workflow; use an allocated workstation/node.

## Configure resources

`resources.json` controls execution independently of the scientific settings:

- `backend`: `local` or `slurm`.
- `gpus_per_node`, `qm_workers_per_node`, `qm_cores_per_worker`,
  `ml_cores_per_worker`: actual worker resources. The effective QM thread count
  follows `qm_cores_per_worker`, overriding the reference config’s thread count
  while leaving the electronic-structure method unchanged.
- `worker_init`: optional shell commands activating the worker environment and
  loading site CUDA modules. No environment path or module name is hardcoded.
- `account`, `qos`, and the `QM` / `ML` partition, walltime, CPU, memory,
  scheduler-options and block limits: editable Slurm settings. Local mode
  uses one block per role and ignores Slurm-only requests.

For Slurm set `backend` to `slurm` and configure **both partitions**. Set GPU
requests using your site's syntax in each role's `scheduler_options`. The
source trial used `shared-gpu-ampere` for labeling, `ml4chem` for training,
`y2020-bf` / `long`, with `--constraint=gpu_count:4` for QM and
`--constraint=gpu_count:4&gpu1_memory:48GB` for training. These are historical
examples; the tracked config makes no site-specific request by default.
Parsl providers start zero blocks until work is submitted. Worker paths must
be visible on the allocated nodes, or equivalent paths/software must be
provided through the worker environment. Scheduler GPU visibility is respected.

## Validate, label, train, and resume

From this directory, with the environment active:

```bash
bash launch.sh --check
bash launch.sh --run-dir "$PWD/runs/smoke" --limit 1 --label-only
bash launch.sh --run-dir "$PWD/runs/trial_001"
```

`--check` validates the input, reference hashes/theory/energy offset, model
config, resources, and installed package versions. It starts no Parsl workers
and creates no run outputs. It checks software metadata, rather than executing
CUDA. The one-frame labeling smoke command is a real GPU calculation; inspect
its outcome before launching the 1,000-frame campaign. One frame is too few
for a train/validation/test split, hence `--label-only`.

`--input-h5 FILE` accepts an alternative compatible pool. `--resources FILE`
selects another resource config. Defaults resolve from the example's location,
so launching from another working directory is supported. `ALF_PYTHON` can
select a Python executable for `launch.sh`. The example also works with:

```bash
python /path/to/copied/cpp9_gpu4pyscf/run_trial.py --check
```

For a Slurm driver, configure resources, activate the driver environment, and
submit **from this directory**:

```bash
sbatch --partition=YOUR_DRIVER_PARTITION --account=YOUR_ACCOUNT \
    submit_driver.slurm --run-dir "$PWD/runs/trial_001"
```

Provide your site's driver QoS if needed. If submitting from elsewhere, export
`CPP_EXAMPLE_DIR=/path/to/copied/cpp9_gpu4pyscf`. The driver allocation and
Parsl worker allocations are separate. The driver must outlive its workers.
For local mode, launch within a GPU allocation instead of using this CPU-only
driver template. No command in the validation tests submits actual jobs.

To label first and train later, launch with `--label-only`, then resume without
it. Resume requires the same inputs, limit, scientific/resource settings,
recorded software identity, and example code:

```bash
bash launch.sh --run-dir "$PWD/runs/trial_001" --resume
```

Completed accepted/rejected outcomes are reused. Recorded QM rejections are
not automatically retried or replaced. Unharvested work can be recomputed after
an interruption. A run lock prevents competing drivers. SIGINT/SIGTERM stops
new submissions, preserves published outcomes, and cleans up Parsl; it does
not guarantee that every in-flight result is harvested. Interrupted training
artifacts remain, and resume starts a new model attempt. A completed ensemble
is not retrained on resume. Runtime identity supports installed ALF packages
without a Git checkout; compatible source files are hashed as well.

## Scientific settings and outputs

The reference is the converged CPP9 S₀ CAM-B3LYP/6-31G, neutral-singlet,
gas-phase optimization from benchmark `17250216`. Four-state labels use
RKS/TDA with density fitting and the saved reference tolerances. All state
energies subtract the **same** optimized S₀ reference:
**−2077.702199766316 Hartree = −56537.15659244438 eV** using the labeling
backend's conversion. Subtraction occurs in float64; forces and energy gaps
retain their original meaning. Reference snapshots and hashes are bundled.
Historical paths appearing inside archived metadata are provenance only;
execution never follows them.

Each calculation returns:

| Quantity | Stored units | Trained |
| --- | --- | --- |
| `sE0`–`sE3` | shifted eV | yes |
| `F0`–`F3` | eV/Å, negative energy gradients | yes |
| `absolute_state_energies` | eV | no |
| `transition_dipoles` | e·bohr, S₀→S₁/S₂/S₃ | no |
| `oscillator_strengths` | dimensionless, length gauge | no |
| `nacr` / `nac_pairs` | Å⁻¹, pairs (1,2), (1,3), (2,3) | no |

NACs retain the solver's arbitrary signs. The example-local QM extension
preserves the trial's staged optical contraction and same-calculation NAC
path; it reuses ALF validation, device selection, and unit conversion. It does
not modify the core driver or public GPU4PySCF interface.

Training uses four HIP-HOP models, 145 features, 20 sensitivities, two
interaction layers, three atom layers, 80/10/10 splits, seed 42, and at most
300 epochs. Gap targets are disabled. Existing trainer filters remain active
(minimum distance 0.7 Å); force/energy outlier filters are disabled as in the
trial. No additional sampling-time or post-QM force screen is introduced.
Every requested label must be finite and correctly shaped for the whole frame
to be accepted. Failed frames reduce the retained count; the workflow attempts
the fixed input pool rather than refilling to a success target. Training starts
only if retained, filtered data support nonempty splits.

`request.json` pins input/config/code/software identity. `outcomes/` records
per-frame success/failure. Accepted labels are atomically published beneath
`h5store/`, with coordinates, forces, NACs and topology IDs permuted together
under upstream storage ordering. `progress.json` reports counts, label
statistics and approximate throughput. `training.json` and `models/` record
training attempts. No active-learning dynamics is run by this finite example.

The geometry pool describes local perturbations around one optimized anchor;
it is not a Boltzmann sample or a complete conformational search. The separate
ALF staged-workflow examples cover acquisition and ALF→GALF continuations.

CPU regression tests are in `tests/test_cpp9_example.py` at the repository
root. They use mocked labeling/training and a real CPU PySCF optical comparison;
GPU production calculations are not part of those tests.
