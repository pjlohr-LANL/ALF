# Portable CPP9 example audit — 2026-10-21

Checkout: `/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF`, branch
`production_excited_states`, starting at `5da581c`.
Source trial: `/vast/home/pjlohr/ALF_LANL/ALF_fork/production/cpp9_trial_001`.

`examples/cpp9_gpu4pyscf` packages the trial's finite workflow: label a fixed
unlabeled geometry pool, train four S0–S3 energy/force models, then stop.
It imports no historical labeled H5 shards, model artifacts, status, or logs.
The initial sampling is reused; no geometry generation or new QM was performed.
The complete example is approximately 2.24 MB, including the 2.14 MB H5 input,
and can be tracked directly without Git LFS or a data download.

## Preserved scientific inputs

The 1,000-frame H5 pool is byte-identical to the source. QM/ML configs and
optimized geometry, full-precision checkpoint, result and effective-reference
settings also match byte for byte. Input coordinates, species, topology IDs,
IDs, shuffle order, and the 500 narrow / 500 broad composition are unchanged.
`inputs/provenance.json` records source fingerprints and pool validation.
Reference manifest paths now name bundled snapshots; historical paths inside
archived records and H5 attributes are documentary and never followed at runtime.

CAM-B3LYP/6-31G RKS/TDA uses the same common optimized S0 energy subtraction,
float64 assembly, S0–S3 energies/forces, staged optical contraction, oscillator
strengths and excited–excited NAC pairs. Four 145-feature HIP-HOP members train
on E/F only, with gap targets disabled. Model hyperparameters and training
filters are unchanged. No new force or geometry screening policy is added.

The example-local QM extension retains the source trial's differing functions.
AST-equivalent shared validators, errors, device selection, unit conversion and
convergence helpers delegate to the production interface. Its task remains
local so it invokes the optical extension. No core runtime module or public QM
interface changes; the finite workflow remains separate from ALF acquisition
and the generic staged runner.

## Portability and execution

The bundled input is the CLI default. Relative paths resolve from the copied
example. ALF is discovered from the imported package rather than a user-specific
checkout path; provenance works with installed packages lacking Git metadata.
Package detection accepts supported CUDA distribution-name variants. The
launch script uses an activated environment and optional `ALF_PYTHON`.

`resources.json` supplies local/Slurm execution settings. The default local
profile retains four GPUs/workers; one-GPU hardware can set both quantities to
one while retaining four models. Slurm partitions, account/QoS, GPU directives
and environment setup are explicit site configuration. Provider construction
starts zero allocations. Scheduler GPU visibility is not overwritten.
Labeling and training run in separate Parsl sessions, releasing QM workers
before training on the same local GPUs. ML CPU threads are divided across
parallel members, preserving the source eight-thread setting at four GPUs.
The QM backend thread count follows the configured worker CPU budget.
Progress estimates use configured
worker capacity rather than a hardcoded eight-worker assumption.

The input is fixed, including failures: rejected labels are counted without
refill, and training requires nonempty filtered splits. Existing-directory
protection, locks, source/config/code/software checks, published-outcome reuse,
partial-H5 recovery and fresh model attempts are retained. Completed rejected
frames are not automatically retried. In-flight results not durably harvested
may be recomputed. A copied example supports fresh runs; migration of active
production checkpoints to another environment is outside this example.

The README documents preflight, real one-frame smoke labeling, full training,
label-only/resume, local resources, Slurm driver/worker separation, and outputs.
Generated runs, locks, caches and scheduler logs are ignored; the scientific
input H5 remains tracked.

## Validation results

| Check | Result |
|---|---|
| Focused CPP9 CPU tests in existing `alf_env` | **28 passed**, 3.62 s |
| Complete fork suite in existing `alf_env` | **473 passed, 2 skipped**, 5 existing warnings, 44.41 s |
| Complete fork suite in existing `atomistic` environment | **447 passed, 1 skipped**, 5 existing warnings, 53.89 s |
| Unchanged upstream suite with fork imports asserted | **47 passed**, 2 existing warnings, 0.88 s |
| Upstream test file integrity | All 9 modules byte-identical |
| Input, scientific config and reference-snapshot fingerprints | Matched originals |
| Relocated directory preflight and installed-ALF-without-Git identity | Passed |
| JSON/Python syntax, launch/Slurm shell syntax, diff whitespace | Passed |

The two `alf_env` skips are the existing installed PySEQM smoke tests because
that environment lacks PySEQM. They execute in `atomistic`. The one `atomistic`
skip is the CPP9 module because it lacks PySCF; all 28 CPP9 tests execute in
`alf_env`. No dependency installation was needed. An initial atomistic test
invocation disabled its user site and therefore could not locate pytest;
rerunning with its established user-site configuration passed as above.

CPP9 checks cover exact bundled input integrity, unchanged reference theory,
shape/finiteness failures, whole-frame rejection, nonfinite input avoiding QM,
force/NAC atom alignment, common-shift gap preservation, ignored auxiliary
training targets, partial publication recovery, interrupted/executor-failure
resume, and refusal of changed inputs. Resource tests mock scheduler probes;
Slurm script generation replaces submission with a fake response and validates
syntax. The relocation check copies the example to a path containing spaces
and runs preflight from another working directory. The optical contraction is
compared with the actual CPU PySCF length-gauge formula. End-to-end finite
labeling/training tests use mock tasks, including failed training followed by
a fresh successful attempt. No real model training or GPU QM is claimed.

Reproduce CPP9 tests from `/ram/tmp` using the existing environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
PYTHONPATH=/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MPLCONFIGDIR=/ram/tmp/cpp9_portable_mpl \
/vast/home/pjlohr/.conda/envs/alf_env/bin/python -m pytest \
  /vast/home/pjlohr/ALF_LANL/ALF_fork/ALF/tests/test_cpp9_example.py \
  -q -p no:cacheprovider -o addopts= \
  -o pythonpath=/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF \
  --basetemp=/ram/tmp/cpp9_reproduction
```

Use the full `tests` directory for the complete suite. Temporary full-suite
artifacts are under `/ram/tmp/cpp9_portable_release_full` and
`/ram/tmp/cpp9_portable_atomistic_full_usersite`.

The source production trial, historical data, running jobs, upstream, prototype
and software environments remain unchanged. No production calculation,
scheduler submission, or remote push was performed.
