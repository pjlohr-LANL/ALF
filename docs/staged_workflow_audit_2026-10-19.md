# Generic staged-workflow audit — 2026-10-19

Implementation checkout: `/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF`,
branch `production_excited_states`. Starting commit:
`29f55143dfb7c9312c4bd5d7f58309699d56875b`.
Pinned upstream main: `a5bb49e95fdfa458bc11700631e076b067483130`.

## Review units

1. **`741f9cf` — lifecycle and storage.** Opt-in driver delegation, durable task
   submissions/results, explicit ensemble success, transactional H5 publication,
   exact generic snapshots, locks, and recovery regressions. The independent
   foundation check passed **15 tests**, including the existing three actual
   driver scenarios and tools dependency checks.
2. ****`e00b95a`** — orchestration and recovery.** Linear stage materialization,
   immutable configuration and label compatibility, verified handoff, explicit
   resume/status/validation, graceful drain/pause, and exact final training.
   Additional inherited-unit consistency checks reject conflicting declarations.
3. **Examples and documentation.** ALF-only and ALF→GALF keto definitions,
   example contracts, user-guide entries, complete CLI/recovery documentation,
   and this audit.

The earlier manual keto preparation/handoff/snapshot commands remain available.
[Their audit](keto_rdkit_alf_galf_audit_2026-10-19.md) records that implementation
separately. Seed preparation remains an explicit prior step.

## Behavior and boundaries

Each stage references ordinary master/builder/sampler/QM/ML/Parsl configuration.
The runner owns budgets, manifests, transitions, and recovery; the core driver
only delegates optional lifecycle boundaries. Sampling strategies and scientific
policies remain in their existing implementations. Without `run_control`, the
standalone driver retains live configuration reload and its normal behavior.

Budgets include finite retained seeds and inherited labels. Actual committed
H5 rows determine progress. Exact selections use numeric shard order, sorted
molecular groups, and stored row order; the final group/shard can be partial.
No independent rows are merged because their scientific IDs happen to match.
Labels, atom axes, topology IDs, arbitrary aligned properties, attributes, and
stored units are copied and verified, without rewriting source files. Legacy
untagged data remain untagged under their existing ordering contract.

At a target, the driver stops new builder/sampler work, finishes submitted
samplers and labels their candidates, accounts for unused builder/buffered
geometry, flushes partial QM batches, waits for outstanding training, freezes
checkpoints, and cleans up Parsl. Raw accepted overshoot remains inspectable.
Only the exact checkpoint enters the next stage or endpoint fit. Fresh initial
ensembles complete before sampling; normal acquisition-time retraining remains.
The default final fit reads the exact final checkpoint. Success requires the
trainer's complete-ensemble result, rather than merely an existing directory.

The local SQLite journal records persistent workflow submission identities,
inputs, harvested results, attempts, storage intents/receipts, and buffer/status
boundaries. H5 publication is validated, fingerprinted, fsynced, and atomic.
Recovery reconciles a published shard before consuming its QM submissions.
Failed or unknown training attempts retain their directories; retry uses a new
model index. A completed worker result that was never harvested can require
recomputation, reported as a possible recomputation rather than an exactly-once
execution claim. Durable accepted rows are not stored twice.

Per-run namespaces isolate independent stages even when model/frame numbering
restarts. Phase membership follows workflow provenance, not QM frame-ID naming.
Counters separately expose converged/unconverged labels, screening reasons,
retained counts, overshoot, unused work, role attempts/outcomes, and elapsed
runtime. Screening reasons can overlap. Observed native Parsl retries count
separately from retained frames.

Workflow and stage locks reject competing controllers. Source definitions,
resolved configuration, inherited data, and checkpoint/model artifacts are
fingerprinted. Changed inputs, unregistered H5 shards, a transaction copied to
another shard path, or corrupt completed artifacts block advancement. Backend
failure leaves an inspectable state; explicit resume retries recorded work.
SIGINT/SIGTERM drains and pauses without launching another stage. An ongoing
final fit finishes before pausing.

These are dataset budgets. They do not impose hard QM-attempt, wall-time, or
raw-retained-row caps. Graceful draining depends on working executors and has
no cancellation deadline. The first version supports linear complete-ALF
stages, plus an internal final-fit operation; other stage kinds, branching,
model warm starts, and cross-theory relabeling remain future extensions.

## Validation

Existing `atomistic` environment, Python 3.12, CPU execution,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. No environment installation was needed.
All nine upstream test modules are byte-identical to pinned main, and upstream
imports were explicitly asserted to resolve inside the production fork.

| Check | Result |
|---|---|
| Complete fork suite, including installed ALCHEMI/PySEQM CPU smoke tests | **447 passed**, 5 existing warnings, **48.71** s |
| Unchanged upstream suite imported against fork | **47 passed**, 2 existing warnings, **1.12 s** |
| Focused foundation/recovery, dependency and existing driver checks | **15 passed**, **5.94 s** |
| New staged cases | **16**, sharing one compact local task helper; 2 example-contract cases |
| Existing upstream test modules | All 9 unchanged, verified byte for byte |
| Python/JSON and example launch-script syntax; whitespace checks | Passed |

Coverage includes ALF-only, legacy ground-state multiple stages, and four-state
ALF→GALF through actual `python -m alframework` subprocesses with local
asynchronous tasks. Tests verify seed rejection/counting, exact inherited pools,
initial four-member success before sampling, strict batching, nested partial
checkpoints, final exact-dataset training, atom/force alignment, multiple groups,
nonfinite rows, duplicate scientific IDs, unit preservation/conflicts, and cache
reloads. The actual installed legacy HIPPYNN H5 reader also reads staged shards.

Faults are injected around submission, result harvesting, H5 publication,
checkpoint creation, handoff, and final-training promotion. Backend builder/QM
failures block transitions; pause/resume and failed final training preserve
already committed work. Extra/moved/corrupt artifacts and pinned-definition
mutations fail explicitly. Training artifacts in orchestration fixtures are
marked mock artifacts; these checks validate coordination, not scientific
model accuracy or production Slurm behavior. Existing installed CPU smoke tests
exercise real dependencies separately. No production GPU run was launched.

Reproduce the fork suite from `/ram/tmp` (use a fresh basetemp):

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/ram/tmp/alf_production_audit_mpl \
/vast/home/pjlohr/.conda/envs/atomistic/bin/python -m pytest \
  /vast/home/pjlohr/ALF_LANL/ALF_fork/ALF/tests -q -p no:cacheprovider \
  -o addopts= -o pythonpath=/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF \
  --basetemp=/ram/tmp/alf_staged_reproduction
```

For upstream, preimport and assert the fork's `alframework.__file__`, then call
`pytest.main` on `/vast/home/pjlohr/ALF_LANL/ALF/tests` with the same import pin.
See [the integration audit](integration_audit_2026-10-19.md) for that pattern.
An existing Parsl configuration test needs network-interface inspection; its
providers start zero allocations, and no jobs are submitted. Remaining warnings
are the existing HIPPYNN/ASE deprecations and unavailable NVML.

Release test artifacts: `/ram/tmp/alf_staged_final_validation/` and
`/ram/tmp/alf_staged_upstream_release_validation/`. Development runs and injected
failures remain under separate temporary directories for inspection.

Upstream, prototype, historical datasets, running jobs, the isolated integration
checkout, and environments were preserved. No push, scheduler submission, full
seed-pool regeneration, or production calculation was performed.
