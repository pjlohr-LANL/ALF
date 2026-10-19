# Staged keto RDKit → ALF → GALF example audit — 2026-10-19

Implementation checkout: `/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF`, branch
`production_excited_states`. Starting tip: `3753b25`; example/preparation/tests
commit: `70e1d2a`. Upstream main remains pinned at
`a5bb49e95fdfa458bc11700631e076b067483130`.

## Delivered workflow

The [example instructions](../examples/keto_rdkit_alf_galf/README.md) cover fresh
KDG/FPS seed generation, four-state PySEQM labeling, ordinary ALF, an exact 40k
handoff into a fresh GALF run, and exact seed-inclusive 50k/60k snapshots.
Existing compatible ALF runs can supply the handoff directly.

All required master, builder, sampler, QM, ML, reference geometry, resource,
and launch files are included. Preparation creates independent directories;
Parsl and the worker environment remain site-configurable. Replay uses the
existing local CPU builder. The example-local CLI exposes `init`, `handoff`,
`snapshot`, and `check`; it does not add a public package API or driver hook.

The default model is the reference four-member, four-state HIP-HOP ensemble
with 145 features and 20 sensitivities. Training remains E/F only. Explicit
adjacent-gap properties support diagnostics and replica-local LCM switching
without adding a gap loss or gap-uncertainty acquisition. Physical force
screening and separate post-QM screening use the existing production helpers.

Handoff selects completed numeric shards and stored rows, with finite geometry
and four-state energies/forces, and no additional geometry filtering. It checks
reference graph/IDs, compatible PySEQM AM1/CIS settings and energy offset, source
fingerprints, exact copied values, and derived gap arithmetic. It preserves
existing atom order, labels, IDs, attributes, and even unused nonfinite metadata.
Missing gap columns are derived only in new copies. No old model or status is
imported. The preflight rejects extra initial training shards and incomplete
initial model artifacts; the trained check verifies all four member completion
logs, checkpoints, summaries, and frozen input counts.

The README identifies production/prototype differences: independent rather than
global LCM switching, supported acquisition/metric adapters and config names,
seed-only nonbonded clearance, and independent resource allocations. Automatic
budgets, stopping, handoff orchestration, and cumulative QM cost accounting are
not implemented. The driver and all production runtime modules are unchanged.

## Validation results

Existing `atomistic` Python 3.12 environment, CPU execution, one OMP/MKL thread.
No dependency installation or environment modification was performed.

| Validation | Result |
| --- | --- |
| Complete fork suite, including installed ALCHEMI/PySEQM CPU smoke tests | **429 passed**, 5 existing warnings, **19.34 s** |
| All unchanged upstream tests with fork imports explicitly pinned | **47 passed**, 2 existing warnings, **1.23 s** |
| Upstream test-file comparison | All **9** upstream modules remain byte-identical |
| Example JSON/Python syntax, launch-script Bash syntax, Git whitespace | Passed |
| Reduced fresh-seed workflow | 12 accepted pool geometries, 6 selected seeds, all six ladder strata; fingerprint-checked replay |
| Actual driver, local asynchronous mocks | Legacy and CFG cases retained; keto seed → labeling → nonselected-state rejection → training → ALF replay → fresh handoff → LCM switching → QM/storage/retraining → nested checkpoints passed |
| Historical keto source selection, read-only | Exact first **40,000** = **4,733 seed + 35,267 ALF**; **68,982** completed valid rows, no nonfinite rows |
| Historical source H5 copy/reload smoke | Five selected frames copied into a temporary fresh GALF directory; labels/IDs verified and missing gaps derived; preflight passed |

The suite grew from 414 to 429 cases: ten focused snapshot/CLI cases and five
extensions to the existing example contracts. The three existing actual-driver
cases remain three cases, with the RDKit case extended through the GALF handoff.
No exclusions, default selection rules, or upstream tests were changed.

The local driver uses mocked QM/training and scripted dynamics; its artifacts
are marked mock data. Installed CPU smoke tests exercise actual dependencies,
but validation does not establish production GPU performance or model accuracy.
CPU preflight may print the installed Warp stack's no-CUDA diagnostic on this
host; no GPU calculation is launched and successful checks exit zero.

Temporary evidence:

- `/ram/tmp/alf_keto_release_validation/`: final full-suite artifacts.
- `/ram/tmp/alf_keto_upstream_1/`: unchanged upstream suite artifacts.
- `/ram/tmp/alf_keto_reference_selection_validation.json`: read-only 40k source summary.
- `/ram/tmp/alf_keto_legacy_handoff_smoke/`: five-frame historical copy/preflight.

Reproduce the fork tests from a writable temporary working directory with:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/ram/tmp/alf_validation_matplotlib
/vast/home/pjlohr/.conda/envs/atomistic/bin/python -m pytest \
  -q -p no:cacheprovider -o addopts= \
  -o pythonpath=/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF \
  /vast/home/pjlohr/ALF_LANL/ALF_fork/ALF/tests \
  --basetemp=/ram/tmp/alf_keto_validation_reproduction
```

For upstream tests, preimport `alframework`, assert its resolved package path is
this fork, and invoke `pytest.main` on `/vast/home/pjlohr/ALF_LANL/ALF/tests`
with the same `pythonpath` override and a distinct temporary base directory.

Full 50k-pool generation, production GPU calculations, submission, and pushing
were not performed. Upstream, the prototype, historical datasets, running jobs,
and the earlier isolated integration checkout remain unchanged.
