# Production excited-state integration audit — 2026-10-19

Current production-checkout validation: **422 fork tests passed** and
**47 unchanged main tests passed**. Both suites were rerun against imports
from `/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF` after the five integration
commits were transferred. Earlier isolated-checkout results remain below
as history.

## Checkout and ancestry

Active implementation checkout: `/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF`
on `production_excited_states`. Subsequent development uses this checkout.

The five integration commits were originally implemented and validated on
`integration/physical-force-main-compat` in the isolated checkout
`/vast/home/pjlohr/github/ALF_production_integration`. They were imported
from that local repository and applied by fast-forward from `452c650` to
`b90e58c`, preserving all five commit identities.

- Production main: `a5bb49e95fdfa458bc11700631e076b067483130`.
- Production fork base (`production_excited_states`):
  `452c650a2a9824981e808b3440098e7e1d04d4d0`.
- The fork base contains main exactly: merge base equals the main SHA;
  main-only/fork-only commit counts at the start were **0 / 25**.
- Prototype reference: `733ab9d98ee244f0825079b40bd41aab6690a101`.

During initial implementation, the isolated checkout was a local clone
without shared hard-linked objects; the production checkouts were unchanged
at that stage. The approved transfer now updates the production fork only.
Production main, the prototype, the isolated integration checkout, running
jobs, and installed environments remain unchanged by the transfer. Main has
two pre-existing untracked example-run directories from July; these were
left untouched. No merge commit, push, or job submission was performed.
No prototype budget orchestration or analysis utilities were ported.

## Review units

1. **`59af6bd` — physical-force screening.** Shared calculator reduction screens
   the largest atom norm across every configured state's ensemble-mean physical
   force, before LCM and confinement. The HIPPYNN, native, and ASE adapters share
   that contract. The task now passes `properties_list` to the run function.
   Required missing, malformed, or nonfinite forces fail explicitly. Cutoff
   equality passes. Propagation, uncertainty acquisition, objective, thresholds,
   and per-replica independent stay-fixed switching retain their original logic.
   Schema/source markers, state maxima, propagation/confinement maxima, and two
   distinct LCM-bias measures make the meaning explicit. Optional bounded audit
   reports survive zero-candidate returns and preserve checked coordinates before
   freezing. See [physical_force_screen.md](physical_force_screen.md).
2. **`41bf32a` — storage compatibility.** The default writer again sorts by
   atomic number, including force and NAC atom axes. Stable ties preserve a
   deterministic legacy convention. Only genuinely supplied IDs are persisted;
   topology-aware replay/sampling attach canonical IDs. Tagged training/replay
   reads normalize identity before validation, supporting old reference-ordered
   tagged shards and new sorted tagged shards together. Untagged validation is
   retained; ambiguous incompatible layouts fail. Existing H5 files are not
   rewritten. The upstream storage test is restored verbatim; a fork-only NAC
   round-trip assertion now expects the required atom permutation. See
   [storage_identity.md](storage_identity.md).
3. **`ce7b0ec` — driver cleanup and the initial audit.** The driver delegates
   its optional sampler debug checks to a helper. That helper initially lived
   under samplers; the tools-boundary follow-up below supersedes that location. No GPU, selected-state,
   topology-debug assertion, or physical-force-screen implementation lives in
   the core driver. Real entry-point tests cover local asynchronous bootstrap,
   four-member training completion flags, sampling, nested outputs, QM storage,
   capacity accounting, and retraining for both legacy and batched samplers.
   These are mock tasks; they do not train scientific models or submit jobs.

## Remaining modifications to upstream runtime files

The fork still changes only three runtime files that already existed in main;
feature implementations remain separate modules. Packaging is also modified.
The table describes the complete fork delta relative to the pinned main SHA,
including changes predating this integration.

| Existing file | Delta (added/removed) | Why it remains necessary |
|---|---:|---|
| `alframework/__main__.py` | 199 / 39 | Generic batching and strict-full buffering; replica-based queue capacity; task argument/config reload support; nested result flattening; storage acceptance hooks and summaries; a tools-owned debug-validation call. The storage trigger uses `>=` so an exact threshold can retrain. |
| `alframework/builders/builders.py` | 5 / 0 | Optional explicit PBC override for molecular CFG inputs; omitted settings retain main behavior. |
| `alframework/tools/tools.py` | 88 / 1 | `pair_atomic` storage and electronic-state pair metadata; validated, opt-in topology ID persistence. Main's sorting contract is restored; stable ties are deterministic. |
| `setup.py` | 17 / 1 | Optional excited-state, HIPPYNN, and GPU-dynamics dependencies, keeping base installs usable. |

The core driver retains generic routing, status, capacity, and storage hooks;
it does not choose excited-state energies, construct LCM forces, define force
screening, or load a GPU backend. Existing ground-state samplers/calculators,
QM/ML task queues, and molecule object definitions are unchanged from main.
The driver delta is smaller by 22 lines than the audited fork base. Future
upstream merges still require reviewing these generic orchestration changes;
passing upstream tests is compatibility evidence, not a guarantee of future
conflict-free merges.

## Initial isolated-checkout validation results

Environment: existing `atomistic` Python 3.12, Torch/HIPPYNN/ALCHEMI stack;
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`; local CPU execution only.

| Check | Before integration | After integration |
|---|---:|---:|
| Complete fork suite | 366 passed | **405 passed**, 13.73 s |
| Unchanged main suite imported against fork | 46 passed, 1 ordering failure | **47 passed**, 1.17 s |
| Actual `python -m alframework` entry-point scenarios | Helpers only | Legacy and batched bootstrap → initial train → sampling → QM → retrain both passed |
| `git diff --check` | — | Passed |

The first sandboxed full run exposed one stale fork-only NAC input-order
assertion, which was corrected to check sorted NAC atom alignment. Its other
failure was the existing Parsl configuration test's network-interface
inspection (`psutil.net_if_addrs`), forbidden by the sandbox. The complete
suite then ran with that inspection permitted and all 405 tests passed. The
Parsl configuration test only constructs executors; its Slurm providers have
`init_blocks=0`, and it does not start or submit them. Existing ORCA regex,
HIPPYNN/ASE deprecation, and unavailable NVML warnings are nonfatal.

New regressions include:

- LCM propagation >16 eV/Å with physical maximum 8.66025 eV/Å remains eligible;
  opposite ensemble-member deviations cancel before the physical norm.
- Excess force on a nonselected physical state rejects without LCM; missing,
  transposed, and nonfinite outputs fail in the shared/native/ASE paths.
- Confinement leaves physical eligibility unchanged; analytic LCM output and
  replica-local propagation remain unchanged; exact cutoff boundaries pass.
- Bounded initial/last-valid/rejected evidence matches actual checked steps and
  forces, including force-only and initial-topology zero-candidate batches.
- Differently ordered molecules sharing a formula keep species, coordinates,
  force, NAC, and supplied IDs aligned. Ordinary stores have no fabricated IDs.
- Mixed old/new tagged training and replay restore canonical order without
  modifying source fingerprints; fractional/duplicate/out-of-range IDs and
  partially tagged formula groups fail explicitly.
- The actual driver accepts nested task outputs, respects replica capacity,
  starts sampling only after completed initial training, and reaches retraining.

No actual GPU trajectory or GPU QM calculation was run. Existing CPU smoke
coverage exercises the ALCHEMI integrator; this does not establish production
GPU performance, trajectory survival, or scientific gap-sampling enrichment.

## Reproduction

From a writable temporary directory (to contain Parsl's runinfo), run the full
fork suite with the existing interpreter:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MPLCONFIGDIR=/ram/tmp/alf_production_audit_mpl \
  /vast/home/pjlohr/.conda/envs/atomistic/bin/python -m pytest \
  /vast/home/pjlohr/ALF_LANL/ALF_fork/ALF/tests \
  -q -p no:cacheprovider -o addopts= \
  -o pythonpath=/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF \
  --basetemp=/ram/tmp/alf_production_checkout_fork_transfer_rerun
```

For upstream compatibility, preimport the production-fork package and assert its
location. Otherwise pytest may prepend main's checkout and accidentally test
main against itself. Use the same environment as above:

```python
import alframework, pytest
assert alframework.__file__.startswith(
    '/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF/')
raise SystemExit(pytest.main([
    '/vast/home/pjlohr/ALF_LANL/ALF/tests', '-q', '-p', 'no:cacheprovider',
    '-o', 'addopts=',
    '-o', 'pythonpath=/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF',
    '--basetemp=/ram/tmp/alf_production_checkout_main_transfer',
]))
```

## Tools-boundary follow-up in the isolated checkout

The three integration commits above are retained. The subsequent refactor moves
reusable validation into `tools/sampling_validation.py` and force-screen policy,
metrics, diagnostics, and audit support into `tools/force_screen.py`. Samplers
retain execution and acquisition strategies and call these helpers. The driver
imports debug validation from tools. There are no helper shim files left under
samplers and tools do not import sampler implementations. Optional Torch imports
are confined to numerical helper functions.

The follow-up retains the current sampling cutoff, diagnostic schema, audit
format, propagation, and switching behavior. Upstream-style training force
filters and the optional post-QM storage screen remain separate policies.

The refactor is committed as **`ebdb793`**; it preserves the initial three
commits. A separate follow-up adds the compatibility regressions and this
updated interpretation of filtering stages. The excited-state training
implementation itself was not changed in this follow-up.

| Follow-up check | Result |
|---|---|
| Complete fork suite, including new regressions | **422 passed**, 13 warnings, 13.62 s |
| Unchanged main suite imported against fork | **47 passed**, 10 warnings, 1.11 s |
| Focused sampling/force/actual-driver suite | **108 passed** |
| Training compatibility, existing trainer, and tools import tests | **62 passed** |
| Source dependency boundary | Tools have no sampler implementation imports; imports work while Torch, HIPPYNN, ALCHEMI, and Warp are blocked. |

The 15 additional training regressions compare retained indices directly with
HIPPYNN's `Database.remove_high_property` for fixed, standardized, combined,
and disabled filtering. Cases include nonzero means, equality at a cutoff,
zero variance, statistics recomputed after removal, and large physical force
norms with no component outliers. Multistate cases verify state-order
independence and alignment of coordinates, all state labels, gaps, indices,
and extra frame IDs. A training-read test verifies that source H5 fingerprints
remain unchanged after filtering. Two dependency tests cover tools imports
without optional GPU dependencies and the absence of sampler helper shims.

No thresholds, defaults, training objectives, acquisition policies, diagnostic
schemas, audit formats, or production jobs changed. Validation remains CPU-only;
no GPU production trajectory was launched. The earlier warning and sandbox
notes apply to these runs as well.

## Production-checkout transfer and validation

The clean `production_excited_states` branch in the production checkout was
fast-forwarded from `452c650` to `b90e58c`. These five commits are retained
unchanged and were verified as ancestors of the updated branch:

1. `59af6bd` — corrected physical-force screening and bounded audit snapshots.
2. `41bf32a` — upstream storage ordering and explicit topology identities.
3. `ce7b0ec` — delegated driver validation and asynchronous entry-point tests.
4. `ebdb793` — tools-owned screening and reusable sampling validation.
5. `b90e58c` — upstream training-filter compatibility regressions.

All runtime code and tests match `b90e58c` exactly; only this audit and its
reproduction paths were updated after the fast-forward. The documentation
update is committed separately.

| Production-checkout check | Result |
|---|---|
| Complete fork suite | **422 passed**, 5 warnings, 11.96 s |
| Unchanged main suite, with production-fork imports asserted and pinned | **47 passed**, 2 warnings, 1.40 s |
| Tools imports/dependency boundary, physical-force regressions, training-filter compatibility, and actual legacy/batched driver integration | Included in the passing complete fork suite |
| Five imported commit identities and ancestry | Verified unchanged |
| Runtime and test tree relative to `b90e58c` | Identical |
| Diff whitespace checks | Passed |

Both test runs asserted that `alframework.__file__` resolves inside the actual
production checkout. Tests ran from `/ram/tmp` with the existing `atomistic`
interpreter and one CPU thread per numerical library. The complete suite needs
network-interface inspection for an existing Parsl executor-configuration test;
its Slurm providers have `init_blocks=0` and no jobs are submitted.

The first full-suite invocation used a Python runner supplied through stdin.
Its spawned PySEQM smoke-test process could not reopen `/ram/tmp/<stdin>`;
421 tests passed and that launcher-dependent test failed. Rerunning through
normal `python -m pytest` passed all 422 tests without source changes. Use the
module invocation above for reproducible multiprocessing tests. Remaining
warnings concern existing HIPPYNN/ASE deprecations and unavailable NVML.

No environment installation, remote push, scheduler submission, production
GPU calculation, or change to running jobs was performed.
