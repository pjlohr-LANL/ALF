# Production excited-state integration audit — 2026-10-19

## Checkout and ancestry

Implementation branch: `integration/physical-force-main-compat` in the isolated
checkout `/vast/home/pjlohr/github/ALF_production_integration`.

- Production main: `a5bb49e95fdfa458bc11700631e076b067483130`.
- Production fork base (`production_excited_states`):
  `452c650a2a9824981e808b3440098e7e1d04d4d0`.
- The fork base contains main exactly: merge base equals the main SHA;
  main-only/fork-only commit counts at the start were **0 / 25**.
- Prototype reference: `733ab9d98ee244f0825079b40bd41aab6690a101`.

This is a local clone, without shared hard-linked objects. The production
fork, production main, prototype, running jobs, and installed environments
were not edited. Main has two pre-existing untracked example-run directories
from July; these were left untouched. No push, merge, or job submission was
performed. No prototype budget orchestration or analysis utilities were ported.

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
3. **Driver cleanup and this audit.** The driver delegates its optional sampler
   debug checks to `samplers/debug_validation.py`. No GPU, selected-state,
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
| `alframework/__main__.py` | 199 / 39 | Generic batching and strict-full buffering; replica-based queue capacity; task argument/config reload support; nested result flattening; storage acceptance hooks and summaries; a sampler-owned debug-validation call. The storage trigger uses `>=` so an exact threshold can retrain. |
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

## Validation results

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
  PYTHONPATH=/vast/home/pjlohr/github/ALF_production_integration \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MPLCONFIGDIR=/ram/tmp/alf_production_audit_mpl \
  /vast/home/pjlohr/.conda/envs/atomistic/bin/python -m pytest \
  /vast/home/pjlohr/github/ALF_production_integration/tests \
  -q -p no:cacheprovider -o addopts= \
  --basetemp=/ram/tmp/alf_complete_integration_final
```

For upstream compatibility, preimport the integration package and assert its
location. Otherwise pytest may prepend main's checkout and accidentally test
main against itself. Use the same environment as above:

```python
import alframework, pytest
assert alframework.__file__.startswith(
    '/vast/home/pjlohr/github/ALF_production_integration/')
raise SystemExit(pytest.main([
    '/vast/home/pjlohr/ALF_LANL/ALF/tests', '-q', '-p', 'no:cacheprovider',
    '-o', 'addopts=',
    '-o', 'pythonpath=/vast/home/pjlohr/github/ALF_production_integration',
    '--basetemp=/ram/tmp/alf_main_contract_integration',
]))
```
