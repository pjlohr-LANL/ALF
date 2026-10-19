# RDKit KDG-ladder + FPS initialization audit — 2026-10-19

Implementation checkout: `/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF`, branch
`production_excited_states`, starting at `e7cbd59`.

## Implementation and boundaries

- `125853d`: optional CPU-only `tools/rdkit_initialization.py` plus portable
  enol/keto JSON configurations and reference geometries.
- `150c522`: thirteen focused cases and one additional actual-driver scenario,
  reusing the existing integration harness rather than duplicating it.
- No core-driver, existing builder/sampler, QM, training, storage-policy,
  packaging, CI, or default test-selection implementation was changed.
- The existing optional `topology` extra supplies RDKit. Tools import without
  RDKit, Torch, HIPPYNN, ALCHEMI, or Warp; generation reports an actionable error
  when RDKit is absent. No environment was installed or modified.

The command prepares the geometry pool and FPS selection explicitly before ALF
starts. Generation is independent of electronic states and QM backend. The
existing external sequential H5 replay path supplies bootstrap structures, then
switches to the run's labeled store. The emitted builder configuration and
provenance's master/sampler fields define that handoff without a core edit.

Sorted storage uses explicit reference IDs from one verified full graph mapping.
No atom IDs are inferred from species equality, labels fabricated, or source
arrays rewritten. The prepared directory includes a fingerprinted reference
snapshot. All artifacts are staged and the directory published only after
successful preparation; existing output paths and insufficient quotas fail.

## Numerical comparison to the original method

A migration-only comparison loaded the existing benchmark functions from
`molecule_smax`. Neither the production implementation nor committed tests
import that repository or the prototype.

For **both enol and keto**, with six strata, seed 42, one thread, **60 accepted
pool frames and 12 FPS selections**, the following matched exactly:

- Reference distance centers and every widened/triangle-smoothed bounds matrix.
- The benchmark's method-4/KDG-mode-2 per-stratum random seeds.
- Every generated coordinate, after undoing the recorded storage permutation;
  the comparison used the reference embedding and geometry-screening functions.
- Pair-distance descriptors and selected pool indices from the reference FPS.

Environment: Python 3.12.12, NumPy 2.3.0, RDKit 2026.03.3, H5py 3.15.1.
Temporary numerical comparison outputs are at
`/ram/tmp/alf_kdg_parity_xsmwd9xu`; they are verification artifacts, not production
data. Exact array equality applies to this pinned environment and reduced pool;
it is not a cross-version or historical-full-dataset identity guarantee.

Reference source fingerprints at comparison time:

| Benchmark file | SHA256 |
|---|---|
| `initial_samples_ALF.py` | `9d221a2100d59652736cc30ef34fd7b305afdc4509386f0b4bd71d1b9095ef41` |
| `workflows/studies/etkdg_ladder/fps.py` | `9c975a3e0d45dc51c66f39188d59b4fbcab5fb0cc979be23a1be5b4176b33803` |
| `workflows/studies/etkdg_ladder/select_fps_before_label.py` | `6580a0d277b93e131eada8e68d90b6152a9a24f6de99859a08b8e14b6648c08f` |
| `workflows/studies/etkdg_ladder/ladder_screens.py` | `c103a926ea2629bceb4dd3da4ef98ed66c42d1ced8c8da8c62a2931863a54bf3` |

## Focused and workflow validation

| Check | Result |
|---|---|
| Focused preparation, actual-driver, and tools-boundary tests | **22 passed**, 2.52 s |
| Complete fork suite, including installed CPU smoke tests | **414 passed**, 5 warnings, 13.66 s |
| Unchanged upstream tests against asserted/pinned production-fork imports | **47 passed**, 2 warnings, 1.27 s |
| Core/previous runtime implementations versus `e7cbd59` | Unchanged; only the new optional tools module is added |
| Upstream's nine test modules | Unchanged |
| Diff whitespace checks | Passed |

The focused tests cover quotas, golden seed values, FPS ties and element-pair
invariance, configuration errors, polar-H/reference-centered widening, geometry
screen boundaries/priority, wrong-connectivity rejection despite matching
formula, repeatable enol/keto preparation, file fingerprints, topology IDs and
partial-final replay, missing-RDKit imports, CLI path resolution, output refusal,
and exhausted embedding without partial outputs.

The extra actual-driver case prepares a small water seed, replays it for two
converged mock QM labels, and rejects one for excessive force on a nonselected
state. It verifies that only the accepted row reaches fresh four-member mock
training, sampling waits for model completion, later starting structures come
from the run's own labeled H5, and retraining completes. Parsl uses only local
threads. This verifies orchestration, not model accuracy or a new QM calculation.

The suite grew from 400 to 414 collected cases for this new functionality; its
previous consolidation is documented separately. Existing HIPPYNN/ASE and
unavailable-NVML warnings remain nonfatal. Validation runs from `/ram/tmp` with
one CPU thread per numerical library and imports pinned to this checkout.
README JSON snippets and the new RST section structure were checked. A full
Sphinx build was not run because docutils is absent from this environment;
no documentation dependencies were installed.

## Count semantics and delivery

The shipped presets request 50,000 geometry-screened pool frames and select
5,000 before labeling. Those complete datasets were **not regenerated** in this
implementation pass. Geometry counts are recorded in provenance; QM convergence
is tracked in existing QM diagnostics and storage acceptance/rejections in the
existing per-batch screening reports/status and actual H5 row counts. The
example's optional 16 eV/Å post-QM screen checks all configured state forces.
Rejected labels are not refilled. Historical 4,908/4,733 retained counts are not
promised for different software or labeling settings.

The configuration-only commands and bootstrap fields are documented in
`examples/rdkit_initialization/README.md` and the Sphinx builder guide. No full
50k-pool regeneration, production ALF/GALF run, GPU calculation, environment
change, remote push, or job submission was performed. Running jobs, historical
datasets, upstream, prototype, and the isolated integration checkout were left
unchanged.
