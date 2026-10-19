# Physical-force screening

`max_force_cutoff` screens the largest per-atom Euclidean norm over the
ensemble-mean physical forces of every configured state. Ground-state runs
screen their physical state. The mean is taken before the norm. Equality at
the cutoff passes. Required state forces must be present, correctly shaped,
and finite; missing or invalid outputs fail explicitly. `properties_list` is
passed through the sampler and declares the required excited-state forces.

`Fmeanmax` retains its name but now has this physical meaning. Candidate
metadata includes `force_screen_schema=alf.physical_force_screen.v1` and
`force_screen_source=ensemble_mean_physical_all_states`. Old unmarked
`Fmeanmax` values must not be interpreted as physical-screen diagnostics.
`physical_F{state}_max` records each state's maximum. `propagation_Fmax`
includes LCM and confinement. `lcm_bias_Fmax` is the LCM force minus the
selected state's physical force; `lcm_correction_Fmax` isolates the analytic
LCM correction relative to the pair-average physical force.
`confinement_Fmax` reports confinement separately. All force units follow the
calculator (the production configurations use eV/Å).

Set optional `force_screen_audit_dir` in the sampler configuration to write
one uniquely named JSON report per batch. Relative paths use
`master_directory`. Reports contain initial coordinates (not yet evaluated),
the last frame passing geometry and physical-force checks, and the actual
rejected frame, at most three snapshots per replica. Checked force snapshots
include the same-frame physical means. Topology rejection before evaluation
has no force arrays. The step is the actual integrator step; the time retains
ALF's legacy warmup clock. Reports survive empty candidate batches and are
written atomically on normal return or exceptions. No full trajectories are
recorded. The directory is opt-in and independent of candidate acquisition.

Propagation, ensemble uncertainties, cutoffs, objectives, and independent
per-replica stay-fixed gap switching are unchanged.

## Module boundaries

Shared screening policy, physical-force maxima, artificial-force diagnostics,
cutoff checks, and audit snapshots live in `alframework/tools/force_screen.py`.
Configuration/input checks, calculator-output shape/finiteness checks, and
`--test_sampler` output validation live in
`alframework/tools/sampling_validation.py`. These modules do not import sampler
implementations. Torch is imported locally only when processing tensors;
configuration and debug helpers work without the optional GPU stack.

The sampler obtains predictions, delegates numerical validation and screening,
and executes the resulting strategy decisions (candidate acquisition, replica
freezing, and independent LCM switching). Runtime execution guards stay alongside
the strategy. Public sampler configuration functions and calculator methods
retain their existing import locations/signatures through delegates. The old
`samplers/force_screen.py` and `samplers/debug_validation.py` helper files have
been removed; their implementation is under tools, without shim modules.

## Sampling, storage, and training filters

These controls operate at different points and keep their existing defaults:

| Stage | Controls | Quantity and effect |
|---|---|---|
| Sampling | `max_force_cutoff` | Maximum physical ensemble-mean atom-force norm over configured states; stops an excessive-force replica. |
| Before storage | `dataset_screening` and `max_force_cutoff` | When enabled, screens QM atom-force norms before writing labels. |
| Training | `remove_high_forces_cut`, `remove_high_forces_std` | Excludes structures with QM force-component outliers when preparing the training pool; original H5 rows remain stored. |

The training options retain upstream HIPPYNN semantics. The fixed cut compares
`abs(component - mean)` to `remove_high_forces_cut`; the standardized cut compares
that deviation divided by the component standard deviation to
`remove_high_forces_std`. Comparisons use strict `>`; equality passes. These
filters do not use per-atom Euclidean force norms. Setting an option to `null`
disables that particular outlier filter; other configured filters still apply.

For one state, the existing sequence is force cut, force standard-deviation cut,
then energy cut and energy standard-deviation cut. Statistics are recomputed
from the current survivors at each stage. For multiple states, static masks are
computed from the common pool and unioned across configured state forces and
energies, followed by standardized masks from the common survivors. This
preserves the existing state-order-independent extension. Every retained array,
including gap labels and frame indices, uses the same final row selection.
Filtering happens before training/validation/test splitting and does not rewrite
source H5 files. Sampling and storage cutoffs are not mapped onto the training
options or automatically enabled by them.
