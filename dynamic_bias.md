# Dynamic UDD Bias Plan For Excited-State ALF

## Summary

This note defines a planned implementation for dynamic uncertainty-driven dynamics (UDD) biasing in the excited-state ALF sampler. The target behavior follows the force-ratio adaptive biasing scheme from van der Oord et al., *Hyperactive learning for data-driven interatomic potentials*, npj Computational Materials 9, 168 (2023):

```text
E_HAL = E - tau * sigma
```

with the adaptive bias strength estimated from Eq. 14:

```text
tau = tau_r * sum(||F_model||) / sum(||F_sigma||)
```

For excited-state ALF, the proposed analogue is to bias dynamics toward uncertainty in an adjacent excited-state energy gap:

```text
E_MD = E_selected_state_mean - tau * std(dEij)
```

where `dEij` is one of `dE01`, `dE12`, or `dE23`, and `std(dEij)` is the direct ensemble standard deviation of the selected gap.

## Current ALF Energy Convention

PySEQM excited-state labels are stored as shifted state energies:

```text
sE# = raw_pyseqm_energy - energy_offset_eV
```

For the keto examples:

```json
"energy_offset_eV": -1391.451136550848
```

This means stored state-energy targets are shifted near relative values:

```text
sE# = raw_pyseqm_energy + 1391.451136550848
```

The shift is constant and therefore does not affect forces:

```text
-grad(E_shifted - tau * sigma) == -grad(E_raw - tau * sigma)
```

Adjacent gap labels are also unaffected by the shift:

```text
dEij = sEj - sEi
     = (raw_Ej - offset) - (raw_Ei - offset)
     = raw_Ej - raw_Ei
```

Therefore dynamic `tau` estimation must be based on force norms, consistent with Eq. 14, not on absolute or shifted energy magnitudes.

## Proposed Sampler Config

Extend the existing `udd` sampler block while preserving current fixed-bias behavior:

```json
"udd": {
  "enabled": false,
  "mode": "gap_std_bias",
  "target": "initial_min_gap_pair",
  "tau_mode": "fixed",
  "bias_weight": 0.0,
  "tau_relative": 0.05,
  "tau_history_steps": 100,
  "tau_update_interval": 100,
  "tau_min": 0.0,
  "tau_max": 10.0,
  "tau_force_epsilon": 1.0e-12
}
```

Config meanings:

- `tau_mode: "fixed"` keeps the current implementation and treats `bias_weight` as fixed `tau`.
- `tau_mode: "force_relative"` enables Eq. 14-style adaptive tau estimation.
- `tau_relative` is the paper's relative biasing parameter `tau_r`.
- `tau_history_steps` is the number of recent MD steps used in the force-norm sums.
- `tau_update_interval` controls how often tau is recomputed after enough force history exists.
- `tau_min` and `tau_max` prevent runaway biasing.
- `tau_force_epsilon` prevents division by zero when `sum(||F_sigma||)` is very small.

Recommended first dynamic value:

```json
"tau_relative": 0.05
```

The paper reports typical `tau_r` values of `0.05` to `0.20`; the conservative end is appropriate for first excited-state gap-uncertainty tests.

## Implementation Plan

Apply changes only in `alframework/samplers/excited_state_sampling.py` for the first implementation.

Keep the current UDD target logic:

```text
1. Evaluate the starting geometry.
2. Pick the adjacent gap with the smallest absolute predicted gap mean.
3. Keep that gap target fixed for the trajectory.
```

Define force components:

```text
F_model = -grad(E_selected_state_mean)
F_sigma = grad(std(dEij))
```

The sign of `F_sigma` is irrelevant for Eq. 14 because only force norms are used.

For `tau_mode: "fixed"`:

```text
E_MD = E_selected_state_mean - bias_weight * std(dEij)
```

For `tau_mode: "force_relative"`:

```text
1. Build/evaluate force-producing graph nodes for:
   - E_selected_state_mean
   - std(dEij)
2. During the first tau_history_steps, collect:
   - ||F_model||
   - ||F_sigma||
3. Compute:
   tau = tau_relative * sum(||F_model||) / max(sum(||F_sigma||), tau_force_epsilon)
4. Clip:
   tau = min(max(tau, tau_min), tau_max)
5. Use:
   E_MD = E_selected_state_mean - tau * std(dEij)
6. Recompute tau every tau_update_interval steps using the latest history window.
```

The implementation must avoid using energy magnitudes to estimate tau. Eq. 14 is a force-ratio rule.

## Calculator Strategy

The current sampler creates a single HIPPYNN ASE calculator from:

```text
energy_for_md = E_selected_state_mean - bias_weight * std(dEij)
```

Dynamic tau needs separate force information for `E_selected_state_mean` and `std(dEij)`. The least invasive first implementation should add internal helper evaluations for the component forces while preserving the main MD calculator path.

Recommended approach:

- Keep the normal MD calculator as the one attached to ASE atoms.
- Add helper calculators or helper graph evaluations to compute force norms for:
  ```text
  E_selected_state_mean
  std(dEij)
  ```
- Use those force norms only for tau estimation and metadata.
- Rebuild or replace the attached MD calculator when tau changes.

This avoids changing ALF task signatures, HDF5 layout, Parsl executors, or training interfaces.

## Metadata

Add UDD tau metadata to sampler JSON output:

```json
{
  "udd_tau_mode": "force_relative",
  "udd_tau_relative": 0.05,
  "udd_tau_history_steps": 100,
  "udd_tau_update_interval": 100,
  "udd_tau_min": 0.0,
  "udd_tau_max": 10.0,
  "udd_tau_trace": [],
  "udd_model_force_norm_trace": [],
  "udd_sigma_force_norm_trace": [],
  "udd_bias_force_ratio_trace": []
}
```

Continue recording existing UDD gap fields:

```json
{
  "udd_gap_key": "dE12",
  "udd_gap_pair": [1, 2],
  "udd_initial_gap_mean": 0.0,
  "udd_initial_gap_std": 0.0,
  "udd_gap_mean_trace": [],
  "udd_gap_std_trace": []
}
```

Trace cadence should be tied to `tau_update_interval` or `Ncheck`, not every MD step, unless detailed debugging is explicitly enabled. This keeps metadata files manageable.

## Failure Modes And Safety Behavior

If UDD is enabled but direct ensemble gap nodes are unavailable, fail fast with a clear error. Dynamic tau requires direct `ensemble_dE##.std` nodes.

If `sum(||F_sigma||)` is near zero, use `tau_force_epsilon` in the denominator and then apply `tau_max`. This prevents extreme `tau` values.

If force helper evaluation fails, fail the sampler task rather than silently falling back to unbiased MD. Silent fallback would make UDD metadata misleading.

If `tau_mode` is missing, default to `"fixed"` for backward compatibility.

If `tau_mode: "force_relative"` is selected but `tau_relative <= 0`, treat this as invalid config and raise a clear error.

## Test Plan

Static checks:

```bash
python -m py_compile alframework/samplers/excited_state_sampling.py
python -m json.tool examples/excited_state_pyseqm_sharedgpu_gap_uncertainty/sampler_config.json
```

Backward compatibility tests:

- `udd.enabled: false` should preserve current sampler behavior.
- `udd.enabled: true` with `tau_mode: "fixed"` should preserve current `bias_weight` behavior.

Dynamic tau validation:

- Run one short sampler with:
  ```json
  "udd": {
    "enabled": true,
    "mode": "gap_std_bias",
    "target": "initial_min_gap_pair",
    "tau_mode": "force_relative",
    "tau_relative": 0.05,
    "tau_history_steps": 10,
    "tau_update_interval": 10
  }
  ```
- Confirm metadata contains non-empty:
  ```text
  udd_tau_trace
  udd_model_force_norm_trace
  udd_sigma_force_norm_trace
  udd_bias_force_ratio_trace
  ```
- Confirm each reported tau satisfies, within floating-point tolerance:
  ```text
  tau = tau_relative * sum(||F_model||) / sum(||F_sigma||)
  ```
  after denominator epsilon and clipping are applied.

Runtime smoke:

- Run a short gap-uncertainty example with low `maxt`.
- Confirm sampler candidates still label normally with PySEQM.
- Confirm HDF5 save and retraining behavior remain unchanged.

Safety tests:

- Force `tau_max` to a small value and confirm tau clipping appears in metadata.
- Force an unavailable gap node and confirm the sampler fails with a clear UDD error.

## Assumptions

- The first implementation applies Eq. 14 to direct adjacent-gap uncertainty, not total-energy uncertainty.
- The selected gap target remains fixed for each trajectory using the initial minimum-gap rule.
- `tau_relative = 0.05` is the recommended first value.
- Dynamic tau is sampler-local and does not require ALF runtime, Parsl, HDF5, QM, or ML interface changes.
- The committed default should remain `udd.enabled: false` and `tau_mode: "fixed"` until dynamic biasing is validated.
