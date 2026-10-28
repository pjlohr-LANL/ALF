# Independent sampling, acquisition, and execution

Select the unified task in your master configuration:

```json
{
  "sampler_task": "alframework.samplers.sampling.sampling_task",
  "sampler_config_path": "sampler_config.json"
}
```

The four choices in the sampler configuration are independent:

| Configuration | Selection |
| --- | --- |
| `sampling.method` | `mlmd` or `udd`, or an import-path method plugin |
| `acquisition.method` | `uncertainty`, or an import-path acquisition plugin |
| `acquisition.policy` | `stop` at the first eligible event, or `continue` and rank candidates |
| `dynamics.driver` | `ase` or `batched` |
| `dynamics.backend` | `torchsim` or `alchemi` when the driver is batched |
| `inference.interface` | `native` ensemble evaluation or `ase` calculator compatibility |

`sampling.method: "udd"` applies the existing signed uncertainty bias to the
propagation energy and forces. It does not choose acquisition behavior.
`udd_bias_weight` is required for UDD; `udd_bias_eps` defaults to `1e-12`.
A positive weight retains the existing `mean_energy + weight * energy_std`
convention. Physical screening and acquisition diagnostics use the unbiased
ensemble. Confinement is excluded from the uncertainty and bias calculation.
Combining UDD with enabled LCM switching is rejected in this milestone.

The `sampler_udd_batched.json` example uses native HIPPYNN with TorchSim.
Set `dynamics.driver` to `ase` to run the same method and acquisition through
ASE; the existing batching and engine settings then become inactive. Native
HIPPYNN inference still requires its Torch/HIPPYNN environment. Set the backend
to `alchemi` for the validated NVIDIA ALCHEMI environment. Use separate worker
environments for TorchSim 0.6.2 and ALCHEMI.

The `sampler_ase.json` example uses a two-member analytic ASE ensemble from
`models.py`. This checks the base ASE path without Torch or a checkpoint;
it is not a physical potential. Replace its loader with your model's ASE
ensemble loader for a real calculation. An ASE inference adapter under
batched dynamics evaluates systems sequentially and is identified as a
compatibility path in metadata.

Run the analytic task directly from the repository root:

```bash
PYTHONPATH="$PWD" python examples/unified_sampling/demo.py --method mlmd
PYTHONPATH="$PWD" python examples/unified_sampling/demo.py --method udd --acquisition-policy continue
PYTHONPATH="$PWD" python examples/unified_sampling/demo.py --method proposals --acquisition-policy continue
```

The last command uses the small `ScaledProposalMethod` plugin in `plugins.py`.
It proposes two structures, evaluates them as a group, applies the same
acquisition and screening, and returns normal ALF candidates. It requires no
MD clock or timestep. This is an interface example rather than an optimizer.

Existing scientific settings remain in the sampler configuration: `dt` in fs,
`maxt` in ps, `Ncheck`, uncertainty cutoffs, state selection, candidate limits,
confinement, and temperature/density schedules. The supplied values illustrate
syntax; retain validated settings for the intended model and application.

Compatible batched systems may differ in size and composition, but share model,
electronic-state, and boundary-condition compatibility. Full batches launch
immediately; timeout batches launch after the oldest queued member waits five
seconds by default. ASE dispatches one system per task.

Temperature and density schedules are independent per system. Their two-element
arrays specify parameter ranges, not values assigned to the first two systems.
Density schedules scale the cell and positions isotropically, preserve
velocities, and refresh forces. Periodic ground-state HIPPYNN checkpoints must
contain a trained cell input identified by `ML_config.cell_key`. NPT, periodic
excited states, and full reactive NEB/dimer batching are subsequent milestones.

Existing sampler task paths and calculator fields remain compatible. Use this
new task to opt into the shared screening and acquisition behavior across
drivers. It rejects invalid geometry before acquisition; legacy task behavior
is preserved through the legacy entrypoints.

## Acceptance and performance

The existing harness now accepts independent execution and method selections:

```bash
python examples/portable_batched_sampling/benchmark.py \
  --driver batched --backend torchsim --device cuda:0 \
  --sampling-method udd --udd-bias-weight 0.45 \
  --master /path/to/master_config.json \
  --ensemble-directory /path/to/models/model-0003 \
  --structures /path/to/representative.extxyz \
  --batch-sizes 1 2 8 32 50 --output /path/to/udd-torchsim.json
```

Repeat with `--driver ase` for sequential singleton ASE execution. Repeat the
batched command with `--backend alchemi` in its NVIDIA environment. Select
`--sampling-method mlmd` to omit the UDD bias. For an excited-state model,
include `--selected-state N` and repeat for each intended surface. UDD requires
an ensemble exposing actual member contributions; aggregate uncertainty alone
is insufficient.

The harness compares total propagated energies/forces and unbiased diagnostics
between packed and individual evaluations, checks independently stopped
systems and periodic cell scaling, and records throughput and peak allocated
Torch memory. ASE timing evaluates singleton systems sequentially; it does not
time packed inference under the ASE label. Artifacts record method, driver,
inference execution, bias parameters, source revision, checkpoint fingerprints,
package versions, hardware identity, and parity tolerances.

The analytic thermal check starts all velocities at zero, equilibrates two
groups at different target temperatures, then swaps their targets and repeats:

```bash
python examples/unified_sampling/thermal_acceptance.py \
  --driver batched --backend torchsim --device cuda:0 \
  --sampling-method udd --udd-bias-weight 0.45 \
  --output /path/to/udd-thermal.json
```

It uses eight independent 32-atom harmonic systems by default. Each phase
equilibrates for 500 steps and collects 20 blocks of 25 steps, at 2 fs and
friction `0.02 fs^-1`. Both windows span 20 friction relaxation times. The
mean kinetic temperature of each group must agree with its target within
the larger of 8% or four lag-one-adjusted block standard errors. The JSON
records the observed errors, correlations, effective block counts, and both
temperature phases. This checks thermostat units and per-system updates;
it does not establish long-term stability of a trained potential.

Float32 parity uses the approved `rtol=1e-5` and `atol=1e-5`, separately from
sampling thresholds. CPU artifacts do not establish GPU acceptance. Repeat the
same trained-checkpoint and policy checks on the intended MI350-family system
with its supported ROCm/PyTorch build before production deployment.
