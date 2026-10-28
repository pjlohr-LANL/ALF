# Independent ALF strategies

Keep ALF's usual master, sampler, builder, QM, and ML configuration files.
Select the unified task in the master:

```json
{
  "sampler_task": "alframework.samplers.sampling.sampling_task",
  "sampler_config_path": "sampler_config.json"
}
```

Keep the existing `ML_task` and `ML_config_path` selections. Training remains a
separate task; only the inference adapter interprets a model family's files.

| Choice | Configuration |
| --- | --- |
| Single or batched grouping | `driver.strategy` |
| ASE, TorchSim, or ALCHEMI engine | `driver.engine` |
| Langevin or a future propagator | `sampling.propagator` |
| No bias, UDD, or existing force controls | `sampling.biases` |
| Uncertainty or summed ranking | `acquisition.strategy` |
| Stop at first eligible frame or continue | `acquisition.trajectory_policy` |
| Immediate frame or cohort-wide top-K | `acquisition.selection` |
| Native evaluation or ASE compatibility | `inference.interface`, `loader`, `options` |

## Examples

- `sampler_ase.json`: single/ASE, analytic ASE ensemble, unbiased Langevin,
  first-trigger return. It needs neither Torch nor a trained model.
- `sampler_udd_batched.json`: batched/TorchSim, native HIPPYNN, Langevin + UDD,
  a full 50-trajectory cohort and 16 selected frames, with physical width 8.
- `sampler_udd_single.json`: the same cohort and acquisition choices through
  single/ASE execution. Physical batching settings are inactive, not the
  acquisition cohort.
- `sampler_summed_cohort.json`: the batched UDD example with summed ranking.
  Its supplied weights illustrate syntax; choose scientific values for the
  intended model and application.

All scientific choices are explicit in these examples. An empty bias list
means no added exploration bias. A UDD bias contains its signed `weight` and
`epsilon`. Its force bias uses real ordered member energies/forces; aggregate
uncertainty alone is insufficient. The existing positive-weight convention
adds `weight * energy_std` to propagation energy. Confinement is excluded from
ensemble moments; screening and acquisition use unbiased physical outputs.
UDD combined with LCM is rejected in this milestone.

The analytic `models.py` fixture is an integration check, not a physical
potential. Use the appropriate inference loader and worker environment for a
real model. TorchSim 0.6.2 and ALCHEMI retain separate environments; ROCm uses a
ROCm PyTorch build and the logical device name `cuda:0`. Native evaluation can
be used with a single engine too. An ASE adapter under batched dynamics remains
sequential inference and is identified in metadata.

## Immediate returns and logical cohorts

`selection.mode: first_trigger` stops at the first eligible frame and returns
one frame per trajectory immediately. It preserves the usual ALF sampling
pattern. A physical batch contains independent trajectories, not a single
larger interacting system.

`selection.mode: cohort_top_k` requires `cohort_size` and `k`. It waits until
all configured trajectories finish and ranks across their eligible frames.
`trajectory_policy` may be `stop` or `continue`; there is no required quota
per trajectory and no structural diversity filter. The existing energy/force
uncertainty gate remains the eligibility rule even for `strategy: summed`;
the weighted sum changes ranking only.

A cohort has one pinned model generation and one selected electronic state.
State cycling advances by logical cohort. Cohort IDs, trajectory IDs, frame
IDs, and ranking ties are independent of GPU grouping and completion order.
If K exceeds the number of eligible frames, all eligible frames are selected.
Empty/rejected trajectories still count as completed members.

Cohorts require full membership. There is no admission timeout or implicit
partial-cohort selection. Physical timeout batching still submits groups after
`batching.max_wait_seconds` and does not close a cohort. Incomplete cohorts and
candidates survive restart; the existing graceful-shutdown support adds no new
pause-control feature. Candidate ingestion and QM handoff are journaled
separately from physical worker-capacity accounting.

A direct analytic demonstration is available from the repository root:

```bash
PYTHONPATH="$PWD" python examples/unified_sampling/demo.py --method mlmd
PYTHONPATH="$PWD" python examples/unified_sampling/demo.py --method udd
PYTHONPATH="$PWD" python examples/unified_sampling/demo.py --method proposals --acquisition-policy continue
PYTHONPATH="$PWD" python examples/unified_sampling/demo.py --method udd --physical-evaluator
PYTHONPATH="$PWD" python examples/unified_sampling/demo.py --method proposals --acquisition geometry --acquisition-policy continue
```

`ScaledProposalMethod` in `plugins.py` demonstrates grouped proposals without
an MD clock. The example program uses the same evaluation, screening, and
acquisition context. It is not an optimizer. New strategies use small callable
interfaces and import paths, following upstream ALF rather than a new workflow
framework. See [the strategy contracts](../../docs/unified_sampling.md).

The physical-evaluator command uses a bare NumPy evaluator that inherits neither an
ALF scientific-controller class nor a Torch module. Its loader accepts an
opaque model reference and requested predictions. The same sampling and
acquisition configuration works through either evaluator interface. This
analytic fixture demonstrates the adapter boundary, not a new production
model family or training implementation.

The geometry command uses the translation-invariant distance to a structure's
geometric centroid, with an explicit threshold, and removes the uncertainty
cutoffs. This isolated-system toy demonstrates a callable acquisition strategy
with its own gate and scalar score. It does not implement a production novelty
metric, periodic descriptor, or structural diversity selector.

## Schedules and model capabilities

Retain `dt` in fs, `maxt` in ps, `Ncheck`, thresholds, state selection,
confinement, and temperature/density settings. Two-element schedule arrays
are parameter ranges, not values for the first two systems. Density updates
scale cells and positions isotropically, preserve velocities, and refresh
forces. Batched periodic models must support fully periodic ground-state
inputs; HIPPYNN checkpoints need the trained cell input. ASE constraints and
partial PBC are checked separately against engine/model capabilities.

Reactive entrypoints retain their existing ASE optimizer workflows. Full
reactive batching, NPT, periodic excited states, UDD–LCM composition,
production distance/diversity methods, and new model families are later work.
Released legacy configurations retain their semantics; an old task-local
`return_top_k` is not silently changed to a cohort-wide budget.

## Acceptance and performance

The runner harness separates grouping, engine, and force controls:

```bash
python examples/portable_batched_sampling/benchmark.py \
  --driver batched --engine torchsim --device cuda:0 \
  --sampling-method udd --udd-bias-weight 0.45 \
  --master /path/to/master_config.json \
  --ensemble-directory /path/to/models/model-0003 \
  --structures /path/to/representative.extxyz \
  --batch-sizes 1 2 8 32 50 --output /path/to/udd-torchsim.json
```

Repeat with `--driver single --engine ase`, or a supported singleton GPU
engine. Use `--engine alchemi` in its NVIDIA environment. `--sampling-method
mlmd` omits the UDD bias. For excited models, add `--selected-state N` and
repeat each intended surface. Existing `--driver ase` and `--backend` command
forms remain accepted compatibility aliases.

The harness retains `--sampling-method mlmd|udd` as convenient mechanics-test
presets. ALF runtime configuration uses separate propagator and bias selections.

The harness exercises the separate physical evaluator and sampling force
pipeline. It compares propagated energies/forces and unbiased diagnostics,
checks one-step analytic motion and velocity/kinetic-energy round trips,
independent freezing, and periodic cell scaling. Single timings evaluate
systems serially; they do not report packed inference under a single label.
Artifacts retain exact configuration, source, fingerprints, versions,
errors, throughput, GPU memory, and hardware identity.

The analytic thermal check starts from zero velocities, equilibrates two
system groups at 200/400 K, swaps their targets, and repeats:

```bash
python examples/unified_sampling/thermal_acceptance.py \
  --driver batched --engine torchsim --device cuda:0 \
  --sampling-method udd --udd-bias-weight 0.45 \
  --output /path/to/udd-thermal.json
```

It uses eight 32-atom harmonic systems. Each phase has 500 equilibration steps
and 20 blocks of 25 steps at 2 fs and friction `0.02 fs^-1`, spanning 20
relaxation times for both equilibration and measurement. Group means must
agree with targets within the larger of 8% or four lag-one-adjusted block
standard errors. JSON records errors, correlations, effective blocks, and both
phases. This checks thermostat behavior, not trained-potential stability.

Cohort acceptance must replay identical candidate observations across widths
1, 2, 8, 32, and 50, including reversed arrival order and restart boundaries,
and compare exact membership, state assignment, scores, and selected IDs.
Stochastic trajectories across engines need not be bitwise identical.

The standalone replay harness records physical observations once, then tests
the production coordinator without introducing different stochastic paths:

```bash
python examples/unified_sampling/cohort_acceptance.py \
  --device cuda:0 --sampling-method udd --udd-bias-weight 0.45 \
  --strategy summed --trajectory-policy continue \
  --master /path/to/master_config.json \
  --ensemble-directory /path/to/model-artifact \
  --structures /path/to/representative.extxyz --selected-state 0 \
  --cohort-size 50 --top-k 16 --physical-widths 1 2 8 32 50 \
  --output /path/to/cohort-replay.json
```

It also checks duplicate delivery, partial-ingestion restart, and a persisted
QM-handoff receipt. It does not execute MD, QM, or training and records
`dynamics_executed: false`; combine its evidence with the mechanics, thermal,
and end-to-end workflow tests. Omit checkpoint arguments for an analytic CPU
check. Journals are retained beside the output in a unique directory.

Float32 parity uses approved `rtol=1e-5, atol=1e-5`, independently of acquisition
thresholds. CPU results do not establish GPU acceptance or potential accuracy.
Retain actual NVIDIA/MI250 results separately and repeat on intended MI350
hardware before production deployment.
