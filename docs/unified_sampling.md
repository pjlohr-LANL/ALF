# Independent ALF execution, sampling, acquisition, and training

The unified task retains ALF's existing master/sampler/ML configuration files,
Parsl tasks, `MoleculesObject` values, and import-string extension points.
Implementation remains in the existing `samplers`, `ml_interfaces`, and `tools`
packages. Upstream ALF is a read-only reference for this fork.

| Component | Configuration | Responsibility |
| --- | --- | --- |
| Driver | `driver.strategy` | Submit single trajectories or compatible groups |
| Engine | `driver.engine` | Integration, units, state synchronization, supported operations |
| Sampling | `sampling.propagator`, `sampling.biases` | Exploration and trajectory-local force controls |
| Acquisition | `acquisition.strategy`, `trajectory_policy`, `selection` | Eligibility, scores, stopping requests, frame selection |
| Inference | `inference.interface`, `loader`, `options` | Physical predictions and model capabilities |
| Training | Existing `ML_task` and `ML_config` | Fit models and report completed artifacts |

## Configuration

Select `alframework.samplers.sampling.sampling_task` as the master's
`sampler_task`. A sampler configuration can select:

```json
{
  "driver": {"strategy": "batched", "engine": "torchsim"},
  "sampling": {
    "propagator": "langevin",
    "biases": [{"strategy": "udd", "weight": 0.45, "epsilon": 1e-12}]
  },
  "acquisition": {
    "strategy": "summed",
    "trajectory_policy": "continue",
    "selection": {"mode": "cohort_top_k", "cohort_size": 50, "k": 16}
  },
  "batching": {
    "batch_size": 8,
    "partial_policy": "timeout",
    "max_wait_seconds": 5.0
  }
}
```

Retain timestep, duration, uncertainty cutoffs, summed-score weights,
temperature/density schedules, state selection, and inference settings. This
fragment describes 50 logical trajectories, executed in physical tasks of at
most eight systems, with at most 16 selected frames across the cohort. The
numbers illustrate syntax, not model-specific scientific recommendations.

`driver.strategy` is `single` or `batched`; `driver.engine` is `ase`,
`torchsim`, or `alchemi`. Singleton execution uses the selected supported
engine. Batched dynamics uses TorchSim or ALCHEMI; unsupported combinations
fail explicitly. Physical batch width does not change the acquisition budget.
An empty `sampling.biases` list gives unbiased Langevin exploration.

Released task entrypoints and calculator fields remain compatibility paths.
An old task-local `return_top_k` is not silently converted into a cohort-wide
budget. New examples make the scientific choices explicit.

Single/ASE execution with an ASE calculator requires no Torch or GPU dynamics
package. Native models may require their own optional stacks. An ASE inference
adapter under batched dynamics still evaluates inference sequentially and is
identified as a compatibility route in metadata.

See [the complete examples](../examples/unified_sampling/README.md).

## Acquisition and logical cohorts

Both built-in strategies retain the existing energy/force uncertainty trigger.
`uncertainty` ranks by the normalized worst violation; `summed` ranks eligible
frames using the existing weighted energy, force, and optional gap score.
A high summed score alone does not establish eligibility. Screening consumes
unbiased physical predictions before acquisition.

- `first_trigger` stops each trajectory at its first eligible frame and returns
  that frame for QM without waiting for a cohort.
- `cohort_top_k` ranks across one full logical cohort. Its trajectory policy
  can be `stop` at the first eligible frame or `continue` collecting frames.
  Both `cohort_size` and `k` are required.

Each cohort has one immutable model generation and one selected electronic
state. State cycling advances by logical cohort, not GPU task. Trajectory
identities, candidate identities, and ranking ties are independent of physical
grouping and task completion order. Zero-candidate and physically rejected
trajectories still finish their membership; operational failures remain
recoverable work rather than silently excluded members.

ALF requires the full configured membership. There is no admission timeout or
automatic partial-cohort selection. Admission and execution proceed
incrementally so cohorts larger than worker capacity can make progress.
Selection returns up to K eligible frames across the pool without requiring
any particular trajectory to contribute. Top-K does not guarantee structural
diversity. Scalar local pruning is permitted only when it preserves exact
global ranking; future selectors must declare their retention requirements.

The coordinator persists candidate arrivals before acknowledging sampler
results and records selected identities and QM handoffs. Finishing trajectories
release physical sampler capacity independently of candidate storage.
Unfinished cohorts survive restart with their original inputs/model/state.
Existing graceful shutdown is retained; no new pause controls or partial-cohort
selection are introduced merely because the process stops.

## Physical inference and sampling forces

Evaluators supply physical member energies `[models, systems]`, forces
`[models, total_atoms, 3]`, optional corresponding state outputs, and declared
signals. Only a model adapter interprets its checkpoint layout. Evaluators do
not own acquisition thresholds, UDD state, LCM switches, or confinement.
A sampling force pipeline produces propagation forces from physical outputs.
Native predictions/transforms stay on their original device; only needed
coordinator observations are converted. New model families need not inherit
a Torch- or HIPPYNN-specific scientific-controller class.

UDD preserves the existing signed formula, using population standard deviation:

```text
E_propagated = mean(E_m) + w * sigma_E + E_confinement
F_bias = w * mean((E_m - mean(E_m)) * (F_m - mean(F_m))) / sigma_E
F_propagated = mean(F_m) + F_bias + F_confinement
```

The force bias is zero when `sigma_E <= epsilon`. Each system has its own
moments; padding and confinement do not contribute. A positive weight retains
the existing uncertainty-energy penalty convention. UDD requires actual ordered
member energies/forces; aggregate uncertainty alone is insufficient. Physical
screening and acquisition remain unbiased.

Existing molecular LCM construction and switching order are preserved.
UDD combined with LCM is explicitly unsupported in this milestone. Temperature
and density timing, initial velocities, independent stopping, isotropic cell
scaling, and force refresh retain their established behavior. Batched periodic
support covers fully periodic ground states with capable models; ASE boundary
and constraint capabilities are checked separately.

## Extension and training handoff

Use small callables and import paths following upstream ALF. Sampling programs
may request grouped evaluations and report optimization progress without an
MD clock. Acquisition strategies declare observations, eligibility, ranking,
and retention requirements. A geometry-only example demonstrates independence
from uncertainty thresholds; production distance/diversity methods are future
work.

A propagator import path names a zero-argument factory or class with
`create_program(context)`; its program implements `run()`. The context exposes
`evaluate_group`, `acquire`, and `finalize`, retaining standard molecule outputs
and provenance. The `ScaledProposalMethod` example uses proposal indices rather
than inventing an MD clock.

A bias import path names a zero-argument factory or class with
`apply(state, physical, propagation, options)`. It returns complete propagation
`energy` and `forces`, plus optional per-system diagnostics. It must preserve
the system/atom shapes and native device/dtype. Plugins execute in configured
order after built-in UDD/LCM controls and before confinement. Their state lives
in the sampling pipeline, never in the evaluator.

An acquisition import path provides `decide(observation, context)` returning
an `AcquisitionDecision`, and `select(candidates, context, final=False)`.
`CoordinateExtentAcquisition` demonstrates a translation-invariant geometry
gate with no energy/force uncertainty cutoffs. Scalar cohort selection requires
the explicit retention contract `scalar_top_k`. A future selector needing
additional candidate payloads must not be silently subjected to scalar pruning.

`ML_task`, `ML_config`, trainer implementations, and return conventions remain
independent. A model-reference compatibility handoff describes an opaque
artifact location, generation, inference adapter, options, and capabilities.
Only the adapter interprets HIPPYNN graphs or another family’s files. New
training and inference adapters must not require edits to driver, sampling,
or acquisition strategies.

Reactive NEB/dimer entrypoints retain their existing ASE workflows. Full
reactive batching, NPT, periodic excited states, new production model families,
structural diversity, UDD–LCM composition, and MI350 execution remain separate
milestones.

## Validation

Report CPU correctness, GPU acceptance, and potential accuracy separately.
The checkpointed draft passed 572 tests in the ALCHEMI environment and 585 in
TorchSim before this further separation. Those counts are a baseline, not a
claim that the new cohort implementation has passed hardware testing.

Acceptance replays identical observations with task widths 1, 2, 8, 32, and 50
and reversed completion order, checking identical membership, state assignment,
scores, and selected IDs. It covers incomplete admission, empty results,
failures, restart at ingestion/selection/QM handoff, model promotion, storage,
retraining, and retained-label budgets.

The standalone harnesses record source identity, versions, model fingerprints,
driver/engine, force controls, errors, throughput, memory, and hardware identity.
Float32 parity uses the approved `rtol=1e-5, atol=1e-5`, separately from
acquisition thresholds. CPU artifacts do not establish GPU acceptance.
