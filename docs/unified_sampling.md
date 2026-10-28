# Unified sampling methods and execution drivers

The unified `alframework.samplers.sampling.sampling_task` separates scientific
sampling and acquisition choices from execution and model inference. MLMD and
UDD use the same acquisition and screening components through ASE, TorchSim,
or ALCHEMI. Existing task entrypoints keep their compatibility behavior.

The ALF driver resolves input cardinality from the execution driver: ASE
receives one system per task, while batched execution groups compatible systems
up to the configured width. Presence of an inference or dynamics configuration
block alone does not turn an ASE task into a batch.

## Configuration

Set `sampling.method` to `mlmd` or `udd`; choose `acquisition.method` and its
`policy` independently. `uncertainty` is the built-in acquisition method, with
`stop` and `continue` policies. Existing thresholds, scores, and candidate
limits retain their meaning under either driver.

Set `dynamics.driver` to `ase` or `batched`. Batched execution requires an
explicit `dynamics.backend` of `torchsim` or `alchemi`. ASE execution ignores
inactive batching/backend settings, so changing the driver retains the method
and acquisition selections. The new task defaults to MLMD, uncertainty
acquisition, stop, and ASE when these selections are omitted.

`inference.interface` selects native ensemble inference or ASE compatibility;
`inference.loader` and `inference.options` identify the model loader. Existing
`batched_calculator`, `alchemi_calculator`, `ase_calculator`, and corresponding
options are accepted as aliases when they agree with explicit new settings.
Contradictory duplicate selections fail validation.

The base ASE/ASE-calculator route does not require Torch or a GPU dynamics
package. Native HIPPYNN and batched engines import their optional stacks only
on the selected worker route. ASE inference under batched dynamics remains a
sequential compatibility path and is recorded as such.

See [the complete examples](../examples/unified_sampling/README.md) for the
public task configuration, analytic ASE fixture, native HIPPYNN configuration,
and acceptance commands.

## Scientific behavior

New unified configurations use the current portable screening, acquisition,
ranking, state-selection, and provenance behavior for both execution drivers.
This includes rejecting invalid geometry before acquisition and keeping the
same uncertainty checks and schedule timing. Switching a driver changes the
numerical engine rather than selecting different acquisition rules.

UDD preserves the existing signed bias exactly. With member energies `E_m`,
member forces `F_m`, population energy standard deviation `sigma_E`, and
configured weight `w`:

```text
E_propagated = mean(E_m) + w * sigma_E + E_confinement
F_bias = w * mean((E_m - mean(E_m)) * (F_m - mean(F_m))) / sigma_E
F_propagated = mean(F_m) + F_bias + F_confinement
```

The force bias is zero when `sigma_E <= udd_bias_eps`. Each system uses its
own ensemble moments; padding never contributes to those reductions.
Confinement does not contribute to ensemble uncertainty. Unbiased physical
state forces and uncertainty remain available for acquisition and screening.
A positive `w` adds an uncertainty energy penalty under the existing sign
convention. No change of bias sign is introduced by the refactor.

Molecular excited-state and LCM behavior remain available with MLMD through
both execution drivers. UDD may bias a selected physical surface when its
member energies and forces are exposed. UDD together with LCM switching is
rejected until that composition has a separate scientific definition.

Temperature and density schedules remain independent per system. Density
scaling is isotropic, preserves velocities and cell shape, and refreshes
geometry-dependent forces. Batched periodic support covers fully periodic
ground-state checkpoints with the required cell inputs. ASE boundary and
constraint capabilities are validated separately from batched-only limits.

## Extension and compatibility

Method and acquisition plugins may be selected by import path. A method owns
its algorithm state and declares the capabilities it requires. The extension
interface permits grouped evaluations and optimization progress without
requiring an MD clock. This provides a route for later NEB/dimer integration;
it does not automatically make arbitrary existing Python samplers batched.

A method import path identifies a zero-argument factory or class implementing
`create_program(context)`. Its program implements `run()` and returns normal
ALF molecule outputs. It declares `required_capabilities`, such as
`grouped_evaluation` or `dynamics`, so unsupported combinations fail before
the program runs. The context exposes:

- `evaluate_group(atoms_list)`: ordered ensemble predictions and per-system
  diagnostics for a group of proposed configurations.
- `acquire(SamplingObservation(...))`: shared screening and acquisition for
  one proposal, returning a candidate, stopping decision, or rejection.
- `finalize(candidates)`: final acquisition selection and normal ALF output
  construction with provenance.

`SamplingObservation` contains the parent system index and molecule, proposed
atoms, diagnostics, a progress step, and a method-specific progress dictionary.
An optimization method can report `optimization_iteration`; a trajectory can
report `time_ps`. The non-MD
[`ScaledProposalMethod` example](../examples/unified_sampling/plugins.py)
demonstrates grouped inference and candidate handling without an MD clock.

An acquisition import path identifies a zero-argument factory or class with
`decide(observation, context) -> AcquisitionDecision` and
`select(candidates, context, final=False) -> candidates`. Decisions carry
capture/stopping flags and method-specific metadata. The built-in uncertainty
policy provides the existing threshold and ranking behavior.

Unsupported method, model, and driver combinations fail explicitly rather
than selecting a different algorithm or silently falling back to ASE.
Existing reactive entrypoints continue to run their established ASE optimizer
workflows. Full reactive batching, NPT, periodic excited states, and MI350
hardware acceptance remain separate milestones.

Task snapshots and reload safeguards include the resolved method, acquisition,
driver, engine, and inference identity. Replica accounting uses actual task
widths, including partial batches and empty results. Existing stage files,
legacy buffer formats, and recovery journal payloads remain readable without
rewriting pinned configurations.
