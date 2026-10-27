# Portable batched dynamics and inference

The portable sampler separates ALF's scientific sampling policies from the
simulation engine. `alframework.samplers.batched_sampling.batched_sampling_task`
uses either ALCHEMI or TorchSim while sharing ensemble inference, population
uncertainty statistics, candidate ranking, physical-force screening, topology
checks, confinement, gap diagnostics, and replica-local LCM controls.

The native HIPPYNN interface represents atoms without padding and tracks the
system offsets. It pads only the HIPPYNN predictor inputs, then restores forces
to the original atom order before reducing uncertainty or applying controls.
System energies have shape `[ensemble_members, systems]`; forces have shape
`[ensemble_members, total_atoms, 3]`. Different sizes and compositions can share
a task when their checkpoint, selected state, and boundary conditions agree.

The neutral configuration has three pieces:

- `batching` sets the maximum systems per task and the partial-batch policy.
  Timeout batches wait up to five seconds by default; queued work is polled
  even when no new builder output arrives.
- `dynamics.backend` selects `alchemi` or `torchsim`, with worker device and
  seed controls in the same block.
- `batched_calculator` selects the shared native model loader;
  `batched_calculator_options` supplies its options. An existing ASE loader
  remains a sequential compatibility path and is identified in metadata.

Existing ALCHEMI task names, loaders, and `alchemi_baoab` settings remain
available. Legacy configurations preserve strict-full batching. Existing
single-system sampler algorithms continue to receive individual molecule
objects. Replica capacity counts the actual task input width even when
sampling returns no candidates or a controlled run resumes.

Periodic support covers fully periodic ground-state HIPPYNN checkpoints with
their trained cell inputs. Density schedules retain MLMD timing and g/cm³
configuration values. Each active system scales its own cell and atom
positions, retains its velocities, and refreshes forces before the next step.
Molecular excited-state checkpoints retain their existing state and gap
policies. NPT and periodic excited-state training are deferred.

ALF uses Å, eV, amu, fs, and native ASE velocities at the runner boundary.
TorchSim's low-level integrator uses natural eV/Å/amu time; the adapter applies
the fs conversion directly, carries explicit initial momenta, and transposes
ASE row-vector cells into TorchSim column-vector cells. It does not apply the
high-level TorchSim runner's additional picosecond conversion.

Use separate worker environments for the two backends. The initial
TorchSim environment pins `torch-sim-atomistic==0.6.2` and requires Python 3.12+.
On ROCm, the native HIPPYNN adapter uses portable PyTorch kernels; ALF does not
dispatch model neighbor computations to NVIDIA-only TorchSim operations.

See [the portable example and hardware acceptance harness](../examples/portable_batched_sampling/README.md)
for environment constraints, checkpoint-based benchmarks, and MI250X/MI350
acceptance instructions. Hardware claims require a GPU artifact from that
platform; CPU conformance is reported independently.

## Validation on October 27, 2026

The fork baseline was synchronized before feature development. Production
includes the three recovery fixes through `c364fa8`, and fork `main` includes
upstream `198f125`. Feature development is on
`feature/portable-batched-sampling` in `pjlohr-LANL/ALF`.

| Check | Result |
| --- | --- |
| Production baseline before promotion | 463 passed, 1 skipped |
| Complete feature suite, original ALCHEMI environment | 508 passed, 14 skipped |
| Complete feature suite, isolated TorchSim environment | 521 passed, 1 skipped |
| Unchanged upstream tests imported against the fork | 47 passed |
| HTML documentation | Built successfully; existing unresolved API-reference warnings remain |
| Wheel packaging | Built `alf-0.1.0-py3-none-any.whl`, including the optional TorchSim extra |

The original worker environment retained Torch 2.9.0, HIPPYNN
0.1.3.post0.dev14, ALCHEMI 0.1.0, ops 0.3.1, and Warp 1.13.0.
The isolated TorchSim environment used TorchSim 0.6.2 and ops 0.4.0, reusing
the installed CPU-capable Torch/HIPPYNN stack without modifying that environment.
The skipped optional TorchSim cases in the original environment passed in
the isolated environment; one pre-existing optional case remains skipped.

The CLI acceptance harness was executed for both engines with saved two-member
HIPPYNN checkpoints, separately for molecular and triclinic periodic systems,
at batch sizes 1, 2, 8, 32, and 50. These were reproducible, untrained mechanics
fixtures, not production potential models. Maximum batch-versus-individual
errors were `2.39e-7` eV in energy and `5.97e-8` eV/Å in force. Independent
freezing and periodic cell scaling passed. JSON artifacts record the source
revision/dirty status, configuration, checkpoint and structure fingerprints,
package versions, timing, and hardware-validation flag.

No GPU was visible during this validation. NVIDIA throughput, Darwin MI250X
acceptance, and production MI350-family acceptance remain to be run using
trained checkpoints and representative structures. Follow the example harness
instructions, retain the resulting JSON, and compare energy/force/uncertainty
parity, candidate provenance, physical-force rejection, thermal behavior,
throughput, and peak GPU memory before production rollout. Stochastic
trajectories across engines or GPU families need not be bitwise identical.
