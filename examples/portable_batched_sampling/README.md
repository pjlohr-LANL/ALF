# Portable batched sampling

Use `alframework.samplers.batched_sampling.batched_sampling_task` as the master
configuration's `sampler_task`, and use `sampler_config.json` as a starting
point. The same configuration selects ALCHEMI with
`"dynamics": {"backend": "alchemi", ...}` or TorchSim with `"backend": "torchsim"`.
Choose temperature, uncertainty thresholds, and trajectory duration for the
model and application; the supplied numbers illustrate configuration syntax.

Both backends consume mixed-size systems through native HIPPYNN inference.
The ML configuration must identify `species_key`, `coordinates_key`,
`energy_key`, and `force_key`; checkpoints must export the requested energy
and force outputs.
Compatible systems share periodic boundary conditions and the selected
electronic state. The driver submits full batches immediately and submits a
smaller batch after `max_wait_seconds`; `batch_size` is an upper bound in this
mode. Old `alchemi_baoab` configurations keep their strict-full behavior.

For fully periodic ground-state models, the HIPPYNN checkpoint must contain a
cell input and the ML configuration should identify it with `cell_key`. Each
system retains its own cell. Density schedules use the existing `end_dens`,
`amp_dens`, and `per_dens` ranges in g/cm³. To enable a water-density schedule,
for example, replace the three null entries with `[0.9, 1.1]`, `[0.0, 0.05]`,
and `[5.0, 50.0]`. Cell scaling preserves fractional coordinates and velocities.
Molecular excited-state models retain state selection, gap diagnostics and
LCM switching. Periodic excited-state models and NPT are outside this example.

## Separate worker environments

Keep the existing ALCHEMI worker environment. Create a separate Python 3.12+
environment for TorchSim, install the cluster-supported PyTorch build, then
install ALF's optional extra:

```bash
python -m pip install -e '.[torchsim_dynamics]' \
  -c examples/portable_batched_sampling/torchsim-constraints.txt
```

Run this from the ALF checkout with the new environment active. For AMD, the
installed Torch must be a ROCm build: `torch.version.hip` must be non-null and
`torch.cuda.is_available()` must be true on an allocated GPU node. PyTorch uses
`cuda:0` as the device name on ROCm too. Configure the Parsl sampler worker's
initialization to activate the selected environment before loading ALF.

The constraint files record the initial CPU validation profiles. ALCHEMI uses
toolkit 0.1.0 and ops 0.3.1; TorchSim 0.6.2 uses ops 0.4.0. The acceptance
artifact records actual package versions, Torch CUDA/ROCm versions, and GPU
identity for each cluster run. Install the checkpoint-compatible HIPPYNN
version used by your workflow in both environments.

## Acceptance and performance

Run the same harness separately in each environment and GPU allocation. It
compares batch predictions with individual predictions, exercises independent
replica freezing and periodic cell scaling, and records throughput and peak
allocated Torch memory. It does not submit scheduler jobs or QM calculations.

Float32 prediction and uncertainty comparisons use `rtol=1e-5` and `atol=1e-5`:
`abs(batch - individual) <= atol + rtol * abs(individual)`. The absolute
tolerance is in eV for energies and energy uncertainty, and eV/Å for forces
and force statistics. These are acceptance tolerances, independent of the
sampler's acquisition and physical-force thresholds. Override them with
`--parity-rtol` and `--parity-atol`; each artifact records the actual values,
inference dtype, and maximum observed energy, force, and diagnostic errors.
The original `atol=1e-6` exposed ordinary float32 roundoff on a trained
four-state checkpoint: a force component differed by about `1.4e-6`
eV/Å, while the same calculation in float64 agreed below `7e-15` eV/Å.

An explicit CPU mechanics check requires no trained model:

```bash
python examples/portable_batched_sampling/benchmark.py \
  --backend torchsim --device cpu --periodic \
  --batch-sizes 1 2 8 --output /tmp/torchsim-cpu.json
```

The analytic harmonic model checks integration units and cell handling. Use
trained HIPPYNN checkpoints and representative structures for model performance
and deployment acceptance:

```bash
python examples/portable_batched_sampling/benchmark.py \
  --backend torchsim --device cuda:0 \
  --master /path/to/run/master_config.json \
  --ensemble-directory /path/to/run/models/model-0003 \
  --structures /path/to/representative.extxyz \
  --batch-sizes 1 2 8 32 50 --output /path/to/results/torchsim-gpu.json
```

For a state-cycling excited-state configuration, supply `--selected-state N`
and repeat for each surface. Repeat the command with `--backend alchemi` in
its NVIDIA environment to compare engines on the same checkpoint and GPU.
Structures must have species and boundary conditions supported by that
checkpoint; multiframe files can contain different atom counts and ordering.

The first AMD acceptance target is Darwin MI250X when available. Repeat on
the production MI350-family hardware with its supported ROCm environment.
CPU results contain `"hardware_validation": false`; they do not certify AMD
or NVIDIA execution. Keep the JSON artifacts with the checkpoint and exact
configuration used for each hardware run.
