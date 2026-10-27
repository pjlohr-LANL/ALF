# Darwin GPU validation, October 27, 2026

These runs evaluate the `feature/portable-batched-sampling` branch in
`pjlohr-LANL/ALF`. They cover GPU execution and integration correctness using
saved HIPPYNN checkpoints. MI350-family hardware and a production sampling
campaign remain separate acceptance steps.

## Hardware and environments

| Target | Partition and node | Engine | PyTorch | Engine dependencies |
| --- | --- | --- | --- | --- |
| AMD Instinct MI250X/MI250, gfx90a | `shared-gpu-amd-mi250`, cn783; rerun cn782 | TorchSim | `2.9.0+rocm6.4`, HIP `6.4.43484-123eb5128` | TorchSim 0.6.2, ops 0.4.0 |
| NVIDIA A100-SXM4-40GB | `shared-redstone`, cn0 | ALCHEMI | 2.9.0, CUDA 12.8 | ALCHEMI 0.1.0, ops 0.3.1 |
| NVIDIA A100-SXM4-40GB | `shared-redstone`, cn0 | TorchSim | 2.9.0, CUDA 12.8 | TorchSim 0.6.2, ops 0.4.0 |

All environments used Python 3.12.12, HIPPYNN 0.1.3.post0.dev14, ASE 3.27.0,
NumPy 2.3.0, h5py 3.15.1, and Warp 1.13.0. The ALCHEMI environment retained
PyTables 3.10.2; the two TorchSim environments used PyTables 3.11.1. The
original ALCHEMI environment was preserved, with isolated CUDA and ROCm
TorchSim environments stored on the shared filesystem.

HIPPYNN custom kernels were disabled for both GPU vendors
(`HIPPYNN_USE_CUSTOM_KERNELS=False`). These numbers describe the portable
PyTorch implementation. They do not measure NVIDIA-specific kernel tuning.
The ROCm loader also selects portable PyTorch operations itself.

Darwin's tested partitions did not advertise GPU GRES. The runs requested
exclusive whole nodes through Slurm, then restricted the worker to one
visible GPU. An unused second GPU in the same NVIDIA allocation ran the
precision diagnostic and targeted reruns. Normal worker configuration must
continue honoring the scheduler's visibility mask.

## Checkpoints and acceptance criteria

The trained fixture is a four-member, four-state HIPPYNN ensemble from the
completed `keto_rdkit_alf_smoke_seed42_v2` pilot, trained on 256 frames.
Eight representative 15-atom structures were copied from its completed
checkpoint, preserving atom identities. The source HDF5 rows were
0, 17, 35, 52, 70, 87, 105, and 123. Checkpoint and structure hashes are
recorded in the artifact manifests. This is an integration fixture; its
potential accuracy is not established by these tests.

Additional saved two-member molecular and triclinic periodic HIPPYNN
checkpoints are untrained mechanics fixtures. Analytic harmonic systems
exercise mixed atom counts independently of checkpoint loading.

Each inference/dynamics case uses batch sizes 1, 2, 8, 32, and 50, five
warmup steps, 20 timed inference calls, and 50 timed dynamics steps at
0.1 fs and 20 K. Acceptance compares individual and batched energies,
forces, and uncertainty statistics, then checks replica freezing and
periodic cell scaling with velocity and kinetic-energy preservation.
The acquisition-policy case uses the trained ensemble to exercise stop,
continue, topology checks, gap diagnostics, LCM entry, physical-force
rejection, and candidate provenance.

Float32 comparisons use `rtol=1e-5, atol=1e-5`, as selected by the user.
The condition is `abs(batch - individual) <= atol + rtol * abs(individual)`;
absolute tolerances use eV for energy quantities and eV/Å for force
quantities. This criterion changes no sampling or physical-force thresholds.
The harness records both tolerances and the observed errors in JSON.

The initial `atol=1e-6` rejected one state-3 force component on AMD by
`1.3262e-6` eV/Å; an earlier NVIDIA run showed the same behavior. A reference
calculation promoted the same float32 coordinates and checkpoint weights to
float64: batched and individual forces then agreed within `6.11e-15` eV/Å.
Float32 batch and individual errors against that reference were both about
`9e-6` eV/Å. This identifies arithmetic roundoff rather than an atom-order
or batching discrepancy. Original failing logs are retained.

## Results

| GPU and engine | Result | Evidence |
| --- | --- | --- |
| MI250X/MI250, TorchSim | 9 of 9 cases accepted | Job 17359798 passed 8 cases at the original tighter tolerance; state 3 passed in job 17359884 at the approved criterion |
| A100, ALCHEMI | 9 of 9 cases passed | Job 17359826 passed the original criterion; the explicit-criterion state-3 rerun also passed |
| A100, TorchSim | 9 of 9 cases passed | Job 17359826 passed the original criterion; the explicit-criterion state-3 rerun also passed |

The NVIDIA matrix completed with exit code 0, as did both targeted reruns.
Across accepted matrix results, the maximum energy difference was
`1.91e-6` eV and the maximum force difference was `2.15e-6` eV/Å. All three
GPU/engine combinations returned two first-step stop candidates, twelve
continue candidates including ten LCM candidates, and two deliberate
physical-force rejections, with the expected replica provenance.

The full CPU suite after the adapter fix passed in both environments:
508 passed / 14 skipped with ALCHEMI, and 521 passed / 1 skipped with
TorchSim. Running the suite from this checkout requires
`PYTHONPATH` to identify the fork so subprocesses do not import another ALF
installation.

Testing also found and corrected an ALCHEMI adapter bug: neutral calculators
were receiving the engine's native batch object. Commit `2871977` supplies
the packed interface, preserves cells and PBC, and refreshes geometry on
each evaluation. Commit `229cf5b` makes the chosen acceptance tolerances
explicit and records them. Frozen source worktrees identify the precise
revision for each run.

The initial `229cf5b` artifacts mistakenly label an integer species-encoder
parameter as the inference dtype. Coordinates and predictions used float32.
The final harness records the coordinate dtype instead; a trained-checkpoint
CPU check verifies `torch.float32` in the corrected report. The original GPU
artifacts are preserved, including that metadata error; prediction errors,
tolerances, timings, and memory measurements are unaffected.

## Throughput and memory

This table uses the trained four-member, four-state ensemble on selected
state 0. Inference includes the surfaces required by the excited-state
diagnostics. Memory is the larger of the inference and dynamics peak Torch
allocations, in MiB; it excludes external allocator and runtime overhead.
These are single-run measurements with portable kernels and short timing
windows, without repeated-run confidence intervals.

| GPU | Engine | Systems | Inference systems/s | Dynamics system-steps/s | Peak MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| MI250X/MI250 | TorchSim | 1 | 5.0 | 5.8 | 374.9 |
| MI250X/MI250 | TorchSim | 2 | 11.5 | 11.5 | 489.1 |
| MI250X/MI250 | TorchSim | 8 | 45.2 | 45.2 | 1421.6 |
| MI250X/MI250 | TorchSim | 32 | 127.8 | 127.7 | 5126.2 |
| MI250X/MI250 | TorchSim | 50 | 139.3 | 139.1 | 7920.2 |
| A100 | ALCHEMI | 1 | 5.1 | 5.0 | 231.8 |
| A100 | ALCHEMI | 2 | 10.1 | 10.1 | 355.4 |
| A100 | ALCHEMI | 8 | 40.2 | 39.7 | 1286.1 |
| A100 | ALCHEMI | 32 | 153.0 | 151.9 | 4987.0 |
| A100 | ALCHEMI | 50 | 218.4 | 216.6 | 7788.8 |
| A100 | TorchSim | 1 | 5.0 | 5.0 | 242.1 |
| A100 | TorchSim | 2 | 9.9 | 9.9 | 356.4 |
| A100 | TorchSim | 8 | 39.3 | 39.1 | 1289.4 |
| A100 | TorchSim | 32 | 152.6 | 152.0 | 4989.0 |
| A100 | TorchSim | 50 | 218.5 | 217.7 | 7783.6 |

Batching substantially improves systems processed per second in this fixture.
At width 50, the two engines have similar throughput on the A100, where shared
HIPPYNN inference dominates these short trajectories. These results do not
establish a universal engine or hardware ranking.

## Artifacts and repeat procedure

Artifacts live under the fork checkout's ignored `debug` directory:

- `portable_gpu_20261027_8795a40`: source `8795a40`, environment builds,
  fixture hashes, hardware probes, and AMD matrix job 17359798.
- `portable_gpu_20261027_adapter`: source `2871977`, NVIDIA matrix job
  17359826, `results/precision-a100.json`, and final CPU regression logs.
- `portable_gpu_20261027_float32`: source `229cf5b`, the approved criterion,
  AMD state-3 rerun job 17359884, and NVIDIA state-3 reruns in job 17359826.

Each directory retains its `manifest.json`, source worktree, execution
scripts in `jobs`, and per-case JSON/logs in `results`. The original
NVIDIA job 17359797 was superseded after the adapter correction; its
failures are retained for diagnosis and are not counted as acceptance.

For MI350-family hardware, use a supported ROCm/PyTorch environment and
repeat the checkpoint command in the
[acceptance harness instructions](../examples/portable_batched_sampling/README.md)
for all four selected states. Include the molecular, periodic, analytic,
and policy cases from the saved `jobs/run_matrix.py` and
`jobs/policy_acceptance.py`. Retain GPU identity, software versions,
checkpoint hashes, tolerances, throughput, and memory measurements.

Periodic GPU validation here establishes mechanics with an untrained
periodic checkpoint. A trained periodic model in its intended domain,
longer thermal and trajectory-stability evaluation, and actual MI350
execution are still needed before production acceptance. Short stochastic
trajectories across engines or GPU families need not be bitwise identical.
