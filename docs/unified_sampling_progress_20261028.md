# Unified sampling development checkpoint — October 28, 2026

Branch: `feature/unified-sampling-drivers` in `pjlohr-LANL/ALF`.
Work is restricted to the fork. The upstream checkout is untouched.

## Saved implementation

- `92ef67d`: checkpoint of the previously tested unified sampling draft.
- `2e1cb64`: independent execution, force controls, acquisition, opaque model
  references, and durable logical cohorts, with regression tests.
- `b8248cc`: explicit configuration examples, documentation, and mechanics,
  thermal, and cohort-replay acceptance harnesses.
- `3dd2c16`: make the finite bootstrap test fixture return empty results after
  exhausting its seed input while asynchronous QM finishes.

The implementation includes first-trigger selection, pooled scalar top-K,
incremental cohort admission, pinned models/states, deterministic identities,
ordinary and staged recovery, and physical-force screening independent of bias.
Additional recovery review fixed stale HDF5/training/input allocators and
model-generation mixing in waiting physical batches. Generic proposal methods
now use the same ranking ties as the coordinator.

## Completed CPU checks

| Check | Result |
| --- | --- |
| Full fork suite, ALCHEMI environment | 629 passed, 15 skipped |
| Full fork suite, TorchSim environment | 643 passed, 1 skipped |
| Unchanged fork-main/upstream compatibility tests | 47 passed |
| Documentation HTML build | Passed; 21 existing warnings |
| Trained-checkpoint canonical policy preflight, single/ASE on CPU | Passed |

Use an absolute fork `PYTHONPATH` when running the full suite: subprocess tests
change their working directory. Earlier relative-path runs imported another
ALF installation; their failing logs are retained rather than counted as passes.
The finite legacy bootstrap fixture also exposed asynchronous oversubmission;
the fixture correction does not change released driver behavior.

Artifacts are retained under the fork's ignored directory
`debug/strategy_separation_20261028/`. Accepted CPU logs are
`results/pytest-alchemi-accepted.log`, `results/pytest-torchsim-accepted.log`,
and `results/upstream-main.log`.

## Background hardware validation

Hardware validation is **in progress**, not complete. Jobs use the immutable,
clean worktree at `b8248cc`; subsequent changes concern tests/documentation.

| Job | Target | Artifact directory |
| --- | --- | --- |
| 17364157 | AMD MI250X/MI250, cn782, TorchSim | `results/amd/17364157/` |
| 17364160 | NVIDIA RTX A5000, cn4074, ALCHEMI and TorchSim | `results/nvidia/17364160/` |

The pending A100 job 17364156 was canceled after an idle RTX A5000 became
available. Both replacement allocations execute independently of this chat.
Each matrix retains per-case JSON/logs and an incrementally written
`summary.json`. The manifest records source identity, checkpoint fingerprints,
script hashes, partitions, and the approved float32 tolerances
`rtol=1e-5`, `atol=1e-5`.

Cases cover MLMD and UDD, supported single/batched engine combinations,
analytic and saved molecular/periodic checkpoints, four trained excited-state
surfaces, thermal statistics, policies, and cohort replay at physical widths
1, 2, 8, 32, and 50. Hardware correctness and production-potential accuracy
must be reported separately. MI350 hardware remains untested.

## Resume checklist

1. Inspect both Slurm jobs and their per-case summaries. Investigate failures
   before writing a final hardware acceptance report; retain original logs.
2. Finish the training-boundary review. In particular, the new
   `submit_training` handoff should tolerate callables whose signature omits
   `ML_config`, and existing `staged_workflow.verify_final_model` still assumes
   `model-NN` members whenever `n_models` is present. Move model-family artifact
   validation behind the adapter without changing legacy trainer conventions.
3. Add focused regression coverage for any remaining changes. Re-run affected
   checks; repeat complete suites if substantive changes require it.
4. Summarize throughput, memory, thermostat statistics, and cohort latency from
   the retained JSON. Publish the final validation report to the fork only.

No claim of completed GPU acceptance or production-potential accuracy is made
at this checkpoint.
