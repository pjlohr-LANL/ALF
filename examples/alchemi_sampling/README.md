# ALCHEMI sampler configuration

This directory contains a sampler configuration fragment for a nonperiodic
ground-state HIPPYNN workflow. Point `sampler_config_path` at
`sampler_config.json` and select:

```json
{
  "sampler_task": "alframework.samplers.alchemi_sampling.alchemi_sampling_task"
}
```

The HIPPYNN configuration must provide `energy_key`, `force_key`,
`species_key`, and `coordinates_key`. The builder must produce identical atom
order often enough to fill each strict batch. Change `uncertainty_policy` to
`continue` and increase `return_top_n` to collect ranked uncertain frames
without stopping those replicas.

This example selects the native HIPPYNN loader with `alchemi_calculator`. A
different native backend can provide a loader returning
`ALFAlchemiCalculator`; compatible ALCHEMI model ensembles can use
`ALFNativeEnsembleModel`. Native loaders receive the model directory, selected
state, device, ALF configurations, and `alchemi_calculator_options`.

If `alchemi_calculator` is omitted, the sampler instead uses the existing
`ase_calculator` and `ase_calculator_options`. This fallback makes established
ALF model loaders available to ALCHEMI, but evaluates calculators and replicas
sequentially and therefore does not provide native batched inference speed.
Candidate metadata and `status.txt` report which interface and loader were
selected.

Temperature parameters are sampled independently for every replica from
`srt_temp`, `end_temp`, `amp_temp`, and `per_temp`. The sampler intentionally
uses base ALF's legacy MLMD timing: it initializes at schedule time zero, runs
one MD step, and performs the first check with reported time zero before each
later `Ncheck` block. Returned candidates include the applied schedule updates
in `temps` plus their sampled `Tamp`, `Tper`, `Tsrt`, and `Tend`. The default
`friction_per_fs` is the ALCHEMI-unit equivalent of base MLMD's ASE friction of
`0.02`.

For excited-state checkpoints, see `sampler_config_excited_states.json` and
provide contiguous `sE#`/`F#` entries in the master `properties_list`. The
example uses `state_selection.mode: "batch_cycle"`. With a batch size of 50,
the trailing indices of 50 consecutive ALF molecule IDs select state 0, the
next 50 select state 1, and so on before the cycle repeats. Each resolved state
is written into molecule metadata before strict batching.

This works with ALF's existing
`alframework.builders.builders.simple_cfg_loader_task`; no excited-state
builder is required. Use a CFG library with compatible atomic-number ordering,
set `shake` to `0.0` for exact geometry reuse, and keep
`maximum_builder_structures` at `1`. Existing labeled HDF5 shards can be placed
directly in the configured HDF5 store so ALF trains from them before sampling.

```json
{
  "builder_task": "alframework.builders.builders.simple_cfg_loader_task",
  "maximum_builder_structures": 1
}
```

```json
{
  "molecule_library_dir": "fragment_library/",
  "shake": 0.0
}
```

`parallel_samplers` and `minimum_QM` retain trajectory-based meanings with
strict batching. For example, one submitted batch of 50 counts as 50 input
replicas, not one Parsl task. Pending builders and incomplete buffered
structures also count toward `parallel_samplers`, preventing the driver from
oversubscribing builders while it waits for a compatible batch.

The status file reports this under `sampler_capacity` and reports incomplete
bucket counts and ages under `sampler_batching`. Incomplete buffers are
memory-only and are not restored after a driver restart.

Batch size, state-selection policy, model mode, sampler task, and batched versus
legacy mode cannot be hot-reloaded until the current buffer is empty. ALF keeps
the previous complete configuration instead of discarding or reinterpreting
waiting molecules. Already-submitted tasks retain their original replica
widths for capacity and completion accounting. Temperature, threshold, gap,
ranking, and calculator-option changes can still reload while a batch waits.

Explicit molecule metadata takes precedence over the configured policy. Use
`state_selection.mode: "fixed"` with `state_selection.state` for a single
dynamics surface. PySEQM still labels every state in `properties_list`; state
selection controls only the surface used for dynamics and uncertainty.

State cycling does not pin a state to a GPU. Every completed state-specific
batch is an ordinary Parsl task and may run on any available sampler GPU.
Candidate metadata records `selected_state`, `sampler_device`,
`sampler_worker_rank`, and `sampler_visible_device` for acceptance checks.

The excited-state example also enables optional direct gap diagnostics. Add
the corresponding gap property, such as
`"dE01": ["gap_01", "system", 1.0]`, to the master `properties_list`.
ALCHEMI computes each model member's state-energy difference before reducing
the ensemble. Candidate metadata records the gap mean, population deviation,
pair, and model count, but gaps do not affect uncertainty stopping or top-K
ranking.

The example also enables one-way Levine–Coe–Martinez gap seeking. Declare each
adjacent gap needed by the configured states (`dE01`, `dE12`, `dE23` for
states 0–3) and keep the corresponding state energies and forces in
`properties_list`. At an ordinary `Ncheck`, every surviving replica tests only
the adjacent declared pairs that contain its selected state. If its smallest
absolute mean gap crosses `trigger_gap_threshold_eV`, that replica enters the
LCM surface and remains there for the rest of the trajectory. Other replicas
are unaffected.

The triggering frame was produced by the direct selected-state surface, so a
candidate captured at that check records `gap_seeking_current_mode: "direct"`.
Candidates from later checks record `"lcm"` and carry the pair, trigger
gap/step/time, and the single entry event. Uncertainty selection and ranking
remain based on the original selected state. Disable `gap_diagnostics` if
those general diagnostics are not wanted; gap seeking still loads the state
energies and forces it needs.

## Darwin GPU acceptance profile

Use `alframework.parsl_resource_configs.darwin.config_atdm_ml_short` for the
first cluster check and the `atomistic` environment containing
`nvalchemi-toolkit 0.1.0`. Start with ground-state `model_mode`, a batch size of
at least two, and `uncertainty_policy: "stop"`. Confirm that:

1. the sampler worker sees CUDA, reports the native calculator path, and loads
   one HIPPYNN ensemble per task;
2. replicas that cross an uncertainty threshold stop independently;
3. every returned candidate is submitted to QM; and
4. incomplete compatibility buckets remain visible in `status.txt`.

Repeat with `uncertainty_policy: "continue"` and verify that no more than the
batch-wide `return_top_n` candidates are returned in descending normalized
uncertainty order. The excited-state acceptance run uses the same profile with
`model_mode: "excited_state"` and enough pending structures to fill at least
one complete batch for each state that should run concurrently. Confirm that
state-specific batches can run on arbitrary GPU worker ranks.
