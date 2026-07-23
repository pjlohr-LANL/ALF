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

For excited-state checkpoints, set `model_mode` to `excited_state`, provide
contiguous `sE#`/`F#` entries in the master `properties_list`, and have the
builder place `selected_state` in each molecule's metadata.

## Darwin GPU acceptance profile

Use `alframework.parsl_resource_configs.darwin.config_atdm_ml_short` for the
first cluster check and the `atomistic` environment containing
`nvalchemi-toolkit 0.1.0`. Start with ground-state `model_mode`, a batch size of
at least two, and `uncertainty_policy: "stop"`. Confirm that:

1. the sampler worker sees CUDA and loads one HIPPYNN ensemble per task;
2. replicas that cross an uncertainty threshold stop independently;
3. every returned candidate is submitted to QM; and
4. incomplete compatibility buckets remain visible in `status.txt`.

Repeat with `uncertainty_policy: "continue"` and verify that no more than the
batch-wide `return_top_n` candidates are returned in descending normalized
uncertainty order. The excited-state acceptance run uses the same profile with
`model_mode: "excited_state"` and a builder that supplies one common
`selected_state` per full compatibility batch.
