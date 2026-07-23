# Single-molecule PySEQM labeling

This example fragment configures the safe single-molecule PySEQM interface.
Set the master configuration's QM task to one of:

```json
{
  "QM_task": "alframework.qm_interfaces.pyseqm_interface.pyseqm_excited_state_task"
}
```

```json
{
  "QM_task": "alframework.qm_interfaces.pyseqm_interface.pyseqm_excited_state_gpu_task"
}
```

Define contiguous flattened properties in the master configuration:

```json
{
  "properties_list": {
    "sE0": ["state_0_energy", "system", 1.0],
    "F0": ["state_0_forces", "atomic", 1.0],
    "sE1": ["state_1_energy", "system", 1.0],
    "F1": ["state_1_forces", "atomic", 1.0],
    "dE01": ["gap_01", "system", 1.0]
  }
}
```

The CPU and GPU tasks share `QM_config.json`. A positive timeout runs PySEQM
in an isolated child process so a stalled solve cannot hold the Parsl worker
indefinitely. This milestone does not batch multiple ALF candidates into one
QM task.

PySEQM SCF convergence is mandatory. ALF checks the Boolean
`driver.notconverged` result before reading any energies or forces. A flagged,
missing, or malformed convergence result rejects the molecule; no partial
state labels enter the HDF5 training store.

Configured gap labels are derived after the state energies are produced:
`dE01 = sE1 - sE0`. Because the same offset is subtracted from both state
energies, it cancels exactly from the gap. Gap properties must be system
properties and use the same storage scale as both state energies.

## Multi-state HIPPYNN training

Point the master configuration at the isolated excited-state trainer:

```json
{
  "ML_task": "alframework.ml_interfaces.excited_state_hippynn_interface.train_excited_state_HIPPYNN_ensemble_task",
  "ML_config_path": "hippynn_config.json"
}
```

The accompanying `hippynn_config.json` trains every contiguous state in
`properties_list`. Each ensemble member contains one shared HipHopNN trunk and
one energy/force head per state. Gap predictions are differences of those
state heads; no independent gap head is added. The HDF5 database names come from the first
entry in each property schema, so the example above creates checkpoint outputs
for `state_0_energy`, `state_0_forces`, `state_1_energy`, and
`state_1_forces`, plus the derived `gap_01` output.

The explicit HDF5 loader infers atom count and allowed species when `n_atoms`
and `network_params.possible_species` are omitted. All input structures must
still have the same exact atomic-number sequence. The loader validates stored
gap values against the corresponding state-energy differences. This trainer
is nonperiodic, requires forces for every state, and rejects independent gap
heads and CSV export.

Model ensemble members—not electronic states—are distributed among the GPUs
visible to `alf_ML_executor`. Every completed model predicts every trained
state. The sampler's `selected_state` chooses which one of those predicted
surfaces drives a dynamics batch; it does not assign that state to a permanent
GPU.
