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
    "F1": ["state_1_forces", "atomic", 1.0]
  }
}
```

The CPU and GPU tasks share `QM_config.json`. A positive timeout runs PySEQM
in an isolated child process so a stalled solve cannot hold the Parsl worker
indefinitely. This milestone does not batch multiple ALF candidates into one
QM task.
