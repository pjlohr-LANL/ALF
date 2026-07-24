# Six-state keto PySEQM production replica

This directory is a fresh, independent reproduction of the Darwin workload
in:

```text
/vast/home/pjlohr/github/ALF/production/6_states_EF_300_2_nodes_topology
```

It uses this branch's streamlined HDF5 replay builder, six-state HIPPYNN
trainer, batched ALCHEMI sampler, convergence-safe PySEQM interface, and
independently scaling Parsl executors. The active source run is never used for
status, models, or output.

## Workflow

```text
copied data-0000.h5
  -> four-member six-state HIPPYNN ensemble
  -> deterministic HDF5 geometry replay
  -> state-specific ALCHEMI batches
  -> screened PySEQM labels
  -> additional HDF5 shards and retraining
```

The flattened property contract is `sE0` through `sE5` and `F0` through
`F5`. Gap targets, UDD, gap diagnostics, and gap seeking are intentionally
disabled for this first production-compatible run.

## Seed data and topology

Only the first source shard belongs in this fresh run:

```bash
mkdir -p h5store
cp --reflink=auto \
  /vast/home/pjlohr/github/ALF/production/6_states_EF_300_2_nodes_topology/h5store_5k_3node_6state_E_F_topology/data-0000.h5 \
  h5store/data-0000.h5
sha256sum \
  /vast/home/pjlohr/github/ALF/production/6_states_EF_300_2_nodes_topology/h5store_5k_3node_6state_E_F_topology/data-0000.h5 \
  h5store/data-0000.h5
```

Both hashes must be:

```text
f47183504f50b191bb4dbee5ef1df13c2b83e76771d7fe30aaa9f774b83a9714
```

The copied shard contains 4,990 structures of one 15-atom keto composition.
ALF HDF5 stores atoms in stable H/C/O order. The replay builder restores the
canonical O/O/C/... order from `topology_atom_ids`, or infers the same mapping
for later ALF-written shards. This makes sampler inputs match
`keto_form_coords.xyz` while every training shard keeps one storage order.

The topology reference hash must be:

```text
254d4cd1674cd353606d12c6b4185d1f9b5e29d323500c9d292f2c66c37d211a
```

## Production settings

- Four HIPPYNN members predict energies and forces for all six states.
- Replay submits 50 structures at a time.
- ALCHEMI uses strict batches of 50 and cycles through states 0–5.
- At most 100 sampler replicas are active, matching two full batches.
- PySEQM labels one molecule per task and uses the production AM1 energy
  offset.
- Candidates and completed labels are screened at `0.7 Å`, `16 eV/Å`, and
  against the fixed keto topology.
- After 2,000 completed QM tasks, accepted labels are screened and stored; an
  all-rejected batch creates no shard and does not advance training.

`master_config_existing_h5.json` is retained as a compatibility alias of
`master_config.json`.

## Darwin resources

`parsl_configs.py` requests independent allocations only when work is queued:

| Stage | Executor | Partition | Workers/node | Maximum blocks |
| --- | --- | --- | ---: | ---: |
| Training | `alf_ML_executor` | `ml4chem` | 1 | 1 |
| Sampling/replay | `alf_sampler_executor` | `shared-gpu-ampere` | 4 | 2 |
| PySEQM | `alf_QM_executor` | `shared-gpu-ampere` | 4 | 2 |

The production defaults are account `y2020-bf`, CUDA `12.2.2`, and:

```text
/vast/home/pjlohr/.conda/envs/atomistic
```

All remain overridable through the `ALF_DARWIN_*` environment variables.

## Stage checks

Run from this directory, in order:

```bash
source /projects/opt/centos8/x86_64/miniconda3/py312_24.11.1/etc/profile.d/conda.sh
conda activate /vast/home/pjlohr/.conda/envs/atomistic
export PYTHONPATH=/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF:${PYTHONPATH:-}

python -m alframework --master master_config_debug.json --test_builder
python -m alframework --master master_config_debug.json --test_qm
python -m alframework --master master_config_debug.json --test_ml
python -m alframework --master master_config_debug.json --test_sampler
```

The debug trainer uses two members and two epochs. Its models and status are
isolated under `models_debug/` and `status_debug.txt`. The builder result must
match the 15-atom reference order; QM must return six finite energies and six
`(15, 3)` force arrays; the sampler must report CUDA execution and selected
state metadata.

## Production launch and restart

After all stage checks pass:

```bash
sbatch submit_darwin.slurm
```

With `h5store/data-0000.h5` present and no `status.txt`, ALF discovers HDF5
index 1, skips bootstrap QM, and trains `models/model-0000`. The same command
resumes from `status.txt`; do not mix debug and production artifacts or copy
the source production status into this directory.

Darwin's `long` QoS permits a two-day driver allocation. If the driver reaches
that limit, resubmit the same script; ALF resumes from the isolated status,
HDF5, and model paths.

The first acceptance milestone is:

1. Four complete members in `model-0000`.
2. Sampling on every state 0–5.
3. Screened PySEQM candidates.
4. `h5store/data-0001.h5`.
5. Start of `model-0001`.

Generated HDF5 shards, models, status, PySEQM scratch/logs, sampling outputs,
and Parsl run information are ignored by Git.
