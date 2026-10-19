# Keto: RDKit → ALF → GALF

This self-contained, four-state example prepares new runs with the production
fork. It imports no historical labels or models unless you explicitly request
an ALF handoff. Configure Parsl and the software environment before launching.

```text
KDG ladder → geometry screens → stratified FPS → single-point QM labels
  → fresh ensemble → ordinary ALF → frozen first 40k (including seed)
  → fresh ensemble → GALF → frozen first 50k and 60k (including inherited data)
```

These are explicit stages. ALF does not stop or switch automatically at a frame
budget. No monitoring command submits tasks or stops a running driver.

## Software and resources

Use a working environment containing ALF, NumPy, H5py, ASE, Parsl, RDKit,
Torch, HIPPYNN with HIP-HOP support, `nvalchemi-toolkit` (the fork's supported
0.1 series), and the PySEQM excited-state implementation. RDKit preparation
runs on CPU; training, ALCHEMI dynamics, and PySEQM labeling use GPUs.

Activate that environment before executing the commands below. From the
repository root:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
EXAMPLE="$PWD/examples/keto_rdkit_alf_galf"
RUN_ROOT=/path/to/new/keto_experiment
```

Edit the copied `parsl_configs.py` in each prepared run, or set its environment
variables. The template has a local CPU builder and independent Slurm ML,
sampling, and QM executors. GPU executor pools scale from zero; importing the
configuration does not submit allocations. It does not reproduce the
prototype's shared two-node allocation.

| Setting | Purpose / default |
| --- | --- |
| `ALF_ACCOUNT` | Your Slurm account; unset by default |
| `ALF_ML_PARTITION`, `ALF_SAMPLER_PARTITION`, `ALF_QM_PARTITION` | Site partitions; default `ml4chem` |
| `ALF_ML_QOS`, `ALF_SAMPLER_QOS`, `ALF_QM_QOS` | Optional site QoS |
| `ALF_ML_SCHEDULER_OPTIONS`, `ALF_SAMPLER_SCHEDULER_OPTIONS`, `ALF_QM_SCHEDULER_OPTIONS` | Site GPU/resource directives |
| `ALF_ENV_ACTIVATION` | Shell commands activating the worker environment, including any CUDA module |
| `ALF_*_MAX_BLOCKS`, `ALF_*_WALLTIME` | Separate executor allocation limits; default one block, four hours |
| `ALF_GPUS_PER_NODE` | GPUs on each allocation; default 4; match `gpus_per_node` in the master config |
| `ALF_PYTHON` | Optional Python executable used by `launch.sh` |

For example, set `ALF_ENV_ACTIVATION` to source your Conda initialization and
activate the same environment on workers. Set GPU directives according to your
site; this template deliberately does not assume a particular TRES syntax.
Run only one driver per run directory. A persistent `tmux` session or a suitable
site-specific driver allocation must outlive the worker allocations.

## 1. Prepare and label fresh seeds

```bash
python "$EXAMPLE/prepare.py" init --output-dir "$RUN_ROOT/alf"
python -m alframework.tools.rdkit_initialization \
    --config "$RUN_ROOT/alf/seed_config.json" \
    --output-dir "$RUN_ROOT/alf/seeds"
python "$RUN_ROOT/alf/prepare.py" check \
    --master "$RUN_ROOT/alf/master_config.json" --require-seeds
bash "$RUN_ROOT/alf/launch.sh" > "$RUN_ROOT/alf/driver.log" 2>&1
```

The last command runs indefinitely. Launch it in the persistent session after
configuring Parsl. It first replays the external seeds sequentially for QM
labeling, screens the results, writes `h5store/data-0000.h5`, and waits for all
four initial members to complete before sampling. Replay subsequently reads
this run's labeled H5 store. Never copy the unlabeled `seeds/seed.h5` into the
labeled store.

The seed config selects 5,000 FPS geometries from 50,000 accepted KDG geometries
in six equal-weight strata, using seed 42 and one CPU thread. It enables basic
knowledge, disables experimental torsion preferences, and uses random
coordinates without force-field minimization or RMSD pruning. Heavy-atom,
hydrogen, and torsion ladders and seed-only geometry screens remain editable
in `seed_config.json`; preparation fails rather than silently reducing quotas.

`bootstrap_set` must equal the selected count (default 5,000), not the number
that ultimately survives QM and storage screening. Check selected counts in
`seeds/provenance.json`, convergence/failure in the QM logs, rejected counts in
the bootstrap screening log, and retained counts in the first H5 shard. The
historical retained count of 4,733 is not guaranteed. Rejected seeds are not
refilled automatically. If bootstrap cannot complete or all labels are rejected,
inspect the logs; do not treat an incomplete bootstrap as a ready training set.

## 2. Monitor ALF and create the 40k handoff

```bash
python "$EXAMPLE/prepare.py" check --master "$RUN_ROOT/alf/master_config.json"
python "$EXAMPLE/prepare.py" handoff \
    --source-master "$RUN_ROOT/alf/master_config.json" \
    --output-dir "$RUN_ROOT/galf" --frames 40000
```

Run handoff after at least 40,000 finite accepted rows are present in completed
shards. It can read an active ALF run: only shard IDs strictly below the
status file's `current_h5_id` are eligible. Fingerprints detect changes during
selection/copying. You may keep the ALF branch running as a pure-ALF baseline,
or stop it manually after checking the completed-shard count.

To start from an existing ALF dataset, skip stage 1 and pass that run's master
JSON as `--source-master`. The source must use matching PySEQM AM1/CIS tolerances
and the keto energy offset; a different QM method/offset is rejected to avoid
mixing incompatible labels. It must have four unscaled `sE0`–`sE3` and
`F0`–`F3` properties and explicit `topology_atom_ids`, with the supplied keto
reference's atom identities and connectivity. Untagged same-element layouts
are ambiguous and are rejected. For a verified quiescent store without a
status file, explicitly supply `--closed-before N` (exclusive shard boundary).

Selection follows numeric shard order and stored row order, skips rows with
nonfinite coordinates or any state energy/force, and slices only the required
portion of the final shard. It applies no new geometry or force filtering.
The manifest records source fingerprints, rows, IDs, topology permutations,
seed membership, total available frames, and retained overshoot. Every copied
property and attribute is preserved; missing adjacent gaps are derived in the
copies as energy differences. Existing source files are never rewritten.

The GALF directory contains an independent configuration, H5 store, topology
reference, and manifest. No old models, status, logs, or sampling outputs are
imported. Existing destination directories are refused.

## 3. Train and run GALF

Configure the copied Parsl file for the new run, then:

```bash
python "$RUN_ROOT/galf/prepare.py" check --master "$RUN_ROOT/galf/master_config.json"
bash "$RUN_ROOT/galf/launch.sh" > "$RUN_ROOT/galf/driver.log" 2>&1
```

ALF discovers the frozen H5 pool and trains a new `models/model-0000`; no seed
labeling is repeated. The normal driver waits for successful completion of
all ensemble members before sampling. In a separate terminal, after promotion:

```bash
python "$RUN_ROOT/galf/prepare.py" check \
    --master "$RUN_ROOT/galf/master_config.json" --require-trained
```

This checks all four initial member checkpoints and summaries, verifies that
each read the frozen 40k input pool, and rechecks initial H5/ML-config hashes.
The summaries also report the pool remaining after normal training filters.
It validates saved artifacts and provenance, not model accuracy. Routine
retraining continues as further accepted labels are stored.

## 4. Freeze 50k/60k checkpoints and stop manually

```bash
python "$EXAMPLE/prepare.py" check --master "$RUN_ROOT/galf/master_config.json"
python "$EXAMPLE/prepare.py" snapshot \
    --source-master "$RUN_ROOT/galf/master_config.json" \
    --output-dir "$RUN_ROOT/checkpoints/hybrid_50k" --frames 50000
python "$EXAMPLE/prepare.py" snapshot \
    --source-master "$RUN_ROOT/galf/master_config.json" \
    --output-dir "$RUN_ROOT/checkpoints/hybrid_60k" --frames 60000
```

Execute each snapshot only after its target is available. They contain exactly
40k ALF + 10k/20k GALF, including retained seed frames. Membership is nested.
The copied handoff boundary identifies the phase even though generating-model
IDs restart in the new run. No frame is deduplicated by ID.

The same command freezes a continued-ALF baseline from the ALF master. Each
checkpoint has independently reloadable H5 files plus a manifest with hashes;
the source and destination hashes can be verified with the example's `verify`
helper. Retained overshoot is separate from selected counts. In-flight and
completed-but-unsaved work is not included in those counts or overshoot.

Stop the driver manually at the desired completed-shard milestone. Coordinate
shutdown with your site's Parsl/Slurm procedures and inspect outstanding worker
allocations; this example adds no draining or stopping orchestration. Capture
QM attempts, failures, screening rejections, and runtime from the logs when
comparing cost. `check` exposes the latest storage-screen summary and task
failures, not a cumulative convergence/rejection ledger.

## Scientific settings and compatibility

- Four-state PySEQM AM1/CIS energies and forces, with offset
  **−1391.451136550848 eV**; gaps are differences of these stored state energies.
- Four HIP-HOP members: 145 features, 20 sensitivities, two interaction layers,
  three atom layers, `l_max=3`, `n_max=4`, and reference cutoffs/training settings.
- Gap loss and gap-uncertainty acquisition are disabled. Explicit `dE01`,
  `dE12`, and `dE23` properties provide switching and diagnostics only.
- Both samplers continue after an uncertainty crossing and rank candidates
  using the supported sum of normalized E/F uncertainties. With cutoffs
  0.001 eV and 0.01 eV/Å and equal normalized weights, ranking is proportional
  to the prototype's `uE + 0.1*uF`. Metrics follow the production adapters.
- At most 500 candidates per batch; no additional per-replica cap. Sampling
  cycles states 0–3 in batches of 50, at 300 K, 0.1 fs, up to 2 ps, with checks
  every 10 steps and a 0.1 ps minimum acquisition time.
- GALF uses adjacent pairs containing the selected state, threshold 0.1 eV,
  α=0.05 eV, σ=3.5, and one-way `stay_fixed` switching. Each replica switches
  independently; the prototype's global batch trigger is not reproduced.
- Sampling rejects excessive ensemble-mean **physical** forces on any state at
  16 eV/Å. LCM/confinement forces remain propagation diagnostics and do not
  enter that cutoff. Bounded initial/last-valid/rejected audits are written
  under `sampling/force_screen_audit` without full trajectories.
- Post-QM force, minimum-distance (0.7 Å), and topology screens are separate.
  Optional training component-outlier filters remain disabled by default.
- Runtime topology checks use the fork's supported bond windows and extra
  connectivity checks. The seed's 1.4× nonbonded clearance is a seed-generation
  screen; the prototype's additional runtime nonbonded option is not supported.

This is a compatible production workflow, not an identical reproduction of the
prototype's batching, acquisition adapters, or scheduler allocation.

## Files and restart limitations

`init` and `handoff` copy all configuration/templates and resolve run-local
paths. Generated H5 stores, models, status, scratch, QM logs, audits, and Parsl
run information remain under each run root. Keep source and snapshot manifests
alongside the data for reproducibility.

Use the same run master for an ordinary resume. Do not copy status between
runs. The unchanged driver does not provide transactional recovery for
interrupted bootstrap, in-flight work, or partially written model directories.
Inspect those artifacts before restarting; do not launch a second driver or
assume a model directory alone proves training completed. The checker rejects
an unpromoted initial model directory rather than permitting accidental reuse.

For reduced CPU validation, use the focused example, workflow, and actual-driver
tests from the repository's test suite. The driver regression uses local Parsl
threads, mocked QM/training, and a scripted dynamics backend; it exercises
seed labeling, screening, replay, fresh handoff, LCM switching, and nested
checkpoints without submitting jobs or claiming GPU performance validation.
