# ALF Workflow Deep Dive

This note is for developers who need to modify ALF later. It is based on the current codepath in `alframework`, with examples used only to show how that code is wired in practice. Where the code and docs disagree, this note follows the code.

## 1. What ALF Actually Is

ALF is a config-driven active-learning loop for interatomic potentials. The framework is organized around four conceptual stages:

1. Structure construction (`builders`)
2. ML-driven exploration (`samplers`)
3. QM labeling (`qm_interfaces`)
4. Model training / retraining (`ml_interfaces`)

The true orchestrator is [`alframework/__main__.py`](./alframework/__main__.py). That file:

- loads all JSON configuration
- loads a Parsl resource configuration
- dynamically resolves the builder, sampler, QM, and ML task callables from dotted strings
- maintains coarse run state in `status.txt`
- optionally bootstraps an initial labeled dataset
- runs the steady-state active-learning loop forever

At a high level, the main loop is:

```text
master config
  -> stage configs
  -> Parsl config
  -> task callables
  -> builder
  -> sampler
  -> QM
  -> HDF5 dataset save
  -> ML retrain
  -> update current model
  -> repeat
```

## 2. Runtime Orchestration in `alframework/__main__.py`

### 2.1 Startup and Config Loading

Execution starts with:

```bash
python -m alframework --master master.json
```

`__main__.py` parses:

- `--master`
- `--test_builder`
- `--test_sampler`
- `--test_qm`
- `--test_ml`

It then loads five JSON files:

1. Master config
2. Builder config
3. Sampler config
4. QM config
5. ML config

The helper that matters here is `load_config_file()` in [`alframework/tools/tools.py`](./alframework/tools/tools.py).

Important behavior:

- If a config declares `"master_directory": "pwd"`, ALF resolves it to the current working directory plus a trailing slash.
- Relative entries ending in `dir` are expanded relative to `master_directory`.
- Relative entries ending in `path` are also expanded, and ALF automatically creates a sibling `...dir` key pointing at the containing directory.

That auto-derived `...dir` behavior is why configs can be inconsistent but still work. Examples:

- `molecule_library_path` in builder config becomes `molecule_library_dir`
- `meta_path` in sampler config becomes `meta_dir`
- `plotting_utility` is loaded directly as a dotted Python path, not via a JSON file

This implicit path-to-dir conversion is convenient, but it also hides naming drift between configs and function signatures.

### 2.2 Parsl Setup and Task Resolution

ALF creates four logical queues:

- `builder_task_queue`
- `sampler_task_queue`
- `QM_task_queue`
- `ML_task_queue`

These are not Parsl queues. They are thin wrappers around lists of Parsl futures, implemented by `parsl_task_queue` in [`alframework/tools/tools.py`](./alframework/tools/tools.py).

Parsl itself is configured from a dotted string in the master config:

- `parsl_configuration` for normal execution
- `parsl_debug_configuration` if a test flag is active and a debug config exists

ALF then dynamically resolves the four stage tasks from strings such as:

- `alframework.builders.builders.simple_condensed_phase_builder_task`
- `alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task`
- `alframework.qm_interfaces.orca5_interface.orca_calculator_task`
- `alframework.ml_interfaces.hippynn_interface.train_HIPPYNN_ensemble_task`

The loader is `load_module_from_string()`.

One subtle feature: after loading each task, ALF checks whether a corresponding standby executor exists. If a task declares an executor like `alf_QM_executor`, ALF will also append `alf_QM_standby_executor` if that label exists in the active Parsl config.

Two practical consequences:

- executor choice is partly declared on the task function and partly rewritten by `__main__.py`
- builders and samplers are usually both backed by `alf_sampler_executor`; there is no separate builder executor in the core resource configs

### 2.3 Directory Creation

After loading config, ALF walks all config dictionaries and creates directories for keys ending in `dir`.

This means a config like:

```json
"meta_path": "sampling/"
```

will create `meta_dir` internally and then create that directory on disk before sampling starts.

### 2.4 Restart and `status.txt`

ALF persists coarse run state in `status.txt`. On startup:

- if `status.txt` exists, it is loaded as JSON
- otherwise ALF initializes a fresh status dictionary

Fresh initialization uses `find_empty_directory()` to infer restart position from the filesystem:

- `current_training_id` is the first missing `model_path`
- `current_model_id` is `current_training_id - 1`, or `-1` if no model exists yet
- `current_h5_id` is the first missing `h5_path`
- `current_molecule_id` starts at `0`

It also initializes lifetime failure counters:

- `lifetime_failed_builder_tasks`
- `lifetime_failed_sampler_tasks`
- `lifetime_failed_ML_tasks`
- `lifetime_failed_QM_tasks`

This restart logic is coarse-grained. ALF does not reconstruct in-flight queues. It only infers where the next model, h5 file, and molecule numbering should start.

### 2.5 Test-Mode Flows

Test mode is handled in the same `__main__.py` file instead of separate scripts.

Behavior:

- `--test_builder` runs one builder task and prints the returned `MoleculesObject`
- `--test_sampler` first runs the builder test path, then requires an existing current model, then runs one sampler task
- `--test_qm` runs builder, optional sampler, then one QM task, prints properties, and writes `qm_test.h5`
- `--test_ml` launches one ML task, updates `current_training_id`, and advances `current_model_id` if training succeeded

This is useful because the same config-driven argument injection is exercised in both test mode and production mode.

### 2.6 Bootstrap Path

If ALF starts with:

- `current_h5_id == 0`
- `current_model_id < 0`

then it enters bootstrap mode.

Bootstrap is different from the steady-state loop:

- builders generate structures
- builder outputs go directly to QM
- samplers are skipped entirely

Bootstrap continues until the number of successful QM tasks reaches `bootstrap_set`. ALF then:

1. collects completed QM results
2. writes `h5store/data-####.h5`
3. trains the first model
4. sets `current_model_id` on success

If the first model fails to train, ALF exits and expects user investigation.

### 2.7 Steady-State Active-Learning Loop

After bootstrap, ALF enters an infinite loop with a 60-second sleep between iterations.

The loop does the following in order.

#### Step A: Re-read all config files

Every iteration, ALF attempts to reload the master, builder, sampler, QM, and ML configs. If reload fails, it logs the exception and keeps the previous in-memory configuration.

This means some config changes can be applied mid-run without restarting ALF.

#### Step B: Submit more builders

Builders are submitted while:

- QM queued work is below `target_queued_QM`
- total QM count is below `maximum_completed_QM` if that limit is set
- combined builder + sampler occupancy is below `parallel_samplers`

Builder molecule IDs are formatted as:

```text
mol-<current_model_id>-<running_index>
```

If `maximum_builder_structures` is set, one builder task may reserve multiple molecule IDs at once.

#### Step C: Move completed builders into samplers

Completed builder results are harvested from `builder_task_queue`.

ALF accepts two builder return shapes:

- a single `MoleculesObject`
- a list of `MoleculesObject` instances

Each returned structure becomes a sampler task.

#### Step D: Move completed samplers into QM

Completed sampler results are harvested from `sampler_task_queue`.

For each returned `MoleculesObject`:

- if `structure.get_atoms()` is not `None`, ALF submits a QM task
- if `structure.get_atoms()` is `None`, the sampler found no escalation candidate and the structure is dropped from the labeling path

This `atoms is None` convention is the main bridge between sampling and labeling.

#### Step E: Save QM results and trigger retraining

When:

- completed QM count exceeds `save_h5_threshold`
- no ML task is currently outstanding

ALF:

1. harvests completed QM results
2. writes a new h5 shard using `store_current_data()`
3. increments `current_h5_id`
4. submits one ML task
5. increments `current_training_id`

#### Step F: Promote the new model

When an ML task completes, ALF checks each returned training result:

- if all ensemble members completed
- and the returned training id is newer than `current_model_id`

then ALF promotes that training id to `current_model_id` and resets `current_molecule_id` to `0`.

That reset is important: molecule numbering restarts for each new current model.

#### Step G: Optional plotting and status persistence

If configured, ALF periodically launches a plotting utility in a separate process.

At the end of every master-loop iteration, ALF:

- prints queue status
- writes `status.txt`
- sleeps for 60 seconds

## 3. Core Data Contracts

## 3.1 `MoleculesObject`

The main container is [`alframework/tools/molecules_class.py`](./alframework/tools/molecules_class.py).

`MoleculesObject` stores:

- `atoms`: current `ase.Atoms` payload, or `None` after a sampler decides not to escalate
- `_moleculeid`: unique structure identifier
- `qm_results`: dictionary populated by QM tasks
- `metadata`: free-form dictionary used by builders and samplers
- `converged`: QM convergence flag

The important stage-by-stage contract is:

- Builders create `MoleculesObject`
- Samplers update metadata and either keep `atoms` as the escalated structure or set it to `None`
- QM tasks store results and set `converged`
- Dataset writing only saves converged structures

## 3.2 Stage Interfaces

### Builder Output Contract

Builder tasks may return:

- one `MoleculesObject`
- a list of `MoleculesObject` instances

ALF explicitly supports both shapes.

### Sampler Output Contract

Sampler tasks may return:

- one `MoleculesObject`
- a list of `MoleculesObject` instances for sampler implementations that support multiple candidates

Meaning of `atoms` on return:

- `atoms is not None`: send this structure to QM
- `atoms is None`: nothing needs QM labeling from this trajectory / reaction attempt

Empty lists are treated as "no sampled candidate". This list-return path is primarily used by the excited-state sampler's optional `return_top_n` mode.

### QM Output Contract

QM tasks return the same `MoleculesObject` with:

- `qm_results` populated
- `converged` set to `True` or `False`

Only converged structures are written into h5 datasets by `store_current_data()`.

### ML Output Contract

ML tasks return:

```text
(completed_flags, training_id)
```

where:

- `completed_flags` is usually a list of booleans, one per ensemble member
- `training_id` is the model directory index that was just trained

ALF only promotes `current_model_id` if all completion flags are `True`.

## 3.3 `properties_list`

`properties_list` in the master config is one of the most important cross-stage interfaces.

Each entry maps:

```text
logical property name -> [h5 field name, "system" or "atomic", unit scale]
```

Example:

```json
"forces": ["forces", "atomic", 51.422067090480645]
```

Effects:

- QM tasks must populate the logical property keys expected by the master config
- `store_current_data()` uses the h5 field name for dataset storage
- `"system"` vs `"atomic"` controls whether values are stored once per structure or once per atom
- the scale factor is applied during h5 writing

This is the bridge between raw QM outputs and the ML training dataset schema.

## 4. State and Persistence

## 4.1 `status.txt`

The current status schema is:

| Field | Meaning |
| --- | --- |
| `current_training_id` | Next model directory index to train into |
| `current_model_id` | Latest model ALF considers active for sampling |
| `current_h5_id` | Next h5 shard index to write |
| `current_molecule_id` | Next per-model molecule counter used in generated IDs |
| `lifetime_failed_builder_tasks` | Cumulative builder failures ALF thinks it has seen |
| `lifetime_failed_sampler_tasks` | Cumulative sampler failures |
| `lifetime_failed_ML_tasks` | Cumulative ML failures |
| `lifetime_failed_QM_tasks` | Cumulative QM failures |

ALF updates `status.txt`:

- once after initialization
- during tests when ML ids change
- during bootstrap
- once per steady-state loop iteration

## 4.2 Model Directories

Model directories live under a pattern such as:

```text
models/model-0000
models/model-0001
...
```

How those directories are used depends on the ML backend:

- HIPPYNN stores ensemble members under `model-####/model-00`, `model-01`, etc.
- NeuroChem writes an ANI ensemble into the top-level model directory

The sampler always references `model_path.format(current_model_id)`.

## 4.3 HDF5 Dataset Shards

ALF writes labeled data to:

```text
h5store/data-0000.h5
h5store/data-0001.h5
...
```

`store_current_data()`:

- drops unconverged structures
- groups structures by empirical formula
- sorts atoms by atomic number before writing
- writes ANI-style arrays via `pyanitools`

The ML interfaces read these h5 directories directly.

## 4.4 Sampling Metadata

Sampling metadata can include:

- `metadata-<moleculeid>.p` pickle files
- `metadata-<moleculeid>.traj` trajectory files

These are written by `mlmd_sampling.py` if `meta_dir` and `trajectory_interval` are set.

The plotting utility in [`alframework/tools/plotting.py`](./alframework/tools/plotting.py) reads these metadata pickles to generate monitoring plots.

## 4.5 QM Scratch Directories

QM tasks create per-structure scratch folders under `QM_scratch_dir`, typically:

```text
<QM_scratch_dir>/<moleculeid>
```

The exact files inside depend on the QM backend:

- ORCA inputs and logs
- VASP input/output files
- SIESTA input/output files
- generic ASE calculator scratch

Scratch reuse is intentionally blocked in several interfaces by raising if the directory already exists.

## 5. Infrastructure Helpers That Matter for Modifications

## 5.1 `build_input_dict()`

`build_input_dict()` is one of the most important helpers in ALF.

It inspects a callable signature and fills parameters by matching names against a list of dictionaries in order.

This is how ALF injects things like:

- `builder_config`
- `sampler_config`
- `QM_config`
- `ML_config`
- `model_path`
- `current_model_id`
- `gpus_per_node`
- `properties_list`
- any auto-derived `...dir` entries

This means ALF is extremely sensitive to parameter names. If you rename a task parameter without updating config keys or call-site dictionaries, wiring can silently change.

## 5.2 `parsl_task_queue`

`parsl_task_queue` is a convenience wrapper over a list of Parsl futures.

It supports:

- `add_task()`
- `get_number()`
- `get_running_number()`
- `get_completed_number()`
- `get_exec_done_number()`
- `get_task_results()`
- `print_status()`

Important design point:

- this is not a persistent queue
- results are removed from the internal list once harvested
- ALF relies on polling these queues every loop iteration

## 5.3 `store_current_data()`

`store_current_data()` is the point where QM results become the training dataset.

It:

- enforces `MoleculesObject`
- checks `system.check_convergence()`
- extracts atoms and results
- applies unit conversion from `properties_list`
- groups records by empirical formula
- writes ANI-style arrays through `pyanitools`

If you change what the QM stage returns or what the ML stage expects, this function is the schema chokepoint.

## 6. Builder Modules

All builder task entrypoints are Parsl `@python_app` functions. In the current codebase they run on `alf_sampler_executor`.

## 6.1 `alframework/builders/builders.py`

This is the main builder module for the modern `MoleculesObject` flow.

Key pieces:

- `simple_cfg_loader_task()`
- `readMolFiles()`
- `condensed_phase_builder()`
- `simple_condensed_phase_builder_task()`
- `simple_multi_condensed_phase_builder_task()`
- `create_atomic_system()`
- `construct_simulation_box()`
- `atomic_system_task()`
- `TiAl_builder_task()`

### `simple_cfg_loader_task()`

- randomly selects a `.cfg` structure from `builder_config['molecule_library_dir']`
- applies `ase_atoms.rattle(shake)`
- wraps it in `MoleculesObject`
- stores the source cfg path in metadata

This is the simplest builder path and is used by the `UO2` example.

### `condensed_phase_builder()`

This is the key condensed-phase assembly routine.

It:

- starts from an empty or partially populated periodic box
- reads fragment geometries from a molecule library
- inserts explicit solutes first
- then inserts solvent molecules until a target density or stop condition is reached
- rejects placements that violate the minimum intermolecular distance
- updates metadata with target vs achieved density

The important design choice is that this function operates on a `MoleculesObject` and mutates its atoms and metadata before returning it.

### `simple_condensed_phase_builder_task()`

This is the primary builder used by `examples/simple_water`.

It:

- samples a random cell from `cell_range`
- creates an empty periodic `Atoms` box
- reads the fragment library
- chooses one solute list from `solute_molecule_options`
- samples a target density from `Rrange`
- uses `build_input_dict()` to feed those values plus the rest of `builder_config` into `condensed_phase_builder()`

Note the config naming trick:

- examples often define `molecule_library_path`
- `load_config_file()` creates `molecule_library_dir`
- the builder task signature expects `molecule_library_dir`

### `simple_multi_condensed_phase_builder_task()`

This is the same condensed-phase logic, but it accepts `moleculeids` and returns a list of `MoleculesObject`.

This is how `examples/simple_water_multi_builder` works together with:

```json
"maximum_builder_structures": 5
```

`__main__.py` knows how to flatten this list into separate sampler submissions.

### Atomic-System Builders

`create_atomic_system()`, `construct_simulation_box()`, and `atomic_system_builder()` support random atomic systems rather than fragment-based molecular systems.

Important behavior:

- `create_atomic_system()` randomly builds a charge-neutral composition
- `construct_simulation_box()` places atoms on a randomized grid while enforcing a minimum distance under PBC
- `atomic_system_task()` samples a box size and returns a `MoleculesObject`

This is the path used by the molten-salt example.

## 6.2 `alframework/builders/moltensalt_builder.py`

This file mostly contains specialized atomic-system generation utilities for neutral ionic systems:

- `create_atomic_system()`
- `construct_simulation_box()`
- `atomic_system_builder()`

Compared with `builders.py`, this module is more specialized and more standalone.

Current state:

- the core molten-salt example actually points at `alframework.builders.builders.atomic_system_task`
- this module does not provide the main Parsl task wrappers used by current example configs
- the logic overlaps heavily with the atomic-system helper code already present in `builders.py`

Practical takeaway:

- treat this file as a specialized utility / partial legacy branch
- if you are changing the active molten-salt workflow, check which module the example config actually calls before editing here

## 6.3 `alframework/builders/reactive_builder.py`

This module is purpose-built for reaction-focused workflows.

Key entrypoint:

- `load_reactive_task()`

It:

- selects a random reaction index from a library
- loads reactant, transition-state, and product structures
- optionally perturbs all three structures
- chooses one of the three as the starting `atoms`
- stores the full triplet in metadata under `reactant`, `ts`, and `product`

This metadata contract is what powers `reactive_sampler.py`.

Important implication:

- the reactive sampler does not infer reaction endpoints itself
- it expects the builder to preload them into the `MoleculesObject.metadata`

## 7. Sampler Modules

## 7.1 `alframework/samplers/ASE_ensemble_constructor.py`

This file provides the ensemble calculator layer used by uncertainty-driven sampling.

### `Well_Potential`

`Well_Potential` is an ASE calculator that adds a spherical restoring force outside a chosen radius.

Use case:

- keep clusters from drifting too far during MD
- optionally mass-weight the restoring force so all atoms feel the same acceleration

### `MLMD_calculator`

`MLMD_calculator` wraps a list of ASE calculators representing ensemble members.

It:

- mixes their shared implemented properties
- can append `Well_Potential`
- computes ensemble mean properties through ASE's `Mixer`
- exposes additional uncertainty metrics:
  - `energy_stdev`
  - `forces_stdev_mean`
  - `forces_stdev_max`

This is the core uncertainty estimator used by `mlmd_sampling.py` and also by reactive NEB / dimer checks.

## 7.2 `alframework/samplers/mlmd_sampling.py`

This is the primary active-learning sampler for non-reactive workflows.

Key pieces:

- `mlmd_sampling()`
- `simple_mlmd_sampling_task()`

### `simple_mlmd_sampling_task()`

This is the task entrypoint invoked by `__main__.py`.

It does the runtime setup:

- pins the worker to a GPU using `PARSL_WORKER_RANK % gpus_per_node`
- samples random temperature and density schedule parameters from the sampler config
- loads the configured ASE calculator path from `sampler_config['ase_calculator']`
- chooses between:
  - a NeuroChem-specific path
  - a general ensemble path using `MLMD_calculator`
- optionally recenters the structure
- optionally disables trajectory writing based on `trajectory_frequency`

It then calls `mlmd_sampling()`.

### `mlmd_sampling()`

This function performs Langevin MD under the current ML ensemble.

Per run it:

- sets the initial temperature with `annealing_schedule()`
- optionally drives density changes by rescaling the cell
- checks uncertainty every `Ncheck` MD steps
- checks:
  - ensemble energy stdev
  - mean force stdev
  - max force stdev
  - minimum interatomic distance
- records metadata and optional trajectory

Return behavior:

- if uncertainty exceeds threshold and the failure was not just a distance collapse, it returns the `MoleculesObject` with `atoms` set to the escalated structure
- otherwise it sets `atoms` to `None`

That `atoms is None` convention is what tells `__main__.py` not to launch QM.

## 7.3 `alframework/samplers/reactive_sampler.py`

This is the reaction-focused alternative to `mlmd_sampling.py`.

Key pieces:

- `reactive_sampling()`
- `neb_sampler()`
- `dimer_sampler()`

Workflow:

- the builder preloads `reactant`, `ts`, and `product` into metadata
- `reactive_sampling()` chooses NEB, dimer, or both based on config
- each generated image / candidate is evaluated with the ensemble uncertainty calculator
- once a structure crosses uncertainty thresholds, it becomes the QM escalation candidate

Notable differences from `mlmd_sampling.py`:

- it searches reaction pathways instead of time-integration trajectories
- it uses metadata-provided endpoints and transition states
- it returns the same `MoleculesObject` pattern: escalated `atoms` or `None`

## 7.4 `alframework/samplers/ml_driven_md_sampling.py`

This file appears to be stale legacy code rather than a live module.

Reasons:

- it references `self` inside a standalone function
- it uses undefined names such as `mol`
- it is not wired by the current example configs
- the active runtime path is `mlmd_sampling.py`

Practical takeaway:

- do not treat this file as part of the modern ALF workflow
- if it is kept, it should be considered archival until proven otherwise

## 8. ML / Training Modules

## 8.1 `alframework/ml_interfaces/hippynn_interface.py`

This is the main modern training backend in the example workflows.

Key pieces:

- `train_HIPNN_model()`
- `train_HIPNN_model_wrapper()`
- `train_HIPPYNN_ensemble_task()`
- `HIPNN_ASE_calculator()`
- `HIPNN_ASE_load_ensemble()`

### Training Flow

`train_HIPNN_model()` is a large all-in-one function that:

- configures CUDA visibility
- builds a HIPNN graph
- optionally includes electrostatics targets
- defines training and validation losses
- loads ANI-style h5 data through `PyAniDirectoryDB`
- filters problematic data
- makes train / valid / test splits
- computes self-energy baselines
- sets up optimizer, scheduler, and controller
- launches training
- optionally exports a LAMMPS-compatible pickle

The function is flexible, but it concentrates a lot of policy in one place.

### Ensemble Training

`train_HIPPYNN_ensemble_task()` is the Parsl entrypoint used by ALF.

It:

- copies `ML_config`
- pops `n_models`
- launches one subprocess per GPU through `multiprocessing.Pool(gpus_per_node)`
- trains each ensemble member under:
  - `models/model-####/model-00`
  - `models/model-####/model-01`
  - etc.
- inspects each `training_log.txt`
- treats `"Training complete"` in the log as the success marker

It returns:

```text
([True/False per model], current_training_id)
```

### Sampling-Side Calculator Loading

`HIPNN_ASE_load_ensemble()` loads every `model-*` subdirectory in a trained ensemble directory and returns a list of ASE calculators.

That list is what `MLMD_calculator` wraps during sampling.

## 8.2 `alframework/ml_interfaces/neurochem_interface.py`

This is the older ANI / NeuroChem backend.

Key pieces:

- `NeuroChemTrainer`
- `NeuroChemCalculator()`
- `train_ANI_model_task()`

### Training Flow

`NeuroChemTrainer.train_models()`:

- creates the ensemble directory
- writes ANI AEV parameter files
- configures training inputs
- builds a strided training cache
- launches ensemble training through `anitraintools`

`train_ANI_model_task()` is the Parsl-facing wrapper used by ALF.

It:

- builds the trainer from `ML_config` plus `gpus_per_node`
- sets `ensemble_path`, `data_store`, and a random seed
- trains the ensemble
- returns `(completed, current_training_id)`

### Sampling-Side Calculator Loading

`NeuroChemCalculator()` constructs an ANI ensemble calculator from a trained model directory and is used when sampler config sets:

```json
"use_potential_specific_code": "neurochem"
```

### Role in the Current Repo

This path is still important because the `UO2` example uses it, but the more actively maintained example workflows use HIPPYNN.

## 9. QM Interface Modules

The QM stage exists to turn escalated structures into labeled data with convergence tracking.

## 9.1 `alframework/qm_interfaces/ase_calculator_interface.py`

This is the generic ASE-based path.

### `ase_calculator_task()`

It:

- creates a per-structure scratch directory
- resolves `QM_config['ASE_calculator']`
- instantiates the ASE calculator with the configured command and options
- runs `calc.calculate(...)`
- stores `calc.results`
- sets `converged` from `calc.converged`

This is the cleanest generic template in the repo.

### `VASP_ase_calculator_task()`

This is the VASP-specific path used by the molten-salt example.

It:

- builds an ASE `Vasp` calculator from `QM_config['input']`
- allows per-structure k-point override via metadata
- runs the calculation
- parses convergence from `OUTCAR`

This function is more aligned with the current `MoleculesObject` flow than the older `vaspase_interface.py`.

## 9.2 `alframework/qm_interfaces/orca5_interface.py`

This is the ORCA-specific implementation used by `simple_water` and `reactive_sampling`.

Key pieces:

- `orcaGenerator`
- `orca_calculator_task()`
- `orca_double_calculator_task()`

### `orcaGenerator`

It:

- writes ORCA input files
- runs ORCA via `os.system`
- parses `engrad`, `.log`, and `_property.txt`
- extracts properties such as:
  - energy
  - forces
  - dipole
  - quadrupole
  - Hirshfeld charges / spin

### `orca_calculator_task()`

This is the normal single-run ORCA task:

- instantiate generator
- run single point
- store parsed properties
- set convergence from parsed ORCA termination markers

### `orca_double_calculator_task()`

This variant runs ORCA twice and averages the results, while also enforcing:

- `Ediff`
- `Fdiff`
- both runs converged

This is useful when the workflow wants internal consistency checks before accepting a label.

## 9.3 `alframework/qm_interfaces/siesta_interface.py`

This is the dedicated SIESTA path.

Key pieces:

- `_siesta_converged()`
- `siesta_calculator_task()`

It behaves like the generic ASE path, but with extra SIESTA-specific convergence inference:

- use `calc.converged` if available
- otherwise inspect output text
- otherwise fall back to checking whether requested properties exist

It also supports per-structure k-point override via `molecule_object.metadata['kpoints']`.

## 9.4 `alframework/qm_interfaces/vaspase_interface.py`

This file is older and much less aligned with the current runtime.

Observations:

- it contains a `VASPGenerator` class rather than the current Parsl task style
- it references legacy concepts such as `Molecule`, `VASP_COMMAND`, `VASP_PP_PATH`, `MPI`, and `gpuid`
- it is not the main VASP path used by the example configs reviewed here

Practical takeaway:

- for current ALF workflow changes, start with `VASP_ase_calculator_task()` in `ase_calculator_interface.py`
- treat `vaspase_interface.py` as legacy code unless a specific workflow still depends on it

## 10. Example Wiring

The examples matter because they show which of the many available modules are actually used together.

## 10.1 `examples/simple_water`: Mainline Modern Flow

This is the cleanest reference workflow for the current `MoleculesObject`-based ALF path.

Wiring:

- Builder: `alframework.builders.builders.simple_condensed_phase_builder_task`
- Sampler: `alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task`
- QM: `alframework.qm_interfaces.orca5_interface.orca_calculator_task`
- ML: `alframework.ml_interfaces.hippynn_interface.train_HIPPYNN_ensemble_task`

Important config patterns:

- builder config uses `molecule_library_path`, which becomes `molecule_library_dir`
- sampler config uses `meta_path`, which becomes `meta_dir`
- sampler calculator is `HIPNN_ASE_load_ensemble`
- master `properties_list` sets the training dataset schema and unit conversion

### Complete Traced Path: `simple_water`

1. `__main__.py` loads `master_config.json`
2. `master_config.json` points to the builder, sampler, QM, and ML JSON files
3. `simple_condensed_phase_builder_task()` creates a periodic water box and returns `MoleculesObject`
4. `simple_mlmd_sampling_task()` loads the current HIPNN ensemble and runs uncertainty-driven MD
5. If uncertainty exceeds thresholds, the same `MoleculesObject` keeps the escalated structure in `atoms`
6. `orca_calculator_task()` runs ORCA on that structure and stores converged QM labels
7. `store_current_data()` writes converged records into `h5store/data-####.h5`
8. `train_HIPPYNN_ensemble_task()` retrains the ensemble in `models/model-####`
9. ALF updates `current_model_id` and starts generating new structures against the new model

## 10.2 `examples/simple_water_multi_builder`: One Builder Task, Multiple Structures

This example is identical in spirit to `simple_water`, except the builder is:

- `alframework.builders.builders.simple_multi_condensed_phase_builder_task`

and the master config sets:

```json
"maximum_builder_structures": 5
```

This demonstrates that:

- one builder submission can fan out into multiple `MoleculesObject` outputs
- `__main__.py` explicitly supports this shape
- the rest of the pipeline still processes one structure at a time downstream

## 10.3 `examples/reactive_sampling`: Reactive Builder + Reactive Sampler

This example changes both the builder and sampler:

- Builder: `alframework.builders.reactive_builder.load_reactive_task`
- Sampler: `alframework.samplers.reactive_sampler.reactive_sampling`
- QM: ORCA
- ML: HIPPYNN

### Complete Traced Path: `reactive_sampling`

1. `load_reactive_task()` loads a reactant / TS / product triplet
2. One of those structures becomes the starting `atoms`
3. The full triplet is stored in metadata
4. `reactive_sampling()` loads the current ensemble and runs NEB and/or dimer searches
5. Intermediate images are checked for ensemble disagreement
6. The first image above threshold becomes the escalated structure
7. ORCA labels that structure
8. ALF writes it to h5 and retrains HIPPYNN as usual

The main difference from the standard flow is that exploration is reaction-path based rather than MD-trajectory based.

## 10.4 `examples/molten_salt`: Atomic-System Builder + VASP Path

This example uses:

- Builder: `alframework.builders.builders.atomic_system_task`
- Sampler: standard MLMD sampling
- QM: `alframework.qm_interfaces.ase_calculator_interface.VASP_ase_calculator_task`
- ML: HIPPYNN

This shows that:

- builders do not need fragment libraries
- ALF can sample periodic atomic systems directly
- the same active-learning loop still applies once the builder returns `MoleculesObject`

## 10.5 `examples/UO2`: Variant Pattern with Older Assumptions

This example is important because it shows a partially older branch of ALF usage:

- Builder: `simple_cfg_loader_task`
- Sampler: standard MLMD sampling
- QM: example-local `custom_modules.ase_calculator_task`
- ML: `train_ANI_model_task` from `neurochem_interface.py`

This variant matters for two reasons:

- it uses NeuroChem rather than HIPPYNN
- the example-local QM code still shows older list-based assumptions in places

Treat `UO2` as evidence that ALF still has legacy interfaces in the repo, even though the core runtime has moved toward `MoleculesObject`.

## 11. Modification Hotspots and Gotchas

This section is intentionally split between confirmed code facts and inferred modification risks.

## 11.1 Confirmed from the Current Code

### Mixed Object Conventions Still Exist

Confirmed:

- core runtime and core builder/sampler/QM code are based on `MoleculesObject`
- some example-local code still uses the older `[metadata, atoms, results]` list convention

Implication:

- do not assume all repo files are on the same interface generation

### Config Key Drift Is Hidden by `load_config_file()`

Confirmed:

- examples use both `molecule_library_path` and `molecule_library_dir`
- examples use both `meta_path` and `meta_dir`

These work because `load_config_file()` auto-derives `...dir` from `...path`.

Implication:

- a config key mismatch may still work today, but only because of implicit helper behavior

### `build_input_dict()` Is a Critical Coupling Point

Confirmed:

- task and helper functions receive many inputs by name-based signature matching
- the first matching dictionary wins

Implication:

- renaming parameters can break wiring without obvious call-site errors

### Builders and Samplers Share the Same Executor Pool

Confirmed:

- builder task functions use `@python_app(executors=['alf_sampler_executor'])`
- sampler task functions also use `alf_sampler_executor`

Implication:

- builder throughput and sampler throughput are coupled at the executor level

### Sampler Code Mutates Config In Place

Confirmed in `simple_mlmd_sampling_task()`:

- if a trajectory is not selected for writing, it sets `sampler_config['trajectory_interval'] = None`

Implication:

- avoid assuming config dictionaries remain immutable after task entry

### Standby Executors Are Appended Automatically

Confirmed:

- `__main__.py` appends `_standby_executor` labels to task executor lists if matching labels exist

Implication:

- effective executor selection is not just whatever is written on the decorator

### Failure Counting in `parsl_task_queue.get_task_results()` Is Broken

Confirmed in `alframework/tools/tools.py`:

- failed tasks do `failed_number += failed_number`

This never increments from zero.

Implication:

- lifetime failure counters in `status.txt` are likely underreported
- queue status printing is more reliable for failures than the persisted counters

### `reactive_builder.load_reactive_task()` Has a Brittle `rattle` Check

Confirmed:

```python
if rattle or rattle.lower() == 'true':
```

If `rattle` is boolean `False`, the second branch will try to call `.lower()` on a bool.

Implication:

- this path is brittle if callers pass boolean `False` instead of a truthy string / default truthy value

### `ml_driven_md_sampling.py` Is Stale

Confirmed:

- the function body references `self` and other undefined names

Implication:

- do not build new work on this file unless it is first repaired or retired

## 11.2 Inferred Risks When Modifying the Workflow

These are not all confirmed bugs, but they are high-risk areas for future edits.

### Changing Return Shapes Will Break Stage Handoffs Quickly

The runtime assumes:

- builder returns one object or a list of objects
- sampler returns one object
- QM returns one object
- ML returns `(completed_flags, training_id)`

If you change any of those shapes, you will likely need matching edits in:

- `__main__.py`
- queue harvesting code
- dataset writing
- example configs

### Mid-Run Config Reload Can Mask State Assumptions

Because ALF re-reads config every master-loop iteration:

- some changes take effect live
- some state derived earlier does not get recomputed

Be careful when adding new config-driven behavior that depends on startup-only initialization.

### Example Code Is Not a Perfect Proxy for Core Runtime

The examples are useful, but they do not all reflect the same maturity level.

In practice:

- `simple_water` is the safest reference for the mainline modern flow
- `reactive_sampling` is the best reference for reaction-centric flow
- `molten_salt` and `UO2` are important variants, but they show more legacy or specialized behavior

## 12. Where to Start When Modifying ALF

If you need to change:

- overall job flow: start in `alframework/__main__.py`
- how arguments are injected: inspect `build_input_dict()` and `load_config_file()`
- handoff structure schema: inspect `MoleculesObject` and `store_current_data()`
- condensed-phase structure generation: start in `builders.py`
- uncertainty-driven MD behavior: start in `mlmd_sampling.py` and `ASE_ensemble_constructor.py`
- reactive exploration: start in `reactive_builder.py` and `reactive_sampler.py`
- HIPPYNN retraining behavior: start in `hippynn_interface.py`
- ANI / NeuroChem behavior: start in `neurochem_interface.py`
- label extraction or convergence behavior: start in the relevant `qm_interfaces` module

## 13. Bottom Line

The cleanest mental model for current ALF is:

- `__main__.py` is the workflow state machine
- config files choose which task implementations are plugged into each stage
- `MoleculesObject` is the main payload passed between stages
- samplers decide whether a structure deserves QM by either keeping or nulling out `atoms`
- `store_current_data()` is the schema bridge from QM results into training data
- model promotion is controlled entirely by the ML task return tuple and `status.txt`

If you keep those contracts intact, most ALF modifications can stay localized. If you change those contracts, you are editing the framework itself rather than just a stage implementation.

## 14. Excited-State Integration Checkpoint (2026-04-20)

This section records the current state of the additive excited-state integration that was brought into ALF from the workflow-development sandbox. The goal of that work was to preserve ALF's existing runtime, Parsl execution model, `status.txt` handling, HDF5 sharding, and `--test_*` flows while adding a new optional task family for multi-state sampling, pyseqm labeling, and multi-state HIPPYNN training.

### 14.1 What Was Added

The excited-state path is currently implemented as new stage modules plus one shared helper module:

- [`alframework/tools/excited_state_tools.py`](./alframework/tools/excited_state_tools.py)
- [`alframework/builders/excited_state_builder.py`](./alframework/builders/excited_state_builder.py)
- [`alframework/samplers/excited_state_sampling.py`](./alframework/samplers/excited_state_sampling.py)
- [`alframework/qm_interfaces/pyseqm_interface.py`](./alframework/qm_interfaces/pyseqm_interface.py)
- [`alframework/ml_interfaces/excited_state_hippynn_interface.py`](./alframework/ml_interfaces/excited_state_hippynn_interface.py)

An example configuration set for the new path was also added:

- [`examples/excited_state_pyseqm`](./examples/excited_state_pyseqm)

And a focused test harness was added:

- [`tests/`](./tests/)

### 14.2 What This Integration Preserves

Confirmed:

- `alframework/__main__.py` was not reworked to support the excited-state path
- stage selection is still entirely config-driven through dotted task strings
- Parsl executors and standby-executor auto-appending still behave the same way
- `store_current_data()` still writes the canonical ALF HDF5 shards
- the ML stage still returns `(completed_flags, training_id)`
- the sampler still signals escalation by returning a `MoleculesObject` whose `atoms` is either the chosen candidate or `None`

This matters because the excited-state workflow was intentionally integrated as an additive stage family, not as a replacement runtime.

### 14.3 Excited-State Data Contract

The current excited-state implementation follows the flattened-property approach described earlier in this note.

Key contract:

- energies are stored as separate logical properties `sE0`, `sE1`, `sE2`, ...
- forces are stored as separate logical properties `F0`, `F1`, `F2`, ...
- the active state list is derived from `master_config["properties_list"]`
- no new tensor-style multi-state storage format was introduced into ALF HDF5 writing

This means the excited-state code is coupled to the same `properties_list` schema bridge used by the rest of ALF.

### 14.4 Builder Checkpoint

The new replay builder is `excited_state_replay_builder_task()`.

Current behavior:

- before any ALF HDF5 shard exists, it can sample start structures from a seed dataset directory containing `R.npy` and `Z.npy`
- after HDF5 shards exist, it can replay structures from `h5store/data-####.h5`
- source selection is controlled by `source_priority`, with `h5_then_seed` as the default
- HDF5 replay samples uniformly over frames, not over files
- optional empirical-formula filtering is supported
- a replay manifest cache can be written through `manifest_cache_dir`
- excited-state assignment is handled through metadata plus `state_selection`

One implementation detail worth remembering:

- replay and sampling seeds were changed to a stable SHA1-based seed helper rather than Python's built-in `hash()` so behavior is reproducible across processes

### 14.5 Sampler Checkpoint

The new sampler is `excited_state_sampling_task()`.

Current behavior:

- it reuses ALF's existing annealing-style temperature controls: `dt`, `maxt`, `Ncheck`, `srt_temp`, `end_temp`, `amp_temp`, `per_temp`
- it loads a multi-state HIPPYNN ensemble from the standard ALF model directory layout
- each trajectory propagates one selected state but evaluates uncertainty metadata for all available states
- candidate ranking is local to one sampler task; ALF does not yet implement a global top-k ranking pool in the core runtime
- `return_top_n` optionally returns the top N local candidate frames from one trajectory as separate `MoleculesObject` instances; the default is `1`, which preserves the original single-candidate return shape
- the sampler writes metadata sidecars and optional trajectory files under `meta_dir`
- if no valid candidate survives filtering, default mode returns the same `MoleculesObject` with `atoms=None`; `return_top_n > 1` returns an empty list

Confirmed bug fixed during integration:

- JSON metadata writing originally failed on ASE `Cell` objects; the serializer now handles array-like objects via `.tolist()`

### 14.6 QM Checkpoint

The new QM task is `pyseqm_excited_state_task()`.

Current behavior:

- it uses a pyseqm backend derived from the workflow-development implementation
- it labels one structure per Parsl task, preserving ALF's existing queueing model
- it flattens results into `sE#` and `F#` keys expected by `properties_list`
- it reads `energy_offset_eV` from `sampler_config`
- it subtracts that offset once before storing results on the `MoleculesObject`
- on backend failure it marks the structure unconverged and stores the exception text in metadata

This is intentionally narrower than the workflow-development batch label scheduler. The batch pool logic was not imported into core ALF.

### 14.7 ML Checkpoint

The new ML task is `train_excited_state_HIPPYNN_ensemble_task()`.

Current behavior:

- it reads ALF's existing HDF5 shard directory directly
- it derives the state/property layout from `properties_list`
- it trains one HIPPYNN model per ensemble member under `models/model-####/model-##`
- it writes a per-model `training_summary.json`
- it exposes `load_excited_state_ensemble()` for the sampler path

Current design choice:

- the existing ground-state `hippynn_interface.py` was not refactored into a dual-mode trainer
- the excited-state trainer is additive and lives beside the existing trainer

### 14.8 Verification Checkpoint

The current integration was verified locally in the `atomistic` Conda environment with:

- import and compile checks for the new modules
- focused unit tests for helpers, replay-building behavior, sampler scoring/selection behavior, pyseqm flattening/offset logic, and ML return-shape behavior
- an async handoff test that exercised builder -> sampler -> QM -> HDF5 save -> ML promotion shape using local Parsl thread executors
- a smoke test that ran `python -m alframework --test_builder --test_sampler --test_qm --test_ml` against synthetic local stage modules

At the time of this checkpoint the focused test subset passed:

- `13 passed, 1 warning`

The warning was Parsl test noise about `strategy=None` deprecation in the local thread-executor test config.

### 14.9 What Is Still Deliberately Deferred

The excited-state integration is intentionally incomplete in a few areas because the goal was minimal ALF-core disruption.

Still deferred:

- no global iteration-wide top-k ranking buffer in ALF core
- no cross-trajectory de-duplication stage in `__main__.py`
- no batch pyseqm scheduler imported into ALF core
- no live cluster-scale validation of real pyseqm labeling or full HIPPYNN training in this checkpoint section
- no changes to the legacy ground-state task family

Interpretation:

- the additive path is real and test-covered at the ALF contract level
- production performance and backend-numerical validation on cluster resources are still separate follow-up tasks

### 14.10 Modification Guidance From This Checkpoint

If future work needs to extend the excited-state path while preserving ALF:

- keep `properties_list` as the single state-schema source of truth
- avoid changing the outer control flow in `__main__.py` unless global candidate ranking becomes absolutely necessary
- treat `store_current_data()` as fixed unless a new training-data format is truly required
- preserve the `(completed_flags, training_id)` ML return tuple
- preserve the sampler convention that `atoms=None` means "no QM escalation"

If future work instead needs workflow-level ranking, dataset-view scheduling, or richer multi-state bookkeeping, that is likely the point where ALF would need a genuine runtime extension rather than another additive stage implementation.
