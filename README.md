# Active Learning Framework:

This code automates the construction of datasets for machine learned (ML) interatomic potentials through active learning. By automating job execution utilizing the [Parsl](https://parsl-project.org/) framework, the active learning process can run for many iterations without human intervention. It breaks the process down into 4 fundamental tasks:
1) system construction (building)
2) sampling (samplers)
3) ML interatomic potential training
4) electronic structure calculations

The framework uses a general purpose ensembling calculator that enables the tracking of model uncertainty during the training process.

This package is closely interfaced to the Parsl framework which enables task execution through a queueing system or smaller clusters. Some requirements, such as the Parsl resource config file in `alframework/parsl_resource_configs` will need to be modified for your computer system. 

This code is currently a work in progress and documentation is still sparse. 

# Requirements: 

The requirements for this software are evolving, though generally, they will include the following: 
1) [NumPy](https://numpy.org/)
2) [ASE](https://wiki.fysik.dtu.dk/ase/)
3) a QM software package with interface (usually ASE)
4) a ML interatomic potential model ([HIPPYNN](https://github.com/lanl/hippynn) is open source and an interface is provided) 

# Configuration:

To control job flow in the active learning framework, 5 json files are used. The master configuration file controls how all pieces of the framework are assembled and defines where to find the other 4 files. Each of the other 4 files passes inputs to one of the four subtasks enumerated in the first section. To see how these files relate to one another, please see the `examples` folder.

With the json files completed, the `PYTHONPATH` environment variable must be set to the directory where `alframework` is held. Eventually, this step will be replaced by making `alframework` an installable package. 

# Testing: 

Once the environment is constructed with the required packages, it is important to test individual operations done by the active learning framework for erorrs. This is done to ensure all processes complete successufuly when run in active learning. Testing each of of the four sub processes is enabled in the following way:

```
python -m alframework --master master.json --test_builder #Test structure building
python -m alframework --master master.json --test_sampler #Test mlmd sampling
python -m alframework --master master.json --test_ml #Test training ML model
python -m alframework --master master.json --test_qm #Test running QM calculation
```

These functions will execute in such a way as to pass errors back to the front end to enable easier debugging. Errors encountered in the active learning phase.

# Execution:

Once each task has been tested, active learning can be started with:
```
python -m alframework --master master.json
```

It is generally advised to run the master process on a head node. It will automatically interface with the queueing system and run future jobs on compute nodes. 

# Citations

[1] Justin S. Smith, Benjamin Nebgen, Nithin Mathew, Jie Chen, Nicholas Lubbers, Leonid Burakovsky, Sergei Tretiak, Hai Ah Nam, Timothy Germann, Saryu Fensin, Kipton Barros "Automated discovery of a robust interatomic potential for aluminum" Nat. Comm. 2021,  12, 1257. 
https://doi.org/10.1038/s41467-021-21376-0

[2] Maksim Kulichenko, Kipton Barros, Nicholas Lubbers, Ying Wai Li, Richard Messerly, Sergei Tretiak, Justin S. Smith, Benjamin Nebgen "Uncertainty-driven dynamics for active learning of interatomic potentials" Nat. Comp. Sci. 2023, 1968. 
https://doi.org/10.1038/s43588-023-00406-5 

[3] Shuhao Zhang, Malgorzata Makos, Ryan Jadrich, Elfi Kraka, Kipton Barros, Benjamin Nebgen, Sergei Tretiak, Olexandr Isayev, Nicholas Lubbers, Richard Messerly, Justin Smith "Exploring the frontiers of chemistry with a general reactive machine learning potential". 
https://doi.org/10.26434/chemrxiv-2022-15ct6

## Physical force screening in excited-state sampling

The excited-state sampler screens the largest per-atom force norm across the
**ensemble-mean physical forces of every modeled state**, using
`max_force_cutoff` in eV/Å. This applies to unbiased, GALF, and UDD sampling and
to both ASE and ALCHEMI backends. Physical forces are negative gradients of the
individual state energies. Artificial propagation forces drive biased sampling
but do not determine physical-force eligibility. Missing or nonfinite physical
force outputs cause an explicit error. Topology, close-contact, uncertainty,
and post-QM dataset screening remain independent.

Sampler metadata schema 2 identifies `force_screen_source=physical_state_mean`
and `force_screen_scope=all_states`. Its `fmax` is the physical screening
quantity; `physical_fmax_by_state`, `propagation_fmax`, and `gap_bias_fmax`
separate state forces, propagation forces, and the Martinez gap-penalty component
(the latter is zero outside active GALF). Diagnostic checks include state
energies, adjacent energy gaps, and active-pair information.

`rejected_snapshot` retains the actual failed coordinates, physical force
arrays, propagation and gap-bias forces, and screening diagnostics **before**
rollback/freezing. `last_valid_snapshot` retains the preceding accepted check;
it is null if no check was accepted. `initial_positions` retains the starting
geometry separately. For an initial topology rejection before model evaluation,
force arrays are null. Existing top-level `positions` still contain the
restored last-valid/initial geometry. Snapshot `integration_step` and
`integration_time_ps` include the existing warmup step; `step` retains the
historical check-step numbering. No full coordinate trajectory is required.

Historical metadata are unchanged. Plot-input readers label records without a
source marker as `propagation_legacy`; their `fmax` must not be reinterpreted as
an all-state physical-force screen. This change also means that future unbiased
excited-state runs screen all states, rather than only the propagation state.

The matched RDKit-handoff controls can be prepared with
`python -m alframework.tools.prepare_physical_force_reruns`. This refuses to
overwrite existing runs, verifies original inherited data and ensemble hashes,
performs topology preflight, and prepares submission scripts without submitting
jobs. The original alpha, sigma, trigger threshold, and 16 eV/Å cutoff are
preserved. Each new directory contains preparation provenance and a source patch.

## Exact-40k ALF to GALF label-budget comparison

`python -m alframework.tools.prepare_budgeted_galf --tautomer both` prepares
independent enol/keto `*_alf40k_galf20k_physical_force` runs. It refuses to
overwrite existing directories, freezes exactly 40,000 original RDKit ALF
labels including seed, verifies sliced H5 contents, and copies no model or
master status. Pure-ALF 40k/50k/60k selections are frozen alongside each run.
The prepared two-node submission scripts are not submitted by this command.

On launch, the ordinary initial-training path trains four fresh HIP-HOP
members before sampling. Training summaries record the input-pool size and
actual split sizes. An artifact/input fingerprint gate requires all four
members to complete on the exact 40k pool, with the standard E/F objective.
Interrupted initial training fails before sampling and requires explicit
repair; a partial model directory is never treated as a complete ensemble.

These runs opt into `accepted_frame_budget` in the master configuration:
`{"handoff_frames":40000,"target_frames":60000,"checkpoints":[50000,60000]}`.
After completed H5 saves, the master freezes exact nested checkpoints, records
batch overshoot and unharvested task counts, and shuts down Parsl after reaching
at least 60k retained finite frames. The option has no effect on other runs.
This budget counts accepted stored labels, not QM attempts or compute cost.

`molecule_smax/benchmarks/conformational_analysis/compare_budgeted_galf.py`
provides source-verified `--stage update`, reduced-cache-only `--stage plot`,
and `--stage validate`. Successful submission scripts run the update stage
after budget completion. It reports per-pair QM gap enrichment and CDFs at
50k/60k, acquisition windows, torsion/pyramidalization diversity, online gap
errors, survival, and separately qualified cost diagnostics. Original paper
caches and conformational fits are preserved.
