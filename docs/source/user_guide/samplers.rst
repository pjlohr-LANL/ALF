Samplers
========

Samplers explore configuration space and decide whether a structure should be
sent to QM labeling. In most ALF workflows, the sampler receives a
``MoleculesObject`` from the builder, evaluates it with an ensemble ML
calculator, and returns either a selected high-uncertainty configuration or a
marker that no QM calculation is needed.

In a master configuration, the selected sampler is specified with the
``sampler_task`` import string:

.. code-block:: json

   {
     "sampler_task": "alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task",
     "sampler_config_path": "mlmd_config.json"
   }

Sampler tasks usually run on the ``alf_sampler_executor`` Parsl executor. GPU
assignment is controlled by ``gpus_per_node`` in the master configuration and
the worker layout in the Parsl resource configuration.

Supported Sampler Tasks
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 34 26

   * - Sampler task
     - What it does
     - Typical use
   * - ``alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task``
     - Runs Langevin molecular dynamics with an ensemble ML calculator,
       monitors energy and force uncertainty, and returns configurations that
       exceed uncertainty or distance criteria.
     - General active-learning loops based on MD exploration.
   * - ``alframework.samplers.udd_sampling.simple_udd_sampling_task``
     - Runs MLMD with an uncertainty-driven dynamics bias that steers
       trajectories toward high ensemble energy uncertainty while retaining the
       usual QM-selection threshold checks.
     - Exploration runs where biasing MD toward model disagreement is desired.
   * - ``alframework.samplers.reactive_sampler.reactive_sampling``
     - Uses NEB and/or dimer-style searches from reaction metadata, monitors
       ensemble uncertainty along reaction pathways, and returns uncertain
       structures for QM labeling.
     - Reaction-pathway active learning with reactant, transition-state, and
       product structures.
   * - ``alframework.samplers.alchemi_sampling.alchemi_sampling_task``
     - Runs strict-full batched BAOAB dynamics with a HIPPYNN ensemble through
       NVIDIA ALCHEMI and supports stop-on-uncertainty or continued sampling.
     - GPU-resident, nonperiodic ground- or excited-state molecular dynamics.

Common Configuration Fields
---------------------------

MLMD dynamics fields
   These fields control the length and cadence of MD sampling.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``dt``
        - MD timestep in femtoseconds.
      * - ``maxt``
        - Maximum simulation time in picoseconds.
      * - ``Ncheck``
        - Number of MD steps between uncertainty checks.
      * - ``min_time``
        - Optional minimum simulation time before uncertainty checks can select
          a structure.
      * - ``distcut``
        - Minimum allowed interatomic distance. Structures can be rejected if
          atoms come too close.

Uncertainty thresholds
   These fields decide when a sampled configuration should be sent to QM.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``Escut``
        - Energy standard-deviation threshold.
      * - ``Fscut``
        - Mean force standard-deviation threshold.
      * - ``forces_stdev_max``
        - The MLMD sampler also checks the maximum per-force uncertainty using
          an internal threshold of ``3 * Fscut``.

Temperature schedule fields
   The sampler randomly chooses temperature schedule parameters from configured
   ranges for each task.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``srt_temp``
        - Range for the starting temperature.
      * - ``end_temp``
        - Range for the ending temperature.
      * - ``amp_temp``
        - Range for sinusoidal temperature fluctuation amplitude.
      * - ``per_temp``
        - Range for temperature fluctuation period.

Density schedule fields
   Density changes are optional. Use ``null`` for ``end_dens``, ``amp_dens``,
   and ``per_dens`` to disable density changes.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``end_dens``
        - Target final density range. ``null`` disables cell rescaling.
      * - ``amp_dens``
        - Range for density fluctuation amplitude.
      * - ``per_dens``
        - Range for density fluctuation period.

ML calculator fields
   These fields control how sampler tasks load and wrap ML models.
   See :ref:`ml-interface-extension-points` and
   :ref:`ml-interface-new-architecture-template` for the corresponding ML
   interface requirements.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``ase_calculator``
        - Import string for the ML calculator or model-loader function.
      * - ``MLMD_calculator_options``
        - Options passed to ``MLMD_calculator``, such as ``well_params``.
      * - ``translate_to_center``
        - Whether to translate the incoming structure so its center of mass is
          near the origin before sampling.

Uncertainty-driven dynamics
   Uncertainty-driven dynamics (UDD) is a biased MLMD sampler that steers
   dynamics toward regions where the model committee has high ensemble energy
   uncertainty. The usual ``Escut`` and ``Fscut`` threshold logic still decides
   whether a sampled configuration is returned for QM labeling.

   Select UDD in ``master_config.json`` with a separate sampler task:

   .. code-block:: json

      {
        "sampler_task": "alframework.samplers.udd_sampling.simple_udd_sampling_task"
      }

   Set the bias strength in the sampler config:

   .. code-block:: json

      {
        "udd_bias_weight": 0.45,
        "MLMD_calculator_options": {
          "well_params": null
        }
      }

   ``udd_bias_weight`` controls the strength of the energy-uncertainty bias.
   Use ``simple_mlmd_sampling_task`` instead when no UDD bias is desired.

Output and trajectory fields
   These fields control sampler metadata and optional trajectory output.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``meta_dir`` or ``meta_path``
        - Directory for sampler metadata files. Examples use both names; the
          task input builder maps compatible names to the sampler function.
      * - ``trajectory_frequency``
        - Probability of writing a trajectory for a sampling task.
      * - ``trajectory_interval``
        - MD-step interval for trajectory writes when trajectory output is
          enabled.

Reactive sampler fields
   These fields are used by ``reactive_sampling``.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``reactive_sampling_method``
        - Reactive search mode. Values containing ``NEB`` run NEB; values
          containing ``DIMER`` run dimer sampling.
      * - ``N_neb``
        - Number of intermediate NEB images.
      * - ``max_iter``
        - Maximum number of repeated reactive-sampling attempts.
      * - ``neb_steps``
        - Optimizer steps for each NEB or dimer call.
      * - ``Escut``, ``Fscut``
        - Energy and force uncertainty thresholds.
      * - ``Fmmult``
        - Multiplier applied to ``Fscut`` for maximum force uncertainty. The
          default is ``3.0``.

ALCHEMI batched sampling
------------------------

Select the dedicated ALCHEMI task in the master configuration. Existing MLMD,
UDD, reactive, and NeuroChem task names continue to use their original
implementations.

.. code-block:: json

   {
     "sampler_task": "alframework.samplers.alchemi_sampling.alchemi_sampling_task"
   }

The sampler configuration retains the existing MLMD thresholds and temperature
ranges and adds the following fields:

.. code-block:: json

   {
     "model_mode": "ground_state",
     "uncertainty_policy": "stop",
     "return_top_n": 1,
     "friction_per_fs": 0.0019645,
     "alchemi_baoab": {
       "batch_size": 50,
       "partial_policy": "full_only",
       "strict_gpu": true,
       "allow_cpu_debug": false,
       "random_seed": 42
     }
   }

``stop`` is the compatibility mode. Each replica is frozen independently at
its first valid uncertainty event, while other replicas in the GPU batch keep
running. At most one candidate is returned for each input replica.

``continue`` keeps uncertain replicas active. At every ``Ncheck`` interval,
qualifying frames are ranked across the whole task batch using
``max(Es/Escut, Fs/Fscut, Fsmax/(3*Fscut))``. Only the global
``return_top_n`` frames are returned. Results are sent directly through Parsl,
so large values of ``return_top_n`` increase task-result serialization and
driver memory use.

Batches are grouped by exact atomic-number sequence. Excited-state batches are
also grouped by selected state. ``partial_policy`` currently accepts only
``full_only``: incomplete buckets stay in the running ALF driver's memory,
remain there across model retraining, and use the latest model when they
eventually fill. ``status.txt`` reports bucket counts and ages. The buffers are
not checkpointed across a driver restart.

The first ALCHEMI implementation supports nonperiodic, fixed-cell HIPPYNN
models. Set ``end_dens``, ``amp_dens``, and ``per_dens`` to ``null``. Existing
``MLMD_calculator_options.well_params`` spherical-well settings are supported.
Periodic cells, density schedules, UDD bias, reactive sampling, and NeuroChem
remain on their existing sampler tasks.

Excited-state mode
~~~~~~~~~~~~~~~~~~

Set ``model_mode`` to ``excited_state`` and attach ``selected_state`` metadata
to each input molecule. The master ``properties_list`` uses a flat state
contract:

.. code-block:: json

   {
     "sE0": ["state_0_energy", "system", 1.0],
     "F0": ["state_0_forces", "atomic", 1.0],
     "sE1": ["state_1_energy", "system", 1.0],
     "F1": ["state_1_forces", "atomic", 1.0]
   }

State numbers must be contiguous from zero, and every sampled state requires a
force property. Dynamics uses the selected state's ensemble-mean energy and
force; uncertainty uses that state's population energy and force deviations.
Gap targets, gap scoring, GUDD, hysteresis, and Martinez--Levine gap seeking are
not enabled by this sampler version.

For ``--test_sampler``, use a debug sampler configuration with
``alchemi_baoab.batch_size`` set to ``1`` because the stage-test command obtains
one builder result.

What The Sampler Returns
------------------------

If uncertainty or another selection criterion triggers, the returned
``MoleculesObject`` keeps its atoms. ALF then passes that structure to the QM
task queue for labeling.

If sampling completes without selecting a configuration, the sampler sets the
atoms to ``None`` on the returned ``MoleculesObject``. ALF treats this as a
successful sampling attempt that does not need QM labeling.

The ALCHEMI sampler instead returns a list containing zero or more
``MoleculesObject`` candidates. The driver normalizes both interfaces before
QM submission.

Sampler metadata records the relevant diagnostics, including uncertainty
values, temperature and density schedule history, distance checks, selection
flags, final positions, and cell information. MLMD metadata is commonly written
to files such as ``sampling/metadata-<moleculeid>.p`` when ``meta_dir`` or a
compatible metadata path is configured.

Related Examples
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Example
     - Sampler task
     - Notes
   * - ``examples/simple_water``
     - ``alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task``
     - HIPPYNN ensemble MLMD sampling.
   * - ``examples/simple_water_multi_builder``
     - ``alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task``
     - Same sampler as simple water; builder throughput changes.
   * - ``examples/molten_salt``
     - ``alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task``
     - MLMD sampling for periodic ionic systems.
   * - ``examples/UO2``
     - ``alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task``
     - Uses ``use_potential_specific_code: "neurochem"``.
   * - ``examples/IL``
     - ``alframework.samplers.mlmd_sampling.simple_mlmd_sampling_task``
     - MLMD sampling for ionic-liquid style mixtures.
   * - ``examples/reactive_sampling``
     - ``alframework.samplers.reactive_sampler.reactive_sampling``
     - NEB/dimer-style reaction-pathway sampling.
   * - UDD workflows
     - ``alframework.samplers.udd_sampling.simple_udd_sampling_task``
     - Biased MLMD sampling toward high ensemble energy uncertainty.

Sampler API Links
-----------------

You can link from this guide directly to API pages:

* :doc:`Samplers package API <../api_documentation/alframework.samplers>`
* :doc:`MLMD sampling module <../api_documentation/alframework.samplers.mlmd_sampling>`
* :doc:`Reactive sampler module <../api_documentation/alframework.samplers.reactive_sampler>`
* :doc:`ASE ensemble calculator module <../api_documentation/alframework.samplers.ASE_ensemble_constructor>`
* :doc:`ALCHEMI sampling module <../api_documentation/alframework.samplers.alchemi_sampling>`
