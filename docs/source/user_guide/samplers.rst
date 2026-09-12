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
     - Runs strict-full batched BAOAB dynamics with a native ALCHEMI calculator
       or an existing ALF ASE calculator and supports stop or continued sampling.
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
     "alchemi_calculator": "alframework.samplers.alchemi_sampling.load_hippynn_alchemi_model",
     "alchemi_calculator_options": {},
     "uncertainty_policy": "stop",
     "return_top_n": 1,
     "max_candidates_per_replica": null,
     "friction_per_fs": 0.0019645,
     "alchemi_baoab": {
       "batch_size": 50,
       "partial_policy": "full_only",
       "strict_gpu": true,
       "allow_cpu_debug": false,
       "random_seed": 42
     }
   }

``max_candidates_per_replica`` caps how many frames one replica may contribute
to the returned batch. Candidates are grouped by ``parent_molecule_id``, and the
cap is applied before the global ``return_top_n`` truncation. The default
``null`` places no per-replica limit, which reproduces the historical behavior.
Under ``uncertainty_policy`` ``continue`` a diverging replica scores higher as it
leaves the training manifold, so without a cap a single trajectory can occupy
every returned slot with frames one ``Ncheck`` interval apart. ``stop`` provides
that guarantee implicitly by freezing each replica at its first uncertain frame.

Fixed-topology gating is separately opt-in:

.. code-block:: json

   {
     "topology_check": {
       "enabled": true,
       "reference_conformer_path": "reference.xyz",
       "reference_format": "auto",
       "reference_charge": 0,
       "bond_min_scale": 0.70,
       "bond_max_scale": 1.35,
       "connectivity_scale": 1.25
     }
   }

The reference path is resolved relative to ALF's ``master_directory``. RDKit
reads XYZ, MOL, or SDF coordinates and determines one fixed reference
connectivity; bond orders are ignored. Every input replica must have the exact
reference atomic-number sequence and atom order. A valid geometry keeps every
reference bond inside its inclusive scaled reference-length window and keeps
every non-reference atom pair above
``connectivity_scale * (r_cov,i + r_cov,j)``.

Topology is checked before the initial legacy MD step and again before every
``Ncheck`` uncertainty evaluation. A violation freezes only that replica and
discards the violating frame before uncertainty qualification, top-K ranking,
gap switching, or QM submission. Earlier valid continued-sampling candidates
remain eligible. ``distcut`` remains an independent close-contact check.
Candidate metadata records the reference content hash, bond/connectivity
metrics, and final task rejection summary. Disable this option for reactive
workflows intended to change connectivity.

Calculator backends
~~~~~~~~~~~~~~~~~~~

The sampler is not tied to HIPPYNN. Calculator selection uses this precedence:

#. If ``alchemi_calculator`` is set, ALF loads that native batched calculator.
#. Otherwise, ALF loads the existing ``ase_calculator`` and
   ``ase_calculator_options`` through a compatibility wrapper.
#. If neither field is configured, the sampler fails before dynamics begins.

A native loader receives the current model directory, device, ``model_mode``,
selected state, ML configuration, property mapping, sampler configuration, and
``alchemi_calculator_options``. It returns an ``ALFAlchemiCalculator``. Native
calculator subclasses provide raw selected-state ensemble contributions with
energy shape ``[models, batch]`` and force shape
``[models, total_atoms, 3]``. ``ALFNativeEnsembleModel`` can combine compatible
ALCHEMI ``BaseModelMixin`` members for model packages that already provide
ALCHEMI wrappers.

The ASE fallback evaluates each replica and calculator sequentially, including
CPU/GPU synchronization, so it provides compatibility rather than accelerated
batched inference. A list of ASE calculators supplies raw ensemble members. A
single ordinary calculator is a one-member ensemble with zero ensemble
deviation. A single uncertainty-aware calculator may instead expose
``energy_stdev``, ``forces_stdev_mean``, and ``forces_stdev_max``; the existing
NeuroChem uncertainty API is also recognized when
``use_potential_specific_code`` is ``neurochem``.

HIPPYNN's native loader consumes its raw ensemble ``.all`` outputs. The shared
calculator layer computes the same population statistics as production ALF:
``Es`` is the standard deviation of model energies, ``Fs`` is the mean absolute
componentwise force deviation, and ``Fsmax`` is its maximum. This common
reduction avoids model-library standard-deviation convention differences.

Direct gap diagnostics are optional in excited-state mode:

.. code-block:: json

   {
     "gap_diagnostics": {
       "enabled": true
     }
   }

When enabled, every ``dE#`` property in ``properties_list`` is evaluated.
Calculator adapters provide paired state-energy contributions for each model
member. The shared layer computes each member's ``sEj - sEi`` first and then
uses a population standard deviation; it never subtracts two state standard
deviations. Native HIPPYNN uses raw ``.all`` outputs. ASE fallback requests
the additional flattened ``sE#`` properties and fails clearly if its
calculator does not provide them. A single ordinary calculator has zero gap
ensemble deviation.

One-way Levine--Coe--Martinez gap seeking is independently opt-in:

.. code-block:: json

   {
     "gap_seeking": {
       "enabled": true,
       "mode": "levine_coe_martinez_switch",
       "switch_policy": "stay_fixed",
       "candidate_pairs": "adjacent",
       "trigger_gap_threshold_eV": 0.05,
       "sigma": 3.5,
       "alpha_eV": 0.05
     }
   }

Gap seeking is available only in ``excited_state`` mode. Eligible pairs are
adjacent, explicitly declared ``dE#`` properties containing the batch's
selected state; both states must also declare ``F#`` properties. At each
existing ``Ncheck`` evaluation, including the legacy time-zero check after the
initial MD step, the sampler first applies its normal uncertainty and geometry
handling. Each remaining replica independently selects the eligible pair with
the smallest absolute ensemble-mean gap, using ascending state pairs to break
ties. A gap at or below ``trigger_gap_threshold_eV`` permanently switches only
that replica to

.. math::

   E_\mathrm{LCM} =
   \frac{E_i + E_j}{2}
   + \sigma \frac{\Delta E^2}
     {\sqrt{\Delta E^2 + 10^{-12}} + \alpha}.

The corresponding force is evaluated analytically from the two state-mean
forces. The model is reevaluated immediately after entry so the next MD step
uses LCM forces. ``stay_fixed`` does not switch back, even if the gap later
grows; hysteresis and LCM exit are intentionally unsupported.

Gap seeking loads its required all-state energy and force contributions even
when ``gap_diagnostics`` is disabled. Native HIPPYNN uses raw ensemble
``.all`` outputs, while ASE fallback requests only the extra flattened
``sE#``/``F#`` properties needed by the eligible pairs. Other native
calculators must implement the optional all-state contribution contract.

``stop`` is the compatibility mode. Each replica is frozen independently at
its first valid uncertainty event, while other replicas in the GPU batch keep
running. At most one candidate is returned for each input replica.

``continue`` keeps uncertain replicas active. At every ``Ncheck`` interval,
qualifying frames are ranked across the whole task batch by ``ranking_score``,
capped per replica by ``max_candidates_per_replica``, and truncated to the
global ``return_top_n``. Results are sent directly through Parsl, so large
values of ``return_top_n`` increase task-result serialization and driver memory
use.

Candidate scoring
~~~~~~~~~~~~~~~~~

Admission is always decided by the legacy thresholds ``Es > Escut``,
``Fs > Fscut``, and ``Fsmax > 3*Fscut``, matching the other ALF samplers.
The optional ``score`` block selects only how admitted candidates are *ranked*.

``mode: "max"`` is the default and is used whenever the block is absent. It
ranks by the normalized worst violation,
``max(Es/Escut, Fs/Fscut, Fsmax/(3*Fscut))``.

``mode: "sum"`` ranks by a weighted sum of normalized terms:

.. code-block:: json

   {
     "score": {
       "mode": "sum",
       "w_energy": 1.0,
       "w_force": 1.0,
       "w_gap_uncertainty": 5.0,
       "gap_std_cut_eV": 0.01,
       "gap_pairs": "selected_adjacent"
     }
   }

.. code-block:: text

   ranking_score = w_energy          * (Es    / Escut)
                 + w_force           * (Fsrms / Fscut)
                 + w_gap_uncertainty * (max_pairs(sigma_gap) / gap_std_cut_eV)

Each term is divided by its own cutoff, so every term is dimensionless and
equals one at that channel's threshold. The weights therefore express only
relative importance and remain meaningful across systems and unit choices. Note
that once weights are free, ``ranking_score > 1`` no longer implies "above
threshold"; the trigger, not the score, carries that meaning.

``Fsrms`` is the root-mean-square of the per-component ensemble force deviations
for the selected state. It is used in place of the mean-absolute ``Fs`` because
it is far less diluted by system size for localized disagreement and is directly
comparable to the force RMSE reported during training. ``Fsmax`` is
deliberately absent from the sum: ``Fsmax >= Fs`` always, and ``3*Fscut`` is
calibrated so the maximum and mean channels fire at similar times for a diffuse
deviation distribution, so including both would double-count force disagreement
and make the individual force weights uninterpretable. ``Fsmax`` remains in the
trigger, where it acts as the localized-failure admission net.

The gap term requires ``model_mode`` ``excited_state``. Its state pairs are
declared inside the ``score`` block rather than through ``dE#`` properties,
because the deviation is derived by subtracting per-member state energies the
model already predicts. No trained gap head, HDF5 gap dataset, QM-interface
change, or retraining is needed. ``gap_pairs`` accepts:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value
     - Monitored pairs
   * - ``"selected_adjacent"``
     - Default. Pairs adjacent to the selected state, so an interior state
       yields two pairs and an edge state yields one.
   * - ``"adjacent"``
     - Every ``(i, i+1)`` pair across the configured states.
   * - ``[[0, 1], [1, 2]]``
     - Exactly the listed pairs, normalized to ascending order and deduplicated.

When several pairs are monitored, the term uses the **largest** deviation, and
the winning pair is recorded in ``score_gap_pair``. Requesting the gap term
loads those pairs into the calculator even when ``gap_diagnostics`` is disabled,
so ``gap_means`` and ``gap_stds`` appear in candidate metadata.

``sum`` mode requires a per-member ensemble. A single calculator that supplies
``energy_stdev``, ``forces_stdev_mean``, and ``forces_stdev_max`` directly cannot
produce ``Fsrms``, and the sampler raises rather than substituting ``Fs``.

Candidate metadata always records both scores: ``uncertainty_score`` holds the
max formula in every mode, while ``ranking_score``, ``score_mode``,
``score_components``, ``score_gap_pair``, and ``score_gap_std`` describe the
score that actually ordered the batch. In ``max`` mode the two are equal, so
enabling the block changes nothing until ``mode`` is set to ``sum``.

Batches are grouped by exact atomic-number sequence. Excited-state batches are
also grouped by selected state. ``partial_policy`` currently accepts only
``full_only``: incomplete buckets stay in the running ALF driver's memory,
remain there across model retraining, and use the latest model when they
eventually fill. ``status.txt`` reports bucket counts and ages. The buffers are
not checkpointed across a driver restart.

ALF accounts for batched sampler concurrency in input replicas rather than
Parsl tasks. ``parallel_samplers`` includes replicas in submitted sampler
tasks, the capacity represented by pending builder tasks, and structures
waiting in incomplete sampler buckets. Likewise, ``minimum_QM`` counts
completed input replicas, so a completed batch is drained even when it returns
no candidates. A batch of 50 therefore counts as 50 trajectories; legacy
single-input samplers retain their existing one-task/one-replica behavior.

The status file exposes this accounting under ``sampler_capacity`` with the
configured limit, submitted sampler replicas, pending builder replicas,
buffered replicas, the accounted total, and available replica slots.
``sampler_batching`` continues to report each incomplete compatibility bucket
and its oldest age.

Configuration reloads preserve incomplete structures. Changes to the sampler
task, batched versus legacy mode, batch size, partial policy, model mode, or
state-selection policy are rejected while a sampler batching epoch is active.
An epoch remains active until the strict-full buffer is empty. ALF retains the
complete previous configuration and reports the changed fields. Once the
buffer is empty, the change is accepted and a changed sampler task is reloaded.
Already-submitted tasks retain their recorded input-replica widths, so capacity
and completion accounting remain correct across that boundary.
Threshold, temperature, gap, ranking, and calculator-option edits remain safe
to reload while structures wait and apply when those structures are eventually
submitted.

The first ALCHEMI implementation supports nonperiodic, fixed-cell models.
HIPPYNN is the first fully validated native accelerated backend; other model
families can use the native calculator contract or ASE fallback. Set
``end_dens``, ``amp_dens``, and ``per_dens`` to ``null``. Existing
``MLMD_calculator_options.well_params`` spherical-well settings are supported.
Periodic cells, density schedules, UDD bias, and reactive sampling remain on
their existing sampler tasks.

Temperature scheduling deliberately follows the established molecular MLMD
loop exactly. Each replica independently samples ``Tamp``, ``Tper``, ``Tsrt``,
and ``Tend`` from the configured ranges and starts at
``annealing_schedule(0, ...)``. Dynamics advances one step before the first
uncertainty check; that check is reported at time zero. Subsequent thermostat
updates use the reported interval time before each ``Ncheck``-step block. This
preserves MLMD's legacy one-``dt`` offset between reported time and the
coordinates being checked. Candidate metadata stores the applied update
history in ``temps`` and the four sampled schedule parameters. Temperature
ranges must be finite, ordered two-value ranges, and sampled ``Tper`` values
must be positive.

``friction_per_fs`` is passed directly to ALCHEMI's BAOAB integrator. Its
default, ``0.02 * ase.units.fs`` (approximately ``0.0019645`` per fs), matches
the established ASE MLMD ``Langevin(..., friction=0.02)`` setting.

Excited-state mode
~~~~~~~~~~~~~~~~~~

Set ``model_mode`` to ``excited_state``. A molecule can provide explicit
``selected_state`` metadata, or the sampler can assign states before buffering
with a fixed or batch-cycle policy. The master ``properties_list`` uses a flat
state contract:

.. code-block:: json

   {
     "sE0": ["state_0_energy", "system", 1.0],
     "F0": ["state_0_forces", "atomic", 1.0],
     "sE1": ["state_1_energy", "system", 1.0],
     "F1": ["state_1_forces", "atomic", 1.0],
     "dE01": ["gap_01", "system", 1.0]
   }

State numbers must be contiguous from zero, and every sampled state requires a
force property. Dynamics uses the selected state's ensemble-mean energy and
force; uncertainty uses that state's population energy and force deviations.
An excited-state ASE fallback calculator must expose the selected flattened
``sE#`` and ``F#`` properties; unsupported calculators fail with an actionable
missing-property error.

Candidate provenance records configured gap means, population deviations,
state pairs, model counts, and the minimum absolute gap. These values do not
affect uncertainty flags, stopping, qualification, or top-K order. The
spherical well is applied only to the selected dynamics mean and is excluded
from ensemble uncertainty, gap diagnostics, and switching decisions. With gap
seeking enabled, provenance also records the replica's direct or LCM mode,
active pair, trigger gap, step and time, physical parameters, and its single
entry event. A candidate generated at the triggering check records ``direct``
because that surface produced the frame; later candidates record ``lcm``.
Uncertainty and ranking always remain based on the original selected state.

For one surface, use a fixed policy:

.. code-block:: json

   {
     "model_mode": "excited_state",
     "state_selection": {
       "mode": "fixed",
       "state": 1
     }
   }

To explore several surfaces in one ALF run, assign complete batch-sized blocks
of molecule IDs to states:

.. code-block:: json

   {
     "model_mode": "excited_state",
     "state_selection": {
       "mode": "batch_cycle",
       "states": [0, 1, 2, 3]
     },
     "alchemi_baoab": {
       "batch_size": 50,
       "partial_policy": "full_only"
     }
   }

The last integer in the molecule ID determines its block. With the configuration
above, IDs 0--49 select state 0, IDs 50--99 select state 1, and so forth before
the cycle repeats. IDs without an integer use a stable hash. Explicit
``selected_state`` metadata overrides the policy; the legacy ``excited_state``
metadata key is also accepted and normalized. Empty, duplicate, negative, or
states absent from ``properties_list`` are rejected.

Batch cycling assigns dynamics states, not hardware. Each complete
state-specific batch is submitted as a normal Parsl task, and the sampler
executor maps it to any available GPU. Candidate metadata records the selected
state and actual sampler device/worker assignment. With
``simple_cfg_loader_task``, keep ``maximum_builder_structures`` at ``1``, use
``shake: 0.0`` for exact CFG reuse, and ensure the CFG library has compatible
atomic-number ordering for strict batches.

Gap scoring, GUDD, hysteresis, and LCM exit are not enabled by this sampler
version.

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
