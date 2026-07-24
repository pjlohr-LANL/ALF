Excited-State PySEQM
====================

The production-first example in ``examples/excited_state_pyseqm`` reproduces
a six-state acetylacetone keto active-learning workload. It starts from one
existing HDF5 shard, trains a shared-trunk HIPPYNN ensemble, replays labeled
geometries, samples state-specific ALCHEMI batches, labels candidates with
PySEQM, and retrains from accepted shards.

Workflow
--------

.. code-block:: text

   Existing six-state HDF5
       -> HIPPYNN
       -> HDF5 replay
       -> batched ALCHEMI
       -> screened PySEQM
       -> new HDF5 and retraining

The configured properties are ``sE0`` through ``sE5`` and ``F0`` through
``F5``. Gap targets and gap-seeking features are disabled in the production
acceptance profile.

ALF Components
--------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Stage
     - Task
   * - Builder
     - ``alframework.builders.h5_replay_builder.h5_replay_builder_task``
   * - Sampler
     - ``alframework.samplers.alchemi_sampling.alchemi_sampling_task``
   * - QM
     - ``alframework.qm_interfaces.pyseqm_interface.pyseqm_excited_state_task``
   * - ML
     - ``alframework.ml_interfaces.excited_state_hippynn_interface.train_excited_state_HIPPYNN_ensemble_task``

The replay builder reads only shards below ``current_h5_id``. ALF stores each
shard in stable atomic-number order; replay restores the fixed topology's
canonical order before dynamics. Completed QM labels are screened by maximum
force, minimum distance, and topology before storage.

Dynamic Darwin Execution
------------------------

The driver uses independent, zero-initial-block Slurm providers:

* HIPPYNN training on ``ml4chem``.
* ALCHEMI sampling and replay on ``shared-gpu-ampere``.
* PySEQM labeling on separate ``shared-gpu-ampere`` allocations.

The supplied submission script runs the persistent driver on ``general``.
Worker account, CUDA setup, environment activation, walltimes, and block
limits remain configurable through ``ALF_DARWIN_*`` variables.

Running the Example
-------------------

Copy only the compatible seed shard to ``h5store/data-0000.h5`` and verify its
checksum as documented in the example README. Run the stage checks in order:

.. code-block:: bash

   cd examples/excited_state_pyseqm
   source /projects/opt/centos8/x86_64/miniconda3/py312_24.11.1/etc/profile.d/conda.sh
   conda activate /vast/home/pjlohr/.conda/envs/atomistic
   export PYTHONPATH=/vast/home/pjlohr/ALF_LANL/ALF_fork/ALF:${PYTHONPATH:-}
   python -m alframework --master master_config_debug.json --test_builder
   python -m alframework --master master_config_debug.json --test_qm
   python -m alframework --master master_config_debug.json --test_ml
   python -m alframework --master master_config_debug.json --test_sampler

Then submit the fresh production replica:

.. code-block:: bash

   sbatch submit_darwin.slurm

The source production status and models are not imported. See the example
README for checksums, acceptance criteria, output paths, and restart guidance.
