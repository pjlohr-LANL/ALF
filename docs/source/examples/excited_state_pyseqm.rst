Excited-State PySEQM
====================

The excited-state PySEQM example is a complete molecular active-learning
workflow using ALF's standard CFG builder, the multi-state HIPPYNN trainer,
batched ALCHEMI dynamics, and single-molecule PySEQM labeling. The example
files are located in ``examples/excited_state_pyseqm``.

Workflow
--------

Two startup modes are provided:

.. code-block:: text

   CFG -> PySEQM bootstrap -> HDF5 -> HIPPYNN -> ALCHEMI sampling

.. code-block:: text

   Existing HDF5 -> HIPPYNN
   CFG -> ALCHEMI sampling -> PySEQM -> additional HDF5 shards

The CFG file supplies only atomic identity, order, coordinates, and cell
information. It is not the storage format for multiple electronic-state
labels. PySEQM generates the flattened ``sE#`` and ``F#`` properties, and ALF
stores those labels in HDF5 for training.

ALF Components
--------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Stage
     - Task
   * - Builder
     - ``alframework.builders.builders.simple_cfg_loader_task``
   * - Sampler
     - ``alframework.samplers.alchemi_sampling.alchemi_sampling_task``
   * - QM
     - ``alframework.qm_interfaces.pyseqm_interface.pyseqm_excited_state_task``
   * - ML
     - ``alframework.ml_interfaces.excited_state_hippynn_interface.train_excited_state_HIPPYNN_ensemble_task``

ASE's CFG reader marks AtomEye CFG inputs periodic. The example therefore uses
the loader's optional ``pbc: false`` override so its water molecule satisfies
ALCHEMI's nonperiodic fixed-cell contract.

Dynamic Darwin Execution
------------------------

The ALF main process runs on a head node. The example's Parsl configuration
dynamically requests independent Slurm blocks:

* HIPPYNN training on ``ml4chem`` through ``alf_ML_executor``.
* ALCHEMI sampling on ``shared-gpu-ampere`` through
  ``alf_sampler_executor``.
* PySEQM labeling on separate ``shared-gpu-ampere`` allocations through
  ``alf_QM_executor``.

Every provider starts with zero blocks. Account, QoS, GPU directives,
walltimes, block limits, CUDA setup, and environment activation are exposed as
settings in ``parsl_configs.py`` for adjustment before a Darwin run.

Running the Example
-------------------

For a self-contained bootstrap:

.. code-block:: bash

   cd examples/excited_state_pyseqm
   python -m alframework --master master_config.json

To begin with existing labeled data, place compatible files at
``h5store/data-0000.h5`` and run:

.. code-block:: bash

   python -m alframework --master master_config_existing_h5.json

When the HDF5 store exists and no status file exists, ALF detects the data,
skips bootstrap labeling, and trains the initial ensemble before sampling.
The existing shards must match the configured state/gap database names and
exact atomic-number ordering.

Stage Checks
------------

The debug master uses one-replica sampling:

.. code-block:: bash

   python -m alframework --master master_config_debug.json --test_builder
   python -m alframework --master master_config_debug.json --test_qm
   python -m alframework --master master_config_debug.json --test_ml
   python -m alframework --master master_config_debug.json --test_sampler

ML testing requires existing HDF5 data, and sampler testing requires an
existing model. See the example README for the expected order, full
configuration inventory, worker-environment settings, topology behavior, and
restart guidance.
