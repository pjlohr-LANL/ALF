Excited-State GPU4PySCF
=======================

The standalone ``examples/excited_state_gpu4pyscf`` workflow labels keto
acetylacetone with density-fitted CAM-B3LYP/6-31G* RKS and five-root TDA. It
uses the flattened six-surface contract ``sE0/F0`` through ``sE5/F5``.

This example is independent of :doc:`excited_state_pyseqm`. In particular,
the two workflows must not share HDF5 labels because they use different
electronic-structure methods.

Execution Model
---------------

GPU4PySCF labeling is single-molecule and nonbatched:

.. code-block:: text

   one MoleculesObject
       -> one Parsl QM task
       -> one A100
       -> S0-S5 energies and forces

All states for a molecule are evaluated on the same GPU. On a four-A100 node,
Parsl can run four independent molecule tasks concurrently. State number is
never used to choose a GPU.

Startup Modes
-------------

The self-contained master follows:

.. code-block:: text

   keto CFG
       -> GPU4PySCF bootstrap
       -> ALF HDF5
       -> multi-state HIPPYNN
       -> state-cycled ALCHEMI
       -> GPU4PySCF labels and retraining

Use ``master_config_existing_h5.json`` to start from a compatible
GPU4PySCF-labeled shard instead. The existing shard must contain finite
``sE0/F0`` through ``sE5/F5`` arrays for the configured 15-atom ordering.
Legacy ALF HDF5 does not carry enough method provenance to detect a manually
copied PySEQM shard, so operators must verify imported data explicitly.

Darwin Resources
----------------

The supplied Parsl profile keeps training, sampling, and labeling on
independent dynamic Slurm providers:

* HIPPYNN training on ``ml4chem``.
* ALCHEMI sampling on ``shared-gpu-ampere``.
* GPU4PySCF labeling on separate ``shared-gpu-ampere`` allocations.

The QM executor exposes four workers and four accelerators per A100 node.
Each worker receives one accelerator and calculates every requested state for
one molecule. Account, QoS, GPU directives, walltime, block limits, worker
environments, source paths, and cache roots are configurable with the
``ALF_DARWIN_*`` variables described in the example README.

Darwin Validation
-----------------

First run the exact CFG geometry through the focused QM check:

.. code-block:: bash

   cd examples/excited_state_gpu4pyscf
   export PYTHONPATH=/path/to/ALF:${PYTHONPATH:-}
   python -m alframework --master master_config_debug.json --test_builder
   python -m alframework --master master_config_debug.json --test_qm

Compare the resulting S0-S5 energies and forces with the five-root
``dataset_workflow`` GPU4PySCF backend using identical method, basis, charge,
and grid settings. Then submit several structures and confirm from metadata
and worker logs that each molecule stays on one GPU while different molecules
can use different A100s.

See ``examples/excited_state_gpu4pyscf/README.md`` for installation,
bootstrap, existing-HDF5, scheduler, restart, and acceptance details.
