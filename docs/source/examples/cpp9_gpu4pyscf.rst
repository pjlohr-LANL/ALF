CPP9 Finite GPU4PySCF Trial
===========================

``examples/cpp9_gpu4pyscf`` is a portable label-and-train example with a bundled
1,000-frame unlabeled C54H36 geometry pool and an optimized S0 reference.
It uses four-state CAM-B3LYP/6-31G RKS/TDA labeling, stores energies, forces,
transition dipoles, oscillator strengths and excited-excited NACs, and trains
four HIP-HOP models on energies and forces only. It stops after this finite
input pool and ensemble training; it does not start acquisition dynamics.

Inputs and reference snapshots are tracked in Git. Copy the example alongside
an installed production excited-state ALF fork, or use it from a cloned fork.
No original benchmark directory, historical labels, models or sampling tools
are required. The example-local optical extension reuses ALF's shared QM
helpers without changing the core workflow or public GPU4PySCF interface.

Bring-up
--------

Activate a compatible GPU4PySCF/HIPPYNN environment, edit ``resources.json``,
and run from the example directory:

.. code-block:: bash

   bash launch.sh --check
   bash launch.sh --run-dir "$PWD/runs/smoke" --limit 1 --label-only
   bash launch.sh --run-dir "$PWD/runs/trial_001"
   bash launch.sh --run-dir "$PWD/runs/trial_001" --resume

The check submits no work. The labeling smoke and full commands perform real
GPU calculations. Defaults use local GPU workers; configure the number of
GPUs/threads to match hardware, or choose Slurm and supply site-specific
partitions, GPU requests and environment initialization. The optional Slurm
driver wrapper also needs site-specific driver resource arguments.

Scientific settings, reference conversion, output shapes, resource details,
checkpoint limitations and transfer instructions are documented in
``examples/cpp9_gpu4pyscf/README.md``. Completed QM rejections are counted
without replacement; only retained, training-filtered rows enter the ensemble.
The initial geometry pool is a local perturbation sample, not a thermal or
comprehensive conformational reference.
