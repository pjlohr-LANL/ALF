Installation
============

ALF is installed as a Python package, but a complete production workflow also
depends on the external QM, ML, scheduler, MPI, and GPU software required.

Prerequisites
-------------

Before installing ALF, use an environment with Python ``>3.9``. A fresh virtual
environment or conda environment is recommended so ALF's Python dependencies do
not conflict with other scientific software stacks.

The Python install includes ALF's core Python dependencies, including Parsl and
ASE. It does not install external executables or site-specific software such as
ORCA, VASP, Q-Chem, scheduler modules, MPI launchers, CUDA/ROCm drivers, or
cluster resource configuration.

Basic Install
-------------

Clone the repository and install ALF in editable mode:

.. code-block:: bash

   git clone https://github.com/lanl/ALF.git
   cd ALF
   python -m pip install -e .

For a quick import check:

.. code-block:: bash

   python -c "import alframework; print('ALF import OK')"

Testing Install
---------------

Install the lightweight unit-test dependencies with the ``tests`` extra:

.. code-block:: bash

   python -m pip install -e ".[tests]"
   python -m pytest

These tests are intended to run without external QM executables, trained ML
models, GPUs, or Parsl executors.

Documentation Install
---------------------

Install the documentation dependencies with the ``docs`` extra:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   sphinx-build -b html docs/source docs/_autobuild/html

The generated HTML pages are written to ``docs/_autobuild/html``.

Optional Scientific Backends
----------------------------

Some workflows need additional packages or external programs. The optional
``full`` extra currently installs ``pyscf`` and ``hippynn``:

.. code-block:: bash

   python -m pip install -e ".[full]"

This is not required for every user and does not cover every supported backend.
HIPPYNN, NeuroChem/ANI, ORCA, VASP, Q-Chem, RDKit, MPI, and GPU toolchains are
workflow-specific. ASE-supported QM engines can still require separately
installed executables, environment variables, pseudopotential paths, and license
or module setup.

ALCHEMI GPU dynamics
~~~~~~~~~~~~~~~~~~~~

Install PyTorch, the supported NVIDIA ALCHEMI release, and HIPPYNN for ALF's
reference native calculator with the ``gpu_dynamics`` extra:

.. code-block:: bash

   python -m pip install -e ".[gpu_dynamics]"

ALF currently targets ``nvalchemi-toolkit>=0.1.0,<0.2``. A CUDA-capable worker
is required for production sampling. CPU execution is available only as an
explicit debugging mode for small smoke tests. Other native model families use
the same calculator interface but may require their own separately installed
dependencies. Existing ALF ASE calculators can use the compatibility fallback
without becoming native batched models.

ALCHEMI topology checking
~~~~~~~~~~~~~~~~~~~~~~~~~

Install RDKit for the optional fixed-topology sampler gate with:

.. code-block:: bash

   python -m pip install -e ".[topology]"

RDKit is imported only when ``topology_check.enabled`` is true, so ordinary
ALF samplers and ALCHEMI runs without topology gating do not require it.

PySEQM excited-state labeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PySEQM is an optional QM backend and is not installed by ALF's base or
``gpu_dynamics`` extras. Install the supported `LANL PYSEQM
<https://github.com/lanl/PYSEQM>`__ source and PyTorch in the environment used
by the selected QM or GPU executor. ALF imports both packages lazily, so other
QM interfaces do not require them.

The initial integration labels one nonperiodic molecule per Parsl task. CPU
execution is supported for development and smoke tests; production GPU use
requires a compatible PyTorch/CUDA installation on the worker.

See :doc:`../user_guide/ml_interfaces`, :doc:`../user_guide/qm_interfaces`, and
:doc:`../user_guide/builders` for backend-specific configuration patterns.

HPC And Parsl Setup
-------------------

ALF installs Parsl as a Python dependency, but users still need to adapt the
Parsl configuration for the machine where ALF will run. In the example
directories, ``parsl_configs.py`` controls scheduler partitions, accounts, QoS,
walltime, worker initialization, launchers, CPU resources, and GPU resources.

See :doc:`../user_guide/parsl` for the executor labels ALF expects and for
templates covering common Slurm CPU/GPU layouts.

Next Steps
----------

* :doc:`quickstart` for the recommended first run.
* :doc:`../examples/simple_water` for the full simple-water walkthrough.
* :doc:`../user_guide/parsl` for cluster and executor setup.
* :doc:`../user_guide/ml_interfaces` for HIPPYNN and ML backend setup.
* :doc:`../user_guide/qm_interfaces` for ORCA, VASP, Q-Chem, and ASE-backed QM
  setup.
