QM Interfaces
=============

QM interfaces connect ALF to electronic structure engines. They receive
candidate structures from the sampler, run or parse a quantum-mechanical
calculation, and return labeled data for storage in ALF's HDF5 training sets.

Supported QM Packages
---------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Package
     - Associated interface
   * - `ORCA <https://www.faccts.de/orca/>`__
     - ``alframework.qm_interfaces.orca5_interface.orca_calculator_task``
   * - `VASP <https://vasp.at/>`__
     - ``alframework.qm_interfaces.ase_calculator_interface.VASP_ase_calculator_task``
   * - `Q-Chem <https://www.q-chem.com/>`__
     - ``alframework.qm_interfaces.qchem_DFT_interface.qchem_dft_calculator_task``
   * - `ASE-supported calculators <https://docs.ase-lib.org/ase/calculators/calculators.html#supported-calculators>`__
     - ``alframework.qm_interfaces.ase_calculator_interface.ase_calculator_task``
   * - `PySEQM <https://github.com/lanl/PYSEQM>`__
     - ``alframework.qm_interfaces.pyseqm_interface.pyseqm_excited_state_task``
   * - `GPU4PySCF <https://github.com/pyscf/gpu4pyscf>`__
     - ``alframework.qm_interfaces.gpu4pyscf_interface.gpu4pyscf_excited_state_task``

When a QM engine already has a reliable ASE calculator, the generic ASE task is
usually the easiest integration route. Write a small QM config that identifies
the ASE calculator class, command, and calculator options, then let ASE handle
input writing, execution, and result extraction.

PySEQM Excited-State Labeling
-----------------------------

The PySEQM interface uses ALF's flattened state contract: ``sE0``, ``F0``,
``sE1``, ``F1``, and so on. State energies must be contiguous from zero.

PySEQM convergence is a hard acceptance condition. Immediately after the
electronic-structure call, ALF requires one Boolean ``driver.notconverged``
flag per input molecule. Any true flag raises
``PySEQMConvergenceError`` before energies or forces are extracted. Missing or
malformed flags are also rejected. The returned molecule is marked
unconverged, stores no requested state labels, and records the error and failed
batch indices in metadata. Successful labels record
``qm_scf_converged: true``.
Force entries may be omitted for energy-only datasets, although excited-state
dynamics and force training require the corresponding ``F#`` properties.

Select either the ordinary QM executor task or the GPU executor task:

.. code-block:: json

   {
     "QM_task": "alframework.qm_interfaces.pyseqm_interface.pyseqm_excited_state_task"
   }

.. code-block:: json

   {
     "QM_task": "alframework.qm_interfaces.pyseqm_interface.pyseqm_excited_state_gpu_task"
   }

The shared QM configuration is:

.. code-block:: json

   {
     "method": "AM1",
     "scf_eps": 1.0e-10,
     "cis_tol": 1.0e-8,
     "energy_offset_eV": 0.0,
     "max_solve_time_seconds": 60,
     "capture_pyseqm_logs": false,
     "pyseqm_log_dir": "pyseqm_logs"
   }

``energy_offset_eV`` is subtracted from every state energy. It belongs in the
QM configuration; the older sampler-configuration location remains a fallback
for compatibility with fork configurations. A positive
``max_solve_time_seconds`` isolates the solve in a child process and marks the
molecule non-converged if the limit is exceeded. Zero or ``null`` disables the
timeout. Optional logs include the selected state, ALCHEMI candidate context,
device, solver settings, elapsed time, and failure details.

Optional flattened gap labels are ordinary ALF system properties:

.. code-block:: json

   {
     "dE01": ["gap_01", "system", 1.0]
   }

PySEQM derives ``dEij = sEj - sEi`` after producing all requested state
energies. The common energy offset therefore cancels. Gap properties must use
the same storage scale as both source energies. Missing or malformed gap
schemas reject the molecule before labels are stored.

PySEQM requires atoms to be ordered by decreasing atomic number. The interface
performs that stable ordering internally and restores every state force to the
original ALF atom order before storing results. Malformed or non-finite backend
outputs are rejected and returned as non-converged molecules with diagnostic
metadata.

This first integration intentionally submits one molecule per ALF QM task.
PySEQM tensor evaluation still has a leading batch dimension of one, but there
is no QM buffer or batching change in the main ALF driver. Periodic systems,
topology filtering, and pre-labeled bypasses are not supported by this
interface version.

GPU4PySCF Excited-State Labeling
--------------------------------

The dedicated GPU4PySCF task produces ground- and excited-state energies and
forces through the same flattened property contract:

.. code-block:: json

   {
     "QM_task": "alframework.qm_interfaces.gpu4pyscf_interface.gpu4pyscf_excited_state_task"
   }

The validated default calculation is a neutral-singlet, density-fitted
CAM-B3LYP/6-31G* RKS calculation with grid level 3 followed by TDA with five
excited roots:

.. code-block:: json

   {
     "xc": "cam-b3lyp",
     "basis": "6-31g*",
     "charge": 0,
     "multiplicity": 1,
     "density_fit": true,
     "auxbasis": null,
     "grids_level": 3,
     "nroots": 5,
     "scf_conv_tol": 1.0e-10,
     "scf_max_cycle": 100,
     "tda_conv_tol": 1.0e-8,
     "tda_max_cycle": 100,
     "num_threads": 8,
     "max_memory_mb": null,
     "verbosity": 0,
     "energy_offset_eV": 0.0
   }

With ``nroots: 5``, ``properties_list`` must contain every ``sE#`` and ``F#``
from state zero through state five. The task stores total state energies in eV
and forces in eV/Angstrom. Explicit ``dE#`` properties are derived after the
common energy offset is applied, so the offset cancels.

SCF and every requested TDA root must expose valid Boolean convergence flags.
The interface rejects the entire molecule on any convergence, gradient,
shape, or finiteness failure and never stores partial labels. Successful and
failed calculations record the backend, selected CUDA device, convergence
status, and elapsed time. Successful calculations also record the calculation
settings and dependency versions.

GPU selection follows ALF's Parsl worker-rank convention. The task requires a
real CUDA device, has no CPU fallback, and does not bind electronic states to
particular GPUs. It submits one molecule per ordinary QM task and relies on
Parsl/Slurm walltime rather than a child-process timeout.

Install the GPU4PySCF package matching the worker CUDA runtime separately, for
example ``gpu4pyscf-cuda12x`` on the supplied Darwin CUDA 12 profile. Imports
remain lazy so ordinary ALF installations do not require GPU4PySCF. The task
calls GPU4PySCF directly because its standard ASE adapter does not expose
ALF's multi-state ``sE#``/``F#`` properties.

Only singlet RKS/TDA is supported in this slice. Roots are energy ordered at
each geometry. Dipoles, transition dipoles, NACVs, full TDDFT, state tracking,
PBC, and QM batching are not included.

See :doc:`../examples/excited_state_gpu4pyscf` for separate CFG-bootstrap,
existing-HDF5, and Darwin validation configurations. The GPU4PySCF and PySEQM
examples intentionally use independent data and output directories.

.. note::

   New VASP configurations should use
   ``alframework.qm_interfaces.ase_calculator_interface.VASP_ase_calculator_task``.
   The older ``alframework.qm_interfaces.vaspase_interface.VASPGenerator`` helper
   is deprecated and kept only for backward compatibility. For custom VASP-style
   workflows, prefer the generic ``ase_calculator_task`` whenever the engine can
   be represented as an ASE calculator.

.. _qm-interface-extension-points:

How ALF Uses QM Interfaces
--------------------------

ALF uses QM code through a small set of master-configuration hooks:

QM task
   The ``QM_task`` field in ``master_config.json`` is an import string for the
   labeling task. ALF submits this task through Parsl on the
   ``alf_QM_executor`` executor.

QM configuration
   The ``QM_config_path`` field points to an engine-specific JSON file. This
   file usually contains the executable command, method settings, CPU count,
   input blocks, and any engine-specific options.

Scratch directory
   The ``QM_scratch_dir`` field controls where per-structure calculation
   directories are created. QM tasks typically create one subdirectory per
   ``MoleculesObject`` id.

Properties
   The ``properties_list`` field defines which properties ALF requests and how
   those properties are written to HDF5. It also records whether each property
   is system-level or atom-level and includes the unit conversion used when
   storing data. See :doc:`units` for how these multipliers are applied.

Common Configuration Fields
---------------------------

Master configuration
   These fields connect the ALF active-learning loop to a QM engine.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``QM_task``
        - Import string for the labeling task.
      * - ``QM_config_path``
        - JSON file containing engine-specific QM settings.
      * - ``QM_scratch_dir``
        - Directory for per-structure QM input, output, and scratch files.
      * - ``properties_list``
        - Requested properties, HDF5 dataset names, property scope, and unit
          conversion factors.

QM configuration
   Exact fields differ by backend, but QM config files commonly include:

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Meaning
      * - ``QM_run_command``
        - Executable or launch command used by the QM task.
      * - ``ncpu``
        - CPU count passed to engines or input writers when supported.
      * - environment file
        - Optional shell setup file, such as ``orca_env_file`` or
          ``qchem_env_file``.
      * - method/input blocks
        - Engine-specific method, basis, SCF, memory, k-point, and property
          settings.
      * - scratch/output paths
        - Backend-specific controls for where input, output, and restart files
          are written.

.. note::
   ALF does not choose a universally appropriate QM method, basis set,
   pseudopotential, dispersion correction, or charge / spin state setting for a
   workflow. These choices should be made explicitly in the engine-specific QM
   configuration and checked against the intended chemistry. For ORCA and
   Q-Chem, the method, basis, and property requests come from the configured
   input/rem blocks, while the current task wrappers call the generators as
   neutral singlets unless customized. For VASP through the current ASE
   interface, INCAR-style settings come from ``QM_config["input"]`` and ASE's
   VASP calculator behavior. The deprecated legacy VASP helper is the exception
   with ALF compatibility defaults documented further below.

For HPC runs, the QM command and Parsl resource configuration must agree. For
example, an MPI VASP command such as ``srun -n 128 vasp_std`` should be paired
with an ``alf_QM_executor`` configuration that requests the matching nodes,
tasks, walltime, modules, and launcher behavior. See :doc:`parsl` for
practical resource examples.

Legacy VASP helper
   ``alframework.qm_interfaces.vaspase_interface.VASPGenerator`` is deprecated
   and kept only for older imports. Do not start new configuration files from
   ``vaspase_interface.py``. Existing imports may continue to work temporarily,
   but should be migrated to
   ``alframework.qm_interfaces.ase_calculator_interface.VASP_ase_calculator_task``
   or the generic ``ase_calculator_task`` described above.

   The legacy helper also applies ALF-specific VASP defaults before merging
   user-provided ``vasp_options``:

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - VASP setting
        - Legacy default
      * - ``xc``
        - ``"pbe"``
      * - ``prec``
        - ``"Accurate"``
      * - ``ncore``
        - ``1`` unless ``vasp_options["ncore"]`` is provided.
      * - ``lreal``
        - ``"Auto"``
      * - ``nelm``
        - ``120`` unless ``vasp_options["nelm"]`` is provided.
      * - ``ivdw``
        - ``0`` unless ``vasp_options["ivdw"]`` is provided.

   These are compatibility defaults from the old helper, not universal
   recommendations for VASP calculations. In particular, users should be aware
   that the legacy path sets ``LREAL = Auto`` by default; review this setting
   when reproducing old calculations, comparing force labels, or migrating to
   the ASE calculator interface. The old helper also maps ``vasp_options["kpoints"]``
   to ASE's ``kpts`` option and falls back to atomic-number magnetic moments
   unless ``magmom`` is provided.

.. _qm-interface-new-engine-template:

Adding New QM Engines
-----------------------------------

Preferred ASE route
   Use this path when ASE already provides a calculator for the engine.
   Configure ``ase_calculator_task`` as the QM task and include an
   ``ASE_calculator`` import string in the QM config:

   .. code-block:: json

      {
        "QM_task": "alframework.qm_interfaces.ase_calculator_interface.ase_calculator_task",
        "QM_config_path": "my_qm_config.json",
        "QM_scratch_dir": "qm_scratch/"
      }

   .. code-block:: json

      {
        "ASE_calculator": "ase.calculators.mycode.MyCode",
        "QM_run_command": "mycode_executable",
        "xc": "PBE",
        "basis": "def2-SVP"
      }

   The generic task loads the ASE calculator class, creates a
   molecule-specific scratch directory, runs ``calc.calculate(...)``, stores
   ``calc.results``, sets the convergence flag from ``calc.converged``, and
   returns the updated ``MoleculesObject``.

Custom parser route
   Use this path when ASE does not support the engine or when ALF needs
   backend-specific parsing and convergence checks. Define a Parsl task with
   the ALF QM task signature:

   .. code-block:: python

      import os
      from parsl import python_app


      @python_app(executors=["alf_QM_executor"])
      def my_qm_task(molecule_object, QM_config, QM_scratch_dir, properties_list):
          molecule_id = molecule_object.get_moleculeid()
          directory = os.path.join(QM_scratch_dir, molecule_id)
          os.makedirs(directory, exist_ok=False)

          atoms = molecule_object.get_atoms()
          requested_properties = list(properties_list.keys())

          write_mycode_input(atoms, QM_config, directory, requested_properties)
          run_mycode(QM_config["QM_run_command"], directory)

          results = parse_mycode_output(directory, requested_properties)
          converged = parse_mycode_convergence(directory)

          molecule_object.store_results(results)
          molecule_object.set_converged_flag(converged)
          return molecule_object

   The ``results`` dictionary should use the same property names requested in
   ``properties_list``. Custom tasks should document the units they place in
   ``MoleculesObject.results`` so the master configuration can apply the
   correct HDF5 conversion factor.

Implementation checklist
   Before using a new QM backend in production, check that:

   * The QM task runs on ``alf_QM_executor`` without requiring ML/GPU imports.
   * The scratch directory is unique for each ``MoleculesObject`` id.
   * Requested properties match the keys in ``properties_list``.
   * Convergence is set explicitly with ``set_converged_flag``.
   * Failed or incomplete calculations return a non-converged
     ``MoleculesObject`` or raise an error that ALF can record.
   * The launch command matches the Parsl resource allocation, especially for
     MPI engines.

QM Interface API Links
----------------------

You can link from this guide directly to API pages:

* :doc:`QM interfaces package API <../api_documentation/alframework.qm_interfaces>`
* :doc:`ASE calculator interface module <../api_documentation/alframework.qm_interfaces.ase_calculator_interface>`
* :doc:`ORCA interface module <../api_documentation/alframework.qm_interfaces.orca5_interface>`
* :doc:`QChem interface module <../api_documentation/alframework.qm_interfaces.qchem_DFT_interface>`
* :doc:`PySEQM interface module <../api_documentation/alframework.qm_interfaces.pyseqm_interface>`
* :doc:`GPU4PySCF interface module <../api_documentation/alframework.qm_interfaces.gpu4pyscf_interface>`
* :doc:`Legacy VASP interface module <../api_documentation/alframework.qm_interfaces.vaspase_interface>`
* :doc:`Parsl execution guide <parsl>`

Related Examples
----------------

See :doc:`../examples/simple_water` and :doc:`../examples/reactive_sampling`
for ORCA-backed workflows, and :doc:`../examples/molten_salt` for a
VASP-backed workflow.
