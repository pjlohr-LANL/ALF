QM Interfaces
=============

QM interfaces connect ALF to electronic structure engines.

Available QM interface modules
------------------------------

ALF currently includes several QM interface modules:

1. ``alframework.qm_interfaces.ase_calculator_interface``
   Generic QM task wrappers for use with ASE calculators.
2. ``alframework.qm_interfaces.orca5_interface``
   Interface for ORCA, including task and parsing utilities for single-point calculations and
   property extraction.
3. ``alframework.qm_interfaces.vaspase_interface``
   Interface for the Vienna ab initio Simulation Package (VASP) plane-wave electronic structure code.
4. ``alframework.qm_interfaces.siesta_interface``
   Interface for SIESTA through ASE's ``Siesta`` calculator.

Choosing a QM interface
-----------------------

Use ``ase_calculator_interface`` when your QM engine is exposed through an ASE
calculator and you want a general integration path.

Use ``orca5_interface`` when your workflow is built around ORCA input blocks
and ORCA-specific output parsing.

Use ``vaspase_interface`` when your workflow requires VASP-specific controls
and file handling beyond the generic ASE path.

Use ``siesta_interface`` when your workflow uses SIESTA through ASE and you
want a dedicated task entry point with ALF's standard ``MoleculesObject`` flow.

SIESTA configuration pattern
----------------------------

Set the QM task in ``master_config.json``:

.. code-block:: json

   "QM_task": "alframework.qm_interfaces.siesta_interface.siesta_calculator_task"

Provide SIESTA settings in ``QM_config.json``:

.. code-block:: json

   {
     "QM_run_command": "siesta < PREFIX.fdf > PREFIX.out",
     "input": {
       "label": "siesta",
       "mesh_cutoff": 300.0,
       "energy_shift": 0.01,
       "basis_set": "DZP",
       "kpts": [1, 1, 1],
       "xc": "GGA",
       "fdf_arguments": {
         "MaxSCFIterations": 200,
         "DM.MixingWeight": 0.1
       }
     }
   }

Notes:

1. Provide pseudopotentials through ``input.pseudo_path`` or ``SIESTA_PP_PATH``.
2. Use ``input.fdf_arguments`` for raw FDF keywords that are not mapped to explicit ASE parameters.

QM interface API links
----------------------

You can link from this guide directly to API pages:

* :doc:`QM interfaces package API <../api_documentation/alframework.qm_interfaces>`
* :doc:`ASE calculator interface module <../api_documentation/alframework.qm_interfaces.ase_calculator_interface>`
* :doc:`ORCA interface module <../api_documentation/alframework.qm_interfaces.orca5_interface>`
* :doc:`VASP interface module <../api_documentation/alframework.qm_interfaces.vaspase_interface>`
* :doc:`SIESTA interface module <../api_documentation/alframework.qm_interfaces.siesta_interface>`
