Seeded Excited-State GPU4PySCF
==============================

The example in ``examples/excited_state_gpu4pyscf`` starts from a labeled
HDF5 shard, trains an excited-state HIPPYNN ensemble, samples new structures
with ALCHEMI, labels them with GPU4PySCF, and retrains from accepted data.

The checked-in configuration uses keto acetylacetone and five excited roots,
but the workflow supports other state counts. For ``nroots = N``, each QM
calculation returns ``N + 1`` surfaces numbered ``0`` through ``N``.

Workflow
--------

.. code-block:: text

   User-provided data-0000.h5
       -> initial HIPPYNN ensemble
       -> deterministic HDF5 replay
       -> state-selected ALCHEMI sampling
       -> GPU4PySCF labels
       -> screened HDF5 shard
       -> retraining

GPU4PySCF labeling is nonbatched. One Parsl task evaluates the ground state
and every requested excited root for one molecule on one GPU. Multiple GPUs
increase the number of molecules evaluated concurrently; they do not divide
one molecule's states among devices.

Seed Contract
-------------

Place the user-supplied seed at ``h5store/data-0000.h5``. For ``nroots = N``,
each molecular HDF5 group must contain:

* ``coordinates`` with shape ``[structures, atoms, 3]`` in Angstrom;
* one consistent ``species`` sequence;
* finite energies ``sE0`` through ``sEN`` in eV; and
* finite forces ``F0`` through ``FN`` with shape
  ``[structures, atoms, 3]`` in eV/Angstrom.

Atoms must use ALF's stable atomic-number storage order or provide valid
``topology_atom_ids``. The seed and future GPU4PySCF labels must use the same
molecule, electronic-structure method, state ordering, units, and common
energy offset. Record the seed's provenance and checksum outside the HDF5;
legacy ALF files do not provide sufficient method provenance automatically.

State Count
-----------

Four settings must agree when changing the number of states:

#. Set ``QM_config.json:nroots`` to ``N``.
#. Define every ``sE0/F0`` through ``sEN/FN`` pair in
   ``master_config.json:properties_list``.
#. Provide those datasets in every seed group.
#. Configure the desired states from ``0`` through ``N`` under
   ``sampler_config.json:state_selection``.

The HIPPYNN trainer derives its state heads from ``properties_list``. The
current GPU4PySCF interface requires at least one excited root and supports
singlet RKS/TDA calculations.

Preparing Another Molecule
--------------------------

Replace the topology-reference XYZ, then update its path and charge in the
sampler configuration. Match ``hippynn_config.json:n_atoms`` and
``network_params.possible_species`` to the seed, review network distance
cutoffs and sampling parameters, and set all QM method and convergence
options consistently with the seed. Also review topology, distance, and force
screening thresholds. The workflow expects one fixed atom count and one exact
atomic-number sequence.

Darwin Execution
----------------

The supplied Parsl profile dynamically requests independent resources for
HIPPYNN training, ALCHEMI sampling, and GPU4PySCF labeling. The QM executor
uses one worker per GPU. Account, QoS, CUDA setup, environment paths,
walltimes, block limits, scheduler directives, and cache roots are
configurable through ``ALF_DARWIN_*`` variables.

After preparing the environment and seed, run the component tests from the
repository root and a replay check from the example directory:

.. code-block:: bash

   python -m pytest -q \
     tests/test_gpu4pyscf_interface.py \
     tests/test_h5_replay_builder.py \
     tests/test_excited_state_hippynn_interface.py

   cd examples/excited_state_gpu4pyscf
   python -m alframework --master master_config.json --test_builder
   python -m alframework --master master_config.json --test_qm

For a fresh production run, ``h5store/data-0000.h5`` must exist while an old
``status.txt`` and old models must not. The recommended launch submits a
persistent CPU-only driver through Slurm:

.. code-block:: bash

   sbatch submit_darwin.slurm

The driver allocation requests no GPU. Parsl independently submits the
training, sampling, and GPU4PySCF worker allocations.

Alternatively, run the headnode driver in a persistent ``tmux`` session:

.. code-block:: bash

   hostname
   tmux new -s alf_gpu4pyscf
   cd /absolute/path/to/ALF/examples/excited_state_gpu4pyscf
   bash launch_headnode.sh

Detach with ``Ctrl-b d`` and reconnect to the same recorded frontend before
running ``tmux attach -t alf_gpu4pyscf``. Darwin frontends have separate local
tmux servers. GNU ``screen`` is also supported: start with
``screen -S alf_gpu4pyscf``, detach with ``Ctrl-a d``, and reattach on the same
frontend with ``screen -r alf_gpu4pyscf``.

The interactive process runs only the ALF driver; Parsl still sends all
worker calculations to Slurm. Never run the Slurm driver and a tmux/screen
driver against the same directory simultaneously.

ALF discovers the seed, trains ``models/model-0000``, and then enters the
sampling, labeling, storage, and retraining loop. Resubmitting the same command
resumes from ``status.txt``. Only one driver may control a run directory.

See ``examples/excited_state_gpu4pyscf/README.md`` for the complete seed
schema, generic adaptation checklist, environment setup, expected outputs,
monitoring, and recovery instructions.
