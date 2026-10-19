import copy

import numpy as np
from ase import Atoms
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.molecular_topology import ReferenceTopology
from ase.calculators.calculator import Calculator, all_changes


class FakeASECalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, atoms=None, directory=None, command=None, fake_results=None,
                 fake_converged=True, fake_expected_command=None, fake_expected_properties=None, **kwargs):
        super().__init__()
        self.init_atoms = atoms
        self.directory = directory
        self.command = command
        self.kwargs = kwargs
        self.fake_results = fake_results
        self.converged = fake_converged
        self.fake_expected_command = fake_expected_command
        self.fake_expected_properties = fake_expected_properties
        self.calculate_calls = []

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calculate_calls.append((atoms, list(properties)))
        if self.fake_expected_command is not None:
            assert self.command == self.fake_expected_command
        if self.fake_expected_properties is not None:
            assert list(properties) == list(self.fake_expected_properties)

        if self.fake_results is None:
            self.results = {"energy": -1.25, "forces": np.ones((len(atoms), 3))}
        else:
            self.results = copy.deepcopy(self.fake_results)


class FixedCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, energy, forces):
        super().__init__()
        self.energy = energy
        self.forces = np.array(forces, dtype=float)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = self.energy
        self.results["forces"] = self.forces.copy()


class FakeUncertaintyCalculator(Calculator):
    implemented_properties = ["energy_stdev", "forces_stdev_mean", "forces_stdev_max"]

    def calculate(self, atoms=None, properties=("energy_stdev",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy_stdev": 0.0,
            "forces_stdev_mean": 0.0,
            "forces_stdev_max": 0.0,
        }


class FakeLangevin:
    def __init__(self, atoms, timestep, friction, temperature_K):
        self.atoms = atoms
        self.timestep = timestep
        self.friction = friction
        self.temperature_K = temperature_K
        self.run_calls = []

    def run(self, steps):
        self.run_calls.append(steps)

    def set_temperature(self, temperature_K):
        self.temperature_K = temperature_K


class FakeTask:
    def __init__(self, status, result=None, done=True, running=False, exception=None):
        self._status = status
        self._result = result
        self._done = done
        self._running = running
        self._exception = exception

    def done(self):
        return self._done

    def running(self):
        return self._running

    def task_status(self):
        return self._status

    def result(self):
        if self._exception is not None:
            raise self._exception
        return self._result


def _molecule(molecule_id, distance=0.74, state=None):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
    molecule = MoleculesObject(atoms, molecule_id)
    if state is not None:
        molecule.update_metadata({"selected_state": state})
    return molecule


def _topology_molecule(molecule_id, positions=None, state=None):
    if positions is None:
        positions = [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    molecule = MoleculesObject(Atoms("CCC", positions=positions), molecule_id)
    if state is not None:
        molecule.update_metadata({"selected_state": state})
    return molecule


def _reference_topology():
    return ReferenceTopology(
        reference_path="/master/reference.mol",
        reference_sha256="reference-hash",
        reference_format="mol",
        reference_charge=0,
        atomic_numbers=(6, 6, 6),
        bonds=((0, 1), (1, 2)),
        bond_lengths=(1.5, 1.5),
        bond_min_scale=0.70,
        bond_max_scale=1.35,
        connectivity_scale=1.25,
    )


def _enable_topology(config):
    config["topology_check"] = {
        "enabled": True,
        "reference_conformer_path": "reference.mol",
        "reference_format": "auto",
        "reference_charge": 0,
        "bond_min_scale": 0.70,
        "bond_max_scale": 1.35,
        "connectivity_scale": 1.25,
    }
    return config


def _config(policy="stop", batch_size=2, return_top_k=2):
    return {
        "model_mode": "ground_state",
        "uncertainty_policy": policy,
        "return_top_k": return_top_k,
        "dt": 100.0,
        "maxt": 0.2,
        "Ncheck": 1,
        "Escut": 1.0,
        "Fscut": 1.0,
        "min_time": 0.0,
        "distcut": 0.1,
        "srt_temp": [300.0, 300.0],
        "end_temp": [300.0, 300.0],
        "amp_temp": [0.0, 0.0],
        "per_temp": [1.0, 1.0],
        "end_dens": None,
        "amp_dens": None,
        "per_dens": None,
        "alchemi_baoab": {
            "batch_size": batch_size,
            "partial_policy": "full_only",
            "random_seed": 7,
        },
    }


def _gap_row():
    return {
        "lower_state": 0,
        "upper_state": 1,
        "gap_key": "dE01",
        "gap_db_name": "dE01",
    }


def _excited_properties_with_gap():
    return {
        "sE0": ["sE0", "system", 1.0],
        "F0": ["F0", "atomic", 1.0],
        "sE1": ["sE1", "system", 1.0],
        "F1": ["F1", "atomic", 1.0],
        "dE01": ["dE01", "system", 1.0],
    }


def _excited_properties_three_states():
    return {
        "sE0": ["sE0", "system", 1.0],
        "F0": ["F0", "atomic", 1.0],
        "sE1": ["sE1", "system", 1.0],
        "F1": ["F1", "atomic", 1.0],
        "sE2": ["sE2", "system", 1.0],
        "F2": ["F2", "atomic", 1.0],
        "dE01": ["dE01", "system", 1.0],
        "dE12": ["dE12", "system", 1.0],
        "dE02": ["dE02", "system", 1.0],
    }


def _enable_gap_seeking(config, **overrides):
    config["model_mode"] = "excited_state"
    config["gap_seeking"] = {
        "enabled": True,
        "mode": "levine_coe_martinez_switch",
        "switch_policy": "stay_fixed",
        "candidate_pairs": "adjacent",
        "trigger_gap_threshold_eV": 0.05,
        "sigma": 3.5,
        "alpha_eV": 0.05,
        **overrides,
    }
    return config


class FakeModel:
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics


class FakeRunner:
    last_instance = None

    def __init__(
        self,
        *,
        model,
        atoms_list,
        dt_fs,
        temperature_K,
        friction_per_fs,
        random_seed,
        device,
    ):
        del dt_fs, random_seed, device
        type(self).last_instance = self
        self.model = model
        self.initial_temperature_K = np.asarray(temperature_K, dtype=float)
        self.friction_per_fs = float(friction_per_fs)
        self.temperature_calls = []
        self.positions = [atoms.get_positions().copy() for atoms in atoms_list]
        self.velocities = [np.zeros_like(value) for value in self.positions]
        self.frozen = set()
        self.lcm_calls = []
        self.nsteps = 0
        self.evaluation_index = -1

    def run(self, steps):
        for _ in range(int(steps)):
            self.nsteps += 1
            for index, positions in enumerate(self.positions):
                if index not in self.frozen:
                    positions[:, 0] += 0.01

    def evaluate(self):
        self.evaluation_index += 1

    def diagnostics_for_graph(self, graph_index):
        sequence = self.model.diagnostics[graph_index]
        return sequence[min(self.evaluation_index, len(sequence) - 1)]

    def freeze_graph(self, graph_index):
        self.frozen.add(int(graph_index))

    def set_lcm_mode(self, graph_index, *, pair, sigma, alpha_eV):
        self.lcm_calls.append(
            {
                "graph_index": int(graph_index),
                "pair": list(pair),
                "sigma": float(sigma),
                "alpha_eV": float(alpha_eV),
                "step": int(self.nsteps),
            }
        )

    def set_temperature(self, temperature_K):
        self.temperature_K = np.asarray(temperature_K)
        self.temperature_calls.append(self.temperature_K.copy())

    def sync_graph_to_atoms(self, graph_index, atoms):
        atoms.set_positions(self.positions[int(graph_index)])
        atoms.set_velocities(self.velocities[int(graph_index)])


class TopologyMutationRunner(FakeRunner):
    def sync_graph_to_atoms(self, graph_index, atoms):
        updates = getattr(self.model, "position_updates", {})
        positions = updates.get(
            self.evaluation_index + 1,
            {},
        ).get(int(graph_index))
        if positions is not None:
            self.positions[int(graph_index)] = np.asarray(
                positions,
                dtype=float,
            ).copy()
        super().sync_graph_to_atoms(graph_index, atoms)


def _torch_batch(torch, positions=None):
    class Batch:
        pass

    batch = Batch()
    if positions is None:
        positions = np.zeros((4, 3), dtype=float)
    batch.positions = torch.as_tensor(positions, dtype=torch.float32)
    batch.atomic_numbers = torch.ones(4, dtype=torch.long)
    batch.atomic_masses = torch.ones(4)
    batch.num_graphs = 2
    batch.num_nodes_per_graph = torch.as_tensor([2, 2])
    return batch
