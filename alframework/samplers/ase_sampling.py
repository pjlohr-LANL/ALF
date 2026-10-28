"""ASE Langevin execution behind the shared sampling-driver protocol."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin

from alframework.ml_interfaces.sampling_inference import evaluate_model


class ASEDynamicsRunner:
    """Independent ASE systems using shared inference and acquisition controls."""

    supports_constraints = True
    supports_partial_pbc = True

    def __init__(self, *, model, atoms_list, dt_fs, temperature_K, friction_per_fs,
                 random_seed, device=None):
        if not atoms_list:
            raise ValueError("ASE dynamics requires at least one system.")
        if not np.isfinite(dt_fs) or float(dt_fs) <= 0:
            raise ValueError("dt_fs must be finite and positive.")
        if not np.isfinite(friction_per_fs) or float(friction_per_fs) < 0:
            raise ValueError("friction_per_fs must be finite and nonnegative.")
        self.model, self.device = model, device
        if device is not None and hasattr(model, "set_device"):
            model.set_device(device)
        self.atoms_list = [atoms.copy() for atoms in atoms_list]
        self._num_graphs = len(self.atoms_list)
        self._slices = [slice(start, stop) for start, stop in zip(
            np.cumsum([0, *[len(atoms) for atoms in atoms_list]])[:-1],
            np.cumsum([0, *[len(atoms) for atoms in atoms_list]])[1:])]
        self._frozen = np.zeros(self._num_graphs, dtype=bool)
        self._temperature = self._temperatures(temperature_K)
        self.nsteps = 0
        self._trajectories = []
        runner = self

        class ModelCalculator(Calculator):
            implemented_properties = ["energy", "forces"]

            def __init__(self, index):
                super().__init__()
                self.index = index

            def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                # ASE owns constraint projection; inference retains physical forces.
                runner.atoms_list[self.index].positions[:] = atoms.positions
                result = runner._evaluate()
                self.results = {"energy": float(np.asarray(result["energy"]).reshape(-1)[self.index]),
                                "forces": np.asarray(result["forces"])[runner._slices[self.index]].copy()}

        sequences = np.random.SeedSequence(int(random_seed)).spawn(self._num_graphs)
        self.dynamics = []
        for index, atoms in enumerate(self.atoms_list):
            atoms.calc = ModelCalculator(index)
            self.dynamics.append(Langevin(
                atoms, timestep=float(dt_fs) * units.fs,
                temperature_K=float(self._temperature[index]),
                friction=float(friction_per_fs) / units.fs,
                fixcm=False, rng=np.random.default_rng(sequences[index]),
            ))
        self.evaluate()

    def _temperatures(self, values):
        temperatures = np.asarray(values, dtype=float).reshape(-1)
        if temperatures.size == 1:
            temperatures = np.full(self._num_graphs, temperatures[0])
        if temperatures.size != self._num_graphs:
            raise ValueError("temperature_K must contain one value per system or a scalar.")
        if not np.isfinite(temperatures).all() or np.any(temperatures < 0):
            raise ValueError("temperature_K must contain finite nonnegative temperatures.")
        return temperatures.copy()

    def _index(self, graph_index):
        index = int(graph_index)
        if index < 0 or index >= self._num_graphs:
            raise IndexError("ASE graph index is outside the batch.")
        return index

    def _evaluate(self):
        self.last_result = evaluate_model(self.model, self.atoms_list, device=self.device)
        return self.last_result

    def evaluate(self):
        result = self._evaluate()
        for index, atoms in enumerate(self.atoms_list):
            atoms.calc.atoms = atoms.copy()
            atoms.calc.results = {"energy": float(np.asarray(result["energy"]).reshape(-1)[index]),
                                  "forces": np.asarray(result["forces"])[self._slices[index]].copy()}

    def run(self, steps):
        count = int(steps)
        if count < 0:
            raise ValueError("steps must be nonnegative.")
        for _ in range(count):
            for index, dynamics in enumerate(self.dynamics):
                if not self._frozen[index]:
                    dynamics.run(1)
            self.nsteps += 1
        if count:
            self.evaluate()

    def diagnostics_for_graph(self, graph_index):
        return self.model.diagnostics_for_graph(self._index(graph_index))

    def freeze_graph(self, graph_index):
        index = self._index(graph_index)
        self._frozen[index] = True
        self.atoms_list[index].set_velocities(np.zeros((len(self.atoms_list[index]), 3)))

    def set_temperature(self, temperature_K):
        self._temperature = self._temperatures(temperature_K)
        for index, dynamics in enumerate(self.dynamics):
            dynamics.set_temperature(temperature_K=float(self._temperature[index]))

    def set_cell(self, graph_index, cell, *, scale_atoms=True, evaluate=True):
        index = self._index(graph_index)
        if self._frozen[index]:
            return
        new_cell = np.asarray(cell, dtype=float)
        if new_cell.shape != (3, 3) or not np.isfinite(new_cell).all():
            raise ValueError("Cell must be a finite 3 by 3 matrix.")
        if np.linalg.det(new_cell) <= 0:
            raise ValueError("Cell must have positive volume.")
        if scale_atoms and self.atoms_list[index].get_volume() <= 0:
            raise ValueError("Cannot scale positions from a cell without positive volume.")
        self.atoms_list[index].set_cell(new_cell, scale_atoms=scale_atoms)
        if evaluate:
            self.evaluate()

    def set_lcm_mode(self, graph_index, *, pair, sigma, alpha_eV):
        self.model.set_lcm_mode(self._index(graph_index), pair=pair, sigma=sigma, alpha_eV=alpha_eV)

    def sync_graph_to_atoms(self, graph_index, atoms):
        source = self.atoms_list[self._index(graph_index)]
        atoms.set_cell(source.cell.array, scale_atoms=False)
        atoms.set_pbc(source.pbc)
        atoms.set_positions(source.positions)
        atoms.set_velocities(source.get_velocities())

    def configure_output(self, sampler_config, molecule_objects, master_directory=None):
        """Attach the existing opt-in ASE trajectory output without changing RNG."""
        interval = sampler_config.get("trajectory_interval")
        frequency = float(sampler_config.get("trajectory_frequency", 0.))
        directory = sampler_config.get("meta_dir", sampler_config.get("trajectory_dir"))
        if not interval or not directory or frequency <= 0:
            return
        from ase.io.trajectory import Trajectory
        root = Path(directory)
        if not root.is_absolute():
            root = Path(master_directory or ".") / root
        root.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(int(dict(sampler_config.get("dynamics") or {}).get("random_seed", 42)) + 2)
        for dynamics, atoms, molecule in zip(self.dynamics, self.atoms_list, molecule_objects):
            if rng.random() > frequency:
                continue
            trajectory = Trajectory(str(root / f"metadata-{molecule.get_moleculeid()}.traj"), mode="w", atoms=atoms)
            dynamics.attach(trajectory.write, interval=int(interval))
            self._trajectories.append(trajectory)

    def close(self):
        for trajectory in self._trajectories:
            trajectory.close()
        self._trajectories.clear()
