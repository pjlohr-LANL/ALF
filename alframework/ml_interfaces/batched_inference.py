"""Packed ensemble inference shared by ALF's optional dynamics backends.

Cells use ASE's row-vector convention, coordinates use Angstroms, energies
use eV, masses use amu, and velocities retain ASE's native units. Backend
adapters alone convert simulation-engine state and units.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import all_changes

from alframework.tools.sampling_validation import (
    normalize_energy_contributions, normalize_contributions,
    validate_required_physical_states, validate_model_count,
    normalize_gap_contributions, normalize_uncertainty_override,
    native_state_contributions, validate_native_results,
    validate_ase_results, validate_ase_force_shape,
    hippynn_members_first, hippynn_state_table, configured_alchemi_gap_rows,
    validate_ase_calculators, validate_hippynn_options,
)
from alframework.tools.force_screen import (
    SCHEMA, SOURCE, physical_force_diagnostics,
    lcm_force_diagnostics, propagation_force_diagnostics,
)


@dataclass
class PackedBatch:
    """Backend-neutral, ordered systems stored without padded atoms."""

    positions: Any
    atomic_numbers: Any
    atomic_masses: Any
    velocities: Any
    cell: Any
    pbc: Any
    batch_ptr: Any

    def __post_init__(self):
        total = int(self.positions.shape[0])
        ptr = self.batch_ptr
        if ptr.ndim != 1 or ptr.numel() < 2 or int(ptr[0]) != 0 or int(ptr[-1]) != total:
            raise ValueError("Packed batch offsets must span every atom.")
        if bool(torch.any(ptr[1:] <= ptr[:-1])):
            raise ValueError("Packed systems must contain at least one atom.")
        if self.positions.shape != (total, 3) or self.velocities.shape != (total, 3):
            raise ValueError("Packed positions and velocities must have shape [atoms, 3].")
        if self.atomic_numbers.shape != (total,) or self.atomic_masses.shape != (total,):
            raise ValueError("Packed species and masses must have one value per atom.")
        if self.cell.shape != (self.num_graphs, 3, 3) or self.pbc.shape != (self.num_graphs, 3):
            raise ValueError("Packed cells/PBC must have one entry per system.")
        if bool(torch.any(self.atomic_numbers <= 0)) or bool(torch.any(self.atomic_masses <= 0)):
            raise ValueError("Packed atoms require positive species and masses.")
        for value in (self.positions, self.velocities, self.atomic_masses, self.cell):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Packed state must contain finite values.")

    @classmethod
    def from_atoms(cls, atoms_list, *, device=None, dtype=None):
        atoms_list = list(atoms_list)
        if not atoms_list or any(len(atoms) < 1 for atoms in atoms_list):
            raise ValueError("A packed batch requires nonempty systems.")
        dtype = torch.float32 if dtype is None else dtype
        def tensor(value, *, kind=dtype):
            return torch.as_tensor(np.asarray(value), dtype=kind, device=device)
        counts = torch.tensor([len(atoms) for atoms in atoms_list], dtype=torch.long, device=device)
        return cls(
            positions=tensor(np.concatenate([atoms.get_positions() for atoms in atoms_list])),
            atomic_numbers=tensor(np.concatenate([atoms.numbers for atoms in atoms_list]), kind=torch.long),
            atomic_masses=tensor(np.concatenate([atoms.get_masses() for atoms in atoms_list])),
            velocities=tensor(np.concatenate([atoms.get_velocities() for atoms in atoms_list])),
            cell=tensor(np.stack([atoms.cell.array for atoms in atoms_list])),
            pbc=tensor(np.stack([atoms.pbc for atoms in atoms_list]), kind=torch.bool),
            batch_ptr=torch.cat([counts.new_zeros(1), counts.cumsum(0)]),
        )

    @property
    def num_graphs(self):
        return int(self.batch_ptr.numel()) - 1

    @property
    def num_nodes_per_graph(self):
        return self.batch_ptr[1:] - self.batch_ptr[:-1]

    @property
    def num_nodes(self):
        return int(self.positions.shape[0])

    @property
    def device(self):
        return self.positions.device

    @property
    def system_idx(self):
        return torch.repeat_interleave(
            torch.arange(self.num_graphs, device=self.device), self.num_nodes_per_graph
        )

    @property
    def batch(self):
        return self.system_idx


def batch_offsets(batch):
    """Offsets for both packed state and the legacy native Batch protocol."""
    ptr = getattr(batch, "batch_ptr", None)
    if ptr is None:
        counts = batch.num_nodes_per_graph.to(dtype=torch.long, device=batch.positions.device)
        ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    return ptr


def batch_slices(batch):
    ptr = batch_offsets(batch).detach().cpu().tolist()
    return [slice(int(start), int(end)) for start, end in zip(ptr[:-1], ptr[1:])]


def _segment_sum(values, batch):
    result = values.new_zeros((int(batch.num_graphs), *values.shape[1:]))
    index = torch.repeat_interleave(
        torch.arange(int(batch.num_graphs), device=values.device), batch.num_nodes_per_graph
    )
    return result.index_add_(0, index, values)


def _segment_max(values, batch):
    result = values.new_full((int(batch.num_graphs),), -float("inf"))
    index = torch.repeat_interleave(
        torch.arange(int(batch.num_graphs), device=values.device), batch.num_nodes_per_graph
    )
    return result.scatter_reduce_(0, index, values, reduce="amax", include_self=True)

class BatchedEnsembleCalculator(torch.nn.Module):
    """Base calculator that applies ALF uncertainty semantics to an ensemble.

    Subclasses implement :meth:`ensemble_forward` and return raw selected-
    state energy and force contributions. Keeping this reduction here makes
    model adapters responsible only for inference and property mapping.
    """

    def __init__(
        self,
        *,
        selected_state_value: int = 0,
        physical_force_states: list[int] | None = None,
        gap_rows: list[dict[str, Any]] | None = None,
        gap_seeking_settings: dict[str, Any] | None = None,
        well_params: dict[str, Any] | None = None,
        device: Any = None,
        model_config_template: Any = None,
        calculator_interface: str = "native",
        calculator_loader: str | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.selected_state_value = int(selected_state_value)
        self.physical_force_states = sorted(set(physical_force_states or [self.selected_state_value]))
        self.gap_rows = [dict(row) for row in (gap_rows or [])]
        self.gap_seeking_settings = dict(
            gap_seeking_settings or {"enabled": False, "rows": []}
        )
        self._lcm_controls: dict[int, dict[str, Any]] = {}
        self.well_params = None if well_params is None else dict(well_params)
        self._device = torch.device(device) if device is not None else None
        self.calculator_interface = str(calculator_interface)
        self.calculator_loader = str(
            calculator_loader or f"{type(self).__module__}.{type(self).__name__}"
        )
        self.supports_pbc = bool(getattr(model_config_template, "supports_pbc", False))
        self.last_diagnostics: dict[str, Any] = {}
        self.udd_bias_weight = None
        self.udd_bias_eps = 1.e-12

    def configure_sampling_method(self, method, options=None):
        from alframework.samplers.force_controls import configure_controls
        return configure_controls(self, method, options)

    def require_physical_states(self, states):
        self.physical_force_states = sorted(set(int(state) for state in states))

    @property
    def embedding_shapes(self) -> dict[str, Any]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        del kwargs
        return data

    def direct_derivative_keys(self) -> set[str]:
        return {"forces"}

    def set_device(self, device: Any) -> None:
        self._device = torch.device(device)
        self.to(self._device)

    def set_lcm_mode(self, graph_index, *, pair, sigma, alpha_eV):
        from alframework.samplers.force_controls import set_lcm_mode
        return set_lcm_mode(self, graph_index, pair=pair, sigma=sigma, alpha_eV=alpha_eV)

    @staticmethod
    def _batch_shape(batch):
        return int(batch.num_graphs), int(batch.num_nodes_per_graph.max().item())

    def _well_energy_forces(self, batch, positions):
        from alframework.samplers.force_controls import torch_well_energy_forces
        return torch_well_energy_forces(self, batch, positions)

    def ensemble_forward(self, batch) -> dict[str, Any]:
        """Return raw ensemble contributions for the current batch."""

        raise NotImplementedError

    @staticmethod
    def _normalize_energy_contributions(batch, energy_values):
        return normalize_energy_contributions(batch, energy_values)

    @staticmethod
    def _normalize_contributions(batch, energy_values, force_values):
        return normalize_contributions(batch, energy_values, force_values)

    @staticmethod
    def _reduce_contributions(batch, energy, forces):
        from alframework.samplers.force_controls import torch_reduce_contributions
        return torch_reduce_contributions(batch, energy, forces)

    @staticmethod
    def _lcm_energy_forces(lower_energy, upper_energy, lower_forces, upper_forces, *, sigma, alpha_eV):
        from alframework.samplers.force_controls import torch_lcm_energy_forces
        return torch_lcm_energy_forces(lower_energy, upper_energy, lower_forces, upper_forces,
                                       sigma=sigma, alpha_eV=alpha_eV)

    def forward(self, batch):
        # Compatibility facade: physical inference is ensemble_forward().
        from alframework.samplers.force_controls import torch_forward
        return torch_forward(self, batch)

    def diagnostics_for_graph(self, graph_index):
        index = int(graph_index)
        if index < 0 or index >= len(self._graph_slices):
            raise IndexError(f"Diagnostic graph index {index} is outside the evaluated batch.")
        values = {"force_screen_schema": SCHEMA, "force_screen_source": SOURCE}
        for key, tensor in self.last_diagnostics.items():
            value = tensor[self._graph_slices[index]] if key in self._diagnostic_force_keys else tensor[index]
            value = np.asarray(value.detach().cpu().numpy())
            values[key] = float(value.reshape(-1)[0]) if value.size == 1 else value.copy()
        return values


class NativeEnsembleModel(BatchedEnsembleCalculator):
    """Combine compatible ALCHEMI-native models into an ALF ensemble."""

    def __init__(
        self,
        models,
        *,
        energy_key: str = "energy",
        force_key: str = "forces",
        selected_state_value: int = 0,
        physical_force_states: list[int] | None = None,
        gap_rows: list[dict[str, Any]] | None = None,
        gap_seeking_settings: dict[str, Any] | None = None,
        well_params: dict[str, Any] | None = None,
        device: Any = None,
        calculator_loader: str | None = None,
    ) -> None:
        model_list = list(models)
        if not model_list:
            raise ValueError("A native ensemble requires at least one model.")
        if not all(isinstance(model, torch.nn.Module) for model in model_list):
            raise TypeError("Every native ensemble member must be a torch.nn.Module.")
        signatures = [model.input_data() for model in model_list if hasattr(model, "input_data")]
        if signatures and (len(signatures) != len(model_list) or any(value != signatures[0] for value in signatures[1:])):
            raise ValueError("Native ensemble members require incompatible batch inputs.")
        super().__init__(
            selected_state_value=selected_state_value,
            physical_force_states=physical_force_states,
            gap_rows=gap_rows,
            gap_seeking_settings=gap_seeking_settings,
            well_params=well_params,
            device=device,
            model_config_template=getattr(model_list[0], "model_config", None),
            calculator_interface="native",
            calculator_loader=calculator_loader,
        )
        self.models = torch.nn.ModuleList(model_list)
        self.energy_key = str(energy_key)
        self.force_key = str(force_key)
        self.set_device(device or "cpu")

    def set_device(self, device: Any) -> None:
        self._device = torch.device(device)
        for model in self.models:
            model.to(self._device)
            if hasattr(model, "set_device"):
                model.set_device(self._device)

    def ensemble_forward(self, batch) -> dict[str, Any]:
        energies = []
        forces = []
        required_lcm_states = set(self.physical_force_states) - {self.selected_state_value} | {
            int(state)
            for row in self.gap_seeking_settings.get("rows", [])
            for state in (row["lower_state"], row["upper_state"])
        }
        state_members = {
            state: {"energy": [], "forces": []}
            for state in sorted(required_lcm_states)
        }
        for model in self.models:
            output = model(batch)
            validate_native_results(output, self.energy_key, self.force_key)
            energies.append(output[self.energy_key])
            forces.append(output[self.force_key])
            if required_lcm_states:
                raw_states = native_state_contributions(
                    output, required_lcm_states, self.selected_state_value,
                    self.energy_key, self.force_key)
                for state, (energy_value, force_value) in raw_states.items():
                    state_members[state]["energy"].append(energy_value)
                    state_members[state]["forces"].append(force_value)
        prediction = {
            "energy_contributions": torch.stack(energies, dim=0),
            "force_contributions": torch.stack(forces, dim=0),
        }
        if state_members:
            prediction["state_contributions"] = {
                state: {
                    "energy_contributions": torch.stack(
                        values["energy"], dim=0
                    ),
                    "force_contributions": torch.stack(
                        values["forces"], dim=0
                    ),
                }
                for state, values in state_members.items()
            }
        return prediction


class ASEEnsembleModel(BatchedEnsembleCalculator):
    """Compatibility wrapper for existing ALF ASE calculator loaders."""

    def __init__(
        self,
        calculators,
        *,
        model_mode: str,
        selected_state_value: int | None,
        physical_force_states: list[int] | None = None,
        gap_rows: list[dict[str, Any]] | None = None,
        gap_seeking_settings: dict[str, Any] | None = None,
        well_params: dict[str, Any] | None = None,
        device: Any = None,
        calculator_loader: str | None = None,
        uncertainty_mode: str | None = None,
    ) -> None:
        if isinstance(calculators, (list, tuple)):
            calculator_list = list(calculators)
        else:
            calculator_list = [calculators]
        validate_ase_calculators(calculator_list)
        selected = 0 if selected_state_value is None else int(selected_state_value)
        super().__init__(
            selected_state_value=selected,
            physical_force_states=physical_force_states,
            gap_rows=gap_rows,
            gap_seeking_settings=gap_seeking_settings,
            well_params=well_params,
            device=device,
            calculator_interface="ase_fallback",
            calculator_loader=calculator_loader,
        )
        mode = str(model_mode).strip().lower()
        self.energy_key = "energy" if mode == "ground_state" else f"sE{selected}"
        self.force_key = "forces" if mode == "ground_state" else f"F{selected}"
        gap_states = set(self.physical_force_states) if mode == "excited_state" else set()
        gap_states |= {
            int(state)
            for row in self.gap_rows
            for state in (row["lower_state"], row["upper_state"])
        }
        self.gap_energy_keys = {
            state: f"sE{state}" for state in sorted(gap_states)
        }
        lcm_states = set(self.physical_force_states) if mode == "excited_state" else set()
        lcm_states |= {
            int(state)
            for row in self.gap_seeking_settings.get("rows", [])
            for state in (row["lower_state"], row["upper_state"])
        }
        self.gap_force_keys = {
            state: f"F{state}" for state in sorted(lcm_states)
        }
        self.calculators = calculator_list
        self.supports_pbc = True
        self.uncertainty_mode = (
            None if uncertainty_mode is None else str(uncertainty_mode).lower()
        )

    def require_physical_states(self, states):
        super().require_physical_states(states)
        if self.energy_key != "energy":
            self.gap_energy_keys.update({state: f"sE{state}" for state in states})
            self.gap_force_keys.update({state: f"F{state}" for state in states})

    def set_device(self, device: Any) -> None:
        self._device = torch.device(device)

    def _evaluate_one(self, calculator, atoms):
        requested = list(
            dict.fromkeys(
                [
                    self.energy_key,
                    self.force_key,
                    *self.gap_energy_keys.values(),
                    *self.gap_force_keys.values(),
                ]
            )
        )
        standard_uncertainty = {
            "energy_stdev",
            "forces_stdev_mean",
            "forces_stdev_max",
        }
        if len(self.calculators) == 1 and standard_uncertainty.issubset(
            set(getattr(calculator, "implemented_properties", []))
        ):
            requested.extend(sorted(standard_uncertainty))
        calculator.calculate(
            atoms,
            properties=requested,
            system_changes=all_changes,
        )
        required = [
            self.energy_key,
            self.force_key,
            *self.gap_energy_keys.values(),
            *self.gap_force_keys.values(),
        ]
        validate_ase_results(calculator.results, required)
        energy = float(np.asarray(calculator.results[self.energy_key]).sum())
        forces = np.asarray(calculator.results[self.force_key], dtype=float)
        state_energies = {
            state: float(np.asarray(calculator.results[key]).sum())
            for state, key in self.gap_energy_keys.items()
        }
        state_forces = {
            state: np.asarray(calculator.results[key], dtype=float)
            for state, key in self.gap_force_keys.items()
        }
        uncertainty = None
        if len(self.calculators) == 1:
            if standard_uncertainty.issubset(calculator.results):
                uncertainty = {
                    "Es": float(calculator.results["energy_stdev"]),
                    "Fs": float(calculator.results["forces_stdev_mean"]),
                    "Fsmax": float(calculator.results["forces_stdev_max"]),
                }
            elif self.uncertainty_mode == "neurochem":
                force_mean, force_max = calculator.get_Fstddev()
                uncertainty = {
                    "Es": float(calculator.Estddev) * 1000.0,
                    "Fs": float(force_mean),
                    "Fsmax": float(force_max),
                }
        return energy, forces, uncertainty, state_energies, state_forces

    def ensemble_forward(self, batch):
        from alframework.ml_interfaces.sampling_inference import NumpyASEEnsembleModel
        positions = batch.positions.detach().cpu().numpy()
        numbers = batch.atomic_numbers.detach().cpu().numpy()
        masses = batch.atomic_masses.detach().cpu().numpy()
        cells, pbc = getattr(batch, "cell", None), getattr(batch, "pbc", None)
        atoms_list = []
        for index, selection in enumerate(batch_slices(batch)):
            atoms_list.append(Atoms(numbers=numbers[selection], positions=positions[selection],
                masses=masses[selection], cell=None if cells is None else cells[index].detach().cpu().numpy(),
                pbc=False if pbc is None else pbc[index].detach().cpu().numpy()))
        raw = NumpyASEEnsembleModel(self.calculators,
            model_mode="ground_state" if self.energy_key == "energy" else "excited_state",
            selected_state_value=self.selected_state_value, physical_force_states=self.physical_force_states,
            gap_rows=self.gap_rows, gap_seeking_settings=self.gap_seeking_settings,
            uncertainty_mode=self.uncertainty_mode).raw_ensemble_forward(atoms_list)
        def tensor(value):
            if isinstance(value, dict):
                return {key: tensor(item) for key, item in value.items()}
            if isinstance(value, np.ndarray):
                return torch.as_tensor(value, dtype=batch.positions.dtype, device=batch.positions.device)
            return value
        return tensor(raw)


class HippynnEnsembleModel(BatchedEnsembleCalculator):
    """Adapt raw HIPPYNN ensemble outputs to the shared calculator base."""

    def __init__(self, *, ensemble_graph, state_nodes, selected_state_value,
                 physical_force_states=None, gap_rows=None, gap_seeking_settings=None,
                 species_key, coordinates_key, cell_key=None, well_params=None,
                 device=None, calculator_loader=None):
        super().__init__(selected_state_value=selected_state_value,
                         physical_force_states=physical_force_states, gap_rows=gap_rows,
                         gap_seeking_settings=gap_seeking_settings, well_params=well_params,
                         device=device, calculator_interface="native", calculator_loader=calculator_loader)
        self.ensemble_graph = ensemble_graph
        self.state_nodes = list(state_nodes)
        self.require_physical_states([int(row["state"]) for row in self.state_nodes])
        self.species_key, self.coordinates_key = str(species_key), str(coordinates_key)
        self.cell_key = None if cell_key in (None, "None") else str(cell_key)
        if self.cell_key is None:
            candidates = [node for node in ensemble_graph.input_nodes
                          if type(node).__name__ == "CellNode" or str(getattr(node, "db_name", "")) == "cell"]
            if len(candidates) == 1:
                self.cell_key = str(getattr(candidates[0], "db_name", None) or candidates[0].name)
        self.supports_pbc = self.cell_key is not None
        self.requires_pbc = self.supports_pbc
        self._supported_species = None
        if hasattr(ensemble_graph, "modules"):
            encoders = [set(int(z) for z in module.species_set.tolist() if int(z) > 0)
                        for module in ensemble_graph.modules() if hasattr(module, "species_set")]
            if encoders:
                self._supported_species = set.intersection(*encoders)
        self._make_predictor()

    def _resolve_input(self, db_name: str):
        for node in self.ensemble_graph.input_nodes:
            if str(getattr(node, "db_name", "")) == db_name or str(
                getattr(node, "name", "")
            ) == db_name:
                return node
        raise RuntimeError(f"Could not find HIPPYNN ensemble input {db_name!r}.")

    def _make_predictor(self):
        from hippynn.graphs import Predictor
        self._species_node = self._resolve_input(self.species_key)
        self._coordinates_node = self._resolve_input(self.coordinates_key)
        inputs = [self._species_node, self._coordinates_node]
        if self.cell_key is not None:
            inputs.append(self._resolve_input(self.cell_key))
        if set(inputs) != set(self.ensemble_graph.input_nodes):
            raise ValueError("HIPPYNN checkpoint inputs must be species, coordinates, and optionally a cell.")
        outputs = [node for row in self.state_nodes for node in (row["energy_all_node"], row["force_all_node"])]
        self._predictor = Predictor(inputs, outputs, return_device=self._device,
                                    model_device=self._device, requires_grad=False)

    def set_device(self, device: Any) -> None:
        self._device = torch.device(device)
        self.ensemble_graph.to(self._device)
        self._predictor.model_device = self._device
        self._predictor.return_device = self._device

    @staticmethod
    def _members_first(values, model_count: int, name: str):
        return hippynn_members_first(values, model_count, name)

    def ensemble_forward(self, batch):
        positions, species = batch.positions.detach(), batch.atomic_numbers.detach()
        num_graphs, max_atoms = self._batch_shape(batch)
        pbc = getattr(batch, "pbc", None)
        periodic = bool(torch.any(pbc)) if pbc is not None else False
        if periodic and not self.supports_pbc:
            raise ValueError("A molecular HIPPYNN checkpoint cannot evaluate periodic systems; load a cell-aware ground-state checkpoint.")
        if self.requires_pbc and (pbc is None or not bool(torch.all(pbc))):
            raise ValueError("A periodic HIPPYNN checkpoint requires fully periodic systems.")
        if self._supported_species is not None:
            missing = sorted(set(int(z) for z in species.detach().cpu().unique().tolist()) - self._supported_species)
            if missing:
                raise ValueError(f"HIPPYNN checkpoint does not support atomic numbers {missing}.")
        counts = batch.num_nodes_per_graph
        mask = torch.arange(max_atoms, device=positions.device).unsqueeze(0) < counts.unsqueeze(1)
        padded_species = species.new_zeros((num_graphs, max_atoms), dtype=torch.long)
        padded_positions = positions.new_zeros((num_graphs, max_atoms, 3))
        padded_species[mask] = species.to(torch.long)
        padded_positions[mask] = positions
        inputs = {self.species_key: padded_species, self.coordinates_key: padded_positions}
        if self.cell_key is not None:
            cells = getattr(batch, "cell", None)
            if cells is None or cells.shape != (num_graphs, 3, 3) or not bool(torch.all(torch.linalg.det(cells) > 0)):
                raise ValueError("Periodic HIPPYNN inference requires a valid cell for every system.")
            inputs[self.cell_key] = cells.detach()
        predictions = self._predictor(**inputs)
        states = {}
        for row in self.state_nodes:
            energy_count, force_count = int(row["energy_model_count"]), int(row["force_model_count"])
            validate_model_count(energy_count, force_count, "HIPPYNN energy and force ensembles must contain the same models.")
            energy_values = self._members_first(predictions[row["energy_all_node"]], energy_count, "energy")
            energy_values = energy_values.reshape(energy_count, num_graphs, -1)
            if energy_values.shape[-1] == max_atoms and max_atoms > 1:
                energy_values = energy_values * mask.unsqueeze(0)
            energy = energy_values.sum(dim=2)
            force_values = self._members_first(predictions[row["force_all_node"]], force_count, "force")
            expected = force_count * num_graphs * max_atoms * 3
            if force_values.numel() != expected:
                raise ValueError("HIPPYNN force outputs must preserve the padded system and atom dimensions.")
            forces = force_values.reshape(force_count, num_graphs, max_atoms, 3)[:, mask, :]
            states[int(row["state"])] = {"energy_contributions": energy, "force_contributions": forces}
        if self.selected_state_value not in states:
            raise RuntimeError(f"Selected state {self.selected_state_value} was not loaded from the HIPPYNN ensemble.")
        return {**states[self.selected_state_value], "state_contributions": states}


def load_hippynn_batched_model(
    ensemble_directory, *, model_mode, selected_state_value, ML_config,
    properties_list, sampler_config, device, **options,
):
    """Load native HIPPYNN ensembles without importing a dynamics backend."""
    import hippynn

    validate_hippynn_options(options)
    if getattr(torch.version, "hip", None) is not None:
        # CUDA-specific HIPPYNN kernels can be importable in a ROCm process.
        # Explicit selection keeps this path on portable GPU tensor operations.
        hippynn.custom_kernels.set_custom_kernels("pytorch")
    graph, (_, output_info) = hippynn.graphs.make_ensemble(
        os.path.join(str(ensemble_directory), "model-*"), quiet=True
    )
    graph.to(device)
    table, selected = hippynn_state_table(model_mode, selected_state_value, ML_config, properties_list)
    gap_rows, gap_seeking = configured_alchemi_gap_rows(sampler_config, properties_list, selected)
    cell_key = ML_config.get("cell_key")
    if str(model_mode).strip().lower() == "excited_state" and cell_key not in (None, "None"):
        raise ValueError("Excited-state HIPPYNN inference currently supports molecular systems only.")
    state_nodes = []
    for row in table:
        energy = graph.node_from_name(f"ensemble_{row['energy_db_name']}")
        forces = graph.node_from_name(f"ensemble_{row['force_db_name']}")
        state_nodes.append({**row, "energy_all_node": energy.all, "force_all_node": forces.all,
                            "energy_model_count": int(output_info[row["energy_db_name"]]),
                            "force_model_count": int(output_info[row["force_db_name"]])})
    return HippynnEnsembleModel(
        ensemble_graph=graph, state_nodes=state_nodes, selected_state_value=selected,
        gap_rows=gap_rows, gap_seeking_settings=gap_seeking,
        species_key=str(ML_config.get("species_key", "species")),
        coordinates_key=str(ML_config.get("coordinates_key", "coordinates")),
        cell_key=cell_key,
        well_params=dict(sampler_config.get("MLMD_calculator_options") or {}).get("well_params"),
        device=device,
        calculator_loader="alframework.ml_interfaces.batched_inference.load_hippynn_batched_model",
    )
