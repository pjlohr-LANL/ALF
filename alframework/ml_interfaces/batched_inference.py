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

    def set_lcm_mode(
        self,
        graph_index: int,
        *,
        pair: list[int] | tuple[int, int],
        sigma: float,
        alpha_eV: float,
    ) -> None:
        """Permanently switch one batch graph to its configured LCM pair."""

        if not bool(self.gap_seeking_settings.get("enabled", False)):
            raise RuntimeError(
                "Cannot enter LCM mode when gap_seeking is disabled."
            )
        index = int(graph_index)
        if index < 0:
            raise ValueError("LCM graph_index must be nonnegative.")
        normalized_pair = (int(pair[0]), int(pair[1]))
        allowed_pairs = {
            (int(row["lower_state"]), int(row["upper_state"]))
            for row in self.gap_seeking_settings["rows"]
        }
        if normalized_pair not in allowed_pairs:
            raise ValueError(
                f"LCM pair {normalized_pair} is not eligible for selected "
                f"state {self.selected_state_value}."
            )
        control = {
            "pair": normalized_pair,
            "sigma": float(sigma),
            "alpha_eV": float(alpha_eV),
        }
        existing = self._lcm_controls.get(index)
        if existing is not None and existing != control:
            raise RuntimeError(
                f"Graph {index} already entered stay-fixed LCM mode with "
                f"pair {existing['pair']}."
            )
        self._lcm_controls[index] = control

    @staticmethod
    def _batch_shape(batch):
        return int(batch.num_graphs), int(batch.num_nodes_per_graph.max().item())

    def _well_energy_forces(self, batch, positions):
        if self.well_params is None:
            return None, None
        params = dict(self.well_params)
        r_start, force_constant = float(params["r_start"]), float(params["force"])
        origin = torch.as_tensor(params.get("origin", [0., 0., 0.]), dtype=positions.dtype, device=positions.device)
        relative = positions - origin.reshape(1, 3)
        radius = torch.linalg.vector_norm(relative, dim=-1)
        depth = torch.clamp(radius - r_start, min=0.)
        weights = batch.atomic_masses if bool(params.get("mass_weighted", True)) else torch.ones_like(radius)
        energy = _segment_sum(weights * depth * force_constant, batch).reshape(-1, 1)
        unit_vectors = relative / torch.clamp(radius, min=1.e-12).unsqueeze(-1)
        active = (radius > r_start).to(positions.dtype).unsqueeze(-1)
        forces = -unit_vectors * active * weights.unsqueeze(-1) * force_constant
        return energy, forces

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
        energy_mean, force_mean = energy.mean(dim=0), forces.mean(dim=0)
        energy_std = energy.std(dim=0, correction=0)
        force_std = forces.std(dim=0, correction=0)
        denominator = batch.num_nodes_per_graph.to(forces.dtype) * 3
        return {
            "energy": energy_mean, "forces": force_mean,
            "energy_std": energy_std, "force_std": force_std,
            "Fs": _segment_sum(force_std.abs().sum(-1), batch) / denominator,
            "Fsmax": _segment_max(force_std.abs().amax(-1), batch),
            "Fsrms": torch.sqrt(_segment_sum(force_std.square().sum(-1), batch) / denominator),
        }

    @staticmethod
    def _lcm_energy_forces(
        lower_energy,
        upper_energy,
        lower_forces,
        upper_forces,
        *,
        sigma: float,
        alpha_eV: float,
    ):
        """Return the fork-compatible smooth LCM mean and analytic force."""

        gap = upper_energy - lower_energy
        smooth_abs_gap = torch.sqrt(gap * gap + 1.0e-12)
        denominator = smooth_abs_gap + float(alpha_eV)
        correction = float(sigma) * gap * gap / denominator
        gap_derivative = float(sigma) * (
            (2.0 * gap * denominator)
            - (gap * gap * gap / smooth_abs_gap)
        ) / (denominator * denominator)
        energy = 0.5 * (lower_energy + upper_energy) + correction
        forces = (
            0.5 * (lower_forces + upper_forces)
            + gap_derivative * (upper_forces - lower_forces)
        )
        return energy, forces

    def forward(self, batch):
        prediction = dict(self.ensemble_forward(batch))
        energy, forces = self._normalize_contributions(batch, prediction["energy_contributions"], prediction["force_contributions"])
        selected = self._reduce_contributions(batch, energy, forces)
        self._graph_slices = batch_slices(batch)
        self._diagnostic_force_keys = set()
        diagnostics = {}
        state_contributions = {int(state): values for state, values in dict(prediction.get("state_contributions") or {}).items()}
        state_contributions.setdefault(self.selected_state_value, {"energy_contributions": energy, "force_contributions": forces})
        validate_required_physical_states(state_contributions, self.physical_force_states)
        state_results = {}
        for state, values in state_contributions.items():
            state_energy, state_forces = self._normalize_contributions(batch, values["energy_contributions"], values["force_contributions"])
            validate_model_count(state_energy.shape[0], energy.shape[0], "Physical states must use the same ordered model count.")
            result = self._reduce_contributions(batch, state_energy, state_forces)
            state_results[state] = result
            diagnostics.update({f"sE{state}": result["energy"], f"sE{state}_stdev": result["energy_std"],
                                f"F{state}": result["forces"], f"F{state}_stdev": result["force_std"]})
            self._diagnostic_force_keys.update((f"F{state}", f"F{state}_stdev"))
        state_energies = {int(state): value for state, value in dict(prediction.get("state_energy_contributions") or {}).items()}
        for state, values in state_contributions.items():
            state_energies.setdefault(state, values["energy_contributions"])
        state_energies.setdefault(self.selected_state_value, energy)
        model_count = int(energy.shape[0])
        for row in self.gap_rows:
            lower, upper = normalize_gap_contributions(batch, state_energies, row, model_count)
            members = upper - lower
            key = str(row["gap_key"])
            diagnostics.update({key: members.mean(0), f"{key}_stdev": members.std(0, correction=0),
                                f"{key}_model_count": torch.full((int(batch.num_graphs),), model_count, dtype=torch.long, device=batch.positions.device)})
        override = prediction.get("uncertainty")
        if override is None:
            diagnostics.update({"Es": selected["energy_std"], "Fs": selected["Fs"], "Fsmax": selected["Fsmax"], "Fsrms": selected["Fsrms"]})
        else:
            diagnostics.update(normalize_uncertainty_override(batch, override))
        ptr = batch_offsets(batch)
        diagnostics.update(physical_force_diagnostics(state_results, int(batch.num_graphs), batch_ptr=ptr))
        selected_energy = selected["energy"].reshape(-1, 1).clone()
        selected_forces = selected["forces"].clone()
        physical_forces = selected_forces.clone()
        correction = torch.zeros_like(selected_forces)
        for graph_index, control in sorted(self._lcm_controls.items()):
            if graph_index >= int(batch.num_graphs):
                raise IndexError(f"LCM graph index {graph_index} is outside batch size {batch.num_graphs}.")
            lower, upper = control["pair"]
            missing = [state for state in (lower, upper) if state not in state_results]
            if missing:
                raise KeyError(f"LCM pair {(lower, upper)} requires raw energy and force contributions for states {missing}.")
            atom_slice = self._graph_slices[graph_index]
            lo, hi = state_results[lower], state_results[upper]
            lcm_energy, lcm_forces = self._lcm_energy_forces(
                lo["energy"][graph_index], hi["energy"][graph_index],
                lo["forces"][atom_slice], hi["forces"][atom_slice],
                sigma=float(control["sigma"]), alpha_eV=float(control["alpha_eV"]))
            selected_energy[graph_index, 0] = lcm_energy
            selected_forces[atom_slice] = lcm_forces
            correction[atom_slice] = lcm_forces - .5 * (lo["forces"][atom_slice] + hi["forces"][atom_slice])
        diagnostics.update(lcm_force_diagnostics(selected_forces, physical_forces, correction, batch_ptr=ptr))
        well_energy, well_forces = self._well_energy_forces(batch, batch.positions)
        if well_energy is not None:
            selected_energy += well_energy
            selected_forces += well_forces
        diagnostics.update(propagation_force_diagnostics(selected_forces, well_forces, batch_ptr=ptr))
        self.last_diagnostics = {key: value.detach() for key, value in diagnostics.items()}
        return {"energy": selected_energy, "forces": selected_forces}

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
        positions = batch.positions.detach().cpu().numpy()
        numbers = batch.atomic_numbers.detach().cpu().numpy()
        masses = batch.atomic_masses.detach().cpu().numpy()
        cells = getattr(batch, "cell", None)
        pbc = getattr(batch, "pbc", None)
        energy_members = [[] for _ in self.calculators]
        force_members = [[] for _ in self.calculators]
        state_energy_members = {state: [[] for _ in self.calculators] for state in self.gap_energy_keys}
        state_force_members = {state: [[] for _ in self.calculators] for state in self.gap_force_keys}
        uncertainty_rows = []
        for graph_index, atom_slice in enumerate(batch_slices(batch)):
            cell_value = None if cells is None else cells[graph_index].detach().cpu().numpy()
            pbc_value = False if pbc is None else pbc[graph_index].detach().cpu().numpy()
            atoms = Atoms(numbers=numbers[atom_slice], positions=positions[atom_slice],
                          masses=masses[atom_slice], cell=cell_value, pbc=pbc_value)
            graph_uncertainty = None
            for model_index, calculator in enumerate(self.calculators):
                energy, forces, uncertainty, state_energies, state_forces = self._evaluate_one(calculator, atoms)
                validate_ase_force_shape(forces, len(atoms), self.force_key)
                energy_members[model_index].append(energy)
                force_members[model_index].append(forces)
                for state, value in state_energies.items():
                    state_energy_members[state][model_index].append(value)
                for state, value in state_forces.items():
                    validate_ase_force_shape(value, len(atoms), f"F{state}")
                    state_force_members[state][model_index].append(value)
                if uncertainty is not None:
                    graph_uncertainty = uncertainty
            uncertainty_rows.append(graph_uncertainty)
        def tensor(value):
            return torch.as_tensor(value, dtype=batch.positions.dtype, device=batch.positions.device)
        def packed_forces(members):
            return tensor(np.stack([np.concatenate(systems, axis=0) for systems in members]))
        prediction = {"energy_contributions": tensor(energy_members), "force_contributions": packed_forces(force_members)}
        if state_energy_members:
            prediction["state_energy_contributions"] = {state: tensor(values) for state, values in state_energy_members.items()}
        if state_force_members:
            prediction["state_contributions"] = {
                state: {"energy_contributions": tensor(state_energy_members[state]), "force_contributions": packed_forces(values)}
                for state, values in state_force_members.items()}
        if all(value is not None for value in uncertainty_rows):
            prediction["uncertainty"] = {key: [row[key] for row in uncertainty_rows] for key in ("Es", "Fs", "Fsmax")}
        return prediction


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
