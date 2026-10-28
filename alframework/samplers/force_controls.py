"""Sampling force controls over physical ensemble predictions.

Numerical engines call these controls after model inference. Physical members
remain unchanged for acquisition and screening. The module itself imports no
optional tensor engine; native operations stay on their original device.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import numpy as np
from alframework.tools.force_screen import (
    SCHEMA, SOURCE, physical_force_diagnostics,
    lcm_force_diagnostics, propagation_force_diagnostics,
)
from alframework.tools.sampling_validation import (
    normalize_energy_contributions, normalize_gap_contributions,
    normalize_uncertainty_override, validate_required_physical_states,
    validate_model_count,
)


def _norm_max(values):
    return float(np.linalg.norm(np.asarray(values), axis=-1).max())


@dataclass
class NumpyForceState:
    """NumPy counterpart of the packed state passed to force-transform plugins."""
    positions: object
    atomic_numbers: object
    atomic_masses: object
    velocities: object
    cell: object
    pbc: object
    batch_ptr: object

    @classmethod
    def from_atoms(cls, atoms_list):
        return cls(np.concatenate([a.positions for a in atoms_list]),
                   np.concatenate([a.numbers for a in atoms_list]),
                   np.concatenate([a.get_masses() for a in atoms_list]),
                   np.concatenate([a.get_velocities() for a in atoms_list]),
                   np.stack([a.cell.array for a in atoms_list]), np.stack([a.pbc for a in atoms_list]),
                   np.cumsum([0, *[len(a) for a in atoms_list]]))

    @property
    def num_graphs(self):
        return len(self.batch_ptr) - 1

    @property
    def num_nodes_per_graph(self):
        return np.diff(self.batch_ptr)

    @property
    def num_nodes(self):
        return len(self.positions)

    @property
    def system_idx(self):
        return np.repeat(np.arange(self.num_graphs), self.num_nodes_per_graph)

    @property
    def batch(self):
        return self.system_idx


def apply_bias_plugins(owner, state, physical, energy, forces):
    """Apply import-path controls before confinement, preserving physical data.

    Plugins are zero-argument factories/classes implementing
    ``apply(state, physical, propagation, options)``. They return complete
    propagation energy/forces and optional per-system diagnostics. Plugin
    instances own their trajectory-local state; the evaluator never does.
    """
    diagnostics = {}
    for plugin, options in getattr(owner, "force_controls", ()):
        missing = set(getattr(plugin, "required_properties", ())) - set(physical)
        if missing:
            raise ValueError(f"Sampling bias requires unavailable physical outputs: {sorted(missing)}.")
        if getattr(plugin, "requires_member_contributions", False) and not physical.get("has_raw_members", "uncertainty" not in physical):
            raise ValueError("Sampling bias requires real ordered ensemble members, not aggregate uncertainty.")
        result = plugin.apply(state, physical, {"energy": energy, "forces": forces}, options)
        if not isinstance(result, dict) or not {"energy", "forces"}.issubset(result):
            raise TypeError("Bias apply() must return energy and forces.")
        new_energy, new_forces = result["energy"], result["forces"]
        if new_energy.shape != energy.shape or new_forces.shape != forces.shape:
            raise ValueError("Bias transforms must preserve system energies and packed force shapes.")
        if hasattr(energy, "detach"):
            import torch
            if any(value.device != old.device or value.dtype != old.dtype
                   for value, old in ((new_energy, energy), (new_forces, forces))):
                raise ValueError("Native bias transforms must preserve device and dtype.")
            if not bool(torch.isfinite(new_energy).all()) or not bool(torch.isfinite(new_forces).all()):
                raise ValueError("Sampling bias produced nonfinite values.")
        elif not np.isfinite(new_energy).all() or not np.isfinite(new_forces).all():
            raise ValueError("Sampling bias produced nonfinite values.")
        energy, forces = new_energy, new_forces
        diagnostics.update(dict(result.get("diagnostics") or {}))
    return energy, forces, diagnostics


def packed_force_state(owner, batch):
    """Expose the neutral state contract even at an engine compatibility edge."""
    if hasattr(batch, "system_idx"):
        return batch
    packed = getattr(owner, "_packed", None)
    if packed is not None:
        return packed
    import torch
    from alframework.ml_interfaces.batched_inference import PackedBatch, batch_offsets
    cells, pbc = getattr(batch, "cell", None), getattr(batch, "pbc", None)
    if cells is None:
        cells = batch.positions.new_zeros((int(batch.num_graphs), 3, 3))
    if pbc is None:
        pbc = torch.zeros((int(batch.num_graphs), 3), dtype=torch.bool, device=batch.positions.device)
    return PackedBatch(batch.positions, batch.atomic_numbers, batch.atomic_masses,
                       batch.velocities, cells, pbc, batch_offsets(batch))

def validated_method_bias(method, options=None):
    """Validate the existing UDD convention without changing its sign."""
    values = dict(options or {})
    if method != "udd":
        return None, 1.e-12
    weight = values.get("udd_bias_weight", values.get("bias_weight"))
    if weight is None:
        raise ValueError("UDD requires udd_bias_weight.")
    weight = float(weight)
    eps = float(values.get("udd_bias_eps", values.get("bias_eps", 1.e-12)))
    if not np.isfinite(weight):
        raise ValueError("udd_bias_weight must be finite.")
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("udd_bias_eps must be finite and nonnegative.")
    return weight, eps


def configure_controls(self, method, options=None):
    """Select propagation controls without changing physical diagnostics."""
    weight, epsilon = validated_method_bias(method, options)
    if weight is not None and self.gap_seeking_settings.get("enabled", False):
        raise ValueError("UDD combined with LCM switching is not supported.")
    self.udd_bias_weight, self.udd_bias_eps = weight, epsilon
    self._lcm_controls.clear()


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
    if self.udd_bias_weight is not None:
        raise ValueError("UDD combined with LCM switching is not supported.")
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


def set_numpy_lcm_mode(self, graph_index, *, pair, sigma, alpha_eV):
    if self.udd_bias_weight is not None:
        raise ValueError("UDD combined with LCM switching is not supported.")
    if not self.gap_seeking_settings.get("enabled", False):
        raise RuntimeError("Cannot enter LCM mode when gap_seeking is disabled.")
    pair = tuple(int(value) for value in pair)
    allowed = {(int(row["lower_state"]), int(row["upper_state"]))
               for row in self.gap_seeking_settings.get("rows", [])}
    if pair not in allowed or int(graph_index) < 0:
        raise ValueError("LCM pair or graph index is not eligible.")
    control = {"pair": pair, "sigma": float(sigma), "alpha_eV": float(alpha_eV)}
    existing = self._lcm_controls.get(int(graph_index))
    if existing is not None and existing != control:
        raise RuntimeError("Replica already entered stay-fixed LCM mode.")
    self._lcm_controls[int(graph_index)] = control


def torch_well_energy_forces(self, batch, positions):
    import torch
    from alframework.ml_interfaces.batched_inference import _segment_sum
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


def torch_reduce_contributions(batch, energy, forces):
    import torch
    from alframework.ml_interfaces.batched_inference import _segment_sum, _segment_max
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


def torch_lcm_energy_forces(
    lower_energy,
    upper_energy,
    lower_forces,
    upper_forces,
    *,
    sigma: float,
    alpha_eV: float,
):
    import torch
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


def torch_forward(self, batch):
    import torch
    from alframework.ml_interfaces.batched_inference import batch_slices, batch_offsets
    prediction = dict(self.ensemble_forward(batch))
    energy, forces = self._normalize_contributions(batch, prediction["energy_contributions"], prediction["force_contributions"])
    self.last_prediction = {"energy_contributions": energy.detach(), "force_contributions": forces.detach()}
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
        self.last_prediction.setdefault("state_contributions", {})[state] = {
            "energy_contributions": state_energy.detach(), "force_contributions": state_forces.detach()}
        diagnostics.update({f"sE{state}": result["energy"], f"sE{state}_stdev": result["energy_std"],
                            f"F{state}": result["forces"], f"F{state}_stdev": result["force_std"]})
        self._diagnostic_force_keys.update((f"F{state}", f"F{state}_stdev"))
    state_energies = {int(state): value for state, value in dict(prediction.get("state_energy_contributions") or {}).items()}
    for state, values in state_contributions.items():
        state_energies.setdefault(state, values["energy_contributions"])
    state_energies.setdefault(self.selected_state_value, energy)
    self.last_prediction["state_energy_contributions"] = {
        state: normalize_energy_contributions(batch, values).detach() for state, values in state_energies.items()}
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
    if self.udd_bias_weight is not None:
        if prediction.get("uncertainty") is not None and not prediction.get("has_raw_members", False):
            raise ValueError("UDD requires ordered ensemble-member energies and forces; aggregate uncertainty is insufficient.")
        indices = torch.repeat_interleave(
            torch.arange(int(batch.num_graphs), device=forces.device), batch.num_nodes_per_graph)
        deviations = energy - selected["energy"].unsqueeze(0)
        force_deviations = forces - selected["forces"].unsqueeze(0)
        std = selected["energy_std"]
        active = std > self.udd_bias_eps
        denominator = torch.where(active, std, torch.ones_like(std))
        bias_force = float(self.udd_bias_weight) * (
            deviations[:, indices, None] * force_deviations).mean(dim=0)
        bias_force = bias_force / denominator[indices, None] * active[indices, None]
        bias_energy = float(self.udd_bias_weight) * std
        selected_energy = selected_energy + bias_energy.reshape(-1, 1)
        selected_forces = selected_forces + bias_force
        from alframework.tools.force_screen import _force_maximum
        diagnostics.update(E_en_bias=bias_energy, F_en_bias=bias_force,
                           udd_bias_Fmax=_force_maximum(bias_force, ptr))
        self._diagnostic_force_keys.add("F_en_bias")
    selected_energy, selected_forces, extra_diagnostics = apply_bias_plugins(
        self, packed_force_state(self, batch) if getattr(self, "force_controls", ()) else batch,
        prediction, selected_energy, selected_forces)
    if set(extra_diagnostics) & set(diagnostics):
        raise ValueError("Bias diagnostics must not replace physical or built-in control diagnostics.")
    for key, value in extra_diagnostics.items():
        value = torch.as_tensor(value, device=selected_energy.device)
        diagnostics[key] = value.expand(int(batch.num_graphs)) if value.ndim == 0 else value
    well_energy, well_forces = self._well_energy_forces(batch, batch.positions)
    if "confinement_energy" in prediction:
        extra_energy = prediction["confinement_energy"].reshape(-1, 1)
        extra_forces = prediction["confinement_forces"].reshape(-1, 3)
        well_energy = extra_energy if well_energy is None else well_energy + extra_energy
        well_forces = extra_forces if well_forces is None else well_forces + extra_forces
    if well_energy is not None:
        selected_energy += well_energy
        selected_forces += well_forces
    diagnostics.update(propagation_force_diagnostics(selected_forces, well_forces, batch_ptr=ptr))
    self.last_diagnostics = {key: value.detach() for key, value in diagnostics.items()}
    return {"energy": selected_energy, "forces": selected_forces}


def numpy_forward(self, atoms_list):
    atoms_list = list(atoms_list)
    prediction = self.raw_ensemble_forward(atoms_list)
    self.last_prediction = prediction
    energy, forces = prediction["energy_contributions"], prediction["force_contributions"]
    ptr = np.cumsum([0, *[len(atoms) for atoms in atoms_list]])
    energy_mean, force_mean = energy.mean(axis=0), forces.mean(axis=0)
    energy_std, force_std = energy.std(axis=0), forces.std(axis=0)
    result_energy, result_forces = energy_mean.copy(), force_mean.copy()
    state_members = prediction["state_contributions"]
    state_means = {state: (values["energy_contributions"].mean(axis=0),
                           values["force_contributions"].mean(axis=0))
                   for state, values in state_members.items()}
    diagnostics = []
    for index, atoms in enumerate(atoms_list):
        selection = slice(ptr[index], ptr[index + 1])
        force_deviation = force_std[selection]
        row = {"force_screen_schema": SCHEMA, "force_screen_source": SOURCE,
               "Es": float(energy_std[index]), "Fs": float(np.abs(force_deviation).mean()),
               "Fsmax": float(np.abs(force_deviation).max()), "Fsrms": float(np.sqrt(np.mean(force_deviation**2)))}
        if "uncertainty" in prediction:
            row.update({key: float(values[index]) for key, values in prediction["uncertainty"].items()})
            row.pop("Fsrms", None)
        for state, (state_e, state_f) in state_means.items():
            values = state_members[state]
            row.update({f"sE{state}": float(state_e[index]),
                        f"sE{state}_stdev": float(values["energy_contributions"][:, index].std()),
                        f"F{state}": state_f[selection].copy(),
                        f"F{state}_stdev": values["force_contributions"][:, selection].std(axis=0),
                        f"physical_F{state}_max": _norm_max(state_f[selection])})
        missing = set(self.physical_force_states) - set(state_means)
        if missing:
            raise ValueError(f"Missing physical state forces: {sorted(missing)}.")
        row["Fmeanmax"] = max(row[f"physical_F{state}_max"] for state in state_means)
        for gap in self.gap_rows:
            lo, hi = int(gap["lower_state"]), int(gap["upper_state"])
            members = (prediction["state_energy_contributions"][hi][:, index]
                       - prediction["state_energy_contributions"][lo][:, index])
            row.update({gap["gap_key"]: float(members.mean()), f"{gap['gap_key']}_stdev": float(members.std()),
                        f"{gap['gap_key']}_model_count": len(members)})
        correction = np.zeros_like(force_mean[selection])
        control = self._lcm_controls.get(index)
        if control is not None:
            lo, hi = control["pair"]
            le, lf = state_means[lo]; ue, uf = state_means[hi]
            gap = ue[index] - le[index]
            smooth = np.sqrt(gap * gap + 1.e-12)
            denominator = smooth + control["alpha_eV"]
            derivative = control["sigma"] * (2 * gap * denominator - gap**3 / smooth) / denominator**2
            result_energy[index] = .5 * (le[index] + ue[index]) + control["sigma"] * gap**2 / denominator
            pair_mean = .5 * (lf[selection] + uf[selection])
            result_forces[selection] = pair_mean + derivative * (uf[selection] - lf[selection])
            correction = result_forces[selection] - pair_mean
        row.update(lcm_bias_Fmax=_norm_max(result_forces[selection] - force_mean[selection]),
                   lcm_correction_Fmax=_norm_max(correction))
        if self.udd_bias_weight is not None:
            if not prediction["has_raw_members"]:
                raise ValueError("UDD requires ordered ensemble-member energies and forces; aggregate uncertainty is insufficient.")
            std = energy_std[index]
            bias = np.zeros_like(force_mean[selection])
            if std > self.udd_bias_eps:
                bias = self.udd_bias_weight * np.mean(
                    (energy[:, index] - energy_mean[index])[:, None, None]
                    * (forces[:, selection] - force_mean[selection]), axis=0) / std
            energy_bias = self.udd_bias_weight * std
            result_energy[index] += energy_bias
            result_forces[selection] += bias
            row.update(E_en_bias=float(energy_bias), F_en_bias=bias.copy(), udd_bias_Fmax=_norm_max(bias))
        diagnostics.append(row)
    if getattr(self, "force_controls", ()):
        controlled_energy, result_forces, extra_diagnostics = apply_bias_plugins(
            self, NumpyForceState.from_atoms(atoms_list), prediction,
            result_energy.reshape(-1, 1), result_forces)
        result_energy = np.asarray(controlled_energy).reshape(-1)
        if any(set(extra_diagnostics) & set(row) for row in diagnostics):
            raise ValueError("Bias diagnostics must not replace physical or built-in control diagnostics.")
        for key, value in extra_diagnostics.items():
            value = np.asarray(value)
            for index, row in enumerate(diagnostics):
                row[key] = copy.deepcopy(value.item() if value.ndim == 0 else value[index])
    for index, atoms in enumerate(atoms_list):
        row = diagnostics[index]
        selection = slice(ptr[index], ptr[index + 1])
        result_energy[index] += prediction["confinement_energy"][index]
        well_force = prediction["confinement_forces"][selection].copy()
        result_forces[selection] += well_force
        if self.well_params is not None:
            p = self.well_params
            relative = atoms.positions - np.asarray(p.get("origin", [0., 0., 0.]))
            radii = np.linalg.norm(relative, axis=1)
            weights = atoms.get_masses() if p.get("mass_weighted", True) else np.ones(len(atoms))
            depth = np.maximum(radii - float(p["r_start"]), 0.)
            result_energy[index] += np.sum(weights * depth * float(p["force"]))
            configured_force = -relative / np.maximum(radii[:, None], 1.e-12) * (
                weights * (radii > float(p["r_start"])) * float(p["force"]))[:, None]
            well_force += configured_force
            result_forces[selection] += configured_force
        row.update(confinement_Fmax=_norm_max(well_force), propagation_Fmax=_norm_max(result_forces[selection]))
    if any(index >= len(atoms_list) for index in self._lcm_controls):
        raise IndexError("LCM graph index is outside the evaluated batch.")
    self.last_diagnostics = diagnostics
    return {"energy": result_energy.reshape(-1, 1), "forces": result_forces,
            "diagnostics": copy.deepcopy(diagnostics), **prediction}


def _numpy_physical(prediction, atoms_list, selected_state):
    """Validate the NumPy implementation of the structural prediction contract."""
    result = dict(prediction)
    energy = np.asarray(result["energy_contributions"], dtype=float)
    forces = np.asarray(result["force_contributions"], dtype=float)
    systems, atoms = len(atoms_list), sum(len(system) for system in atoms_list)
    if energy.ndim != 2 or energy.shape[1] != systems or energy.shape[0] < 1:
        raise ValueError("Physical energies must have shape [models, systems].")
    if forces.shape != (energy.shape[0], atoms, 3):
        raise ValueError("Physical forces must have shape [models, total_atoms, 3].")
    if not np.isfinite(energy).all() or not np.isfinite(forces).all():
        raise ValueError("Physical predictions must be finite.")
    result["energy_contributions"], result["force_contributions"] = energy, forces
    states = {int(state): dict(values) for state, values in dict(result.get("state_contributions") or {}).items()}
    states.setdefault(int(selected_state), {"energy_contributions": energy, "force_contributions": forces})
    for state, values in states.items():
        values["energy_contributions"] = np.asarray(values["energy_contributions"], dtype=float)
        values["force_contributions"] = np.asarray(values["force_contributions"], dtype=float)
        if values["energy_contributions"].shape != energy.shape or values["force_contributions"].shape != forces.shape:
            raise ValueError("Physical states must preserve the same ordered ensemble and packed atoms.")
        if not np.isfinite(values["energy_contributions"]).all() or not np.isfinite(values["force_contributions"]).all():
            raise ValueError("Physical state predictions must be finite.")
    result["state_contributions"] = states
    state_energy = {int(state): np.asarray(value, dtype=float)
                    for state, value in dict(result.get("state_energy_contributions") or {}).items()}
    for state, values in states.items():
        state_energy.setdefault(state, values["energy_contributions"])
    if any(value.shape != energy.shape or not np.isfinite(value).all() for value in state_energy.values()):
        raise ValueError("Physical state energies must preserve the same finite ordered ensemble.")
    result["state_energy_contributions"] = state_energy
    result.setdefault("has_raw_members", "uncertainty" not in result)
    # Additive entries exist only in the compatibility bridge for an ASE
    # calculator that already contains a well; physical members exclude them.
    result.setdefault("confinement_energy", np.zeros(systems))
    result.setdefault("confinement_forces", np.zeros((atoms, 3)))
    return result


def _metadata(evaluator, name, default=None):
    capabilities = dict(getattr(evaluator, "capabilities", {}) or {})
    return getattr(evaluator, name, capabilities.get(name, default))


class NumpySamplingModel:
    """NumPy sampling controls around a model-family-independent evaluator."""
    is_sampling_model = True

    def __init__(self, evaluator, *, selected_state_value, physical_force_states,
                 gap_rows, gap_seeking_settings, well_params, device=None):
        self.evaluator = evaluator
        self.selected_state_value = int(selected_state_value)
        self.physical_force_states = list(physical_force_states)
        self.gap_rows, self.gap_seeking_settings = list(gap_rows), dict(gap_seeking_settings)
        self.well_params = well_params
        self._device = device
        self._lcm_controls = {}
        self.udd_bias_weight, self.udd_bias_eps = None, 1.e-12
        self.last_diagnostics, self.last_prediction = [], None
        self.calculator_interface = _metadata(evaluator, "calculator_interface", "native")
        self.calculator_loader = _metadata(evaluator, "calculator_loader", type(evaluator).__name__)
        for name in ("supports_pbc", "requires_pbc", "supports_partial_pbc", "supports_constraints", "supports_member_contributions"):
            setattr(self, name, _metadata(evaluator, name, name == "supports_member_contributions"))

    def set_device(self, device):
        self._device = device
        if hasattr(self.evaluator, "set_device"):
            self.evaluator.set_device(device)

    def require_physical_states(self, states):
        self.physical_force_states = sorted(set(int(state) for state in states))
        if hasattr(self.evaluator, "request_states"):
            self.evaluator.request_states(self.physical_force_states, self.gap_rows,
                                          self.gap_seeking_settings.get("rows", []))

    def configure_sampling_method(self, method, options=None):
        return configure_controls(self, method, options)

    def set_lcm_mode(self, index, *, pair, sigma, alpha_eV):
        return set_numpy_lcm_mode(self, index, pair=pair, sigma=sigma, alpha_eV=alpha_eV)

    def raw_ensemble_forward(self, atoms_list):
        predict = getattr(self.evaluator, "predict_atoms", None) or self.evaluator.predict
        return _numpy_physical(predict(atoms_list), atoms_list, self.selected_state_value)

    def evaluate_atoms(self, atoms_list):
        return numpy_forward(self, atoms_list)

    def diagnostics_for_graph(self, index):
        return copy.deepcopy(self.last_diagnostics[int(index)])


def _torch_sampling_model(evaluator, **settings):
    """Construct the optional tensor wrapper only on a tensor-engine path."""
    import torch
    from ase import Atoms
    from alframework.ml_interfaces.batched_inference import BatchedEnsembleCalculator, batch_slices

    class TorchSamplingModel(BatchedEnsembleCalculator):
        is_sampling_model = True

        def __init__(self):
            super().__init__(**settings,
                calculator_interface=_metadata(evaluator, "calculator_interface", "native"),
                calculator_loader=_metadata(evaluator, "calculator_loader", type(evaluator).__name__))
            self.evaluator = evaluator
            for name in ("supports_pbc", "requires_pbc", "supports_partial_pbc", "supports_constraints", "supports_member_contributions"):
                setattr(self, name, _metadata(evaluator, name, name == "supports_member_contributions"))

        def set_device(self, device):
            self._device = torch.device(device)
            if hasattr(self.evaluator, "set_device"):
                self.evaluator.set_device(device)

        def require_physical_states(self, states):
            super().require_physical_states(states)
            if hasattr(self.evaluator, "request_states"):
                self.evaluator.request_states(self.physical_force_states, self.gap_rows,
                                              self.gap_seeking_settings.get("rows", []))

        def ensemble_forward(self, batch):
            if self.evaluator.array_backend == "torch":
                return self.evaluator.predict(batch)
            systems = []
            for index, selection in enumerate(batch_slices(batch)):
                atoms = Atoms(numbers=batch.atomic_numbers[selection].detach().cpu().numpy(),
                    positions=batch.positions[selection].detach().cpu().numpy(),
                    masses=batch.atomic_masses[selection].detach().cpu().numpy(),
                    cell=batch.cell[index].detach().cpu().numpy(), pbc=batch.pbc[index].detach().cpu().numpy())
                atoms.set_velocities(batch.velocities[selection].detach().cpu().numpy())
                systems.append(atoms)
            predict = getattr(self.evaluator, "predict_atoms", None) or self.evaluator.predict
            raw = _numpy_physical(predict(systems), systems, self.selected_state_value)

            def tensor(value):
                if isinstance(value, dict):
                    return {key: tensor(item) for key, item in value.items()}
                if isinstance(value, np.ndarray):
                    return torch.as_tensor(value, dtype=batch.positions.dtype, device=batch.positions.device)
                return value
            return tensor(raw)

    return TorchSamplingModel()


def build_sampling_model(evaluator, sampler_config, *, properties_list=None,
                         selected_state_value=None, device=None):
    """Compose sampling force controls without adding science to inference."""
    from alframework.ml_interfaces.sampling_inference import adapt_physical_evaluator
    from alframework.tools.sampling_validation import configured_alchemi_gap_rows, _physical_force_states
    evaluator = adapt_physical_evaluator(evaluator, device=device)
    config = copy.deepcopy(sampler_config)
    mode = config.get("model_mode", "ground_state")
    selected = int(_metadata(evaluator, "selected_state_value", 0) if selected_state_value is None else selected_state_value)
    if properties_list is not None:
        physical = _physical_force_states(mode, selected, properties_list)
        gaps, seeking = configured_alchemi_gap_rows(config, properties_list, selected)
    else:
        legacy = getattr(evaluator, "model", evaluator)
        physical = list(getattr(legacy, "physical_force_states", [selected]))
        gaps = list(getattr(legacy, "gap_rows", []))
        seeking = dict(getattr(legacy, "gap_seeking_settings", {"enabled": False, "rows": []}))
        if "gap_seeking" in config and not dict(config.get("gap_seeking") or {}).get("enabled", False):
            seeking = {"enabled": False, "rows": []}
    well = dict(config.get("MLMD_calculator_options") or {}).get("well_params")
    if well is None and "driver" not in config:
        well = getattr(getattr(evaluator, "model", evaluator), "well_params", None)
    udd, extra_controls = None, []
    sampling = dict(config.get("sampling") or {})
    for raw in sampling.get("biases", []):
        bias = {"strategy": raw} if isinstance(raw, str) else dict(raw)
        name = str(bias.get("strategy", bias.get("method", "")))
        if "." not in name:
            name = name.lower()
        options = dict(bias.get("options") or {})
        if name == "udd":
            if udd is not None:
                raise ValueError("Configure the UDD bias once.")
            udd = {"udd_bias_weight": bias.get("weight", bias.get("udd_bias_weight", options.get("weight", config.get("udd_bias_weight")))),
                   "udd_bias_eps": bias.get("epsilon", bias.get("udd_bias_eps", options.get("epsilon", config.get("udd_bias_eps", 1.e-12))))}
        elif name in {"well", "confinement"}:
            well = options or {key: value for key, value in bias.items() if key not in {"strategy", "method"}}
        elif name in {"none", ""}:
            continue
        elif "." in name:
            from alframework.tools.tools import load_module_from_string
            factory = load_module_from_string(name)
            plugin = factory() if callable(factory) else factory
            if not callable(getattr(plugin, "apply", None)):
                raise TypeError(f"Sampling bias {name!r} must implement apply().")
            values = {key: value for key, value in bias.items() if key not in {"strategy", "method", "options"}}
            values.update(options)
            extra_controls.append((plugin, values))
        else:
            raise ValueError(f"Unsupported sampling force control {name!r}; configure an available bias strategy.")
    if udd is None and sampling.get("method") == "udd":
        options = dict(sampling.get("options") or {})
        udd = {"udd_bias_weight": options.get("udd_bias_weight", config.get("udd_bias_weight")),
               "udd_bias_eps": options.get("udd_bias_eps", config.get("udd_bias_eps", 1.e-12))}
    if udd is not None and seeking.get("enabled", False):
        raise ValueError("UDD combined with LCM switching is not supported.")
    settings = dict(selected_state_value=selected, physical_force_states=physical,
                    gap_rows=gaps, gap_seeking_settings=seeking,
                    well_params=None if well is None else copy.deepcopy(well), device=device)
    engine = dict(config.get("driver") or {}).get("engine")
    if engine is None:
        dynamics = dict(config.get("dynamics") or {})
        engine = "ase" if dynamics.get("driver", "ase") == "ase" else dynamics.get("backend")
    if evaluator.array_backend == "numpy" and engine == "ase":
        model = NumpySamplingModel(evaluator, **settings)
    else:
        model = _torch_sampling_model(evaluator, **settings)
    model.require_physical_states(physical)
    model.force_controls = extra_controls
    if udd is not None:
        model.configure_sampling_method("udd", udd)
    return model


