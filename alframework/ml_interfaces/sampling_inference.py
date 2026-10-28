"""Inference interfaces shared by ASE and optional tensor-based execution.

Importing this module requires only base ALF dependencies.  Native model and
tensor imports happen only after a native inference adapter is selected.
"""

from __future__ import annotations

import copy
import numpy as np
from ase.calculators.calculator import all_changes

from alframework.tools.force_screen import SCHEMA, SOURCE
from alframework.tools.sampling_validation import (
    configured_alchemi_gap_rows, _physical_force_states, validate_ase_calculators,
    validate_ase_results, validate_ase_force_shape,
)
from alframework.tools.tools import load_module_from_string


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


def _norm_max(values):
    return float(np.linalg.norm(np.asarray(values), axis=-1).max())


class NumpyASEEnsembleModel:
    """Ordered ASE ensemble inference with the same scientific reductions.

    ASE calculators are evaluated sequentially.  This adapter does not imply
    accelerated inference, even when its caller uses a batched dynamics engine.
    """

    supports_pbc = True
    supports_partial_pbc = True
    supports_constraints = True
    requires_pbc = False
    calculator_interface = "ase_fallback"

    def __init__(self, calculators, *, model_mode="ground_state", selected_state_value=None,
                 physical_force_states=None, gap_rows=None, gap_seeking_settings=None,
                 well_params=None, device=None, calculator_loader=None, uncertainty_mode=None):
        self.calculators = list(calculators) if isinstance(calculators, (list, tuple)) else [calculators]
        validate_ase_calculators(self.calculators)
        self.selected_state_value = 0 if selected_state_value is None else int(selected_state_value)
        self.model_mode = str(model_mode)
        self.physical_force_states = sorted(set(physical_force_states or [self.selected_state_value]))
        self.gap_rows = list(gap_rows or [])
        self.gap_seeking_settings = dict(gap_seeking_settings or {"enabled": False, "rows": []})
        self.well_params = None if well_params is None else dict(well_params)
        self.calculator_loader = str(calculator_loader or type(self).__name__)
        self.uncertainty_mode = uncertainty_mode
        self._device = device
        self.udd_bias_weight, self.udd_bias_eps = None, 1.e-12
        self._lcm_controls = {}
        self.last_diagnostics = []
        self.last_prediction = None

    def set_device(self, device):
        self._device = device

    def require_physical_states(self, states):
        self.physical_force_states = sorted(set(int(state) for state in states))

    def configure_sampling_method(self, method, options=None):
        weight, epsilon = validated_method_bias(method, options)
        if weight is not None and self.gap_seeking_settings.get("enabled", False):
            raise ValueError("UDD combined with LCM switching is not supported.")
        self.udd_bias_weight, self.udd_bias_eps = weight, epsilon
        self._lcm_controls.clear()

    def set_lcm_mode(self, graph_index, *, pair, sigma, alpha_eV):
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

    def _states(self):
        states = set(self.physical_force_states)
        states.add(self.selected_state_value)
        for row in self.gap_rows:
            states.update((int(row["lower_state"]), int(row["upper_state"])))
        return sorted(states)

    def raw_ensemble_forward(self, atoms_list):
        """Return unmodified member energies/forces in packed atom order."""
        atoms_list = list(atoms_list)
        if not atoms_list or any(not len(atoms) for atoms in atoms_list):
            raise ValueError("Inference requires nonempty systems.")
        states = self._states()
        force_states = set(self.physical_force_states) | {self.selected_state_value}
        for row in self.gap_seeking_settings.get("rows", []):
            force_states.update((int(row["lower_state"]), int(row["upper_state"])))
        energy_key = "energy" if self.model_mode == "ground_state" else f"sE{self.selected_state_value}"
        force_key = "forces" if self.model_mode == "ground_state" else f"F{self.selected_state_value}"
        required = [energy_key, force_key]
        if self.model_mode == "excited_state":
            required += [f"sE{state}" for state in states]
            required += [f"F{state}" for state in sorted(force_states)]
        required = list(dict.fromkeys(required))
        rows = []
        aggregate_rows = []
        standard = {"energy_stdev", "forces_stdev_mean", "forces_stdev_max"}
        has_raw_members = True
        for atoms in atoms_list:
            energies, forces = [], []
            confinement_energy, confinement_forces = 0., np.zeros((len(atoms), 3))
            state_energy, state_force = {state: [] for state in states}, {state: [] for state in force_states}
            override = None
            for calculator in self.calculators:
                properties = list(required)
                if len(self.calculators) == 1 and standard.issubset(set(calculator.implemented_properties)):
                    properties += sorted(standard)
                calculator.calculate(atoms, properties=properties, system_changes=all_changes)
                result = calculator.results
                validate_ase_results(result, required)
                validate_ase_force_shape(np.asarray(result[force_key]), len(atoms), force_key)
                raw = (len(self.calculators) == 1 and self.model_mode == "ground_state"
                       and "energy_contributions" in result and "forces_contributions" in result)
                if raw:
                    if getattr(calculator, "udd_bias_weight", None) is not None:
                        raise ValueError("ASE inference must expose unbiased members; configure UDD through sampling.method instead.")
                    members = np.asarray(result["energy_contributions"], dtype=float)
                    if members.ndim == 0:
                        raise ValueError("Raw ensemble energy contributions require a member axis.")
                    members = members.reshape(members.shape[0], -1).sum(axis=1)
                    member_forces = np.asarray(result["forces_contributions"], dtype=float)
                    member_count = int(getattr(calculator, "N_models", len(members)))
                    contains_confinement = len(members) > member_count
                    members, member_forces = members[:member_count], member_forces[:member_count]
                    if member_forces.shape != (len(members), len(atoms), 3):
                        raise ValueError("Raw ensemble force contributions must have shape [models, atoms, 3].")
                    if contains_confinement:
                        # MLMD_calculator appends the well as an additive mixer
                        # entry. Keep it out of ensemble moments and UDD bias.
                        confinement_energy = float(np.asarray(result[energy_key]).sum()) - members.mean()
                        confinement_forces = np.asarray(result[force_key], dtype=float) - member_forces.mean(axis=0)
                    energies.extend(members)
                    forces.extend(member_forces)
                    state_energy[self.selected_state_value].extend(members)
                    state_force[self.selected_state_value].extend(member_forces)
                else:
                    energies.append(float(np.asarray(result[energy_key]).sum()))
                    forces.append(np.asarray(result[force_key], dtype=float).copy())
                    for state in states:
                        ekey, fkey = (energy_key, force_key) if self.model_mode == "ground_state" else (f"sE{state}", f"F{state}")
                        state_energy[state].append(float(np.asarray(result[ekey]).sum()))
                        if state in force_states:
                            validate_ase_force_shape(np.asarray(result[fkey]), len(atoms), fkey)
                            state_force[state].append(np.asarray(result[fkey], dtype=float).copy())
                if len(self.calculators) == 1 and not raw:
                    if standard.issubset(result):
                        override = {"Es": float(result["energy_stdev"]), "Fs": float(result["forces_stdev_mean"]),
                                    "Fsmax": float(result["forces_stdev_max"])}
                    elif str(self.uncertainty_mode).lower() == "neurochem":
                        fmean, fmax = calculator.get_Fstddev()
                        override = {"Es": float(calculator.Estddev) * 1000., "Fs": float(fmean), "Fsmax": float(fmax)}
                    if override is not None:
                        if any(not np.isfinite(value) or value < 0 for value in override.values()):
                            raise ValueError("Aggregate uncertainty must contain finite nonnegative statistics.")
                        has_raw_members = False
            rows.append((np.asarray(energies), np.asarray(forces), state_energy, state_force,
                         confinement_energy, confinement_forces))
            aggregate_rows.append(override)
        counts = {len(row[0]) for row in rows}
        if len(counts) != 1 or next(iter(counts)) < 1:
            raise ValueError("Systems must use the same nonempty ordered ensemble.")
        prediction = {
            "energy_contributions": np.stack([row[0] for row in rows], axis=1),
            "force_contributions": np.concatenate([row[1] for row in rows], axis=1),
            "confinement_energy": np.asarray([row[4] for row in rows]),
            "confinement_forces": np.concatenate([row[5] for row in rows], axis=0),
            "state_energy_contributions": {
                state: np.stack([np.asarray(row[2][state]) for row in rows], axis=1) for state in states},
            "state_contributions": {
                state: {"energy_contributions": np.stack([np.asarray(row[2][state]) for row in rows], axis=1),
                        "force_contributions": np.concatenate([np.asarray(row[3][state]) for row in rows], axis=1)}
                for state in sorted(force_states)},
            "has_raw_members": has_raw_members,
        }
        for key in ("energy_contributions", "force_contributions"):
            if not np.isfinite(prediction[key]).all():
                raise ValueError("Ensemble predictions must be finite.")
        if all(row is not None for row in aggregate_rows):
            prediction["uncertainty"] = {key: np.asarray([row[key] for row in aggregate_rows])
                                         for key in ("Es", "Fs", "Fsmax")}
        return prediction

    def evaluate_atoms(self, atoms_list):
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
            diagnostics.append(row)
        if any(index >= len(atoms_list) for index in self._lcm_controls):
            raise IndexError("LCM graph index is outside the evaluated batch.")
        self.last_diagnostics = diagnostics
        return {"energy": result_energy.reshape(-1, 1), "forces": result_forces,
                "diagnostics": copy.deepcopy(diagnostics), **prediction}

    def diagnostics_for_graph(self, graph_index):
        return copy.deepcopy(self.last_diagnostics[int(graph_index)])


def evaluate_model(model, atoms_list, device=None):
    """Evaluate a grouped proposal without requiring a dynamics or MD clock."""
    if hasattr(model, "evaluate_atoms"):
        return model.evaluate_atoms(atoms_list)
    from alframework.ml_interfaces.batched_inference import PackedBatch
    batch = PackedBatch.from_atoms(atoms_list, device=device or getattr(model, "_device", None))
    output = model(batch)
    def numpy(value):
        if isinstance(value, dict):
            return {key: numpy(item) for key, item in value.items()}
        return value.detach().cpu().numpy().copy() if hasattr(value, "detach") else np.asarray(value).copy()
    result = {key: numpy(value) for key, value in output.items()}
    result["diagnostics"] = [model.diagnostics_for_graph(i) for i in range(len(atoms_list))]
    # Native forward retains the exact raw prediction used for propagation.
    raw = getattr(model, "last_prediction", None)
    if raw is not None:
        result.update({key: numpy(raw[key]) for key in ("energy_contributions", "force_contributions",
                       "state_contributions", "state_energy_contributions") if key in raw})
    return result


def load_sampling_calculator(*, sampler_config, ensemble_directory, model_mode,
                            selected_state_value, ML_config, properties_list, device=None):
    """Load inference independently of the selected execution driver."""
    from alframework.tools.batched_config import inference_settings
    settings = inference_settings(sampler_config)
    loader_path, options = settings["loader"], dict(settings.get("options") or {})
    if not loader_path:
        raise ValueError("Sampling requires inference.loader or a compatible existing calculator loader setting.")
    loader = load_module_from_string(loader_path)
    well = dict(sampler_config.get("MLMD_calculator_options") or {}).get("well_params")
    gaps, seeking = configured_alchemi_gap_rows(sampler_config, properties_list, selected_state_value)
    physical = _physical_force_states(model_mode, selected_state_value, properties_list)
    if settings["interface"] == "native":
        from alframework.ml_interfaces.batched_inference import BatchedEnsembleCalculator
        model = loader(ensemble_directory, model_mode=model_mode, selected_state_value=selected_state_value,
                       ML_config=ML_config, properties_list=properties_list,
                       sampler_config=sampler_config, device=device, **options)
        if not isinstance(model, BatchedEnsembleCalculator):
            raise TypeError("Native inference loader must return a BatchedEnsembleCalculator.")
        model.require_physical_states(physical)
        model.well_params, model.gap_rows, model.gap_seeking_settings = well, list(gaps), dict(seeking)
        model.calculator_interface, model.calculator_loader = "native", loader_path
        return model
    mode = sampler_config.get("use_potential_specific_code")
    if str(mode).lower() == "neurochem":
        details = {"model_path": str(ensemble_directory).rstrip("/") + "/", "Nn": 8, "gpu": "0", **options}
        calculators = loader(details)
    else:
        options.setdefault("device", str(device or "cpu"))
        calculators = loader(str(ensemble_directory).rstrip("/") + "/", **options)
    kwargs = dict(model_mode=model_mode, selected_state_value=selected_state_value,
                  physical_force_states=physical, gap_rows=gaps, gap_seeking_settings=seeking,
                  well_params=well, device=device, calculator_loader=loader_path, uncertainty_mode=mode)
    if dict(sampler_config.get("dynamics") or {}).get("driver", "ase") == "batched":
        from alframework.ml_interfaces.batched_inference import ASEEnsembleModel
        return ASEEnsembleModel(calculators, **kwargs)
    return NumpyASEEnsembleModel(calculators, **kwargs)
