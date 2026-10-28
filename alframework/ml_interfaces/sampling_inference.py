"""Inference interfaces shared by ASE and optional tensor-based execution.

Importing this module requires only base ALF dependencies.  Native model and
tensor imports happen only after a native inference adapter is selected.
"""

from __future__ import annotations

import copy
import inspect
import numpy as np
from ase.calculators.calculator import all_changes

from alframework.tools.force_screen import SCHEMA, SOURCE
from alframework.tools.sampling_validation import (
    configured_alchemi_gap_rows, _physical_force_states, validate_ase_calculators,
    validate_ase_results, validate_ase_force_shape,
)
from alframework.tools.tools import load_module_from_string


def validated_method_bias(method, options=None):
    from alframework.samplers.force_controls import validated_method_bias as validate
    return validate(method, options)


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
        from alframework.samplers.force_controls import configure_controls
        return configure_controls(self, method, options)

    def set_lcm_mode(self, graph_index, *, pair, sigma, alpha_eV):
        from alframework.samplers.force_controls import set_numpy_lcm_mode
        return set_numpy_lcm_mode(self, graph_index, pair=pair, sigma=sigma, alpha_eV=alpha_eV)

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
        # Compatibility facade; sampling controls are not inference internals.
        from alframework.samplers.force_controls import numpy_forward
        return numpy_forward(self, atoms_list)

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


class LegacyPhysicalEvaluator:
    """Expose legacy adapters' physical members without invoking their controls.

    This is a compatibility bridge. New adapters can implement ``predict``
    directly and do not inherit any ALF or Torch model base class.
    """

    def __init__(self, model, device=None):
        self.model = model
        self.array_backend = "numpy" if hasattr(model, "raw_ensemble_forward") else "torch"
        self.calculator_interface = getattr(model, "calculator_interface", "native")
        self.calculator_loader = getattr(model, "calculator_loader", type(model).__name__)
        self.selected_state_value = int(getattr(model, "selected_state_value", 0))
        self.capabilities = {
            key: getattr(model, key, default) for key, default in (
                ("supports_pbc", False), ("requires_pbc", False),
                ("supports_partial_pbc", self.array_backend == "numpy"),
                ("supports_constraints", self.array_backend == "numpy"),
                ("supports_member_contributions", True))
        }
        self._device = device if device is not None else getattr(model, "_device", None)

    def set_device(self, device):
        self._device = device
        if hasattr(self.model, "set_device"):
            self.model.set_device(device)

    def request_states(self, physical_states, gap_rows, seeking_rows=()):
        # Translate prediction requirements for legacy adapters. No switching
        # thresholds, bias weights, or acquisition decisions enter inference.
        if hasattr(self.model, "require_physical_states"):
            self.model.require_physical_states(physical_states)
        if self.array_backend == "numpy":
            self.model.gap_rows = list(gap_rows)
            self.model.gap_seeking_settings = {"enabled": False, "rows": list(seeking_rows)}

    def predict(self, data):
        if self.array_backend == "numpy":
            return self.model.raw_ensemble_forward(data)
        return self.model.ensemble_forward(data)

    def predict_atoms(self, atoms_list):
        if self.array_backend == "numpy":
            return self.predict(atoms_list)
        from alframework.ml_interfaces.batched_inference import PackedBatch
        return self.predict(PackedBatch.from_atoms(atoms_list, device=self._device))


def adapt_physical_evaluator(model, device=None):
    """Adapt a structural evaluator or an existing raw-member model."""
    if getattr(model, "is_sampling_model", False):
        return model.evaluator
    if isinstance(model, LegacyPhysicalEvaluator):
        return model
    if callable(getattr(model, "ensemble_forward", None)) or callable(getattr(model, "raw_ensemble_forward", None)):
        return LegacyPhysicalEvaluator(model, device=device)
    if callable(getattr(model, "predict", None)) or callable(getattr(model, "predict_atoms", None)):
        if getattr(model, "array_backend", None) not in {"numpy", "torch"}:
            raise TypeError("A physical evaluator must declare array_backend='numpy' or 'torch'.")
        return model
    raise TypeError("Inference must provide physical predict()/predict_atoms() or raw ensemble outputs.")


def load_physical_evaluator(model_reference, *, model_mode="ground_state", selected_state_value=None,
                            ML_config=None, properties_list=None, requested_properties=None, device=None):
    """Load physical inference from an opaque artifact reference.

    New factories accept ``model_reference`` and ``requested_properties``.
    Compatibility factories receive only prediction configuration; sampling
    biases, acquisition scores, thresholds, and selection budgets are excluded.
    """
    from alframework.ml_interfaces.model_reference import ModelReference
    reference = ModelReference.from_value(model_reference)
    model_config = dict(reference.metadata.get("ML_config", ML_config or {}))
    properties = dict(reference.metadata.get("properties_list", properties_list or {}))
    request = dict(requested_properties or {})
    request.setdefault("model_mode", model_mode)
    request.setdefault("selected_state_value", selected_state_value)
    request.setdefault("properties_list", properties)
    request.setdefault("model_config", model_config)
    if "ML_config" in reference.metadata:
        request["model_config"] = model_config
    if "properties_list" in reference.metadata:
        request["properties_list"] = properties
    options = dict(reference.options)
    loader = load_module_from_string(reference.adapter)
    parameters = inspect.signature(loader).parameters
    new_api = "model_reference" in parameters or "requested_properties" in parameters or "reference" in parameters
    if new_api:
        evaluator = loader(reference, requested_properties=request, device=device, **options)
    elif reference.interface == "ase":
        if options.pop("use_potential_specific_code", None) == "neurochem":
            calculators = loader({"model_path": reference.artifact_location.rstrip("/") + "/",
                                  "Nn": 8, "gpu": "0", **options})
            uncertainty_mode = "neurochem"
        else:
            options.setdefault("device", str(device or "cpu"))
            calculators = loader(reference.artifact_location.rstrip("/") + "/", **options)
            uncertainty_mode = None
        evaluator = NumpyASEEnsembleModel(calculators, model_mode=model_mode,
            selected_state_value=selected_state_value, device=device,
            calculator_loader=reference.adapter, uncertainty_mode=uncertainty_mode)
    else:
        if reference.adapter.endswith(("load_hippynn_batched_model", "load_hippynn_alchemi_model")):
            # The HIPPYNN compatibility adapter owns its checkpoint layout and
            # does not load the NVIDIA engine when used by another driver.
            from alframework.ml_interfaces.batched_inference import load_hippynn_batched_model
            loader = load_hippynn_batched_model
        evaluator = loader(reference.artifact_location, model_mode=model_mode,
            selected_state_value=selected_state_value, ML_config=model_config,
            properties_list=properties, sampler_config={"model_mode": model_mode},
            device=device, **options)
    evaluator = adapt_physical_evaluator(evaluator, device=device)
    evaluator.model_reference = reference
    if hasattr(evaluator, "set_device") and device is not None:
        evaluator.set_device(device)
    return evaluator


def build_sampling_model(evaluator, sampler_config, *, properties_list=None,
                         selected_state_value=None, device=None):
    """Compatibility import for the sampling-owned force-pipeline factory."""
    from alframework.samplers.force_controls import build_sampling_model as build
    return build(evaluator, sampler_config, properties_list=properties_list,
                 selected_state_value=selected_state_value, device=device)
