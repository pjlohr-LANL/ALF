"""Calculator-agnostic batched molecular dynamics with NVIDIA ALCHEMI.

This module intentionally implements a new sampler task instead of changing
ALF's established ASE samplers. Native ALCHEMI calculators provide batched
ensemble contributions; existing ASE calculator loaders remain available as a
compatibility fallback. Optional GPU-stack imports are guarded so a base ALF
installation can still import and test normally.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from parsl import python_app

from alframework.tools.excited_state_tools import derive_state_property_table
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampler_batching import sampler_batch_size, selected_state
from alframework.tools.tools import annealing_schedule, load_module_from_string


KB_EV_PER_K = 8.617333262145e-5
DEFAULT_FRICTION_PER_FS = 0.02 * units.fs


def _missing_alchemi_error(exc: BaseException | None = None) -> ImportError:
    detail = "" if exc is None else f" Original error: {type(exc).__name__}: {exc}"
    return ImportError(
        "ALCHEMI sampling requires ALF's gpu_dynamics extra "
        "(nvalchemi-toolkit>=0.1.0,<0.2)." + detail
    )


try:
    import torch
    from nvalchemi.models.base import BaseModelMixin, ModelConfig
except Exception as exc:  # pragma: no cover - depends on optional environment
    torch = None
    BaseModelMixin = object
    ModelConfig = None
    _ALCHEMI_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised in the optional integration environment
    _ALCHEMI_IMPORT_ERROR = None


def ensure_alchemi_available() -> None:
    """Raise an actionable error when the optional GPU stack is unavailable."""

    if _ALCHEMI_IMPORT_ERROR is not None or torch is None or ModelConfig is None:
        raise _missing_alchemi_error(_ALCHEMI_IMPORT_ERROR)


def calculate_uncertainty(
    energy_contributions: Any,
    force_contributions: Any,
) -> dict[str, float]:
    """Calculate the exact population statistics used by production MLMD."""

    energies = np.asarray(energy_contributions, dtype=float)
    forces = np.asarray(force_contributions, dtype=float)
    if energies.shape[0] != forces.shape[0]:
        raise ValueError("Energy and force contributions must have the same model count.")
    if energies.shape[0] < 1:
        raise ValueError("At least one model contribution is required.")
    force_stdev = np.std(forces, axis=0)
    return {
        "Es": float(np.std(energies)),
        "Fs": float(np.mean(np.abs(force_stdev))),
        "Fsmax": float(np.max(np.abs(force_stdev))),
    }


def uncertainty_flags(
    diagnostics: dict[str, Any],
    *,
    Escut: float,
    Fscut: float,
) -> dict[str, Any]:
    """Apply ALF's existing energy, mean-force, and max-force thresholds."""

    energy_cutoff = float(Escut)
    force_cutoff = float(Fscut)
    if energy_cutoff <= 0 or force_cutoff <= 0:
        raise ValueError("Escut and Fscut must be positive for ALCHEMI sampling.")
    Es = float(diagnostics["Es"])
    Fs = float(diagnostics["Fs"])
    Fsmax = float(diagnostics["Fsmax"])
    Ecrit = Es > energy_cutoff
    Fcrit = Fs > force_cutoff
    Fmcrit = Fsmax > 3.0 * force_cutoff
    ratios = {
        "energy": Es / energy_cutoff,
        "force_mean": Fs / force_cutoff,
        "force_max": Fsmax / (3.0 * force_cutoff),
    }
    return {
        "Ecrit": bool(Ecrit),
        "Fcrit": bool(Fcrit),
        "Fmcrit": bool(Fmcrit),
        "uncertain": bool(Ecrit or Fcrit or Fmcrit),
        "uncertainty_ratios": ratios,
        "uncertainty_score": float(max(ratios.values())),
        "uncertainty_reasons": [
            name
            for name, active in (
                ("energy", Ecrit),
                ("force_mean", Fcrit),
                ("force_max", Fmcrit),
            )
            if active
        ],
    }


def _sample_range(rng: np.random.Generator, value: Any, name: str) -> float:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            f"{name} must be a two-value numeric range [minimum, maximum]."
        )
    try:
        lower, upper = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain two finite numeric values; received {value!r}."
        ) from exc
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(
            f"{name} must contain two finite numeric values; received {value!r}."
        )
    if lower > upper:
        raise ValueError(
            f"{name} minimum must not exceed its maximum; received {value!r}."
        )
    return float(rng.uniform(lower, upper))


def _temperature_parameters(
    sampler_config: dict[str, Any],
    *,
    count: int,
    random_seed: int,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(int(random_seed))
    parameters = []
    for replica_index in range(int(count)):
        row = {
            "Tamp": _sample_range(rng, sampler_config.get("amp_temp"), "amp_temp"),
            "Tper": _sample_range(rng, sampler_config.get("per_temp"), "per_temp"),
            "Tsrt": _sample_range(rng, sampler_config.get("srt_temp"), "srt_temp"),
            "Tend": _sample_range(rng, sampler_config.get("end_temp"), "end_temp"),
        }
        if row["Tper"] <= 0:
            raise ValueError(
                "per_temp sampled a nonpositive temperature period "
                f"({row['Tper']}) for replica {replica_index}; configure "
                "strictly positive period bounds."
            )
        parameters.append(row)
    return parameters


def _minimum_distance(atoms) -> float:
    distances = np.asarray(atoms.get_all_distances(mic=False), dtype=float)
    nonzero = distances[distances > 0]
    return float(np.min(nonzero)) if nonzero.size else float("inf")


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    record = candidate["record"]
    return (
        -float(record["uncertainty_score"]),
        int(record["step"]),
        int(record["batch_index"]),
        str(record["parent_molecule_id"]),
    )


def _validate_sampling_inputs(
    molecule_objects: list[MoleculesObject],
    sampler_config: dict[str, Any],
) -> tuple[str, int | None]:
    if not isinstance(molecule_objects, list) or not molecule_objects:
        raise ValueError("molecule_objects must be a non-empty list.")
    if not all(isinstance(item, MoleculesObject) for item in molecule_objects):
        raise TypeError("molecule_objects may contain only MoleculesObject instances.")
    expected_size = sampler_batch_size(sampler_config)
    if len(molecule_objects) != expected_size:
        raise ValueError(
            "Strict-full ALCHEMI sampling expected exactly "
            f"{expected_size} molecules but received {len(molecule_objects)}."
        )

    mode = str(sampler_config.get("model_mode", "ground_state")).strip().lower()
    if mode not in {"ground_state", "excited_state"}:
        raise ValueError("model_mode must be 'ground_state' or 'excited_state'.")
    policy = str(sampler_config.get("uncertainty_policy", "stop")).strip().lower()
    if policy not in {"stop", "continue"}:
        raise ValueError("uncertainty_policy must be 'stop' or 'continue'.")

    reference_numbers: tuple[int, ...] | None = None
    states: set[int] = set()
    for molecule in molecule_objects:
        atoms = molecule.get_atoms()
        if atoms is None:
            raise ValueError("ALCHEMI cannot sample a molecule whose atoms are None.")
        if bool(np.any(atoms.get_pbc())):
            raise NotImplementedError("ALCHEMI sampling currently supports only nonperiodic systems.")
        numbers = tuple(int(value) for value in atoms.get_atomic_numbers())
        if reference_numbers is None:
            reference_numbers = numbers
        elif numbers != reference_numbers:
            raise ValueError(
                "Every molecule in an ALCHEMI batch must have identical atom order."
            )
        if mode == "excited_state":
            states.add(selected_state(molecule))
    if len(states) > 1:
        raise ValueError("Every molecule in an excited-state batch must select the same state.")

    for density_key in ("end_dens", "amp_dens", "per_dens"):
        if sampler_config.get(density_key) is not None:
            raise NotImplementedError(
                "ALCHEMI sampling does not yet support density or cell schedules; "
                f"set {density_key} to null."
            )
    return mode, (next(iter(states)) if states else None)


def _candidate_record(
    *,
    diagnostics: dict[str, Any],
    flags: dict[str, Any],
    molecule: MoleculesObject,
    batch_index: int,
    step: int,
    time_ps: float,
    distance: float,
    sampler_config: dict[str, Any],
    policy: str,
    temperature_parameters: dict[str, float],
    temperature_history: list[float],
    selected_state_value: int | None,
) -> dict[str, Any]:
    return {
        "parent_molecule_id": molecule.get_moleculeid(),
        "batch_index": int(batch_index),
        "step": int(step),
        "time_ps": float(time_ps),
        "uncertainty_policy": policy,
        "uncertainty_score": float(flags["uncertainty_score"]),
        "uncertainty_ratios": dict(flags["uncertainty_ratios"]),
        "uncertainty_reasons": list(flags["uncertainty_reasons"]),
        "Es": float(diagnostics["Es"]),
        "Fs": float(diagnostics["Fs"]),
        "Fsmax": float(diagnostics["Fsmax"]),
        "Ecrit": bool(flags["Ecrit"]),
        "Fcrit": bool(flags["Fcrit"]),
        "Fmcrit": bool(flags["Fmcrit"]),
        "Escut": float(sampler_config["Escut"]),
        "Fscut": float(sampler_config["Fscut"]),
        "distmin": float(distance),
        "distcut": float(sampler_config.get("distcut", 1.2)),
        "selected_state": selected_state_value,
        "temps": [float(value) for value in temperature_history],
        **temperature_parameters,
    }


def run_alchemi_sampling(
    molecule_objects: list[MoleculesObject],
    sampler_config: dict[str, Any],
    model: Any,
    *,
    device: Any = None,
    runner_factory: Callable[..., Any] | None = None,
) -> list[MoleculesObject]:
    """Run one strict-full ALCHEMI batch using stop or continue selection."""

    mode, selected_state_value = _validate_sampling_inputs(molecule_objects, sampler_config)
    policy = str(sampler_config.get("uncertainty_policy", "stop")).strip().lower()
    backend = dict(sampler_config["alchemi_baoab"])
    random_seed = int(backend.get("random_seed", 42))
    dt = float(sampler_config["dt"])
    maxt = float(sampler_config["maxt"])
    ncheck = int(sampler_config["Ncheck"])
    min_time = float(sampler_config.get("min_time", 0.0))
    if dt <= 0 or maxt <= 0 or ncheck < 1 or min_time < 0:
        raise ValueError("dt and maxt must be positive; Ncheck >= 1 and min_time >= 0.")
    return_top_n = int(sampler_config.get("return_top_n", 1))
    if return_top_n < 1:
        raise ValueError("return_top_n must be at least one.")

    atoms_list = [molecule.get_atoms().copy() for molecule in molecule_objects]
    if sampler_config.get("translate_to_center", False):
        for atoms in atoms_list:
            atoms.set_positions(atoms.get_positions() - atoms.get_center_of_mass())
    temperature_parameters = _temperature_parameters(
        sampler_config,
        count=len(molecule_objects),
        random_seed=random_seed,
    )
    initial_temperatures = np.asarray(
        [
            annealing_schedule(0.0, maxt, row["Tamp"], row["Tper"], row["Tsrt"], row["Tend"])
            for row in temperature_parameters
        ],
        dtype=float,
    )
    friction_per_fs = float(
        sampler_config.get("friction_per_fs", DEFAULT_FRICTION_PER_FS)
    )
    if not np.isfinite(friction_per_fs) or friction_per_fs < 0:
        raise ValueError("friction_per_fs must be a finite nonnegative value.")
    if runner_factory is None:
        runner_factory = AlchemiDynamicsRunner
    runner = runner_factory(
        model=model,
        atoms_list=atoms_list,
        dt_fs=dt,
        temperature_K=initial_temperatures,
        friction_per_fs=friction_per_fs,
        random_seed=random_seed,
        device=device,
    )

    active = np.ones(len(molecule_objects), dtype=bool)
    temperature_histories: list[list[float]] = [[] for _ in molecule_objects]
    candidates: list[dict[str, Any]] = []
    n_outer = int(np.ceil((1000.0 * maxt) / (dt * ncheck)))
    # Preserve molecular MLMD's legacy clock: advance once, then label the
    # first uncertainty check and thermostat update as time zero.
    runner.run(1)
    for iteration in range(n_outer):
        if not np.any(active):
            break
        time_ps = float(iteration * ncheck * dt / 1000.0)
        runner.evaluate()
        for batch_index in np.where(active)[0]:
            index = int(batch_index)
            runner.sync_graph_to_atoms(index, atoms_list[index])
            diagnostics = dict(runner.diagnostics_for_graph(index))
            flags = uncertainty_flags(
                diagnostics,
                Escut=float(sampler_config["Escut"]),
                Fscut=float(sampler_config["Fscut"]),
            )
            distance = _minimum_distance(atoms_list[index])
            if time_ps < min_time:
                continue
            if distance < float(sampler_config.get("distcut", 1.2)):
                active[index] = False
                runner.freeze_graph(index)
                continue
            if not flags["uncertain"]:
                continue

            record = _candidate_record(
                diagnostics=diagnostics,
                flags=flags,
                molecule=molecule_objects[index],
                batch_index=index,
                step=int(runner.nsteps),
                time_ps=time_ps,
                distance=distance,
                sampler_config=sampler_config,
                policy=policy,
                temperature_parameters=temperature_parameters[index],
                temperature_history=temperature_histories[index],
                selected_state_value=selected_state_value,
            )
            candidates.append(
                {
                    "record": record,
                    "atoms": atoms_list[index].copy(),
                    "parent_metadata": dict(molecule_objects[index].get_metadata()),
                }
            )
            if policy == "stop":
                active[index] = False
                runner.freeze_graph(index)
            else:
                candidates.sort(key=_candidate_sort_key)
                del candidates[return_top_n:]

        if not np.any(active):
            break
        target_temperatures = np.asarray(
            [
                annealing_schedule(
                    time_ps,
                    maxt,
                    row["Tamp"],
                    row["Tper"],
                    row["Tsrt"],
                    row["Tend"],
                )
                for row in temperature_parameters
            ],
            dtype=float,
        )
        for index in np.where(active)[0]:
            temperature_histories[int(index)].append(
                float(target_temperatures[int(index)])
            )
        runner.set_temperature(target_temperatures)
        runner.run(ncheck)

    if policy == "stop":
        candidates.sort(key=lambda item: int(item["record"]["batch_index"]))
    else:
        candidates.sort(key=_candidate_sort_key)
        candidates = candidates[:return_top_n]

    outputs: list[MoleculesObject] = []
    for rank, candidate in enumerate(candidates):
        parent_id = str(candidate["record"]["parent_molecule_id"])
        molecule = MoleculesObject(candidate["atoms"], f"{parent_id}-cand-{rank:04d}")
        metadata = dict(candidate["parent_metadata"])
        metadata.update(candidate["record"])
        metadata.update(
            {
                "candidate_rank": int(rank),
                "candidate_count": len(candidates),
                "dynamics_backend": "alchemi_baoab",
                "model_mode": mode,
                "calculator_interface": str(
                    getattr(model, "calculator_interface", "native")
                ),
                "calculator_loader": str(
                    getattr(model, "calculator_loader", type(model).__name__)
                ),
            }
        )
        molecule.update_metadata(metadata)
        outputs.append(molecule)
    return outputs


if _ALCHEMI_IMPORT_ERROR is None:

    class ALFAlchemiCalculator(torch.nn.Module, BaseModelMixin):
        """Base calculator that applies ALF uncertainty semantics to an ensemble.

        Subclasses implement :meth:`ensemble_forward` and return raw selected-
        state energy and force contributions. Keeping this reduction here makes
        model adapters responsible only for inference and property mapping.
        """

        def __init__(
            self,
            *,
            selected_state_value: int = 0,
            well_params: dict[str, Any] | None = None,
            device: Any = None,
            model_config_template: Any = None,
            calculator_interface: str = "native",
            calculator_loader: str | None = None,
        ) -> None:
            torch.nn.Module.__init__(self)
            self.selected_state_value = int(selected_state_value)
            self.well_params = None if well_params is None else dict(well_params)
            self._device = torch.device(device) if device is not None else None
            self.calculator_interface = str(calculator_interface)
            self.calculator_loader = str(
                calculator_loader or f"{type(self).__module__}.{type(self).__name__}"
            )
            template = model_config_template
            self.model_config = ModelConfig(
                outputs=frozenset({"energy", "forces"}),
                autograd_outputs=frozenset(),
                autograd_inputs=frozenset(),
                required_inputs=(
                    frozenset() if template is None else template.required_inputs
                ),
                optional_inputs=(
                    frozenset() if template is None else template.optional_inputs
                ),
                supports_pbc=(False if template is None else template.supports_pbc),
                needs_pbc=(False if template is None else template.needs_pbc),
                neighbor_config=(None if template is None else template.neighbor_config),
                active_outputs={"energy", "forces"},
            )
            self.last_diagnostics: dict[str, Any] = {}

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

        @staticmethod
        def _batch_shape(batch) -> tuple[int, int]:
            counts = batch.num_nodes_per_graph
            if not bool(torch.all(counts == counts[0]).detach().cpu().item()):
                raise ValueError("ALCHEMI calculator batches require equal atom counts.")
            return int(batch.num_graphs), int(counts[0].detach().cpu().item())

        def _well_energy_forces(self, batch, positions_batched):
            if self.well_params is None:
                return None, None
            params = dict(self.well_params)
            r_start = float(params["r_start"])
            force_constant = float(params["force"])
            origin = torch.as_tensor(
                params.get("origin", [0.0, 0.0, 0.0]),
                dtype=positions_batched.dtype,
                device=positions_batched.device,
            ).reshape(1, 1, 3)
            relative = positions_batched - origin
            radius = torch.linalg.vector_norm(relative, dim=-1)
            depth = torch.clamp(radius - r_start, min=0.0)
            if bool(params.get("mass_weighted", True)):
                weights = batch.atomic_masses.reshape_as(radius)
            else:
                weights = torch.ones_like(radius)
            well_energy = torch.sum(weights * depth * force_constant, dim=1, keepdim=True)
            unit_vectors = relative / torch.clamp(radius, min=1.0e-12).unsqueeze(-1)
            active = (radius > r_start).to(positions_batched.dtype).unsqueeze(-1)
            well_forces = -unit_vectors * active * weights.unsqueeze(-1) * force_constant
            return well_energy, well_forces

        def ensemble_forward(self, batch) -> dict[str, Any]:
            """Return raw ensemble contributions for the current batch."""

            raise NotImplementedError

        @staticmethod
        def _normalize_contributions(batch, energy_values, force_values):
            num_graphs = int(batch.num_graphs)
            total_atoms = int(batch.positions.shape[0])
            energy = torch.as_tensor(
                energy_values,
                dtype=batch.positions.dtype,
                device=batch.positions.device,
            )
            forces = torch.as_tensor(
                force_values,
                dtype=batch.positions.dtype,
                device=batch.positions.device,
            )
            if energy.ndim < 2 or int(energy.shape[1]) != num_graphs:
                raise ValueError(
                    "ALCHEMI energy contributions must have shape [models, batch, ...]."
                )
            model_count = int(energy.shape[0])
            energy = energy.reshape(model_count, num_graphs, -1).sum(dim=2)
            if forces.ndim < 3 or int(forces.shape[0]) != model_count:
                raise ValueError(
                    "ALCHEMI force contributions must use the same model count as energies."
                )
            if int(forces.numel()) != model_count * total_atoms * 3:
                raise ValueError(
                    "ALCHEMI force contributions must have shape [models, total_atoms, 3]."
                )
            forces = forces.reshape(model_count, total_atoms, 3)
            if model_count < 1:
                raise ValueError("At least one calculator contribution is required.")
            return energy, forces

        @staticmethod
        def _reduce_contributions(batch, energy, forces) -> dict[str, Any]:
            num_graphs = int(batch.num_graphs)
            num_atoms = int(batch.num_nodes_per_graph[0].detach().cpu().item())
            model_count = int(energy.shape[0])
            energy_mean = torch.mean(energy, dim=0)
            force_mean = torch.mean(forces, dim=0)
            if model_count == 1:
                energy_std = torch.zeros_like(energy_mean)
                force_std = torch.zeros_like(force_mean)
            else:
                energy_std = torch.std(energy, dim=0, correction=0)
                force_std = torch.std(forces, dim=0, correction=0)
            force_std_batched = force_std.reshape(num_graphs, num_atoms, 3)
            return {
                "energy": energy_mean,
                "forces": force_mean,
                "energy_std": energy_std,
                "force_std": force_std_batched,
                "Fs": torch.mean(torch.abs(force_std_batched), dim=(1, 2)),
                "Fsmax": torch.amax(torch.abs(force_std_batched), dim=(1, 2)),
            }

        def forward(self, batch):
            prediction = dict(self.ensemble_forward(batch))
            energy, forces = self._normalize_contributions(
                batch,
                prediction["energy_contributions"],
                prediction["force_contributions"],
            )
            selected = self._reduce_contributions(batch, energy, forces)
            diagnostics: dict[str, Any] = {}
            state_contributions = dict(prediction.get("state_contributions") or {})
            state_contributions.setdefault(
                self.selected_state_value,
                {
                    "energy_contributions": energy,
                    "force_contributions": forces,
                },
            )
            for state_value, values in state_contributions.items():
                state_energy, state_forces = self._normalize_contributions(
                    batch,
                    values["energy_contributions"],
                    values["force_contributions"],
                )
                state_result = self._reduce_contributions(
                    batch, state_energy, state_forces
                )
                state = int(state_value)
                diagnostics[f"sE{state}"] = state_result["energy"]
                diagnostics[f"sE{state}_stdev"] = state_result["energy_std"]
                diagnostics[f"F{state}"] = state_result["forces"].reshape(
                    int(batch.num_graphs), -1, 3
                )
                diagnostics[f"F{state}_stdev"] = state_result["force_std"]

            uncertainty_override = prediction.get("uncertainty")
            if uncertainty_override is None:
                diagnostics["Es"] = selected["energy_std"]
                diagnostics["Fs"] = selected["Fs"]
                diagnostics["Fsmax"] = selected["Fsmax"]
            else:
                for key in ("Es", "Fs", "Fsmax"):
                    value = torch.as_tensor(
                        uncertainty_override[key],
                        dtype=batch.positions.dtype,
                        device=batch.positions.device,
                    ).reshape(-1)
                    if int(value.shape[0]) != int(batch.num_graphs):
                        raise ValueError(
                            f"Calculator-provided {key} must have one value per graph."
                        )
                    diagnostics[key] = value

            num_graphs, num_atoms = self._batch_shape(batch)
            positions_batched = batch.positions.reshape(num_graphs, num_atoms, 3)
            selected_energy = selected["energy"].reshape(num_graphs, 1)
            selected_forces = selected["forces"].reshape(num_graphs, num_atoms, 3)
            well_energy, well_forces = self._well_energy_forces(
                batch, positions_batched
            )
            if well_energy is not None:
                selected_energy = selected_energy + well_energy
                selected_forces = selected_forces + well_forces
            self.last_diagnostics = {
                key: value.detach() if hasattr(value, "detach") else value
                for key, value in diagnostics.items()
            }
            return {
                "energy": selected_energy,
                "forces": selected_forces.reshape(-1, 3),
            }

        def diagnostics_for_graph(self, graph_index: int) -> dict[str, Any]:
            values: dict[str, Any] = {}
            for key, tensor in self.last_diagnostics.items():
                value = tensor[int(graph_index)]
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                value = np.asarray(value)
                values[key] = (
                    float(value.reshape(-1)[0]) if value.size == 1 else value.copy()
                )
            return values


    class ALFNativeEnsembleModel(ALFAlchemiCalculator):
        """Combine compatible ALCHEMI-native models into an ALF ensemble."""

        def __init__(
            self,
            models,
            *,
            energy_key: str = "energy",
            force_key: str = "forces",
            selected_state_value: int = 0,
            well_params: dict[str, Any] | None = None,
            device: Any = None,
            calculator_loader: str | None = None,
        ) -> None:
            model_list = list(models)
            if not model_list:
                raise ValueError("A native ALCHEMI ensemble requires at least one model.")
            if not all(isinstance(model, BaseModelMixin) for model in model_list):
                raise TypeError("Every native ensemble member must implement BaseModelMixin.")
            input_signature = model_list[0].input_data()
            if any(model.input_data() != input_signature for model in model_list[1:]):
                raise ValueError("Native ensemble members require incompatible batch inputs.")
            super().__init__(
                selected_state_value=selected_state_value,
                well_params=well_params,
                device=device,
                model_config_template=model_list[0].model_config,
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
            for model in self.models:
                output = model(batch)
                if self.energy_key not in output or self.force_key not in output:
                    raise KeyError(
                        "Native calculator output must contain configured energy and force keys."
                    )
                energies.append(output[self.energy_key])
                forces.append(output[self.force_key])
            return {
                "energy_contributions": torch.stack(energies, dim=0),
                "force_contributions": torch.stack(forces, dim=0),
            }


    class ALFASEAlchemiModel(ALFAlchemiCalculator):
        """Compatibility wrapper for existing ALF ASE calculator loaders."""

        def __init__(
            self,
            calculators,
            *,
            model_mode: str,
            selected_state_value: int | None,
            well_params: dict[str, Any] | None = None,
            device: Any = None,
            calculator_loader: str | None = None,
            uncertainty_mode: str | None = None,
        ) -> None:
            if isinstance(calculators, (list, tuple)):
                calculator_list = list(calculators)
            else:
                calculator_list = [calculators]
            if not calculator_list or not all(
                isinstance(calculator, Calculator) for calculator in calculator_list
            ):
                raise TypeError(
                    "ase_calculator must return an ASE Calculator or a non-empty list of them."
                )
            selected = 0 if selected_state_value is None else int(selected_state_value)
            super().__init__(
                selected_state_value=selected,
                well_params=well_params,
                device=device,
                calculator_interface="ase_fallback",
                calculator_loader=calculator_loader,
            )
            mode = str(model_mode).strip().lower()
            self.energy_key = "energy" if mode == "ground_state" else f"sE{selected}"
            self.force_key = "forces" if mode == "ground_state" else f"F{selected}"
            self.calculators = calculator_list
            self.uncertainty_mode = (
                None if uncertainty_mode is None else str(uncertainty_mode).lower()
            )

        def set_device(self, device: Any) -> None:
            self._device = torch.device(device)

        def _evaluate_one(self, calculator, atoms):
            requested = [self.energy_key, self.force_key]
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
            missing = [key for key in (self.energy_key, self.force_key) if key not in calculator.results]
            if missing:
                raise KeyError(
                    "ASE fallback calculator did not provide required properties: "
                    + ", ".join(missing)
                )
            energy = float(np.asarray(calculator.results[self.energy_key]).sum())
            forces = np.asarray(calculator.results[self.force_key], dtype=float)
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
            return energy, forces, uncertainty

        def ensemble_forward(self, batch) -> dict[str, Any]:
            num_graphs, num_atoms = self._batch_shape(batch)
            positions = batch.positions.detach().cpu().numpy().reshape(
                num_graphs, num_atoms, 3
            )
            numbers = batch.atomic_numbers.detach().cpu().numpy().reshape(
                num_graphs, num_atoms
            )
            energy_members = [[] for _ in self.calculators]
            force_members = [[] for _ in self.calculators]
            uncertainty_rows = []
            for graph_index in range(num_graphs):
                atoms = Atoms(
                    numbers=numbers[graph_index],
                    positions=positions[graph_index],
                )
                graph_uncertainty = None
                for model_index, calculator in enumerate(self.calculators):
                    energy, forces, uncertainty = self._evaluate_one(calculator, atoms)
                    if forces.shape != (num_atoms, 3):
                        raise ValueError(
                            f"ASE fallback {self.force_key} must have shape "
                            f"({num_atoms}, 3)."
                        )
                    energy_members[model_index].append(energy)
                    force_members[model_index].append(forces)
                    if uncertainty is not None:
                        graph_uncertainty = uncertainty
                uncertainty_rows.append(graph_uncertainty)

            prediction = {
                "energy_contributions": torch.as_tensor(
                    energy_members,
                    dtype=batch.positions.dtype,
                    device=batch.positions.device,
                ),
                "force_contributions": torch.as_tensor(
                    np.asarray(force_members).reshape(
                        len(self.calculators), num_graphs * num_atoms, 3
                    ),
                    dtype=batch.positions.dtype,
                    device=batch.positions.device,
                ),
            }
            if all(value is not None for value in uncertainty_rows):
                prediction["uncertainty"] = {
                    key: [row[key] for row in uncertainty_rows]
                    for key in ("Es", "Fs", "Fsmax")
                }
            return prediction


    class ALFHippynnAlchemiModel(ALFAlchemiCalculator):
        """Adapt raw HIPPYNN ensemble outputs to the shared calculator base."""

        def __init__(
            self,
            *,
            ensemble_graph: Any,
            state_nodes: list[dict[str, Any]],
            selected_state_value: int,
            species_key: str,
            coordinates_key: str,
            well_params: dict[str, Any] | None = None,
            device: Any = None,
            calculator_loader: str | None = None,
        ) -> None:
            super().__init__(
                selected_state_value=selected_state_value,
                well_params=well_params,
                device=device,
                calculator_interface="native",
                calculator_loader=calculator_loader,
            )
            self.ensemble_graph = ensemble_graph
            self.state_nodes = list(state_nodes)
            self.species_key = str(species_key)
            self.coordinates_key = str(coordinates_key)
            self._make_predictor()

        def _resolve_input(self, db_name: str):
            for node in self.ensemble_graph.input_nodes:
                if str(getattr(node, "db_name", "")) == db_name or str(
                    getattr(node, "name", "")
                ) == db_name:
                    return node
            raise RuntimeError(f"Could not find HIPPYNN ensemble input {db_name!r}.")

        def _make_predictor(self) -> None:
            from hippynn.graphs import Predictor

            self._species_node = self._resolve_input(self.species_key)
            self._coordinates_node = self._resolve_input(self.coordinates_key)
            output_nodes = [
                node
                for row in self.state_nodes
                for node in (row["energy_all_node"], row["force_all_node"])
            ]
            self._predictor = Predictor(
                [self._species_node, self._coordinates_node],
                output_nodes,
                return_device=self._device,
                model_device=self._device,
                requires_grad=False,
            )

        def set_device(self, device: Any) -> None:
            self._device = torch.device(device)
            self.ensemble_graph.to(self._device)
            self._predictor.model_device = self._device
            self._predictor.return_device = self._device

        @staticmethod
        def _members_first(values, model_count: int, name: str):
            if values.ndim < 2 or int(values.shape[1]) != int(model_count):
                raise ValueError(
                    f"HIPPYNN {name} .all output must place the model axis second."
                )
            return torch.movedim(values, 1, 0)

        def ensemble_forward(self, batch) -> dict[str, Any]:
            positions = batch.positions.detach()
            species = batch.atomic_numbers.detach()
            num_graphs, num_atoms = self._batch_shape(batch)
            predictions = self._predictor(
                **{
                    self.species_key: species.reshape(num_graphs, num_atoms).to(torch.long),
                    self.coordinates_key: positions.reshape(num_graphs, num_atoms, 3),
                }
            )
            states = {}
            for row in self.state_nodes:
                energy_count = int(row["energy_model_count"])
                force_count = int(row["force_model_count"])
                if energy_count != force_count:
                    raise ValueError(
                        "HIPPYNN energy and force ensembles must contain the same models."
                    )
                energy = self._members_first(
                    predictions[row["energy_all_node"]], energy_count, "energy"
                ).reshape(energy_count, num_graphs, -1).sum(dim=2)
                forces = self._members_first(
                    predictions[row["force_all_node"]], force_count, "force"
                ).reshape(force_count, num_graphs * num_atoms, 3)
                states[int(row["state"])] = {
                    "energy_contributions": energy,
                    "force_contributions": forces,
                }
            if self.selected_state_value not in states:
                raise RuntimeError(
                    f"Selected state {self.selected_state_value} was not loaded "
                    "from the HIPPYNN ensemble."
                )
            selected = states[self.selected_state_value]
            return {**selected, "state_contributions": states}


    class AlchemiDynamicsRunner:
        """Small fixed-batch wrapper around ALCHEMI's BAOAB integrator."""

        def __init__(
            self,
            *,
            model: ALFAlchemiCalculator,
            atoms_list: list[Any],
            dt_fs: float,
            temperature_K: Any,
            friction_per_fs: float,
            random_seed: int,
            device: Any,
        ) -> None:
            from nvalchemi.data import AtomicData, Batch
            from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin

            self.model = model
            self.device = torch.device(device)
            self.model.set_device(self.device)
            data_list = []
            for atoms in atoms_list:
                positions = torch.as_tensor(atoms.get_positions(), dtype=torch.float32, device=self.device)
                velocities = atoms.get_velocities()
                if velocities is None:
                    velocity_tensor = torch.zeros_like(positions)
                else:
                    velocity_tensor = torch.as_tensor(
                        np.asarray(velocities) * units.fs,
                        dtype=torch.float32,
                        device=self.device,
                    )
                data = AtomicData(
                    positions=positions,
                    atomic_numbers=torch.as_tensor(
                        atoms.get_atomic_numbers(), dtype=torch.long, device=self.device
                    ),
                    atomic_masses=torch.as_tensor(
                        atoms.get_masses(), dtype=torch.float32, device=self.device
                    ),
                    velocities=velocity_tensor,
                    forces=torch.zeros_like(positions),
                    energy=torch.zeros((1, 1), dtype=torch.float32, device=self.device),
                )
                data.add_system_property(
                    "status", torch.zeros((1,), dtype=torch.long, device=self.device)
                )
                data_list.append(data)
            self.batch = Batch.from_data_list(data_list, device=self.device)
            initial_temperature = torch.as_tensor(
                temperature_K, dtype=torch.float32, device=self.device
            ).reshape(-1)
            self.dynamics = NVTLangevin(
                model=self.model,
                dt=float(dt_fs),
                temperature=initial_temperature,
                friction=float(friction_per_fs),
                random_seed=int(random_seed),
                device_type=self.device.type,
            )
            self.nsteps = 0
            self.dynamics.compute(self.batch)

        def run(self, steps: int) -> None:
            self.dynamics.run(self.batch, n_steps=int(steps))
            self.nsteps = int(self.dynamics.step_count)

        def evaluate(self) -> None:
            self.dynamics.compute(self.batch)

        def diagnostics_for_graph(self, graph_index: int) -> dict[str, Any]:
            return self.model.diagnostics_for_graph(graph_index)

        def freeze_graph(self, graph_index: int) -> None:
            index = int(graph_index)
            self.batch.status[index] = 1
            start = int(self.batch.batch_ptr[index].detach().cpu().item())
            end = int(self.batch.batch_ptr[index + 1].detach().cpu().item())
            self.batch.velocities[start:end] = 0.0

        def set_temperature(self, temperature_K: Any) -> None:
            values = torch.as_tensor(
                temperature_K, dtype=torch.float32, device=self.device
            ).reshape(-1)
            self.dynamics._temperature_init = values
            state = getattr(self.dynamics, "_state", None)
            if state is not None:
                state.temperature.copy_(
                    values.reshape_as(state.temperature) * KB_EV_PER_K
                )

        def sync_graph_to_atoms(self, graph_index: int, atoms) -> None:
            index = int(graph_index)
            start = int(self.batch.batch_ptr[index].detach().cpu().item())
            end = int(self.batch.batch_ptr[index + 1].detach().cpu().item())
            positions = self.batch.positions[start:end].detach().cpu().numpy()
            velocities = self.batch.velocities[start:end].detach().cpu().numpy() / units.fs
            atoms.set_positions(np.asarray(positions, dtype=float))
            atoms.set_velocities(np.asarray(velocities, dtype=float))

else:

    class ALFAlchemiCalculator:  # pragma: no cover - dependency guard
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise _missing_alchemi_error(_ALCHEMI_IMPORT_ERROR)


    class ALFNativeEnsembleModel(ALFAlchemiCalculator):  # pragma: no cover
        pass


    class ALFASEAlchemiModel(ALFAlchemiCalculator):  # pragma: no cover
        pass


    class ALFHippynnAlchemiModel:  # pragma: no cover - dependency guard
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise _missing_alchemi_error(_ALCHEMI_IMPORT_ERROR)


    class AlchemiDynamicsRunner:  # pragma: no cover - dependency guard
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise _missing_alchemi_error(_ALCHEMI_IMPORT_ERROR)


def load_hippynn_alchemi_model(
    ensemble_directory: str,
    *,
    model_mode: str,
    selected_state_value: int | None,
    ML_config: dict[str, Any],
    properties_list: dict[str, Any],
    sampler_config: dict[str, Any],
    device: Any,
    **options: Any,
) -> ALFHippynnAlchemiModel:
    """Load ground- or excited-state ensemble nodes from HIPPYNN checkpoints."""

    ensure_alchemi_available()
    import hippynn

    if options:
        raise TypeError(
            "Unknown HIPPYNN ALCHEMI calculator options: "
            + ", ".join(sorted(options))
        )

    ensemble_graph, (_, output_info) = hippynn.graphs.make_ensemble(
        os.path.join(str(ensemble_directory), "model-*"), quiet=True
    )
    ensemble_graph.to(device)
    mode = str(model_mode).strip().lower()
    if mode == "ground_state":
        energy_key = str(ML_config.get("energy_key", "energy"))
        force_key = ML_config.get("force_key")
        if force_key is None:
            raise ValueError("Ground-state ALCHEMI requires ML_config.force_key.")
        table = [
            {
                "state": 0,
                "energy_db_name": energy_key,
                "force_db_name": str(force_key),
            }
        ]
        selected = 0
    elif mode == "excited_state":
        table = derive_state_property_table(properties_list, require_forces=True)
        if selected_state_value is None:
            raise ValueError("Excited-state ALCHEMI requires selected_state metadata.")
        selected = int(selected_state_value)
        if selected not in {int(row["state"]) for row in table}:
            raise ValueError(f"Selected state {selected} is not present in properties_list.")
    else:
        raise ValueError("model_mode must be 'ground_state' or 'excited_state'.")

    state_nodes = []
    for row in table:
        energy_target = ensemble_graph.node_from_name(
            f"ensemble_{row['energy_db_name']}"
        )
        force_target = ensemble_graph.node_from_name(
            f"ensemble_{row['force_db_name']}"
        )
        state_nodes.append(
            {
                **row,
                "energy_all_node": energy_target.all,
                "force_all_node": force_target.all,
                "energy_model_count": int(output_info[row["energy_db_name"]]),
                "force_model_count": int(output_info[row["force_db_name"]]),
            }
        )
    well_params = dict(sampler_config.get("MLMD_calculator_options") or {}).get(
        "well_params"
    )
    return ALFHippynnAlchemiModel(
        ensemble_graph=ensemble_graph,
        state_nodes=state_nodes,
        selected_state_value=selected,
        species_key=str(ML_config.get("species_key", "species")),
        coordinates_key=str(ML_config.get("coordinates_key", "coordinates")),
        well_params=well_params,
        device=device,
        calculator_loader=(
            "alframework.samplers.alchemi_sampling.load_hippynn_alchemi_model"
        ),
    )


def alchemi_calculator_status(sampler_config: dict[str, Any]) -> dict[str, str]:
    """Describe the configured ALCHEMI calculator path without loading it."""

    native_loader = sampler_config.get("alchemi_calculator")
    if native_loader:
        return {"interface": "native", "loader": str(native_loader)}
    ase_loader = sampler_config.get("ase_calculator")
    if ase_loader:
        return {"interface": "ase_fallback", "loader": str(ase_loader)}
    return {"interface": "unconfigured", "loader": ""}


def load_alchemi_calculator(
    *,
    sampler_config: dict[str, Any],
    ensemble_directory: str,
    model_mode: str,
    selected_state_value: int | None,
    ML_config: dict[str, Any],
    properties_list: dict[str, Any],
    device: Any,
) -> ALFAlchemiCalculator:
    """Load a native calculator or bridge an existing ASE calculator loader."""

    ensure_alchemi_available()
    well_params = dict(sampler_config.get("MLMD_calculator_options") or {}).get(
        "well_params"
    )
    native_loader_path = sampler_config.get("alchemi_calculator")
    if native_loader_path:
        loader = load_module_from_string(str(native_loader_path))
        model = loader(
            ensemble_directory,
            model_mode=model_mode,
            selected_state_value=selected_state_value,
            ML_config=ML_config,
            properties_list=properties_list,
            sampler_config=sampler_config,
            device=device,
            **dict(sampler_config.get("alchemi_calculator_options") or {}),
        )
        if not isinstance(model, ALFAlchemiCalculator):
            raise TypeError(
                "alchemi_calculator must return an ALFAlchemiCalculator instance."
            )
        model.well_params = None if well_params is None else dict(well_params)
        model.calculator_interface = "native"
        model.calculator_loader = str(native_loader_path)
        return model

    ase_loader_path = sampler_config.get("ase_calculator")
    if not ase_loader_path:
        raise ValueError(
            "ALCHEMI sampling requires either alchemi_calculator for native "
            "batched inference or ase_calculator for compatibility fallback."
        )
    loader = load_module_from_string(str(ase_loader_path))
    options = dict(sampler_config.get("ase_calculator_options") or {})
    potential_mode = sampler_config.get("use_potential_specific_code")
    if str(potential_mode).lower() == "neurochem":
        model_details = {
            "model_path": str(ensemble_directory).rstrip("/") + "/",
            "Nn": 8,
            "gpu": "0",
        }
        model_details.update(options)
        calculators = loader(model_details)
        uncertainty_mode = "neurochem"
    else:
        options.setdefault("device", str(device))
        calculators = loader(str(ensemble_directory).rstrip("/") + "/", **options)
        uncertainty_mode = None
    return ALFASEAlchemiModel(
        calculators,
        model_mode=model_mode,
        selected_state_value=selected_state_value,
        well_params=well_params,
        device=device,
        calculator_loader=str(ase_loader_path),
        uncertainty_mode=uncertainty_mode,
    )


@python_app(executors=["alf_sampler_executor"])
def alchemi_sampling_task(
    molecule_objects,
    sampler_config,
    model_path,
    current_model_id,
    gpus_per_node,
    ML_config,
    properties_list,
):
    """Parsl task for strict-full batched ALCHEMI sampling."""

    gpu_count = max(1, int(gpus_per_node))
    worker_rank = int(os.environ.get("PARSL_WORKER_RANK", "0"))
    visible_device = worker_rank % gpu_count
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_device)
    os.environ["ROCR_VISIBLE_DEVICES"] = str(visible_device)
    ensure_alchemi_available()

    backend = dict(sampler_config.get("alchemi_baoab") or {})
    allow_cpu_debug = bool(backend.get("allow_cpu_debug", False))
    strict_gpu = bool(backend.get("strict_gpu", True))
    if torch.cuda.is_available():
        # When CUDA visibility was applied before runtime initialization, the
        # selected physical GPU is remapped to cuda:0. If another dependency
        # initialized CUDA while importing the task module, select the worker's
        # explicit index from the still-visible device list instead.
        available_devices = max(1, int(torch.cuda.device_count()))
        device_index = 0 if available_devices == 1 else visible_device % available_devices
        device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        if strict_gpu and not allow_cpu_debug:
            raise RuntimeError(
                "ALCHEMI sampling requires CUDA by default. Set "
                "alchemi_baoab.allow_cpu_debug=true only for local smoke tests."
            )

    mode, selected_state_value = _validate_sampling_inputs(
        molecule_objects, sampler_config
    )
    model = load_alchemi_calculator(
        sampler_config=sampler_config,
        ensemble_directory=model_path.format(int(current_model_id)),
        model_mode=mode,
        selected_state_value=selected_state_value,
        ML_config=ML_config,
        properties_list=properties_list,
        device=device,
    )
    return run_alchemi_sampling(
        molecule_objects,
        sampler_config,
        model,
        device=device,
    )
