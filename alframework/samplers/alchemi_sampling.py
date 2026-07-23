"""Batched HIPPYNN molecular dynamics with NVIDIA ALCHEMI.

This module intentionally implements a new sampler task instead of changing
ALF's established ASE samplers.  The pure sampling policy functions remain
usable with test doubles, while imports of PyTorch, HIPPYNN, and ALCHEMI are
guarded so a base ALF installation can still import and test normally.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
from ase import units
from parsl import python_app

from alframework.tools.excited_state_tools import derive_state_property_table
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampler_batching import sampler_batch_size, selected_state
from alframework.tools.tools import annealing_schedule


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
        raise ValueError(f"{name} must be a two-value range.")
    return float(rng.uniform(float(value[0]), float(value[1])))


def _temperature_parameters(
    sampler_config: dict[str, Any],
    *,
    count: int,
    random_seed: int,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(int(random_seed))
    parameters = []
    for _ in range(int(count)):
        parameters.append(
            {
                "Tamp": _sample_range(rng, sampler_config["amp_temp"], "amp_temp"),
                "Tper": _sample_range(rng, sampler_config["per_temp"], "per_temp"),
                "Tsrt": _sample_range(rng, sampler_config["srt_temp"], "srt_temp"),
                "Tend": _sample_range(rng, sampler_config["end_temp"], "end_temp"),
            }
        )
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
    temperatures: dict[str, float],
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
        **temperatures,
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
    temperatures = _temperature_parameters(
        sampler_config,
        count=len(molecule_objects),
        random_seed=random_seed,
    )
    initial_temperatures = np.asarray(
        [
            annealing_schedule(0.0, maxt, row["Tamp"], row["Tper"], row["Tsrt"], row["Tend"])
            for row in temperatures
        ],
        dtype=float,
    )
    if runner_factory is None:
        runner_factory = AlchemiDynamicsRunner
    runner = runner_factory(
        model=model,
        atoms_list=atoms_list,
        dt_fs=dt,
        temperature_K=initial_temperatures,
        friction_per_fs=float(sampler_config.get("friction_per_fs", DEFAULT_FRICTION_PER_FS)),
        random_seed=random_seed,
        device=device,
    )

    active = np.ones(len(molecule_objects), dtype=bool)
    candidates: list[dict[str, Any]] = []
    n_outer = int(np.ceil((1000.0 * maxt) / (dt * ncheck)))
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
                temperatures=temperatures[index],
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
                for row in temperatures
            ],
            dtype=float,
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
            }
        )
        molecule.update_metadata(metadata)
        outputs.append(molecule)
    return outputs


if _ALCHEMI_IMPORT_ERROR is None:

    class ALFHippynnAlchemiModel(torch.nn.Module, BaseModelMixin):
        """Adapt a HIPPYNN ensemble graph to ALCHEMI's model contract."""

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
        ) -> None:
            torch.nn.Module.__init__(self)
            self.ensemble_graph = ensemble_graph
            self.state_nodes = list(state_nodes)
            self.selected_state_value = int(selected_state_value)
            self.species_key = str(species_key)
            self.coordinates_key = str(coordinates_key)
            self.well_params = None if well_params is None else dict(well_params)
            self._device = torch.device(device) if device is not None else None
            self.model_config = ModelConfig(
                outputs=frozenset({"energy", "forces"}),
                autograd_outputs=frozenset(),
                autograd_inputs=frozenset(),
                active_outputs=frozenset({"energy", "forces"}),
            )
            self.last_diagnostics: dict[str, Any] = {}
            self._make_predictor()

        @property
        def embedding_shapes(self) -> dict[str, Any]:
            return {}

        def compute_embeddings(self, data, **kwargs):
            del kwargs
            return data

        def direct_derivative_keys(self) -> set[str]:
            return {"forces"}

        def _resolve_input(self, db_name: str):
            for node in self.ensemble_graph.input_nodes:
                if str(getattr(node, "db_name", "")) == db_name or str(getattr(node, "name", "")) == db_name:
                    return node
            raise RuntimeError(f"Could not find HIPPYNN ensemble input {db_name!r}.")

        def _make_predictor(self) -> None:
            from hippynn.graphs import Predictor

            self._species_node = self._resolve_input(self.species_key)
            self._coordinates_node = self._resolve_input(self.coordinates_key)
            output_nodes = []
            for row in self.state_nodes:
                output_nodes.extend(
                    [
                        row["energy_mean_node"],
                        row["energy_std_node"],
                        row["force_mean_node"],
                        row["force_std_node"],
                    ]
                )
            self._predictor = Predictor(
                [self._species_node, self._coordinates_node],
                output_nodes,
                return_device=self._device,
                model_device=self._device,
                requires_grad=False,
            )

        def set_device(self, device: Any) -> None:
            self._device = torch.device(device)
            self._predictor.model_device = self._device
            self._predictor.return_device = self._device

        @staticmethod
        def _batch_shape(batch, positions) -> tuple[int, int]:
            counts = batch.num_nodes_per_graph
            if not bool(torch.all(counts == counts[0]).detach().cpu().item()):
                raise ValueError("HIPPYNN ALCHEMI batches require equal atom counts.")
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

        @staticmethod
        def _population_std(sample_std, model_count: int):
            count = int(model_count)
            if count <= 1:
                return torch.zeros_like(sample_std)
            return sample_std * np.sqrt(float(count - 1) / float(count))

        def forward(self, batch):
            positions = batch.positions.detach()
            species = batch.atomic_numbers.detach()
            num_graphs, num_atoms = self._batch_shape(batch, positions)
            positions_batched = positions.reshape(num_graphs, num_atoms, 3)
            predictions = self._predictor(
                **{
                    self.species_key: species.reshape(num_graphs, num_atoms).to(torch.long),
                    self.coordinates_key: positions_batched,
                }
            )

            diagnostics: dict[str, Any] = {}
            selected_energy = None
            selected_forces = None
            for row in self.state_nodes:
                state = int(row["state"])
                energy_mean = predictions[row["energy_mean_node"]].reshape(num_graphs, -1).sum(dim=1)
                energy_std = self._population_std(
                    predictions[row["energy_std_node"]].reshape(num_graphs, -1).sum(dim=1),
                    int(row["energy_model_count"]),
                )
                force_mean = predictions[row["force_mean_node"]].reshape(num_graphs, num_atoms, 3)
                force_std = self._population_std(
                    predictions[row["force_std_node"]].reshape(num_graphs, num_atoms, 3),
                    int(row["force_model_count"]),
                )
                diagnostics[f"sE{state}"] = energy_mean
                diagnostics[f"sE{state}_stdev"] = energy_std
                diagnostics[f"F{state}"] = force_mean
                diagnostics[f"F{state}_stdev"] = force_std
                if state == self.selected_state_value:
                    selected_energy = energy_mean
                    selected_forces = force_mean
                    diagnostics["Es"] = energy_std
                    diagnostics["Fs"] = torch.mean(torch.abs(force_std), dim=(1, 2))
                    diagnostics["Fsmax"] = torch.amax(torch.abs(force_std), dim=(1, 2))

            if selected_energy is None or selected_forces is None:
                raise RuntimeError(
                    f"Selected state {self.selected_state_value} was not loaded from the HIPPYNN ensemble."
                )
            well_energy, well_forces = self._well_energy_forces(batch, positions_batched)
            if well_energy is not None:
                selected_energy = selected_energy.reshape(num_graphs, 1) + well_energy
                selected_forces = selected_forces + well_forces
            else:
                selected_energy = selected_energy.reshape(num_graphs, 1)
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
                value = tensor[int(graph_index)].detach().cpu().numpy()
                values[key] = float(np.asarray(value).reshape(-1)[0]) if np.asarray(value).size == 1 else value.copy()
            return values


    class AlchemiDynamicsRunner:
        """Small fixed-batch wrapper around ALCHEMI's BAOAB integrator."""

        def __init__(
            self,
            *,
            model: ALFHippynnAlchemiModel,
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
) -> ALFHippynnAlchemiModel:
    """Load ground- or excited-state ensemble nodes from HIPPYNN checkpoints."""

    ensure_alchemi_available()
    import hippynn

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
                "energy_mean_node": energy_target.mean,
                "energy_std_node": energy_target.std,
                "force_mean_node": force_target.mean,
                "force_std_node": force_target.std,
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
    model = load_hippynn_alchemi_model(
        model_path.format(int(current_model_id)),
        model_mode=mode,
        selected_state_value=selected_state_value,
        ML_config=ML_config,
        properties_list=properties_list,
        sampler_config=sampler_config,
        device=device,
    )
    return run_alchemi_sampling(
        molecule_objects,
        sampler_config,
        model,
        device=device,
    )
