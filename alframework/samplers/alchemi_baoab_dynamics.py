from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


KB_EV_PER_K = 8.617333262145e-5


def _missing_alchemi_error(exc: BaseException | None = None) -> ImportError:
    detail = "" if exc is None else f" Original import error: {type(exc).__name__}: {exc}"
    return ImportError(
        "dynamics_backend='alchemi_baoab' requires NVIDIA ALCHEMI. "
        "Install ALF with the gpu_dynamics extra or install nvalchemi-toolkit."
        + detail
    )


def ensure_alchemi_available() -> None:
    try:
        import nvalchemi  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised with fakes in sampler tests
        raise _missing_alchemi_error(exc) from exc


try:  # Keep normal ALF imports independent of nvalchemi.
    import torch
    from nvalchemi.models.base import BaseModelMixin, ModelConfig
except Exception as exc:  # pragma: no cover - import guard
    torch = None
    BaseModelMixin = object
    ModelConfig = None
    _alchemi_model_import_error = exc
else:
    _alchemi_model_import_error = None


def _as_tensor(value: Any):
    if torch is None:
        raise _missing_alchemi_error(_alchemi_model_import_error)
    return value if hasattr(value, "detach") else torch.as_tensor(value)


def _node_label(node: Any) -> str:
    for attr in ("db_name", "name"):
        value = getattr(node, attr, None)
        if value is not None:
            return str(value)
    return str(node)


def _prediction_value(predictions: Any, node: Any, fallback_key: str):
    keys = [node, _node_label(node), fallback_key]
    if isinstance(predictions, dict):
        for key in keys:
            try:
                return predictions[key]
            except (KeyError, TypeError):
                continue
    for key in keys:
        if isinstance(key, str) and hasattr(predictions, key):
            return getattr(predictions, key)
    if isinstance(predictions, (tuple, list)):
        # Hippynn Predictor usually returns an ordered tuple/list. The caller
        # passes nodes in energy + extras order and handles positional unpacking.
        raise KeyError
    raise KeyError(f"Could not find prediction output for {fallback_key!r} ({_node_label(node)!r}).")


def _prediction_tensor(value: Any, reference, *, dtype=None):
    if torch is None:
        raise _missing_alchemi_error(_alchemi_model_import_error)
    tensor = value if hasattr(value, "detach") else torch.as_tensor(value)
    target_dtype = dtype
    if target_dtype is None and torch.is_floating_point(tensor):
        target_dtype = reference.dtype
    if target_dtype is not None and torch.is_floating_point(tensor):
        return tensor.to(device=reference.device, dtype=target_dtype)
    return tensor.to(device=reference.device)


def _tensor_to_numpy(value: Any):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value)
    if array.size == 1:
        return float(array.reshape(-1)[0])
    return array.copy()


def _batch_get(batch: Any, key: str):
    if hasattr(batch, key):
        return getattr(batch, key)
    try:
        return batch[key]
    except Exception as exc:
        raise AttributeError(f"ALCHEMI batch is missing required field {key!r}.") from exc


def _batch_set(batch: Any, key: str, value: Any) -> None:
    try:
        setattr(batch, key, value)
    except Exception:
        batch[key] = value


if _alchemi_model_import_error is None:

    class ALFExcitedStateAlchemiModel(torch.nn.Module, BaseModelMixin):
        """Hippynn excited-state ensemble adapter for ALCHEMI dynamics.

        The sampler runs one molecule per task, so this adapter intentionally
        supports a single graph in the ALCHEMI batch for v1.
        """

        def __init__(
            self,
            *,
            ensemble_graph: Any,
            energy_node: Any,
            extra_properties: dict[str, Any] | None = None,
            force_node: Any | None = None,
            species_key: str = "species",
            coordinates_key: str = "coordinates",
            offset_eV: float = 0.0,
            device: Any | None = None,
            predictor_factory: Callable[..., Any] | None = None,
        ) -> None:
            torch.nn.Module.__init__(self)
            if ModelConfig is None:
                raise _missing_alchemi_error(_alchemi_model_import_error)
            self.ensemble_graph = ensemble_graph
            self.species_key = str(species_key)
            self.coordinates_key = str(coordinates_key)
            self.offset_eV = float(offset_eV)
            self._device = torch.device(device) if device is not None else None
            self.extra_properties = dict(extra_properties or {})
            self._predictor_factory = predictor_factory
            self._input_nodes = self._resolve_input_nodes()
            self.model_config = ModelConfig(
                outputs=frozenset({"energy", "forces"}),
                autograd_outputs=frozenset(),
                autograd_inputs=frozenset(),
                active_outputs=frozenset({"energy", "forces"}),
            )
            self.last_results: dict[str, Any] = {}
            self._mode: dict[str, Any] = {}
            self.set_energy_node(energy_node, force_node=force_node)

        def set_device(self, device: Any) -> None:
            self._device = torch.device(device)
            predictor = getattr(self, "_predictor", None)
            if predictor is not None and hasattr(predictor, "to"):
                predictor.to(self._device)

        @property
        def embedding_shapes(self) -> dict[str, Any]:
            return {}

        def direct_derivative_keys(self) -> set[str]:
            return {"forces"}

        def compute_embeddings(self, data):
            return {}

        def _resolve_input_nodes(self) -> list[Any]:
            nodes: list[Any] = []
            for node_name in (self.species_key, self.coordinates_key):
                try:
                    nodes.append(self.ensemble_graph.node_from_name(node_name))
                except Exception as exc:
                    raise RuntimeError(
                        "Could not resolve Hippynn input node "
                        f"{node_name!r} for ALCHEMI BAOAB dynamics. "
                        "Set alchemi_baoab.species_key and alchemi_baoab.coordinates_key "
                        "if the model used non-default input names."
                    ) from exc
            return nodes

        def _set_output_nodes(self, nodes: list[Any]) -> None:
            seen: set[str] = set()
            self._output_nodes = []
            for node in [*nodes, *self.extra_properties.values()]:
                label = _node_label(node)
                if label in seen:
                    continue
                seen.add(label)
                self._output_nodes.append(node)
            if self._predictor_factory is None:
                from hippynn.graphs import Predictor

                self._predictor = Predictor(self._input_nodes, self._output_nodes)
            else:
                self._predictor = self._predictor_factory(self._input_nodes, self._output_nodes)
            if self._device is not None and hasattr(self._predictor, "to"):
                self._predictor.to(self._device)

        def set_energy_node(
            self,
            energy_node: Any,
            *,
            force_node: Any | None = None,
            offset_eV: float | None = None,
        ) -> None:
            if offset_eV is not None:
                self.offset_eV = float(offset_eV)
            self.energy_node = energy_node
            self.force_node = force_node
            self._mode = {"name": "direct"}
            nodes = [self.energy_node]
            if self.force_node is not None:
                nodes.append(self.force_node)
            self._set_output_nodes(nodes)

        def set_lcm_gap_mode(
            self,
            *,
            lower_energy_node: Any,
            upper_energy_node: Any,
            lower_force_node: Any,
            upper_force_node: Any,
            gap_node: Any | None,
            sigma: float,
            alpha_eV: float,
        ) -> None:
            self._mode = {
                "name": "lcm_gap",
                "lower_energy_node": lower_energy_node,
                "upper_energy_node": upper_energy_node,
                "lower_force_node": lower_force_node,
                "upper_force_node": upper_force_node,
                "gap_node": gap_node,
                "sigma": float(sigma),
                "alpha_eV": float(alpha_eV),
            }
            nodes = [lower_energy_node, upper_energy_node, lower_force_node, upper_force_node]
            if gap_node is not None:
                nodes.append(gap_node)
            self._set_output_nodes(nodes)

        def _batch_shape(self, batch, positions) -> tuple[int, int]:
            num_graphs = int(getattr(batch, "num_graphs", 1))
            if num_graphs <= 0:
                num_graphs = 1
            if positions.shape[0] % num_graphs != 0:
                raise ValueError(
                    "ALCHEMI BAOAB batching requires every graph in the batch to have "
                    "the same number of atoms."
                )
            return num_graphs, int(positions.shape[0] // num_graphs)

        def _predict(self, species, positions, *, batch=None):
            num_graphs, num_atoms = self._batch_shape(batch, positions) if batch is not None else (1, positions.shape[0])
            kwargs = {
                self.species_key: species.reshape(num_graphs, num_atoms).to(torch.long),
                self.coordinates_key: positions.reshape(num_graphs, num_atoms, 3),
            }
            return self._predictor(**kwargs)

        def forward(self, batch):
            positions = _batch_get(batch, "positions").detach().clone().requires_grad_(True)
            species = _batch_get(batch, "atomic_numbers").detach()
            num_graphs, num_atoms = self._batch_shape(batch, positions)
            positions_batched = positions.reshape(num_graphs, num_atoms, 3)
            predictions = self._predict(species, positions, batch=batch)

            if isinstance(predictions, (tuple, list)):
                raise RuntimeError("ALCHEMI BAOAB Hippynn adapter requires dict-like Predictor outputs.")
            else:
                extra_values = [
                    _prediction_tensor(_prediction_value(predictions, node, key), positions_batched)
                    for key, node in self.extra_properties.items()
                ]

            if self._mode.get("name") == "lcm_gap":
                lower_energy = _prediction_tensor(
                    _prediction_value(predictions, self._mode["lower_energy_node"], "lower_energy"),
                    positions_batched,
                )
                upper_energy = _prediction_tensor(
                    _prediction_value(predictions, self._mode["upper_energy_node"], "upper_energy"),
                    positions_batched,
                )
                lower_forces = _prediction_tensor(
                    _prediction_value(predictions, self._mode["lower_force_node"], "lower_forces"),
                    positions_batched,
                )
                upper_forces = _prediction_tensor(
                    _prediction_value(predictions, self._mode["upper_force_node"], "upper_forces"),
                    positions_batched,
                )
                if self._mode["gap_node"] is None:
                    gap = upper_energy - lower_energy
                else:
                    gap = _prediction_tensor(
                        _prediction_value(predictions, self._mode["gap_node"], "gap"),
                        positions_batched,
                    )
                smooth_abs_gap = ((gap * gap) + 1.0e-12) ** 0.5
                denom = smooth_abs_gap + float(self._mode["alpha_eV"])
                sigma = float(self._mode["sigma"])
                energy = 0.5 * (lower_energy + upper_energy) + sigma * gap * gap / denom
                gap_derivative = sigma * (
                    (2.0 * gap * denom) - (gap * gap * gap / smooth_abs_gap)
                ) / (denom * denom)
                lower_forces = lower_forces.reshape_as(positions_batched)
                upper_forces = upper_forces.reshape_as(positions_batched)
                forces = 0.5 * (lower_forces + upper_forces) + gap_derivative.reshape(num_graphs, 1, 1) * (
                    upper_forces - lower_forces
                )
            else:
                energy = _prediction_tensor(_prediction_value(predictions, self.energy_node, "energy"), positions_batched)
                if self.force_node is None:
                    if not getattr(energy, "requires_grad", False):
                        raise RuntimeError(
                            "ALCHEMI BAOAB could not compute forces from the selected Hippynn energy because "
                            f"{_node_label(self.energy_node)!r} is detached and no force node was configured. "
                            "Pass the corresponding ensemble force mean node for selected-state dynamics."
                        )
                    energy_for_grad = energy.reshape(-1).sum()
                    forces = -torch.autograd.grad(energy_for_grad, positions, create_graph=False, retain_graph=False)[0]
                    forces = forces.reshape_as(positions_batched)
                else:
                    forces = _prediction_tensor(
                        _prediction_value(predictions, self.force_node, "forces"),
                        positions_batched,
                    ).reshape_as(positions_batched)

            energy = energy.reshape(num_graphs, -1).sum(dim=1, keepdim=True) + torch.as_tensor(
                self.offset_eV,
                dtype=positions_batched.dtype,
                device=positions_batched.device,
            )
            forces = forces.reshape(-1, 3)

            results: dict[str, Any] = {
                "energy": energy.reshape(num_graphs, 1),
                "forces": forces,
            }
            for key, value in zip(self.extra_properties, extra_values):
                results[key] = value
            self.last_results = {key: value.detach() if hasattr(value, "detach") else value for key, value in results.items()}
            return {"energy": results["energy"], "forces": results["forces"]}

        def results_from_batch(self, batch) -> dict[str, Any]:
            self.forward(batch)
            return {key: _tensor_to_numpy(value) for key, value in self.last_results.items()}

        def force_norm_for_energy(self, batch, energy_node: Any) -> float:
            original = self.energy_node
            original_offset = self.offset_eV
            try:
                self.set_energy_node(energy_node, offset_eV=0.0)
                self.forward(batch)
                forces = self.last_results["forces"]
                return float(torch.sqrt(torch.sum(forces * forces)).detach().cpu().item())
            finally:
                self.set_energy_node(original, offset_eV=original_offset)

else:

    class ALFExcitedStateAlchemiModel:  # pragma: no cover - simple import guard
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise _missing_alchemi_error(_alchemi_model_import_error)


@dataclass
class AlchemiBackendConfig:
    random_seed: int
    allow_cpu_debug: bool = False
    strict_gpu: bool = True
    scalar_sync_policy: str = "control_only"
    species_key: str = "species"
    coordinates_key: str = "coordinates"
    batch_size: int = 1
    allow_partial_batches: bool = False
    require_same_selected_state: bool = True
    batched_gap_switch_policy: str = "global_lowest_gap_trigger"


def alchemi_config_from_sampler(
    sampler_config: dict[str, Any],
    *,
    default_seed: int,
) -> AlchemiBackendConfig:
    raw = dict(sampler_config.get("alchemi_baoab") or {})
    seed = int(raw.get("random_seed", default_seed))
    return AlchemiBackendConfig(
        random_seed=seed,
        allow_cpu_debug=bool(raw.get("allow_cpu_debug", False)),
        strict_gpu=bool(raw.get("strict_gpu", True)),
        scalar_sync_policy=str(raw.get("scalar_sync_policy", "control_only")),
        species_key=str(raw.get("species_key", "species")),
        coordinates_key=str(raw.get("coordinates_key", "coordinates")),
        batch_size=max(1, int(raw.get("batch_size", 1))),
        allow_partial_batches=bool(raw.get("allow_partial_batches", False)),
        require_same_selected_state=bool(raw.get("require_same_selected_state", True)),
        batched_gap_switch_policy=str(raw.get("batched_gap_switch_policy", "global_lowest_gap_trigger")),
    )


def validate_alchemi_sampler_support(atoms, feed: dict[str, Any], device, config: AlchemiBackendConfig) -> None:
    if config.scalar_sync_policy != "control_only":
        raise ValueError(
            "dynamics_backend='alchemi_baoab' currently supports only "
            "alchemi_baoab.scalar_sync_policy='control_only'."
        )
    if bool(np.any(atoms.get_pbc())):
        raise NotImplementedError("dynamics_backend='alchemi_baoab' currently supports only non-periodic sampling.")
    if feed.get("Rend") is not None:
        raise NotImplementedError("dynamics_backend='alchemi_baoab' does not yet support density scaling.")
    if config.strict_gpu and str(getattr(device, "type", device)) != "cuda" and not config.allow_cpu_debug:
        raise RuntimeError(
            "dynamics_backend='alchemi_baoab' requires a CUDA device by default. "
            "Set alchemi_baoab.allow_cpu_debug=True only for local CPU debugging."
        )


def build_alchemi_batch(atoms, *, device, dtype=None):
    return build_alchemi_batch_from_atoms_list([atoms], device=device, dtype=dtype)


def build_alchemi_batch_from_atoms_list(atoms_list, *, device, dtype=None):
    ensure_alchemi_available()
    if torch is None:
        raise _missing_alchemi_error(_alchemi_model_import_error)
    from nvalchemi.data import AtomicData, Batch

    dtype = dtype or torch.float32
    data_list = []
    for atoms in atoms_list:
        positions = torch.as_tensor(atoms.get_positions(), dtype=dtype, device=device)
        atomic_numbers = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
        masses = torch.as_tensor(atoms.get_masses(), dtype=dtype, device=device).reshape(-1)
        velocities_np = atoms.get_velocities()
        if velocities_np is None:
            velocities = torch.zeros_like(positions)
        else:
            velocities = torch.as_tensor(velocities_np, dtype=dtype, device=device)
        data_list.append(
            AtomicData(
                positions=positions,
                atomic_numbers=atomic_numbers,
                atomic_masses=masses,
                velocities=velocities,
                forces=torch.zeros_like(positions),
                energy=torch.zeros((1, 1), dtype=dtype, device=device),
            )
        )
    batch = Batch.from_data_list(data_list, device=device)
    return batch


def _graph_slice(batch: Any, graph_index: int) -> slice:
    ptr = _as_tensor(getattr(batch, "batch_ptr"))
    start = int(ptr[int(graph_index)].detach().cpu().item())
    end = int(ptr[int(graph_index) + 1].detach().cpu().item())
    return slice(start, end)


def _result_for_graph(value: Any, batch, graph_index: int):
    tensor = _as_tensor(value)
    if tensor.ndim >= 1 and tensor.shape[0] == int(getattr(batch, "num_nodes", -1)):
        tensor = tensor[_graph_slice(batch, graph_index)]
    elif tensor.ndim >= 1 and tensor.shape[0] == int(getattr(batch, "num_graphs", 1)):
        tensor = tensor[int(graph_index)]
    return _tensor_to_numpy(tensor)


class AlchemiBaoabRunner:
    def __init__(
        self,
        *,
        model: ALFExcitedStateAlchemiModel,
        batch,
        dt_fs: float,
        temperature_K: float,
        friction_per_fs: float,
        random_seed: int,
        device,
    ) -> None:
        ensure_alchemi_available()
        from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin

        self.model = model
        if hasattr(self.model, "set_device"):
            self.model.set_device(device)
        self.batch = batch
        self.nsteps = 0
        self._callbacks: list[tuple[Callable[[], None], int]] = []
        self._frozen_graphs: dict[int, dict[str, Any]] = {}
        self._device = device
        initial_temperature = torch.as_tensor(temperature_K, dtype=torch.float32, device=device).reshape(-1)
        self.dynamics = NVTLangevin(
            model,
            dt=float(dt_fs),
            temperature=float(initial_temperature[0].detach().cpu().item()),
            friction=float(friction_per_fs),
            random_seed=int(random_seed),
            device_type=str(getattr(device, "type", device)),
        )
        self.results: dict[str, Any] = {}
        self.set_temperature(temperature_K=temperature_K)
        self.evaluate_results()

    def attach(self, callback: Callable[[], None], interval: int = 1) -> None:
        self._callbacks.append((callback, max(1, int(interval))))

    def set_temperature(self, *, temperature_K: float) -> None:
        value = temperature_K
        scalar_value = None
        try:
            scalar_value = float(value)
        except (TypeError, ValueError):
            scalar_value = None
        if hasattr(self.dynamics, "_temperature_init"):
            self.dynamics._temperature_init = scalar_value if scalar_value is not None else value
        state = getattr(self.dynamics, "_state", None)
        if state is not None and hasattr(state, "temperature"):
            temperature = state.temperature
            temperature_values = torch.as_tensor(
                value,
                dtype=temperature.dtype,
                device=temperature.device,
            )
            if temperature_values.ndim == 0:
                temperature_values = temperature_values.expand_as(temperature)
            else:
                temperature_values = temperature_values.reshape_as(temperature)
            temperature[...] = temperature_values * KB_EV_PER_K

    def run(self, steps: int) -> None:
        for _ in range(int(steps)):
            self.dynamics.run(self.batch, n_steps=1)
            self._restore_frozen_graphs()
            self.nsteps = int(getattr(self.dynamics, "step_count", self.nsteps + 1))
            for callback, interval in self._callbacks:
                if self.nsteps % interval == 0:
                    callback()

    def freeze_graph(self, graph_index: int, *, positions=None) -> None:
        """Freeze one graph at a known-valid geometry inside a shared batch."""

        graph_index = int(graph_index)
        atom_slice = _graph_slice(self.batch, graph_index)
        batch_positions = _as_tensor(_batch_get(self.batch, "positions"))
        if positions is not None:
            replacement = torch.as_tensor(
                positions,
                dtype=batch_positions.dtype,
                device=batch_positions.device,
            ).reshape_as(batch_positions[atom_slice])
            batch_positions[atom_slice] = replacement
        frozen_positions = batch_positions[atom_slice].detach().clone()
        try:
            batch_velocities = _as_tensor(_batch_get(self.batch, "velocities"))
            batch_velocities[atom_slice] = 0.0
            frozen_velocities = torch.zeros_like(batch_velocities[atom_slice])
        except Exception:
            frozen_velocities = None
        self._frozen_graphs[graph_index] = {
            "positions": frozen_positions,
            "velocities": frozen_velocities,
        }
        self._restore_frozen_graphs()

    def _restore_frozen_graphs(self) -> None:
        if not self._frozen_graphs:
            return
        batch_positions = _as_tensor(_batch_get(self.batch, "positions"))
        try:
            batch_velocities = _as_tensor(_batch_get(self.batch, "velocities"))
        except Exception:
            batch_velocities = None
        for graph_index, frozen in self._frozen_graphs.items():
            atom_slice = _graph_slice(self.batch, graph_index)
            batch_positions[atom_slice] = frozen["positions"]
            if batch_velocities is not None and frozen["velocities"] is not None:
                batch_velocities[atom_slice] = frozen["velocities"]

    def evaluate_results(self) -> dict[str, Any]:
        if hasattr(self.dynamics, "compute"):
            self.dynamics.compute(self.batch)
        else:  # pragma: no cover - compatibility with simple fakes
            outputs = self.model(self.batch)
            for key, value in outputs.items():
                _batch_set(self.batch, key, value)
        self.results = {key: _tensor_to_numpy(value) for key, value in self.model.last_results.items()}
        return dict(self.results)

    def sync_to_atoms(self, atoms) -> None:
        self.sync_graph_to_atoms(0, atoms)

    def results_for_graph(self, graph_index: int) -> dict[str, Any]:
        return {
            key: _result_for_graph(value, self.batch, int(graph_index))
            for key, value in self.model.last_results.items()
        }

    def sync_graph_to_atoms(self, graph_index: int, atoms) -> None:
        atom_slice = _graph_slice(self.batch, int(graph_index))
        positions = _tensor_to_numpy(_batch_get(self.batch, "positions")[atom_slice])
        atoms.set_positions(np.asarray(positions, dtype=float))
        try:
            velocities = _tensor_to_numpy(_batch_get(self.batch, "velocities")[atom_slice])
            atoms.set_velocities(np.asarray(velocities, dtype=float))
        except Exception:
            pass

    def kinetic_energy_eV(self) -> float:
        masses = _as_tensor(_batch_get(self.batch, "atomic_masses")).reshape(-1, 1)
        velocities = _as_tensor(_batch_get(self.batch, "velocities"))
        per_atom = 0.5 * torch.sum(masses * velocities * velocities, dim=1)
        num_graphs = int(getattr(self.batch, "num_graphs", 1))
        if num_graphs == 1:
            return float(torch.sum(per_atom).detach().cpu().item())
        values = torch.zeros(num_graphs, dtype=per_atom.dtype, device=per_atom.device)
        values.scatter_add_(0, _as_tensor(getattr(self.batch, "batch_idx")).to(torch.long), per_atom)
        return _tensor_to_numpy(values)

    def temperature_K(self) -> float:
        kinetic = self.kinetic_energy_eV()
        num_graphs = int(getattr(self.batch, "num_graphs", 1))
        if num_graphs == 1:
            masses = _as_tensor(_batch_get(self.batch, "atomic_masses")).reshape(-1, 1)
            dof = max(1, int(3 * masses.shape[0]))
            return float((2.0 * float(kinetic)) / (dof * KB_EV_PER_K))
        nodes_per_graph = _tensor_to_numpy(getattr(self.batch, "num_nodes_per_graph"))
        dof = np.maximum(1, 3 * np.asarray(nodes_per_graph, dtype=float))
        return (2.0 * np.asarray(kinetic, dtype=float)) / (dof * KB_EV_PER_K)
