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
from ase.calculators.calculator import all_changes
from parsl import python_app

from alframework.tools.sampling_validation import (
    _configured_replica_candidate_limit, _configured_return_top_k,
    _physical_force_states, _validate_sampling_inputs, configured_alchemi_gap_rows,
    configured_score,
    normalize_energy_contributions, normalize_contributions, validated_batch_shape,
    validate_required_physical_states, validate_model_count,
    normalize_gap_contributions, normalize_uncertainty_override,
    validate_ase_results, validate_ase_force_shape, native_state_contributions,
    validate_native_results, hippynn_members_first, hippynn_state_table,
    validate_numeric_range, validate_temperature_period, uncertainty_cutoffs,
    validated_numpy_contributions, validate_dynamics_parameters, validate_friction,
    validate_alchemi_device, validate_gap_model, gap_seeking_settings_for_model,
    validate_score_diagnostics, validated_seeking_gap, _gap_candidate_metadata,
    validate_topology_identities, validate_native_models, validate_ase_calculators,
    validate_hippynn_options,
)
# Public configuration helpers remain importable from this sampler module.
from alframework.tools.sampling_validation import (
    SCORE_WEIGHT_KEYS as SCORE_WEIGHT_KEYS,
    configured_gap_diagnostics as configured_gap_diagnostics,
    configured_gap_seeking as configured_gap_seeking,
)
from alframework.tools.force_screen import (
    SCHEMA, SOURCE, ForceScreenAudit, force_screen_metadata, physical_force_exceeds,
    physical_force_diagnostics, lcm_force_diagnostics, propagation_force_diagnostics,
)
from alframework.tools.atom_order import canonicalize_tagged_atoms, attach_topology_ids
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.molecular_topology import (
    TopologyValidationResult,
    load_reference_topology,
    topology_metadata,
    validate_fixed_topology,
)
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


from alframework.samplers.batched_sampling import (
    calculate_uncertainty,
    uncertainty_flags,
    _sample_range,
    _temperature_parameters,
    _minimum_distance,
    _candidate_sort_key,
    candidate_score,
    _prune_candidates,
    select_gap_seeking_trigger,
    _candidate_record,
    run_batched_sampling as _run_batched_sampling,
)


def run_alchemi_sampling(molecule_objects, sampler_config, model, *, properties_list=None,
                         device=None, runner_factory=None, master_directory=None):
    """Compatibility entrypoint using the shared ALF acquisition loop."""
    return _run_batched_sampling(molecule_objects, sampler_config, model,
        properties_list=properties_list, device=device,
        runner_factory=runner_factory or AlchemiDynamicsRunner, master_directory=master_directory)


if _ALCHEMI_IMPORT_ERROR is None:
    from alframework.ml_interfaces.batched_inference import (
        BatchedEnsembleCalculator, NativeEnsembleModel, ASEEnsembleModel, HippynnEnsembleModel,
    )

    class ALFAlchemiCalculator(BatchedEnsembleCalculator, BaseModelMixin):
        """Legacy native calculator interface with ALCHEMI model configuration."""

        def __init__(self, *, model_config_template=None, **kwargs):
            super().__init__(model_config_template=model_config_template, **kwargs)
            template = model_config_template
            self.model_config = ModelConfig(
                outputs=frozenset({"energy", "forces"}),
                autograd_outputs=frozenset(), autograd_inputs=frozenset(),
                required_inputs=frozenset() if template is None else template.required_inputs,
                optional_inputs=frozenset() if template is None else template.optional_inputs,
                supports_pbc=False if template is None else template.supports_pbc,
                needs_pbc=False if template is None else template.needs_pbc,
                neighbor_config=None if template is None else template.neighbor_config,
                active_outputs={"energy", "forces"},
            )

    class ALFNativeEnsembleModel(NativeEnsembleModel, ALFAlchemiCalculator):
        """Compatibility interface for native ALCHEMI ensembles."""

        def __init__(self, models, **kwargs):
            models = list(models)
            validate_native_models(models, BaseModelMixin)
            super().__init__(models, **kwargs)

    class ALFASEAlchemiModel(ASEEnsembleModel, ALFAlchemiCalculator):
        """Compatibility interface for established ASE model loaders."""

    class ALFHippynnAlchemiModel(HippynnEnsembleModel, ALFAlchemiCalculator):
        """Compatibility interface for native HIPPYNN inference."""

    class _AlchemiModelAdapter(ALFAlchemiCalculator):
        """Expose a backend-neutral calculator to ALCHEMI's model protocol."""

        def __init__(self, model, atoms_list):
            super().__init__(selected_state_value=model.selected_state_value,
                physical_force_states=model.physical_force_states, gap_rows=model.gap_rows,
                gap_seeking_settings=model.gap_seeking_settings, well_params=model.well_params,
                device=model._device, calculator_interface=model.calculator_interface,
                calculator_loader=model.calculator_loader)
            self.wrapped = model
            self.supports_pbc = model.supports_pbc
            self.udd_bias_weight = getattr(model, "udd_bias_weight", None)
            self.udd_bias_eps = getattr(model, "udd_bias_eps", 1.e-12)
            self.force_controls = getattr(model, "force_controls", ())
            self.register_buffer("_input_cells", torch.as_tensor(
                np.stack([atoms.cell.array for atoms in atoms_list]), dtype=torch.float32))
            self.register_buffer("_input_pbc", torch.as_tensor(
                np.stack([atoms.pbc for atoms in atoms_list]), dtype=torch.bool))

        def set_device(self, device):
            super().set_device(device)
            self.wrapped.set_device(device)

        def ensemble_forward(self, batch):
            from alframework.ml_interfaces.batched_inference import PackedBatch
            cells = getattr(batch, "cell", None)
            pbc = getattr(batch, "pbc", None)
            if not hasattr(self, "_packed"):
                if cells is None:
                    cells = self._input_cells
                if pbc is None:
                    pbc = self._input_pbc
                self._packed = PackedBatch(
                    positions=batch.positions, atomic_numbers=batch.atomic_numbers,
                    atomic_masses=batch.atomic_masses, velocities=batch.velocities,
                    cell=cells, pbc=pbc, batch_ptr=batch.batch_ptr)
            else:
                # Membership is fixed during a task. Refresh engine buffers
                # while keeping the calculator on the shared packed protocol.
                self._packed.positions = batch.positions
                self._packed.velocities = batch.velocities
                if cells is not None:
                    self._packed.cell = cells
                if pbc is not None:
                    self._packed.pbc = pbc
            return self.wrapped.ensemble_forward(self._packed)

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

            if not isinstance(model, BaseModelMixin):
                model = _AlchemiModelAdapter(model, atoms_list)
            self.model = model
            self.device = torch.device(device)
            self.model.set_device(self.device)
            # HIPPYNN resolves periodic capability after cooperative base
            # initialization; update the engine's protocol to that capability.
            self.model.model_config.supports_pbc = bool(getattr(
                self.model, "supports_pbc", self.model.model_config.supports_pbc))
            data_list = []
            for atoms in atoms_list:
                if atoms.constraints:
                    raise NotImplementedError("Batched ALCHEMI dynamics does not yet support ASE constraints.")
                positions = torch.as_tensor(atoms.get_positions(), dtype=torch.float32, device=self.device)
                velocities = atoms.get_velocities()
                if velocities is None:
                    velocity_tensor = torch.zeros_like(positions)
                else:
                    velocity_tensor = torch.as_tensor(
                        # ASE and ALCHEMI kernels both use sqrt(eV/amu)
                        # velocities. Only the public dt uses femtoseconds.
                        np.asarray(velocities),
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
                    cell=(torch.as_tensor(np.asarray(atoms.cell), dtype=torch.float32,
                                          device=self.device).reshape(1, 3, 3)
                          if np.any(atoms.pbc) else None),
                    pbc=(torch.as_tensor(atoms.pbc, dtype=torch.bool,
                                         device=self.device).reshape(1, 3)
                         if np.any(atoms.pbc) else None),
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

        def set_lcm_mode(
            self,
            graph_index: int,
            *,
            pair: list[int] | tuple[int, int],
            sigma: float,
            alpha_eV: float,
        ) -> None:
            self.model.set_lcm_mode(
                graph_index,
                pair=pair,
                sigma=sigma,
                alpha_eV=alpha_eV,
            )

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
            velocities = self.batch.velocities[start:end].detach().cpu().numpy()
            atoms.set_positions(np.asarray(positions, dtype=float))
            atoms.set_velocities(np.asarray(velocities, dtype=float))
            if getattr(self.batch, "cell", None) is not None:
                atoms.set_cell(self.batch.cell[index].detach().cpu().numpy(), scale_atoms=False)
                atoms.set_pbc(self.batch.pbc[index].detach().cpu().numpy())

        def set_cell(self, graph_index, cell, *, scale_atoms=True, evaluate=True):
            index = int(graph_index)
            if bool(self.batch.status[index] >= 1):
                return
            if getattr(self.batch, "cell", None) is None:
                raise ValueError("Cell updates require a periodic batch.")
            value = torch.as_tensor(cell, dtype=self.batch.positions.dtype, device=self.device)
            if value.shape != (3, 3) or not bool(torch.isfinite(value).all()) or float(torch.linalg.det(value)) <= 0:
                raise ValueError("Cell updates require a finite nonsingular right-handed cell.")
            start, end = int(self.batch.batch_ptr[index]), int(self.batch.batch_ptr[index + 1])
            if scale_atoms:
                fractional = self.batch.positions[start:end] @ torch.linalg.inv(self.batch.cell[index])
                self.batch.positions[start:end] = fractional @ value
            self.batch.cell[index].copy_(value)
            if evaluate:
                self.evaluate()

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

    validate_hippynn_options(options)

    ensemble_graph, (_, output_info) = hippynn.graphs.make_ensemble(
        os.path.join(str(ensemble_directory), "model-*"), quiet=True
    )
    ensemble_graph.to(device)
    table, selected = hippynn_state_table(model_mode, selected_state_value, ML_config, properties_list)
    gap_rows, gap_seeking_settings = configured_alchemi_gap_rows(
        sampler_config,
        properties_list,
        selected,
    )

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
        gap_rows=gap_rows,
        gap_seeking_settings=gap_seeking_settings,
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
    gap_rows, gap_seeking_settings = configured_alchemi_gap_rows(
        sampler_config,
        properties_list,
        selected_state_value,
    )
    physical_states = _physical_force_states(model_mode, selected_state_value, properties_list)
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
        model.require_physical_states(physical_states)
        model.well_params = None if well_params is None else dict(well_params)
        model.gap_rows = [dict(row) for row in gap_rows]
        model.gap_seeking_settings = dict(gap_seeking_settings)
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
        physical_force_states=physical_states,
        model_mode=model_mode,
        selected_state_value=selected_state_value,
        gap_rows=gap_rows,
        gap_seeking_settings=gap_seeking_settings,
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
    master_directory=None,
):
    """Parsl task for strict-full batched ALCHEMI sampling."""

    if "batching" in sampler_config or "dynamics" in sampler_config:
        from alframework.samplers.batched_sampling import batched_sampling_task
        return batched_sampling_task.func(
            molecule_objects, sampler_config, model_path, current_model_id,
            gpus_per_node, ML_config, properties_list, master_directory)

    mode, selected_state_value = _validate_sampling_inputs(
        molecule_objects, sampler_config, properties_list
    )
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
        validate_alchemi_device(False, strict_gpu=strict_gpu, allow_cpu_debug=allow_cpu_debug)

    model = load_alchemi_calculator(
        sampler_config=sampler_config,
        ensemble_directory=model_path.format(int(current_model_id)),
        model_mode=mode,
        selected_state_value=selected_state_value,
        ML_config=ML_config,
        properties_list=properties_list,
        device=device,
    )
    run_options = {"device": device, "properties_list": properties_list}
    if master_directory is not None:
        run_options["master_directory"] = master_directory
    outputs = run_alchemi_sampling(
        molecule_objects,
        sampler_config,
        model,
        **run_options,
    )
    for molecule in outputs:
        molecule.update_metadata(
            {
                "sampler_device": str(device),
                "sampler_worker_rank": int(worker_rank),
                "sampler_visible_device": int(visible_device),
            }
        )
    return outputs
