"""Compatibility entrypoints for portable batched ALF sampling.

The scientific controller lives in :mod:`alframework.samplers.sampling`.
Existing import strings and task arguments remain supported.
"""
from __future__ import annotations

import os
from parsl import python_app
from alframework.tools.batched_config import (
    batching_settings, dynamics_settings, configured_native_loader, configured_native_options,
)
from alframework.tools.sampling_validation import (
    _validate_sampling_inputs, configured_alchemi_gap_rows, _physical_force_states,
)
from alframework.tools.tools import load_module_from_string
from alframework.samplers.sampling import (
    KB_EV_PER_K, DEFAULT_FRICTION_PER_FS, dynamics_runner,
    calculate_uncertainty, uncertainty_flags, _sample_range, _temperature_parameters,
    _minimum_distance, _density_parameters, _candidate_sort_key, candidate_score,
    _prune_candidates, select_gap_seeking_trigger, _candidate_record, run_batched_sampling,
)

def load_batched_calculator(*, sampler_config, ensemble_directory, model_mode,
                           selected_state_value, ML_config, properties_list, device):
    """Load a backend-independent native ensemble or an ASE compatibility bridge."""
    from alframework.ml_interfaces.batched_inference import BatchedEnsembleCalculator, ASEEnsembleModel
    native = configured_native_loader(sampler_config)
    well = dict(sampler_config.get("MLMD_calculator_options") or {}).get("well_params")
    gaps, seeking = configured_alchemi_gap_rows(sampler_config, properties_list, selected_state_value)
    physical = _physical_force_states(model_mode, selected_state_value, properties_list)
    if native:
        model = load_module_from_string(native)(
            ensemble_directory, model_mode=model_mode, selected_state_value=selected_state_value,
            ML_config=ML_config, properties_list=properties_list, sampler_config=sampler_config,
            device=device, **configured_native_options(sampler_config))
        if not isinstance(model, BatchedEnsembleCalculator):
            raise TypeError("batched_calculator must return a BatchedEnsembleCalculator instance.")
        model.require_physical_states(physical)
        model.well_params = None if well is None else dict(well)
        model.gap_rows = [dict(row) for row in gaps]
        model.gap_seeking_settings = dict(seeking)
        model.calculator_interface = "native"
        model.calculator_loader = native
        return model
    loader_path = sampler_config.get("ase_calculator")
    if not loader_path:
        raise ValueError("Configure batched_calculator for native inference or ase_calculator for compatibility.")
    loader = load_module_from_string(loader_path)
    options = dict(sampler_config.get("ase_calculator_options") or {})
    uncertainty_mode = None
    if str(sampler_config.get("use_potential_specific_code")).lower() == "neurochem":
        model_details = {"model_path": str(ensemble_directory).rstrip("/") + "/", "Nn": 8, "gpu": "0"}
        model_details.update(options)
        calculators = loader(model_details)
        uncertainty_mode = "neurochem"
    else:
        options.setdefault("device", str(device))
        calculators = loader(str(ensemble_directory).rstrip("/") + "/", **options)
    return ASEEnsembleModel(
        calculators, model_mode=model_mode, selected_state_value=selected_state_value,
        physical_force_states=physical, gap_rows=gaps, gap_seeking_settings=seeking,
        well_params=well, device=device, calculator_loader=str(loader_path),
        uncertainty_mode=uncertainty_mode)


def sampler_device(sampler_config, gpus_per_node, *, torch_module=None):
    """Select a logical worker GPU while respecting externally assigned visibility."""
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    options = dynamics_settings(sampler_config)
    version = getattr(torch, "version", None)
    vendor = "amd" if getattr(version, "hip", None) else "nvidia"
    if vendor == "amd" and options["backend"] == "alchemi":
        raise RuntimeError("ALCHEMI requires NVIDIA CUDA; select dynamics.backend='torchsim' for ROCm.")
    rank = int(os.environ.get("PARSL_WORKER_RANK", "0"))
    count = max(1, int(gpus_per_node))
    mask_names = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES") if vendor == "amd" else ("CUDA_VISIBLE_DEVICES",)
    mask = next((os.environ[name] for name in mask_names if name in os.environ), None)
    if mask is None:
        # Do this before the first runtime query, and use only the vendor's
        # physical visibility mask. A scheduler-provided mask is authoritative.
        os.environ[mask_names[0]] = str(rank % count)
        mask = os.environ[mask_names[0]]
    if torch.cuda.is_available():
        visible = int(torch.cuda.device_count())
        index = rank % max(1, min(count, visible))
        device = torch.device(f"cuda:{index}")
        torch.cuda.set_device(device)
        values = [item.strip() for item in mask.split(",") if item.strip()]
        physical = values[index] if index < len(values) else index
        if isinstance(physical, str) and physical.isdigit():
            physical = int(physical)
    else:
        if options.get("strict_gpu", True) and not options.get("allow_cpu_debug", False):
            raise RuntimeError("Batched dynamics requires a GPU; set dynamics.allow_cpu_debug=true for CPU checks.")
        device, physical, vendor = torch.device("cpu"), None, "cpu"
    return device, {"sampler_device": str(device), "sampler_worker_rank": rank,
                    "sampler_visible_device": physical, "sampler_gpu_vendor": vendor}


@python_app(executors=["alf_sampler_executor"])
def batched_sampling_task(molecule_objects, sampler_config, model_path, current_model_id,
                          gpus_per_node, ML_config, properties_list, master_directory=None):
    """Parsl entrypoint for portable ALF batching with either dynamics engine."""
    from alframework.tools.batched_config import unified_sampling
    if unified_sampling(sampler_config):
        from alframework.samplers.sampling import sampling_task
        return sampling_task.func(molecule_objects, sampler_config, model_path, current_model_id,
                                  gpus_per_node, ML_config, properties_list, master_directory)
    mode, selected = _validate_sampling_inputs(molecule_objects, sampler_config, properties_list)
    device, metadata = sampler_device(sampler_config, gpus_per_node)
    model = load_batched_calculator(
        sampler_config=sampler_config, ensemble_directory=model_path.format(int(current_model_id)),
        model_mode=mode, selected_state_value=selected, ML_config=ML_config,
        properties_list=properties_list, device=device)
    outputs = run_batched_sampling(molecule_objects, sampler_config, model,
                                   properties_list=properties_list, device=device,
                                   master_directory=master_directory)
    for molecule in outputs:
        molecule.update_metadata(metadata)
    return outputs
