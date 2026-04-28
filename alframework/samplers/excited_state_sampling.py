from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from ase import units
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from parsl import python_app

from alframework.ml_interfaces.excited_state_hippynn_interface import load_excited_state_ensemble
from alframework.tools.excited_state_tools import select_excited_state, stable_uint32_seed
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import annealing_schedule


def _state_selection_config(sampler_config: dict[str, Any]) -> dict[str, Any]:
    return dict(sampler_config.get("state_selection") or {})


def _score_config(sampler_config: dict[str, Any]) -> dict[str, Any]:
    return dict(sampler_config.get("score") or {})


def _uncertainty_config(sampler_config: dict[str, Any]) -> dict[str, Any]:
    return dict(sampler_config.get("uncertainty") or {})


def _force_rms(array: np.ndarray) -> float:
    data = np.asarray(array, dtype=np.float64)
    return float(np.sqrt(np.mean(data * data)))


def _minimum_gap(energies: dict[int, float]) -> tuple[float, tuple[int, int]]:
    state_ids = sorted(energies)
    best_gap = float("inf")
    pair = (state_ids[0], state_ids[0])
    for idx, state_i in enumerate(state_ids):
        for state_j in state_ids[idx + 1 :]:
            gap = abs(float(energies[state_i]) - float(energies[state_j]))
            if gap < best_gap:
                best_gap = gap
                pair = (int(state_i), int(state_j))
    return float(best_gap), pair


def _gap_term(min_gap: float, score_config: dict[str, Any]) -> float:
    threshold = float(score_config.get("gap_threshold_eV", 0.01))
    mode = str(score_config.get("gap_mode", "linear")).strip().lower()
    if mode == "inverse":
        eps = float(score_config.get("gap_inverse_eps_eV", 1e-4))
        return max(0.0, (1.0 / (float(min_gap) + eps)) - (1.0 / (threshold + eps)))
    return max(0.0, threshold - float(min_gap))


def _effective_uncertainties(metrics: dict[str, Any], score_config: dict[str, Any]) -> tuple[float, float]:
    aggregate = str(score_config.get("uncertainty_aggregate", "max")).strip().lower()
    if aggregate == "rms":
        return float(metrics["uE_rms"]), float(metrics["uF_rms"])
    return float(metrics["uE_max"]), float(metrics["uF_max"])


def _passes_uncertainty_gate(metrics: dict[str, Any], sampler_config: dict[str, Any]) -> bool:
    gate = _uncertainty_config(sampler_config)
    if not bool(gate.get("enabled", False)):
        return True
    uE_value, uF_value = _effective_uncertainties(metrics, _score_config(sampler_config))
    min_uE = float(gate.get("min_uE", 0.0))
    min_uF = float(gate.get("min_uF", 0.0))
    if str(gate.get("logic", "either")).strip().lower() == "both":
        return uE_value >= min_uE and uF_value >= min_uF
    return uE_value >= min_uE or uF_value >= min_uF


def compute_excited_state_score(metrics: dict[str, Any], sampler_config: dict[str, Any]) -> float:
    score_config = _score_config(sampler_config)
    uE_value, uF_value = _effective_uncertainties(metrics, score_config)
    gap_value = _gap_term(float(metrics["min_gap"]), score_config)
    return (
        float(score_config.get("w_energy", 1.0)) * uE_value
        + float(score_config.get("w_force", 1.0)) * uF_value
        + float(score_config.get("w_gap", 0.0)) * gap_value
    )


def _results_to_metrics(atoms, results: dict[str, Any], state_table: list[dict[str, Any]], selected_state: int) -> dict[str, Any]:
    energies = {int(row["state"]): float(results[f"E_mean_S{int(row['state'])}"]) for row in state_table}
    energy_stds = {int(row["state"]): float(results[f"E_std_S{int(row['state'])}"]) for row in state_table}
    min_gap, pair = _minimum_gap(energies)
    force_stds: dict[int, float] = {}
    for row in state_table:
        state = int(row["state"])
        force_stds[state] = _force_rms(np.asarray(results[f"F_std_S{state}"], dtype=np.float64))

    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    all_distances = np.asarray(atoms.get_all_distances(mic=True), dtype=np.float64)
    np.fill_diagonal(all_distances, np.inf)

    return {
        "selected_state": int(selected_state),
        "energies": energies,
        "energy_stds": energy_stds,
        "force_stds": force_stds,
        "uE_max": float(max(energy_stds.values())),
        "uE_rms": float(np.sqrt(np.mean(np.square(list(energy_stds.values()))))),
        "uF_max": float(max(force_stds.values())),
        "uF_rms": float(np.sqrt(np.mean(np.square(list(force_stds.values()))))),
        "min_gap": float(min_gap),
        "min_gap_pair": [int(pair[0]), int(pair[1])],
        "selected_energy_eV": float(energies[int(selected_state)]),
        "fmax": float(np.linalg.norm(forces, axis=1).max()),
        "min_dist": float(all_distances.min()),
    }


def _temperature_feed_parameters(sampler_config: dict[str, Any], rng: np.random.Generator) -> dict[str, float | None]:
    feed = {
        "Tamp": float(rng.uniform(*sampler_config["amp_temp"])),
        "Tper": float(rng.uniform(*sampler_config["per_temp"])),
        "Tsrt": float(rng.uniform(*sampler_config["srt_temp"])),
        "Tend": float(rng.uniform(*sampler_config["end_temp"])),
        "Ramp": None,
        "Rper": None,
        "Rend": None,
    }
    if sampler_config.get("amp_dens") is not None:
        feed["Ramp"] = float(rng.uniform(*sampler_config["amp_dens"]))
    if sampler_config.get("per_dens") is not None:
        feed["Rper"] = float(rng.uniform(*sampler_config["per_dens"]))
    if sampler_config.get("end_dens") is not None:
        feed["Rend"] = float(rng.uniform(*sampler_config["end_dens"]))
    return feed


def _sample_trajectory_interval(sampler_config: dict[str, Any], rng: np.random.Generator) -> int | None:
    interval = sampler_config.get("trajectory_interval")
    if interval is None:
        return None
    frequency = float(sampler_config.get("trajectory_frequency", 1.0))
    if rng.random() > frequency:
        return None
    return int(interval)


def _write_metadata(meta_dir: str | None, moleculeid: str, payload: dict[str, Any], metadata_format: str) -> None:
    if meta_dir is None:
        return
    meta_root = Path(meta_dir).expanduser().resolve()
    meta_root.mkdir(parents=True, exist_ok=True)
    fmt = str(metadata_format).strip().lower()
    if fmt in {"pickle", "both"}:
        with open(meta_root / f"metadata-{moleculeid}.p", "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if fmt in {"json", "both"}:
        with open(meta_root / f"metadata-{moleculeid}.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=_json_default)


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def run_excited_state_sampling(
    molecule_object: MoleculesObject,
    *,
    sampler_config: dict[str, Any],
    model_path: str,
    current_model_id: int,
    gpus_per_node: int,
    properties_list: dict[str, list[Any]],
) -> MoleculesObject:
    import torch
    from hippynn.interfaces.ase_interface import HippynnCalculator

    if not isinstance(molecule_object, MoleculesObject):
        raise TypeError("molecule_object must be a MoleculesObject instance.")

    worker_rank = int(os.environ.get("PARSL_WORKER_RANK", "0"))
    if torch.cuda.is_available() and int(gpus_per_node) > 0:
        gpu_index = worker_rank % int(gpus_per_node)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    rng = np.random.default_rng(stable_uint32_seed(molecule_object.get_moleculeid(), int(current_model_id)))
    ensemble_graph, state_table = load_excited_state_ensemble(
        model_path.format(int(current_model_id)),
        properties_list=properties_list,
        device=str(device),
    )
    missing_forces = [int(row["state"]) for row in state_table if row["force_node_base"] is None]
    if missing_forces:
        raise ValueError(
            "excited_state_sampling_task requires force properties for every state. "
            f"Missing force nodes for states {missing_forces}."
        )
    state_ids = [int(row["state"]) for row in state_table]
    selected_state = select_excited_state(
        moleculeid=molecule_object.get_moleculeid(),
        available_states=state_ids,
        metadata=molecule_object.get_metadata(),
        selection_config=_state_selection_config(sampler_config),
        rng=rng,
    )
    row_by_state = {int(row["state"]): row for row in state_table}
    state_row = row_by_state[int(selected_state)]

    energy_node = ensemble_graph.node_from_name(str(state_row["energy_node_base"]))
    extra_properties: dict[str, Any] = {}
    for row in state_table:
        state = int(row["state"])
        energy_base = ensemble_graph.node_from_name(str(row["energy_node_base"]))
        extra_properties[f"E_mean_S{state}"] = energy_base.mean
        extra_properties[f"E_std_S{state}"] = energy_base.std
        if row["force_node_base"] is not None:
            force_base = ensemble_graph.node_from_name(str(row["force_node_base"]))
            extra_properties[f"F_std_S{state}"] = force_base.std

    calculator = HippynnCalculator(
        energy=energy_node.mean,
        extra_properties=extra_properties,
        en_unit=units.eV,
        offset=float(sampler_config.get("energy_offset_eV", 0.0)),
    )
    calculator.to(torch.float32)
    calculator.to(device)

    if sampler_config.get("translate_to_center", False):
        positions = molecule_object.atoms.get_positions() - molecule_object.atoms.get_center_of_mass()
        molecule_object.atoms.set_positions(positions)

    ase_atoms = molecule_object.get_atoms().copy()
    ase_atoms.calc = calculator

    feed = _temperature_feed_parameters(sampler_config, rng)
    dt = float(sampler_config["dt"])
    maxt = float(sampler_config["maxt"])
    ncheck = int(sampler_config["Ncheck"])
    min_time = float(sampler_config.get("min_time", 0.0))
    friction = float(sampler_config.get("friction", 0.02))
    trajectory_interval = _sample_trajectory_interval(sampler_config, rng)
    meta_dir = sampler_config.get("meta_dir")
    metadata_format = str(sampler_config.get("metadata_format", "pickle"))
    write_xyz = bool(sampler_config.get("write_traj_xyz", False))
    write_binary = bool(sampler_config.get("write_traj_binary", trajectory_interval is not None))

    dyn = Langevin(
        ase_atoms,
        dt * units.fs,
        friction=friction,
        temperature_K=annealing_schedule(0.0, maxt, feed["Tamp"], feed["Tper"], feed["Tsrt"], feed["Tend"]),
    )

    traj_writer = None
    xyz_handle = None
    if trajectory_interval is not None and meta_dir is not None:
        meta_root = Path(str(meta_dir)).expanduser().resolve()
        meta_root.mkdir(parents=True, exist_ok=True)
        if write_binary:
            traj_writer = Trajectory(str(meta_root / f"metadata-{molecule_object.get_moleculeid()}.traj"), mode="w", atoms=ase_atoms)
        if write_xyz:
            xyz_handle = open(meta_root / f"metadata-{molecule_object.get_moleculeid()}.xyz", "w", encoding="utf-8")

    best_record = None
    hard_close_contact = False
    start_time = time.time()
    temperatures: list[float] = []
    total_energies: list[float] = []
    selected_atoms = None

    try:
        dyn.run(1)
        n_outer = int(np.ceil((1000.0 * maxt) / (dt * ncheck)))
        density_trace: list[float] = []
        if feed["Rend"] is None:
            initial_density = None
        else:
            initial_density = (1.66054e-24 / 1.0e-24) * (np.sum(ase_atoms.get_masses()) / ase_atoms.get_volume())
        for step_index in range(n_outer):
            current_time_ps = float(step_index * ncheck * dt) / 1000.0
            target_temperature = annealing_schedule(
                current_time_ps,
                maxt,
                feed["Tamp"],
                feed["Tper"],
                feed["Tsrt"],
                feed["Tend"],
            )
            dyn.set_temperature(temperature_K=target_temperature)

            if initial_density is not None:
                target_density = (1.0e-24 / 1.66054e-24) * annealing_schedule(
                    current_time_ps,
                    maxt,
                    float(feed["Ramp"]),
                    float(feed["Rper"]),
                    float(initial_density),
                    float(feed["Rend"]),
                )
                scale = np.power(np.sum(ase_atoms.get_masses()) / (ase_atoms.get_volume() * target_density), 1.0 / 3.0)
                ase_atoms.set_cell(scale * ase_atoms.get_cell(), scale_atoms=True)
                density_trace.append(float(target_density))

            dyn.run(ncheck)
            if traj_writer is not None:
                traj_writer.write(ase_atoms)
            if xyz_handle is not None:
                write(xyz_handle, ase_atoms, format="xyz")

            metrics = _results_to_metrics(ase_atoms, dict(ase_atoms.calc.results), state_table, selected_state)
            current_temperature = float(ase_atoms.get_temperature())
            current_total_energy = float(ase_atoms.get_potential_energy() + ase_atoms.get_kinetic_energy())
            temperatures.append(current_temperature)
            total_energies.append(current_total_energy)

            if float(metrics["min_dist"]) < float(sampler_config.get("min_distance_cutoff", 0.3)):
                hard_close_contact = True
                break
            if float(metrics["fmax"]) > float(sampler_config.get("max_force_cutoff", 10.0)):
                break

            if current_time_ps < min_time:
                continue
            if not _passes_uncertainty_gate(metrics, sampler_config):
                continue

            score = compute_excited_state_score(metrics, sampler_config)
            if best_record is None or float(score) > float(best_record["score"]):
                selected_atoms = ase_atoms.copy()
                best_record = {
                    "score": float(score),
                    "step": int((step_index + 1) * ncheck),
                    "time_ps": float(current_time_ps),
                    "selected_state": int(selected_state),
                    "temperature_K": float(current_temperature),
                    "temperature_target_K": float(target_temperature),
                    **metrics,
                }
    finally:
        if traj_writer is not None:
            traj_writer.close()
        if xyz_handle is not None:
            xyz_handle.close()

    drift_eV_per_ps = None
    if len(total_energies) >= 2:
        times = np.arange(len(total_energies), dtype=np.float64) * (float(dt) * float(ncheck) / 1000.0)
        if times[-1] > times[0]:
            slope, _ = np.polyfit(times, np.asarray(total_energies, dtype=np.float64), 1)
            drift_eV_per_ps = float(slope)

    meta_dict = {
        "realtime_simulation": float(time.time() - start_time),
        "selected_state": int(selected_state),
        "best_candidate": best_record,
        "hard_close_contact": bool(hard_close_contact),
        "trajectory_temperature_trace_K": temperatures,
        "trajectory_total_energy_trace_eV": total_energies,
        "density_trace": density_trace if "density_trace" in locals() else [],
        "temperature_feed_parameters": feed,
        "energy_drift_eV_per_ps": drift_eV_per_ps,
        "chemical_symbols": ase_atoms.get_chemical_symbols(),
        "positions": ase_atoms.get_positions(wrap=True),
        "cell": ase_atoms.get_cell(),
    }
    meta_dict.update(molecule_object.get_metadata())
    _write_metadata(meta_dir, molecule_object.get_moleculeid(), meta_dict, metadata_format)

    ase_atoms.calc = None
    molecule_object.update_metadata(meta_dict)
    if best_record is not None and not hard_close_contact:
        selected_atoms.calc = None
        molecule_object.update_atoms(selected_atoms)
    else:
        molecule_object.update_atoms(None)
    return molecule_object


@python_app(executors=["alf_sampler_executor"])
def excited_state_sampling_task(
    molecule_object,
    sampler_config,
    model_path,
    current_model_id,
    gpus_per_node,
    properties_list,
):
    return run_excited_state_sampling(
        molecule_object=molecule_object,
        sampler_config=dict(sampler_config or {}),
        model_path=str(model_path),
        current_model_id=int(current_model_id),
        gpus_per_node=int(gpus_per_node),
        properties_list=dict(properties_list or {}),
    )
