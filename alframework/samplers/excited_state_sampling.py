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
from alframework.samplers.sampling_timing import SamplerTiming
from alframework.tools.excited_state_tools import gap_key_for_pair, select_excited_state, stable_uint32_seed
from alframework.tools.molecule_payloads import (
    clean_atoms,
    plain_metadata_dict,
    write_sampler_error_ref,
    write_sampler_result_ref,
)
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import annealing_schedule


def _state_selection_config(sampler_config: dict[str, Any]) -> dict[str, Any]:
    return dict(sampler_config.get("state_selection") or {})


def _score_config(sampler_config: dict[str, Any]) -> dict[str, Any]:
    return dict(sampler_config.get("score") or {})


def _uncertainty_config(sampler_config: dict[str, Any]) -> dict[str, Any]:
    return dict(sampler_config.get("uncertainty") or {})


def _udd_config(sampler_config: dict[str, Any]) -> dict[str, Any]:
    return dict(sampler_config.get("udd") or {})


def _gap_seeking_config(sampler_config: dict[str, Any]) -> dict[str, Any]:
    return dict(sampler_config.get("gap_seeking") or {})


def _dynamics_backend(sampler_config: dict[str, Any]) -> str:
    backend = str(sampler_config.get("dynamics_backend", "ase")).strip().lower()
    aliases = {
        "ase_langevin": "ase",
        "alchemi": "alchemi_baoab",
        "nvalchemi": "alchemi_baoab",
        "nvalchemi_langevin": "alchemi_baoab",
    }
    backend = aliases.get(backend, backend)
    if backend not in {"ase", "alchemi_baoab"}:
        raise ValueError(
            f"Unsupported dynamics_backend {backend!r}. Use 'ase' or 'alchemi_baoab'."
        )
    return backend


def _alchemi_batch_size(sampler_config: dict[str, Any]) -> int:
    return max(1, int(dict(sampler_config.get("alchemi_baoab") or {}).get("batch_size", 1)))


def _force_rms(array: np.ndarray) -> float:
    data = np.asarray(array, dtype=np.float64)
    return float(np.sqrt(np.mean(data * data)))


def _force_norm(array: np.ndarray) -> float:
    data = np.asarray(array, dtype=np.float64)
    return float(np.sqrt(np.sum(data * data)))


def _make_hippynn_calculator(
    HippynnCalculator,
    torch,
    *,
    energy,
    extra_properties: dict[str, Any] | None,
    sampler_config: dict[str, Any],
    device,
    offset: float | None = None,
):
    calc = HippynnCalculator(
        energy=energy,
        extra_properties=extra_properties,
        en_unit=units.eV,
        offset=float(sampler_config.get("energy_offset_eV", 0.0) if offset is None else offset),
    )
    calc.to(torch.float32)
    calc.to(device)
    return calc


def _calculator_force_norm(atoms, calculator) -> float:
    probe_atoms = atoms.copy()
    probe_atoms.calc = calculator
    try:
        return _force_norm(np.asarray(probe_atoms.get_forces(), dtype=np.float64))
    finally:
        probe_atoms.calc = None


def _required_sampler_result_keys(state_table: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for row in state_table:
        state = int(row["state"])
        keys.extend([f"E_mean_S{state}", f"E_std_S{state}", f"F_std_S{state}"])
    return keys


def _validate_sampler_results(
    results: dict[str, Any],
    state_table: list[dict[str, Any]],
    *,
    selected_state: int,
    step: int | None,
    context: str,
    gap_seeking_switched: bool,
    gap_seeking_pair: list[int] | None,
) -> None:
    missing = [key for key in _required_sampler_result_keys(state_table) if key not in results]
    if not missing:
        return
    available = sorted(str(key) for key in results)
    raise RuntimeError(
        "Excited-state sampler calculator results are missing required keys after evaluation. "
        f"context={context!r}, step={step}, selected_state={int(selected_state)}, "
        f"gap_seeking_switched={bool(gap_seeking_switched)}, gap_seeking_pair={gap_seeking_pair}, "
        f"missing_keys={missing}, available_keys={available}"
    )


def _fresh_sampler_results(
    atoms,
    state_table: list[dict[str, Any]],
    *,
    selected_state: int,
    step: int | None,
    context: str,
    gap_seeking_switched: bool,
    gap_seeking_pair: list[int] | None,
) -> dict[str, Any]:
    atoms.get_potential_energy()
    atoms.get_forces()
    results = dict(atoms.calc.results)
    _validate_sampler_results(
        results,
        state_table,
        selected_state=selected_state,
        step=step,
        context=context,
        gap_seeking_switched=gap_seeking_switched,
        gap_seeking_pair=gap_seeking_pair,
    )
    return results


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


def _gap_key_from_pair(pair: tuple[int, int] | list[int]) -> str:
    lower, upper = int(pair[0]), int(pair[1])
    return gap_key_for_pair(lower, upper)


def _adjacent_gap_pairs(state_ids: list[int]) -> list[tuple[int, int]]:
    states = sorted(int(state) for state in state_ids)
    if len(states) < 2:
        return []
    return [(states[idx], states[idx + 1]) for idx in range(len(states) - 1)]


def _allowed_gap_pairs(
    state_ids: list[int],
    selected_state: int,
    candidate_pairs: str,
) -> list[tuple[int, int]]:
    adjacent_pairs = _adjacent_gap_pairs(state_ids)
    mode = str(candidate_pairs).strip().lower()
    if mode == "adjacent":
        return [pair for pair in adjacent_pairs if int(selected_state) in pair]
    if mode in {"all", "all_adjacent"}:
        return adjacent_pairs
    raise ValueError(
        f"Unsupported gap_seeking.candidate_pairs: {candidate_pairs!r}. "
        "Use 'adjacent', 'all_adjacent', or legacy alias 'all'."
    )


def _gap_infos_from_metric_data(
    metrics: dict[str, Any],
    pairs: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    gap_means = dict(metrics.get("gap_means") or {})
    energies = dict(metrics.get("energies") or {})
    for lower_state, upper_state in pairs:
        gap_key = _gap_key_from_pair((lower_state, upper_state))
        if gap_key in gap_means:
            gap = float(gap_means[gap_key])
            gap_source = "direct_gap_head"
        elif lower_state in energies and upper_state in energies:
            gap = float(energies[int(upper_state)]) - float(energies[int(lower_state)])
            gap_source = "state_energy_difference"
        else:
            continue
        infos.append(
            {
                "pair": [int(lower_state), int(upper_state)],
                "gap_key": gap_key,
                "gap_source": gap_source,
                "gap_eV": float(gap),
                "abs_gap_eV": abs(float(gap)),
            }
        )
    return infos


def _gap_infos_from_energies(
    energies: dict[int, float],
    pairs: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for lower_state, upper_state in pairs:
        if lower_state not in energies or upper_state not in energies:
            continue
        gap = float(energies[int(upper_state)]) - float(energies[int(lower_state)])
        infos.append(
            {
                "pair": [int(lower_state), int(upper_state)],
                "gap_key": _gap_key_from_pair((lower_state, upper_state)),
                "gap_source": "state_energy_difference",
                "gap_eV": float(gap),
                "abs_gap_eV": abs(float(gap)),
            }
        )
    return infos


def _gap_infos_from_results(
    results: dict[str, Any],
    pairs: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    infos: list[dict[str, Any]] = []
    for lower_state, upper_state in pairs:
        gap_key = _gap_key_from_pair((lower_state, upper_state))
        direct_gap_key = f"{gap_key}_mean"
        if direct_gap_key in results:
            gap = float(np.asarray(results[direct_gap_key], dtype=np.float64).reshape(-1)[0])
            gap_source = "direct_gap_head"
        else:
            energy_i_key = f"E_mean_S{int(lower_state)}"
            energy_j_key = f"E_mean_S{int(upper_state)}"
            if energy_i_key not in results or energy_j_key not in results:
                raise ValueError(
                    "Gap-seeking switch requires either direct gap result "
                    f"{direct_gap_key!r} or state energy results {energy_i_key!r} and {energy_j_key!r}."
                )
            energy_i = float(np.asarray(results[energy_i_key], dtype=np.float64).reshape(-1)[0])
            energy_j = float(np.asarray(results[energy_j_key], dtype=np.float64).reshape(-1)[0])
            gap = energy_j - energy_i
            gap_source = "state_energy_difference"
        infos.append(
            {
                "pair": [int(lower_state), int(upper_state)],
                "gap_key": gap_key,
                "gap_source": gap_source,
                "gap_eV": float(gap),
                "abs_gap_eV": abs(float(gap)),
            }
        )
    return infos


def _gap_seeking_settings(
    gap_seeking: dict[str, Any],
    state_ids: list[int],
    selected_state: int,
) -> dict[str, Any]:
    enabled = bool(gap_seeking.get("enabled", False))
    mode = str(gap_seeking.get("mode", "levine_coe_martinez_switch")).strip().lower()
    trigger_gap_threshold = float(gap_seeking.get("trigger_gap_threshold_eV", 0.05))
    candidate_pairs = str(gap_seeking.get("candidate_pairs", "adjacent")).strip().lower()
    switch_policy = str(gap_seeking.get("switch_policy", "stay_fixed")).strip().lower()
    sigma = float(gap_seeking.get("sigma", 3.5))
    alpha_eV = float(gap_seeking.get("alpha_eV", 0.05))
    switch_check_interval = int(gap_seeking.get("switch_check_interval", 1))
    exit_gap_threshold = gap_seeking.get("exit_gap_threshold_eV", None)
    exit_gap_threshold = None if exit_gap_threshold is None else float(exit_gap_threshold)
    min_lcm_steps = int(gap_seeking.get("min_lcm_steps", 0))

    if not enabled:
        return {
            "enabled": False,
            "mode": mode,
            "trigger_gap_threshold_eV": trigger_gap_threshold,
            "exit_gap_threshold_eV": exit_gap_threshold,
            "candidate_pairs": candidate_pairs,
            "switch_policy": switch_policy,
            "sigma": sigma,
            "alpha_eV": alpha_eV,
            "switch_check_interval": switch_check_interval,
            "min_lcm_steps": min_lcm_steps,
            "pairs": [],
        }
    pairs = _allowed_gap_pairs(state_ids, int(selected_state), candidate_pairs)
    if mode != "levine_coe_martinez_switch":
        raise ValueError(
            f"Unsupported gap_seeking.mode: {mode!r}. Use 'levine_coe_martinez_switch'."
        )
    if switch_policy not in {"stay_fixed", "hysteresis"}:
        raise ValueError(
            f"Unsupported gap_seeking.switch_policy: {switch_policy!r}. "
            "Use 'stay_fixed' or 'hysteresis'."
        )
    if trigger_gap_threshold <= 0.0:
        raise ValueError("gap_seeking.trigger_gap_threshold_eV must be > 0.")
    if switch_policy == "hysteresis":
        if exit_gap_threshold is None:
            raise ValueError("gap_seeking.exit_gap_threshold_eV is required for switch_policy='hysteresis'.")
        if exit_gap_threshold <= trigger_gap_threshold:
            raise ValueError(
                "gap_seeking.exit_gap_threshold_eV must be greater than "
                "gap_seeking.trigger_gap_threshold_eV for switch_policy='hysteresis'."
            )
    if exit_gap_threshold is not None and exit_gap_threshold <= 0.0:
        raise ValueError("gap_seeking.exit_gap_threshold_eV must be > 0.")
    if min_lcm_steps < 0:
        raise ValueError("gap_seeking.min_lcm_steps must be >= 0.")
    if sigma <= 0.0:
        raise ValueError("gap_seeking.sigma must be > 0.")
    if alpha_eV <= 0.0:
        raise ValueError("gap_seeking.alpha_eV must be > 0.")
    if switch_check_interval <= 0:
        raise ValueError("gap_seeking.switch_check_interval must be > 0.")
    if not pairs:
        raise ValueError("gap_seeking requires at least one valid state-energy pair.")
    return {
        "enabled": True,
        "mode": mode,
        "trigger_gap_threshold_eV": trigger_gap_threshold,
        "exit_gap_threshold_eV": exit_gap_threshold,
        "candidate_pairs": candidate_pairs,
        "switch_policy": switch_policy,
        "sigma": sigma,
        "alpha_eV": alpha_eV,
        "switch_check_interval": switch_check_interval,
        "min_lcm_steps": min_lcm_steps,
        "pairs": pairs,
    }


def _make_lcm_energy_node(
    energy_node_by_state: dict[int, Any],
    gap_node_by_pair: dict[tuple[int, int], Any],
    pair: tuple[int, int] | list[int],
    *,
    sigma: float,
    alpha_eV: float,
):
    lower_state, upper_state = int(pair[0]), int(pair[1])
    if lower_state not in energy_node_by_state or upper_state not in energy_node_by_state:
        raise ValueError(f"LCM gap seeking requested missing state pair {lower_state}, {upper_state}.")
    energy_i = energy_node_by_state[lower_state].mean
    energy_j = energy_node_by_state[upper_state].mean
    gap_node = gap_node_by_pair.get((lower_state, upper_state))
    gap = gap_node.mean if gap_node is not None else energy_j - energy_i
    # HIPPYNN graph nodes support pow but not abs. This is a smooth |gap|
    # surrogate for the Levine-Coe-Martinez denominator.
    abs_gap = ((gap * gap) + 1.0e-12) ** 0.5
    return 0.5 * (energy_i + energy_j) + float(sigma) * gap * gap / (abs_gap + float(alpha_eV))


def _gap_term(min_gap: float, score_config: dict[str, Any]) -> float:
    threshold = float(score_config.get("gap_threshold_eV", 0.01))
    mode = str(score_config.get("gap_mode", "linear")).strip().lower()
    if mode == "inverse":
        eps = float(score_config.get("gap_inverse_eps_eV", 1e-4))
        return max(0.0, (1.0 / (float(min_gap) + eps)) - (1.0 / (threshold + eps)))
    return max(0.0, threshold - float(min_gap))


def _effective_uncertainties(metrics: dict[str, Any], score_config: dict[str, Any]) -> tuple[float, float]:
    scope = str(score_config.get("uncertainty_scope", "selected_state")).strip().lower()
    if scope == "selected_state":
        return float(metrics["uE_selected"]), float(metrics["uF_selected"])
    if scope != "all_states":
        raise ValueError(
            f"Unsupported excited-state uncertainty_scope: {scope!r}. "
            "Use 'selected_state' or 'all_states'."
        )
    aggregate = str(score_config.get("uncertainty_aggregate", "max")).strip().lower()
    if aggregate == "rms":
        return float(metrics["uE_rms"]), float(metrics["uF_rms"])
    if aggregate != "max":
        raise ValueError(
            f"Unsupported excited-state uncertainty_aggregate: {aggregate!r}. "
            "Use 'max' or 'rms'."
        )
    return float(metrics["uE_max"]), float(metrics["uF_max"])


def _score_uncertainty_metadata(sampler_config: dict[str, Any]) -> dict[str, Any]:
    score_config = _score_config(sampler_config)
    scope = str(score_config.get("uncertainty_scope", "selected_state")).strip().lower()
    aggregate = str(score_config.get("uncertainty_aggregate", "max")).strip().lower()
    if scope == "selected_state":
        source = "selected_state"
    elif scope == "all_states":
        source = f"all_states_{aggregate}"
    else:
        source = scope
    return {
        "score_uncertainty_scope": scope,
        "score_uncertainty_aggregate": aggregate,
        "score_uncertainty_source": source,
    }


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


def compute_excited_state_score_components(
    metrics: dict[str, Any],
    sampler_config: dict[str, Any],
    *,
    udd_gap_key: str | None = None,
) -> dict[str, float]:
    score_config = _score_config(sampler_config)
    uE_value, uF_value = _effective_uncertainties(metrics, score_config)
    gap_value = _gap_term(float(metrics["min_gap"]), score_config)
    components = {
        "energy_uncertainty": float(score_config.get("w_energy", 1.0)) * uE_value,
        "force_uncertainty": float(score_config.get("w_force", 1.0)) * uF_value,
        "gap": float(score_config.get("w_gap", 0.0)) * gap_value,
        "gap_uncertainty": 0.0,
    }
    gap_uncertainty_weight = float(score_config.get("w_gap_uncertainty", 0.0))
    if gap_uncertainty_weight != 0.0:
        target = str(score_config.get("gap_uncertainty_target", "udd_target")).strip().lower()
        if target != "udd_target":
            raise ValueError(
                f"Unsupported excited-state gap_uncertainty_target: {target!r}. "
                "Use 'udd_target'."
            )
        if udd_gap_key is None:
            raise ValueError("w_gap_uncertainty > 0 requires UDD to select a target gap.")
        if udd_gap_key not in metrics["gap_stds"]:
            raise ValueError(
                f"w_gap_uncertainty > 0 requires gap std for UDD target {udd_gap_key!r}, "
                "but it was not present in sampler metrics."
            )
        components["gap_uncertainty"] = gap_uncertainty_weight * float(metrics["gap_stds"][udd_gap_key])
    return components


def compute_excited_state_score(
    metrics: dict[str, Any],
    sampler_config: dict[str, Any],
    *,
    udd_gap_key: str | None = None,
) -> float:
    components = compute_excited_state_score_components(metrics, sampler_config, udd_gap_key=udd_gap_key)
    return float(sum(components.values()))


def _results_to_metrics(
    atoms,
    results: dict[str, Any],
    state_table: list[dict[str, Any]],
    gap_table: list[dict[str, Any]],
    selected_state: int,
    *,
    forces_override: np.ndarray | None = None,
) -> dict[str, Any]:
    energies = {int(row["state"]): float(results[f"E_mean_S{int(row['state'])}"]) for row in state_table}
    energy_stds = {int(row["state"]): float(results[f"E_std_S{int(row['state'])}"]) for row in state_table}
    gap_means: dict[str, float] = {}
    gap_stds: dict[str, float] = {}
    gap_pairs: dict[str, list[int]] = {}
    for row in gap_table:
        gap_key = str(row["gap_key"])
        mean_key = f"{gap_key}_mean"
        std_key = f"{gap_key}_std"
        if mean_key in results:
            gap_means[gap_key] = float(results[mean_key])
            gap_pairs[gap_key] = [int(row["lower_state"]), int(row["upper_state"])]
        if std_key in results:
            gap_stds[gap_key] = float(results[std_key])
    if gap_means:
        min_gap_key, min_gap = min(gap_means.items(), key=lambda item: abs(float(item[1])))
        pair = tuple(gap_pairs[min_gap_key])
        min_gap = abs(float(min_gap))
    else:
        min_gap, pair = _minimum_gap(energies)
    force_stds: dict[int, float] = {}
    for row in state_table:
        state = int(row["state"])
        force_stds[state] = _force_rms(np.asarray(results[f"F_std_S{state}"], dtype=np.float64))
    uncertainty_state = int(selected_state)
    if uncertainty_state not in energy_stds or uncertainty_state not in force_stds:
        raise ValueError(
            f"Selected state {uncertainty_state} is missing from sampler uncertainty metrics."
        )

    if forces_override is None:
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    else:
        forces = np.asarray(forces_override, dtype=np.float64)
    all_distances = np.asarray(atoms.get_all_distances(mic=True), dtype=np.float64)
    np.fill_diagonal(all_distances, np.inf)
    nearest_neighbor_distances = all_distances.min(axis=1)

    return {
        "selected_state": int(selected_state),
        "energies": energies,
        "energy_stds": energy_stds,
        "gap_means": gap_means,
        "gap_stds": gap_stds,
        "gap_pairs": gap_pairs,
        "force_stds": force_stds,
        "uncertainty_state": uncertainty_state,
        "uE_selected": float(energy_stds[uncertainty_state]),
        "uF_selected": float(force_stds[uncertainty_state]),
        "uE_max": float(max(energy_stds.values())),
        "uE_rms": float(np.sqrt(np.mean(np.square(list(energy_stds.values()))))),
        "uF_max": float(max(force_stds.values())),
        "uF_rms": float(np.sqrt(np.mean(np.square(list(force_stds.values()))))),
        "min_gap": float(min_gap),
        "min_gap_pair": [int(pair[0]), int(pair[1])],
        "selected_energy_eV": float(energies[int(selected_state)]),
        "fmax": float(np.linalg.norm(forces, axis=1).max()),
        "min_dist": float(all_distances.min()),
        "nearest_neighbor_distances": nearest_neighbor_distances.astype(float),
        "max_nearest_neighbor_distance": float(nearest_neighbor_distances.max()),
    }


def _candidate_min_distance(atoms) -> float | None:
    if atoms is None or len(atoms) < 2:
        return None
    distances = np.asarray(atoms.get_all_distances(mic=True), dtype=np.float64)
    np.fill_diagonal(distances, np.inf)
    min_distance = float(np.min(distances))
    if not np.isfinite(min_distance):
        return None
    return min_distance


def _qm_candidate_reject_reason(atoms, sampler_config: dict[str, Any]) -> str | None:
    min_distance = _candidate_min_distance(atoms)
    if min_distance is not None and min_distance < float(sampler_config.get("min_distance_cutoff", 0.3)):
        return "min_distance"
    return None


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


def _return_top_n(sampler_config: dict[str, Any]) -> int:
    try:
        return max(1, int(sampler_config.get("return_top_n", 1)))
    except (TypeError, ValueError):
        return 1


def _select_initial_udd_gap_target(results: dict[str, Any], gap_table: list[dict[str, Any]]) -> dict[str, Any]:
    if not gap_table:
        raise ValueError("UDD gap_std_bias requires direct ensemble gap nodes, but no gap nodes were loaded.")

    candidates: list[dict[str, Any]] = []
    for row in gap_table:
        gap_key = str(row["gap_key"])
        mean_key = f"{gap_key}_mean"
        std_key = f"{gap_key}_std"
        if mean_key not in results or std_key not in results:
            raise ValueError(f"UDD gap_std_bias requires {mean_key!r} and {std_key!r} calculator results.")
        mean = float(np.asarray(results[mean_key], dtype=np.float64).reshape(-1)[0])
        std = float(np.asarray(results[std_key], dtype=np.float64).reshape(-1)[0])
        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError(f"UDD gap_std_bias found non-finite values for {gap_key}: mean={mean}, std={std}.")
        candidates.append(
            {
                "gap_key": gap_key,
                "gap_pair": [int(row["lower_state"]), int(row["upper_state"])],
                "initial_gap_mean": mean,
                "initial_gap_std": std,
            }
        )
    return min(candidates, key=lambda item: abs(float(item["initial_gap_mean"])))


def _udd_tau_settings(udd: dict[str, Any], total_md_steps: int) -> dict[str, Any]:
    tau_mode = str(udd.get("tau_mode", "fixed")).strip().lower()
    if tau_mode not in {"fixed", "force_relative"}:
        raise ValueError(f"Unsupported excited-state UDD tau_mode: {tau_mode!r}. Use 'fixed' or 'force_relative'.")
    if tau_mode == "fixed":
        return {
            "tau_mode": tau_mode,
            "tau_relative": None,
            "tau_history_steps": None,
            "tau_update_interval": None,
            "tau_min": None,
            "tau_max": None,
            "tau_force_epsilon": None,
        }

    tau_relative = float(udd.get("tau_relative", 0.05))
    tau_history_steps = int(udd.get("tau_history_steps", 100))
    tau_update_interval = int(udd.get("tau_update_interval", tau_history_steps))
    tau_min = float(udd.get("tau_min", 0.0))
    tau_max = float(udd.get("tau_max", 10.0))
    tau_force_epsilon = float(udd.get("tau_force_epsilon", 1.0e-12))

    if tau_relative <= 0.0:
        raise ValueError("excited-state UDD force_relative mode requires tau_relative > 0.")
    if tau_history_steps <= 0:
        raise ValueError("excited-state UDD force_relative mode requires tau_history_steps > 0.")
    if tau_update_interval <= 0:
        raise ValueError("excited-state UDD force_relative mode requires tau_update_interval > 0.")
    if tau_min < 0.0:
        raise ValueError("excited-state UDD force_relative mode requires tau_min >= 0.")
    if tau_max < tau_min:
        raise ValueError("excited-state UDD force_relative mode requires tau_max >= tau_min.")
    if tau_force_epsilon <= 0.0:
        raise ValueError("excited-state UDD force_relative mode requires tau_force_epsilon > 0.")
    if int(total_md_steps) < tau_history_steps:
        raise ValueError(
            "excited-state UDD force_relative mode requires total trajectory steps "
            f"({int(total_md_steps)}) >= tau_history_steps ({tau_history_steps})."
        )

    return {
        "tau_mode": tau_mode,
        "tau_relative": float(tau_relative),
        "tau_history_steps": int(tau_history_steps),
        "tau_update_interval": int(tau_update_interval),
        "tau_min": float(tau_min),
        "tau_max": float(tau_max),
        "tau_force_epsilon": float(tau_force_epsilon),
    }


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


def _write_qm_candidate_xyz(
    sampler_config: dict[str, Any],
    parent_molecule_id: str,
    candidate_items: list[dict[str, Any]],
    candidate_ids: list[str],
) -> None:
    if not bool(sampler_config.get("write_qm_candidate_xyz", False)):
        return
    if not candidate_items:
        return

    output_dir = Path(str(sampler_config.get("qm_candidate_xyz_dir", "sampling/qm_candidates"))).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"qm-candidates-{parent_molecule_id}.xyz"

    with open(output_path, "w", encoding="utf-8") as handle:
        for rank, (candidate, molecule_id) in enumerate(zip(candidate_items, candidate_ids)):
            atoms = candidate["atoms"].copy()
            atoms.calc = None
            record = dict(candidate["record"])
            atoms.info.update(
                {
                    "molecule_id": molecule_id,
                    "parent_molecule_id": parent_molecule_id,
                    "candidate_rank": int(rank),
                    "candidate_score": float(record["score"]),
                    "candidate_step": int(record["step"]),
                    "candidate_time_ps": float(record["time_ps"]),
                    "selected_state": int(record["selected_state"]),
                    "min_dist": float(record["min_dist"]),
                    "fmax": float(record["fmax"]),
                }
            )
            write(handle, atoms, format="xyz")


def _validate_same_alchemi_batch_inputs(molecules: list[MoleculesObject]) -> None:
    if not molecules:
        raise ValueError("molecule_objects must contain at least one MoleculesObject.")
    reference = molecules[0].get_atoms()
    if reference is None:
        raise ValueError(f"Batch molecule {molecules[0].get_moleculeid()} has no atoms.")
    reference_numbers = np.asarray(reference.get_atomic_numbers(), dtype=int)
    for molecule in molecules:
        if not isinstance(molecule, MoleculesObject):
            raise TypeError("Every batch item must be a MoleculesObject instance.")
        atoms = molecule.get_atoms()
        if atoms is None:
            raise ValueError(f"Batch molecule {molecule.get_moleculeid()} has no atoms.")
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=int)
        if len(numbers) != len(reference_numbers) or not np.array_equal(numbers, reference_numbers):
            raise ValueError(
                "Batched ALCHEMI BAOAB currently requires every molecule to have "
                "the same atom count and atomic-number order."
            )


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


def run_excited_state_sampling_batch(
    molecule_objects: list[MoleculesObject],
    *,
    sampler_config: dict[str, Any],
    model_path: str,
    current_model_id: int,
    gpus_per_node: int,
    properties_list: dict[str, list[Any]],
) -> list[MoleculesObject]:
    import torch

    dynamics_backend = _dynamics_backend(sampler_config)
    if dynamics_backend != "alchemi_baoab":
        raise ValueError("alchemi_baoab.batch_size > 1 requires dynamics_backend='alchemi_baoab'.")
    if bool(_udd_config(sampler_config).get("enabled", False)):
        raise NotImplementedError("Batched ALCHEMI BAOAB does not yet support UDD.")

    from alframework.samplers.alchemi_baoab_dynamics import (
        ALFExcitedStateAlchemiModel,
        AlchemiBaoabRunner,
        alchemi_config_from_sampler,
        build_alchemi_batch_from_atoms_list,
        ensure_alchemi_available,
        validate_alchemi_sampler_support,
    )

    ensure_alchemi_available()
    _validate_same_alchemi_batch_inputs(molecule_objects)

    worker_rank = int(os.environ.get("PARSL_WORKER_RANK", "0"))
    if torch.cuda.is_available() and int(gpus_per_node) > 0:
        gpu_index = worker_rank % int(gpus_per_node)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    actual_batch_size = len(molecule_objects)
    timing = SamplerTiming.from_sampler_config(
        sampler_config,
        backend=dynamics_backend,
        device=device,
        batch_size=actual_batch_size,
    )

    rngs = [
        np.random.default_rng(stable_uint32_seed(molecule.get_moleculeid(), int(current_model_id)))
        for molecule in molecule_objects
    ]
    ensemble_graph, state_table, gap_table = load_excited_state_ensemble(
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
    selected_states: list[int] = []
    for molecule, rng in zip(molecule_objects, rngs):
        selected_states.append(
            select_excited_state(
                moleculeid=molecule.get_moleculeid(),
                available_states=state_ids,
                metadata=molecule.get_metadata(),
                selection_config=_state_selection_config(sampler_config),
                rng=rng,
            )
        )
    if len(set(selected_states)) != 1:
        raise ValueError(
            "Batched ALCHEMI BAOAB requires all molecules in one batch to use the same selected state. "
            f"Found selected_states={selected_states}."
        )
    selected_state = int(selected_states[0])

    energy_node_by_state: dict[int, Any] = {}
    force_node_by_state: dict[int, Any] = {}
    extra_properties: dict[str, Any] = {}
    gap_node_by_key: dict[str, Any] = {}
    gap_node_by_pair: dict[tuple[int, int], Any] = {}
    for row in state_table:
        state = int(row["state"])
        energy_base = ensemble_graph.node_from_name(str(row["energy_node_base"]))
        energy_node_by_state[state] = energy_base
        extra_properties[f"E_mean_S{state}"] = energy_base.mean
        extra_properties[f"E_std_S{state}"] = energy_base.std
        if row["force_node_base"] is not None:
            force_base = ensemble_graph.node_from_name(str(row["force_node_base"]))
            force_node_by_state[state] = force_base
            extra_properties[f"F_std_S{state}"] = force_base.std
    for row in gap_table:
        gap_base = ensemble_graph.node_from_name(str(row["gap_node_base"]))
        gap_key = str(row["gap_key"])
        gap_node_by_key[gap_key] = gap_base
        gap_node_by_pair[(int(row["lower_state"]), int(row["upper_state"]))] = gap_base
        extra_properties[f"{gap_key}_mean"] = gap_base.mean
        extra_properties[f"{gap_key}_std"] = gap_base.std

    atoms_list = [molecule.get_atoms().copy() for molecule in molecule_objects]
    feeds = [_temperature_feed_parameters(sampler_config, rng) for rng in rngs]
    dt = float(sampler_config["dt"])
    maxt = float(sampler_config["maxt"])
    ncheck = int(sampler_config["Ncheck"])
    min_time = float(sampler_config.get("min_time", 0.0))
    friction = float(sampler_config.get("friction", 0.02))
    total_md_steps = int(np.ceil((1000.0 * maxt) / dt))
    n_outer = int(np.ceil(float(total_md_steps) / float(ncheck)))
    meta_dir = sampler_config.get("meta_dir")
    metadata_format = str(sampler_config.get("metadata_format", "pickle"))
    return_top_n = _return_top_n(sampler_config)
    if _sample_trajectory_interval(sampler_config, rngs[0]) is not None and bool(
        sampler_config.get("write_traj_xyz", False) or sampler_config.get("write_traj_binary", False)
    ):
        raise NotImplementedError("Batched ALCHEMI BAOAB does not yet support trajectory writing.")

    initial_temperatures = np.asarray(
        [
            annealing_schedule(0.0, maxt, feed["Tamp"], feed["Tper"], feed["Tsrt"], feed["Tend"])
            for feed in feeds
        ],
        dtype=np.float64,
    )
    alchemi_options = alchemi_config_from_sampler(
        sampler_config,
        default_seed=stable_uint32_seed(molecule_objects[0].get_moleculeid(), int(current_model_id) + 7919),
    )
    if str(alchemi_options.batched_gap_switch_policy).strip().lower() != "global_lowest_gap_trigger":
        raise ValueError("Batched ALCHEMI BAOAB currently supports only batched_gap_switch_policy='global_lowest_gap_trigger'.")
    for atoms, feed in zip(atoms_list, feeds):
        validate_alchemi_sampler_support(atoms, feed, device, alchemi_options)
    alchemi_batch = build_alchemi_batch_from_atoms_list(atoms_list, device=device)
    alchemi_model = ALFExcitedStateAlchemiModel(
        ensemble_graph=ensemble_graph,
        energy_node=energy_node_by_state[selected_state].mean,
        extra_properties=extra_properties,
        force_node=force_node_by_state[selected_state].mean,
        species_key=alchemi_options.species_key,
        coordinates_key=alchemi_options.coordinates_key,
        offset_eV=float(sampler_config.get("energy_offset_eV", 0.0)),
        device=device,
    )
    dyn = AlchemiBaoabRunner(
        model=alchemi_model,
        batch=alchemi_batch,
        dt_fs=dt,
        temperature_K=torch.as_tensor(initial_temperatures, dtype=torch.float32, device=device),
        friction_per_fs=friction,
        random_seed=int(alchemi_options.random_seed),
        device=device,
    )

    gap_seeking_settings = _gap_seeking_settings(_gap_seeking_config(sampler_config), state_ids, selected_state)
    gap_seeking_enabled = bool(gap_seeking_settings["enabled"])
    gap_seeking_switched = False
    gap_seeking_lcm_active = False
    gap_seeking_switch_step = None
    gap_seeking_switch_time_ps = None
    gap_seeking_pair = None
    gap_seeking_gap_key = None
    gap_seeking_gap_source = None
    gap_seeking_trigger_gap_eV = None
    gap_seeking_trigger_molecule_id = None
    gap_seeking_trigger_batch_index = None
    gap_seeking_switch_events: list[dict[str, Any]] = []

    active = np.ones(actual_batch_size, dtype=bool)
    replica_stop_reasons: list[str | None] = [None for _ in molecule_objects]
    top_candidates: list[dict[str, Any]] = []
    per_molecule_records: list[list[dict[str, Any]]] = [[] for _ in molecule_objects]
    rejected_qm_candidates_min_distance: list[int] = [0 for _ in molecule_objects]
    per_molecule_geometry_trace: list[list[dict[str, Any]]] = [[] for _ in molecule_objects]
    start_time = time.time()

    def _evaluate_results() -> dict[str, Any]:
        with timing.scope("model_eval", cuda=True):
            results = dyn.evaluate_results()
        timing.increment("scalar_sync_count", 1)
        return results

    def _run_md_steps(steps: int) -> None:
        with timing.scope("md_run", cuda=True):
            dyn.run(int(steps))
        timing.increment("num_md_steps_completed", int(steps))
        timing.increment("num_md_chunks", 1)

    def _sync_graph(graph_index: int) -> None:
        with timing.scope("host_sync"):
            dyn.sync_graph_to_atoms(int(graph_index), atoms_list[int(graph_index)])
        timing.increment("full_sync_count", 1)

    def _set_batched_lcm_mode(trigger: dict[str, Any], current_step: int) -> None:
        nonlocal gap_seeking_switched, gap_seeking_lcm_active, gap_seeking_switch_step, gap_seeking_switch_time_ps
        nonlocal gap_seeking_pair, gap_seeking_gap_key, gap_seeking_gap_source, gap_seeking_trigger_gap_eV
        nonlocal gap_seeking_trigger_molecule_id, gap_seeking_trigger_batch_index
        gap_seeking_switched = True
        gap_seeking_lcm_active = True
        gap_seeking_switch_step = int(current_step)
        gap_seeking_switch_time_ps = float(current_step * dt) / 1000.0
        gap_seeking_pair = [int(value) for value in trigger["pair"]]
        gap_seeking_gap_key = str(trigger["gap_key"])
        gap_seeking_gap_source = str(trigger["gap_source"])
        gap_seeking_trigger_gap_eV = float(trigger["gap_eV"])
        gap_seeking_trigger_batch_index = int(trigger["batch_index"])
        gap_seeking_trigger_molecule_id = molecule_objects[gap_seeking_trigger_batch_index].get_moleculeid()
        gap_seeking_switch_events.append(
            {
                "event": "enter_lcm",
                "step": int(current_step),
                "time_ps": float(current_step * dt) / 1000.0,
                "batch_index": int(gap_seeking_trigger_batch_index),
                "molecule_id": gap_seeking_trigger_molecule_id,
                **{key: trigger[key] for key in ["pair", "gap_key", "gap_source", "gap_eV", "abs_gap_eV"]},
            }
        )
        lower_state, upper_state = int(gap_seeking_pair[0]), int(gap_seeking_pair[1])
        gap_base = gap_node_by_pair.get((lower_state, upper_state))
        alchemi_model.set_lcm_gap_mode(
            lower_energy_node=energy_node_by_state[lower_state].mean,
            upper_energy_node=energy_node_by_state[upper_state].mean,
            lower_force_node=force_node_by_state[lower_state].mean,
            upper_force_node=force_node_by_state[upper_state].mean,
            gap_node=gap_base.mean if gap_base is not None else None,
            sigma=float(gap_seeking_settings["sigma"]),
            alpha_eV=float(gap_seeking_settings["alpha_eV"]),
        )

    def _set_batched_direct_mode(current_step: int, *, reason: str, gap_info: dict[str, Any] | None = None) -> None:
        nonlocal gap_seeking_lcm_active
        gap_seeking_lcm_active = False
        alchemi_model.set_energy_node(
            energy_node_by_state[int(selected_state)].mean,
            force_node=force_node_by_state[int(selected_state)].mean,
        )
        event = {
            "event": "exit_lcm",
            "step": int(current_step),
            "time_ps": float(current_step * dt) / 1000.0,
            "reason": str(reason),
            "batch_index": gap_seeking_trigger_batch_index,
            "molecule_id": gap_seeking_trigger_molecule_id,
        }
        if gap_info is not None:
            event.update({key: gap_info[key] for key in ["pair", "gap_key", "gap_source", "gap_eV", "abs_gap_eV"]})
        gap_seeking_switch_events.append(event)

    def _active_batched_gap_info() -> dict[str, Any] | None:
        if gap_seeking_trigger_batch_index is None or gap_seeking_pair is None:
            return None
        graph_results = dyn.results_for_graph(int(gap_seeking_trigger_batch_index))
        gap_infos = _gap_infos_from_results(graph_results, list(gap_seeking_settings["pairs"]))
        active_pair = [int(gap_seeking_pair[0]), int(gap_seeking_pair[1])]
        for gap_info in gap_infos:
            if [int(value) for value in gap_info["pair"]] == active_pair:
                return gap_info
        return None

    def _update_gap_seeking_switch() -> None:
        if not gap_seeking_enabled or not np.any(active):
            return
        current_step = int(dyn.nsteps)
        if current_step <= 0:
            return
        if gap_seeking_lcm_active:
            if str(gap_seeking_settings["switch_policy"]) != "hysteresis":
                return
            if gap_seeking_switch_step is not None and (
                current_step - int(gap_seeking_switch_step)
            ) < int(gap_seeking_settings["min_lcm_steps"]):
                return
            if gap_seeking_trigger_batch_index is None or not bool(active[int(gap_seeking_trigger_batch_index)]):
                _set_batched_direct_mode(current_step, reason="trigger_replica_inactive")
            else:
                _evaluate_results()
                active_gap = _active_batched_gap_info()
                if active_gap is not None and float(active_gap["abs_gap_eV"]) >= float(
                    gap_seeking_settings["exit_gap_threshold_eV"]
                ):
                    _set_batched_direct_mode(current_step, reason="exit_gap_threshold", gap_info=active_gap)
            if not gap_seeking_lcm_active:
                results = _evaluate_results()
                _validate_sampler_results(
                    results,
                    state_table,
                    selected_state=selected_state,
                    step=current_step,
                    context="batched_gap_seeking_exit",
                    gap_seeking_switched=gap_seeking_switched,
                    gap_seeking_pair=gap_seeking_pair,
                )
            return
        triggers: list[dict[str, Any]] = []
        _evaluate_results()
        for graph_index in np.where(active)[0]:
            graph_results = dyn.results_for_graph(int(graph_index))
            gap_infos = _gap_infos_from_results(graph_results, list(gap_seeking_settings["pairs"]))
            if not gap_infos:
                continue
            trigger = min(gap_infos, key=lambda item: float(item["abs_gap_eV"]))
            if float(trigger["abs_gap_eV"]) <= float(gap_seeking_settings["trigger_gap_threshold_eV"]):
                triggers.append({"batch_index": int(graph_index), **trigger})
        if not triggers:
            return
        trigger = min(triggers, key=lambda item: float(item["abs_gap_eV"]))
        _set_batched_lcm_mode(trigger, current_step)
        results = _evaluate_results()
        _validate_sampler_results(
            results,
            state_table,
            selected_state=selected_state,
            step=current_step,
            context="batched_gap_seeking_switch",
            gap_seeking_switched=gap_seeking_switched,
            gap_seeking_pair=gap_seeking_pair,
        )

    if gap_seeking_enabled:
        dyn.attach(_update_gap_seeking_switch, interval=int(gap_seeking_settings["switch_check_interval"]))

    _run_md_steps(1)
    for step_index in range(n_outer):
        if not np.any(active):
            break
        current_time_ps = float(step_index * ncheck * dt) / 1000.0
        target_temperatures = np.asarray(
            [
                annealing_schedule(current_time_ps, maxt, feed["Tamp"], feed["Tper"], feed["Tsrt"], feed["Tend"])
                for feed in feeds
            ],
            dtype=np.float64,
        )
        dyn.set_temperature(temperature_K=torch.as_tensor(target_temperatures, dtype=torch.float32, device=device))
        _run_md_steps(ncheck)
        current_step = int((step_index + 1) * ncheck)
        _evaluate_results()
        kinetic = np.asarray(dyn.kinetic_energy_eV(), dtype=np.float64).reshape(-1)
        temperatures = np.asarray(dyn.temperature_K(), dtype=np.float64).reshape(-1)

        for graph_index in np.where(active)[0]:
            graph_index = int(graph_index)
            _sync_graph(graph_index)
            graph_results = dyn.results_for_graph(graph_index)
            _validate_sampler_results(
                graph_results,
                state_table,
                selected_state=selected_state,
                step=current_step,
                context="batched_post_md_chunk",
                gap_seeking_switched=gap_seeking_switched,
                gap_seeking_pair=gap_seeking_pair,
            )
            with timing.scope("metrics"):
                metrics = _results_to_metrics(
                    atoms_list[graph_index],
                    graph_results,
                    state_table,
                    gap_table,
                    selected_state,
                    forces_override=np.asarray(graph_results["forces"], dtype=np.float64),
                )
            current_temperature = float(temperatures[graph_index])
            current_total_energy = float(np.asarray(graph_results["energy"]).reshape(-1)[0] + kinetic[graph_index])
            per_molecule_geometry_trace[graph_index].append(
                {
                    "step": int(current_step),
                    "time_ps": float(current_time_ps),
                    "min_dist": float(metrics["min_dist"]),
                    "fmax": float(metrics["fmax"]),
                    "max_nearest_neighbor_distance": float(metrics["max_nearest_neighbor_distance"]),
                    "nearest_neighbor_distances": np.asarray(metrics["nearest_neighbor_distances"], dtype=float),
                }
            )

            stop_reason = None
            if float(metrics["min_dist"]) < float(sampler_config.get("min_distance_cutoff", 0.3)):
                stop_reason = "min_distance"
            elif bool(sampler_config.get("max_nearest_neighbor_distance_check", False)):
                max_nn_cutoff = float(sampler_config["max_nearest_neighbor_distance_cutoff"])
                if float(metrics["max_nearest_neighbor_distance"]) > max_nn_cutoff:
                    stop_reason = "max_nearest_neighbor_distance"
            if stop_reason is None and float(metrics["fmax"]) > float(sampler_config.get("max_force_cutoff", 10.0)):
                stop_reason = "max_force"
            if stop_reason is not None:
                active[graph_index] = False
                replica_stop_reasons[graph_index] = stop_reason
                continue

            if current_time_ps < min_time or not _passes_uncertainty_gate(metrics, sampler_config):
                continue
            score_components = compute_excited_state_score_components(metrics, sampler_config, udd_gap_key=None)
            score = float(sum(score_components.values()))
            record = {
                "score": score,
                "score_components": score_components,
                **_score_uncertainty_metadata(sampler_config),
                "score_gap_uncertainty_key": None,
                "score_gap_uncertainty_std": None,
                "step": int(current_step),
                "time_ps": float(current_time_ps),
                "selected_state": int(selected_state),
                "batch_index": int(graph_index),
                "batch_size": int(actual_batch_size),
                "parent_molecule_id": molecule_objects[graph_index].get_moleculeid(),
                "gap_seeking_switched": bool(gap_seeking_switched),
                "gap_seeking_current_mode": "lcm_gap" if gap_seeking_lcm_active else "direct",
                "gap_seeking_pair": gap_seeking_pair,
                "gap_seeking_gap_key": gap_seeking_gap_key,
                "gap_seeking_gap_source": gap_seeking_gap_source,
                "temperature_K": current_temperature,
                "temperature_target_K": float(target_temperatures[graph_index]),
                "total_energy_eV": current_total_energy,
                **metrics,
            }
            candidate_atoms = atoms_list[graph_index].copy()
            if _qm_candidate_reject_reason(candidate_atoms, sampler_config) == "min_distance":
                rejected_qm_candidates_min_distance[graph_index] += 1
                continue
            item = {"record": record, "atoms": candidate_atoms, "parent_index": graph_index}
            top_candidates.append(item)
            per_molecule_records[graph_index].append(dict(record))
            top_candidates.sort(key=lambda candidate: float(candidate["record"]["score"]), reverse=True)
            del top_candidates[return_top_n:]

    realtime_simulation = float(time.time() - start_time)
    timing.set_count("num_md_steps_completed", int(getattr(dyn, "nsteps", 0)))
    timing_metadata = timing.metadata(total_wall_s=realtime_simulation, num_atoms=len(atoms_list[0]))
    output_candidates: list[MoleculesObject] = []
    candidate_ids: list[str] = []
    for rank, candidate in enumerate(top_candidates):
        parent_index = int(candidate["parent_index"])
        parent_id = molecule_objects[parent_index].get_moleculeid()
        candidate_id = f"{parent_id}-cand-{rank:02d}"
        candidate_ids.append(candidate_id)
        candidate_atoms = clean_atoms(candidate["atoms"])
        candidate_metadata = dict(candidate["record"])
        candidate_metadata.update(
            {
                "candidate_rank": int(rank),
                "candidate_score": float(candidate["record"]["score"]),
                "candidate_step": int(candidate["record"]["step"]),
                "candidate_time_ps": float(candidate["record"]["time_ps"]),
                "dynamics_backend": dynamics_backend,
                "alchemi_baoab_options": dict(vars(alchemi_options)),
                "timing": timing_metadata,
            }
        )
        molecule = MoleculesObject(candidate_atoms, candidate_id)
        molecule.update_metadata(plain_metadata_dict(candidate_metadata))
        output_candidates.append(molecule)

    grouped_candidates: dict[str, tuple[list[dict[str, Any]], list[str]]] = {}
    for candidate, candidate_id in zip(top_candidates, candidate_ids):
        parent_id = molecule_objects[int(candidate["parent_index"])].get_moleculeid()
        grouped_candidates.setdefault(parent_id, ([], []))[0].append(candidate)
        grouped_candidates[parent_id][1].append(candidate_id)
    for parent_id, (items, ids) in grouped_candidates.items():
        _write_qm_candidate_xyz(sampler_config, parent_id, items, ids)

    for graph_index, molecule in enumerate(molecule_objects):
        molecule_records = sorted(per_molecule_records[graph_index], key=lambda item: float(item["score"]), reverse=True)
        last_geometry_metrics = (
            dict(per_molecule_geometry_trace[graph_index][-1]) if per_molecule_geometry_trace[graph_index] else {}
        )
        meta_dict = {
            "realtime_simulation": realtime_simulation,
            "selected_state": int(selected_state),
            "best_candidate": molecule_records[0] if molecule_records else None,
            "top_candidates": molecule_records[:return_top_n],
            "return_top_n": int(return_top_n),
            "hard_close_contact": replica_stop_reasons[graph_index] == "min_distance",
            "geometry_reject_reason": replica_stop_reasons[graph_index],
            "rejected_qm_candidates_min_distance": int(rejected_qm_candidates_min_distance[graph_index]),
            "geometry_metrics_trace": per_molecule_geometry_trace[graph_index],
            "max_nearest_neighbor_distance": last_geometry_metrics.get("max_nearest_neighbor_distance"),
            "nearest_neighbor_distances": last_geometry_metrics.get("nearest_neighbor_distances"),
            "trajectory_temperature_trace_K": [],
            "trajectory_total_energy_trace_eV": [],
            "density_trace": [],
            "temperature_feed_parameters": feeds[graph_index],
            "energy_drift_eV_per_ps": None,
            "dynamics_backend": dynamics_backend,
            "alchemi_baoab_options": dict(vars(alchemi_options)),
            **_score_uncertainty_metadata(sampler_config),
            "batch_size": int(actual_batch_size),
            "batch_index": int(graph_index),
            "batch_molecule_ids": [item.get_moleculeid() for item in molecule_objects],
            "active_replica_count": int(np.sum(active)),
            "replica_stop_reasons": list(replica_stop_reasons),
            "timing": timing_metadata,
            "chemical_symbols": atoms_list[graph_index].get_chemical_symbols(),
            "positions": atoms_list[graph_index].get_positions(wrap=True),
            "cell": atoms_list[graph_index].get_cell(),
            "gap_seeking_enabled": bool(gap_seeking_enabled),
            "gap_seeking_mode": str(gap_seeking_settings["mode"]),
            "gap_seeking_candidate_pairs": str(gap_seeking_settings["candidate_pairs"]),
            "gap_seeking_switch_policy": str(gap_seeking_settings["switch_policy"]),
            "gap_seeking_trigger_gap_threshold_eV": float(gap_seeking_settings["trigger_gap_threshold_eV"]),
            "gap_seeking_exit_gap_threshold_eV": gap_seeking_settings["exit_gap_threshold_eV"],
            "gap_seeking_min_lcm_steps": int(gap_seeking_settings["min_lcm_steps"]),
            "gap_seeking_sigma": float(gap_seeking_settings["sigma"]),
            "gap_seeking_alpha_eV": float(gap_seeking_settings["alpha_eV"]),
            "gap_seeking_switch_check_interval": int(gap_seeking_settings["switch_check_interval"]),
            "gap_seeking_switched": bool(gap_seeking_switched),
            "gap_seeking_current_mode": "lcm_gap" if gap_seeking_lcm_active else "direct",
            "gap_seeking_switch_step": gap_seeking_switch_step,
            "gap_seeking_switch_time_ps": gap_seeking_switch_time_ps,
            "gap_seeking_pair": gap_seeking_pair,
            "gap_seeking_gap_key": gap_seeking_gap_key,
            "gap_seeking_gap_source": gap_seeking_gap_source,
            "gap_seeking_trigger_gap_eV": gap_seeking_trigger_gap_eV,
            "gap_seeking_trigger_molecule_id": gap_seeking_trigger_molecule_id,
            "gap_seeking_trigger_batch_index": gap_seeking_trigger_batch_index,
            "gap_seeking_switch_events": gap_seeking_switch_events,
            "gap_seeking_min_gap_trace": [],
            "gap_seeking_active_pair_gap_trace": [],
            "udd_enabled": False,
        }
        meta_dict.update(molecule.get_metadata())
        _write_metadata(meta_dir, molecule.get_moleculeid(), meta_dict, metadata_format)

    return output_candidates


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

    dynamics_backend = _dynamics_backend(sampler_config)
    use_alchemi = dynamics_backend == "alchemi_baoab"
    if not use_alchemi and _alchemi_batch_size(sampler_config) > 1:
        raise ValueError("alchemi_baoab.batch_size > 1 requires dynamics_backend='alchemi_baoab'.")
    if use_alchemi:
        from alframework.samplers.alchemi_baoab_dynamics import (
            ALFExcitedStateAlchemiModel,
            AlchemiBaoabRunner,
            alchemi_config_from_sampler,
            build_alchemi_batch,
            ensure_alchemi_available,
            validate_alchemi_sampler_support,
        )

        ensure_alchemi_available()
        HippynnCalculator = None
    else:
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
    timing = SamplerTiming.from_sampler_config(
        sampler_config,
        backend=dynamics_backend,
        device=device,
        batch_size=1,
    )

    rng = np.random.default_rng(stable_uint32_seed(molecule_object.get_moleculeid(), int(current_model_id)))
    ensemble_graph, state_table, gap_table = load_excited_state_ensemble(
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
    energy_node_by_state: dict[int, Any] = {}
    force_node_by_state: dict[int, Any] = {}
    extra_properties: dict[str, Any] = {}
    gap_node_by_key: dict[str, Any] = {}
    gap_node_by_pair: dict[tuple[int, int], Any] = {}
    for row in state_table:
        state = int(row["state"])
        energy_base = ensemble_graph.node_from_name(str(row["energy_node_base"]))
        energy_node_by_state[state] = energy_base
        extra_properties[f"E_mean_S{state}"] = energy_base.mean
        extra_properties[f"E_std_S{state}"] = energy_base.std
        if row["force_node_base"] is not None:
            force_base = ensemble_graph.node_from_name(str(row["force_node_base"]))
            force_node_by_state[state] = force_base
            extra_properties[f"F_std_S{state}"] = force_base.std
    for row in gap_table:
        gap_base = ensemble_graph.node_from_name(str(row["gap_node_base"]))
        gap_key = str(row["gap_key"])
        gap_node_by_key[gap_key] = gap_base
        gap_node_by_pair[(int(row["lower_state"]), int(row["upper_state"]))] = gap_base
        extra_properties[f"{gap_key}_mean"] = gap_base.mean
        extra_properties[f"{gap_key}_std"] = gap_base.std
    energy_node = energy_node_by_state[int(selected_state)]

    if sampler_config.get("translate_to_center", False):
        positions = molecule_object.atoms.get_positions() - molecule_object.atoms.get_center_of_mass()
        molecule_object.atoms.set_positions(positions)

    ase_atoms = molecule_object.get_atoms().copy()
    feed = _temperature_feed_parameters(sampler_config, rng)
    dt = float(sampler_config["dt"])
    maxt = float(sampler_config["maxt"])
    ncheck = int(sampler_config["Ncheck"])
    min_time = float(sampler_config.get("min_time", 0.0))
    friction = float(sampler_config.get("friction", 0.02))
    total_md_steps = int(np.ceil((1000.0 * maxt) / dt))
    n_outer = int(np.ceil(float(total_md_steps) / float(ncheck)))
    trajectory_interval = _sample_trajectory_interval(sampler_config, rng)
    meta_dir = sampler_config.get("meta_dir")
    metadata_format = str(sampler_config.get("metadata_format", "pickle"))
    write_xyz = bool(sampler_config.get("write_traj_xyz", False))
    write_binary = bool(sampler_config.get("write_traj_binary", trajectory_interval is not None))
    return_top_n = _return_top_n(sampler_config)
    initial_temperature = annealing_schedule(0.0, maxt, feed["Tamp"], feed["Tper"], feed["Tsrt"], feed["Tend"])
    alchemi_options = None
    alchemi_batch = None
    alchemi_model = None
    alchemi_model_force_model = None
    alchemi_sigma_force_model = None
    if use_alchemi:
        alchemi_options = alchemi_config_from_sampler(
            sampler_config,
            default_seed=stable_uint32_seed(molecule_object.get_moleculeid(), int(current_model_id) + 7919),
        )
        validate_alchemi_sampler_support(ase_atoms, feed, device, alchemi_options)
        alchemi_batch = build_alchemi_batch(ase_atoms, device=device)

    energy_for_md = energy_node.mean
    if use_alchemi:
        alchemi_model = ALFExcitedStateAlchemiModel(
            ensemble_graph=ensemble_graph,
            energy_node=energy_for_md,
            extra_properties=extra_properties,
            force_node=force_node_by_state[int(selected_state)].mean,
            species_key=alchemi_options.species_key,
            coordinates_key=alchemi_options.coordinates_key,
            offset_eV=float(sampler_config.get("energy_offset_eV", 0.0)),
            device=device,
        )
    gap_seeking_settings = _gap_seeking_settings(_gap_seeking_config(sampler_config), state_ids, int(selected_state))
    gap_seeking_enabled = bool(gap_seeking_settings["enabled"])
    gap_seeking_switched = False
    gap_seeking_lcm_active = False
    gap_seeking_switch_step = None
    gap_seeking_switch_time_ps = None
    gap_seeking_pair = None
    gap_seeking_gap_key = None
    gap_seeking_gap_source = None
    gap_seeking_trigger_gap_eV = None
    udd = _udd_config(sampler_config)
    udd_enabled = bool(udd.get("enabled", False))
    udd_tau = float(udd.get("bias_weight", 0.0))
    udd_tau_settings = _udd_tau_settings(udd, total_md_steps) if udd_enabled else _udd_tau_settings({}, total_md_steps)
    udd_tau_mode = str(udd_tau_settings["tau_mode"])
    udd_metadata: dict[str, Any] = {
        "udd_enabled": bool(udd_enabled),
        "udd_mode": str(udd.get("mode", "gap_std_bias")),
        "udd_target": str(udd.get("target", "initial_min_gap_pair")),
        "udd_bias_weight": float(udd_tau),
        "udd_tau_mode": udd_tau_mode,
        "udd_tau_relative": udd_tau_settings["tau_relative"],
        "udd_tau_history_steps": udd_tau_settings["tau_history_steps"],
        "udd_tau_update_interval": udd_tau_settings["tau_update_interval"],
        "udd_tau_min": udd_tau_settings["tau_min"],
        "udd_tau_max": udd_tau_settings["tau_max"],
        "udd_tau_force_epsilon": udd_tau_settings["tau_force_epsilon"],
        "udd_gap_key": None,
        "udd_gap_pair": None,
        "udd_initial_gap_mean": None,
        "udd_initial_gap_std": None,
    }
    model_force_calculator = None
    sigma_force_calculator = None
    if udd_enabled:
        if str(udd_metadata["udd_mode"]).strip().lower() != "gap_std_bias":
            raise ValueError(f"Unsupported excited-state UDD mode: {udd_metadata['udd_mode']}")
        if str(udd_metadata["udd_target"]).strip().lower() != "initial_min_gap_pair":
            raise ValueError(f"Unsupported excited-state UDD target: {udd_metadata['udd_target']}")

        if use_alchemi:
            if alchemi_model is None or alchemi_batch is None:
                raise RuntimeError("ALCHEMI backend was selected without an initialized model and batch.")
            udd_target = _select_initial_udd_gap_target(alchemi_model.results_from_batch(alchemi_batch), gap_table)
        else:
            probe_calculator = _make_hippynn_calculator(
                HippynnCalculator,
                torch,
                energy=energy_node.mean,
                extra_properties=extra_properties,
                sampler_config=sampler_config,
                device=device,
            )
            probe_atoms = ase_atoms.copy()
            probe_atoms.calc = probe_calculator
            probe_atoms.get_potential_energy()
            udd_target = _select_initial_udd_gap_target(dict(probe_atoms.calc.results), gap_table)
            probe_atoms.calc = None
            del probe_calculator

        udd_gap_key = str(udd_target["gap_key"])
        if udd_gap_key not in gap_node_by_key:
            raise ValueError(f"UDD selected gap {udd_gap_key!r}, but no direct ensemble node was loaded for it.")
        if udd_tau_mode == "fixed":
            energy_for_md = energy_node.mean - float(udd_tau) * gap_node_by_key[udd_gap_key].std
        else:
            # Start force-relative UDD unbiased. Tau is estimated from Eq. 14
            # using helper force norms and then applied to subsequent MD chunks.
            energy_for_md = energy_node.mean
            if use_alchemi:
                alchemi_model_force_model = ALFExcitedStateAlchemiModel(
                    ensemble_graph=ensemble_graph,
                    energy_node=energy_node.mean,
                    extra_properties=None,
                    force_node=force_node_by_state[int(selected_state)].mean,
                    species_key=alchemi_options.species_key,
                    coordinates_key=alchemi_options.coordinates_key,
                    offset_eV=float(sampler_config.get("energy_offset_eV", 0.0)),
                    device=device,
                )
                alchemi_sigma_force_model = ALFExcitedStateAlchemiModel(
                    ensemble_graph=ensemble_graph,
                    energy_node=gap_node_by_key[udd_gap_key].std,
                    extra_properties=None,
                    species_key=alchemi_options.species_key,
                    coordinates_key=alchemi_options.coordinates_key,
                    offset_eV=0.0,
                    device=device,
                )
            else:
                model_force_calculator = _make_hippynn_calculator(
                    HippynnCalculator,
                    torch,
                    energy=energy_node.mean,
                    extra_properties=None,
                    sampler_config=sampler_config,
                    device=device,
                )
                sigma_force_calculator = _make_hippynn_calculator(
                    HippynnCalculator,
                    torch,
                    energy=gap_node_by_key[udd_gap_key].std,
                    extra_properties=None,
                    sampler_config=sampler_config,
                    device=device,
                    offset=0.0,
                )
        udd_metadata.update(
            {
                "udd_gap_key": udd_gap_key,
                "udd_gap_pair": udd_target["gap_pair"],
                "udd_initial_gap_mean": float(udd_target["initial_gap_mean"]),
                "udd_initial_gap_std": float(udd_target["initial_gap_std"]),
            }
        )

    if use_alchemi:
        if alchemi_model is None or alchemi_batch is None:
            raise RuntimeError("ALCHEMI backend was selected without an initialized model and batch.")
        alchemi_model.set_energy_node(
            energy_for_md,
            force_node=force_node_by_state[int(selected_state)].mean if energy_for_md is energy_node.mean else None,
        )
        calculator = None
        ase_atoms.calc = None
        dyn = AlchemiBaoabRunner(
            model=alchemi_model,
            batch=alchemi_batch,
            dt_fs=dt,
            temperature_K=initial_temperature,
            friction_per_fs=friction,
            random_seed=int(alchemi_options.random_seed),
            device=device,
        )
    else:
        calculator = _make_hippynn_calculator(
            HippynnCalculator,
            torch,
            energy=energy_for_md,
            extra_properties=extra_properties,
            sampler_config=sampler_config,
            device=device,
        )
        ase_atoms.calc = calculator

        dyn = Langevin(
            ase_atoms,
            dt * units.fs,
            friction=friction,
            temperature_K=initial_temperature,
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

    top_candidates: list[dict[str, Any]] = []
    hard_close_contact = False
    geometry_reject_reason = None
    rejected_qm_candidates_min_distance = 0
    start_time = time.time()
    temperatures: list[float] = []
    total_energies: list[float] = []
    udd_gap_mean_trace: list[float] = []
    udd_gap_std_trace: list[float] = []
    udd_tau_trace: list[float] = []
    udd_tau_raw_trace: list[float] = []
    udd_tau_clipped_trace: list[bool] = []
    udd_bias_force_ratio_trace: list[float] = []
    udd_tau_update_steps: list[int] = []
    udd_model_force_norm_sum_trace: list[float] = []
    udd_sigma_force_norm_sum_trace: list[float] = []
    udd_force_history: list[tuple[float, float]] = []
    last_tau_update_step = 0
    geometry_metrics_trace: list[dict[str, Any]] = []
    gap_seeking_min_gap_trace: list[dict[str, Any]] = []
    gap_seeking_active_pair_gap_trace: list[dict[str, Any]] = []
    gap_seeking_switch_events: list[dict[str, Any]] = []

    def _run_md_steps(steps: int) -> None:
        with timing.scope("md_run", cuda=use_alchemi):
            dyn.run(int(steps))
        timing.increment("num_md_steps_completed", int(steps))
        timing.increment("num_md_chunks", 1)

    def _sync_alchemi_to_atoms() -> None:
        if not use_alchemi:
            return
        with timing.scope("host_sync"):
            dyn.sync_to_atoms(ase_atoms)
        timing.increment("full_sync_count", 1)

    def _evaluate_alchemi_results() -> dict[str, Any]:
        with timing.scope("model_eval", cuda=True):
            results = dyn.evaluate_results()
        timing.increment("scalar_sync_count", 1)
        return results

    def _fresh_ase_results(*, step: int | None, context: str) -> dict[str, Any]:
        with timing.scope("model_eval", cuda=str(device).startswith("cuda")):
            return _fresh_sampler_results(
                ase_atoms,
                state_table,
                selected_state=selected_state,
                step=step,
                context=context,
                gap_seeking_switched=gap_seeking_switched,
                gap_seeking_pair=gap_seeking_pair,
            )

    def _update_force_relative_udd_tau() -> None:
        nonlocal calculator, energy_for_md, last_tau_update_step, udd_tau
        if not (udd_enabled and udd_tau_mode == "force_relative"):
            return
        if gap_seeking_lcm_active:
            return
        if use_alchemi:
            if alchemi_model_force_model is None or alchemi_sigma_force_model is None:
                raise RuntimeError("UDD force_relative mode was enabled without initialized ALCHEMI helper models.")
        elif model_force_calculator is None or sigma_force_calculator is None:
            raise RuntimeError("UDD force_relative mode was enabled without initialized helper calculators.")

        current_step = int(dyn.nsteps)
        if current_step <= 0:
            return
        if use_alchemi:
            with timing.scope("model_eval", cuda=True):
                model_force_norm = alchemi_model_force_model.force_norm_for_energy(dyn.batch, energy_node.mean)
                sigma_force_norm = alchemi_sigma_force_model.force_norm_for_energy(
                    dyn.batch,
                    gap_node_by_key[str(udd_metadata["udd_gap_key"])].std,
                )
            timing.increment("scalar_sync_count", 2)
        else:
            with timing.scope("model_eval", cuda=str(device).startswith("cuda")):
                model_force_norm = _calculator_force_norm(ase_atoms, model_force_calculator)
                sigma_force_norm = _calculator_force_norm(ase_atoms, sigma_force_calculator)
        udd_force_history.append((model_force_norm, sigma_force_norm))
        del udd_force_history[: -int(udd_tau_settings["tau_history_steps"])]

        enough_history = len(udd_force_history) >= int(udd_tau_settings["tau_history_steps"])
        due_for_update = (current_step - last_tau_update_step) >= int(udd_tau_settings["tau_update_interval"])
        if not (enough_history and due_for_update):
            return

        model_sum = float(sum(item[0] for item in udd_force_history))
        sigma_sum = float(sum(item[1] for item in udd_force_history))
        tau_raw = float(udd_tau_settings["tau_relative"]) * model_sum / max(
            sigma_sum,
            float(udd_tau_settings["tau_force_epsilon"]),
        )
        tau_new = float(np.clip(tau_raw, float(udd_tau_settings["tau_min"]), float(udd_tau_settings["tau_max"])))
        clipped = bool(not np.isclose(tau_new, tau_raw))
        bias_force_ratio = tau_new * sigma_sum / max(model_sum, float(udd_tau_settings["tau_force_epsilon"]))

        udd_tau = tau_new
        udd_tau_update_steps.append(int(current_step))
        udd_tau_trace.append(float(tau_new))
        udd_tau_raw_trace.append(float(tau_raw))
        udd_tau_clipped_trace.append(clipped)
        udd_model_force_norm_sum_trace.append(float(model_sum))
        udd_sigma_force_norm_sum_trace.append(float(sigma_sum))
        udd_bias_force_ratio_trace.append(float(bias_force_ratio))
        last_tau_update_step = int(current_step)

        if udd_metadata["udd_gap_key"] is None:
            raise RuntimeError("UDD force_relative mode has no selected gap key.")
        energy_for_md = energy_node.mean - float(udd_tau) * gap_node_by_key[str(udd_metadata["udd_gap_key"])].std
        if use_alchemi:
            alchemi_model.set_energy_node(
                energy_for_md,
                force_node=force_node_by_state[int(selected_state)].mean if energy_for_md is energy_node.mean else None,
            )
            results = _evaluate_alchemi_results()
            _validate_sampler_results(
                results,
                state_table,
                selected_state=selected_state,
                step=current_step,
                context="udd_tau_update",
                gap_seeking_switched=gap_seeking_switched,
                gap_seeking_pair=gap_seeking_pair,
            )
        else:
            calculator = _make_hippynn_calculator(
                HippynnCalculator,
                torch,
                energy=energy_for_md,
                extra_properties=extra_properties,
                sampler_config=sampler_config,
                device=device,
            )
            ase_atoms.calc = calculator
            _fresh_ase_results(step=current_step, context="udd_tau_update")

    def _update_gap_seeking_switch() -> None:
        nonlocal calculator, energy_for_md
        nonlocal gap_seeking_switched, gap_seeking_lcm_active, gap_seeking_switch_step, gap_seeking_switch_time_ps
        nonlocal gap_seeking_pair, gap_seeking_gap_key, gap_seeking_gap_source, gap_seeking_trigger_gap_eV
        if not gap_seeking_enabled:
            return

        current_step = int(dyn.nsteps)
        if current_step <= 0:
            return

        if use_alchemi:
            current_results = _evaluate_alchemi_results()
        else:
            current_results = _fresh_ase_results(step=current_step, context="gap_seeking_monitor")
        gap_infos = _gap_infos_from_results(current_results, list(gap_seeking_settings["pairs"]))
        if not gap_infos:
            raise RuntimeError("Gap-seeking switch found no valid state-energy gaps to monitor.")
        if gap_seeking_lcm_active:
            if str(gap_seeking_settings["switch_policy"]) != "hysteresis":
                return
            if gap_seeking_switch_step is not None and (
                current_step - int(gap_seeking_switch_step)
            ) < int(gap_seeking_settings["min_lcm_steps"]):
                return
            active_gap = None
            if gap_seeking_pair is not None:
                active_pair = [int(gap_seeking_pair[0]), int(gap_seeking_pair[1])]
                for gap_info in gap_infos:
                    if [int(value) for value in gap_info["pair"]] == active_pair:
                        active_gap = gap_info
                        break
            if active_gap is None or float(active_gap["abs_gap_eV"]) < float(gap_seeking_settings["exit_gap_threshold_eV"]):
                return

            gap_seeking_lcm_active = False
            energy_for_md = energy_node.mean
            gap_seeking_switch_events.append(
                {
                    "event": "exit_lcm",
                    "step": int(current_step),
                    "time_ps": float(current_step * dt) / 1000.0,
                    "reason": "exit_gap_threshold",
                    **{key: active_gap[key] for key in ["pair", "gap_key", "gap_source", "gap_eV", "abs_gap_eV"]},
                }
            )
            if use_alchemi:
                alchemi_model.set_energy_node(
                    energy_for_md,
                    force_node=force_node_by_state[int(selected_state)].mean,
                )
                results = _evaluate_alchemi_results()
                _validate_sampler_results(
                    results,
                    state_table,
                    selected_state=selected_state,
                    step=current_step,
                    context="gap_seeking_exit",
                    gap_seeking_switched=gap_seeking_switched,
                    gap_seeking_pair=gap_seeking_pair,
                )
            else:
                calculator = _make_hippynn_calculator(
                    HippynnCalculator,
                    torch,
                    energy=energy_for_md,
                    extra_properties=extra_properties,
                    sampler_config=sampler_config,
                    device=device,
                )
                ase_atoms.calc = calculator
                _fresh_ase_results(step=current_step, context="gap_seeking_exit")
            return

        trigger = min(gap_infos, key=lambda item: float(item["abs_gap_eV"]))
        if float(trigger["abs_gap_eV"]) > float(gap_seeking_settings["trigger_gap_threshold_eV"]):
            return

        gap_seeking_switched = True
        gap_seeking_lcm_active = True
        gap_seeking_switch_step = int(current_step)
        gap_seeking_switch_time_ps = float(current_step * dt) / 1000.0
        gap_seeking_pair = [int(value) for value in trigger["pair"]]
        gap_seeking_gap_key = str(trigger["gap_key"])
        gap_seeking_gap_source = str(trigger["gap_source"])
        gap_seeking_trigger_gap_eV = float(trigger["gap_eV"])
        gap_seeking_switch_events.append(
            {
                "event": "enter_lcm",
                "step": int(current_step),
                "time_ps": float(current_step * dt) / 1000.0,
                **{key: trigger[key] for key in ["pair", "gap_key", "gap_source", "gap_eV", "abs_gap_eV"]},
            }
        )
        energy_for_md = _make_lcm_energy_node(
            energy_node_by_state,
            gap_node_by_pair,
            gap_seeking_pair,
            sigma=float(gap_seeking_settings["sigma"]),
            alpha_eV=float(gap_seeking_settings["alpha_eV"]),
        )
        if use_alchemi:
            lower_state, upper_state = int(gap_seeking_pair[0]), int(gap_seeking_pair[1])
            gap_base = gap_node_by_pair.get((lower_state, upper_state))
            alchemi_model.set_lcm_gap_mode(
                lower_energy_node=energy_node_by_state[lower_state].mean,
                upper_energy_node=energy_node_by_state[upper_state].mean,
                lower_force_node=force_node_by_state[lower_state].mean,
                upper_force_node=force_node_by_state[upper_state].mean,
                gap_node=gap_base.mean if gap_base is not None else None,
                sigma=float(gap_seeking_settings["sigma"]),
                alpha_eV=float(gap_seeking_settings["alpha_eV"]),
            )
            results = _evaluate_alchemi_results()
            _validate_sampler_results(
                results,
                state_table,
                selected_state=selected_state,
                step=current_step,
                context="gap_seeking_switch",
                gap_seeking_switched=gap_seeking_switched,
                gap_seeking_pair=gap_seeking_pair,
            )
        else:
            calculator = _make_hippynn_calculator(
                HippynnCalculator,
                torch,
                energy=energy_for_md,
                extra_properties=extra_properties,
                sampler_config=sampler_config,
                device=device,
            )
            ase_atoms.calc = calculator
            _fresh_ase_results(step=current_step, context="gap_seeking_switch")

    if udd_enabled and udd_tau_mode == "force_relative":
        dyn.attach(_update_force_relative_udd_tau, interval=1)
    if gap_seeking_enabled:
        dyn.attach(_update_gap_seeking_switch, interval=int(gap_seeking_settings["switch_check_interval"]))

    try:
        _run_md_steps(1)
        _sync_alchemi_to_atoms()
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

            _run_md_steps(ncheck)
            _sync_alchemi_to_atoms()
            with timing.scope("trajectory_io"):
                if traj_writer is not None:
                    traj_writer.write(ase_atoms)
                if xyz_handle is not None:
                    write(xyz_handle, ase_atoms, format="xyz")

            current_step = int((step_index + 1) * ncheck)
            if use_alchemi:
                results = _evaluate_alchemi_results()
                _validate_sampler_results(
                    results,
                    state_table,
                    selected_state=selected_state,
                    step=current_step,
                    context="post_md_chunk",
                    gap_seeking_switched=gap_seeking_switched,
                    gap_seeking_pair=gap_seeking_pair,
                )
                with timing.scope("metrics"):
                    metrics = _results_to_metrics(
                        ase_atoms,
                        results,
                        state_table,
                        gap_table,
                        selected_state,
                        forces_override=np.asarray(results["forces"], dtype=np.float64),
                    )
            else:
                results = _fresh_ase_results(step=current_step, context="post_md_chunk")
                with timing.scope("metrics"):
                    metrics = _results_to_metrics(ase_atoms, results, state_table, gap_table, selected_state)
            timing.record_chunk(step=current_step, chunk_steps=int(ncheck))
            if udd_enabled and udd_metadata["udd_gap_key"] is not None:
                udd_gap_key = str(udd_metadata["udd_gap_key"])
                if udd_gap_key in metrics["gap_means"]:
                    udd_gap_mean_trace.append(float(metrics["gap_means"][udd_gap_key]))
                if udd_gap_key in metrics["gap_stds"]:
                    udd_gap_std_trace.append(float(metrics["gap_stds"][udd_gap_key]))
            if gap_seeking_enabled:
                gap_infos = _gap_infos_from_metric_data(
                    metrics,
                    list(gap_seeking_settings["pairs"]),
                )
                if gap_infos:
                    min_gap_info = min(gap_infos, key=lambda item: float(item["abs_gap_eV"]))
                    gap_seeking_min_gap_trace.append(
                        {
                            "step": int(current_step),
                            "time_ps": float(current_step * dt) / 1000.0,
                            **min_gap_info,
                        }
                    )
                    if gap_seeking_pair is not None:
                        active_pair = [int(gap_seeking_pair[0]), int(gap_seeking_pair[1])]
                        for gap_info in gap_infos:
                            if [int(value) for value in gap_info["pair"]] == active_pair:
                                gap_seeking_active_pair_gap_trace.append(
                                    {
                                        "step": int(current_step),
                                        "time_ps": float(current_step * dt) / 1000.0,
                                        **gap_info,
                                    }
                                )
                                break
            if use_alchemi:
                with timing.scope("model_eval", cuda=True):
                    current_temperature = float(dyn.temperature_K())
                    current_total_energy = float(results["energy"] + dyn.kinetic_energy_eV())
                timing.increment("scalar_sync_count", 2)
            else:
                current_temperature = float(ase_atoms.get_temperature())
                current_total_energy = float(ase_atoms.get_potential_energy() + ase_atoms.get_kinetic_energy())
            temperatures.append(current_temperature)
            total_energies.append(current_total_energy)
            geometry_metrics_trace.append(
                {
                    "step": int(current_step),
                    "time_ps": float(current_time_ps),
                    "min_dist": float(metrics["min_dist"]),
                    "fmax": float(metrics["fmax"]),
                    "max_nearest_neighbor_distance": float(metrics["max_nearest_neighbor_distance"]),
                    "nearest_neighbor_distances": np.asarray(metrics["nearest_neighbor_distances"], dtype=float),
                }
            )

            if float(metrics["min_dist"]) < float(sampler_config.get("min_distance_cutoff", 0.3)):
                hard_close_contact = True
                geometry_reject_reason = "min_distance"
                break
            if bool(sampler_config.get("max_nearest_neighbor_distance_check", False)):
                max_nn_cutoff = float(sampler_config["max_nearest_neighbor_distance_cutoff"])
                if float(metrics["max_nearest_neighbor_distance"]) > max_nn_cutoff:
                    geometry_reject_reason = "max_nearest_neighbor_distance"
                    break
            if float(metrics["fmax"]) > float(sampler_config.get("max_force_cutoff", 10.0)):
                break

            if current_time_ps < min_time:
                continue
            if not _passes_uncertainty_gate(metrics, sampler_config):
                continue

            score_gap_uncertainty_key = None
            if float(_score_config(sampler_config).get("w_gap_uncertainty", 0.0)) != 0.0:
                if udd_metadata["udd_gap_key"] is not None:
                    score_gap_uncertainty_key = str(udd_metadata["udd_gap_key"])
            score_components = compute_excited_state_score_components(
                metrics,
                sampler_config,
                udd_gap_key=score_gap_uncertainty_key,
            )
            score = float(sum(score_components.values()))
            score_gap_uncertainty_std = None
            if score_gap_uncertainty_key is not None:
                score_gap_uncertainty_std = float(metrics["gap_stds"][score_gap_uncertainty_key])
            candidate_record = {
                    "score": float(score),
                    "score_components": score_components,
                    **_score_uncertainty_metadata(sampler_config),
                    "score_gap_uncertainty_key": score_gap_uncertainty_key,
                    "score_gap_uncertainty_std": score_gap_uncertainty_std,
                    "step": int((step_index + 1) * ncheck),
                    "time_ps": float(current_time_ps),
                    "selected_state": int(selected_state),
                    "gap_seeking_switched": bool(gap_seeking_switched),
                    "gap_seeking_current_mode": "lcm_gap" if gap_seeking_lcm_active else "direct",
                    "gap_seeking_pair": gap_seeking_pair,
                    "gap_seeking_gap_key": gap_seeking_gap_key,
                    "gap_seeking_gap_source": gap_seeking_gap_source,
                    "temperature_K": float(current_temperature),
                    "temperature_target_K": float(target_temperature),
                    **metrics,
                }
            candidate_atoms = ase_atoms.copy()
            if _qm_candidate_reject_reason(candidate_atoms, sampler_config) == "min_distance":
                rejected_qm_candidates_min_distance += 1
                continue
            top_candidates.append({"record": candidate_record, "atoms": candidate_atoms})
            top_candidates.sort(key=lambda item: float(item["record"]["score"]), reverse=True)
            del top_candidates[return_top_n:]
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

    best_record = dict(top_candidates[0]["record"]) if top_candidates else None
    top_candidate_records = [dict(candidate["record"]) for candidate in top_candidates]
    last_geometry_metrics = dict(geometry_metrics_trace[-1]) if geometry_metrics_trace else {}
    realtime_simulation = float(time.time() - start_time)
    if hasattr(dyn, "nsteps"):
        timing.set_count("num_md_steps_completed", int(getattr(dyn, "nsteps", 0)))
    timing_metadata = timing.metadata(total_wall_s=realtime_simulation, num_atoms=len(ase_atoms))
    meta_dict = {
        "realtime_simulation": realtime_simulation,
        "selected_state": int(selected_state),
        "best_candidate": best_record,
        "top_candidates": top_candidate_records,
        "return_top_n": int(return_top_n),
        "hard_close_contact": bool(hard_close_contact),
        "geometry_reject_reason": geometry_reject_reason,
        "rejected_qm_candidates_min_distance": int(rejected_qm_candidates_min_distance),
        "geometry_metrics_trace": geometry_metrics_trace,
        "max_nearest_neighbor_distance": last_geometry_metrics.get("max_nearest_neighbor_distance"),
        "nearest_neighbor_distances": last_geometry_metrics.get("nearest_neighbor_distances"),
        "trajectory_temperature_trace_K": temperatures,
        "trajectory_total_energy_trace_eV": total_energies,
        "density_trace": density_trace if "density_trace" in locals() else [],
        "temperature_feed_parameters": feed,
        "energy_drift_eV_per_ps": drift_eV_per_ps,
        "dynamics_backend": dynamics_backend,
        "alchemi_baoab_options": dict(vars(alchemi_options)) if use_alchemi and alchemi_options is not None else None,
        **_score_uncertainty_metadata(sampler_config),
        "chemical_symbols": ase_atoms.get_chemical_symbols(),
        "positions": ase_atoms.get_positions(wrap=True),
        "cell": ase_atoms.get_cell(),
        "gap_seeking_enabled": bool(gap_seeking_enabled),
        "gap_seeking_mode": str(gap_seeking_settings["mode"]),
        "gap_seeking_candidate_pairs": str(gap_seeking_settings["candidate_pairs"]),
        "gap_seeking_switch_policy": str(gap_seeking_settings["switch_policy"]),
        "gap_seeking_trigger_gap_threshold_eV": float(gap_seeking_settings["trigger_gap_threshold_eV"]),
        "gap_seeking_exit_gap_threshold_eV": gap_seeking_settings["exit_gap_threshold_eV"],
        "gap_seeking_min_lcm_steps": int(gap_seeking_settings["min_lcm_steps"]),
        "gap_seeking_sigma": float(gap_seeking_settings["sigma"]),
        "gap_seeking_alpha_eV": float(gap_seeking_settings["alpha_eV"]),
        "gap_seeking_switch_check_interval": int(gap_seeking_settings["switch_check_interval"]),
        "gap_seeking_switched": bool(gap_seeking_switched),
        "gap_seeking_current_mode": "lcm_gap" if gap_seeking_lcm_active else "direct",
        "gap_seeking_switch_step": gap_seeking_switch_step,
        "gap_seeking_switch_time_ps": gap_seeking_switch_time_ps,
        "gap_seeking_pair": gap_seeking_pair,
        "gap_seeking_gap_key": gap_seeking_gap_key,
        "gap_seeking_gap_source": gap_seeking_gap_source,
        "gap_seeking_trigger_gap_eV": gap_seeking_trigger_gap_eV,
        "gap_seeking_switch_events": gap_seeking_switch_events,
        "gap_seeking_min_gap_trace": gap_seeking_min_gap_trace,
        "gap_seeking_active_pair_gap_trace": gap_seeking_active_pair_gap_trace,
        "udd_gap_mean_trace": udd_gap_mean_trace,
        "udd_gap_std_trace": udd_gap_std_trace,
        "udd_final_tau": float(udd_tau) if udd_enabled else None,
        "udd_tau_update_steps": udd_tau_update_steps,
        "udd_tau_trace": udd_tau_trace,
        "udd_tau_raw_trace": udd_tau_raw_trace,
        "udd_tau_clipped_trace": udd_tau_clipped_trace,
        "udd_model_force_norm_sum_trace": udd_model_force_norm_sum_trace,
        "udd_sigma_force_norm_sum_trace": udd_sigma_force_norm_sum_trace,
        "udd_bias_force_ratio_trace": udd_bias_force_ratio_trace,
    }
    if timing_metadata is not None:
        meta_dict["timing"] = timing_metadata
    meta_dict.update(udd_metadata)
    meta_dict.update(molecule_object.get_metadata())
    _write_metadata(meta_dir, molecule_object.get_moleculeid(), meta_dict, metadata_format)

    ase_atoms.calc = None
    if return_top_n == 1:
        molecule_object.update_metadata(meta_dict)
        if top_candidates and geometry_reject_reason is None:
            selected_atoms = top_candidates[0]["atoms"]
            selected_atoms.calc = None
            _write_qm_candidate_xyz(
                sampler_config,
                molecule_object.get_moleculeid(),
                [top_candidates[0]],
                [molecule_object.get_moleculeid()],
            )
            molecule_object.update_atoms(selected_atoms)
        else:
            molecule_object.update_atoms(None)
        return molecule_object

    if geometry_reject_reason is not None:
        return []

    output_candidates: list[MoleculesObject] = []
    candidate_ids = [f"{molecule_object.get_moleculeid()}-cand-{rank:02d}" for rank in range(len(top_candidates))]
    _write_qm_candidate_xyz(sampler_config, molecule_object.get_moleculeid(), top_candidates, candidate_ids)
    for rank, candidate in enumerate(top_candidates):
        selected_atoms = clean_atoms(candidate["atoms"])
        candidate_record = dict(candidate["record"])
        candidate_metadata = dict(meta_dict)
        candidate_metadata.update(
            {
                "parent_molecule_id": molecule_object.get_moleculeid(),
                "candidate_rank": int(rank),
                "candidate_score": float(candidate_record["score"]),
                "candidate_step": int(candidate_record["step"]),
                "candidate_time_ps": float(candidate_record["time_ps"]),
                "candidate_record": candidate_record,
            }
        )
        candidate_molecule = MoleculesObject(
            selected_atoms,
            candidate_ids[rank],
        )
        candidate_molecule.update_metadata(plain_metadata_dict(candidate_metadata))
        output_candidates.append(candidate_molecule)
    return output_candidates


@python_app(executors=["alf_sampler_executor"])
def excited_state_sampling_task(
    molecule_object,
    sampler_config,
    model_path,
    current_model_id,
    gpus_per_node,
    properties_list,
    molecule_objects=None,
):
    local_sampler_config = dict(sampler_config or {})
    try:
        if molecule_objects is not None:
            result = run_excited_state_sampling_batch(
                molecule_objects=list(molecule_objects),
                sampler_config=local_sampler_config,
                model_path=str(model_path),
                current_model_id=int(current_model_id),
                gpus_per_node=int(gpus_per_node),
                properties_list=dict(properties_list or {}),
            )
            return write_sampler_result_ref(result, local_sampler_config)
        result = run_excited_state_sampling(
            molecule_object=molecule_object,
            sampler_config=local_sampler_config,
            model_path=str(model_path),
            current_model_id=int(current_model_id),
            gpus_per_node=int(gpus_per_node),
            properties_list=dict(properties_list or {}),
        )
        return write_sampler_result_ref(result, local_sampler_config)
    except BaseException as exc:
        return write_sampler_error_ref(exc, local_sampler_config)


@python_app(executors=["alf_gpu_executor"])
def excited_state_sampling_gpu_task(
    molecule_object,
    sampler_config,
    model_path,
    current_model_id,
    gpus_per_node,
    properties_list,
    molecule_objects=None,
):
    local_sampler_config = dict(sampler_config or {})
    try:
        if molecule_objects is not None:
            result = run_excited_state_sampling_batch(
                molecule_objects=list(molecule_objects),
                sampler_config=local_sampler_config,
                model_path=str(model_path),
                current_model_id=int(current_model_id),
                gpus_per_node=int(gpus_per_node),
                properties_list=dict(properties_list or {}),
            )
            return write_sampler_result_ref(result, local_sampler_config)
        result = run_excited_state_sampling(
            molecule_object=molecule_object,
            sampler_config=local_sampler_config,
            model_path=str(model_path),
            current_model_id=int(current_model_id),
            gpus_per_node=int(gpus_per_node),
            properties_list=dict(properties_list or {}),
        )
        return write_sampler_result_ref(result, local_sampler_config)
    except BaseException as exc:
        return write_sampler_error_ref(exc, local_sampler_config)
