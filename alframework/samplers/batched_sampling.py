"""Backend-neutral batched ALF acquisition and dynamics task.

Model inference and dynamics adapters share this acquisition loop so physical
screening, trajectory clocks, state selection, and ranking stay consistent.
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


from alframework.tools.batched_config import (
    batching_settings, dynamics_settings, configured_native_loader,
    configured_native_options,
)


def dynamics_runner(backend):
    """Import only the selected optional dynamics engine."""
    if backend == "alchemi":
        from alframework.samplers.alchemi_sampling import AlchemiDynamicsRunner
        return AlchemiDynamicsRunner
    if backend == "torchsim":
        from alframework.samplers.torchsim_sampling import TorchSimDynamicsRunner
        return TorchSimDynamicsRunner
    raise ValueError(f"Unknown dynamics backend: {backend!r}.")


def calculate_uncertainty(
    energy_contributions: Any,
    force_contributions: Any,
) -> dict[str, float]:
    """Calculate the exact population statistics used by production MLMD."""

    energies, forces = validated_numpy_contributions(energy_contributions, force_contributions)
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

    energy_cutoff, force_cutoff = uncertainty_cutoffs(Escut, Fscut)
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
    lower, upper = validate_numeric_range(value, name)
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
        validate_temperature_period(row["Tper"], replica_index)
        parameters.append(row)
    return parameters


def _minimum_distance(atoms) -> float:
    distances = np.asarray(atoms.get_all_distances(mic=bool(np.any(atoms.pbc))), dtype=float)
    np.fill_diagonal(distances, np.inf)
    minimum = float(distances.min()) if distances.size else float("inf")
    if np.any(atoms.pbc):
        # Include neighboring images of the same atom, including primitive
        # one-atom cells. Only the zero-translation self pair is excluded.
        from ase.geometry import minkowski_reduce
        reduced, _ = minkowski_reduce(atoms.cell, pbc=atoms.pbc)
        lengths = np.linalg.norm(np.asarray(reduced)[atoms.pbc], axis=1)
        minimum = min(minimum, float(lengths.min()))
    return minimum


def _density_parameters(sampler_config, atoms_list, random_seed):
    if sampler_config.get("end_dens") is None:
        return None
    rng = np.random.default_rng(int(random_seed) + 1)
    rows = []
    for atoms in atoms_list:
        row = {
            "Rsrt": 1.66054 * float(atoms.get_masses().sum()) / atoms.get_volume(),
            "Rend": _sample_range(rng, sampler_config["end_dens"], "end_dens"),
            "Ramp": (0.0 if sampler_config.get("amp_dens") is None else
                     _sample_range(rng, sampler_config["amp_dens"], "amp_dens")),
            "Rper": (1.0 if sampler_config.get("per_dens") is None else
                     _sample_range(rng, sampler_config["per_dens"], "per_dens")),
        }
        if row["Rend"] <= 0 or row["Rper"] <= 0 or row["Ramp"] < 0:
            raise ValueError("Density targets and periods must be positive; amplitude must be nonnegative.")
        rows.append(row)
    return rows


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    record = candidate["record"]
    return (
        -float(record["ranking_score"]),
        int(record["step"]),
        int(record["batch_index"]),
        str(record["parent_molecule_id"]),
    )


def candidate_score(
    diagnostics: dict[str, Any],
    flags: dict[str, Any],
    score: dict[str, Any],
    *,
    Escut: float,
    Fscut: float,
) -> dict[str, Any]:
    """Return the ranking score for one candidate frame.

    ``max`` mode returns ALF's existing worst-violation score unchanged. Each
    ``sum`` term is normalized by its own cutoff, so a value of one means "at
    threshold" for that channel and the weights express only relative
    importance.
    """

    if str(score.get("mode", "max")).strip().lower() != "sum":
        return {
            "ranking_score": float(flags["uncertainty_score"]),
            "score_mode": "max",
            "score_components": {},
            "score_gap_pair": None,
            "score_gap_std": None,
        }

    weights = dict(score["weights"])
    gap_values = validate_score_diagnostics(diagnostics, score)
    components = {
        "energy_uncertainty": (
            weights["w_energy"] * float(diagnostics["Es"]) / float(Escut)
        ),
        "force_uncertainty": (
            weights["w_force"] * float(diagnostics["Fsrms"]) / float(Fscut)
        ),
        "gap_uncertainty": 0.0,
    }
    gap_pair: list[int] | None = None
    gap_std: float | None = None
    if weights["w_gap_uncertainty"] > 0:
        gap_std, row = max(gap_values, key=lambda item: item[0])
        gap_pair = [int(row["lower_state"]), int(row["upper_state"])]
        components["gap_uncertainty"] = (
            weights["w_gap_uncertainty"]
            * gap_std
            / float(score["gap_std_cut_eV"])
        )
    return {
        "ranking_score": float(sum(components.values())),
        "score_mode": "sum",
        "score_components": {
            key: float(value) for key, value in components.items()
        },
        "score_gap_pair": gap_pair,
        "score_gap_std": None if gap_std is None else float(gap_std),
    }


def _prune_candidates(
    candidates: list[dict[str, Any]],
    return_top_k: int,
    max_candidates_per_replica: int | None,
) -> list[dict[str, Any]]:
    """Rank candidates, cap each trajectory, then apply the global top-K limit.

    ``return_top_k`` alone is a global budget: under ``uncertainty_policy``
    ``continue`` one diverging replica can occupy every slot with frames a
    single check apart. Capping per replica first keeps the returned batch
    spread across distinct starting structures.
    """

    ranked = sorted(candidates, key=_candidate_sort_key)
    if max_candidates_per_replica is not None:
        limit = int(max_candidates_per_replica)
        counts: dict[str, int] = {}
        kept: list[dict[str, Any]] = []
        for candidate in ranked:
            parent = str(candidate["record"]["parent_molecule_id"])
            if counts.get(parent, 0) >= limit:
                continue
            counts[parent] = counts.get(parent, 0) + 1
            kept.append(candidate)
        ranked = kept
    return ranked[: int(return_top_k)]


def select_gap_seeking_trigger(
    diagnostics: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any] | None:
    """Select the deterministic, minimum-absolute eligible gap crossing."""

    if not bool(settings.get("enabled", False)):
        return None
    candidates = []
    for row in settings["rows"]:
        gap_key = str(row["gap_key"])
        gap = validated_seeking_gap(diagnostics, gap_key)
        candidates.append(
            {
                "pair": [
                    int(row["lower_state"]),
                    int(row["upper_state"]),
                ],
                "gap_key": gap_key,
                "gap_eV": gap,
                "abs_gap_eV": abs(gap),
            }
        )
    trigger = min(
        candidates,
        key=lambda item: (
            float(item["abs_gap_eV"]),
            int(item["pair"][0]),
            int(item["pair"][1]),
        ),
    )
    if float(trigger["abs_gap_eV"]) > float(
        settings["trigger_gap_threshold_eV"]
    ):
        return None
    return trigger


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
    gap_seeking_state: dict[str, Any],
    score: dict[str, Any] | None = None,
    density_parameters: dict[str, float] | None = None,
    density_history: list[float] | None = None,
) -> dict[str, Any]:
    seeking_metadata = {
        "gap_seeking_enabled": bool(gap_seeking_state["enabled"]),
        "gap_seeking_current_mode": str(
            gap_seeking_state["current_mode"]
        ),
        "gap_seeking_switched": bool(gap_seeking_state["switched"]),
        "gap_seeking_pair": gap_seeking_state["pair"],
        "gap_seeking_trigger_gap_eV": gap_seeking_state[
            "trigger_gap_eV"
        ],
        "gap_seeking_trigger_step": gap_seeking_state["trigger_step"],
        "gap_seeking_trigger_time_ps": gap_seeking_state[
            "trigger_time_ps"
        ],
        "gap_seeking_switch_events": [
            dict(event) for event in gap_seeking_state["events"]
        ],
    }
    if gap_seeking_state["enabled"]:
        seeking_metadata.update(
            {
                "gap_seeking_mode": str(gap_seeking_state["mode"]),
                "gap_seeking_switch_policy": "stay_fixed",
                "gap_seeking_trigger_gap_threshold_eV": float(
                    gap_seeking_state["trigger_gap_threshold_eV"]
                ),
                "gap_seeking_sigma": float(gap_seeking_state["sigma"]),
                "gap_seeking_alpha_eV": float(
                    gap_seeking_state["alpha_eV"]
                ),
            }
        )
    score_metadata = candidate_score(
        diagnostics,
        flags,
        score if score is not None else {"mode": "max"},
        Escut=float(sampler_config["Escut"]),
        Fscut=float(sampler_config["Fscut"]),
    )
    record = {
        "parent_molecule_id": molecule.get_moleculeid(),
        "batch_index": int(batch_index),
        "step": int(step),
        "time_ps": float(time_ps),
        "uncertainty_policy": policy,
        "uncertainty_score": float(flags["uncertainty_score"]),
        **score_metadata,
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
        **seeking_metadata,
        **_gap_candidate_metadata(diagnostics),
        **temperature_parameters,
    }
    record.update(force_screen_metadata(diagnostics))
    if density_parameters is not None:
        record.update(density_parameters)
        record["denss"] = list(density_history or [])
    if "Fsrms" in diagnostics:
        record["Fsrms"] = float(diagnostics["Fsrms"])
    return record


def run_batched_sampling(
    molecule_objects: list[MoleculesObject],
    sampler_config: dict[str, Any],
    model: Any,
    *,
    properties_list: dict[str, Any] | None = None,
    device: Any = None,
    runner_factory: Callable[..., Any] | None = None,
    master_directory: str | None = None,
) -> list[MoleculesObject]:
    """Run one ALF batch using shared stop/continue acquisition policies."""

    validate_gap_model(sampler_config, properties_list, model)
    mode, selected_state_value = _validate_sampling_inputs(
        molecule_objects,
        sampler_config,
        properties_list,
    )
    periodic = bool(np.any(molecule_objects[0].get_atoms().pbc))
    template = getattr(model, "model_config", None)
    if periodic and not getattr(model, "supports_pbc", getattr(template, "supports_pbc", False)):
        raise ValueError("The configured calculator does not support periodic systems.")
    if not periodic and getattr(model, "requires_pbc", getattr(template, "needs_pbc", False)):
        raise ValueError("The configured periodic calculator requires periodic input cells.")
    if properties_list is not None and hasattr(model, "require_physical_states"):
        model.require_physical_states(_physical_force_states(mode, selected_state_value, properties_list))
    gap_seeking_settings = gap_seeking_settings_for_model(
        sampler_config, properties_list, model, selected_state_value)
    policy = str(sampler_config.get("uncertainty_policy", "stop")).strip().lower()
    backend = dynamics_settings(sampler_config)
    random_seed = int(backend.get("random_seed", 42))
    dt = float(sampler_config["dt"])
    maxt = float(sampler_config["maxt"])
    ncheck = int(sampler_config["Ncheck"])
    min_time = float(sampler_config.get("min_time", 0.0))
    validate_dynamics_parameters(dt, maxt, ncheck, min_time)
    return_top_k = _configured_return_top_k(sampler_config)
    max_candidates_per_replica = _configured_replica_candidate_limit(sampler_config)
    score = configured_score(
        sampler_config,
        properties_list,
        selected_state_value,
    )

    atoms_list = [molecule.get_atoms().copy() for molecule in molecule_objects]
    if sampler_config.get("translate_to_center", False):
        for atoms in atoms_list:
            atoms.set_positions(atoms.get_positions() - atoms.get_center_of_mass())
    reference_topology = load_reference_topology(
        sampler_config,
        master_directory=master_directory,
    )
    if reference_topology is not None:
        atoms_list = [canonicalize_tagged_atoms(atoms) for atoms in atoms_list]
        for atoms in atoms_list:
            if np.array_equal(atoms.numbers, reference_topology.atomic_numbers):
                attach_topology_ids(atoms, reference_topology.atomic_numbers)
    audit = ForceScreenAudit(sampler_config.get("force_screen_audit_dir"), master_directory,
                             molecule_objects, atoms_list, sampler_config.get("max_force_cutoff"))
    try:
        topology_results: list[TopologyValidationResult] = [
            validate_fixed_topology(atoms, reference_topology)
            for atoms in atoms_list
        ]
        validate_topology_identities(topology_results)
        active = np.asarray(
            [result.valid for result in topology_results],
            dtype=bool,
        )
        topology_rejections: list[dict[str, Any]] = [
            {
                "parent_molecule_id": molecule_objects[index].get_moleculeid(),
                "batch_index": int(index),
                "stage": "initial",
                "step": 0,
                "time_ps": 0.0,
                "reason": result.reason,
                "violations": [
                    dict(violation) for violation in result.violations
                ],
            }
            for index, result in enumerate(topology_results)
            if not result.valid
        ]
        for index, result in enumerate(topology_results):
            if not result.valid:
                audit.record(index, "rejected", atoms_list[index], 0, 0.0, reason=result.reason)
        if reference_topology is not None and not np.any(active):
            print(
                "ALF batched topology summary: "
                f"{len(topology_rejections)} initial replica(s) rejected; "
                "dynamics and uncertainty evaluation skipped."
            )
            return []
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
        if not np.isfinite(initial_temperatures).all() or np.any(initial_temperatures < 0):
            raise ValueError("Temperature schedules must produce finite nonnegative temperatures.")
        density_parameters = _density_parameters(sampler_config, atoms_list, random_seed)
        density_histories = [[] for _ in molecule_objects]
        friction_per_fs = float(
            sampler_config.get("friction_per_fs", backend.get("friction_per_fs", DEFAULT_FRICTION_PER_FS))
        )
        validate_friction(friction_per_fs)
        if runner_factory is None:
            runner_factory = dynamics_runner(backend["backend"])
        runner = runner_factory(
            model=model,
            atoms_list=atoms_list,
            dt_fs=dt,
            temperature_K=initial_temperatures,
            friction_per_fs=friction_per_fs,
            random_seed=random_seed,
            device=device,
        )

        for index, result in enumerate(topology_results):
            if result.valid:
                continue
            runner.freeze_graph(index)
        temperature_histories: list[list[float]] = [[] for _ in molecule_objects]
        gap_seeking_states: list[dict[str, Any]] = []
        for _ in molecule_objects:
            gap_seeking_states.append(
                {
                    "enabled": bool(gap_seeking_settings.get("enabled", False)),
                    "mode": gap_seeking_settings.get("mode"),
                    "current_mode": "direct",
                    "switched": False,
                    "pair": None,
                    "trigger_gap_eV": None,
                    "trigger_step": None,
                    "trigger_time_ps": None,
                    "trigger_gap_threshold_eV": gap_seeking_settings.get(
                        "trigger_gap_threshold_eV"
                    ),
                    "sigma": gap_seeking_settings.get("sigma"),
                    "alpha_eV": gap_seeking_settings.get("alpha_eV"),
                    "events": [],
                }
            )
        candidates: list[dict[str, Any]] = []
        n_outer = int(np.ceil((1000.0 * maxt) / (dt * ncheck)))
        # Preserve molecular MLMD's legacy clock: advance once, then label the
        # first uncertainty check and thermostat update as time zero.
        if np.any(active):
            runner.run(1)
        for iteration in range(n_outer):
            if not np.any(active):
                break
            time_ps = float(iteration * ncheck * dt / 1000.0)
            for batch_index in np.where(active)[0]:
                index = int(batch_index)
                runner.sync_graph_to_atoms(index, atoms_list[index])
                topology_result = validate_fixed_topology(
                    atoms_list[index],
                    reference_topology,
                )
                topology_results[index] = topology_result
                if not topology_result.valid:
                    audit.record(index, "rejected", atoms_list[index], runner.nsteps, time_ps,
                                 reason=topology_result.reason)
                    active[index] = False
                    runner.freeze_graph(index)
                    topology_rejections.append(
                        {
                            "parent_molecule_id": (
                                molecule_objects[index].get_moleculeid()
                            ),
                            "batch_index": index,
                            "stage": "ncheck",
                            "step": int(runner.nsteps),
                            "time_ps": float(time_ps),
                            "reason": topology_result.reason,
                            "violations": [
                                dict(violation)
                                for violation in topology_result.violations
                            ],
                        }
                    )
                    continue
            if not np.any(active):
                break
            runner.evaluate()
            diagnostics_by_index: dict[int, dict[str, Any]] = {}
            for batch_index in np.where(active)[0]:
                index = int(batch_index)
                topology_result = topology_results[index]
                diagnostics = dict(runner.diagnostics_for_graph(index))
                diagnostics_by_index[index] = diagnostics
                flags = uncertainty_flags(
                    diagnostics,
                    Escut=float(sampler_config["Escut"]),
                    Fscut=float(sampler_config["Fscut"]),
                )
                distance = _minimum_distance(atoms_list[index])
                if distance < float(sampler_config.get("distcut", 1.2)):
                    audit.record(index, "rejected", atoms_list[index], runner.nsteps, time_ps,
                                 diagnostics, "distance_cutoff")
                    active[index] = False
                    runner.freeze_graph(index)
                    continue
                try:
                    rejected = physical_force_exceeds(diagnostics, sampler_config.get("max_force_cutoff"))
                except (ValueError, TypeError):
                    audit.record(index, "rejected", atoms_list[index], runner.nsteps, time_ps,
                                 diagnostics, "invalid_physical_force_output")
                    raise
                if rejected:
                    audit.record(index, "rejected", atoms_list[index], runner.nsteps, time_ps,
                                 diagnostics, "physical_force_cutoff")
                    active[index] = False
                    runner.freeze_graph(index)
                    continue
                audit.record(index, "last_valid", atoms_list[index], runner.nsteps, time_ps, diagnostics)
                if time_ps < min_time:
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
                    gap_seeking_state=gap_seeking_states[index],
                    score=score,
                    density_parameters=(None if density_parameters is None else density_parameters[index]),
                    density_history=density_histories[index],
                )
                if reference_topology is not None:
                    record.update(topology_metadata(topology_result))
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
                    candidates[:] = _prune_candidates(
                        candidates,
                        return_top_k,
                        max_candidates_per_replica,
                    )

            if not np.any(active):
                break
            switched_this_check = False
            if bool(gap_seeking_settings.get("enabled", False)):
                for batch_index in np.where(active)[0]:
                    index = int(batch_index)
                    state = gap_seeking_states[index]
                    if state["switched"]:
                        continue
                    trigger = select_gap_seeking_trigger(
                        diagnostics_by_index[index],
                        gap_seeking_settings,
                    )
                    if trigger is None:
                        continue
                    runner.set_lcm_mode(
                        index,
                        pair=trigger["pair"],
                        sigma=float(gap_seeking_settings["sigma"]),
                        alpha_eV=float(gap_seeking_settings["alpha_eV"]),
                    )
                    state.update(
                        {
                            "current_mode": "lcm",
                            "switched": True,
                            "pair": list(trigger["pair"]),
                            "trigger_gap_eV": float(trigger["gap_eV"]),
                            "trigger_step": int(runner.nsteps),
                            "trigger_time_ps": float(time_ps),
                        }
                    )
                    state["events"].append(
                        {
                            "event": "enter_lcm",
                            "step": int(runner.nsteps),
                            "time_ps": float(time_ps),
                            "pair": list(trigger["pair"]),
                            "gap_key": str(trigger["gap_key"]),
                            "gap_eV": float(trigger["gap_eV"]),
                            "abs_gap_eV": float(trigger["abs_gap_eV"]),
                        }
                    )
                    switched_this_check = True
            if switched_this_check:
                # NVTLangevin consumes the currently stored force at the start of
                # its next step, so refresh the batch after changing graph modes.
                runner.evaluate()
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
            if not np.isfinite(target_temperatures).all() or np.any(target_temperatures < 0):
                raise ValueError("Temperature schedules must produce finite nonnegative temperatures.")
            for index in np.where(active)[0]:
                temperature_histories[int(index)].append(
                    float(target_temperatures[int(index)])
                )
            runner.set_temperature(target_temperatures)
            if density_parameters is not None:
                for batch_index in np.where(active)[0]:
                    index = int(batch_index)
                    row = density_parameters[index]
                    density = annealing_schedule(time_ps, maxt, row["Ramp"], row["Rper"], row["Rsrt"], row["Rend"]) / 1.66054
                    if not np.isfinite(density) or density <= 0:
                        raise ValueError("Density schedule produced a nonpositive or nonfinite density.")
                    atoms = atoms_list[index]
                    scale = (atoms.get_masses().sum() / (atoms.get_volume() * density)) ** (1.0 / 3.0)
                    runner.set_cell(index, np.asarray(atoms.cell) * scale, scale_atoms=True, evaluate=False)
                    runner.sync_graph_to_atoms(index, atoms)
                    density_histories[index].append(float(density))
                runner.evaluate()
            runner.run(ncheck)

        if policy == "stop":
            candidates.sort(key=lambda item: int(item["record"]["batch_index"]))
        else:
            candidates = _prune_candidates(
                candidates,
                return_top_k,
                max_candidates_per_replica,
            )

        outputs: list[MoleculesObject] = []
        topology_summary = (
            {
                "topology_rejected_replica_count": len(topology_rejections),
                "topology_rejections": [
                    dict(rejection) for rejection in topology_rejections
                ],
            }
            if reference_topology is not None
            else {}
        )
        if reference_topology is not None and not candidates:
            print(
                "ALF batched topology summary: "
                f"{len(topology_rejections)} replica(s) rejected; "
                "no valid uncertainty candidates returned."
            )
        for rank, candidate in enumerate(candidates):
            parent_id = str(candidate["record"]["parent_molecule_id"])
            molecule = MoleculesObject(candidate["atoms"], f"{parent_id}-cand-{rank:04d}")
            metadata = dict(candidate["parent_metadata"])
            metadata.update(candidate["record"])
            metadata.update(
                {
                    **({"force_screen_audit_path": str(audit.path)} if audit.path else {}),
                    "candidate_rank": int(rank),
                    "candidate_count": len(candidates),
                    "dynamics_backend": ("alchemi_baoab" if backend["backend"] == "alchemi" else "torchsim_baoab"),
                    "model_mode": mode,
                    "calculator_interface": str(
                        getattr(model, "calculator_interface", "native")
                    ),
                    "calculator_loader": str(
                        getattr(model, "calculator_loader", type(model).__name__)
                    ),
                    **topology_summary,
                    **(
                        {"sampler_device": str(device)}
                        if device is not None
                        else {}
                    ),
                }
            )
            molecule.update_metadata(metadata)
            outputs.append(molecule)
        return outputs
    finally:
        audit.write()


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
