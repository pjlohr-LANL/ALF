"""Sampling configuration and numerical contracts, independent of strategies.

NumPy/ASE configuration and debug checks work without the optional GPU stack.
Torch is imported locally only by helpers processing tensor predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alframework.tools.excited_state_tools import (
    _gap_state_index, derive_gap_property_table, derive_state_property_table,
    gap_key_for_pair, parse_gap_key,
)
from alframework.tools.force_screen import validated_force_cutoff
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampler_batching import (
    sampler_batch_size, selected_state, validate_state_selection,
)

SCORE_WEIGHT_KEYS = ("w_energy", "w_force", "w_gap_uncertainty")

def _score_weight(raw: dict[str, Any], name: str) -> float:
    """Return one validated, nonnegative scoring weight."""

    value = raw.get(name, 0.0)
    if isinstance(value, bool):
        raise ValueError(f"score.{name} must be a nonnegative number.")
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"score.{name} must be a nonnegative number; received {value!r}."
        ) from exc
    if not np.isfinite(weight) or weight < 0:
        raise ValueError(
            f"score.{name} must be finite and nonnegative; received {value!r}."
        )
    return weight


def _score_gap_rows(
    raw: dict[str, Any],
    sampler_config: dict[str, Any],
    properties_list: dict[str, Any] | None,
    selected_state_value: int | None,
) -> list[dict[str, Any]]:
    """Resolve the state pairs whose ensemble gap deviation the score consumes.

    Pairs are declared inside the ``score`` block rather than through ``dE#``
    entries in ``properties_list``. The deviation is derived from per-member
    state energies the model already predicts, so no trained gap head, HDF5
    dataset, or QM-interface change is required.
    """

    if (
        str(sampler_config.get("model_mode", "ground_state")).strip().lower()
        != "excited_state"
    ):
        raise ValueError(
            "score.w_gap_uncertainty > 0 requires model_mode='excited_state'."
        )
    if properties_list is None:
        raise ValueError(
            "score.w_gap_uncertainty > 0 requires properties_list so gap "
            "state pairs can be validated."
        )
    available = sorted(
        int(row["state"]) for row in derive_state_property_table(properties_list)
    )
    if len(available) < 2:
        raise ValueError(
            "score.w_gap_uncertainty > 0 requires at least two excited-state "
            f"energies in properties_list; found states {available}."
        )

    raw_pairs = raw.get("gap_pairs", "selected_adjacent")
    if isinstance(raw_pairs, str):
        mode = raw_pairs.strip().lower()
        if mode == "adjacent":
            pair_items = [
                (lower, upper)
                for lower, upper in zip(available, available[1:])
                if upper == lower + 1
            ]
        elif mode == "selected_adjacent":
            if selected_state_value is None:
                raise ValueError(
                    "score.gap_pairs='selected_adjacent' requires a resolved "
                    "selected state for the batch."
                )
            selected = int(selected_state_value)
            candidate_pairs = ((selected - 1, selected), (selected, selected + 1))
            pair_items = [
                (lower, upper)
                for lower, upper in candidate_pairs
                if lower in set(available) and upper in set(available)
            ]
            if not pair_items:
                raise ValueError(
                    "score.gap_pairs='selected_adjacent' found no adjacent "
                    f"state pair for selected state {selected}; available "
                    f"states are {available}."
                )
        else:
            raise ValueError(
                "score.gap_pairs must be 'selected_adjacent', 'adjacent', or a "
                f"list of state pairs; received {raw_pairs!r}."
            )
    elif isinstance(raw_pairs, (list, tuple)):
        if not raw_pairs:
            raise ValueError("score.gap_pairs must not be an empty list.")
        pair_items = []
        for index, pair in enumerate(raw_pairs):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(
                    "Each score.gap_pairs entry must contain exactly two "
                    f"states; entry {index} is {pair!r}."
                )
            lower = _gap_state_index(pair[0], context=f"score.gap_pairs[{index}][0]")
            upper = _gap_state_index(pair[1], context=f"score.gap_pairs[{index}][1]")
            if lower == upper:
                raise ValueError(
                    f"score.gap_pairs entry {index} cannot reference one state "
                    "twice."
                )
            if lower > upper:
                lower, upper = upper, lower
            pair_items.append((lower, upper))
    else:
        raise TypeError(
            "score.gap_pairs must be a string or a list of state pairs."
        )

    unavailable = sorted(
        {
            state
            for lower, upper in pair_items
            for state in (lower, upper)
            if state not in set(available)
        }
    )
    if unavailable:
        raise ValueError(
            f"score.gap_pairs references states {unavailable} that are not "
            f"present in properties_list; available states are {available}."
        )

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for lower, upper in sorted(set(pair_items)):
        if (lower, upper) in seen:
            continue
        seen.add((lower, upper))
        rows.append(
            {
                "lower_state": int(lower),
                "upper_state": int(upper),
                "gap_key": gap_key_for_pair(lower, upper),
            }
        )
    return rows


def configured_score(
    sampler_config: dict[str, Any],
    properties_list: dict[str, Any] | None = None,
    selected_state_value: int | None = None,
) -> dict[str, Any]:
    """Validate the optional candidate-scoring policy.

    ``max`` mode reproduces ALF's normalized worst-violation score exactly.
    ``sum`` mode ranks candidates by a weighted sum of the same normalized
    ratios plus an ensemble gap-deviation term, so relative importance becomes
    configurable while every term remains dimensionless.
    """

    raw = sampler_config.get("score")
    if raw is None:
        return {"mode": "max", "weights": {}, "gap_std_cut_eV": None, "gap_rows": []}
    if not isinstance(raw, dict):
        raise TypeError("score must be a dictionary when provided.")
    allowed = {"mode", "gap_std_cut_eV", "gap_pairs", *SCORE_WEIGHT_KEYS}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError("Unknown score options: " + ", ".join(unknown))

    mode = str(raw.get("mode", "max")).strip().lower()
    if mode not in {"max", "sum"}:
        raise ValueError("score.mode must be either 'max' or 'sum'.")
    if mode == "max":
        return {"mode": "max", "weights": {}, "gap_std_cut_eV": None, "gap_rows": []}

    weights = {name: _score_weight(raw, name) for name in SCORE_WEIGHT_KEYS}
    if not any(weight > 0 for weight in weights.values()):
        raise ValueError(
            "score.mode='sum' requires at least one strictly positive weight; "
            "received " + ", ".join(f"{k}={v}" for k, v in weights.items()) + "."
        )

    gap_std_cut: float | None = None
    gap_rows: list[dict[str, Any]] = []
    if weights["w_gap_uncertainty"] > 0:
        if "gap_std_cut_eV" not in raw:
            raise ValueError(
                "score.w_gap_uncertainty > 0 requires score.gap_std_cut_eV."
            )
        try:
            gap_std_cut = float(raw["gap_std_cut_eV"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "score.gap_std_cut_eV must be finite and positive; received "
                f"{raw['gap_std_cut_eV']!r}."
            ) from exc
        if not np.isfinite(gap_std_cut) or gap_std_cut <= 0:
            raise ValueError(
                "score.gap_std_cut_eV must be finite and positive; received "
                f"{raw['gap_std_cut_eV']!r}."
            )
        gap_rows = _score_gap_rows(
            raw, sampler_config, properties_list, selected_state_value
        )
    return {
        "mode": "sum",
        "weights": weights,
        "gap_std_cut_eV": gap_std_cut,
        "gap_rows": gap_rows,
    }


def _configured_replica_candidate_limit(
    sampler_config: dict[str, Any],
) -> int | None:
    """Return the validated per-replica candidate cap, or None for unlimited."""

    value = sampler_config.get("max_candidates_per_replica")
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            "max_candidates_per_replica must be a positive integer or null."
        )
    try:
        limit = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "max_candidates_per_replica must be a positive integer or null; "
            f"received {value!r}."
        ) from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(
            "max_candidates_per_replica must be a positive integer or null; "
            f"received {value!r}."
        )
    if limit < 1:
        raise ValueError(
            "max_candidates_per_replica must be at least one when provided."
        )
    return limit


def _configured_return_top_k(sampler_config: dict[str, Any]) -> int:
    """Return the validated per-task top-K candidate budget."""

    if "return_top_n" in sampler_config:
        raise ValueError(
            "return_top_n was renamed to return_top_k; update the sampler "
            "configuration. The old key is rejected rather than ignored "
            "because silently falling back to the default would return one "
            "candidate per task instead of the configured budget."
        )
    value = sampler_config.get("return_top_k", 1)
    if isinstance(value, bool):
        raise ValueError("return_top_k must be a positive integer.")
    try:
        limit = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"return_top_k must be a positive integer; received {value!r}."
        ) from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(
            f"return_top_k must be a positive integer; received {value!r}."
        )
    if limit < 1:
        raise ValueError("return_top_k must be at least one.")
    return limit


def configured_gap_diagnostics(
    sampler_config: dict[str, Any],
    properties_list: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return configured direct-gap diagnostics without loading a backend."""

    raw = sampler_config.get("gap_diagnostics")
    if raw is None:
        return []
    if isinstance(raw, bool):
        enabled = raw
    elif isinstance(raw, dict):
        enabled = bool(raw.get("enabled", False))
        unknown = sorted(set(raw) - {"enabled"})
        if unknown:
            raise ValueError(
                "Unknown gap_diagnostics options: " + ", ".join(unknown)
            )
    else:
        raise TypeError("gap_diagnostics must be a Boolean or dictionary.")
    if not enabled:
        return []
    if (
        str(sampler_config.get("model_mode", "ground_state")).strip().lower()
        != "excited_state"
    ):
        raise ValueError(
            "gap_diagnostics requires model_mode='excited_state'."
        )
    if properties_list is None:
        raise ValueError(
            "gap_diagnostics requires properties_list so state-gap pairs can "
            "be validated."
        )
    rows = derive_gap_property_table(properties_list)
    if not rows:
        raise ValueError(
            "gap_diagnostics.enabled=true requires at least one dE# system "
            "property in properties_list."
        )
    return rows


def configured_gap_seeking(
    sampler_config: dict[str, Any],
    properties_list: dict[str, Any] | None,
    selected_state_value: int | None,
) -> dict[str, Any]:
    """Validate one-way, selected-adjacent LCM gap seeking."""

    raw = sampler_config.get("gap_seeking")
    if raw is None:
        return {"enabled": False, "rows": []}
    if not isinstance(raw, dict):
        raise TypeError("gap_seeking must be a dictionary.")
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return {"enabled": False, "rows": []}

    allowed = {
        "enabled",
        "mode",
        "switch_policy",
        "candidate_pairs",
        "trigger_gap_threshold_eV",
        "sigma",
        "alpha_eV",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            "Unknown gap_seeking options: " + ", ".join(unknown)
        )
    if (
        str(sampler_config.get("model_mode", "ground_state")).strip().lower()
        != "excited_state"
    ):
        raise ValueError(
            "gap_seeking requires model_mode='excited_state'."
        )
    mode = str(raw.get("mode", "")).strip().lower()
    if mode != "levine_coe_martinez_switch":
        raise ValueError(
            "gap_seeking.mode must be 'levine_coe_martinez_switch'."
        )
    switch_policy = str(raw.get("switch_policy", "")).strip().lower()
    if switch_policy != "stay_fixed":
        raise ValueError(
            "gap_seeking.switch_policy must be 'stay_fixed'; hysteresis and "
            "LCM exit are not supported."
        )
    candidate_pairs = str(raw.get("candidate_pairs", "")).strip().lower()
    if candidate_pairs != "adjacent":
        raise ValueError(
            "gap_seeking.candidate_pairs must be 'adjacent'."
        )
    required_parameters = (
        "trigger_gap_threshold_eV",
        "sigma",
        "alpha_eV",
    )
    missing = [name for name in required_parameters if name not in raw]
    if missing:
        raise ValueError(
            "Enabled gap_seeking requires explicit values for: "
            + ", ".join(missing)
        )
    parameters: dict[str, float] = {}
    for name in required_parameters:
        try:
            value = float(raw[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"gap_seeking.{name} must be finite and positive."
            ) from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"gap_seeking.{name} must be finite and positive."
            )
        parameters[name] = value
    if properties_list is None:
        raise ValueError(
            "gap_seeking requires properties_list so eligible dE# pairs can "
            "be validated."
        )
    if selected_state_value is None:
        raise ValueError(
            "gap_seeking requires a resolved selected state for the batch."
        )

    selected = int(selected_state_value)
    rows = [
        dict(row)
        for row in derive_gap_property_table(properties_list)
        if int(row["upper_state"]) == int(row["lower_state"]) + 1
        and selected in {
            int(row["lower_state"]),
            int(row["upper_state"]),
        }
    ]
    rows.sort(
        key=lambda row: (
            int(row["lower_state"]),
            int(row["upper_state"]),
        )
    )
    if not rows:
        raise ValueError(
            "gap_seeking found no explicitly configured adjacent dE# property "
            f"containing selected state {selected}."
        )
    state_rows = {
        int(row["state"]): row
        for row in derive_state_property_table(properties_list)
    }
    required_force_states = sorted(
        {
            int(state)
            for row in rows
            for state in (row["lower_state"], row["upper_state"])
        }
    )
    missing_force_keys = [
        f"F{state}"
        for state in required_force_states
        if state_rows[state]["force_key"] is None
    ]
    if missing_force_keys:
        raise ValueError(
            "gap_seeking requires force properties for every eligible LCM "
            "state; missing " + ", ".join(missing_force_keys) + "."
        )
    return {
        "enabled": True,
        "mode": mode,
        "switch_policy": switch_policy,
        "candidate_pairs": candidate_pairs,
        "selected_state": selected,
        "rows": rows,
        **parameters,
    }


def configured_alchemi_gap_rows(
    sampler_config: dict[str, Any],
    properties_list: dict[str, Any] | None,
    selected_state_value: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the union of diagnostic, scoring, and dynamics-required gap rows."""

    diagnostic_rows = configured_gap_diagnostics(
        sampler_config,
        properties_list,
    )
    seeking = configured_gap_seeking(
        sampler_config,
        properties_list,
        selected_state_value,
    )
    score = configured_score(
        sampler_config,
        properties_list,
        selected_state_value,
    )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for row in [*diagnostic_rows, *seeking["rows"], *score["gap_rows"]]:
        pair = (int(row["lower_state"]), int(row["upper_state"]))
        if pair not in seen:
            seen.add(pair)
            rows.append(dict(row))
    return rows, seeking


def _validate_sampling_inputs(
    molecule_objects: list[MoleculesObject],
    sampler_config: dict[str, Any],
    properties_list: dict[str, Any] | None = None,
) -> tuple[str, int | None]:
    from alframework.tools.batched_config import batching_settings, dynamics_settings, unified_sampling, acquisition_settings
    settings = batching_settings(sampler_config)
    portable = settings["portable"]
    ase_driver = unified_sampling(sampler_config) and dynamics_settings(sampler_config)["backend"] == "ase"
    if portable:
        dynamics_settings(sampler_config)
    if not isinstance(molecule_objects, list) or not molecule_objects:
        raise ValueError("molecule_objects must be a non-empty list.")
    if not all(isinstance(item, MoleculesObject) for item in molecule_objects):
        raise TypeError("molecule_objects may contain only MoleculesObject instances.")
    expected_size = sampler_batch_size(sampler_config)
    if (len(molecule_objects) > expected_size or
            (settings["partial_policy"] == "full_only" and len(molecule_objects) != expected_size)):
        raise ValueError(
            "Strict-full ALCHEMI sampling expected exactly "
            f"{expected_size} molecules but received {len(molecule_objects)}."
        )

    mode = str(sampler_config.get("model_mode", "ground_state")).strip().lower()
    if mode not in {"ground_state", "excited_state"}:
        raise ValueError("model_mode must be 'ground_state' or 'excited_state'.")
    validate_state_selection(sampler_config, properties_list)
    policy = str(sampler_config.get("uncertainty_policy", "stop")).strip().lower()
    if (not unified_sampling(sampler_config) or acquisition_settings(sampler_config)["method"] == "uncertainty") and policy not in {"stop", "continue"}:
        raise ValueError("uncertainty_policy must be 'stop' or 'continue'.")
    _configured_return_top_k(sampler_config)
    _configured_replica_candidate_limit(sampler_config)

    reference_numbers: tuple[int, ...] | None = None
    states: set[int] = set()
    boundary_conditions: set[tuple[bool, ...]] = set()
    for molecule in molecule_objects:
        atoms = molecule.get_atoms()
        if atoms is None:
            raise ValueError("ALCHEMI cannot sample a molecule whose atoms are None.")
        if not len(atoms) or not np.isfinite(atoms.positions).all():
            raise ValueError("Sampling requires nonempty systems with finite positions.")
        if not np.isfinite(atoms.get_masses()).all() or np.any(atoms.get_masses() <= 0):
            raise ValueError("Sampling requires finite positive atomic masses.")
        if atoms.constraints and not ase_driver:
            raise NotImplementedError("Batched sampling does not yet support atom constraints.")
        boundary_conditions.add(tuple(bool(value) for value in atoms.pbc))
        if bool(np.any(atoms.get_pbc())):
            if not portable:
                raise NotImplementedError("ALCHEMI sampling currently supports only nonperiodic systems.")
            if (not np.all(atoms.pbc) and not ase_driver) or mode != "ground_state":
                raise NotImplementedError("Periodic batched sampling supports fully periodic ground states only.")
            if not np.isfinite(atoms.cell).all():
                raise ValueError("Periodic sampling requires finite cell vectors.")
            if ase_driver and not np.all(atoms.pbc):
                vectors = np.asarray(atoms.cell)[atoms.pbc]
                if np.linalg.matrix_rank(vectors) != int(np.sum(atoms.pbc)):
                    raise ValueError("Periodic cell vectors must be linearly independent.")
            elif not np.isfinite(atoms.get_volume()) or atoms.get_volume() <= 0:
                raise ValueError("Periodic sampling requires a finite nonsingular cell.")
        numbers = tuple(int(value) for value in atoms.get_atomic_numbers())
        if reference_numbers is None:
            reference_numbers = numbers
        elif not portable and numbers != reference_numbers:
            raise ValueError(
                "Every molecule in an ALCHEMI batch must have identical atom order."
            )
        if mode == "excited_state":
            states.add(selected_state(molecule, sampler_config))
    if len(boundary_conditions) > 1:
        raise ValueError("Every system in a batch must use the same periodic boundary conditions.")
    if len(states) > 1:
        raise ValueError("Every molecule in an excited-state batch must select the same state.")
    if mode == "excited_state" and properties_list is not None and states:
        available_states = {
            int(row["state"])
            for row in derive_state_property_table(properties_list)
        }
        unavailable_states = sorted(states - available_states)
        if unavailable_states:
            raise ValueError(
                f"Selected states {unavailable_states} are not present in "
                f"properties_list; available states are {sorted(available_states)}."
            )
    selected_for_validation = next(iter(states)) if states else None
    if properties_list is not None:
        # Also validates the score block, including its gap state pairs.
        configured_alchemi_gap_rows(
            sampler_config,
            properties_list,
            selected_for_validation,
        )
    else:
        configured_score(sampler_config, None, selected_for_validation)

    for density_key in ("end_dens", "amp_dens", "per_dens"):
        if not portable and sampler_config.get(density_key) is not None:
            raise NotImplementedError(
                "ALCHEMI sampling does not yet support density or cell schedules; "
                f"set {density_key} to null."
            )
    if portable and sampler_config.get("end_dens") is not None:
        if boundary_conditions != {(True, True, True)}:
            raise ValueError("Density schedules require fully periodic ground-state systems.")
    if "max_force_cutoff" in sampler_config:
        validated_force_cutoff(sampler_config["max_force_cutoff"], positive=True)
    return mode, (next(iter(states)) if states else None)


def _physical_force_states(mode, selected, properties_list):
    if mode == "ground_state":
        return [0 if selected is None else int(selected)]
    return [int(row["state"]) for row in
            derive_state_property_table(properties_list, require_forces=True)]


def validated_batch_shape(batch) -> tuple[int, int]:
    import torch

    counts = batch.num_nodes_per_graph
    if not bool(torch.all(counts == counts[0]).detach().cpu().item()):
        raise ValueError("ALCHEMI calculator batches require equal atom counts.")
    return int(batch.num_graphs), int(counts[0].detach().cpu().item())


def normalize_energy_contributions(batch, energy_values):
    import torch

    num_graphs = int(batch.num_graphs)
    energy = torch.as_tensor(
        energy_values,
        dtype=batch.positions.dtype,
        device=batch.positions.device,
    )
    if energy.ndim < 2 or int(energy.shape[1]) != num_graphs:
        raise ValueError(
            "ALCHEMI energy contributions must have shape [models, batch, ...]."
        )
    model_count = int(energy.shape[0])
    energy = energy.reshape(model_count, num_graphs, -1).sum(dim=2)
    if model_count < 1:
        raise ValueError("At least one calculator contribution is required.")
    return energy


def normalize_contributions(batch, energy_values, force_values):
    import torch

    total_atoms = int(batch.positions.shape[0])
    energy = normalize_energy_contributions(
        batch, energy_values
    )
    model_count = int(energy.shape[0])
    forces = torch.as_tensor(
        force_values,
        dtype=batch.positions.dtype,
        device=batch.positions.device,
    )
    if forces.ndim < 3 or int(forces.shape[0]) != model_count:
        raise ValueError(
            "ALCHEMI force contributions must use the same model count as energies."
        )
    allowed_shapes = {(model_count, total_atoms, 3)}
    if bool(torch.all(batch.num_nodes_per_graph == batch.num_nodes_per_graph[0])):
        allowed_shapes.add((model_count, int(batch.num_graphs),
                            int(batch.num_nodes_per_graph[0]), 3))
    if tuple(forces.shape) not in allowed_shapes:
        raise ValueError(
            "ALCHEMI force contributions must have shape [models, total_atoms, 3]."
        )
    forces = forces.reshape(model_count, total_atoms, 3)
    if not bool(torch.isfinite(forces).all()) or not bool(torch.isfinite(energy).all()):
        raise ValueError("Physical energy/force contributions must be finite.")
    return energy, forces


def validate_sampler_debug_output(molecule, sampler_config):
    from alframework.tools.batched_config import dynamics_settings
    metadata = molecule.get_metadata()
    backend = dynamics_settings(sampler_config)
    if backend.get("strict_gpu", False) and not backend.get("allow_cpu_debug", False):
        assert str(metadata.get("sampler_device", "")).startswith("cuda:"), (
            "Strict-GPU sampler testing must return CUDA device metadata.")
    if sampler_config.get("model_mode") == "excited_state":
        selection = dict(sampler_config.get("state_selection", {}))
        states = ({selection.get("state")} if selection.get("mode") == "fixed"
                  else set(selection.get("states", [])))
        assert metadata.get("selected_state") in states, (
            "Excited-state sampler testing must return configured selected_state metadata.")
    if sampler_config.get("topology_check", {}).get("enabled", False):
        assert metadata.get("topology_valid") is True, (
            "Topology-enabled sampler testing must return topology_valid=true.")


def validated_numpy_contributions(energy_values, force_values):
    energies = np.asarray(energy_values, dtype=float)
    forces = np.asarray(force_values, dtype=float)
    if energies.shape[0] != forces.shape[0]:
        raise ValueError("Energy and force contributions must have the same model count.")
    if energies.shape[0] < 1:
        raise ValueError("At least one model contribution is required.")
    return energies, forces


def uncertainty_cutoffs(Escut, Fscut):
    energy_cutoff, force_cutoff = float(Escut), float(Fscut)
    if energy_cutoff <= 0 or force_cutoff <= 0:
        raise ValueError("Escut and Fscut must be positive for ALCHEMI sampling.")
    return energy_cutoff, force_cutoff


def validate_numeric_range(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value numeric range [minimum, maximum].")
    try:
        lower, upper = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain two finite numeric values; received {value!r}."
        ) from exc
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"{name} must contain two finite numeric values; received {value!r}.")
    if lower > upper:
        raise ValueError(f"{name} minimum must not exceed its maximum; received {value!r}.")
    return lower, upper


def validate_temperature_period(period, replica_index):
    if period <= 0:
        raise ValueError(
            "per_temp sampled a nonpositive temperature period "
            f"({period}) for replica {replica_index}; configure strictly positive period bounds."
        )


def validate_dynamics_parameters(dt, maxt, ncheck, min_time):
    if not np.isfinite([dt, maxt, min_time]).all() or dt <= 0 or maxt <= 0 or ncheck < 1 or min_time < 0:
        raise ValueError("dt and maxt must be finite and positive; Ncheck >= 1 and min_time finite and nonnegative.")


def validate_friction(friction_per_fs):
    if not np.isfinite(friction_per_fs) or friction_per_fs < 0:
        raise ValueError("friction_per_fs must be a finite nonnegative value.")


def validate_alchemi_device(is_cuda, *, strict_gpu, allow_cpu_debug):
    if not is_cuda and strict_gpu and not allow_cpu_debug:
        raise RuntimeError(
            "ALCHEMI sampling requires CUDA by default. Set "
            "alchemi_baoab.allow_cpu_debug=true only for local smoke tests."
        )


def validate_gap_model(sampler_config, properties_list, model):
    raw = sampler_config.get("gap_diagnostics")
    diagnostics_enabled = bool(raw.get("enabled", False)) if isinstance(raw, dict) else bool(raw)
    raw_seeking = sampler_config.get("gap_seeking")
    seeking_enabled = bool(raw_seeking.get("enabled", False)) if isinstance(raw_seeking, dict) else False
    if ((diagnostics_enabled or seeking_enabled) and properties_list is None
            and not getattr(model, "gap_rows", None)):
        raise ValueError(
            "Gap diagnostics or gap seeking requires properties_list or a "
            "calculator loaded with validated gap rows."
        )


def gap_seeking_settings_for_model(sampler_config, properties_list, model, selected_state_value):
    if properties_list is not None:
        return configured_alchemi_gap_rows(sampler_config, properties_list, selected_state_value)[1]
    settings = dict(getattr(model, "gap_seeking_settings", {"enabled": False, "rows": []}))
    raw = sampler_config.get("gap_seeking")
    if isinstance(raw, dict) and raw.get("enabled", False) and not settings.get("enabled", False):
        raise ValueError(
            "gap_seeking requires properties_list or a calculator loaded "
            "with validated gap-seeking settings."
        )
    return settings


def validate_required_physical_states(state_contributions, states):
    missing = sorted(set(states) - set(state_contributions))
    if missing:
        raise KeyError(f"Physical force screening requires state_contributions for states {missing}.")


def validate_model_count(actual, expected, message):
    if int(actual) != int(expected):
        raise ValueError(message)


def normalize_gap_contributions(batch, state_energy_contributions, row, model_count):
    lower, upper = int(row["lower_state"]), int(row["upper_state"])
    missing = [state for state in (lower, upper) if state not in state_energy_contributions]
    if missing:
        raise KeyError(
            f"Gap diagnostic {row['gap_key']} requires raw energy contributions for states {missing}."
        )
    lower_energy = normalize_energy_contributions(batch, state_energy_contributions[lower])
    upper_energy = normalize_energy_contributions(batch, state_energy_contributions[upper])
    message = (f"Gap diagnostic {row['gap_key']} requires the same ordered model members "
               "for both states and the selected dynamics ensemble.")
    validate_model_count(lower_energy.shape[0], model_count, message)
    validate_model_count(upper_energy.shape[0], model_count, message)
    return lower_energy, upper_energy


def normalize_uncertainty_override(batch, values):
    import torch

    normalized = {}
    for key in ("Es", "Fs", "Fsmax"):
        value = torch.as_tensor(values[key], dtype=batch.positions.dtype,
                                device=batch.positions.device).reshape(-1)
        if int(value.shape[0]) != int(batch.num_graphs):
            raise ValueError(f"Calculator-provided {key} must have one value per graph.")
        normalized[key] = value
    return normalized


def validate_native_results(output, energy_key, force_key):
    if energy_key not in output or force_key not in output:
        raise KeyError("Native calculator output must contain configured energy and force keys.")


def native_state_contributions(output, states, selected_state, energy_key, force_key):
    raw_states = {int(state): values for state, values in dict(output.get("state_contributions") or {}).items()}
    raw_states.setdefault(selected_state, {"energy": output[energy_key], "forces": output[force_key]})
    missing = sorted(set(states) - set(raw_states))
    if missing:
        raise KeyError(
            "Native gap seeking requires state_contributions "
            f"with energy and force values for states {missing}."
        )
    normalized = {}
    for state in sorted(states):
        values = raw_states[state]
        energy = values.get("energy", values.get("energy_contributions"))
        forces = values.get("forces", values.get("force_contributions"))
        if energy is None or forces is None:
            raise KeyError(
                "Native gap-seeking state_contributions must provide energy/forces "
                "or energy_contributions/force_contributions."
            )
        normalized[state] = energy, forces
    return normalized


def validate_ase_results(results, required):
    missing = [key for key in required if key not in results]
    if missing:
        raise KeyError("ASE fallback calculator did not provide required properties: " + ", ".join(missing))


def validate_ase_force_shape(forces, num_atoms, key):
    if forces.shape != (num_atoms, 3):
        raise ValueError(f"ASE fallback {key} must have shape ({num_atoms}, 3).")


def hippynn_members_first(values, model_count, name):
    import torch

    if values.ndim < 2 or int(values.shape[1]) != int(model_count):
        raise ValueError(f"HIPPYNN {name} .all output must place the model axis second.")
    return torch.movedim(values, 1, 0)


def hippynn_state_table(model_mode, selected_state_value, ML_config, properties_list):
    mode = str(model_mode).strip().lower()
    if mode == "ground_state":
        force_key = ML_config.get("force_key")
        if force_key is None:
            raise ValueError("Ground-state ALCHEMI requires ML_config.force_key.")
        return [{"state": 0, "energy_db_name": str(ML_config.get("energy_key", "energy")),
                 "force_db_name": str(force_key)}], 0
    if mode == "excited_state":
        table = derive_state_property_table(properties_list, require_forces=True)
        if selected_state_value is None:
            raise ValueError("Excited-state ALCHEMI requires selected_state metadata.")
        selected = int(selected_state_value)
        if selected not in {int(row["state"]) for row in table}:
            raise ValueError(f"Selected state {selected} is not present in properties_list.")
        return table, selected
    raise ValueError("model_mode must be 'ground_state' or 'excited_state'.")


def validate_score_diagnostics(diagnostics, score):
    """Validate the inputs a summed acquisition strategy consumes."""
    if "Fsrms" not in diagnostics:
        raise ValueError(
            "score.mode='sum' requires the Fsrms force deviation, which is "
            "only available from a per-member ensemble. The single-calculator "
            "uncertainty override supplies energy_stdev, forces_stdev_mean, "
            "and forces_stdev_max only; use an ensemble calculator or score.mode='max'."
        )
    gaps = []
    if score["weights"]["w_gap_uncertainty"] > 0:
        for row in score["gap_rows"]:
            key = f"{row['gap_key']}_stdev"
            if key not in diagnostics:
                raise ValueError(
                    f"score gap term requires calculator diagnostic {key!r}; "
                    "the ALCHEMI calculator was built without that state pair."
                )
            value = float(diagnostics[key])
            if not np.isfinite(value):
                raise ValueError(f"score gap term received nonfinite {key}={value!r}.")
            gaps.append((value, row))
        if not gaps:
            raise ValueError("score.w_gap_uncertainty > 0 resolved no gap state pairs.")
    return gaps


def validated_seeking_gap(diagnostics, gap_key):
    if gap_key not in diagnostics:
        raise KeyError(f"gap_seeking requires calculator diagnostic {gap_key!r}.")
    gap = float(diagnostics[gap_key])
    if not np.isfinite(gap):
        raise ValueError(f"gap_seeking received nonfinite diagnostic {gap_key}={gap!r}.")
    return gap


def validate_topology_identities(topology_results):
    failures = [index for index, result in enumerate(topology_results)
                if result.reason == "topology_identity"]
    if failures:
        first = failures[0]
        violation = topology_results[first].violations[0]
        raise ValueError(
            "ALCHEMI topology checking requires the exact reference atom "
            f"order. Replica {first} has atomic numbers "
            f"{violation['actual_atomic_numbers']}, expected "
            f"{violation['expected_atomic_numbers']}."
        )


def validate_native_models(models, base_model_type):
    if not models:
        raise ValueError("A native ALCHEMI ensemble requires at least one model.")
    if not all(isinstance(model, base_model_type) for model in models):
        raise TypeError("Every native ensemble member must implement BaseModelMixin.")
    input_signature = models[0].input_data()
    if any(model.input_data() != input_signature for model in models[1:]):
        raise ValueError("Native ensemble members require incompatible batch inputs.")


def validate_ase_calculators(calculators):
    from ase.calculators.calculator import Calculator

    if not calculators or not all(isinstance(calculator, Calculator) for calculator in calculators):
        raise TypeError("ase_calculator must return an ASE Calculator or a non-empty list of them.")


def validate_hippynn_options(options):
    if options:
        raise TypeError("Unknown HIPPYNN ALCHEMI calculator options: " + ", ".join(sorted(options)))


def _gap_candidate_metadata(
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    means: dict[str, float] = {}
    stdevs: dict[str, float] = {}
    pairs: dict[str, list[int]] = {}
    model_counts: dict[str, int] = {}
    for key, value in diagnostics.items():
        pair = parse_gap_key(str(key))
        if pair is None:
            continue
        gap_key = str(key)
        std_key = f"{gap_key}_stdev"
        if std_key not in diagnostics:
            raise ValueError(
                f"Gap diagnostic {gap_key!r} is missing {std_key!r}."
            )
        means[gap_key] = float(value)
        stdevs[gap_key] = float(diagnostics[std_key])
        pairs[gap_key] = [int(pair[0]), int(pair[1])]
        count_key = f"{gap_key}_model_count"
        if count_key in diagnostics:
            model_counts[gap_key] = int(diagnostics[count_key])
    if not means:
        return {}
    minimum_key = min(means, key=lambda key: (abs(means[key]), key))
    return {
        "gap_means": means,
        "gap_stds": stdevs,
        "gap_pairs": pairs,
        "gap_model_counts": model_counts,
        "minimum_abs_gap": abs(means[minimum_key]),
        "minimum_abs_gap_key": minimum_key,
        "minimum_abs_gap_pair": pairs[minimum_key],
    }
