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

from alframework.tools.excited_state_tools import (
    _gap_state_index,
    derive_gap_property_table,
    derive_state_property_table,
    gap_key_for_pair,
    parse_gap_key,
)
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.molecular_topology import (
    TopologyValidationResult,
    load_reference_topology,
    topology_metadata,
    validate_fixed_topology,
)
from alframework.tools.sampler_batching import (
    sampler_batch_size,
    selected_state,
    validate_state_selection,
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
        -float(record["ranking_score"]),
        int(record["step"]),
        int(record["batch_index"]),
        str(record["parent_molecule_id"]),
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
    if "Fsrms" not in diagnostics:
        raise ValueError(
            "score.mode='sum' requires the Fsrms force deviation, which is "
            "only available from a per-member ensemble. The single-calculator "
            "uncertainty override supplies energy_stdev, forces_stdev_mean, "
            "and forces_stdev_max only; use an ensemble calculator or "
            "score.mode='max'."
        )
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
        best: tuple[float, dict[str, Any]] | None = None
        for row in score["gap_rows"]:
            std_key = f"{row['gap_key']}_stdev"
            if std_key not in diagnostics:
                raise ValueError(
                    f"score gap term requires calculator diagnostic {std_key!r}; "
                    "the ALCHEMI calculator was built without that state pair."
                )
            value = float(diagnostics[std_key])
            if not np.isfinite(value):
                raise ValueError(
                    f"score gap term received nonfinite {std_key}={value!r}."
                )
            if best is None or value > best[0]:
                best = (value, row)
        if best is None:
            raise ValueError(
                "score.w_gap_uncertainty > 0 resolved no gap state pairs."
            )
        gap_std, row = best
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


def _prune_candidates(
    candidates: list[dict[str, Any]],
    return_top_n: int,
    max_candidates_per_replica: int | None,
) -> list[dict[str, Any]]:
    """Rank candidates, cap each trajectory, then apply the global limit.

    ``return_top_n`` alone is a global budget: under ``uncertainty_policy``
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
    return ranked[: int(return_top_n)]


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
        if gap_key not in diagnostics:
            raise KeyError(
                f"gap_seeking requires calculator diagnostic {gap_key!r}."
            )
        gap = float(diagnostics[gap_key])
        if not np.isfinite(gap):
            raise ValueError(
                f"gap_seeking received nonfinite diagnostic {gap_key}={gap!r}."
            )
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


def _validate_sampling_inputs(
    molecule_objects: list[MoleculesObject],
    sampler_config: dict[str, Any],
    properties_list: dict[str, Any] | None = None,
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
    validate_state_selection(sampler_config, properties_list)
    policy = str(sampler_config.get("uncertainty_policy", "stop")).strip().lower()
    if policy not in {"stop", "continue"}:
        raise ValueError("uncertainty_policy must be 'stop' or 'continue'.")
    _configured_replica_candidate_limit(sampler_config)

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
            states.add(selected_state(molecule, sampler_config))
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
        if sampler_config.get(density_key) is not None:
            raise NotImplementedError(
                "ALCHEMI sampling does not yet support density or cell schedules; "
                f"set {density_key} to null."
            )
    if "max_force_cutoff" in sampler_config:
        max_force_cutoff = float(sampler_config["max_force_cutoff"])
        if not np.isfinite(max_force_cutoff) or max_force_cutoff <= 0:
            raise ValueError(
                "max_force_cutoff must be finite and positive when provided."
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
    gap_seeking_state: dict[str, Any],
    score: dict[str, Any] | None = None,
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
    if "Fmeanmax" in diagnostics:
        record["Fmeanmax"] = float(diagnostics["Fmeanmax"])
    if "Fsrms" in diagnostics:
        record["Fsrms"] = float(diagnostics["Fsrms"])
    return record


def run_alchemi_sampling(
    molecule_objects: list[MoleculesObject],
    sampler_config: dict[str, Any],
    model: Any,
    *,
    properties_list: dict[str, Any] | None = None,
    device: Any = None,
    runner_factory: Callable[..., Any] | None = None,
    master_directory: str | None = None,
) -> list[MoleculesObject]:
    """Run one strict-full ALCHEMI batch using stop or continue selection."""

    raw_gap_options = sampler_config.get("gap_diagnostics")
    diagnostics_enabled = (
        bool(raw_gap_options.get("enabled", False))
        if isinstance(raw_gap_options, dict)
        else bool(raw_gap_options)
    )
    raw_seeking_options = sampler_config.get("gap_seeking")
    seeking_enabled = bool(
        raw_seeking_options.get("enabled", False)
    ) if isinstance(raw_seeking_options, dict) else False
    if (
        (diagnostics_enabled or seeking_enabled)
        and properties_list is None
        and not getattr(model, "gap_rows", None)
    ):
        raise ValueError(
            "Gap diagnostics or gap seeking requires properties_list or a "
            "calculator loaded with validated gap rows."
        )
    mode, selected_state_value = _validate_sampling_inputs(
        molecule_objects,
        sampler_config,
        properties_list,
    )
    if properties_list is not None:
        _, gap_seeking_settings = configured_alchemi_gap_rows(
            sampler_config,
            properties_list,
            selected_state_value,
        )
    else:
        gap_seeking_settings = dict(
            getattr(
                model,
                "gap_seeking_settings",
                {"enabled": False, "rows": []},
            )
        )
        if seeking_enabled and not bool(
            gap_seeking_settings.get("enabled", False)
        ):
            raise ValueError(
                "gap_seeking requires properties_list or a calculator loaded "
                "with validated gap-seeking settings."
            )
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
    topology_results: list[TopologyValidationResult] = [
        validate_fixed_topology(atoms, reference_topology)
        for atoms in atoms_list
    ]
    identity_failures = [
        index
        for index, result in enumerate(topology_results)
        if result.reason == "topology_identity"
    ]
    if identity_failures:
        first = identity_failures[0]
        violation = topology_results[first].violations[0]
        raise ValueError(
            "ALCHEMI topology checking requires the exact reference atom "
            f"order. Replica {first} has atomic numbers "
            f"{violation['actual_atomic_numbers']}, expected "
            f"{violation['expected_atomic_numbers']}."
        )
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
    if reference_topology is not None and not np.any(active):
        print(
            "ALCHEMI topology summary: "
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
                active[index] = False
                runner.freeze_graph(index)
                continue
            if (
                sampler_config.get("max_force_cutoff") is not None
                and diagnostics.get("Fmeanmax") is not None
                and float(diagnostics["Fmeanmax"])
                > float(sampler_config["max_force_cutoff"])
            ):
                active[index] = False
                runner.freeze_graph(index)
                continue
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
                    return_top_n,
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
        for index in np.where(active)[0]:
            temperature_histories[int(index)].append(
                float(target_temperatures[int(index)])
            )
        runner.set_temperature(target_temperatures)
        runner.run(ncheck)

    if policy == "stop":
        candidates.sort(key=lambda item: int(item["record"]["batch_index"]))
    else:
        candidates = _prune_candidates(
            candidates,
            return_top_n,
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
            "ALCHEMI topology summary: "
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
            gap_rows: list[dict[str, Any]] | None = None,
            gap_seeking_settings: dict[str, Any] | None = None,
            well_params: dict[str, Any] | None = None,
            device: Any = None,
            model_config_template: Any = None,
            calculator_interface: str = "native",
            calculator_loader: str | None = None,
        ) -> None:
            torch.nn.Module.__init__(self)
            self.selected_state_value = int(selected_state_value)
            self.gap_rows = [dict(row) for row in (gap_rows or [])]
            self.gap_seeking_settings = dict(
                gap_seeking_settings or {"enabled": False, "rows": []}
            )
            self._lcm_controls: dict[int, dict[str, Any]] = {}
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

        def set_lcm_mode(
            self,
            graph_index: int,
            *,
            pair: list[int] | tuple[int, int],
            sigma: float,
            alpha_eV: float,
        ) -> None:
            """Permanently switch one batch graph to its configured LCM pair."""

            if not bool(self.gap_seeking_settings.get("enabled", False)):
                raise RuntimeError(
                    "Cannot enter LCM mode when gap_seeking is disabled."
                )
            index = int(graph_index)
            if index < 0:
                raise ValueError("LCM graph_index must be nonnegative.")
            normalized_pair = (int(pair[0]), int(pair[1]))
            allowed_pairs = {
                (int(row["lower_state"]), int(row["upper_state"]))
                for row in self.gap_seeking_settings["rows"]
            }
            if normalized_pair not in allowed_pairs:
                raise ValueError(
                    f"LCM pair {normalized_pair} is not eligible for selected "
                    f"state {self.selected_state_value}."
                )
            control = {
                "pair": normalized_pair,
                "sigma": float(sigma),
                "alpha_eV": float(alpha_eV),
            }
            existing = self._lcm_controls.get(index)
            if existing is not None and existing != control:
                raise RuntimeError(
                    f"Graph {index} already entered stay-fixed LCM mode with "
                    f"pair {existing['pair']}."
                )
            self._lcm_controls[index] = control

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
        def _normalize_energy_contributions(batch, energy_values):
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

        @staticmethod
        def _normalize_contributions(batch, energy_values, force_values):
            total_atoms = int(batch.positions.shape[0])
            energy = ALFAlchemiCalculator._normalize_energy_contributions(
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
            if int(forces.numel()) != model_count * total_atoms * 3:
                raise ValueError(
                    "ALCHEMI force contributions must have shape [models, total_atoms, 3]."
                )
            forces = forces.reshape(model_count, total_atoms, 3)
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
                "Fsrms": torch.sqrt(
                    torch.mean(force_std_batched * force_std_batched, dim=(1, 2))
                ),
            }

        @staticmethod
        def _lcm_energy_forces(
            lower_energy,
            upper_energy,
            lower_forces,
            upper_forces,
            *,
            sigma: float,
            alpha_eV: float,
        ):
            """Return the fork-compatible smooth LCM mean and analytic force."""

            gap = upper_energy - lower_energy
            smooth_abs_gap = torch.sqrt(gap * gap + 1.0e-12)
            denominator = smooth_abs_gap + float(alpha_eV)
            correction = float(sigma) * gap * gap / denominator
            gap_derivative = float(sigma) * (
                (2.0 * gap * denominator)
                - (gap * gap * gap / smooth_abs_gap)
            ) / (denominator * denominator)
            energy = 0.5 * (lower_energy + upper_energy) + correction
            forces = (
                0.5 * (lower_forces + upper_forces)
                + gap_derivative * (upper_forces - lower_forces)
            )
            return energy, forces

        def forward(self, batch):
            prediction = dict(self.ensemble_forward(batch))
            energy, forces = self._normalize_contributions(
                batch,
                prediction["energy_contributions"],
                prediction["force_contributions"],
            )
            selected = self._reduce_contributions(batch, energy, forces)
            diagnostics: dict[str, Any] = {}
            state_contributions = {
                int(state): values
                for state, values in dict(
                    prediction.get("state_contributions") or {}
                ).items()
            }
            state_contributions.setdefault(
                self.selected_state_value,
                {
                    "energy_contributions": energy,
                    "force_contributions": forces,
                },
            )
            state_results: dict[int, dict[str, Any]] = {}
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
                state_results[state] = state_result
                diagnostics[f"sE{state}"] = state_result["energy"]
                diagnostics[f"sE{state}_stdev"] = state_result["energy_std"]
                diagnostics[f"F{state}"] = state_result["forces"].reshape(
                    int(batch.num_graphs), -1, 3
                )
                diagnostics[f"F{state}_stdev"] = state_result["force_std"]

            state_energy_contributions = {
                int(state): values
                for state, values in dict(
                    prediction.get("state_energy_contributions") or {}
                ).items()
            }
            for state_value, values in state_contributions.items():
                state_energy_contributions.setdefault(
                    int(state_value), values["energy_contributions"]
                )
            state_energy_contributions.setdefault(
                self.selected_state_value, energy
            )
            selected_model_count = int(energy.shape[0])
            for row in self.gap_rows:
                lower = int(row["lower_state"])
                upper = int(row["upper_state"])
                missing_states = [
                    state
                    for state in (lower, upper)
                    if state not in state_energy_contributions
                ]
                if missing_states:
                    raise KeyError(
                        f"Gap diagnostic {row['gap_key']} requires raw energy "
                        f"contributions for states {missing_states}."
                    )
                lower_energy = self._normalize_energy_contributions(
                    batch, state_energy_contributions[lower]
                )
                upper_energy = self._normalize_energy_contributions(
                    batch, state_energy_contributions[upper]
                )
                if (
                    int(lower_energy.shape[0]) != selected_model_count
                    or int(upper_energy.shape[0]) != selected_model_count
                ):
                    raise ValueError(
                        f"Gap diagnostic {row['gap_key']} requires the same "
                        "ordered model members for both states and the selected "
                        "dynamics ensemble."
                    )
                gap_members = upper_energy - lower_energy
                gap_mean = torch.mean(gap_members, dim=0)
                if selected_model_count == 1:
                    gap_std = torch.zeros_like(gap_mean)
                else:
                    gap_std = torch.std(
                        gap_members, dim=0, correction=0
                    )
                gap_key = str(row["gap_key"])
                diagnostics[gap_key] = gap_mean
                diagnostics[f"{gap_key}_stdev"] = gap_std
                diagnostics[f"{gap_key}_model_count"] = torch.full(
                    (int(batch.num_graphs),),
                    selected_model_count,
                    dtype=torch.long,
                    device=batch.positions.device,
                )

            uncertainty_override = prediction.get("uncertainty")
            if uncertainty_override is None:
                diagnostics["Es"] = selected["energy_std"]
                diagnostics["Fs"] = selected["Fs"]
                diagnostics["Fsmax"] = selected["Fsmax"]
                diagnostics["Fsrms"] = selected["Fsrms"]
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
            selected_energy = selected["energy"].reshape(num_graphs, 1).clone()
            selected_forces = selected["forces"].reshape(
                num_graphs, num_atoms, 3
            ).clone()
            for graph_index, control in sorted(self._lcm_controls.items()):
                if graph_index >= num_graphs:
                    raise IndexError(
                        f"LCM graph index {graph_index} is outside batch size "
                        f"{num_graphs}."
                    )
                lower, upper = control["pair"]
                missing_states = [
                    state
                    for state in (lower, upper)
                    if state not in state_results
                ]
                if missing_states:
                    raise KeyError(
                        f"LCM pair {(lower, upper)} requires raw energy and "
                        f"force contributions for states {missing_states}."
                    )
                lower_result = state_results[lower]
                upper_result = state_results[upper]
                lcm_energy, lcm_forces = self._lcm_energy_forces(
                    lower_result["energy"][graph_index],
                    upper_result["energy"][graph_index],
                    lower_result["forces"].reshape(
                        num_graphs, num_atoms, 3
                    )[graph_index],
                    upper_result["forces"].reshape(
                        num_graphs, num_atoms, 3
                    )[graph_index],
                    sigma=float(control["sigma"]),
                    alpha_eV=float(control["alpha_eV"]),
                )
                selected_energy[graph_index, 0] = lcm_energy
                selected_forces[graph_index] = lcm_forces
            well_energy, well_forces = self._well_energy_forces(
                batch, positions_batched
            )
            if well_energy is not None:
                selected_energy = selected_energy + well_energy
                selected_forces = selected_forces + well_forces
            diagnostics["Fmeanmax"] = torch.amax(
                torch.linalg.vector_norm(selected_forces, dim=2),
                dim=1,
            )
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
            gap_rows: list[dict[str, Any]] | None = None,
            gap_seeking_settings: dict[str, Any] | None = None,
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
                gap_rows=gap_rows,
                gap_seeking_settings=gap_seeking_settings,
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
            required_lcm_states = {
                int(state)
                for row in self.gap_seeking_settings.get("rows", [])
                for state in (row["lower_state"], row["upper_state"])
            }
            state_members = {
                state: {"energy": [], "forces": []}
                for state in sorted(required_lcm_states)
            }
            for model in self.models:
                output = model(batch)
                if self.energy_key not in output or self.force_key not in output:
                    raise KeyError(
                        "Native calculator output must contain configured energy and force keys."
                    )
                energies.append(output[self.energy_key])
                forces.append(output[self.force_key])
                if required_lcm_states:
                    raw_states = {
                        int(state): values
                        for state, values in dict(
                            output.get("state_contributions") or {}
                        ).items()
                    }
                    missing = sorted(required_lcm_states - set(raw_states))
                    if missing:
                        raise KeyError(
                            "Native gap seeking requires state_contributions "
                            f"with energy and force values for states {missing}."
                        )
                    for state in sorted(required_lcm_states):
                        values = raw_states[state]
                        energy_value = values.get(
                            "energy",
                            values.get("energy_contributions"),
                        )
                        force_value = values.get(
                            "forces",
                            values.get("force_contributions"),
                        )
                        if energy_value is None or force_value is None:
                            raise KeyError(
                                "Native gap-seeking state_contributions must "
                                "provide energy/forces or "
                                "energy_contributions/force_contributions."
                            )
                        state_members[state]["energy"].append(energy_value)
                        state_members[state]["forces"].append(force_value)
            prediction = {
                "energy_contributions": torch.stack(energies, dim=0),
                "force_contributions": torch.stack(forces, dim=0),
            }
            if state_members:
                prediction["state_contributions"] = {
                    state: {
                        "energy_contributions": torch.stack(
                            values["energy"], dim=0
                        ),
                        "force_contributions": torch.stack(
                            values["forces"], dim=0
                        ),
                    }
                    for state, values in state_members.items()
                }
            return prediction


    class ALFASEAlchemiModel(ALFAlchemiCalculator):
        """Compatibility wrapper for existing ALF ASE calculator loaders."""

        def __init__(
            self,
            calculators,
            *,
            model_mode: str,
            selected_state_value: int | None,
            gap_rows: list[dict[str, Any]] | None = None,
            gap_seeking_settings: dict[str, Any] | None = None,
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
                gap_rows=gap_rows,
                gap_seeking_settings=gap_seeking_settings,
                well_params=well_params,
                device=device,
                calculator_interface="ase_fallback",
                calculator_loader=calculator_loader,
            )
            mode = str(model_mode).strip().lower()
            self.energy_key = "energy" if mode == "ground_state" else f"sE{selected}"
            self.force_key = "forces" if mode == "ground_state" else f"F{selected}"
            gap_states = {
                int(state)
                for row in self.gap_rows
                for state in (row["lower_state"], row["upper_state"])
            }
            self.gap_energy_keys = {
                state: f"sE{state}" for state in sorted(gap_states)
            }
            lcm_states = {
                int(state)
                for row in self.gap_seeking_settings.get("rows", [])
                for state in (row["lower_state"], row["upper_state"])
            }
            self.gap_force_keys = {
                state: f"F{state}" for state in sorted(lcm_states)
            }
            self.calculators = calculator_list
            self.uncertainty_mode = (
                None if uncertainty_mode is None else str(uncertainty_mode).lower()
            )

        def set_device(self, device: Any) -> None:
            self._device = torch.device(device)

        def _evaluate_one(self, calculator, atoms):
            requested = list(
                dict.fromkeys(
                    [
                        self.energy_key,
                        self.force_key,
                        *self.gap_energy_keys.values(),
                        *self.gap_force_keys.values(),
                    ]
                )
            )
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
            required = [
                self.energy_key,
                self.force_key,
                *self.gap_energy_keys.values(),
                *self.gap_force_keys.values(),
            ]
            missing = [
                key for key in required if key not in calculator.results
            ]
            if missing:
                raise KeyError(
                    "ASE fallback calculator did not provide required properties: "
                    + ", ".join(missing)
                )
            energy = float(np.asarray(calculator.results[self.energy_key]).sum())
            forces = np.asarray(calculator.results[self.force_key], dtype=float)
            state_energies = {
                state: float(np.asarray(calculator.results[key]).sum())
                for state, key in self.gap_energy_keys.items()
            }
            state_forces = {
                state: np.asarray(calculator.results[key], dtype=float)
                for state, key in self.gap_force_keys.items()
            }
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
            return energy, forces, uncertainty, state_energies, state_forces

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
            state_energy_members = {
                state: [[] for _ in self.calculators]
                for state in self.gap_energy_keys
            }
            state_force_members = {
                state: [[] for _ in self.calculators]
                for state in self.gap_force_keys
            }
            uncertainty_rows = []
            for graph_index in range(num_graphs):
                atoms = Atoms(
                    numbers=numbers[graph_index],
                    positions=positions[graph_index],
                )
                graph_uncertainty = None
                for model_index, calculator in enumerate(self.calculators):
                    (
                        energy,
                        forces,
                        uncertainty,
                        state_energies,
                        state_forces,
                    ) = self._evaluate_one(calculator, atoms)
                    if forces.shape != (num_atoms, 3):
                        raise ValueError(
                            f"ASE fallback {self.force_key} must have shape "
                            f"({num_atoms}, 3)."
                        )
                    energy_members[model_index].append(energy)
                    force_members[model_index].append(forces)
                    for state, state_energy in state_energies.items():
                        state_energy_members[state][model_index].append(
                            state_energy
                        )
                    for state, state_force in state_forces.items():
                        if state_force.shape != (num_atoms, 3):
                            raise ValueError(
                                f"ASE fallback F{state} must have shape "
                                f"({num_atoms}, 3)."
                            )
                        state_force_members[state][model_index].append(
                            state_force
                        )
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
            if state_energy_members:
                prediction["state_energy_contributions"] = {
                    state: torch.as_tensor(
                        values,
                        dtype=batch.positions.dtype,
                        device=batch.positions.device,
                    )
                    for state, values in state_energy_members.items()
                }
            if state_force_members:
                prediction["state_contributions"] = {
                    state: {
                        "energy_contributions": torch.as_tensor(
                            state_energy_members[state],
                            dtype=batch.positions.dtype,
                            device=batch.positions.device,
                        ),
                        "force_contributions": torch.as_tensor(
                            np.asarray(values).reshape(
                                len(self.calculators),
                                num_graphs * num_atoms,
                                3,
                            ),
                            dtype=batch.positions.dtype,
                            device=batch.positions.device,
                        ),
                    }
                    for state, values in state_force_members.items()
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
            gap_rows: list[dict[str, Any]] | None = None,
            gap_seeking_settings: dict[str, Any] | None = None,
            species_key: str,
            coordinates_key: str,
            well_params: dict[str, Any] | None = None,
            device: Any = None,
            calculator_loader: str | None = None,
        ) -> None:
            super().__init__(
                selected_state_value=selected_state_value,
                gap_rows=gap_rows,
                gap_seeking_settings=gap_seeking_settings,
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
        if strict_gpu and not allow_cpu_debug:
            raise RuntimeError(
                "ALCHEMI sampling requires CUDA by default. Set "
                "alchemi_baoab.allow_cpu_debug=true only for local smoke tests."
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
    run_options = {"device": device}
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
