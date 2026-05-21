from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
from ase.data import chemical_symbols

from alframework.tools.tools import compute_empirical_formula


_ENERGY_KEY_RE = re.compile(r"^sE(\d+)$")
_FORCE_KEY_RE = re.compile(r"^F(\d+)$")
_GAP_KEY_RE = re.compile(r"^dE(?:(\d+)[_-](\d+)|(\d)(\d))$")
_MOLECULE_ID_INT_RE = re.compile(r"(\d+)(?!.*\d)")


def _decode_symbol(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (int, np.integer)):
        return chemical_symbols[int(value)]
    return str(value)


def stable_uint32_seed(*parts: Any) -> int:
    digest = hashlib.sha1(repr(tuple(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


def empirical_formula_from_numbers(atomic_numbers: np.ndarray | list[int]) -> str:
    numbers = np.asarray(atomic_numbers, dtype=np.int64).reshape(-1)
    numbers = numbers[numbers > 0]
    if numbers.size == 0:
        raise ValueError("Cannot compute empirical formula from an empty atomic-number list.")
    symbols = [chemical_symbols[int(z)] for z in numbers]
    return compute_empirical_formula(symbols)


def empirical_formula_from_species(species: list[str] | np.ndarray) -> str:
    symbols = [_decode_symbol(value) for value in np.asarray(species).reshape(-1).tolist()]
    if not symbols:
        raise ValueError("Cannot compute empirical formula from an empty species list.")
    return compute_empirical_formula(symbols)


def derive_state_property_table(
    properties_list: dict[str, list[Any]],
    *,
    require_forces: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(properties_list, dict) or not properties_list:
        raise ValueError("properties_list must be a non-empty dictionary.")

    energy_entries: dict[int, dict[str, Any]] = {}
    force_entries: dict[int, dict[str, Any]] = {}

    for prop_key, schema in properties_list.items():
        if not isinstance(schema, (list, tuple)) or len(schema) < 1:
            raise ValueError(f"Property schema for {prop_key!r} must be a non-empty list or tuple.")
        match_energy = _ENERGY_KEY_RE.match(str(prop_key))
        match_force = _FORCE_KEY_RE.match(str(prop_key))
        if match_energy:
            state = int(match_energy.group(1))
            energy_entries[state] = {
                "state": state,
                "energy_key": str(prop_key),
                "energy_db_name": str(schema[0]),
            }
        elif match_force:
            state = int(match_force.group(1))
            force_entries[state] = {
                "state": state,
                "force_key": str(prop_key),
                "force_db_name": str(schema[0]),
            }

    if not energy_entries:
        raise ValueError("properties_list does not define any excited-state energy keys like sE0, sE1, ...")

    state_ids = sorted(energy_entries)
    expected = list(range(state_ids[-1] + 1))
    if state_ids != expected:
        raise ValueError(f"Excited-state energy keys must be contiguous from 0. Found {state_ids}.")

    rows: list[dict[str, Any]] = []
    for state in state_ids:
        row = dict(energy_entries[state])
        if state in force_entries:
            row.update(force_entries[state])
        elif require_forces:
            raise ValueError(f"Missing force key F{state} for excited-state sampling or training.")
        else:
            row["force_key"] = None
            row["force_db_name"] = None
        rows.append(row)
    return rows


def derive_state_ids(properties_list: dict[str, list[Any]]) -> list[int]:
    return [int(row["state"]) for row in derive_state_property_table(properties_list)]


def gap_key_for_pair(lower_state: int, upper_state: int) -> str:
    lower_state = int(lower_state)
    upper_state = int(upper_state)
    if 0 <= lower_state <= 9 and 0 <= upper_state <= 9:
        return f"dE{lower_state}{upper_state}"
    return f"dE{lower_state}_{upper_state}"


def parse_gap_key(prop_key: str) -> tuple[int, int] | None:
    match = _GAP_KEY_RE.match(str(prop_key))
    if match is None:
        return None
    if match.group(1) is not None:
        lower_state = int(match.group(1))
        upper_state = int(match.group(2))
    else:
        lower_state = int(match.group(3))
        upper_state = int(match.group(4))
    if lower_state == upper_state:
        raise ValueError(f"Gap key {prop_key!r} uses the same state twice.")
    return lower_state, upper_state


def derive_gap_property_table(
    properties_list: dict[str, list[Any]],
    *,
    gap_config: dict[str, Any] | None = None,
    require_properties: bool = False,
) -> list[dict[str, Any]]:
    if not isinstance(properties_list, dict) or not properties_list:
        raise ValueError("properties_list must be a non-empty dictionary.")

    state_ids = set(derive_state_ids(properties_list))
    config = dict(gap_config or {})
    configured_pairs = config.get("pairs")
    if configured_pairs is None:
        pair_items = []
        for prop_key in properties_list:
            parsed = parse_gap_key(str(prop_key))
            if parsed is not None:
                pair_items.append(parsed)
        pair_items = sorted(set(pair_items))
    else:
        pair_items = []
        for pair in configured_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"Gap target pair must contain exactly two states. Got {pair!r}.")
            pair_items.append((int(pair[0]), int(pair[1])))

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for lower_state, upper_state in pair_items:
        lower_state = int(lower_state)
        upper_state = int(upper_state)
        if lower_state >= upper_state:
            raise ValueError(f"Gap targets must be ordered lower-to-higher state. Got {(lower_state, upper_state)}.")
        if lower_state not in state_ids or upper_state not in state_ids:
            raise ValueError(
                f"Gap target {(lower_state, upper_state)} references states outside available states {sorted(state_ids)}."
            )
        pair = (lower_state, upper_state)
        if pair in seen:
            continue
        seen.add(pair)

        gap_key = gap_key_for_pair(lower_state, upper_state)
        if gap_key not in properties_list:
            if require_properties:
                raise ValueError(f"Gap target {gap_key!r} is not present in properties_list.")
            continue

        schema = properties_list[gap_key]
        if not isinstance(schema, (list, tuple)) or len(schema) < 2:
            raise ValueError(f"Property schema for {gap_key!r} must include database name and property kind.")
        prop_kind = str(schema[1]).strip().lower()
        if prop_kind != "system":
            raise ValueError(f"Gap property {gap_key!r} must be a system property, not {prop_kind!r}.")

        rows.append(
            {
                "lower_state": lower_state,
                "upper_state": upper_state,
                "gap_key": gap_key,
                "gap_db_name": str(schema[0]),
            }
        )
    return rows


def validate_gap_results(
    results: dict[str, Any],
    properties_list: dict[str, list[Any]],
    *,
    gap_config: dict[str, Any] | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-7,
) -> list[str]:
    errors: list[str] = []
    for row in derive_gap_property_table(properties_list, gap_config=gap_config):
        lower_key = f"sE{int(row['lower_state'])}"
        upper_key = f"sE{int(row['upper_state'])}"
        gap_key = str(row["gap_key"])
        missing = [key for key in (lower_key, upper_key, gap_key) if key not in results]
        if missing:
            errors.append(f"{gap_key} validation missing keys: {missing}")
            continue
        lower = float(np.asarray(results[lower_key], dtype=np.float64).reshape(-1)[0])
        upper = float(np.asarray(results[upper_key], dtype=np.float64).reshape(-1)[0])
        gap = float(np.asarray(results[gap_key], dtype=np.float64).reshape(-1)[0])
        expected = upper - lower
        if not np.all(np.isfinite([lower, upper, gap])):
            errors.append(f"{gap_key} validation found non-finite values")
            continue
        if not np.isclose(gap, expected, atol=float(atol), rtol=float(rtol)):
            errors.append(f"{gap_key}={gap} does not match {upper_key}-{lower_key}={expected}")
    return errors


def select_excited_state(
    *,
    moleculeid: str,
    available_states: list[int],
    metadata: dict[str, Any] | None = None,
    selection_config: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    if not available_states:
        raise ValueError("available_states must be non-empty.")

    metadata = metadata or {}
    if "excited_state" in metadata:
        selected = int(metadata["excited_state"])
        if selected not in available_states:
            raise ValueError(
                f"Metadata excited_state={selected} is not one of the available states {available_states}."
            )
        return selected

    config = selection_config or {}
    mode = str(config.get("mode", "cyclic")).strip().lower()
    if mode == "fixed":
        fixed_state = int(config.get("fixed_state", available_states[0]))
        if fixed_state not in available_states:
            raise ValueError(
                f"Fixed excited state {fixed_state} is not one of the available states {available_states}."
            )
        return fixed_state

    if mode == "random":
        if rng is None:
            rng = np.random.default_rng()
        return int(rng.choice(np.asarray(available_states, dtype=np.int64)))

    if mode != "cyclic":
        raise ValueError(f"Unsupported excited-state selection mode: {mode}")

    match = _MOLECULE_ID_INT_RE.search(str(moleculeid))
    if match:
        cycle_index = int(match.group(1))
    else:
        cycle_index = stable_uint32_seed(str(moleculeid))
    return int(available_states[cycle_index % len(available_states)])
