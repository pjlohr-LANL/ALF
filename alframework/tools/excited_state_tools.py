from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
from ase.data import chemical_symbols

from alframework.tools.tools import compute_empirical_formula


_ENERGY_KEY_RE = re.compile(r"^sE(\d+)$")
_FORCE_KEY_RE = re.compile(r"^F(\d+)$")
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
