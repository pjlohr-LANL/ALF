"""Small, dependency-free contracts shared by excited-state components."""

from __future__ import annotations

import re
from typing import Any


_ENERGY_KEY = re.compile(r"^sE(\d+)$")
_FORCE_KEY = re.compile(r"^F(\d+)$")


def derive_state_property_table(
    properties_list: dict[str, list[Any]],
    *,
    require_forces: bool = False,
) -> list[dict[str, Any]]:
    """Validate flattened ``sE#``/``F#`` properties and describe each state."""

    if not isinstance(properties_list, dict) or not properties_list:
        raise ValueError("properties_list must be a non-empty dictionary.")

    energies: dict[int, dict[str, Any]] = {}
    forces: dict[int, dict[str, Any]] = {}
    for key, schema in properties_list.items():
        if not isinstance(schema, (list, tuple)) or not schema:
            raise ValueError(f"Property schema for {key!r} must be a non-empty list.")
        energy_match = _ENERGY_KEY.fullmatch(str(key))
        force_match = _FORCE_KEY.fullmatch(str(key))
        if energy_match:
            state = int(energy_match.group(1))
            energies[state] = {
                "state": state,
                "energy_key": str(key),
                "energy_db_name": str(schema[0]),
            }
        elif force_match:
            state = int(force_match.group(1))
            forces[state] = {
                "force_key": str(key),
                "force_db_name": str(schema[0]),
            }

    if not energies:
        raise ValueError(
            "properties_list must define excited-state energies named sE0, sE1, ..."
        )
    states = sorted(energies)
    if states != list(range(states[-1] + 1)):
        raise ValueError(
            f"Excited-state energy keys must be contiguous from zero; found {states}."
        )

    rows: list[dict[str, Any]] = []
    for state in states:
        row = dict(energies[state])
        if state in forces:
            row.update(forces[state])
        elif require_forces:
            raise ValueError(f"Missing excited-state force property F{state}.")
        else:
            row.update({"force_key": None, "force_db_name": None})
        rows.append(row)
    return rows
