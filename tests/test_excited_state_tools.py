from __future__ import annotations

import numpy as np
import pytest

from alframework.tools.excited_state_tools import (
    derive_state_property_table,
    empirical_formula_from_species,
    select_excited_state,
    stable_uint32_seed,
)


def test_derive_state_property_table_parses_states_and_forces():
    properties_list = {
        "sE0": ["sE0", "system", 1.0],
        "F0": ["F0", "atomic", 1.0],
        "sE1": ["sE1", "system", 1.0],
        "F1": ["F1", "atomic", 1.0],
    }

    rows = derive_state_property_table(properties_list, require_forces=True)

    assert [row["state"] for row in rows] == [0, 1]
    assert rows[0]["energy_key"] == "sE0"
    assert rows[0]["force_key"] == "F0"
    assert rows[1]["energy_db_name"] == "sE1"
    assert rows[1]["force_db_name"] == "F1"


def test_derive_state_property_table_rejects_state_gaps():
    properties_list = {
        "sE0": ["sE0", "system", 1.0],
        "F0": ["F0", "atomic", 1.0],
        "sE2": ["sE2", "system", 1.0],
        "F2": ["F2", "atomic", 1.0],
    }

    with pytest.raises(ValueError, match="contiguous"):
        derive_state_property_table(properties_list)


def test_select_excited_state_modes_are_stable():
    available = [0, 1, 2]

    assert select_excited_state(
        moleculeid="traj_0010",
        available_states=available,
        selection_config={"mode": "fixed", "fixed_state": 2},
    ) == 2
    assert select_excited_state(
        moleculeid="traj_0010",
        available_states=available,
        selection_config={"mode": "cyclic"},
    ) == 1

    first = select_excited_state(
        moleculeid="traj_without_digits",
        available_states=available,
        selection_config={"mode": "cyclic"},
    )
    second = select_excited_state(
        moleculeid="traj_without_digits",
        available_states=available,
        selection_config={"mode": "cyclic"},
    )
    assert first == second

    rng = np.random.default_rng(7)
    random_state = select_excited_state(
        moleculeid="traj_0000",
        available_states=available,
        selection_config={"mode": "random"},
        rng=rng,
    )
    assert random_state in available


def test_stable_seed_and_formula_helpers_handle_numeric_species():
    assert stable_uint32_seed("abc", 3) == stable_uint32_seed("abc", 3)
    assert empirical_formula_from_species(np.array([8, 1, 1])) == "H02_O01"
