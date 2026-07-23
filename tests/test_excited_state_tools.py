import pytest

from alframework.tools.excited_state_tools import derive_state_property_table


def test_flattened_excited_state_contract():
    table = derive_state_property_table(
        {
            "sE0": ["singlet_0_energy", "system", 1.0],
            "F0": ["singlet_0_forces", "atomic", 1.0],
            "sE1": ["singlet_1_energy", "system", 1.0],
            "F1": ["singlet_1_forces", "atomic", 1.0],
        },
        require_forces=True,
    )

    assert [row["state"] for row in table] == [0, 1]
    assert table[1]["energy_key"] == "sE1"
    assert table[1]["force_db_name"] == "singlet_1_forces"


def test_excited_state_contract_requires_contiguous_energies():
    with pytest.raises(ValueError, match="contiguous"):
        derive_state_property_table(
            {"sE0": ["e0"], "F0": ["f0"], "sE2": ["e2"], "F2": ["f2"]},
            require_forces=True,
        )


def test_excited_state_contract_requires_each_force_for_sampling():
    with pytest.raises(ValueError, match="F1"):
        derive_state_property_table(
            {"sE0": ["e0"], "F0": ["f0"], "sE1": ["e1"]},
            require_forces=True,
        )
