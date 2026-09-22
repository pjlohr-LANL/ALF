"""Validate the checked-in NACR labeling example end to end, offline.

The example computes state energies, forces, and excited-excited nonadiabatic
coupling, but trains only on energies and forces. These tests assert both halves
of that contract from the real configuration files, without a GPU.
"""

import json
from pathlib import Path

import pytest

from alframework.ml_interfaces.excited_state_hippynn_interface import (
    validate_excited_state_training_config,
)
from alframework.qm_interfaces.gpu4pyscf_interface import (
    NACR_PROPERTY_KEY,
    excited_excited_nac_pairs,
    validate_gpu4pyscf_config,
    validate_gpu4pyscf_properties,
)
from alframework.samplers.alchemi_sampling import (
    _configured_replica_candidate_limit,
    _configured_return_top_k,
    configured_score,
)
from alframework.tools.molecular_topology import load_reference_topology


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NACR_DIR = EXAMPLES / "excited_state_gpu4pyscf_nacr"
SUMMED_DIR = EXAMPLES / "excited_state_gpu4pyscf"


def _load(directory, filename):
    return json.loads((directory / filename).read_text(encoding="utf-8"))


def _configs():
    master = _load(NACR_DIR, "master_config.json")
    return (
        master,
        _load(NACR_DIR, master["sampler_config_path"]),
        _load(NACR_DIR, master["QM_config_path"]),
        _load(NACR_DIR, master["ML_config_path"]),
    )


def test_every_example_json_parses_and_paths_resolve():
    master, _, _, _ = _configs()
    for filename in sorted(NACR_DIR.glob("*.json")):
        json.loads(filename.read_text(encoding="utf-8"))
    for key in (
        "sampler_config_path",
        "QM_config_path",
        "ML_config_path",
        "builder_config_path",
    ):
        assert (NACR_DIR / master[key]).is_file()


def test_qm_config_enables_coupling_with_the_declared_property():
    master, _, qm_config, _ = _configs()
    options = validate_gpu4pyscf_config(qm_config)

    assert options["compute_nacr"] is True
    assert options["nroots"] == 5
    state_table, _ = validate_gpu4pyscf_properties(
        master["properties_list"],
        nroots=options["nroots"],
        compute_nacr=options["compute_nacr"],
    )
    assert len(state_table) == 6
    assert master["properties_list"][NACR_PROPERTY_KEY][1] == "pair_atomic"
    # nroots=5 yields the ten ordered excited-excited pairs.
    assert len(excited_excited_nac_pairs(options["nroots"])) == 10


def test_coupling_is_stored_but_never_a_training_target():
    master, _, _, ml_config = _configs()
    state_table = validate_excited_state_training_config(
        ml_config, master["properties_list"]
    )

    trainable = {str(row["energy_key"]) for row in state_table} | {
        str(row["force_key"]) for row in state_table
    }
    assert trainable == {f"sE{i}" for i in range(6)} | {
        f"F{i}" for i in range(6)
    }
    # The whole point: nacr is declared for storage and excluded from training.
    assert NACR_PROPERTY_KEY in master["properties_list"]
    assert NACR_PROPERTY_KEY not in trainable


def test_sampler_inherits_the_summed_score_settings():
    _, sampler_config, _, _ = _configs()

    assert sampler_config["uncertainty_policy"] == "continue"
    assert sampler_config["score"]["mode"] == "sum"
    assert _configured_return_top_k(sampler_config) == 100
    assert _configured_replica_candidate_limit(sampler_config) == 10
    assert sampler_config["Ncheck"] == 10


def test_sampler_and_training_configs_match_the_summed_score_variant():
    """Only the NACR additions differ; sampling and training are inherited."""

    _, sampler_config, _, ml_config = _configs()

    assert sampler_config == _load(
        SUMMED_DIR, "sampler_config_summed_score.json"
    )
    assert ml_config == _load(SUMMED_DIR, "hippynn_config_summed_score.json")


def test_master_config_differs_from_summed_score_only_in_paths_and_nacr():
    master, _, _, _ = _configs()
    summed = _load(SUMMED_DIR, "master_config_summed_score.json")

    changed = {
        key
        for key in set(master) | set(summed)
        if master.get(key) != summed.get(key)
    }
    assert changed == {
        "sampler_config_path",
        "ML_config_path",
        "properties_list",
    }
    # properties_list differs by exactly the nacr addition.
    assert set(master["properties_list"]) - set(summed["properties_list"]) == {
        NACR_PROPERTY_KEY
    }
    assert not set(summed["properties_list"]) - set(master["properties_list"])


@pytest.mark.parametrize("state", range(6))
def test_score_gap_pairs_resolve_for_every_cycled_state(state):
    master, sampler_config, _, _ = _configs()
    score = configured_score(sampler_config, master["properties_list"], state)

    pairs = [
        (int(row["lower_state"]), int(row["upper_state"]))
        for row in score["gap_rows"]
    ]
    expected = [
        pair
        for pair in ((state - 1, state), (state, state + 1))
        if 0 <= pair[0] and pair[1] <= 5
    ]
    assert pairs == expected


def test_topology_reference_loads_from_the_example_directory():
    _, sampler_config, _, _ = _configs()
    topology = load_reference_topology(
        sampler_config, master_directory=str(NACR_DIR)
    )

    assert topology is not None
    assert len(topology.atomic_numbers) == 15
