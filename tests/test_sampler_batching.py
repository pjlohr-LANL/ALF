import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from alframework.builders.builders import simple_cfg_loader_task
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampler_batching import (
    SamplerBatchBuffer,
    flatten_molecule_output,
    sampler_batch_key,
    sampler_batch_size,
    selected_state,
    validate_state_selection,
)


def _molecule(molecule_id, symbols="H2", state=None):
    atoms = Atoms(symbols, positions=np.zeros((len(symbols), 3)))
    molecule = MoleculesObject(atoms, molecule_id)
    if state is not None:
        molecule.update_metadata({"selected_state": state})
    return molecule


def test_strict_full_buffer_groups_exact_atom_order():
    config = {"model_mode": "ground_state", "alchemi_baoab": {"batch_size": 2}}
    buffer = SamplerBatchBuffer()
    first = _molecule("first", "OH")
    second = _molecule("second", "OH")
    different_order = _molecule("different", "HO")

    buffer.add(first, config, buffered_at=10.0)
    buffer.add(different_order, config, buffered_at=20.0)
    assert buffer.pop_ready(2) == []
    assert len(buffer) == 2

    buffer.add(second, config, buffered_at=30.0)
    batches = buffer.pop_ready(2)
    assert [[item.get_moleculeid() for item in batch] for batch in batches] == [
        ["first", "second"]
    ]
    assert [item.get_moleculeid() for item in buffer.molecules()] == ["different"]


def test_excited_state_is_part_of_batch_key():
    config = {"model_mode": "excited_state", "alchemi_baoab": {"batch_size": 2}}
    state_zero = _molecule("zero", state=0)
    state_one = _molecule("one", state=1)

    assert sampler_batch_key(state_zero, config) != sampler_batch_key(state_one, config)


def test_batch_cycle_assigns_complete_state_blocks_and_repeats():
    config = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "batch_cycle", "states": [0, 1]},
        "alchemi_baoab": {"batch_size": 2},
    }
    buffer = SamplerBatchBuffer()

    for index in range(8):
        buffer.add(_molecule(f"mol-0003-{index:010d}"), config)

    batches = buffer.pop_ready(2)
    assert [
        [item.get_metadata()["selected_state"] for item in batch]
        for batch in batches
    ] == [[0, 0], [0, 0], [1, 1], [1, 1]]
    assigned_by_id = {
        item.get_moleculeid(): item.get_metadata()["selected_state"]
        for batch in batches
        for item in batch
    }
    assert [assigned_by_id[f"mol-0003-{index:010d}"] for index in range(8)] == [
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
    ]


@pytest.mark.parametrize(
    "sequence_index,expected_state",
    [
        (0, 0),
        (49, 0),
        (50, 1),
        (99, 1),
        (100, 2),
        (149, 2),
        (150, 3),
        (199, 3),
        (200, 0),
    ],
)
def test_batch_cycle_matches_documented_fifty_replica_boundaries(
    sequence_index, expected_state
):
    config = {
        "model_mode": "excited_state",
        "state_selection": {
            "mode": "batch_cycle",
            "states": [0, 1, 2, 3],
        },
        "alchemi_baoab": {"batch_size": 50},
    }
    molecule = _molecule(f"mol-0000-{sequence_index:010d}")

    assert selected_state(molecule, config) == expected_state


def test_state_metadata_overrides_cycle_and_legacy_key_is_normalized():
    config = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "batch_cycle", "states": [0, 1]},
        "alchemi_baoab": {"batch_size": 2},
    }
    explicit = _molecule("mol-0000000000", state=3)
    legacy = _molecule("mol-0000000000")
    legacy.update_metadata({"excited_state": 2})

    assert selected_state(explicit, config) == 3
    assert explicit.get_metadata()["selected_state_source"] == "metadata.selected_state"
    assert selected_state(legacy, config) == 2
    assert legacy.get_metadata()["selected_state"] == 2
    assert legacy.get_metadata()["selected_state_source"] == "metadata.excited_state"


def test_fixed_state_and_non_numeric_ids_are_deterministic():
    fixed_config = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "fixed", "state": 2},
        "alchemi_baoab": {"batch_size": 4},
    }
    assert selected_state(_molecule("without-digits"), fixed_config) == 2

    cycle_config = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "batch_cycle", "states": [0, 1, 2]},
        "alchemi_baoab": {"batch_size": 1},
    }
    first = selected_state(_molecule("same-nonnumeric-id"), cycle_config)
    second = selected_state(_molecule("same-nonnumeric-id"), cycle_config)
    assert first == second


@pytest.mark.parametrize(
    "selection,error",
    [
        ({"mode": "batch_cycle", "states": []}, "non-empty"),
        ({"mode": "batch_cycle", "states": [0, 0]}, "duplicate"),
        ({"mode": "batch_cycle", "states": [0, -1]}, "nonnegative"),
        ({"mode": "fixed"}, "requires"),
        ({"mode": "random", "states": [0, 1]}, "fixed.*batch_cycle"),
    ],
)
def test_invalid_state_selection_is_rejected(selection, error):
    config = {
        "model_mode": "excited_state",
        "state_selection": selection,
        "alchemi_baoab": {"batch_size": 2},
    }
    with pytest.raises(ValueError, match=error):
        validate_state_selection(config)


def test_configured_states_are_validated_against_properties():
    config = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "batch_cycle", "states": [0, 2]},
        "alchemi_baoab": {"batch_size": 2},
    }
    properties = {
        "sE0": ["state_0_energy", "system", 1.0],
        "F0": ["state_0_forces", "atomic", 1.0],
        "sE1": ["state_1_energy", "system", 1.0],
        "F1": ["state_1_forces", "atomic", 1.0],
    }
    with pytest.raises(ValueError, match=r"states \[2\].*available states.*\[0, 1\]"):
        validate_state_selection(config, properties)


def test_cfg_loader_hands_structures_to_state_specific_batches(tmp_path):
    library = tmp_path / "cfg_library"
    library.mkdir()
    write(
        library / "water.cfg",
        Atoms(
            "OH2",
            positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ],
        ),
        format="cfg",
    )
    config = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "batch_cycle", "states": [0, 1]},
        "alchemi_baoab": {"batch_size": 2},
    }
    buffer = SamplerBatchBuffer()

    for index in range(4):
        molecule = simple_cfg_loader_task.func(
            moleculeid=f"mol-0000-{index:010d}",
            builder_config={"molecule_library_dir": str(library)},
            shake=0.0,
        )
        buffer.add(molecule, config)

    batches = buffer.pop_ready(2)
    assert len(batches) == 2
    assert [batch[0].get_metadata()["selected_state"] for batch in batches] == [0, 1]
    assert all(
        item.get_metadata()["selected_state_source"]
        == "state_selection.batch_cycle"
        for batch in batches
        for item in batch
    )


def test_incomplete_batches_remain_visible_without_being_dropped():
    config = {"model_mode": "ground_state", "alchemi_baoab": {"batch_size": 4}}
    buffer = SamplerBatchBuffer()
    buffer.add(_molecule("waiting"), config, buffered_at=100.0)

    assert buffer.pop_ready(4) == []
    summary = buffer.status(now=160.0)
    assert summary["buffered_structures"] == 1
    assert summary["oldest_age_seconds"] == 60.0
    assert summary["buckets"][0]["count"] == 1


def test_buffer_status_reports_cycle_assigned_state():
    config = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "batch_cycle", "states": [0, 1]},
        "alchemi_baoab": {"batch_size": 2},
    }
    buffer = SamplerBatchBuffer()
    buffer.add(_molecule("mol-0000000002"), config, buffered_at=100.0)

    summary = buffer.status(now=110.0)
    assert summary["buckets"][0]["selected_state"] == 1
    assert next(buffer.molecules()).get_metadata()["selected_state"] == 1


def test_flatten_molecule_output_handles_single_nested_and_empty():
    first = _molecule("first")
    second = _molecule("second")

    assert flatten_molecule_output(first) == [first]
    assert flatten_molecule_output([first, (second,)]) == [first, second]
    assert flatten_molecule_output(None) == []
    with pytest.raises(TypeError, match="Sampler output"):
        flatten_molecule_output("not-a-molecule")


def test_batch_configuration_accepts_only_full_only_policy():
    assert sampler_batch_size({"alchemi_baoab": {"batch_size": 3}}) == 3
    with pytest.raises(ValueError, match="full_only"):
        sampler_batch_size(
            {"alchemi_baoab": {"batch_size": 3, "partial_policy": "timeout"}}
        )
