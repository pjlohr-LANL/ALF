import numpy as np
import pytest
from ase import Atoms

from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampler_batching import (
    SamplerBatchBuffer,
    flatten_molecule_output,
    sampler_batch_key,
    sampler_batch_size,
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


def test_incomplete_batches_remain_visible_without_being_dropped():
    config = {"model_mode": "ground_state", "alchemi_baoab": {"batch_size": 4}}
    buffer = SamplerBatchBuffer()
    buffer.add(_molecule("waiting"), config, buffered_at=100.0)

    assert buffer.pop_ready(4) == []
    summary = buffer.status(now=160.0)
    assert summary["buffered_structures"] == 1
    assert summary["oldest_age_seconds"] == 60.0
    assert summary["buckets"][0]["count"] == 1


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
