from __future__ import annotations

from ase import Atoms

from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampler_batching import (
    add_molecule_to_same_state_buffer,
    alchemi_allow_partial_batches,
    alchemi_sampler_batch_size,
    pop_ready_same_state_batches,
)


def _molecule(molecule_id: str, state: int) -> MoleculesObject:
    molecule = MoleculesObject(
        Atoms(symbols=["H"], positions=[[0.0, 0.0, 0.0]]),
        molecule_id,
    )
    molecule.update_metadata({"excited_state": int(state)})
    return molecule


def test_same_state_buffer_submits_only_full_batches_by_default():
    buffers = {}
    for molecule in [_molecule("a", 0), _molecule("b", 1), _molecule("c", 0)]:
        add_molecule_to_same_state_buffer(buffers, molecule)

    ready = pop_ready_same_state_batches(buffers, batch_size=2)

    assert [[m.get_moleculeid() for m in batch] for batch in ready] == [["a", "c"]]
    assert list(buffers) == [1]
    assert [m.get_moleculeid() for m in buffers[1]] == ["b"]


def test_partial_batches_are_opt_in_and_forced():
    buffers = {}
    add_molecule_to_same_state_buffer(buffers, _molecule("a", 0))

    assert pop_ready_same_state_batches(buffers, batch_size=4, allow_partial=True, force_partial=False) == []
    ready = pop_ready_same_state_batches(buffers, batch_size=4, allow_partial=True, force_partial=True)

    assert [[m.get_moleculeid() for m in batch] for batch in ready] == [["a"]]
    assert buffers == {}


def test_alchemi_batch_config_defaults():
    assert alchemi_sampler_batch_size({"dynamics_backend": "ase"}) == 1
    assert alchemi_sampler_batch_size({"dynamics_backend": "alchemi_baoab"}) == 1
    assert (
        alchemi_sampler_batch_size(
            {"dynamics_backend": "alchemi_baoab", "alchemi_baoab": {"batch_size": 8}}
        )
        == 8
    )
    assert alchemi_allow_partial_batches({"dynamics_backend": "alchemi_baoab"}) is False
