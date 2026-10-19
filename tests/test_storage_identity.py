"""Main's species contract plus explicit topology identities across generations."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from ase import Atoms

from alframework.tools.atom_order import TOPOLOGY_ATOM_IDS, canonical_orders
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import store_current_data
from alframework.builders import h5_replay_builder as replay
from alframework.ml_interfaces.excited_state_hippynn_interface import load_excited_state_h5_arrays

NUMBERS = np.array([8, 6, 1, 1])
POSITIONS = np.arange(12).reshape(4,3) * .1
FORCES = np.arange(12).reshape(4,3) * .01
NAC = np.arange(24).reshape(2,4,3) * .001
PROPERTIES = {'sE0': ['sE0','system',1.], 'F0': ['F0','atomic',1.],
              'sE1': ['sE1','system',1.], 'F1': ['F1','atomic',1.],
              'nacr': ['nacr','pair_atomic',2.]}


def molecule(order, *, tagged=True):
    atoms = Atoms(numbers=NUMBERS[order], positions=POSITIONS[order])
    if tagged:
        atoms.set_array(TOPOLOGY_ATOM_IDS, np.array(order))
    mol = MoleculesObject(atoms, 'fixture')
    mol.store_results({'sE0': 1., 'sE1': 2., 'F0': FORCES[order],
                       'F1': 3 * FORCES[order], 'nacr': NAC[:,order]})
    mol.set_converged_flag(True)
    mol.update_metadata({'nac_pairs': [[0,1],[1,2]]})
    return mol


@pytest.mark.parametrize('tagged', [False, True])
def test_differently_ordered_same_formula_species_coordinates_force_nac(tmp_path, tagged):
    path = tmp_path / 'data-0000.h5'
    orders = [np.array([0,1,2,3]), np.array([3,0,2,1])]
    store_current_data(str(path), [molecule(order, tagged=tagged) for order in orders], PROPERTIES)
    with h5py.File(path) as handle:
        group = handle[next(iter(handle))]
        assert list(group['species'][()]) == [b'H',b'H',b'C',b'O']
        assert (TOPOLOGY_ATOM_IDS in group) is tagged
        for row, order in enumerate(orders):
            stored_order = order[np.argsort(NUMBERS[order], kind='stable')]
            np.testing.assert_allclose(group['coordinates'][row], POSITIONS[stored_order])
            np.testing.assert_allclose(group['F0'][row], FORCES[stored_order])
            np.testing.assert_allclose(group['F1'][row], 3 * FORCES[stored_order])
            np.testing.assert_allclose(group['nacr'][row], 2 * NAC[:,stored_order])
            if tagged:
                np.testing.assert_array_equal(group[TOPOLOGY_ATOM_IDS][row], stored_order)
                restore = np.argsort(group[TOPOLOGY_ATOM_IDS][row])
                np.testing.assert_allclose(group['nacr'][row][:,restore], 2*NAC)
        np.testing.assert_array_equal(group['nac_pairs'], [[0,1],[1,2]])


def test_mixed_old_and_new_tagged_training_and_replay_without_rewriting(tmp_path, monkeypatch):
    old = tmp_path / 'data-0000.h5'
    with h5py.File(old, 'w') as h:
        g = h.create_group('old')
        for key, value in {'species': np.array(['O','C','H','H'], dtype='S1'),
                           'coordinates': POSITIONS[None], TOPOLOGY_ATOM_IDS: np.arange(4)[None],
                           'sE0': [1.], 'sE1': [2.], 'F0': FORCES[None], 'F1': 3*FORCES[None]}.items():
            g[key] = value
    new = tmp_path / 'data-0001.h5'
    store_current_data(str(new), [molecule(np.array([3,0,2,1]))], PROPERTIES)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [old,new]}
    training_props = {key: value for key,value in PROPERTIES.items() if key != 'nacr'}
    arrays, summary = load_excited_state_h5_arrays(str(tmp_path), training_props)
    assert summary['n_structures'] == 2
    np.testing.assert_array_equal(arrays['species'], np.tile(NUMBERS, (2,1)))
    np.testing.assert_allclose(arrays['coordinates'], np.tile(POSITIONS, (2,1,1)))
    np.testing.assert_allclose(arrays['F0'], np.tile(FORCES, (2,1,1)))
    np.testing.assert_allclose(arrays['F1'], np.tile(3*FORCES, (2,1,1)))
    monkeypatch.setattr(replay, 'load_reference_topology', lambda *a, **k: SimpleNamespace(atomic_numbers=NUMBERS))
    for entry in replay.build_h5_replay_manifest(str(tmp_path/'data-{:04d}.h5'), 2):
        atoms, metadata = replay._load_replay_frame(entry, 0, sampler_config={}, master_directory=None)
        np.testing.assert_array_equal(atoms.numbers, NUMBERS)
        np.testing.assert_allclose(atoms.positions, POSITIONS)
        np.testing.assert_array_equal(atoms.arrays[TOPOLOGY_ATOM_IDS], np.arange(4))
        assert metadata['replay_canonicalized_to_topology']
        # A replay/store round trip retains exact identities even for equal species.
        mol = MoleculesObject(atoms, 'round-trip')
        mol.store_results({'energy': 1.})
        mol.set_converged_flag(True)
        store_current_data(str(tmp_path/(Path(entry.shard_path).stem + '.roundtrip.h5')), [mol], {'energy':['energy','system',1.]})
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [old,new]}


@pytest.mark.parametrize('ids', [[0,0,2,3], [-1,1,2,3], [0,1,2,4], [0.,1.,2.,3.], [[0,1],[2,3]]])
def test_invalid_ids_fail_before_write(tmp_path, ids):
    mol = molecule(np.arange(4))
    # Bypass ASE's shape guard to exercise persisted input validation.
    mol.get_atoms().arrays[TOPOLOGY_ATOM_IDS] = np.array(ids)
    path = tmp_path/'bad.h5'
    with pytest.raises(ValueError, match='permutation'):
        store_current_data(str(path), [mol], PROPERTIES)
    assert not path.exists()


def test_mixed_tagging_in_same_group_is_explicit_error(tmp_path):
    path = tmp_path/'mixed.h5'
    with pytest.raises(ValueError, match='tagged and untagged'):
        store_current_data(str(path), [molecule(np.arange(4)), molecule(np.arange(4), tagged=False)], PROPERTIES)
    assert not path.exists()


def test_invalid_tagged_training_read_does_not_cast_fractional_ids(tmp_path):
    path = tmp_path/'data-0000.h5'
    store_current_data(str(path), [molecule(np.arange(4))], PROPERTIES)
    with h5py.File(path,'a') as h:
        g = h[next(iter(h))]
        del g[TOPOLOGY_ATOM_IDS]
        g[TOPOLOGY_ATOM_IDS] = [[0., 1.5, 2., 3.]]
    with pytest.raises(ValueError, match='permutation'):
        load_excited_state_h5_arrays(str(tmp_path), {k:v for k,v in PROPERTIES.items() if k != 'nacr'})


def test_sampler_normalizes_supplied_ids_before_topology_check(monkeypatch):
    from tests.test_alchemi_sampling import (
        _topology_molecule, _reference_topology, _enable_topology, _config, FakeModel, FakeRunner)
    from alframework.samplers import alchemi_sampling as mod
    monkeypatch.setattr(mod, 'load_reference_topology', lambda *a, **k: _reference_topology())
    mol = _topology_molecule('tagged')
    original = mol.get_atoms().positions.copy()
    atoms = mol.get_atoms()[[2,0,1]]
    atoms.set_array(TOPOLOGY_ATOM_IDS, np.array([2,0,1]))
    mol = MoleculesObject(atoms, 'tagged')
    model = FakeModel({0: [{'Es': 2., 'Fs': 0., 'Fsmax': 0.}]})
    out = mod.run_alchemi_sampling([mol], _enable_topology(_config(batch_size=1)), model,
                                   runner_factory=FakeRunner)
    assert len(out) == 1
    np.testing.assert_allclose(out[0].get_atoms().positions, original + [.01,0,0])
    np.testing.assert_array_equal(out[0].get_atoms().arrays[TOPOLOGY_ATOM_IDS], np.arange(3))
    np.testing.assert_array_equal(mol.get_atoms().arrays[TOPOLOGY_ATOM_IDS], [2,0,1])
