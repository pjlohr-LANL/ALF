"""Compare training selection to the upstream HIPPYNN implementation."""
import hashlib

import h5py
import numpy as np
import pytest

from alframework.ml_interfaces import excited_state_hippynn_interface as ml


PROPERTIES = {'sE0': ['sE0', 'system', 1.], 'F0': ['F0', 'atomic', 1.]}
TABLE = ml.validate_excited_state_training_properties(PROPERTIES)


def arrays(count=12):
    coordinates = np.zeros((count,2,3), dtype=np.float64)
    coordinates[:,1,0] = 1.
    return {'coordinates': coordinates,
            'species': np.tile([1,1], (count,1)).astype(np.int64),
            'sE0': np.full(count, -2.), 'F0': np.full((count,2,3),10.),
            'indices': np.arange(count), 'frame_ids': 500 + np.arange(count)}


def upstream_selection(data, options):
    torch = pytest.importorskip('torch')
    Database = pytest.importorskip('hippynn.databases.database').Database
    values = {key: torch.as_tensor(value.copy()) for key,value in data.items()}
    values['sE0'] = values['sE0'].reshape(-1,1)
    database = Database(values, inputs=['species','coordinates'], targets=['sE0','F0'],
                        seed=42, quiet=True)
    database.remove_high_property('F0', True, species_key='species',
                                  cut=options.get('remove_high_forces_cut'),
                                  std_factor=options.get('remove_high_forces_std'))
    database.remove_high_property('sE0', False, species_key='species',
                                  cut=options.get('remove_high_energy_cut'),
                                  std_factor=options.get('remove_high_energy_std'))
    return database.arr_dict['indices'].cpu().numpy()


def compare(data, options):
    before = {key: value.copy() for key,value in data.items()}
    expected = upstream_selection(data, options)
    filtered, summary = ml.filter_excited_state_training_arrays(
        data, TABLE, coordinates_key='coordinates', dist_soft_min=.5, **options)
    np.testing.assert_array_equal(filtered['indices'], expected)
    for key,value in before.items():
        np.testing.assert_array_equal(data[key], value)  # Source arrays are unchanged.
        np.testing.assert_array_equal(filtered[key], value[expected])
    return filtered, summary


@pytest.mark.parametrize('options', [
    {'remove_high_forces_cut':40., 'remove_high_energy_cut':4.},
    {'remove_high_forces_std':2., 'remove_high_energy_std':2.},
    {'remove_high_forces_cut':40., 'remove_high_forces_std':2.,
     'remove_high_energy_cut':4., 'remove_high_energy_std':2.},
    {'remove_high_forces_cut':None, 'remove_high_forces_std':None},
])
def test_fixed_standardized_combined_and_disabled_filters_match_upstream(options):
    data = arrays()
    data['F0'][-1] = 110.
    data['F0'][-2] = 20.
    data['sE0'][-3] = 8.
    compare(data, options)


@pytest.mark.parametrize('cut, expected', [(2., list(range(6))),
                                         (np.nextafter(2.,0.), list(range(2,6)))])
def test_nonzero_mean_and_exact_component_cutoff(cut, expected):
    data = arrays(6)
    data['F0'][0], data['F0'][1] = 8., 12.
    filtered, _ = compare(data, {'remove_high_forces_cut':cut})
    assert filtered['indices'].tolist() == expected


@pytest.mark.parametrize('options', [{'remove_high_forces_cut':0.}, {'remove_high_forces_std':0.},
                                    {'remove_high_forces_cut':0.,'remove_high_forces_std':2.}])
def test_zero_variance_large_forces_are_not_norm_filtered(options):
    data = arrays()
    # Every vector has a large norm, but every component equals the mean.
    data['F0'][:] = 1000.
    filtered, _ = compare(data, options)
    assert len(filtered['indices']) == 12


def test_std_statistics_are_recomputed_after_fixed_cut():
    data = arrays()
    data['F0'][-1], data['F0'][-2] = 110.,20.
    filtered, summary = compare(data, {'remove_high_forces_cut':40., 'remove_high_forces_std':2.})
    assert filtered['indices'].tolist() == list(range(10))
    assert summary['static_cut_failures']['F0'] == 1
    assert summary['standard_deviation_failures']['F0'] == 1


def test_h5_training_force_filter_keeps_original_store(tmp_path):
    path = tmp_path/'data-0000.h5'
    data = arrays()
    data['F0'][-1] = 110.
    data['F0'][-2] = 20.
    with h5py.File(path,'w') as handle:
        group = handle.create_group('H02')
        for key in ('coordinates','sE0','F0'):
            group[key] = data[key]
        group['species'] = np.array([b'H',b'H'])
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded, _ = ml.load_excited_state_h5_arrays(str(tmp_path), PROPERTIES)
    filtered, _ = ml.filter_excited_state_training_arrays(
        loaded, TABLE, coordinates_key='coordinates', dist_soft_min=.5,
        remove_high_forces_cut=40., remove_high_forces_std=2.)
    assert len(filtered['coordinates']) == 10
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def multistate_arrays():
    data = arrays(20)
    data['sE1'] = data['sE0'] + .5
    data['F1'] = data['F0'].copy()
    data['dE01'] = data['sE1'] - data['sE0']
    properties = {**PROPERTIES, 'sE1': ['sE1', 'system', 1.],
                  'F1': ['F1', 'atomic', 1.], 'dE01': ['dE01', 'system', 1.]}
    return data, ml.validate_excited_state_training_properties(properties)


@pytest.mark.parametrize('options', [
    {'remove_high_forces_cut': 5., 'remove_high_energy_cut': 5.},
    {'remove_high_forces_std': 2., 'remove_high_energy_std': 2.},
    {'remove_high_forces_cut': 5., 'remove_high_forces_std': 2.,
     'remove_high_energy_cut': 5., 'remove_high_energy_std': 2.},
    {'remove_high_forces_cut': None, 'remove_high_forces_std': None},
], ids=['fixed', 'standardized', 'combined', 'disabled'])
def test_multistate_union_is_order_independent_and_preserves_all_rows(options):
    data, table = multistate_arrays()
    data['sE1'][16] = 20.
    data['F0'][17] = 30.
    data['F1'][18] = -10.
    data['dE01'] = data['sE1'] - data['sE0']
    before = {key: value.copy() for key, value in data.items()}
    forward, summary = ml.filter_excited_state_training_arrays(
        data, table, coordinates_key='coordinates', dist_soft_min=.5, **options)
    reverse, _ = ml.filter_excited_state_training_arrays(
        data, list(reversed(table)), coordinates_key='coordinates', dist_soft_min=.5, **options)
    expected = [i for i in range(20) if i not in (16, 17, 18)] if any(
        value is not None for value in options.values()) else list(range(20))
    assert forward['indices'].tolist() == expected
    for key, value in before.items():
        np.testing.assert_array_equal(data[key], value)
        np.testing.assert_array_equal(forward[key], reverse[key])
        np.testing.assert_array_equal(forward[key], value[expected])
    np.testing.assert_array_equal(forward['dE01'], forward['sE1'] - forward['sE0'])
    if len(expected) < len(data['indices']):
        stage = 'static_cut' if options.get('remove_high_forces_cut') is not None else 'standard_deviation'
        assert summary['per_state_failures']['0']['forces'][stage] == 1
        assert summary['per_state_failures']['1']['forces'][stage] == 1
        assert summary['per_state_failures']['1']['energy'][stage] == 1
        assert summary['removed_static_cut_union'] + summary['removed_standard_deviation_union'] == 3
    assert summary['final_structures'] == len(expected)


def test_minimum_distance_is_applied_before_multistate_statistics():
    data, table = multistate_arrays()
    data['coordinates'][0, 1, 0] = .4
    data['sE1'][0] = 1000.
    data['dE01'] = data['sE1'] - data['sE0']
    filtered, summary = ml.filter_excited_state_training_arrays(
        data, table, coordinates_key='coordinates', dist_soft_min=.5,
        remove_high_energy_std=2.)
    expected = list(range(1, 20))
    for key, value in data.items():
        np.testing.assert_array_equal(filtered[key], value[expected])
    assert summary['removed_min_distance'] == 1
    assert summary['removed_standard_deviation_union'] == 0


def test_multistate_zero_variance_retains_all_rows():
    data, table = multistate_arrays()
    filtered, summary = ml.filter_excited_state_training_arrays(
        data, table, coordinates_key='coordinates', dist_soft_min=.5,
        remove_high_energy_std=2., remove_high_forces_std=2.)
    for key, value in data.items():
        np.testing.assert_array_equal(filtered[key], value)
    assert summary['final_structures'] == len(data['indices'])
