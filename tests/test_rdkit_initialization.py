"""Small, self-contained regressions for reference-centred KDG/FPS seeds."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
from ase.data import atomic_numbers

from alframework.builders.h5_replay_builder import build_h5_replay_structures
from alframework.tools import rdkit_initialization as seed


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'examples/rdkit_initialization'


def example_config(name):
    return json.loads((EXAMPLES / f'{name}.json').read_text())


def test_quotas_descriptor_and_fps_are_deterministic():
    assert seed.ladder_quotas(50000, 6) == [8334, 8334, 8333, 8333, 8333, 8333]
    assert seed.ladder_quotas(5000, 6) == [834, 834, 833, 833, 833, 833]
    assert [seed.stratum_seed(42, level, 0) for level in range(6)] == [
        438055385, 1262664333, 1985621771, 1596702514, 702239142, 649246379]
    features = np.arange(4, dtype=float)[:, None]
    np.testing.assert_array_equal(seed.farthest_point_order(features, 4), [1, 3, 0, 2])
    np.testing.assert_array_equal(seed.farthest_point_order(np.zeros((4, 2)), 4), np.arange(4))
    coordinates = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]]])
    numbers = np.array([8, 1, 1])
    descriptor = seed.pair_descriptor(coordinates, numbers)
    np.testing.assert_allclose(descriptor, [[np.sqrt(5), 1., 2.]])
    order = [2, 0, 1]
    np.testing.assert_array_equal(seed.pair_descriptor(coordinates[:, order] + 10., numbers[order]), descriptor)
    with pytest.raises(ValueError, match='FPS'):
        seed.farthest_point_order(features, 5)


@pytest.mark.parametrize('override', [
    {'seed_size': 50001}, {'threads': True}, {'bond_scales': [0., float('nan')]},
    {'angle_scales': [0.]}, {'geometry_screen': {'typo': 1.}},
])
def test_invalid_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        seed.resolve_config({**example_config('keto'), **override}, base_directory=EXAMPLES)


def test_seed_screen_boundaries_and_rejection_priority():
    context = {'numbers': np.array([1, 1]), 'bonds': np.array([[0, 1]]),
               'bond_centers': np.array([1.]), 'pairs': np.array([[0, 1]]),
               'nonbonded': np.empty((0, 2), dtype=int)}
    screen = seed.DEFAULT_SETTINGS['geometry_screen']
    for distance in (.85, 1., 1.50):
        assert seed.geometry_failure([[0., 0., 0.], [distance, 0., 0.]], context, screen) is None
    assert seed.geometry_failure([[0., 0., 0.], [.849, 0., 0.]], context, screen) == 'bond_window'
    assert seed.geometry_failure([[0., 0., 0.], [1.501, 0., 0.]], context, screen) == 'bond_window'
    assert seed.geometry_failure(np.full((2, 3), np.nan), context, screen) == 'non_finite'
    loose = {**screen, 'bond_min_scale': .1}
    assert seed.geometry_failure([[0., 0., 0.], [.69, 0., 0.]], context, loose) == 'min_distance'
    context.update(numbers=np.array([1, 1, 1]), pairs=np.array([[0, 1], [0, 2], [1, 2]]),
                   nonbonded=np.array([[0, 2], [1, 2]]))
    assert seed.geometry_failure([[0., 0., 0.], [1., 0., 0.], [0., .1, 0.]], context, screen) == 'inferred_extra_bond'
    assert seed.geometry_failure([[0., 0., 0.], [1., 0., 0.], [0., .82, 0.]], context, screen) == 'nonbonded_close'


def test_reference_centers_and_polar_hydrogen_widening(tmp_path):
    pytest.importorskip('rdkit')
    reference = tmp_path / 'water.xyz'
    reference.write_text('3\nwater\nO 0 0 0\nH .96 0 0\nH -.24 .93 0\n')
    config = seed.resolve_config({'smiles': 'O', 'reference_geometry': str(reference)})
    context = seed._reference_context(config)
    polar = seed.widened_bounds_matrix(context['molecule'], context['centers'], config, 5)
    ordinary = seed.widened_bounds_matrix(context['molecule'], context['centers'], {**config, 'polar_h': False}, 5)
    center = context['centers'][0, 1]
    assert polar[0, 1] == pytest.approx(center * 1.1)
    assert polar[1, 0] == pytest.approx(center * .9)
    assert polar[0, 1] > ordinary[0, 1]
    assert polar[1, 0] < ordinary[1, 0]


def test_matching_species_are_not_a_topology_mapping():
    pytest.importorskip('rdkit')
    config = {**example_config('keto'), 'reference_geometry': 'enol.xyz'}
    with pytest.raises(ValueError, match='connectivity|bond set'):
        seed._reference_context(seed.resolve_config(config, base_directory=EXAMPLES))


@pytest.mark.parametrize('name', ['enol', 'keto'])
def test_preparation_is_reproducible_and_replays_tagged_geometry_only(tmp_path, name):
    pytest.importorskip('rdkit')
    config = {**example_config(name), 'pool_size': 24, 'seed_size': 12}
    output = tmp_path / name
    report = seed.prepare_rdkit_seed(config, output, config_directory=EXAMPLES)
    repeated = tmp_path / (name + '-repeat')
    seed.prepare_rdkit_seed(config, repeated, config_directory=EXAMPLES)
    assert report['pool_count'] == 24 and report['selected_count'] == 12
    assert report['embedding_settings']['useBasicKnowledge'] is True
    assert report['embedding_settings']['useExpTorsionAnglePrefs'] is False
    assert report['embedding_settings']['force_field_optimization'] is False
    for file, digest in report['artifacts_sha256'].items():
        assert hashlib.sha256((output / file).read_bytes()).hexdigest() == digest
    with h5py.File(output / 'pool.h5') as pool, h5py.File(output / 'seed.h5') as selected, h5py.File(repeated / 'seed.h5') as again:
        p, s, r = pool['molecule'], selected['molecule'], again['molecule']
        indices = s['pool_index'][()]
        assert len(np.unique(indices)) == 12
        species = s['species'][()].astype('U2')
        assert species.ndim == 1
        assert np.all(np.diff([atomic_numbers[value] for value in species]) >= 0)
        np.testing.assert_array_equal(np.bincount(s['strata'][()][:, 0]), [2] * 6)
        np.testing.assert_array_equal(s['fps_rank'][()], [0, 1] * 6)
        np.testing.assert_array_equal(s['coordinates'][()], p['coordinates'][()][indices])
        np.testing.assert_array_equal(s['strata'][()], p['strata'][()][indices])
        np.testing.assert_array_equal(s['fps_rank'][()], p['fps_rank'][()][indices])
        for key in s:
            np.testing.assert_array_equal(s[key][()], r[key][()])
        assert not any(key.startswith(('sE', 'F', 'dE')) for key in s)
        reorder = np.argsort(s['topology_atom_ids'][()])
        canonical = s['coordinates'][()][:, reorder]
    molecules = build_h5_replay_structures(
        moleculeids=[f'mol-boot-{index:010d}' for index in range(14)],
        builder_config=json.loads((output / 'builder_config.json').read_text()),
        sampler_config=report['alf_configuration']['sampler'],
        h5_path=str(tmp_path / 'labels/data-{:04d}.h5'), current_h5_id=0)
    assert len(molecules) == 12
    for index, molecule in enumerate(molecules):
        assert molecule.get_results() == {}
        assert molecule.get_metadata()['replay_global_frame_index'] == index
        np.testing.assert_array_equal(molecule.get_atoms().positions, canonical[index])
        np.testing.assert_array_equal(molecule.get_atoms().arrays['topology_atom_ids'], np.arange(len(reorder)))


def test_exhausted_embedding_and_existing_outputs_fail_without_overwrite(tmp_path, monkeypatch):
    pytest.importorskip('rdkit')
    from rdkit.Chem import AllChem
    config = {**example_config('keto'), 'pool_size': 6, 'seed_size': 6}
    output = tmp_path / 'failed'
    monkeypatch.setattr(AllChem, 'EmbedMultipleConfs', lambda *args, **kwargs: [])
    with pytest.raises(RuntimeError, match='quota'):
        seed.prepare_rdkit_seed(config, output, config_directory=EXAMPLES)
    assert not output.exists()
    output.mkdir()
    marker = output / 'keep'
    marker.write_text('unchanged')
    with pytest.raises(FileExistsError):
        seed.prepare_rdkit_seed(config, output, config_directory=EXAMPLES)
    assert marker.read_text() == 'unchanged'


def test_cli_resolves_reference_relative_to_config(tmp_path):
    pytest.importorskip('rdkit')
    config_dir = tmp_path / 'config'
    config_dir.mkdir()
    config = {**example_config('enol'), 'pool_size': 12, 'seed_size': 6}
    (config_dir / 'enol.xyz').write_bytes((EXAMPLES / 'enol.xyz').read_bytes())
    path = config_dir / 'seed.json'
    path.write_text(json.dumps(config))
    output = tmp_path / 'prepared'
    result = subprocess.run([sys.executable, '-m', 'alframework.tools.rdkit_initialization',
                             '--config', str(path), '--output-dir', str(output)], cwd=tmp_path,
                            env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1'),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / 'provenance.json').read_text())
    assert report['selected_count'] == 6
    assert report['settings']['reference_geometry'] == str(config_dir / 'enol.xyz')
