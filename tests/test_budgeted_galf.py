from pathlib import Path

import h5py
import numpy as np
import pytest

from alframework.tools.budgeted_galf import (
    copy_selected_shard, fingerprint, freeze_selection, load_selection,
    read_json, select_frames, update_budget_checkpoints, verify_handoff_inputs, verify_initial_ensemble, write_json,
)


def shard(path, n, generation=0, start=0, permutation=None, bad=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    permutation = np.array([0, 1, 2] if permutation is None else permutation)
    xyz = np.arange(n * 9, dtype=float).reshape(n, 3, 3) + start
    with h5py.File(path, 'w') as h:
        h.attrs['units'] = 'eV'
        g = h.create_group('molecule')
        g.attrs['provenance'] = 'fixture'
        g['coordinates'] = xyz[:, permutation]
        g['species'] = np.array([b'C', b'O', b'H'])[permutation]
        g['topology_atom_ids'] = np.tile(permutation, (n, 1))
        g['_id'] = np.array([f'mol-{generation:04d}-{start+i:010d}'.encode() if generation >= 0
                             else f'mol-boot-{start+i:010d}'.encode() for i in range(n)])
        for s in range(4):
            g[f'sE{s}'] = np.arange(n, dtype=float) + s
            g[f'F{s}'] = np.full((n, 3, 3), s, dtype=float)
        g['F3'].attrs['units'] = 'eV/Angstrom'
        if bad:
            g[bad[0]][bad[1]] = np.nan
    return xyz


@pytest.fixture
def run(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    write_json(run / 'master_config.json', {'h5_path': 'h5/data-{:04d}.h5', 'status_path': 'status.json',
                                          'model_path': 'models/model-{:04d}', 'ML_config_path': 'ml.json'})
    write_json(run / 'ml.json', {'n_models': 4})
    return run


def test_seed_finite_partial_closed_order_and_reordering(run):
    seed = shard(run / 'h5/data-0000.h5', 3, -1)
    original = shard(run / 'h5/data-0001.h5', 5, 2, 100, permutation=[2, 0, 1], bad=('F3', 1))
    shard(run / 'h5/data-0002.h5', 4, 99, 200)
    write_json(run / 'status.json', {'current_h5_id': 2})
    a, m = select_frames(run, 6)
    assert a['ids'].shape == (6,)
    assert np.array_equal(a['coordinates'][:3], seed)
    assert np.array_equal(a['coordinates'][3:], original[[0, 2, 3]])
    assert a['generations'].tolist() == [-1, -1, -1, 2, 2, 2]
    assert a['rows'].tolist() == [0, 1, 2, 0, 2, 3]
    assert m['sources'][-1]['rejected_nonfinite'] == 1
    with pytest.raises(ValueError, match='needs 8 finite frames, has 7'):
        select_frames(run, 8)
    target = run / 'copied.h5'
    copy_selected_shard(m['sources'][-1], target)
    with h5py.File(target) as h:
        assert h.attrs['units'] == 'eV'
        assert h['molecule'].attrs['provenance'] == 'fixture'
        assert h['molecule/F3'].attrs['units'] == 'eV/Angstrom'
        assert h['molecule/F3'].shape == (3, 3, 3)
        assert h['molecule/species'][:].tolist() == [b'H', b'C', b'O']
        assert np.array_equal(h['molecule/coordinates'][:], original[[0, 2, 3]][:, [2, 0, 1]])


def test_freeze_reload_fingerprint_and_nested_budget_with_reset_ids(run):
    shard(run / 'h5/data-0000.h5', 3, -1)
    shard(run / 'h5/data-0001.h5', 2, 0)
    # Identical IDs in an independent branch must remain distinct retained rows.
    shard(run / 'h5/data-0002.h5', 4, 0)
    master = read_json(run / 'master_config.json')
    master['accepted_frame_budget'] = {'handoff_frames': 5, 'target_frames': 8, 'checkpoints': [6, 8]}
    write_json(run / 'master_config.json', master)
    assert not update_budget_checkpoints(run, {'current_h5_id': 2})
    assert update_budget_checkpoints(run, {'current_h5_id': 3})
    assert read_json(run / 'budget_progress.json')['overshoot_frames'] == 1
    a = load_selection(run / 'budget_checkpoints/6')
    b = load_selection(run / 'budget_checkpoints/8')
    assert np.array_equal(a['ids'], b['ids'][:6])
    assert a['phases'].tolist() == ['seed'] * 3 + ['alf'] * 2 + ['galf']
    assert len(np.unique(b['ids'])) < len(b['ids'])
    freeze_selection(run, 6, run / 'budget_checkpoints/6', closed_before=3, handoff_frames=5)
    with h5py.File(run / 'h5/data-0002.h5', 'r+') as h:
        h['molecule/sE0'][0] = 8
    with pytest.raises(RuntimeError, match='Frozen source changed'):
        load_selection(run / 'budget_checkpoints/6')


def test_invalid_topology_and_cache_corruption(run):
    shard(run / 'h5/data-0000.h5', 3, -1)
    freeze_selection(run, 2, run / 'snapshot')
    with (run / 'snapshot/arrays.npz').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(RuntimeError, match='Frozen cache changed'):
        load_selection(run / 'snapshot')
    with h5py.File(run / 'h5/data-0000.h5', 'r+') as h:
        h['molecule/topology_atom_ids'][0] = [0, 0, 2]
    with pytest.raises(ValueError, match='topology-ID permutation'):
        select_frames(run, 2)


def test_initial_ensemble_gate_requires_all_fresh_members_and_exact_pool(run):
    shard(run / 'h5/data-0000.h5', 5, -1)
    write_json(run / 'budgeted_galf_provenance.json', {'handoff_frames': 5,
               'copied_h5': [fingerprint(run / 'h5/data-0000.h5')]})
    for i in range(4):
        member = run / f'models/model-0000/model-{i:02d}'
        member.mkdir(parents=True)
        (member / 'training_log.txt').write_text('Training complete')
        write_json(member / 'training_summary.json', {'dataset_frames': 5, 'split_frames': {'train': 3, 'valid': 1, 'test': 1},
                   'gap_target_mode': 'disabled', 'train_forces': True})
        for name in ('best_model.pt', 'best_checkpoint.pt', 'experiment_structure.pt'):
            (member / name).write_bytes(b'fixture')
    member = run / 'models/model-0000/model-03'
    (member / 'training_log.txt').write_text('Interrupted')
    with pytest.raises(RuntimeError, match='did not complete'):
        verify_initial_ensemble(run)
    (member / 'training_log.txt').write_text('Training complete')
    summary = read_json(member / 'training_summary.json')
    summary['dataset_frames'] = 6
    write_json(member / 'training_summary.json', summary)
    with pytest.raises(RuntimeError, match='exact handoff pool'):
        verify_initial_ensemble(run)
    summary['dataset_frames'] = 5
    write_json(member / 'training_summary.json', summary)
    report = verify_initial_ensemble(run)
    assert len(report['members']) == 4
    assert verify_initial_ensemble(run) == report
    (member / 'best_model.pt').write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='Frozen source changed'):
        verify_initial_ensemble(run)


def test_handoff_gate_rejects_nonstandard_extra_training_shards(run):
    shard(run / 'h5/data-0000.h5', 5, -1)
    write_json(run / 'budgeted_galf_provenance.json', {'handoff_frames': 5,
               'copied_h5': [fingerprint(run / 'h5/data-0000.h5')]})
    verify_handoff_inputs(run)
    shard(run / 'h5/unexpected-training-data.h5', 3)
    with pytest.raises(RuntimeError, match='only the frozen handoff shards'):
        verify_handoff_inputs(run)


def test_cpu_fresh_four_member_training_records_pool_and_passes_gate(run):
    """Exercise real HIP-HOP E/F training and the startup gate on a tiny pool."""
    from alframework.ml_interfaces.excited_state_hippynn_interface import train_excited_state_ensemble
    import torch
    torch.set_num_threads(1)
    shard(run / 'h5/data-0000.h5', 24, -1)
    rng = np.random.default_rng(42)
    with h5py.File(run / 'h5/data-0000.h5', 'r+') as h:
        g = h['molecule']
        for key in ('coordinates', 'species', 'topology_atom_ids', 'F0', 'F1', 'F2', 'F3'):
            del g[key]
        base = np.array([[x, y, z] for x in range(3) for y in range(3) for z in range(2)])[:15] * 1.1
        g['coordinates'] = base[None] + rng.normal(0, .03, (24, 15, 3))
        g['species'] = np.array([b'O'] * 2 + [b'C'] * 5 + [b'H'] * 8)
        g['topology_atom_ids'] = np.tile(np.arange(15), (24, 1))
        for state in range(4):
            g[f'sE{state}'][:] = state + rng.normal(0, .01, 24)
            g[f'F{state}'] = rng.normal(0, .01, (24, 15, 3))
    write_json(run / 'budgeted_galf_provenance.json', {'handoff_frames': 24,
               'copied_h5': [fingerprint(run / 'h5/data-0000.h5')]})
    config = {'n_models': 4, 'coordinates_key': 'coordinates', 'species_key': 'species', 'cell_key': None,
              'train_forces': True, 'network_choice': 1, 'n_atoms': 15, 'device_string': 'cpu',
              'network_params': {'possible_species': [0, 1, 6, 8], 'n_features': 4, 'n_sensitivities': 4,
                                 'n_interaction_layers': 1, 'n_atom_layers': 1, 'l_max': 1, 'n_max': 2,
                                 'dist_soft_min': .7, 'dist_soft_max': 3., 'dist_hard_max': 3.5},
              'gap_targets': {'enabled': False}, 'valid_size': .1, 'test_size': .1,
              'max_epochs': 1, 'batch_size': 8, 'eval_batch_size': 8,
              'exports': {'pdf': False, 'png': False, 'csv': False}}
    write_json(run / 'ml.json', config)
    properties = {key: [key, 'system' if key.startswith('sE') else 'atomic', 1.]
                  for state in range(4) for key in (f'sE{state}', f'F{state}')}
    complete, index = train_excited_state_ensemble(ML_config=config, h5_dir=str(run / 'h5'),
                        model_path=str(run / 'models/model-{:04d}'), current_training_id=0,
                        gpus_per_node=0, properties_list=properties)
    assert complete == [True] * 4 and index == 0
    report = verify_initial_ensemble(run)
    assert all(member['dataset_frames'] == 24 for member in report['members'])
    assert all(sum(member['split_frames'].values()) == 24 for member in report['members'])


def test_master_trains_before_sampling_then_stops_and_freezes_nested_checkpoints(run, monkeypatch):
    """Run the actual master with local async tasks and small retained budgets."""
    import runpy
    import sys
    import time
    import parsl
    from parsl import python_app
    from parsl.config import Config
    from parsl.executors import ThreadPoolExecutor
    from ase import Atoms
    from alframework.tools import tools as tool_module
    from alframework.tools.molecules_class import MoleculesObject
    from alframework.tools.molecular_topology import topology_atom_id_array

    shard(run / 'h5/data-0000.h5', 5, -1)
    write_json(run / 'budgeted_galf_provenance.json', {'handoff_frames': 5,
               'copied_h5': [fingerprint(run / 'h5/data-0000.h5')]})
    properties = {key: [key, 'system' if key.startswith('sE') else 'atomic', 1.]
                  for state in range(4) for key in (f'sE{state}', f'F{state}')}
    training_calls = []

    @python_app(executors=['alf_builder_executor'])
    def builder(moleculeid):
        assert (run / 'initial_ensemble_verified.json').exists()
        atoms = Atoms('COH', positions=[[0, 0, 0], [1.2, 0, 0], [1.8, .9, 0]])
        atoms.set_array(topology_atom_id_array({}), np.arange(3))
        return MoleculesObject(atoms, moleculeid)

    @python_app(executors=['alf_sampler_executor'])
    def sampler(molecule_object):
        return molecule_object

    @python_app(executors=['alf_QM_executor'])
    def qm(molecule_object):
        molecule_object.store_results({key: float(state) if key.startswith('sE') else np.zeros((3, 3))
                                      for state in range(4) for key in (f'sE{state}', f'F{state}')})
        molecule_object.set_converged_flag(True)
        return molecule_object

    @python_app(executors=['alf_ML_executor'])
    def ml(current_training_id, model_path):
        count = len(select_frames(run)[0]['ids'])
        training_calls.append((current_training_id, count))
        for i in range(4):
            member = Path(model_path.format(current_training_id)) / f'model-{i:02d}'
            member.mkdir(parents=True)
            (member / 'training_log.txt').write_text('Training complete')
            write_json(member / 'training_summary.json', {'dataset_frames': count, 'split_frames': {'train': count},
                       'gap_target_mode': 'disabled', 'train_forces': True})
            for name in ('best_model.pt', 'best_checkpoint.pt', 'experiment_structure.pt'):
                (member / name).write_bytes(b'fixture')
        return [True] * 4, current_training_id

    master = read_json(run / 'master_config.json')
    master.update(master_directory='pwd', builder_task='fake.builder', sampler_task='fake.sampler',
                  QM_task='fake.qm', ML_task='fake.ml', builder_config_path='builder.json',
                  sampler_config_path='sampler.json', QM_config_path='qm.json', parsl_configuration='fake.config',
                  properties_list=properties, parallel_samplers=1, target_queued_QM=1, minimum_QM=0,
                  save_h5_threshold=2, bootstrap_set=5, gpus_per_node=0,
                  accepted_frame_budget={'handoff_frames': 5, 'target_frames': 8, 'checkpoints': [6, 8]})
    write_json(run / 'master_config.json', master)
    for name in ('builder', 'sampler', 'qm'):
        write_json(run / f'{name}.json', {})
    config = Config(executors=[ThreadPoolExecutor(label=f'alf_{name}_executor', max_threads=2)
                              for name in ('builder', 'sampler', 'QM', 'ML')], strategy='none',
                    run_dir=str(run / 'runinfo'))
    modules = {'fake.config': config, 'fake.builder': builder, 'fake.sampler': sampler, 'fake.qm': qm, 'fake.ml': ml}
    monkeypatch.setattr(tool_module, 'load_module_from_string', modules.__getitem__)
    sleep = time.sleep
    ticks = []

    def short_sleep(seconds):
        if seconds == 60:
            ticks.append(seconds)
            assert len(ticks) < 200, 'Master did not stop at its accepted-frame budget'
            sleep(.01)
        else:
            sleep(seconds)

    monkeypatch.setattr(time, 'sleep', short_sleep)
    monkeypatch.chdir(run)
    monkeypatch.setattr(sys, 'argv', ['alframework', '--master', str(run / 'master_config.json')])
    try:
        runpy.run_module('alframework', run_name='__main__')
    finally:
        try:
            parsl.dfk().cleanup()
        except parsl.errors.NoDataFlowKernelError:
            pass
    assert training_calls[0] == (0, 5)
    assert read_json(run / 'budget_progress.json')['complete']
    a, b = [load_selection(run / 'budget_checkpoints' / str(n)) for n in (6, 8)]
    assert np.array_equal(a['ids'], b['ids'][:6])
    assert len(a['ids']) == 6 and len(b['ids']) == 8
