"""Actual driver coverage for timed mixed-size periodic batches and debug feeds."""

import json
from pathlib import Path
import runpy
import sqlite3
import sys
import threading
import time

import h5py
import numpy as np
import parsl
from parsl import python_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
import pytest
from ase import Atoms
from ase.io import write

from alframework.builders.builders import simple_cfg_loader_task
from alframework.tools import tools as tool_module


class DriverFinished(Exception):
    pass


@pytest.mark.parametrize('debug', [False, True], ids=['timeout-and-empty-result', 'single-debug-input'])
def test_actual_portable_driver_mixed_periodic_partial_batch(tmp_path, monkeypatch, debug):
    try:
        parsl.dfk().cleanup()
    except parsl.errors.NoDataFlowKernelError:
        pass
    parsl.clear()
    monkeypatch.chdir(tmp_path)
    for index, atoms in enumerate((Atoms('H2', positions=[[0, 0, 0], [0, 0, .74]]),
                                   Atoms('OH2', positions=[[0, 0, 0], [.96, 0, 0], [-.24, .93, 0]]))):
        directory = tmp_path / f'library{index}'
        directory.mkdir()
        atoms.set_cell([[7., 0, 0], [1., 8., 0], [0, 0, 9.]])
        atoms.set_pbc(True)
        write(directory / 'structure.cfg', atoms, format='cfg')
    model_pattern = str(tmp_path / 'models/model-{:04d}')
    h5_pattern = str(tmp_path / 'h5/data-{:04d}.h5')
    if debug:
        Path(model_pattern.format(0)).mkdir(parents=True)
    batches, trained_frames = [], []

    @python_app(executors=['alf_builder_executor'])
    def builder(moleculeids, builder_config):
        rows = [simple_cfg_loader_task.func(moleculeid,
                    {'molecule_library_dir': str(tmp_path / f'library{index % 2}'), 'pbc': True}, shake=0.)
                for index, moleculeid in enumerate(moleculeids)]
        return rows if debug or 'boot' in moleculeids[0] else [[rows[0]], rows[1:]]

    @python_app(executors=['alf_sampler_executor'])
    def sampler(molecule_objects, current_model_id):
        assert Path(model_pattern.format(current_model_id)).is_dir()
        assert len(molecule_objects) < 4
        assert all(molecule.get_atoms().pbc.all() for molecule in molecule_objects)
        batches.append([len(molecule.get_atoms()) for molecule in molecule_objects])
        if not debug and len(batches) == 1:
            return []
        for molecule in molecule_objects:
            molecule.update_metadata({'sampled': True})
        return molecule_objects

    @python_app(executors=['alf_QM_executor'])
    def qm(molecule_object):
        if 'boot' not in molecule_object.get_moleculeid():
            assert molecule_object.get_metadata()['sampled']
        molecule_object.store_results({'energy': 1., 'forces': np.zeros((len(molecule_object.get_atoms()), 3))})
        molecule_object.set_converged_flag(True)
        return [molecule_object]

    @python_app(executors=['alf_ML_executor'])
    def ml(current_training_id):
        frames = 0
        for path in (tmp_path / 'h5').glob('data-*.h5'):
            with h5py.File(path) as handle:
                frames += sum(len(group['coordinates']) for group in handle.values())
        trained_frames.append(frames)
        Path(model_pattern.format(current_training_id)).mkdir(parents=True)
        if current_training_id == 0:
            # Bootstrap uses the legacy QM threshold. Raise it before active
            # sampling so each timed width-two batch is below minimum_QM.
            master['minimum_QM'] = 3
            (tmp_path / 'master.json').write_text(json.dumps(master))
        return [True], current_training_id

    config = Config(executors=[ThreadPoolExecutor(label=f'alf_{role}_executor', max_threads=2)
                              for role in ('builder', 'sampler', 'QM', 'ML')],
                    strategy='none', run_dir=str(tmp_path / 'runinfo'))
    modules = {'mock.config': config, 'mock.builder': builder, 'mock.sampler': sampler,
               'mock.qm': qm, 'mock.ml': ml}
    monkeypatch.setattr(tool_module, 'load_module_from_string', modules.__getitem__)
    master = dict(master_directory='pwd', builder_task='mock.builder', sampler_task='mock.sampler',
                  QM_task='mock.qm', ML_task='mock.ml', parsl_configuration='mock.config',
                  builder_config_path='builder.json', sampler_config_path='sampler.json',
                  QM_config_path='qm.json', ML_config_path='ml.json', status_path='status.json',
                  model_path=model_pattern, h5_path=h5_pattern, bootstrap_set=2,
                  properties_list={'energy': ['energy', 'system', 1.], 'forces': ['forces', 'atomic', 1.]},
                  parallel_samplers=3, maximum_builder_structures=2, target_queued_QM=2,
                  minimum_QM=0, save_h5_threshold=2, maximum_completed_QM=4, gpus_per_node=0)
    sampler_config = {'model_mode': 'ground_state', 'batching': {'batch_size': 4, 'max_wait_seconds': .001},
                      'dynamics': {'backend': 'torchsim', 'strict_gpu': False, 'allow_cpu_debug': True}}
    for name, value in [('master', master), ('sampler', sampler_config), ('builder', {}), ('qm', {}), ('ml', {})]:
        (tmp_path / f'{name}.json').write_text(json.dumps(value))
    argv = ['alframework', '--master', str(tmp_path / 'master.json')]
    if debug:
        argv.append('--test_sampler')
    monkeypatch.setattr(sys, 'argv', argv)
    original_sleep = time.sleep
    ticks = []

    def local_sleep(seconds):
        if threading.current_thread() is not threading.main_thread():
            return original_sleep(seconds)
        ticks.append(seconds)
        status = json.loads((tmp_path / 'status.json').read_text())
        assert status['lifetime_failed_sampler_tasks'] == 0
        assert status['sampler_capacity']['accounted_replicas'] <= 3
        assert status['sampler_batching']['backend'] == 'torchsim'
        assert status['sampler_batching']['partial_policy'] == 'timeout'
        assert status['sampler_batching']['max_wait_seconds'] == .001
        if status['current_model_id'] >= 1:
            raise DriverFinished
        assert len(ticks) < 1000, 'Timed partial batches failed to release capacity and retrain'
        original_sleep(min(seconds, .005))

    monkeypatch.setattr(time, 'sleep', local_sleep)
    try:
        if debug:
            with pytest.raises(SystemExit):
                runpy.run_module('alframework', run_name='__main__')
        else:
            with pytest.raises(DriverFinished):
                runpy.run_module('alframework', run_name='__main__')
    finally:
        parsl.dfk().cleanup()
        parsl.clear()
    if debug:
        assert batches == [[2]]
        assert trained_frames == []
    else:
        assert len(batches) >= 2
        assert all(batch == [2, 3] for batch in batches)
        assert master['minimum_QM'] == 3
        assert trained_frames[0] >= 2
        assert trained_frames[-1] >= trained_frames[0] + 2
        assert any(delay < 60 for delay in ticks)


def test_controlled_partial_batch_resume_restores_actual_input_width(tmp_path):
    from alframework.tools.staged_workflow import prepare_workflow, WorkflowRunner
    from alframework.tools.workflow_io import atomic_json, read_json
    from tests.test_staged_workflow import fixture_workflow

    path = fixture_workflow(tmp_path, excited=True, stages=1)
    sampler = path.parent / 'sampler0.json'
    config = read_json(sampler)
    config.pop('alchemi_baoab')
    config.update(batching={'batch_size': 4, 'max_wait_seconds': .001},
                  dynamics={'backend': 'torchsim'}, fail_first_sampler=True)
    atomic_json(sampler, config)
    root = tmp_path / 'run'
    prepare_workflow(path, root)
    with pytest.raises(RuntimeError, match='Stage alf0 failed'):
        WorkflowRunner(root).run()
    stage = root / 'stages/00-alf0'
    with sqlite3.connect(stage / 'control/journal.sqlite') as db:
        seed_attempts = dict(db.execute("SELECT key,attempts FROM tasks WHERE role='QM' AND phase='seed'"))
        assert db.execute("SELECT COUNT(*) FROM tasks WHERE role='sampler' AND state='failed'").fetchone()[0] > 0
    assert WorkflowRunner(root).run() == 0
    with sqlite3.connect(stage / 'control/journal.sqlite') as db:
        assert dict(db.execute("SELECT key,attempts FROM tasks WHERE role='QM' AND phase='seed'")) == seed_attempts
        assert db.execute("SELECT COUNT(*) FROM attempts a JOIN tasks t ON a.task_key=t.key WHERE t.role='sampler' AND a.number>1 AND a.state='succeeded'").fetchone()[0] > 0
        import pickle
        widths = [len(pickle.loads(row[0])['molecule_objects'])
                  for row in db.execute("SELECT input FROM tasks WHERE role='sampler'")]
        assert widths and all(0 < width < 4 for width in widths)
    assert read_json(root / 'workflow_status.json')['final_training']['input_frames'] == 3
