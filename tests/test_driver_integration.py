"""Execute ALF's real entry point with asynchronous, local-only mock tasks."""
import json
from pathlib import Path
import runpy
import sys
import time

import h5py
import numpy as np
import parsl
from parsl import python_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
import pytest
from ase import Atoms

from alframework.tools import tools as tool_module
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampling_validation import validate_sampler_debug_output


class DriverFinished(Exception):
    pass


def stored_ids(pattern):
    ids = []
    for path in sorted(Path(pattern).parent.glob('data-*.h5')):
        with h5py.File(path) as handle:
            for group in handle.values():
                ids.extend(value.decode() for value in group['_id'][()])
    return ids


@pytest.mark.parametrize('batched', [False, True])
def test_actual_driver_bootstrap_training_sampling_qm_and_retraining(tmp_path, monkeypatch, batched):
    # No scheduler, GPU, or production files: four local Parsl thread executors.
    try:
        parsl.dfk().cleanup()
    except parsl.errors.NoDataFlowKernelError:
        pass
    parsl.clear()
    monkeypatch.chdir(tmp_path)
    training_calls, sampler_calls = [], []
    model_pattern = str(tmp_path/'models/model-{:04d}')
    h5_pattern = str(tmp_path/'h5/data-{:04d}.h5')

    @python_app(executors=['alf_builder_executor'])
    def builder(moleculeid):
        mol = MoleculesObject(Atoms('OH2', positions=[[0,0,0],[.96,0,0],[-.24,.93,0]]), moleculeid)
        # Bootstrap accepts the legacy object. Active learning tests nested lists.
        return [[mol]] if batched and 'boot' not in moleculeid else mol

    def sample_one(molecule, current_model_id):
        assert (Path(model_pattern.format(current_model_id))/'complete').is_file()
        sampler_calls.append((current_model_id, molecule.get_moleculeid()))
        molecule.update_metadata({'sampled': True})
        return molecule

    @python_app(executors=['alf_sampler_executor'])
    def legacy(molecule_object, current_model_id):
        return sample_one(molecule_object, current_model_id)

    @python_app(executors=['alf_sampler_executor'])
    def batch(molecule_objects, current_model_id):
        assert len(molecule_objects) == 2
        rows = [sample_one(molecule, current_model_id) for molecule in molecule_objects]
        return [rows[0], [rows[1]]]

    @python_app(executors=['alf_QM_executor'])
    def qm(molecule_object):
        if 'boot' not in molecule_object.get_moleculeid():
            assert molecule_object.get_metadata()['sampled']
        molecule_object.store_results({'energy': 1., 'forces': np.arange(9).reshape(3,3)})
        molecule_object.set_converged_flag(True)
        return [[molecule_object]]  # Generic nested QM result extraction.

    @python_app(executors=['alf_ML_executor'])
    def ml(current_training_id):
        ids = stored_ids(h5_pattern)
        training_calls.append((current_training_id, ids))
        target = Path(model_pattern.format(current_training_id))
        target.mkdir(parents=True, exist_ok=False)
        (target/'complete').write_text('four mock members completed\n')
        return [True]*4, current_training_id

    config = Config(executors=[ThreadPoolExecutor(label=f'alf_{name}_executor', max_threads=2)
                              for name in ('builder','sampler','QM','ML')],
                    strategy='none', run_dir=str(tmp_path/'runinfo'))
    modules = {'mock.config': config, 'mock.builder': builder, 'mock.sampler': batch if batched else legacy,
               'mock.qm': qm, 'mock.ml': ml}
    monkeypatch.setattr(tool_module, 'load_module_from_string', modules.__getitem__)
    master = dict(master_directory='pwd', builder_task='mock.builder', sampler_task='mock.sampler',
                  QM_task='mock.qm', ML_task='mock.ml', parsl_configuration='mock.config',
                  builder_config_path='builder.json', sampler_config_path='sampler.json',
                  QM_config_path='qm.json', ML_config_path='ml.json', status_path='status.json',
                  model_path=model_pattern, h5_path=h5_pattern, bootstrap_set=2,
                  properties_list={'energy':['energy','system',1.], 'forces':['forces','atomic',1.]},
                  parallel_samplers=2, target_queued_QM=2, minimum_QM=0, save_h5_threshold=2,
                  maximum_completed_QM=4, gpus_per_node=0)
    sampler_config = {'model_mode':'ground_state'}
    if batched:
        sampler_config['alchemi_baoab'] = {'batch_size':2, 'partial_policy':'full_only'}
    for name, values in [('master',master), ('builder',{}), ('sampler',sampler_config), ('qm',{}), ('ml',{})]:
        (tmp_path/f'{name}.json').write_text(json.dumps(values))
    monkeypatch.setattr(sys, 'argv', ['alframework','--master',str(tmp_path/'master.json')])
    original_sleep = time.sleep
    ticks = []

    def local_sleep(seconds):
        if seconds == 60:
            ticks.append(seconds)
            status = json.loads((tmp_path/'status.json').read_text())
            capacity = status['sampler_capacity']
            assert capacity['submitted_sampler_replicas'] <= 2
            if status['current_model_id'] >= 1:
                raise DriverFinished
            assert len(ticks) < 500, 'Driver failed to complete bootstrap and retraining'
            original_sleep(.005)
        else:
            original_sleep(seconds)

    monkeypatch.setattr(time, 'sleep', local_sleep)
    try:
        with pytest.raises(DriverFinished):
            runpy.run_module('alframework', run_name='__main__')
    finally:
        try:
            parsl.dfk().cleanup()
        finally:
            parsl.clear()
    assert training_calls[0][0] == 0
    assert len(training_calls[0][1]) >= 2
    assert all('mol-boot-' in value for value in training_calls[0][1])
    assert any(index == 1 and any('mol-0000-' in value for value in ids)
               for index, ids in training_calls)
    assert sampler_calls and sampler_calls[0][0] == 0
    assert (tmp_path/'h5/data-0000.h5').exists() and (tmp_path/'h5/data-0001.h5').exists()
    status = json.loads((tmp_path/'status.json').read_text())
    assert all(status[f'lifetime_failed_{name}_tasks'] == 0 for name in ('builder','sampler','QM','ML'))
    with h5py.File(tmp_path/'h5/data-0001.h5') as handle:
        group = handle['H02_O01']
        assert group['species'][()].tolist() == [b'H',b'H',b'O']
        assert 'topology_atom_ids' not in group


@pytest.mark.parametrize('metadata,passes', [
    ({'sampler_device':'cuda:0','selected_state':1,'topology_valid':True}, True),
    ({'sampler_device':'cpu','selected_state':1,'topology_valid':True}, False),
    ({'sampler_device':'cuda:0','selected_state':0,'topology_valid':True}, False),
    ({'sampler_device':'cuda:0','selected_state':1,'topology_valid':False}, False),
])
def test_sampler_owned_debug_validation(metadata, passes):
    mol = MoleculesObject(Atoms('H'), 'debug')
    mol.update_metadata(metadata)
    cfg = {'alchemi_baoab': {'strict_gpu':True}, 'model_mode':'excited_state',
           'state_selection':{'mode':'fixed','state':1}, 'topology_check':{'enabled':True}}
    if passes:
        validate_sampler_debug_output(mol, cfg)
    else:
        with pytest.raises(AssertionError):
            validate_sampler_debug_output(mol, cfg)


def test_legacy_debug_has_no_excited_state_or_gpu_requirements():
    validate_sampler_debug_output(MoleculesObject(Atoms('H'), 'legacy'), {})
