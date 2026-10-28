"""Execute ALF's real entry point with asynchronous, local-only mock tasks."""
import json
import importlib.util
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
from ase.io import read, write as write_atoms

from alframework.builders.builders import simple_cfg_loader_task
from alframework.builders.h5_replay_builder import build_h5_replay_structures
from alframework.tools.rdkit_initialization import prepare_rdkit_seed
from alframework.samplers.alchemi_sampling import run_alchemi_sampling
from alframework.tools.force_screen import SCHEMA, SOURCE
from tests.helpers.fakes import FakeModel, FakeRunner

from alframework.tools import tools as tool_module
from alframework.tools.molecules_class import MoleculesObject


class DriverFinished(Exception):
    pass


def stored_ids(pattern):
    ids = []
    for path in sorted(Path(pattern).parent.glob('data-*.h5')):
        with h5py.File(path) as handle:
            for group in handle.values():
                ids.extend(value.decode() for value in group['_id'][()])
    return ids


@pytest.mark.parametrize('batched,rdkit_seed', [(False, False), (True, False), (True, True)], ids=['legacy', 'cfg', 'rdkit'])
def test_actual_driver_bootstrap_training_sampling_qm_and_retraining(tmp_path, monkeypatch, batched, rdkit_seed):
    # No scheduler, GPU, or production files: four local Parsl thread executors.
    try:
        parsl.dfk().cleanup()
    except parsl.errors.NoDataFlowKernelError:
        pass
    parsl.clear()
    monkeypatch.chdir(tmp_path)
    training_calls, sampler_calls, bootstrap_labels = [], [], []
    galf_phase = False
    example = Path(__file__).resolve().parents[1] / "examples/keto_rdkit_alf_galf"
    model_pattern = str(tmp_path/'models/model-{:04d}')
    h5_pattern = str(tmp_path/'h5/data-{:04d}.h5')
    library = tmp_path / 'cfg_library'
    library.mkdir()
    reference = Atoms('OH2', positions=[[0,0,0],[.96,0,0],[-.24,.93,0]], cell=np.eye(3)*8)
    if rdkit_seed:
        reference = read(example / 'keto.xyz')
    write_atoms(library / 'water.cfg', reference, format='cfg')
    builder_config = {'molecule_library_dir': str(library), 'pbc': False}
    if rdkit_seed:
        pytest.importorskip('rdkit')
        reference_path = tmp_path / 'reference.xyz'
        write_atoms(reference_path, reference)
        seed_report = prepare_rdkit_seed({
            'smiles': 'CC(=O)CC(C)=O', 'reference_geometry': str(reference_path),
            'pool_size': 4, 'seed_size': 2,
            **{key: [0.] for key in ('bond_scales', 'angle_scales', 'xh_bond_scales', 'xh_angle_scales', 'torsion_scales')},
        }, tmp_path / 'rdkit_seed')
        builder_config = json.loads((tmp_path / 'rdkit_seed/builder_config.json').read_text())

    @python_app(executors=['alf_builder_executor'])
    def builder(moleculeid, moleculeids, builder_config, sampler_config, h5_path, current_h5_id, master_directory):
        if rdkit_seed:
            # Ordinary ALF may request more builders while seed QM is still
            # running. This finite test source has only two seed frames;
            # report exhaustion as an empty builder result, not a task failure.
            if moleculeid.startswith('mol-boot-') and int(moleculeid.rsplit('-', 1)[1]) >= 2:
                return []
            return build_h5_replay_structures(
                moleculeids=moleculeids, builder_config=builder_config,
                sampler_config=sampler_config, h5_path=h5_path,
                current_h5_id=current_h5_id, master_directory=master_directory)
        mol = simple_cfg_loader_task.func(moleculeid, builder_config, shake=0.) if batched else MoleculesObject(reference.copy(), moleculeid)
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
        assert all(mol.get_metadata()['selected_state'] == 0 for mol in molecule_objects)
        if rdkit_seed:
            assert all(not mol.get_metadata()['replay_external_bootstrap'] for mol in molecule_objects)
            assert all(Path(mol.get_metadata()['replay_source_path']).parent == Path(h5_pattern).parent for mol in molecule_objects)
            if not galf_phase:
                assert all(mol.get_metadata()['replay_source_path'] == h5_pattern.format(0) for mol in molecule_objects)
        else:
            assert all(mol.get_metadata()['molecule_library_file'] == str(library / 'water.cfg') for mol in molecule_objects)
        assert all(not mol.get_atoms().pbc.any() for mol in molecule_objects)
        rows = [sample_one(molecule, current_model_id) for molecule in molecule_objects]
        if galf_phase:
            diagnostic = {'Es': .002, 'Fs': 0., 'Fsrms': 0., 'Fsmax': 0., 'Fmeanmax': 1.,
                          'force_screen_schema': SCHEMA, 'force_screen_source': SOURCE,
                          **{key: value for gap in ('dE01','dE12','dE23')
                             for key,value in ((gap,.05),(gap+'_stdev',0.),(gap+'_model_count',4))}}
            outputs = run_alchemi_sampling(rows, sampler_config, FakeModel({i:[diagnostic,{**diagnostic,'Es':.004}] for i in range(2)}),
                properties_list=master['properties_list'], runner_factory=FakeRunner, master_directory=str(tmp_path))
            assert len(FakeRunner.last_instance.lcm_calls) == 2
            assert any(mol.get_metadata()['gap_seeking_switched'] for mol in outputs)
            return outputs
        return [rows[0], [rows[1]]]

    @python_app(executors=['alf_QM_executor'])
    def qm(molecule_object):
        if 'boot' not in molecule_object.get_moleculeid():
            assert molecule_object.get_metadata()['sampled']
        if 'boot' in molecule_object.get_moleculeid():
            bootstrap_labels.append(molecule_object.get_moleculeid())
        if rdkit_seed:
            # Both seed labels converge; reject the second on a nonselected state.
            excessive = molecule_object.get_moleculeid() == 'mol-boot-0000000001'
            results = {f'sE{state}': 1. + state*.05 for state in range(4)}
            results.update({f'F{state}': np.full((len(reference),3),20. if excessive and state==1 else 0.) for state in range(4)})
            results.update({f'dE{i}{i+1}':results[f'sE{i+1}']-results[f'sE{i}'] for i in range(3)})
            molecule_object.store_results(results)
        else:
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
        if rdkit_seed:
            for member in range(4):
                directory = target/f'model-{member:02d}'
                directory.mkdir()
                (directory/'training_summary.json').write_text(json.dumps({
                    'data': {'n_structures':len(ids),'n_structures_before_filtering':len(ids)},
                    'gap_targets': {'enabled':False}}))
                (directory/'training_log.txt').write_text('Training complete\n')
                for artifact in ('experiment_structure.pt','best_checkpoint.pt'):
                    (directory/artifact).write_bytes(b'mock checkpoint')
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
        sampler_config.update(model_mode='excited_state', state_selection={'mode': 'fixed', 'state': 0})
        sampler_config['alchemi_baoab'] = {'batch_size':2, 'partial_policy':'full_only'}
    if rdkit_seed:
        master['properties_list'] = {f'{prefix}{state}': [f'{prefix}{state}', scope, 1.]
                                     for state in range(4) for prefix, scope in [('sE', 'system'), ('F', 'atomic')]}
        master['properties_list'].update({f'dE{i}{i+1}':[f'dE{i}{i+1}','system',1.] for i in range(3)})
        sampler_config.update(seed_report['alf_configuration']['sampler'])
        sampler_config.update(max_force_cutoff=16., dataset_screening={
            'enabled': True, 'force': True, 'min_distance': True, 'topology': True})
    for name, values in [('master',master), ('builder',builder_config), ('sampler',sampler_config), ('qm',json.loads((example/'QM_config.json').read_text()) if rdkit_seed else {}), ('ml',{})]:
        (tmp_path/f'{name}.json').write_text(json.dumps(values))
    monkeypatch.setattr(sys, 'argv', ['alframework','--master',str(tmp_path/'master.json')])
    original_sleep = time.sleep
    ticks = []

    def local_sleep(seconds):
        if seconds == 60:
            ticks.append(seconds)
            status = json.loads((tmp_path/('galf/status.json' if galf_phase else 'status.json')).read_text())
            assert status['lifetime_failed_sampler_tasks'] == 0, 'Local sampler task failed; inspect Parsl log'
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
    if rdkit_seed:
        assert sorted(bootstrap_labels) == ['mol-boot-0000000000', 'mol-boot-0000000001']
        assert training_calls[0][1] == ['mol-boot-0000000000']
    else:
        assert len(training_calls[0][1]) >= 2
    assert all('mol-boot-' in value for value in training_calls[0][1])
    assert any(index == 1 and any('mol-0000-' in value for value in ids)
               for index, ids in training_calls)
    assert sampler_calls and sampler_calls[0][0] == 0
    assert (tmp_path/'h5/data-0000.h5').exists() and (tmp_path/'h5/data-0001.h5').exists()
    status = json.loads((tmp_path/'status.json').read_text())
    assert all(status[f'lifetime_failed_{name}_tasks'] == 0 for name in ('builder','sampler','QM','ML'))
    with h5py.File(tmp_path/'h5/data-0001.h5') as handle:
        group = next(iter(handle.values()))
        assert group['species'].asstr()[()].tolist() == sorted(reference.get_chemical_symbols(), key=lambda symbol: {'H':1,'C':6,'O':8}[symbol])
        assert ('topology_atom_ids' in group) is rdkit_seed

    if rdkit_seed:
        # Freeze a reduced exact handoff, then exercise the real driver again.
        spec = importlib.util.spec_from_file_location('keto_driver_example', example/'prepare.py')
        workflow = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workflow)
        galf = tmp_path/'galf'
        master['QM_task']='alframework.qm_interfaces.pyseqm_interface.pyseqm_excited_state_task'
        workflow.write_json(tmp_path/'master.json',master)
        manifest = workflow.freeze(tmp_path/'master.json',galf,2,handoff=True)
        assert manifest['phase_counts'] == {'seed':1,'alf':1,'galf':0}
        master = workflow.read_json(galf/'master_config.json')
        model_pattern, h5_pattern = master['model_path'], master['h5_path']
        sampler_config = workflow.read_json(galf/'sampler_galf.json')
        sampler_config.update(dt=.1,maxt=.0002,Ncheck=1,min_time=0.,return_top_k=2)
        sampler_config['alchemi_baoab']['batch_size']=2
        sampler_config['state_selection']={'mode':'fixed','state':0}
        master.update(builder_task='mock.builder',sampler_task='mock.sampler',QM_task='mock.qm',ML_task='mock.ml',
            parsl_configuration='mock.config',save_h5_threshold=2,parallel_samplers=2,maximum_builder_structures=2,gpus_per_node=0)
        workflow.write_json(galf/'master_config.json',master)
        workflow.write_json(galf/'sampler_galf.json',sampler_config)
        galf_phase=True
        training_calls.clear()
        ticks.clear()
        monkeypatch.setattr(sys,'argv',['alframework','--master',str(galf/'master_config.json')])
        modules['mock.config']=Config(executors=[ThreadPoolExecutor(label=f'alf_{name}_executor',max_threads=2)
            for name in ('builder','sampler','QM','ML')],strategy='none',run_dir=str(galf/'runinfo'))
        monkeypatch.chdir(galf)
        try:
            with pytest.raises(DriverFinished):
                runpy.run_module('alframework',run_name='__main__')
        finally:
            parsl.dfk().cleanup()
            parsl.clear()
        assert training_calls[0] == (0,[identifier for record in manifest['sources'] for identifier in record['ids']])
        assert len(workflow.check(galf/'master_config.json',require_trained=True)['initial_training'])==4
        assert (galf/'h5store/data-0002.h5').exists()
        small=workflow.freeze(galf/'master_config.json',tmp_path/'checkpoint3',3)
        large=workflow.freeze(galf/'master_config.json',tmp_path/'checkpoint4',4)
        keys=lambda result:[(record['sha256'],row) for record in result['sources'] for row in record['rows']]
        assert keys(small)==keys(large)[:3]
        assert large['phase_counts']=={'seed':1,'alf':1,'galf':2}
