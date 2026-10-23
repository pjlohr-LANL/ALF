"""Retained-frame budgets, durable publication/replay, and linear workflows."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import h5py
import numpy as np
import pytest
from ase import Atoms
from ase.io import read,write

from alframework.tools.workflow_io import (
    ROW_KEY, PHASE_KEY, ExclusiveLock, atomic_json, fingerprint, freeze_snapshot,
    read_json, scan_store, verify_snapshot,
)
from alframework.tools.staged_workflow import definition,prepare_workflow,WorkflowRunner

ROOT=Path(__file__).resolve().parents[1]


def fixture_workflow(tmp_path, *, excited=False, stages=2, fail=False):
    inputs=tmp_path/'inputs'; inputs.mkdir()
    if excited:
        atoms=read(ROOT/'examples/keto_rdkit_alf_galf/keto.xyz')
    else:
        atoms=Atoms('OH2',positions=[[0,0,0],[.96,0,0],[-.24,.93,0]])
    reference=inputs/'reference.xyz'; write(reference,atoms)
    order=np.argsort(atoms.numbers,kind='stable')
    with h5py.File(inputs/'seed.h5','w') as handle:
        group=handle.create_group('seed')
        group['coordinates']=np.stack([atoms.positions[order]]*2)
        group['species']=np.asarray(atoms.get_chemical_symbols(),dtype='S2')[order]
        group['topology_atom_ids']=order
        group['_id']=np.asarray(['same-source-id','same-source-id'],dtype='S20')
    properties=({'energy':['energy','system',1.],'forces':['forces','atomic',1.]} if not excited else
                {f'{prefix}{state}':[f'{prefix}{state}',scope,1.] for state in range(4) for prefix,scope in [('sE','system'),('F','atomic')]})
    if excited:
        properties.update({f'dE{i}{i+1}':[f'dE{i}{i+1}','system',1.] for i in range(3)})
    builder={'source_priority':'h5_only','selection_seed':42,'bootstrap_h5_path':str(inputs/'seed.h5'),
             'bootstrap_h5_count':1,'bootstrap_selection_mode':'sequential'}
    sampler={'model_mode':'excited_state' if excited else 'ground_state','dt':.1,'maxt':.0002,'Ncheck':1,
             'min_time':0.,'Escut':.001,'Fscut':.01,'distcut':.7,'min_distance_cutoff':.7,'max_force_cutoff':16.,
             'srt_temp':[300.,300.],'end_temp':[300.,300.],'amp_temp':[0.,0.],'per_temp':[1.,1.],
             'end_dens':None,'amp_dens':None,'per_dens':None,'return_top_k':2,'uncertainty_policy':'continue',
             'score':{'mode':'sum','w_energy':1.,'w_force':1.,'w_gap_uncertainty':0.},
             'dataset_screening':{'enabled':True,'force':excited,'min_distance':True,'topology':excited}}
    if excited:
        sampler.update(state_selection={'mode':'fixed','state':0},alchemi_baoab={'batch_size':2,'partial_policy':'full_only'},
                       topology_check={'enabled':True,'reference_conformer_path':'reference.xyz','reference_format':'xyz','reference_charge':0,
                                       'bond_min_scale':.85,'bond_max_scale':1.5,'connectivity_scale':1.25})
    master={'master_directory':'pwd','h5_path':'h5store/data-{:04d}.h5','model_path':'models/model-{:04d}',
            'status_path':'status.json','QM_task':'tests.helpers.staged_apps.qm','QM_config_path':'QM.json',
            'ML_task':'tests.helpers.staged_apps.train','ML_config_path':'ML.json',
            'builder_task':'alframework.builders.h5_replay_builder.h5_replay_builder_local_task','builder_config_path':'builder.json',
            'sampler_task':'tests.helpers.staged_apps.sample','sampler_config_path':'sampler.json',
            'properties_list':properties,'parsl_configuration':'tests.helpers.staged_apps.config',
            'parsl_debug_configuration':'unused_debug.configuration',
            'bootstrap_set':2,'parallel_samplers':2,'maximum_builder_structures':2 if excited else 1,
            'target_queued_QM':2,'minimum_QM':0,'save_h5_threshold':2,'gpus_per_node':0,
            'run_control':{'poll_interval_seconds':.01}}
    for filename,value in [('master.json',master),('builder.json',builder),('sampler.json',sampler),
                           ('QM.json',{'events_dir':'events','reject_seed':True,'fail_first_qm':fail}),('ML.json',{'n_models':4})]:
        atomic_json(inputs/filename,value)
    plan=[]
    for index in range(stages):
        stage_master=dict(master)
        stage_sampler=dict(sampler)
        if excited and index==1:
            stage_sampler['gap_seeking']={'enabled':True,'mode':'levine_coe_martinez_switch','switch_policy':'stay_fixed',
                                          'candidate_pairs':'adjacent','trigger_gap_threshold_eV':.1,'alpha_eV':.05,'sigma':3.5}
        atomic_json(inputs/f'sampler{index}.json',stage_sampler)
        stage_master['sampler_config_path']=f'sampler{index}.json'
        atomic_json(inputs/f'master{index}.json',stage_master)
        plan.append({'name':'galf' if excited and index==1 else f'alf{index}', 'master_config':f'master{index}.json',
                     'target_total_frames':3+index*2,'checkpoints':[4] if index==1 else [],
                     'output_paths':{'QM.events_dir':'events'}})
    path=inputs/'workflow.json'; atomic_json(path,{'stages':plan,'final_training':True})
    return path


def test_snapshot_multiple_groups_alignment_units_metadata_and_duplicates(tmp_path):
    properties={'energy':['energy','system',27.2114],'forces':['forces','atomic',51.422]}
    directory=tmp_path/'raw'; directory.mkdir()
    for shard in (10,2):
        with h5py.File(directory/f'data-{shard:04d}.h5','w') as handle:
            for formula,atoms in [('O01',1),('H02',2)]:
                group=handle.create_group(formula)
                group['coordinates']=np.ones((3,atoms,3))*shard
                group['species']=np.asarray(['O'] if atoms==1 else ['H','H'],dtype='S1')
                group['_id']=np.asarray(['same']*3,dtype='S4')
                group['energy']=np.arange(3,dtype=float)
                group['forces']=np.arange(3*atoms*3).reshape(3,atoms,3)
                group['unused_metadata']=np.array([np.nan,2.,3.])
                group['nac_pairs']=np.array([[0,1],[1,2],[0,2]]) # Static first axis equals frame count.
                group['energy'].attrs['units']='stored energy units'
                if shard==2 and atoms==2:
                    group['coordinates'][1,0,0]=np.inf
    pattern=str(directory/'data-{:04d}.h5')
    selection=scan_store(pattern,properties,count=4,closed_before=10)
    assert selection['frames']==4 and selection['available_frames']==5 and selection['nonfinite_frames']==1
    assert selection['sources'][0]['groups'][0]['rows']==[0,2]
    snapshot=freeze_snapshot(pattern,properties,4,tmp_path/'checkpoint',closed_before=10)
    assert snapshot==verify_snapshot(tmp_path/'checkpoint')
    with h5py.File(tmp_path/'checkpoint/h5store/data-0000.h5') as handle:
        assert handle['H02/coordinates'].shape==(2,2,3)
        assert handle['H02/nac_pairs'].shape==(3,2)
        assert handle['H02/_id'].asstr()[:].tolist()==['same','same']
        assert handle['O01/energy'].attrs['units']=='stored energy units'
        assert handle['H02/energy'][:].tolist()==[0.,2.]
    with pytest.raises(ValueError,match='Need 6'):
        freeze_snapshot(pattern,properties,6,tmp_path/'short',closed_before=10)
    assert not (tmp_path/'short').exists()
    with ExclusiveLock(tmp_path/'lock'):
        with pytest.raises(RuntimeError,match='Another controller'):
            ExclusiveLock(tmp_path/'lock')
    assert snapshot['units']=={'energy':{'units':'stored energy units'}}
    with h5py.File(directory/'data-0010.h5','a') as handle:
        handle['H02/energy'].attrs['units']='incompatible energy units'
    with pytest.raises(ValueError,match='Conflicting stored units'):
        scan_store(pattern,properties)


@pytest.mark.parametrize('excited,stages',[(False,1),(False,2),(True,2)],ids=['alf-only','legacy-stages','alf-galf'])
def test_full_workflow_actual_driver_targets_and_final_model(tmp_path,excited,stages):
    path=fixture_workflow(tmp_path,excited=excited,stages=stages)
    root=tmp_path/'run'
    prepare_workflow(path,root)
    assert WorkflowRunner(root).run()==0
    status=read_json(root/'workflow_status.json')
    assert status['state']=='completed'
    last=3+(stages-1)*2
    assert status['final_training']['input_frames']==last
    model=Path(status['final_training']['directory'])
    assert read_json(model/'input_frames.json')['frames']==last
    for index,stage in enumerate(read_json(root/'workflow.json')['stages']):
        directory=Path(stage['run_directory'])
        progress=read_json(directory/'control/progress.json')
        assert progress['state']=='completed' and progress['accepted_frames']>=stage['target_total_frames']
        manifest=verify_snapshot(directory/'checkpoints'/str(stage['target_total_frames']))
        assert manifest['frames']==stage['target_total_frames']
        inputs=read_json(directory/'models/model-0000/input_frames.json')
        assert inputs['frames']==(1 if index==0 else 3) # First seed converges; second is rejected.
        if index:
            small=verify_snapshot(directory/'checkpoints/4')
            big=verify_snapshot(directory/'checkpoints/5')
            keys=lambda m:[(r['sha256'],g['group'],row) for r in m['sources'] for g in r['groups'] for row in g['rows']]
            assert keys(small)==keys(big)[:4]
        with sqlite3.connect(directory/'control/journal.sqlite') as db:
            assert db.execute("SELECT COUNT(*) FROM tasks WHERE role='QM' AND state!='consumed'").fetchone()[0]==0
            if index:
                acquired=db.execute("SELECT COUNT(*) FROM tasks WHERE role='QM' AND state='consumed'").fetchone()[0]
                assert progress['accepted_frames']==3+acquired # Independent branches may reuse frame IDs, never workflow row identities.

    if excited:
        import warnings
        from hippynn.databases.h5_pyanitools import PyAniDirectoryDB
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Skipping key')
            database=PyAniDirectoryDB(directory=str(root/'stages/01-galf/checkpoints/5/h5store'),
                inputs=None,targets=None,allow_unfound=True,quiet=True,seed=42)
        assert database.arr_dict['coordinates'].shape[0]==last
        assert database.arr_dict['F0'].shape==(last,15,3)
    before=fingerprint(model/'input_frames.json')
    assert WorkflowRunner(root).run()==0
    assert fingerprint(model/'input_frames.json')==before


def test_validation_rejects_incompatible_labels_and_mutation(tmp_path):
    path=fixture_workflow(tmp_path)
    config=read_json(path)
    config['stages'][1]['target_total_frames']=3
    atomic_json(path,config)
    with pytest.raises(ValueError,match='strictly increase'):
        definition(path)
    config['stages'][1]['target_total_frames']=5
    atomic_json(path,config)
    stage=path.parent/'QM.json'
    original=read_json(stage)
    other={**original,'method':'different'}
    atomic_json(path.parent/'otherQM.json',other)
    master=read_json(path.parent/'master1.json'); master['QM_config_path']='otherQM.json'
    atomic_json(path.parent/'master1.json',master)
    with pytest.raises(ValueError,match='incompatible'):
        definition(path)
    master['QM_config_path']='QM.json'
    master['QM_task']='tests.helpers.staged_apps.other_label_method'
    atomic_json(path.parent/'master1.json',master)
    with pytest.raises(ValueError,match='incompatible'):
        definition(path)


class ImmediateApp:
    """Journal tests use the same Future interface as Parsl without an executor."""
    def __init__(self, callback):
        self.callback=callback
        self.func=callback
        self.executors=[]

    def __call__(self,**kwargs):
        callback=self.callback
        class Future:
            def task_status(self): return 'exec_done'
            def running(self): return False
            def done(self): return True
            def result(self): return callback(**kwargs)
        return Future()


def journal_control(root, *, target=2):
    from alframework.tools.run_control import create_run_control
    from alframework.tools.tools import load_config_file,parsl_task_queue
    from alframework.tools.sampler_batching import SamplerBatchBuffer
    master_path=root/'master.json'
    if not master_path.exists():
        root.mkdir()
        for role in ('builder','sampler','QM','ML'):
            atomic_json(root/f'{role}.json',{})
        atomic_json(master_path,{'master_directory':str(root),'status_path':'status.json',
            'h5_path':'h5store/data-{:04d}.h5','model_path':'models/model-{:04d}',
            **{f'{role}_config_path':f'{role}.json' for role in ('builder','sampler','QM','ML')},
            'properties_list':{'energy':['energy','system',1.],'forces':['forces','atomic',1.]},
            'bootstrap_set':0,'run_control':{'master_path':str(master_path),'stage_name':'test','target_total_frames':target}})
    master=load_config_file(str(master_path))
    configs=[master]+[load_config_file(master[f'{role}_config_path'],str(root)) for role in ('builder','sampler','QM','ML')]
    control=create_run_control(master,configs)
    queues={role:parsl_task_queue() for role in ('builder','sampler','QM','ML')}
    status={'current_model_id':-1,'current_training_id':0,'current_h5_id':0,'current_molecule_id':0}
    return control,master,queues,status,SamplerBatchBuffer()


def candidate(origin):
    from alframework.tools.molecules_class import MoleculesObject
    molecule=MoleculesObject(Atoms('H2',positions=[[0,0,0],[.75,0,0]]),'same-id')
    molecule.update_metadata({'_workflow_origin':origin,'_workflow_phase':'seed'})
    return molecule


@pytest.mark.parametrize('fault',['submission','result','publication','checkpoint'])
def test_journal_recovers_boundary_without_duplicate_retained_rows(tmp_path,monkeypatch,fault):
    from alframework.tools import run_control as module
    control,master,queues,status,buffer=journal_control(tmp_path/'control')
    calls=[]
    def label(molecule_object):
        calls.append(molecule_object.get_metadata()['_workflow_origin'])
        molecule_object.store_results({'energy':1.,'forces':np.zeros((2,3))})
        molecule_object.set_converged_flag(True)
        return [[molecule_object]]
    app=ImmediateApp(label)
    for role in queues:
        control.wrap(role,app)
    control.bind(status,queues,buffer)
    qm=module.JournalApp(control,'QM',app)
    futures=[qm(molecule_object=candidate(origin)) for origin in ('independent-a','independent-b')]
    if fault!='submission':
        results=[mol for future in futures for mol in module.flatten_molecule_output(future.result())]
    if fault in ('publication','checkpoint'):
        if fault=='publication':
            original=control.db
            class CrashBeforeJournalCommit:
                def execute(self,sql,*args):
                    if "SET state='consumed'" in sql:
                        raise RuntimeError('Injected crash after publication')
                    return original.execute(sql,*args)
                def __getattr__(self,key): return getattr(original,key)
            control.db=CrashBeforeJournalCommit()
        else:
            monkeypatch.setattr(module,'freeze_snapshot',lambda *args,**kwargs: (_ for _ in ()).throw(RuntimeError('Injected checkpoint crash')))
        with pytest.raises(RuntimeError,match='Injected'):
            control.store(master['h5_path'].format(0),results,master['properties_list'],{},master_directory=str(tmp_path/'control'))
        control.db=original if fault=='publication' else control.db
    control.abort()
    monkeypatch.undo()
    restored,master,queues,status,buffer=journal_control(tmp_path/'control')
    for role in queues:
        restored.wrap(role,app)
    restored.bind(status,queues,buffer)
    recovered=[]
    for future in list(queues['QM'].task_list):
        recovered.extend(module.flatten_molecule_output(future.result()))
    if recovered:
        stored,_,_=restored.store(master['h5_path'].format(status['current_h5_id']),recovered,master['properties_list'],{},master_directory=str(tmp_path/'control'))
    restored.make_checkpoints()
    snapshot=verify_snapshot(tmp_path/'control/checkpoints/2')
    assert snapshot['frames']==2 and snapshot['phase_counts']=={'seed':2}
    with h5py.File(tmp_path/'control/h5store/data-0000.h5') as handle:
        group=next(iter(handle.values()))
        assert len(set(group[ROW_KEY].asstr()[:]))==2
        assert group['_id'].asstr()[:].tolist()==['same-id','same-id']
    assert len(calls)==2 # Each successful result is labeled once; submitted-only work is retried once.
    assert scan_store(master['h5_path'],master['properties_list'])['frames']==2
    restored.write_progress()
    assert restored.state['screening_totals']['kept']==2
    assert restored.state['screening_totals']['converged']==2
    assert restored.db.execute("SELECT COUNT(*) FROM tasks WHERE role='QM' AND state!='consumed'").fetchone()[0]==0
    restored.close()
    if fault=='publication':
        import shutil
        shutil.copy2(Path(master['h5_path'].format(0)),Path(master['h5_path'].format(9999)))
        with pytest.raises(RuntimeError,match='different shard path'):
            journal_control(tmp_path/'control')


def test_training_result_and_partial_attempt_recovery(tmp_path):
    from alframework.tools.run_control import JournalApp
    control,master,queues,status,buffer=journal_control(tmp_path/'models')
    def train(**kwargs):
        directory=Path(kwargs['model_path'].format(kwargs['current_training_id']))
        directory.mkdir(parents=True)
        (directory/'complete').write_text('complete')
        return [True]*4,kwargs['current_training_id']
    app=ImmediateApp(train)
    for role in queues: control.wrap(role,app)
    control.bind(status,queues,buffer)
    Path(master['h5_dir']).mkdir(parents=True,exist_ok=True)
    future=JournalApp(control,'ML',app)(ML_config={'n_models':4},h5_dir=master['h5_dir'],model_path=master['model_path'],current_training_id=0)
    assert future.result()==([True]*4,0)
    control.abort() # Successful result exists, but driver status has not promoted it.
    restored,master,queues,status,buffer=journal_control(tmp_path/'models')
    for role in queues: restored.wrap(role,app)
    restored.bind(status,queues,buffer)
    assert status['current_model_id']==0 and (Path(master['model_path'].format(0))/'complete').exists()
    restored.close()
    # A different run dies with an unfinished initial model directory.
    control,master,queues,status,buffer=journal_control(tmp_path/'partial')
    for role in queues: control.wrap(role,app)
    control.bind(status,queues,buffer)
    Path(master['h5_dir']).mkdir(parents=True,exist_ok=True)
    future=JournalApp(control,'ML',app)(ML_config={'n_models':4},h5_dir=master['h5_dir'],model_path=master['model_path'],current_training_id=0)
    Path(master['model_path'].format(0)).mkdir(parents=True)
    control.abort()
    restored,master,queues,status,buffer=journal_control(tmp_path/'partial')
    for role in queues: restored.wrap(role,app)
    restored.bind(status,queues,buffer)
    assert queues['ML'].task_list[0].result()==([True]*4,1)
    assert Path(master['model_path'].format(0)).exists() # Partial attempt preserved.
    restored.close()
    with h5py.File(Path(master['h5_dir'])/'unexpected.h5','w'):
        pass
    with pytest.raises(RuntimeError,match='outside the configured shard pattern'):
        journal_control(tmp_path/'partial')


@pytest.mark.parametrize('ignored_interrupt',[False,True],ids=['normal-signals','SIGINT-overridden'])
def test_pause_drains_and_resume_reuses_committed_work(tmp_path,monkeypatch,ignored_interrupt):
    if ignored_interrupt:
        monkeypatch.setenv('ALF_TEST_IGNORE_DRIVER_SIGINT','1')
    import threading,time
    path=fixture_workflow(tmp_path,stages=1)
    root=tmp_path/'run'; prepare_workflow(path,root)
    runner=WorkflowRunner(root)
    def pause_at_first_shard():
        deadline=time.monotonic()+20
        shard=root/'stages/00-alf0/h5store/data-0000.h5'
        while not shard.exists() and time.monotonic()<deadline:
            time.sleep(.002)
        runner.interrupt()
    thread=threading.Thread(target=pause_at_first_shard,daemon=True); thread.start()
    assert runner.run()==130
    thread.join(timeout=1)
    progress=read_json(root/'stages/00-alf0/control/progress.json')
    assert progress['state']=='paused'
    with sqlite3.connect(root/'stages/00-alf0/control/journal.sqlite') as db:
        completed=dict(db.execute("SELECT key,attempts FROM tasks WHERE role='QM' AND state='consumed'"))
    assert WorkflowRunner(root).run()==0
    with sqlite3.connect(root/'stages/00-alf0/control/journal.sqlite') as db:
        assert all(dict(db.execute('SELECT key,attempts FROM tasks'))[key]==attempts for key,attempts in completed.items())
    assert read_json(root/'workflow_status.json')['state']=='completed'


@pytest.mark.parametrize('role',['QM','builder'])
def test_backend_failure_blocks_advancement_and_explicit_resume_retries(tmp_path,role):
    path=fixture_workflow(tmp_path,fail=role=='QM')
    if role=='builder':
        for index in range(2):
            master=read_json(path.parent/f'master{index}.json')
            master['builder_task']='tests.helpers.staged_apps.builder'
            atomic_json(path.parent/f'master{index}.json',master)
    root=tmp_path/'run'; prepare_workflow(path,root)
    with pytest.raises(RuntimeError,match='Stage alf0 failed'):
        WorkflowRunner(root).run()
    assert not (root/'stages/01-alf1/control/progress.json').exists()
    assert read_json(root/'workflow_status.json')['state']=='failed'
    with pytest.raises(RuntimeError,match='Stage alf1 failed'):
        WorkflowRunner(root).run()
    assert WorkflowRunner(root).run()==0
    state=read_json(root/'stages/00-alf0/control/progress.json')
    assert state['possible_recomputations']>=1
    assert state['tasks'][role]['attempt_outcomes']['failed']>=1
    assert state['tasks'][role]['attempt_outcomes']['succeeded']>=1


@pytest.mark.parametrize('fault',['handoff','final-training'])
def test_interrupted_handoff_and_final_training_resume(tmp_path,monkeypatch,fault):
    import shutil
    from alframework.tools import staged_workflow as module
    path=fixture_workflow(tmp_path)
    if fault=='final-training':
        atomic_json(path.parent/'ML.json',{'n_models':4,'fail_final_once':True})
    root=tmp_path/'run'; prepare_workflow(path,root)
    if fault=='handoff':
        def partial_copy(checkpoint,stage):
            manifest=verify_snapshot(checkpoint)
            first=manifest['artifacts'][0]
            destination=Path(stage['run_directory'])/first['relative_path']
            shutil.copy2(first['path'],destination)
            raise RuntimeError('Injected handoff interruption')
        monkeypatch.setattr(module,'inherit_checkpoint',partial_copy)
    with pytest.raises(RuntimeError,match='Injected handoff|Final training failed'):
        WorkflowRunner(root).run()
    before=read_json(root/'stages/00-alf0/control/progress.json')['tasks']['QM']['attempts']
    monkeypatch.undo()
    assert WorkflowRunner(root).run()==0
    assert read_json(root/'stages/00-alf0/control/progress.json')['tasks']['QM']['attempts']==before
    final=read_json(root/'workflow_status.json')['final_training']
    assert final['input_frames']==5
    if fault=='final-training':
        assert final['model_index']==1
        assert (root/'final_training/models/model-0000/partial').exists()
    artifact=Path(final['artifacts'][0]['path'])
    artifact.write_bytes(b'changed completed model')
    with pytest.raises(RuntimeError,match='fingerprint changed'):
        WorkflowRunner(root).run()


def test_cli_validation_status_and_pinned_resume(tmp_path,capsys):
    from alframework.tools.staged_workflow import main
    path=fixture_workflow(tmp_path,stages=1)
    root=tmp_path/'run'
    assert main(['--config',str(path),'--run-dir',str(root),'--validate'])==0
    assert not root.exists()
    assert json.loads(capsys.readouterr().out)['valid']
    prepare_workflow(path,root)
    assert main(['--run-dir',str(root),'--status'])==0
    report=json.loads(capsys.readouterr().out)
    assert report['workflow']['state']=='prepared' and report['stages']=={}
    value=read_json(path); value['final_training']=False; atomic_json(path,value)
    with pytest.raises(RuntimeError,match='definition changed'):
        main(['--config',str(path),'--run-dir',str(root),'--resume'])


def test_bootstrap_completion_between_polls_does_not_report_exhausted_seed(tmp_path):
    """A QM result can finish after journaling but before the queue status poll."""
    control,master,queues,status,buffer=journal_control(tmp_path/'bootstrap-race',target=3)
    def label(molecule_object):
        molecule_object.store_results({'energy':1.,'forces':np.zeros((2,3))})
        molecule_object.set_converged_flag(True)
        return molecule_object
    class BoundaryApp(ImmediateApp):
        def __call__(self,**kwargs):
            class Future:
                polls=0
                def task_status(self):
                    self.polls+=1
                    return 'running' if self.polls==1 else 'exec_done'
                def running(self):return self.polls<2
                def result(self):return label(**kwargs)
            return Future()
    for role in queues:control.wrap(role,ImmediateApp(label))
    control.bind(status,queues,buffer)
    master['bootstrap_set']=2
    control.bootstrap_required=True;control.state['bootstrap_required']=True
    status['current_molecule_id']=2
    first=control.submit('QM',ImmediateApp(label),{'molecule_object':candidate('first')})
    last=control.submit('QM',BoundaryApp(label),{'molecule_object':candidate('last')})
    queues['QM'].add_task(first);queues['QM'].add_task(last)
    try:
        for _ in range(3):
            wait=control.bootstrap_wait()
            assert control.failure is None, 'Task completed between status polls; quota is not exhausted'
            if not wait:break
        assert not wait and not control.bootstrap_required
        assert control.db.execute("SELECT COUNT(*) FROM tasks WHERE role='QM' AND state='ready'").fetchone()[0]==2
    finally:control.close()


def test_bootstrap_real_shortfall_still_reports_exhausted_seed(tmp_path):
    control,master,queues,status,buffer=journal_control(tmp_path/'bootstrap-shortfall',target=3)
    def label(molecule_object):return molecule_object
    for role in queues:control.wrap(role,ImmediateApp(label))
    control.bind(status,queues,buffer)
    master['bootstrap_set']=2;control.bootstrap_required=True;status['current_molecule_id']=2
    future=control.submit('QM',ImmediateApp(label),{'molecule_object':candidate('only-one')})
    queues['QM'].add_task(future)
    try:
        assert control.bootstrap_wait() is False
        assert 'exhausted' in control.failure
    finally:control.close()


def test_resume_after_bootstrap_store_failure_reuses_all_seed_labels(tmp_path):
    """Recover enol's exact boundary: seed results stored, initial ML not started."""
    from alframework.tools.run_control import create_run_control,JournalApp
    from alframework.tools.tools import load_config_file,parsl_task_queue
    from alframework.tools.sampler_batching import SamplerBatchBuffer
    from alframework.tools.molecules_class import MoleculesObject
    path=fixture_workflow(tmp_path,stages=1)
    root=tmp_path/'run';prepare_workflow(path,root)
    stage=root/'stages/00-alf0'
    master=load_config_file(str(stage/'master_config.json'))
    configs=[master]+[load_config_file(master[f'{role}_config_path'],str(stage)) for role in ('builder','sampler','QM','ML')]
    control=create_run_control(master,configs)
    queues={role:parsl_task_queue() for role in ('builder','sampler','QM','ML')}
    status={'current_model_id':-1,'current_training_id':0,'current_h5_id':0,'current_molecule_id':2,
            **{f'lifetime_failed_{role}_tasks':0 for role in ('builder','sampler','QM','ML')}}
    def label(molecule_object):
        molecule_object.store_results({'energy':1.,'forces':np.zeros((3,3))})
        molecule_object.set_converged_flag(molecule_object.get_moleculeid().endswith('0'))
        return molecule_object
    for role in queues:control.wrap(role,ImmediateApp(label))
    control.bind(status,queues,SamplerBatchBuffer())
    atoms=read(path.parent/'reference.xyz');results=[]
    for i in range(2):
        molecule=MoleculesObject(atoms.copy(),f'mol-boot-{i:010d}')
        molecule.update_metadata({'_workflow_origin':f'builder:{i}','_workflow_phase':'seed'})
        future=JournalApp(control,'QM',ImmediateApp(label))(molecule_object=molecule)
        results.extend([future.result()])
    # Preserve the old fault and unfinished bootstrap flag as found in the pilot.
    control.failure='Bootstrap builders exhausted the selected seed quota before producing all labeling tasks.'
    assert control.bootstrap_required
    stored,_,_=control.store(master['h5_path'].format(0),results,master['properties_list'],configs[2],master_directory=str(stage))
    assert stored and control.accepted==1
    status['current_h5_id']=1;control.save(status);control.abort()
    with sqlite3.connect(stage/'control/journal.sqlite') as db:
        before=dict(db.execute("SELECT key,attempts FROM tasks WHERE role='QM'"))
    assert WorkflowRunner(root).run()==0
    with sqlite3.connect(stage/'control/journal.sqlite') as db:
        after=dict(db.execute("SELECT key,attempts FROM tasks WHERE role='QM'"))
        assert all(after[key]==attempts for key,attempts in before.items())
        assert db.execute("SELECT COUNT(*) FROM tasks WHERE role='QM' AND phase='seed'").fetchone()[0]==2
    assert read_json(stage/'models/model-0000/input_frames.json')['frames']==1
    assert verify_snapshot(stage/'checkpoints/3')['frames']==3


@pytest.mark.parametrize('code,outcome',[(1,'failed'),(-15,'paused'),(1,'paused'),(0,'failed')])
def test_pause_request_does_not_mask_failed_or_killed_driver(tmp_path,monkeypatch,code,outcome):
    path=fixture_workflow(tmp_path,stages=1)
    root=tmp_path/'run';prepare_workflow(path,root)
    runner=WorkflowRunner(root)
    def interrupted_execute(master):
        runner.interrupt()
        atomic_json(Path(master).parent/'control/progress.json',{'state':outcome})
        return code
    monkeypatch.setattr(runner,'execute',interrupted_execute)
    with pytest.raises(RuntimeError,match='Stage alf0 failed'):
        runner.run()
    assert read_json(root/'workflow_status.json')['state']=='failed'
    assert not (root/'final_training').exists()


@pytest.mark.parametrize('at_budget',[False,True])
def test_resume_budget_retires_only_unfinished_acquisition_not_qm(tmp_path,at_budget):
    """Do not rerun an 80-minute sampler after enough labels are committed."""
    from alframework.tools.run_control import flatten_molecule_output
    target=2 if at_budget else 5
    root=tmp_path/'budget-recovery'
    control,master,queues,status,buffer=journal_control(root,target=target)
    calls=[]
    def label(molecule_object):
        calls.append(molecule_object.get_metadata()['_workflow_origin'])
        molecule_object.store_results({'energy':1.,'forces':np.zeros((2,3))})
        molecule_object.set_converged_flag(True)
        return molecule_object
    qm_app=ImmediateApp(label)
    other_app=ImmediateApp(lambda **kwargs:[])
    for role in queues:control.wrap(role,qm_app if role=='QM' else other_app)
    control.bind(status,queues,buffer)
    results=[control.submit('QM',qm_app,{'molecule_object':candidate(f'seed:{i}')}).result() for i in range(2)]
    control.store(master['h5_path'].format(0),results,master['properties_list'],{},master_directory=str(root))
    # One failed sampler, one unknown builder, and two QM results (ready and
    # unknown) survive the allocation. All existing attempt evidence remains.
    sampler=control.submit('sampler',other_app,{'molecule_objects':[candidate('parent')]})
    control.db.execute("UPDATE tasks SET state='failed',error='deadline' WHERE key=?",(sampler.key,))
    control.db.execute("UPDATE attempts SET state='failed',error='deadline' WHERE task_key=?",(sampler.key,))
    builder=control.submit('builder',other_app,{'moleculeids':['mol-0000-0000000000']})
    ready=control.submit('QM',qm_app,{'molecule_object':candidate('pending-ready')});ready.result()
    pending=control.submit('QM',qm_app,{'molecule_object':candidate('pending-unknown')})
    control.abort()
    restored,master,queues,status,buffer=journal_control(root,target=target)
    for role in queues:restored.wrap(role,qm_app if role=='QM' else other_app)
    restored.bind(status,queues,buffer)
    assert restored.task(sampler.key)['state']==('abandoned' if at_budget else 'submitted')
    assert restored.task(builder.key)['state']==('abandoned' if at_budget else 'submitted')
    assert restored.db.execute("SELECT error FROM attempts WHERE task_key=? AND number=1",(sampler.key,)).fetchone()[0]=='deadline'
    assert len(restored.state.get('abandoned_acquisition',[]))==(2 if at_budget else 0)
    assert queues['sampler'].get_number()==(0 if at_budget else 1)
    recovered=[m for future in queues['QM'].task_list for m in flatten_molecule_output(future.result())]
    assert len(recovered)==2 and len(calls)==4
    restored.store(master['h5_path'].format(1),recovered,master['properties_list'],{},master_directory=str(root))
    assert restored.accepted==4
    assert restored.task(ready.key)['state']==restored.task(pending.key)['state']=='consumed'
    assert scan_store(master['h5_path'],master['properties_list'])['frames']==4
    restored.close()


def test_actual_driver_finishes_budget_recovery_and_exact_final_fit(tmp_path):
    """Reproduce keto's boundary: enough stored labels, interrupted sampler."""
    import pickle,time
    path=fixture_workflow(tmp_path,stages=1)
    root=tmp_path/'run';prepare_workflow(path,root)
    assert WorkflowRunner(root).run()==0
    stage=root/'stages/00-alf0'
    with sqlite3.connect(stage/'control/journal.sqlite') as db:
        counts=dict(db.execute("SELECT key,attempts FROM tasks WHERE role='QM'"))
        payload=db.execute("SELECT input FROM tasks WHERE role='sampler' LIMIT 1").fetchone()[0]
        now=time.time();key='failed-after-budget'
        db.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?)',
                   (key,'sampler',payload,None,'failed',1,'allocation deadline','alf0',now,now))
        db.execute('INSERT INTO attempts (task_key,number,state,input,error,started,finished) VALUES (?,?,?,?,?,?,?)',
                   (key,1,'failed',payload,'allocation deadline',now,now))
    progress=read_json(stage/'control/progress.json');progress.update(state='failed',failure='allocation deadline')
    atomic_json(stage/'control/progress.json',progress)
    state=read_json(root/'workflow_status.json');state.pop('final_training')
    state.update(state='failed',stage_states={'alf0':'failed'})
    atomic_json(root/'workflow_status.json',state)
    (root/'final_training').rename(root/'prior_test_final_training')
    assert WorkflowRunner(root).run()==0
    with sqlite3.connect(stage/'control/journal.sqlite') as db:
        assert db.execute('SELECT state,attempts FROM tasks WHERE key=?',(key,)).fetchone()==('abandoned',1)
        assert dict(db.execute("SELECT key,attempts FROM tasks WHERE role='QM'"))==counts
        assert db.execute('SELECT state FROM attempts WHERE task_key=?',(key,)).fetchone()[0]=='failed'
    assert read_json(root/'workflow_status.json')['final_training']['input_frames']==3
    assert read_json(root/'final_training/models/model-0000/input_frames.json')['frames']==3
