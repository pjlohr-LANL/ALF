"""Retained-frame snapshots and durable lifecycle recovery."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from ase import Atoms

from alframework.tools.workflow_io import (
    ROW_KEY, PHASE_KEY, ExclusiveLock, atomic_json, fingerprint, freeze_snapshot,
    read_json, scan_store, verify_snapshot,
)


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
