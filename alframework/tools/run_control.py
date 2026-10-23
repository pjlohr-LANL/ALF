"""Opt-in ALF lifecycle, durable task journal, and transactional H5 publication.

Journal payloads are local trusted pickle records, not an interchange format.
No sampler, QM, ML implementation or optional GPU stack is imported here.
"""
from __future__ import annotations

import atexit
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import sqlite3
import time
import uuid

import h5py
import numpy as np

from alframework.tools.workflow_io import (
    ROW_KEY, PHASE_KEY, ExclusiveLock, atomic_json, fingerprint, freeze_snapshot,
    read_json, scan_store, shards, sync_directory, verify,
)
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampler_batching import flatten_molecule_output


def positive_integer(value, name):
    if isinstance(value,bool) or not isinstance(value,int) or value < 1:
        raise ValueError(f'{name} must be a positive integer.')
    return value



def input_fingerprints(configs, output_fields=()):
    def leaves(config,prefix):
        for key,value in config.items():
            name=prefix+'.'+key
            if isinstance(value,dict):
                yield from leaves(value,name)
            else:
                yield name,key,value
    records={}
    for role,config in zip(('master','builder','sampler','QM','ML'),configs):
        for qualified,key,value in leaves(config,role):
            if (not isinstance(value,str) or not value or key in {'status_path','h5_path','model_path'}
                    or qualified in output_fields):
                continue
            paths=[]
            if key=='bootstrap_h5_path':
                paths=[Path(value.format(i)) for i in range(int(config.get('bootstrap_h5_count',1)))]
            elif key.endswith('_path') or key.endswith('_geometry'):
                paths=[Path(value)]
            elif key.endswith('library_dir') and Path(value).is_dir():
                paths=[path for path in Path(value).rglob('*') if path.is_file()]
            for path in paths:
                if path.is_file():
                    records[str(path.resolve())]=fingerprint(path)
    return list(records.values())


def clean_payload(value):
    if isinstance(value,MoleculesObject):
        result=copy.copy(value)
        result.atoms=value.get_atoms().copy() if value.get_atoms() is not None else None
        if result.atoms is not None:
            result.atoms.calc=None
        result.metadata=copy.deepcopy(value.get_metadata())
        result.qm_results=copy.deepcopy(value.get_results())
        return result
    if isinstance(value,dict):
        return {key:clean_payload(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return type(value)(clean_payload(item) for item in value)
    return value


def dump(value):
    return pickle.dumps(clean_payload(value),protocol=pickle.HIGHEST_PROTOCOL)


class JournalFuture:
    def __init__(self, control, key, native=None, empty=False):
        self.control,self.key,self.native,self.empty=control,key,native,empty
        self.observed_retries=0
        self.attempt=None
        if native is not None:
            self.attempt=control.db.execute('SELECT MAX(number) FROM attempts WHERE task_key=?',(key,)).fetchone()[0]

    def observe_retries(self):
        retries=int(getattr(self.native,'task_record',{}).get('try_id',0)) if self.native is not None else 0
        if retries>self.observed_retries:
            self.control.db.execute('UPDATE tasks SET attempts=attempts+? WHERE key=?',(retries-self.observed_retries,self.key))
            self.control.db.execute('UPDATE attempts SET native_retries=? WHERE task_key=? AND number=?',(retries,self.key,self.attempt))
            self.control.db.commit()
            self.observed_retries=retries

    def task_status(self):
        self.observe_retries()
        if self.empty:
            return 'exec_done'
        row=self.control.task(self.key)
        if row['state']=='failed':
            return 'failed'
        return self.native.task_status() if self.native is not None and row['state']=='submitted' else 'exec_done'

    def running(self):
        return bool(self.native is not None and self.native.running())

    def done(self):
        return self.task_status() in {'exec_done','memo_done','failed','dep_fail'}

    def result(self):
        if self.empty:
            return []
        row=self.control.task(self.key)
        if row['state']=='failed':
            raise RuntimeError(row['error'])
        if row['state'] in {'consumed','delivered'} and row['role']!='ML':
            return []
        if row['state']=='submitted':
            try:
                output=self.native.result()
                self.observe_retries()
                if row['role']=='ML':
                    inputs=self.control.db.execute('SELECT value FROM metadata WHERE key=?',(f'training-input:{self.key}',)).fetchone()
                    for record in pickle.loads(inputs['value']):
                        verify(record)
                    self.control.validate_model_result(output,pickle.loads(row['input']))
                else:
                    for index,molecule in enumerate(flatten_molecule_output(output)):
                        molecule.update_metadata({'_workflow_origin':f'{self.key}:{index}',
                                                  '_workflow_submission':self.key,
                                                  '_workflow_phase':row['phase']})
                self.control.db.execute('UPDATE tasks SET output=?,state=?,updated=? WHERE key=?',
                    (dump(output),'ready',time.time(),self.key))
                self.control.db.execute('UPDATE attempts SET state=?,finished=? WHERE task_key=? AND number=?',
                    ('succeeded',time.time(),self.key,self.attempt))
                self.control.db.commit()
            except BaseException as exc:
                self.observe_retries()
                self.control.db.execute('UPDATE tasks SET state=?,error=?,updated=? WHERE key=?',
                    ('failed',f'{type(exc).__name__}: {exc}',time.time(),self.key))
                self.control.db.execute('UPDATE attempts SET state=?,error=?,finished=? WHERE task_key=? AND number=?',
                    ('failed',str(exc),time.time(),self.key,self.attempt))
                self.control.db.commit()
                self.control.failure=f'{row["role"]} task failed: {exc}'
                raise
            row=self.control.task(self.key)
        if row['role'] in {'builder','sampler','ML'}:
            self.control.deliveries.add(self.key)
        return pickle.loads(row['output'])


class JournalApp:
    def __init__(self, control, role, app):
        self.control,self.role,self.app=control,role,app
        self.func,self.executors=app.func,app.executors

    def __call__(self, **kwargs):
        return self.control.submit(self.role,self.app,kwargs)


class RunControl:
    def __init__(self, master, configs):
        self.master=master
        self.root=Path(master['master_directory']).resolve()
        self.options=master['run_control']
        self.target=positive_integer(self.options['target_total_frames'],'target_total_frames')
        self.checkpoints=sorted(set([self.target,*self.options.get('checkpoints',[])]))
        if any(positive_integer(x,'checkpoint')>self.target for x in self.checkpoints):
            raise ValueError('Checkpoint exceeds stage target.')
        self.mode=self.options.get('mode','active')
        if self.mode not in {'active','final_training'}:
            raise ValueError('Unsupported controlled execution mode.')
        self.name=self.options.get('stage_name','alf')
        self.poll_interval=float(self.options.get('poll_interval_seconds',1.))
        if not np.isfinite(self.poll_interval) or self.poll_interval<=0:
            raise ValueError('poll_interval_seconds must be finite and positive.')
        self.directory=self.root/'control'
        self.directory.mkdir(parents=True,exist_ok=True)
        self.lock=ExclusiveLock(self.directory/'run.lock')
        self.state_path=self.directory/'progress.json'
        self.db=sqlite3.connect(self.directory/'journal.sqlite')
        self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS tasks (key TEXT PRIMARY KEY, role TEXT, input BLOB, output BLOB, state TEXT, attempts INTEGER, error TEXT, phase TEXT, created REAL, updated REAL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS attempts (task_key TEXT,number INTEGER,state TEXT,input BLOB,error TEXT,started REAL,finished REAL,native_retries INTEGER DEFAULT 0, PRIMARY KEY(task_key,number))')
        self.db.execute('CREATE TABLE IF NOT EXISTS storage_events (id TEXT PRIMARY KEY,summary TEXT,submissions TEXT,output_path TEXT,file_record TEXT,state TEXT,created REAL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value BLOB)')
        self.db.commit()
        self.futures={}
        self.apps={}
        self.deliveries=set()
        self.failure=None
        self.pause_requested=False
        self.closed=False
        self.started=time.time()
        self.signals={}
        self.configs=configs
        records=[fingerprint(self.options['master_path'])]+[fingerprint(master[f'{role}_config_path']) for role in ('builder','sampler','QM','ML')]
        records.extend(input_fingerprints(configs,self.options.get('output_fields',())))
        records=list({record['path']:record for record in records}.values())
        old=read_json(self.state_path) if self.state_path.exists() else None
        if old:
            for record in old['configuration']:
                verify(record)
            if records!=old['configuration']:
                raise RuntimeError('Active stage configuration changed; explicit repair or a new workflow is required.')
            self.state=old
            if old['state']=='completed':
                raise RuntimeError('Stage is already complete; use the workflow status command.')
        else:
            if list(Path(master['model_dir']).glob('model-*')):
                raise RuntimeError('New controlled runs cannot infer models from existing directories.')
            self.state={'schema':'alf.run_control.v1','stage':self.name,'state':'running',
                        'configuration':records,'elapsed_seconds':0.,'possible_recomputations':0,
                        'unused_builder_outputs':0,'unused_buffered_structures':0,
                        'screening_totals':{},'bootstrap_required':not shards(master['h5_path']) and master.get('bootstrap_set',0)>0}
        if old is None:
            inheritance=self.root/'inheritance.json'
            if inheritance.exists():
                records=[]
                for artifact in read_json(inheritance)['artifacts']:
                    record=dict(artifact)
                    record['path']=str(self.root/artifact.get('relative_path',Path(artifact['path']).name))
                    verify(record)
                    records.append({key:record[key] for key in ('path','sha256','size')})
                self.state['initial_h5']=records
            else:
                self.state['initial_h5']=[fingerprint(path) for _,path in shards(master['h5_path'])]
        self.initial_files={record['path']:record for record in self.state['initial_h5']}
        self.namespace=self.state.setdefault('namespace',uuid.uuid4().hex)
        self.bootstrap_required=self.state['bootstrap_required']
        self.reconcile()
        scan=scan_store(master['h5_path'],master['properties_list'])
        self.accepted=scan['frames']
        self.state['phase_counts']=scan['phase_counts']
        self.prior_elapsed=self.state['elapsed_seconds']
        self.write_progress()
        for sig in (signal.SIGINT,signal.SIGTERM):
            self.signals[sig]=signal.getsignal(sig)
            signal.signal(sig,self.request_pause)
        atexit.register(self.abort)

    @property
    def draining(self):
        return self.pause_requested or self.failure is not None or self.accepted>=self.target

    @property
    def allow_acquisition(self):
        return not self.draining

    def request_pause(self, *unused):
        self.pause_requested=True

    def task(self,key):
        return self.db.execute('SELECT * FROM tasks WHERE key=?',(key,)).fetchone()

    def wrap(self,role,app):
        self.apps[role]=app
        return JournalApp(self,role,app)

    def logical_key(self,role,kwargs):
        if role=='builder':
            identity=kwargs.get('moleculeids') or [kwargs['moleculeid']]
        elif role=='sampler':
            molecules=kwargs.get('molecule_objects',[kwargs.get('molecule_object')])
            identity=[mol.get_metadata()['_workflow_origin'] for mol in molecules]
        elif role=='QM':
            identity=kwargs['molecule_object'].get_metadata()['_workflow_origin']
        else:
            identity=int(kwargs['current_training_id'])
        return self.namespace+':'+role+':'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()

    def submit(self,role,app,kwargs, *, recovery_key=None):
        key=recovery_key or self.logical_key(role,kwargs)
        if key in self.futures:
            return JournalFuture(self,key,empty=True)
        row=self.task(key)
        if row is not None and row['state'] in {'consumed','delivered'}:
            return JournalFuture(self,key,empty=True)
        if row is None:
            phase=self.name
            if role=='builder':
                names=kwargs.get('moleculeids') or [kwargs['moleculeid']]
                phase='seed' if all(name.startswith('mol-boot-') for name in names) else self.name
            elif role=='QM':
                phase=kwargs['molecule_object'].get_metadata().get('_workflow_phase',self.name)
            self.db.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?)',
                (key,role,dump(kwargs),None,'submitted',0,None,phase,time.time(),time.time()))
            self.db.commit()
            row=self.task(key)
        if row['state']=='ready':
            result=JournalFuture(self,key)
        else:
            if role=='ML':
                inputs=[fingerprint(p) for p in sorted(Path(kwargs['h5_dir']).glob('*.h5'))]
                self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(f'training-input:{key}',dump(inputs)))
            number=self.db.execute('SELECT COALESCE(MAX(number),0)+1 FROM attempts WHERE task_key=?',(key,)).fetchone()[0]
            self.db.execute('INSERT INTO attempts (task_key,number,state,input,started) VALUES (?,?,?,?,?)',
                            (key,number,'submitted',dump(kwargs),time.time()))
            self.db.execute('UPDATE tasks SET state=?,attempts=attempts+1,input=?,error=NULL,updated=? WHERE key=?',
                ('submitted',dump(kwargs),time.time(),key))
            self.db.commit()
            try:
                result=JournalFuture(self,key,app(**kwargs))
            except BaseException as exc:
                self.db.execute('UPDATE tasks SET state=?,error=? WHERE key=?',('failed',str(exc),key))
                self.db.execute('UPDATE attempts SET state=?,error=?,finished=? WHERE task_key=? AND number=?',
                    ('failed',str(exc),time.time(),key,number))
                self.db.commit()
                self.failure=f'{role} submission failed: {exc}'
                raise
        self.futures[key]=result
        return result

    def validate_model_result(self,result,kwargs):
        flags,index=result
        expected=kwargs.get('ML_config',{}).get('n_models')
        if len(flags)==0 or not all(flags) or (expected is not None and len(flags)!=int(expected)):
            raise RuntimeError('Trainer did not report a complete ensemble.')
        if int(index)!=int(kwargs['current_training_id']):
            raise RuntimeError('Trainer returned an unexpected model index.')
        directory=Path(str(kwargs['model_path']).format(int(index)))
        if not directory.is_dir():
            raise RuntimeError('Trainer reported success without an output model directory.')

    def bind(self,status,queues,buffer, *, sampler_replica_counts=None):
        self.queues,self.buffer=queues,buffer
        self.status=status
        model_id=-1
        model_files=Path(self.master['model_dir'])
        next_index=0
        for row in self.db.execute("SELECT * FROM tasks WHERE role='ML'").fetchall():
            kwargs=pickle.loads(row['input'])
            next_index=max(next_index,int(kwargs['current_training_id'])+1)
            if row['state'] in {'ready','delivered'}:
                result=pickle.loads(row['output'])
                self.validate_model_result(result,kwargs)
                model_id=max(model_id,int(result[1]))
        from alframework.tools.tools import find_empty_directory
        status['current_training_id']=max(next_index,find_empty_directory(self.master['model_path']))
        status['current_model_id']=model_id
        paths=shards(self.master['h5_path'])
        status['current_h5_id']=max((i+1 for i,_ in paths),default=0)
        row=self.db.execute("SELECT value FROM metadata WHERE key='buffer'").fetchone()
        if row:
            buffer.__dict__.update(pickle.loads(row['value']).__dict__)
        for row in self.db.execute('SELECT * FROM tasks ORDER BY created').fetchall():
            role=row['role']
            if row['state'] in {'consumed','delivered','abandoned'}:
                continue
            if (self.accepted>=self.target and role in {'builder','sampler'}
                    and row['state'] in {'submitted','failed'}):
                # Completed labels, including any pending QM results, remain
                # authoritative. Do not restart interrupted acquisition after
                # the retained-frame budget is already satisfied. Preserve
                # failed attempts and distinguish this from delivered work.
                now=time.time()
                self.state.setdefault('abandoned_acquisition',[]).append({
                    'key':row['key'],'role':role,'previous_state':row['state'],
                    'reason':'retained-frame target already met on resume','time':now})
                self.db.execute("UPDATE attempts SET state='unknown',finished=? WHERE task_key=? AND state='submitted'",(now,row['key']))
                self.db.execute("UPDATE tasks SET state='abandoned',updated=? WHERE key=?",(now,row['key']))
                self.db.commit()
                continue
            kwargs=pickle.loads(row['input'])
            if role=='ML' and row['state'] in {'submitted','failed'}:
                kwargs['current_training_id']=status['current_training_id']
                status['current_training_id']+=1
            if row['state']=='submitted':
                self.db.execute("UPDATE attempts SET state='unknown',finished=? WHERE task_key=? AND state='submitted'",(time.time(),row['key']))
                self.db.commit()
            if row['state'] in {'submitted','failed'}:
                self.state['possible_recomputations']+=int(row['attempts']>0)
            future=self.submit(role,self.apps[role],kwargs,recovery_key=row['key'])
            queues[role].add_task(future)
            if role=='sampler' and sampler_replica_counts is not None:
                # Recovery submits outside submit_sampler_batch(). Register
                # the original input width before the driver polls capacity;
                # output candidates (including empty output) are unrelated.
                molecules=kwargs.get('molecule_objects')
                sampler_replica_counts[id(future)]=len(molecules) if molecules is not None else 1
        # Reconcile the allocator against durable submissions, not a possibly older status file.
        prefix='mol-boot-' if status['current_model_id']<0 else f"mol-{status['current_model_id']:04d}-"
        indices=[]
        for row in self.db.execute("SELECT input FROM tasks WHERE role='builder'"):
            kwargs=pickle.loads(row['input'])
            for name in kwargs.get('moleculeids') or [kwargs['moleculeid']]:
                if name.startswith(prefix):
                    indices.append(int(name.rsplit('-',1)[1])+1)
        status['current_molecule_id']=max([status.get('current_molecule_id',0),*indices])
        self.save(status)

    def bootstrap_wait(self):
        # Bootstrap holds completed results until its single storage boundary.
        for future in self.queues['QM'].task_list:
            if future.done():
                try:
                    future.result()
                except Exception:
                    pass  # The journal records failure; the lifecycle now drains.
        if self.draining:
            return (self.queues['builder'].get_number()>0 or
                    self.queues['QM'].get_number()>self.queues['QM'].get_completed_number())
        completed=self.db.execute("SELECT COUNT(*) FROM tasks WHERE role='QM' AND phase='seed' AND state IN ('ready','consumed')").fetchone()[0]
        if completed>=self.master['bootstrap_set']:
            self.bootstrap_required=False
            self.state['bootstrap_required']=False
            return False
        # Use the same durable journal for completion and exhaustion. A native
        # future may finish after the result-harvest pass above; polling it again
        # here would falsely declare exhaustion before its result is journaled.
        pending=self.db.execute("SELECT COUNT(*) FROM tasks WHERE role='QM' AND phase='seed' AND state='submitted'").fetchone()[0]
        if (self.status['current_molecule_id']>=self.master['bootstrap_set'] and
                self.queues['builder'].get_number()==0 and pending==0):
            self.failure='Bootstrap builders exhausted the selected seed quota before producing all labeling tasks.'
            return False
        return True

    def reconcile(self):
        self.stored_tokens=set()
        paths=shards(self.master['h5_path'])
        expected={path.resolve() for _,path in paths}
        directory=Path(self.master['h5_dir'])
        if directory.exists() and any(path.resolve() not in expected for path in directory.iterdir() if path.suffix in {'.h5','.hdf5'}):
            raise RuntimeError('Unexpected training file outside the configured shard pattern.')
        for _,path in paths:
            initial=self.initial_files.get(str(path.resolve()))
            if initial is not None:
                verify(initial)
            with h5py.File(path) as handle:
                event=handle.attrs.get('alf_workflow_transaction')
                record=self.db.execute('SELECT * FROM storage_events WHERE id=?',(event,)).fetchone() if event else None
                if record is None and initial is None:
                    raise RuntimeError('Unregistered H5 shard in the controlled training store.')
                if record is not None:
                    if str(path.resolve())!=str(Path(record['output_path']).resolve()):
                        raise RuntimeError('Published H5 transaction is registered to a different shard path.')
                    if record['file_record'] is None:
                        raise RuntimeError('Published H5 lacks its durable transaction fingerprint.')
                    verify(json.loads(record['file_record']))
                    self.db.execute("UPDATE storage_events SET state='committed' WHERE id=?",(event,))
                    for key in json.loads(record['submissions']):
                        self.db.execute("UPDATE tasks SET state='consumed' WHERE key=?",(key,))
                    self.db.commit()
                for group in handle.values():
                    if ROW_KEY in group:
                        self.stored_tokens.update(group[ROW_KEY].asstr()[:].tolist())

    def valid_molecule(self,molecule):
        if not isinstance(molecule,MoleculesObject) or molecule.get_atoms() is None or not molecule.check_convergence():
            return False
        if not np.isfinite(molecule.get_atoms().positions).all():
            return False
        for key in self.master['properties_list']:
            values=np.asarray(molecule.get_results().get(key))
            if values.dtype.kind not in 'biufc' or not np.isfinite(values).all():
                return False
        return True

    def store(self,path,molecules,properties,sampler_config, *, master_directory=None):
        from alframework.tools.dataset_screening import filter_dataset_screening
        from alframework.tools.tools import store_current_data,compute_empirical_formula
        unique={}
        submissions=set()
        for molecule in molecules:
            key=molecule.get_metadata()['_workflow_submission']
            submissions.add(key)
            token=molecule.get_metadata()['_workflow_origin']
            if token not in self.stored_tokens:
                unique[token]=molecule
        kept,summary=filter_dataset_screening(list(unique.values()),properties,sampler_config,master_directory=master_directory)
        valid=[mol for mol in kept if self.valid_molecule(mol)]
        summary['unconverged']=sum(not bool(mol.check_convergence()) for mol in unique.values())
        summary['converged']=len(unique)-summary['unconverged']
        summary['rejected_unusable']+=len(kept)-len(valid)
        summary['kept']=len(valid)
        event=uuid.uuid4().hex
        self.db.execute('INSERT INTO storage_events VALUES (?,?,?,?,?,?,?)',
            (event,json.dumps(summary),json.dumps(sorted(submissions)),str(path),None,'prepared',time.time()))
        self.db.commit()
        if valid:
            path=Path(path)
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_name('.'+path.name+'.pending')
            if path.exists():
                raise FileExistsError(f'Refusing committed shard overwrite: {path}')
            if temporary.exists():
                temporary.unlink()  # Unpublished transaction; durable results remain in the journal.
            store_current_data(str(temporary),valid,properties)
            groups={}
            for mol in valid:
                groups.setdefault(compute_empirical_formula(mol.get_atoms().get_chemical_symbols()),[]).append(mol)
            with h5py.File(temporary,'a') as handle:
                for name,items in groups.items():
                    group=handle[name]
                    group[ROW_KEY]=np.asarray([mol.get_metadata()['_workflow_origin'] for mol in items],dtype=h5py.string_dtype())
                    group[PHASE_KEY]=np.asarray([mol.get_metadata()['_workflow_phase'] for mol in items],dtype=h5py.string_dtype())
                handle.attrs['alf_workflow_transaction']=event
                handle.flush()
            # Validate serialized values/shapes before the atomic commit.
            with h5py.File(temporary) as handle:
                from alframework.tools.workflow_io import group_valid_rows
                if sum(len(group_valid_rows(group,properties)) for group in handle.values())!=len(valid):
                    raise ValueError('Serialized H5 frame count does not match accepted labels.')
            with temporary.open('rb') as handle:
                os.fsync(handle.fileno())
            record=fingerprint(temporary); record['path']=str(path.resolve())
            self.db.execute('UPDATE storage_events SET file_record=? WHERE id=?',(json.dumps(record),event))
            self.db.commit()
            os.replace(temporary,path)
            sync_directory(path.parent)
            self.reconcile()
            scan=scan_store(self.master['h5_path'],properties)
            self.accepted=scan['frames']
            self.state['phase_counts']=scan['phase_counts']
        for key in submissions:
            self.db.execute("UPDATE tasks SET state='consumed' WHERE key=?",(key,))
        self.db.execute("UPDATE storage_events SET state='committed' WHERE id=?",(event,))
        self.db.commit()
        self.write_progress()
        self.make_checkpoints()
        return bool(valid),valid,summary

    def make_checkpoints(self):
        if self.mode=='final_training':
            return
        for count in self.checkpoints:
            if self.accepted>=count:
                freeze_snapshot(self.master['h5_path'],self.master['properties_list'],count,self.root/'checkpoints'/str(count))

    def train_only(self,app,status):
        from alframework.tools.tools import build_input_dict
        queue=self.queues['ML']
        if status['current_model_id']<0:
            if queue.get_number()==0:
                kwargs=build_input_dict(app.func,[{'ML_config':self.configs[-1]},*self.configs,status],raise_on_fail=True)
                queue.add_task(app(**kwargs))
                status['current_training_id']+=1
                self.save(status)
            result=queue.task_list[0].result()
            status['current_model_id']=int(result[1])
        queue.get_task_results()
        self.save(status)

    def unused(self,molecules):
        self.state['unused_builder_outputs']+=len(flatten_molecule_output(molecules))

    def save(self,status):
        for key in self.deliveries:
            self.db.execute("UPDATE tasks SET state='delivered' WHERE key=? AND state='ready'",(key,))
        self.deliveries.clear()
        if hasattr(self,'buffer'):
            self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('buffer',dump(self.buffer)))
        self.db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',('status',dump(status)))
        self.db.commit()
        atomic_json(self.master['status_path'],status)
        self.write_progress()

    def write_progress(self, terminal=None):
        if terminal:
            self.state['state']=terminal
        elif self.draining:
            self.state['state']='draining'
        else:
            self.state['state']='running'
        self.state.update(accepted_frames=self.accepted,target_total_frames=self.target,
                          overshoot_frames=max(0,self.accepted-self.target),
                          elapsed_seconds=self.state.get('elapsed_seconds',0.) if not hasattr(self,'prior_elapsed') else self.prior_elapsed+time.time()-self.started,
                          failure=self.failure,pause_requested=self.pause_requested)
        self.state['screening_totals']={}
        for row in self.db.execute("SELECT summary FROM storage_events WHERE state='committed'").fetchall():
            for key,value in json.loads(row['summary']).items():
                if key in {'total','kept','converged','unconverged'} or key.startswith('rejected_'):
                    self.state['screening_totals'][key]=self.state['screening_totals'].get(key,0)+int(value)
        self.state['tasks']={}
        for role in ('builder','sampler','QM','ML'):
            rows=self.db.execute('SELECT state,COUNT(*),SUM(attempts) FROM tasks WHERE role=? GROUP BY state',(role,)).fetchall()
            self.state['tasks'][role]={'states':{row[0]:row[1] for row in rows},'attempts':sum(row[2] for row in rows),
                'attempt_outcomes':{row[0]:row[1] for row in self.db.execute('SELECT a.state,COUNT(*) FROM attempts a JOIN tasks t ON a.task_key=t.key WHERE t.role=? GROUP BY a.state',(role,)).fetchall()}}
        if hasattr(self,'queues'):
            self.state['queues']={role:{'total':queue.get_number(),'running':queue.get_running_number(),
                'completed':queue.get_completed_number()} for role,queue in self.queues.items()}
        atomic_json(self.state_path,self.state)

    def finish_ready(self):
        return self.draining and all(queue.get_number()==0 for queue in self.queues.values())

    def finish(self,status):
        self.state['unused_buffered_structures']=len(self.buffer)
        self.buffer._buckets.clear()
        self.save(status)
        self.make_checkpoints()
        terminal='failed' if self.failure else 'paused' if self.pause_requested else 'completed'
        self.write_progress(terminal)
        self.close()
        return terminal

    def close(self):
        if self.closed:
            return
        self.closed=True
        for sig,previous in self.signals.items():
            signal.signal(sig,previous)
        self.db.close()
        self.lock.close()
        atexit.unregister(self.abort)

    def abort(self):
        if not self.closed:
            self.failure=self.failure or 'Driver interrupted before a clean lifecycle boundary; resume required.'
            self.write_progress('failed')
            self.close()


def create_run_control(master,configs):
    if not master.get('run_control'):
        return None
    control=RunControl.__new__(RunControl)
    try:
        control.__init__(master,configs)
        return control
    except BaseException:
        if hasattr(control,'db'):
            control.db.close()
        if hasattr(control,'lock'):
            control.lock.close()
        raise
