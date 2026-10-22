"""Run linear, budgeted ALF stages with exact inherited checkpoints.

Seed preparation remains explicit. Use --validate, --status, or --resume for
inspection and recovery. Every stage references an ordinary ALF master config.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from alframework.tools.run_control import positive_integer
from alframework.tools.workflow_io import (
    ExclusiveLock, atomic_json, copy_snapshot_data, fingerprint, publish_directory,
    read_json, scan_store, shards, verify, verify_snapshot, validate_reference_layout, sync_directory,
)

OPERATIONAL_QM = {'max_solve_time_seconds','capture_pyseqm_logs','pyseqm_log_dir','log_dir',
                  'scratch_dir','batch_size','pyseqm_batch_size','capture_logs','qm_log_dir'}
OUTPUTS = {
    'master': {'h5_path':'h5store/data-{:04d}.h5','model_path':'models/model-{:04d}',
               'status_path':'status.json','QM_scratch_dir':'qm_scratch','plot_dir':'plots'},
    'QM': {'pyseqm_log_dir':'pyseqm_logs','qm_log_dir':'qm_logs','log_dir':'qm_logs','scratch_dir':'qm_scratch'},
    'sampler': {'force_screen_audit_dir':'sampling/force_screen_audit','meta_dir':'sampling','metadata_dir':'sampling',
                'qm_candidate_xyz_dir':'sampling/qm_candidates','trajectory_dir':'sampling/trajectories'},
}


def source_master(path):
    path=Path(path).resolve(strict=True)
    master=read_json(path)
    directory=master.get('master_directory')
    root=path.parent if directory in (None,'pwd') else Path(directory).expanduser()
    if not root.is_absolute():
        root=path.parent/root
    root=root.resolve()
    configs={role:read_json((root/master[f'{role}_config_path']).resolve()) for role in ('builder','sampler','QM','ML')}
    return master,configs,root


def scientific_contract(master,configs,root):
    qm={key:value for key,value in configs['QM'].items() if key not in OPERATIONAL_QM}
    qm['energy_offset_eV']=qm.get('energy_offset_eV',configs['sampler'].get('energy_offset_eV',0.))
    contract={'properties':master['properties_list'],'QM_task':master['QM_task'],'QM':qm}
    topology=configs['sampler'].get('topology_check',{})
    if topology.get('enabled',False):
        from alframework.tools.molecular_topology import load_reference_topology
        reference=load_reference_topology(configs['sampler'],master_directory=str(root))
        contract['topology']={'atomic_numbers':list(reference.atomic_numbers),'bonds':[list(b) for b in reference.bonds],
                              'charge':reference.reference_charge}
    return contract


def definition(path):
    path=Path(path).expanduser().resolve(strict=True)
    config=read_json(path)
    unknown=set(config)-{'stages','final_training'}
    if unknown:
        raise ValueError(f'Unknown workflow settings: {sorted(unknown)}')
    if not isinstance(config.get('final_training',True),bool):
        raise ValueError('final_training must be boolean.')
    if not isinstance(config.get('stages'),list) or not config['stages']:
        raise ValueError('Workflow requires at least one stage.')
    names=set()
    previous=0
    contract=None
    records=[fingerprint(path)]
    resolved=[]
    for item in config['stages']:
        if set(item)-{'name','master_config','target_total_frames','checkpoints','output_paths'}:
            raise ValueError('Unknown stage settings.')
        name=item['name']
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',name) or name in names:
            raise ValueError('Stage names must be unique safe directory names.')
        names.add(name)
        target=positive_integer(item['target_total_frames'],'target_total_frames')
        if target<=previous:
            raise ValueError('Inherited stage targets must strictly increase.')
        checkpoints=item.get('checkpoints',[])
        if not isinstance(checkpoints,list) or any(not previous<positive_integer(c,'checkpoint')<=target for c in checkpoints):
            raise ValueError('Checkpoints must lie above the inherited pool and at or below the stage target.')
        previous=target
        master_path=(path.parent/item['master_config']).resolve(strict=True)
        master,configs,root=source_master(master_path)
        current=scientific_contract(master,configs,root)
        if contract is not None and current!=contract:
            raise ValueError('Inherited stages have incompatible properties, QM settings/units, or topology identities.')
        contract=current
        records.append(fingerprint(master_path))
        records.extend(fingerprint(root/master[f'{role}_config_path']) for role in configs)
        for key in ('parsl_configuration','parsl_debug_configuration'):
            if key in master:
                module=master[key].split('.')[0]
                if (root/(module+'.py')).is_file():
                    records.append(fingerprint(root/(module+'.py')))
        resolved.append({**item,'master_config':str(master_path),'source_root':str(root)})
    return {'schema':'alf.workflow.v1','configuration':records,'definition':config,'stages':resolved,
            'final_training':config.get('final_training',True),'label_contract':contract}


def resolve_paths(config,root):
    result=copy.deepcopy(config)
    for key,value in result.items():
        if isinstance(value,dict):
            result[key]=resolve_paths(value,root)
        elif isinstance(value,list):
            result[key]=[resolve_paths(item,root) if isinstance(item,dict) else item for item in value]
        elif isinstance(value,str) and value and (key.endswith('_path') or key.endswith('_dir')):
            result[key]=str((root/Path(value).expanduser()).resolve())
    return result


def prepare_workflow(config_path,run_dir):
    plan=definition(config_path)
    destination=Path(run_dir).expanduser().resolve()
    with publish_directory(destination) as temporary:
        for index,stage in enumerate(plan['stages']):
            relative=f'stages/{index:02d}-{stage["name"]}'
            local=temporary/relative
            final=destination/relative
            local.mkdir(parents=True)
            master,configs,root=source_master(stage['master_config'])
            for role,config in configs.items():
                configs[role]=resolve_paths(config,root)
                for key,value in OUTPUTS.get(role,{}).items():
                    if key in config:
                        configs[role][key]=str(final/value)
                master[f'{role}_config_path']=str(final/f'{role}_config.json')
            for key,value in OUTPUTS['master'].items():
                if key in master or key in {'h5_path','model_path','status_path'}:
                    master[key]=str(final/value)
            for qualified,value in stage.get('output_paths',{}).items():
                if qualified in {'master.h5_path','master.model_path','master.status_path','master.master_directory','master.run_control'}:
                    raise ValueError('Core data/model/status layout is managed by the workflow runner.')
                role,key=qualified.split('.',1)
                if role not in {*configs,'master'} or not isinstance(value,str) or Path(value).is_absolute() or '..' in Path(value).parts:
                    raise ValueError('output_paths needs role.key and a safe run-relative path.')
                target_config=master if role=='master' else configs[role]
                parts=key.split('.')
                for part in parts[:-1]:
                    target_config=target_config[part]
                target_config[parts[-1]]=str(final/value)
            master['master_directory']=str(final)
            control_options=master.pop('run_control',{})
            for key in ('h5_dir','model_dir','status_dir'):
                master.pop(key,None)
            master['run_control']={**control_options,'target_total_frames':stage['target_total_frames'],'checkpoints':stage.get('checkpoints',[]),
                                   'stage_name':stage['name'],'master_path':str(final/'master_config.json'),
                                   'output_fields':[role+'.'+key for role,fields in OUTPUTS.items() for key in fields]+list(stage.get('output_paths',{}))}
            if index:
                master['bootstrap_set']=0
            for role,config in configs.items():
                atomic_json(local/f'{role}_config.json',config)
            atomic_json(local/'master_config.json',master)
            for module_key in ('parsl_configuration','parsl_debug_configuration'):
                if module_key in master:
                    filename=master[module_key].split('.')[0]+'.py'
                    if (root/filename).is_file():
                        shutil.copy2(root/filename,local/filename)
            (local/'h5store').mkdir()
            (local/'models').mkdir()
            stage['run_directory']=str(final)
            if index==0:
                original,_,original_root=source_master(stage['master_config'])
                source_pattern=str((original_root/original['h5_path']).resolve())
                status=original_root/original['status_path']
                boundary=read_json(status)['current_h5_id'] if status.exists() else None
                existing=shards(source_pattern,boundary)
                if existing:
                    selection=scan_store(source_pattern,original['properties_list'],closed_before=boundary)
                    if selection['frames']:
                        if 'topology' in plan['label_contract']:
                            validate_reference_layout(selection,plan['label_contract']['topology']['atomic_numbers'])
                        selection['artifacts']=copy_snapshot_data(selection,local,final)
                        atomic_json(local/'inheritance.json',selection)
                elif master.get('bootstrap_set',0)>0:
                    bootstrap=configs['builder'].get('bootstrap_h5_path')
                    if bootstrap and not Path(bootstrap.format(0)).is_file():
                        raise FileNotFoundError('Generate external seed geometries before starting the workflow.')
        for record in plan['configuration']:
            verify(record)
        atomic_json(temporary/'workflow.json',plan)
        plan_record=fingerprint(temporary/'workflow.json'); plan_record['path']=str(destination/'workflow.json')
        atomic_json(temporary/'workflow_status.json',{'state':'prepared','stage_states':{},'final_training':None,'workflow_fingerprint':plan_record})
    return plan


def inherit_checkpoint(checkpoint,stage):
    manifest=verify_snapshot(checkpoint)
    root=Path(stage['run_directory'])
    master=read_json(root/'master_config.json')
    sampler=read_json(master['sampler_config_path'])
    if sampler.get('topology_check',{}).get('enabled',False):
        from alframework.tools.molecular_topology import load_reference_topology
        reference=load_reference_topology(sampler,master_directory=str(root))
        validate_reference_layout(manifest,reference.atomic_numbers)
    for artifact in manifest['artifacts']:
        verify(artifact)
        destination=root/artifact['relative_path']
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if fingerprint(destination)['sha256']!=artifact['sha256']:
                raise RuntimeError('Handoff destination contains conflicting data; explicit repair required.')
            continue
        temporary=destination.with_name('.'+destination.name+'.pending')
        shutil.copy2(artifact['path'],temporary)
        if fingerprint(temporary)['sha256']!=artifact['sha256']:
            raise RuntimeError('Handoff copy fingerprint mismatch.')
        with temporary.open('rb') as saved:
            os.fsync(saved.fileno())
        os.replace(temporary,destination)
        sync_directory(destination.parent)
    unexpected={p.resolve() for p in (root/'h5store').glob('*.h5')}-{(root/a['relative_path']).resolve() for a in manifest['artifacts']}
    if unexpected:
        raise RuntimeError('Fresh handoff contains additional training shards.')
    atomic_json(root/'inheritance.json',{'checkpoint':str(Path(checkpoint).resolve()),'frames':manifest['frames'],
                                       'artifacts':manifest['artifacts'],'label_contract_inherited':True})


def verify_final_model(master,expected_frames):
    status=read_json(master['status_path'])
    index=status['current_model_id']
    if index<0:
        raise RuntimeError('Final training did not promote a completed ensemble.')
    directory=Path(master['model_path'].format(index))
    if not directory.is_dir():
        raise RuntimeError('Final model directory is missing.')
    ml=read_json(master['ML_config_path'])
    if 'n_models' in ml:
        members=[directory/f'model-{i:02d}' for i in range(int(ml['n_models']))]
        if any(not member.is_dir() for member in members):
            raise RuntimeError('Final ensemble members are incomplete.')
        if '.hippynn_interface.' in master['ML_task'] or '.excited_state_hippynn_interface.' in master['ML_task']:
            for member in members:
                if 'Training complete' not in (member/'training_log.txt').read_text():
                    raise RuntimeError(f'Final member training incomplete: {member}')
                if (member/'training_summary.json').exists():
                    data=read_json(member/'training_summary.json')['data']
                    if data.get('n_structures_before_filtering',data['n_structures'])!=expected_frames:
                        raise RuntimeError('Final model did not read the exact selected dataset.')
    return {'model_index':index,'directory':str(directory),'input_frames':expected_frames,
            'artifacts':[fingerprint(path) for path in sorted(directory.rglob('*')) if path.is_file()]}


class WorkflowRunner:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.plan=read_json(self.root/'workflow.json')
        self.state=read_json(self.root/'workflow_status.json')
        verify(self.state['workflow_fingerprint'])
        self.child=None
        self.pause=False
        self.training=False

    def save(self):
        atomic_json(self.root/'workflow_status.json',self.state)

    def interrupt(self,*unused):
        self.pause=True
        if self.child is not None and self.child.poll() is None and not self.training:
            # Numerical runtimes can replace SIGINT after the lifecycle is
            # initialized. SIGTERM uses RunControl's explicit drain handler.
            self.child.send_signal(signal.SIGTERM)

    def execute(self,master, *, training=False):
        self.training=training
        root=Path(master).parent
        environment={**os.environ,'PYTHONDONTWRITEBYTECODE':'1',
                     'PYTHONPATH':str(root)+os.pathsep+str(Path(__file__).resolve().parents[2])+os.pathsep+os.environ.get('PYTHONPATH','')}
        command=[sys.executable,'-u','-m','alframework','--master',str(master)]
        if training:
            command.append('--test_ml')
        with (root/'driver.log').open('ab') as log:
            self.child=subprocess.Popen(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=self.child.wait()
        self.child=None
        self.training=False
        return code

    def final_training(self,stage,checkpoint):
        manifest=verify_snapshot(checkpoint)
        for artifact in manifest['artifacts']:
            verify(artifact)
        completed=self.state.get('final_training')
        if completed and completed.get('state')=='completed':
            for record in completed['artifacts']+completed['inputs']:
                verify(record)
            return completed
        root=self.root/'final_training'
        root.mkdir(exist_ok=True)
        master_path=root/'master_config.json'
        if not master_path.exists():
            master=read_json(Path(stage['run_directory'])/'master_config.json')
            master.pop('run_control',None)
            master.pop('parsl_debug_configuration',None)
            master['run_control']={'mode':'final_training','stage_name':'final_training',
                                   'target_total_frames':manifest['frames'],'master_path':str(master_path)}
            master.update(master_directory=str(root),h5_path=str(checkpoint/'h5store/data-{:04d}.h5'),
                          model_path=str(root/'models/model-{:04d}'),status_path=str(root/'status.json'))
            atomic_json(master_path,master)
        master=read_json(master_path)
        pin=root/'inputs.json'
        if pin.exists():
            for record in read_json(pin)['configuration']:
                verify(record)
        else:
            atomic_json(pin,{'configuration':[fingerprint(master_path),fingerprint(master['ML_config_path'])],
                             'inputs':manifest['artifacts']})
        if Path(master['status_path']).exists():
            status=read_json(master['status_path'])
            progress=root/'control/progress.json'
            if status['current_model_id']>=0 and progress.exists() and read_json(progress)['state']=='completed':
                result=verify_final_model(master,manifest['frames'])
                result['inputs']=manifest['artifacts']
                return result
        from alframework.tools.tools import find_empty_directory
        index=find_empty_directory(master['model_path'])
        # Explicit status prevents legacy directory-based inference from promoting partial attempts.
        atomic_json(master['status_path'],{'current_model_id':-1,'current_training_id':index,
            'current_h5_id':len(manifest['artifacts']),'current_molecule_id':0,
            **{f'lifetime_failed_{role}_tasks':0 for role in ('builder','sampler','QM','ML')}})
        self.state['final_training']={'state':'running','attempt_model_index':index,'input_frames':manifest['frames'],
                                     'inputs':manifest['artifacts']}
        self.save()
        # Resolve the copied stage's Parsl module from that directory without sharing any DFK.
        stage_root=Path(stage['run_directory'])
        source=read_json(stage_root/'master_config.json')
        filename=source['parsl_configuration'].split('.')[0]+'.py'
        if (stage_root/filename).is_file():
            shutil.copy2(stage_root/filename,root/filename)
        if self.execute(master_path,training=True)!=0:
            raise RuntimeError('Final training failed; incomplete artifacts are preserved for resume.')
        for artifact in manifest['artifacts']:
            verify(artifact)
        result=verify_final_model(master,manifest['frames'])
        result['inputs']=manifest['artifacts']
        return result

    def run(self):
        with ExclusiveLock(self.root/'workflow.lock'):
            self.state.pop('failure',None)
            for record in self.plan['configuration']:
                verify(record)
            previous={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
            for sig in previous:
                signal.signal(sig,self.interrupt)
            try:
                checkpoint=None
                for stage in self.plan['stages']:
                    root=Path(stage['run_directory'])
                    progress=root/'control/progress.json'
                    if progress.exists() and read_json(progress)['state']=='completed':
                        for record in read_json(progress)['configuration']:
                            verify(record)
                        checkpoint=root/'checkpoints'/str(stage['target_total_frames'])
                        verify_snapshot(checkpoint)
                        self.state['stage_states'][stage['name']]='completed'
                        continue
                    if self.pause:
                        break
                    if checkpoint is not None and not progress.exists():
                        inherit_checkpoint(checkpoint,stage)
                    self.state.update(state='running',active_stage=stage['name'])
                    self.state['stage_states'][stage['name']]='running'
                    self.save()
                    code=self.execute(root/'master_config.json')
                    outcome=read_json(progress)['state'] if progress.exists() else 'failed'
                    self.state['stage_states'][stage['name']]=outcome
                    if self.pause or outcome=='paused':
                        self.state['state']='paused'
                        self.save()
                        return 130
                    if code!=0 or outcome!='completed':
                        raise RuntimeError(f'Stage {stage["name"]} failed; inspect {root / "driver.log"}.')
                    checkpoint=root/'checkpoints'/str(stage['target_total_frames'])
                    verify_snapshot(checkpoint)
                    self.save()
                if self.pause:
                    self.state['state']='paused'
                    self.save()
                    return 130
                if self.plan['final_training']:
                    self.state['final_training']={**self.final_training(self.plan['stages'][-1],checkpoint),'state':'completed'}
                self.state['state']='paused' if self.pause else 'completed'
                self.save()
                return 130 if self.pause else 0
            except BaseException as exc:
                self.state.setdefault('failure_history',[]).append({'time':time.time(),'stage':self.state.get('active_stage'),
                    'error':f'{type(exc).__name__}: {exc}'})
                self.state.update(state='failed',failure=f'{type(exc).__name__}: {exc}')
                self.save()
                raise
            finally:
                for sig,handler in previous.items():
                    signal.signal(sig,handler)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--run-dir',required=True,type=Path)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--validate',action='store_true')
    mode.add_argument('--status',action='store_true')
    mode.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if args.status:
        state=read_json(args.run_dir/'workflow_status.json')
        stages=read_json(args.run_dir/'workflow.json')['stages']
        live={stage['name']:read_json(Path(stage['run_directory'])/'control/progress.json')
              for stage in stages if (Path(stage['run_directory'])/'control/progress.json').exists()}
        final=args.run_dir/'final_training/control/progress.json'
        print(json.dumps({'workflow':state,'stages':live,'final_training':read_json(final) if final.exists() else None},indent=2))
        return 0
    if args.config is None:
        parser.error('--config is required unless --status is used.')
    if args.validate:
        plan=definition(args.config)
        print(json.dumps({'valid':True,'stages':[{k:s[k] for k in ('name','target_total_frames')} for s in plan['stages']],
                          'final_training':plan['final_training']},indent=2))
        return 0
    if args.resume:
        saved=read_json(args.run_dir/'workflow.json')
        if fingerprint(args.config)!=saved['configuration'][0]:
            raise RuntimeError('Workflow definition changed; resume requires the pinned original definition.')
    else:
        prepare_workflow(args.config,args.run_dir)
    return WorkflowRunner(args.run_dir).run()


if __name__=='__main__':
    raise SystemExit(main())
