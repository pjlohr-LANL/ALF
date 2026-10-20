"""Local-only tasks for staged integration tests; never used by examples."""
import json
from pathlib import Path
import time

import h5py
import numpy as np
from parsl import python_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor

config=Config(executors=[ThreadPoolExecutor(label=f'alf_{role}_executor',max_threads=2)
                        for role in ('builder','sampler','QM','ML')],strategy='none',run_dir='runinfo')


@python_app(executors=['alf_sampler_executor'])
def sample(molecule_object=None,molecule_objects=None,current_model_id=0,model_path=None,sampler_config=None,properties_list=None,master_directory=None):
    assert Path(model_path.format(current_model_id)).is_dir()
    molecules=molecule_objects if molecule_objects is not None else [molecule_object]
    if sampler_config.get('gap_seeking',{}).get('enabled'):
        from alframework.samplers.alchemi_sampling import run_alchemi_sampling
        from alframework.tools.force_screen import SCHEMA,SOURCE
        from tests.helpers.fakes import FakeModel,FakeRunner
        base={'Es':.002,'Fs':0.,'Fsrms':0.,'Fsmax':0.,'Fmeanmax':1.,
              'force_screen_schema':SCHEMA,'force_screen_source':SOURCE,
              **{key:value for gap in ('dE01','dE12','dE23') for key,value in ((gap,.05),(gap+'_stdev',0.),(gap+'_model_count',4))}}
        output=run_alchemi_sampling(molecules,sampler_config,FakeModel({i:[base,{**base,'Es':.004}] for i in range(len(molecules))}),
            properties_list=properties_list,runner_factory=FakeRunner,master_directory=master_directory)
        assert len(FakeRunner.last_instance.lcm_calls)==len(molecules)
        return output
    for molecule in molecules:
        molecule.update_metadata({'sampled':True})
    return [molecules]


@python_app(executors=['alf_QM_executor'])
def qm(molecule_object,properties_list,QM_config):
    time.sleep(.01)
    events=Path(QM_config['events_dir']); events.mkdir(exist_ok=True)
    if QM_config.get('fail_first_qm') and not (events/'failure_seen').exists():
        (events/'failure_seen').write_text('injected failure')
        raise RuntimeError('Injected QM backend failure')
    results={}
    for key,schema in properties_list.items():
        if key.startswith('dE'):
            continue
        results[key]=np.zeros((len(molecule_object.get_atoms()),3)) if schema[1]=='atomic' else float(int(key[2:])*.05 if key.startswith('sE') else 1.)
    for key in properties_list:
        if key.startswith('dE'):
            results[key]=results['sE'+key[-1]]-results['sE'+key[-2]]
    molecule_object.store_results(results)
    reject=QM_config.get('reject_seed') and molecule_object.get_moleculeid()=='mol-boot-0000000001'
    molecule_object.set_converged_flag(not reject)
    return [[molecule_object]]


@python_app(executors=['alf_ML_executor'])
def train(ML_config,h5_dir,model_path,current_training_id,properties_list):
    ids=[]
    for path in sorted(Path(h5_dir).glob('*.h5')):
        with h5py.File(path) as handle:
            for group in handle.values():
                ids.extend(group['_id'].asstr()[:].tolist())
    directory=Path(model_path.format(current_training_id))
    directory.mkdir(parents=True,exist_ok=False)
    marker=directory.parent.parent/'final_failure_seen'
    if ML_config.get('fail_final_once') and 'final_training' in directory.parts and not marker.exists():
        marker.write_text('injected partial final training')
        (directory/'partial').write_text('unfinished')
        raise RuntimeError('Injected final training failure')
    for index in range(ML_config['n_models']):
        member=directory/f'model-{index:02d}'; member.mkdir()
        (member/'training_log.txt').write_text('Training complete\n')
        (member/'training_summary.json').write_text(json.dumps({'data':{'n_structures':len(ids),'n_structures_before_filtering':len(ids)},'gap_targets':{'enabled':False}}))
        for name in ('experiment_structure.pt','best_checkpoint.pt'):
            (member/name).write_bytes(b'mock artifact')
    (directory/'input_frames.json').write_text(json.dumps({'frames':len(ids),'ids':ids,'input_directory':str(h5_dir)}))
    return [True]*ML_config['n_models'],current_training_id


@python_app(executors=['alf_builder_executor'])
def builder(moleculeid,moleculeids,builder_config,sampler_config,h5_path,current_h5_id,master_directory):
    from alframework.builders.h5_replay_builder import build_h5_replay_structures
    marker=Path(master_directory)/'builder_failure_seen'
    if not marker.exists():
        marker.write_text('injected builder failure')
        raise RuntimeError('Injected builder backend failure')
    return build_h5_replay_structures(moleculeids=moleculeids,builder_config=builder_config,
        sampler_config=sampler_config,h5_path=h5_path,current_h5_id=current_h5_id,master_directory=master_directory)
