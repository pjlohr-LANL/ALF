#!/usr/bin/env python3
"""Label the bundled CPP9 pool, train four E/F models, and stop."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
from pathlib import Path
import signal
import subprocess
import threading

from cpp_trial.workflow import (PROPERTIES, atomic_json, file_hash, input_frames,
                                label_all, validate_reference)
from cpp_trial.resources import build_config,validate_resources

ROOT=Path(__file__).resolve().parent


def dependency_versions():
    variants={
        'gpu4pyscf':('gpu4pyscf-cuda12x','gpu4pyscf-cuda11x','gpu4pyscf-cuda13x','gpu4pyscf'),
        'cupy':('cupy-cuda12x','cupy-cuda11x','cupy-cuda13x','cupy'),
        **{name:(name,) for name in ('pyscf','parsl','hippynn','torch','h5py','numpy','ase')},
    }
    result={}
    for name,choices in variants.items():
        for distribution in choices:
            try:
                result[name]={'distribution':distribution,'version':importlib.metadata.version(distribution)}
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        else:
            raise ImportError(f'Missing {name}; install the compatible GPU4PySCF environment described in README.md.')
    return result


def alf_identity():
    """Support both a source checkout and an installed ALF without Git metadata."""
    import alframework
    root=Path(alframework.__file__).resolve().parent
    commit=None
    if (root.parent/'.git').exists():
        try:
            commit=subprocess.check_output(['git','-C',str(root.parent),'rev-parse','HEAD'],text=True).strip()
        except (OSError,subprocess.CalledProcessError):
            pass
    try:
        version=importlib.metadata.version('alf')
    except importlib.metadata.PackageNotFoundError:
        version=None
    modules=['qm_interfaces/gpu4pyscf_interface.py','ml_interfaces/excited_state_hippynn_interface.py',
             'tools/tools.py','tools/atom_order.py','tools/molecules_class.py','tools/excited_state_tools.py',
             'builders/h5_replay_builder.py']
    return {'commit':commit,'version':version,'source_sha256':{name:file_hash(root/name) for name in modules}}
def validate_training(directory, config):
    from alframework.ml_interfaces.excited_state_hippynn_interface import (
        load_excited_state_h5_arrays, filter_excited_state_training_arrays,
        validate_filtered_split_capacity, validate_excited_state_training_config,
    )
    states = validate_excited_state_training_config(config, PROPERTIES)
    arrays, summary = load_excited_state_h5_arrays(
        str(Path(directory) / 'h5store'), PROPERTIES, configured_n_atoms=90,
        configured_possible_species=[0, 1, 6])
    arrays, filters = filter_excited_state_training_arrays(
        arrays, states, coordinates_key='coordinates',
        dist_soft_min=config['network_params']['dist_soft_min'],
        remove_high_energy_cut=config.get('remove_high_energy_cut'),
        remove_high_energy_std=config.get('remove_high_energy_std'),
        remove_high_forces_cut=config.get('remove_high_forces_cut'),
        remove_high_forces_std=config.get('remove_high_forces_std'))
    count = len(arrays['coordinates'])
    validate_filtered_split_capacity(count, test_size=config['test_size'], valid_size=config['valid_size'])
    return {'before_filtering': summary['n_structures'], 'after_filtering': count, 'filters': filters}


def train_ensemble(directory,ml,reference,settings,stop):
    """Fresh resources after labeling, including on a single multi-GPU workstation."""
    import parsl
    from alframework.ml_interfaces.excited_state_hippynn_interface import train_excited_state_HIPPYNN_ensemble_task
    capacity=validate_training(directory,ml)
    record_path=directory/'training.json'
    if record_path.exists() and json.loads(record_path.read_text()).get('status')=='complete':
        print('Ensemble already complete; no training resubmitted.',flush=True)
        return 0
    model_root=directory/'models'
    existing=[int(p.name.split('-')[-1]) for p in model_root.glob('model-*') if p.is_dir()]
    training_id=max(existing,default=-1)+1
    model_dir=model_root/f'model-{training_id:04d}'
    model_dir.mkdir(parents=True)
    atomic_json(model_dir/'energy_reference.json',reference)
    atomic_json(record_path,{'status':'running','training_id':training_id,'data':capacity})
    parsl.load(build_config(ROOT,directory,settings,roles=('ML',)))
    try:
        future=train_excited_state_HIPPYNN_ensemble_task(
            ml,str(directory/'h5store'),str(model_root/'model-{:04d}'),
            training_id,settings['gpus_per_node'],PROPERTIES)
        while not future.done():
            if stop.wait(1):
                raise InterruptedError('Stopped during training; previous attempt is preserved.')
        completed,returned_id=future.result()
        if returned_id!=training_id or len(completed)!=4:
            raise RuntimeError('Unexpected ensemble completion contract.')
        for member in model_dir.glob('model-*'):
            atomic_json(member/'energy_reference.json',reference)
        atomic_json(record_path,{'status':'complete' if all(completed) else 'failed',
            'training_id':training_id,'members_complete':completed,'data':capacity})
        return 0 if all(completed) else 3
    except Exception as exc:
        atomic_json(record_path,{'status':'interrupted' if isinstance(exc,InterruptedError) else 'failed',
                                'training_id':training_id,'error':str(exc),'data':capacity})
        raise
    finally:
        parsl.dfk().cleanup()
        parsl.clear()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-h5',type=Path,default=ROOT/'inputs/bootstrap.h5')
    parser.add_argument('--run-dir',type=Path,default=ROOT/'runs/trial_001')
    parser.add_argument('--resources',type=Path,default=ROOT/'resources.json')
    parser.add_argument('--check',action='store_true',help='Validate without starting Parsl or writing run outputs.')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--limit',type=int,help='Use the first N input rows in a separate smoke directory.')
    parser.add_argument('--label-only',action='store_true')
    args=parser.parse_args(argv)
    directory=args.run_dir.expanduser().resolve()
    source=args.input_h5.expanduser().resolve()
    if source.is_relative_to(directory):
        raise ValueError('Input dataset must remain outside the run output directory.')
    qm=json.loads((ROOT/'QM_config.json').read_text())
    ml=json.loads((ROOT/'hippynn_config.json').read_text())
    if ml['n_models']!=4:
        raise ValueError('The CPP9 example requires a four-member ensemble.')
    settings=validate_resources(json.loads(args.resources.expanduser().read_text()))
    qm['num_threads']=settings['qm_cores_per_worker']
    reference,numbers=validate_reference(ROOT/'reference',qm)
    frames=input_frames(source,numbers,args.limit)
    if source==ROOT/'inputs/bootstrap.h5':
        expected=json.loads((ROOT/'inputs/provenance.json').read_text())['source_bootstrap_sha256']
        if file_hash(source)!=expected:
            raise ValueError('Bundled input fingerprint changed; use --input-h5 for an independent input pool.')
    from cpp_trial.qm import validate_gpu4pyscf_config,validate_gpu4pyscf_properties
    validate_gpu4pyscf_config(qm)
    validate_gpu4pyscf_properties(PROPERTIES,nroots=3,compute_nacr=True)
    from alframework.ml_interfaces.excited_state_hippynn_interface import validate_excited_state_training_config
    validate_excited_state_training_config(ml,PROPERTIES)
    versions=dependency_versions()
    request={'input_path':str(source),'input_sha256':file_hash(source),'frames':len(frames),
        'QM_config':qm,'ML_config':ml,'resources':settings,'reference':reference,'versions':versions,
        'ALF':alf_identity(),'local_code_sha256':{str(path.relative_to(ROOT)):file_hash(path) for path in
            [ROOT/'run_trial.py',*sorted((ROOT/'cpp_trial').glob('*.py'))]}}
    print(json.dumps({'input':str(source),'frames':len(frames),'energy_reference_eV':reference['energy_eV'],
                      'backend':settings['backend'],'versions':versions},indent=2),flush=True)
    if args.check:
        print('Check passed. No jobs submitted and no run outputs written.')
        return 0
    if not args.resume and directory.exists():
        raise FileExistsError('Run directory exists; use --resume or a new --run-dir.')
    if args.resume and not (directory/'request.json').exists():
        raise FileNotFoundError('Cannot resume without request.json.')
    directory.parent.mkdir(parents=True,exist_ok=True)
    with (directory.parent/(directory.name+'.driver.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stop=threading.Event()
        previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
        for sig in previous:
            signal.signal(sig,lambda signum,frame:stop.set())
        import parsl
        from cpp_trial.qm import gpu4pyscf_excited_state_task
        try:
            parsl.load(build_config(ROOT,directory,settings,roles=('QM',)))
            try:
                max_pending=settings['qm_workers_per_node']*(settings['QM']['max_blocks'] if settings['backend']=='slurm' else 1)
                progress=label_all(frames=frames,directory=directory,request=request,reference=reference,
                    resume=args.resume,allow_created=not args.resume,max_pending=max_pending,
                    stop=stop,submit=lambda molecule:gpu4pyscf_excited_state_task(
                        molecule,qm,PROPERTIES,gpus_per_node=settings['gpus_per_node']))
            finally:
                parsl.dfk().cleanup()
                parsl.clear()
            if args.label_only:
                return 0
            if stop.is_set():
                raise InterruptedError('Stopped before training; resume to continue.')
            if progress['accepted']==0:
                atomic_json(directory/'training.json',{'status':'not_started','reason':'No accepted labels.'})
                return 2
            try:
                validate_training(directory,ml)
            except (ValueError,KeyError) as exc:
                atomic_json(directory/'training.json',{'status':'not_started','reason':str(exc)})
                print(f'Training was not started: {exc}',flush=True)
                return 2
            return train_ensemble(directory,ml,reference,settings,stop)
        finally:
            for sig,handler in previous.items():
                signal.signal(sig,handler)


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except InterruptedError as exc:
        print(str(exc),flush=True)
        raise SystemExit(130)
