"""Portable finite CPP9 labeling/training, using CPU mocks and reference optics."""
from concurrent.futures import Future
import json
from pathlib import Path
import shutil
import threading

import h5py
import numpy as np
import pytest
from ase.io import read
pytest.importorskip('pyscf')

import sys
EXAMPLE=Path(__file__).resolve().parents[1]/'examples/cpp9_gpu4pyscf'
sys.path.insert(0,str(EXAMPLE))
from cpp_trial import qm
from cpp_trial import workflow as w
from cpp_trial.resources import build_config,validate_resources,worker_init
import run_trial as runner
from run_trial import ROOT, validate_training
sys.path.remove(str(EXAMPLE))


@pytest.fixture
def source(tmp_path):
    atoms = read(ROOT / 'reference/optimized.xyz')
    path = tmp_path / 'source.h5'
    with h5py.File(path, 'w') as handle:
        group = handle.create_group('cpp9')
        group['species'] = atoms.numbers
        group['coordinates'] = np.stack([atoms.positions + i * 0.001 for i in range(12)])
        group['_id'] = np.array([f'source-{i}'.encode() for i in range(12)])
        group['topology_atom_ids'] = np.arange(90)
    return path, atoms


@pytest.fixture
def settings():
    config = json.loads((ROOT / 'QM_config.json').read_text())
    reference, _ = w.validate_reference(ROOT / 'reference', config)
    return config, reference


def successful(molecule, reference):
    energies = np.arange(4, dtype=np.float64) + reference['energy_eV'] + 0.1234567
    molecule.store_results({
        **{f'sE{i}': float(energies[i] - reference['energy_eV']) for i in range(4)},
        **{f'F{i}': np.full((90, 3), i * 0.001) for i in range(4)},
        'absolute_state_energies': energies,
        'transition_dipoles': np.full((3, 3), 0.2),
        'oscillator_strengths': np.array([0.0, 0.1, 0.2]),
        'nacr': np.full((3, 90, 3), 0.01),
    })
    molecule.update_metadata({'nac_pairs': w.PAIRS.tolist(), 'qm_elapsed_seconds': 10.0,
                              'qm_scf_seconds': 3.0})
    molecule.set_converged_flag(True)
    return molecule


def completed(value=None, error=None):
    future = Future()
    if error is not None:
        future.set_exception(error)
    else:
        future.set_result(value)
    return future


def run(source, settings, tmp_path, submit, **kwargs):
    path, atoms = source
    _, reference = settings
    frames = w.input_frames(path, atoms.numbers)
    return w.label_all(frames=frames, submit=submit, directory=tmp_path / 'run',
                       request={'input_sha256': w.file_hash(path)}, reference=reference,
                       **kwargs)


def test_reference_is_benchmark_optimized_s0(settings):
    config, reference = settings
    assert reference['energy_hartree'] == -2077.702199766316
    assert reference['energy_eV'] == pytest.approx(-56537.15659244438, abs=1e-8)
    assert config['energy_offset_eV'] == reference['energy_eV']


def test_reference_rejects_changed_method_or_hash(settings, tmp_path):
    config, _ = settings
    with pytest.raises(ValueError, match='differs'):
        w.validate_reference(ROOT / 'reference', config | {'xc': 'b3lyp'})
    snapshot = tmp_path / 'reference'
    shutil.copytree(ROOT / 'reference', snapshot)
    with (snapshot / 'optimized.xyz').open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        w.validate_reference(snapshot, config)


def test_mixed_failures_exhaust_input_and_resume_without_duplicates(source, settings, tmp_path):
    _, reference = settings
    submitted = []
    def submit(molecule):
        index = molecule.get_metadata()['source_global_index']
        submitted.append(index)
        if index in (2, 11):
            molecule.set_converged_flag(False)
            molecule.update_metadata({'qm_failed_stage': 'nac', 'qm_error': 'failed'})
        else:
            successful(molecule, reference)
        return completed(molecule)
    result = run(source, settings, tmp_path, submit)
    assert result['completed'] == 12 and result['accepted'] == 10 and result['rejected'] == 2
    assert sorted(submitted) == list(range(12))
    assert result['failure_stages'] == {'nac': 2}
    assert len(list((tmp_path / 'run/h5store').glob('*.h5'))) == 10
    resumed = run(source, settings, tmp_path, lambda m: pytest.fail('resubmitted'), resume=True)
    assert resumed['completed'] == 12
    with h5py.File(tmp_path / 'run/h5store/data-0000000000.h5') as h:
        g = next(iter(h.values()))
        assert g['nacr'].shape == (1, 3, 90, 3)
        assert g['transition_dipoles'].shape == (1, 3, 3)
        assert g['oscillator_strengths'].shape == (1, 3)
        assert g['absolute_state_energies'].shape == (1, 4)
        np.testing.assert_array_equal(g['nac_pairs'], w.PAIRS)
        assert g.attrs['source_id'] == 'source-0'
        assert json.loads(h.attrs['units_json'])['transition_dipoles'] == 'e*bohr'
    ml = json.loads((ROOT / 'hippynn_config.json').read_text())
    assert validate_training(tmp_path / 'run', ml)['after_filtering'] == 10


def test_all_failed_does_not_write_any_shard(source, settings, tmp_path):
    def submit(molecule):
        molecule.set_converged_flag(False)
        molecule.update_metadata({'qm_failed_stage': 'scf', 'qm_error': 'unconverged'})
        return completed(molecule)
    result = run(source, settings, tmp_path, submit)
    assert result['completed'] == 12 and result['accepted'] == 0
    assert not list((tmp_path / 'run/h5store').glob('*.h5'))


@pytest.mark.parametrize('key,bad', [('nacr', np.zeros((2, 90, 3))),
                                    ('transition_dipoles', np.full((3, 3), np.nan)),
                                    ('oscillator_strengths', np.array([0, -1, 0])),
                                    ('F2', np.full((90, 3), np.inf))])
def test_incomplete_frame_is_rejected_whole(source, settings, tmp_path, key, bad):
    _, reference = settings
    def submit(molecule):
        successful(molecule, reference)
        molecule.get_results()[key] = bad
        return completed(molecule)
    result = run(source, settings, tmp_path, submit)
    assert result['accepted'] == 0 and result['rejected'] == 12
    assert not list((tmp_path / 'run/h5store').glob('*.h5'))


def test_crash_after_shard_publish_is_recovered(source, settings, tmp_path, monkeypatch):
    _, reference = settings
    original = w.atomic_json
    def crash(path, value):
        if Path(path).parent.name == 'outcomes':
            raise OSError('simulated crash')
        original(path, value)
    monkeypatch.setattr(w, 'atomic_json', crash)
    with pytest.raises(OSError):
        run(source, settings, tmp_path, lambda m: completed(successful(m, reference)))
    published = list((tmp_path / 'run/h5store').glob('*.h5'))
    assert len(published) == 1
    recovered_index = int(published[0].stem.split('-')[-1])
    monkeypatch.setattr(w, 'atomic_json', original)
    submitted = []
    def submit(molecule):
        submitted.append(molecule.get_metadata()['source_global_index'])
        return completed(successful(molecule, reference))
    result = run(source, settings, tmp_path, submit, resume=True)
    assert result['accepted'] == 12 and recovered_index not in submitted


def test_stop_preserves_published_outcomes(source, settings, tmp_path):
    _, reference = settings
    stop = threading.Event()
    def submit(molecule):
        stop.set()
        return completed(successful(molecule, reference))
    with pytest.raises(InterruptedError):
        run(source, settings, tmp_path, submit, stop=stop, max_pending=1)
    assert len(list((tmp_path / 'run/outcomes').glob('*.json'))) == 1
    result = run(source, settings, tmp_path,
                 lambda m: completed(successful(m, reference)), resume=True)
    assert result['accepted'] == 12


def test_systematic_error_stops_without_rejecting_dataset(source, settings, tmp_path):
    def submit(molecule):
        molecule.set_converged_flag(False)
        molecule.update_metadata({'qm_failed_stage': 'dependency', 'qm_error': 'CUDA missing'})
        return completed(molecule)
    with pytest.raises(RuntimeError, match='Systematic'):
        run(source, settings, tmp_path, submit, max_pending=1)
    assert (tmp_path / 'run/fatal_error.json').exists()
    assert not list((tmp_path / 'run/outcomes').glob('*.json'))


def test_executor_failure_is_resumable_not_a_bad_frame(source, settings, tmp_path):
    with pytest.raises(RuntimeError, match='executor failed'):
        run(source, settings, tmp_path, lambda m: completed(error=RuntimeError('worker lost')))
    assert not list((tmp_path / 'run/outcomes').glob('*.json'))


def test_resume_rejects_changed_input(source, settings, tmp_path):
    _, reference = settings
    run(source, settings, tmp_path, lambda m: completed(successful(m, reference)))
    with h5py.File(source[0], 'a') as h:
        h['cpp9/coordinates'][0, 0, 0] += 0.01
    with pytest.raises(ValueError, match='do not match'):
        run(source, settings, tmp_path, lambda m: pytest.fail('submitted'), resume=True)


def test_input_duplicate_ids_and_permuted_topology_rejected(source):
    path, atoms = source
    with h5py.File(path, 'a') as h:
        h['cpp9/_id'][1] = h['cpp9/_id'][0]
    with pytest.raises(ValueError, match='unique'):
        w.input_frames(path, atoms.numbers)
    with h5py.File(path, 'a') as h:
        h['cpp9/_id'][1] = b'source-1'
        h['cpp9/topology_atom_ids'][0] = 1
    with pytest.raises(ValueError, match='canonical'):
        w.input_frames(path, atoms.numbers)


def test_qm_shift_preserves_gaps_forces_and_optics(source, settings, monkeypatch):
    path, atoms = source
    config, reference = settings
    molecule = w.build_molecule(w.input_frames(path, atoms.numbers)[0])
    absolute = reference['energy_eV'] + np.array([0.000012345, 3.7, 4.2, 4.4])
    forces = np.arange(4 * 90 * 3, dtype=np.float64).reshape(4, 90, 3) * 1e-5
    dipoles, strengths = np.ones((3, 3)), np.array([0.1, 0.2, 0.3])
    monkeypatch.setattr(qm, 'select_gpu4pyscf_device', lambda n: {
        'index': 0, 'label': 'cuda:0', 'visible_count': 1, 'worker_rank': 0,
        'cuda_visible_devices': '0'})
    monkeypatch.setattr(qm, 'run_gpu4pyscf_states', lambda *a, **k: (
        absolute, forces, np.zeros((3, 90, 3)), w.PAIRS,
        {'_transition_dipoles': dipoles, '_oscillator_strengths': strengths}))
    labeled = qm.label_gpu4pyscf_excited_state_molecule(
        molecule, QM_config=config, properties_list=w.PROPERTIES, gpus_per_node=4)
    w.validate_labels(labeled, reference)
    r = labeled.get_results()
    np.testing.assert_array_equal(r['absolute_state_energies'], absolute)
    np.testing.assert_array_equal(r['F2'], forces[2])
    np.testing.assert_array_equal(r['transition_dipoles'], dipoles)
    shifted = np.array([r[f'sE{i}'] for i in range(4)])
    np.testing.assert_allclose(shifted[1:] - shifted[0], absolute[1:] - absolute[0], atol=1e-10)
    assert abs(shifted[0] - 0.000012345) < 1e-8
    assert '_transition_dipoles' not in labeled.get_metadata()
    r['sE0'] -= reference['energy_eV']
    with pytest.raises(ValueError, match='exactly once'):
        w.validate_labels(labeled, reference)


def test_trainer_ignores_auxiliary_arrays(source, settings, tmp_path):
    _, reference = settings
    run(source, settings, tmp_path, lambda m: completed(successful(m, reference)))
    from alframework.ml_interfaces.excited_state_hippynn_interface import (
        load_excited_state_h5_arrays, validate_excited_state_training_properties)
    arrays, _ = load_excited_state_h5_arrays(str(tmp_path / 'run/h5store'), w.PROPERTIES)
    assert set(arrays) == {'coordinates', 'species', *[f'sE{i}' for i in range(4)], *[f'F{i}' for i in range(4)]}
    assert len(validate_excited_state_training_properties(w.PROPERTIES)) == 4
    assert arrays['sE0'].dtype == np.float32
    assert arrays['sE0'][0] == pytest.approx(0.1234567, abs=1e-7)


@pytest.mark.parametrize('backend',['local','slurm'])
def test_portable_resources_do_not_start_allocations(tmp_path,monkeypatch,backend):
    from parsl.executors.high_throughput import executor
    from parsl.providers import SlurmProvider
    monkeypatch.setattr(executor,'get_all_addresses',lambda:{'127.0.0.1'})
    monkeypatch.setattr(SlurmProvider,'execute_wait',lambda *args,**kwargs:(0,'',''))
    settings=json.loads((ROOT/'resources.json').read_text())
    settings.update(backend=backend,account='test-account')
    for role in ('QM','ML'):
        settings[role]['partition']='test-partition'
    config=build_config(ROOT,tmp_path,settings)
    qm_executor,ml_executor=config.executors
    assert qm_executor.available_accelerators==['0','1','2','3']
    assert qm_executor.max_workers_per_node==4 and ml_executor.max_workers_per_node==1
    assert all(e.provider.init_blocks==e.provider.min_blocks==0 for e in config.executors)
    assert config.retries==0
    shell=worker_init(ROOT,settings,'QM')
    assert 'CUDA_VISIBLE_DEVICES=' not in shell
    assert '/vast/home/pjlohr/.conda/' not in shell
    assert 'OMP_NUM_THREADS=8' in worker_init(ROOT,settings,'ML')
    if backend=='slurm':
        assert all(e.provider.partition=='test-partition' for e in config.executors)
        assert qm_executor.provider.max_blocks==2
    settings['qm_workers_per_node']=5
    with pytest.raises(ValueError,match='one QM worker'):
        validate_resources(settings)


def test_relocated_example_preflight_submits_nothing(tmp_path):
    import os,subprocess
    moved=tmp_path/'copied example'; shutil.copytree(ROOT,moved)
    run_dir=tmp_path/'not_created'
    environment={**os.environ,'PYTHONDONTWRITEBYTECODE':'1',
        'PYTHONPATH':str(EXAMPLE.parents[1])+os.pathsep+os.environ.get('PYTHONPATH','')}
    process=subprocess.run([sys.executable,str(moved/'run_trial.py'),'--run-dir',str(run_dir),'--check'],
        cwd=tmp_path,env=environment,capture_output=True,text=True,timeout=60)
    assert process.returncode==0,process.stdout+process.stderr
    assert str(moved/'inputs/bootstrap.h5') in process.stdout
    assert 'No jobs submitted' in process.stdout and not run_dir.exists()
    with h5py.File(moved/'inputs/bootstrap.h5') as handle:
        assert handle['cpp9/coordinates'].shape==(1000,90,3)
    original=json.loads((ROOT/'inputs/provenance.json').read_text())
    assert w.file_hash(moved/'inputs/bootstrap.h5')==original['source_bootstrap_sha256']


def test_staged_dipoles_and_oscillators_match_pyscf_length_gauge():
    from types import SimpleNamespace
    from pyscf import gto
    from pyscf.tdscf.rhf import transition_dipole, oscillator_strength, _contract_multipole
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    td = SimpleNamespace(mol=mol, singlet=True,
                         _scf=SimpleNamespace(mo_coeff=np.eye(2), mo_occ=np.array([2, 0])),
                         xy=[(np.array([[0.7]]), 0), (np.array([[0.2]]), 0), (np.array([[0.1]]), 0)],
                         e=np.array([0.135, 0.155, 0.165]))
    td._contract_multipole = lambda *a, **k: _contract_multipole(td, *a, **k)
    td.get_frozen_mask = lambda: np.ones(2, dtype=bool)
    gpu = SimpleNamespace(asarray=np.asarray, einsum=np.einsum, ndarray=np.ndarray,
                          asnumpy=np.asarray)
    staged = qm.transition_dipoles_staged(td, gpu)
    np.testing.assert_allclose(staged, transition_dipole(td), atol=1e-12)
    computed = (2.0 / 3.0) * td.e * np.einsum('si,si->s', staged, staged)
    np.testing.assert_allclose(computed, oscillator_strength(td, gauge='length'), atol=1e-12)


def test_nonfinite_source_frame_is_dropped_without_qm(source, settings, tmp_path):
    _, reference = settings
    with h5py.File(source[0], 'a') as handle:
        handle['cpp9/coordinates'][11, 0, 0] = np.nan
    submitted = []
    def submit(molecule):
        submitted.append(molecule.get_metadata()['source_global_index'])
        return completed(successful(molecule, reference))
    result = run(source, settings, tmp_path, submit)
    assert result['accepted'] == 11 and result['rejected'] == 1
    assert 11 not in submitted


def test_unpublished_partial_is_replaced_on_resume(source, settings, tmp_path):
    _, reference = settings
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        run(source, settings, tmp_path, lambda m: pytest.fail('submitted'), stop=stop)
    partial = tmp_path / 'run/h5store/data-0000000000.h5.partial'
    partial.write_bytes(b'incomplete HDF5')
    result = run(source, settings, tmp_path, lambda m: completed(successful(m, reference)), resume=True)
    assert result['accepted'] == 12 and not partial.exists()


def test_too_few_survivors_rejects_training_before_submission(source, settings, tmp_path):
    _, reference = settings
    frames = w.input_frames(source[0], source[1].numbers, limit=1)
    w.label_all(frames=frames, submit=lambda m: completed(successful(m, reference)),
                directory=tmp_path / 'run', request={}, reference=reference)
    config = json.loads((ROOT / 'hippynn_config.json').read_text())
    with pytest.raises(ValueError, match='too few structures'):
        validate_training(tmp_path / 'run', config)


@pytest.mark.parametrize('executor_index',[0,1])
def test_generated_slurm_script_without_submission(tmp_path,monkeypatch,executor_index):
    import subprocess
    from parsl.executors.high_throughput import executor
    from parsl.providers import SlurmProvider
    monkeypatch.setattr(executor,'get_all_addresses',lambda:{'127.0.0.1'})
    monkeypatch.setattr(SlurmProvider,'execute_wait',lambda *args,**kwargs:(0,'',''))
    settings=json.loads((ROOT/'resources.json').read_text()); settings['backend']='slurm'
    for role in ('QM','ML'): settings[role]['partition']='test-partition'
    provider=build_config(ROOT,tmp_path,settings).executors[executor_index].provider
    provider.script_dir=str(tmp_path)
    commands=[]
    def pretend_sbatch(command,**kwargs):
        commands.append(command)
        return 0,'Submitted batch job 999999\n',''
    monkeypatch.setattr(provider,'execute_wait',pretend_sbatch)
    assert provider.submit('true',1,job_name='cpu-test')=='999999'
    assert len(commands)==1 and commands[0].startswith('sbatch ')
    script=Path(commands[0].removeprefix('sbatch '))
    content=script.read_text()
    assert 'export PYTORCH_ALLOC_CONF=expandable_segments:True\n' in content
    assert 'Trueexport' not in content
    subprocess.run(['bash','-n',str(script)],check=True)


def test_installed_alf_identity_needs_no_git(tmp_path,monkeypatch):
    import alframework
    fake=tmp_path/'installed/alframework'; fake.mkdir(parents=True)
    original=runner.alf_identity()
    for name in original['source_sha256']:
        path=fake/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('installed module')
    (fake/'__init__.py').write_text('')
    monkeypatch.setattr(alframework,'__file__',str(fake/'__init__.py'))
    assert runner.alf_identity()['commit'] is None
    assert set(runner.alf_identity()['source_sha256'])==set(original['source_sha256'])


@pytest.mark.parametrize('fail_training',[False,True],ids=['complete','retry-partial-model'])
def test_finite_driver_labels_trains_then_resume(source,settings,tmp_path,monkeypatch,fail_training):
    from types import SimpleNamespace
    import parsl
    from alframework.ml_interfaces import excited_state_hippynn_interface as trainer
    _,reference=settings
    directory=tmp_path/'full'
    submitted=[]; trained=[]; phases=[]
    resources=json.loads((ROOT/'resources.json').read_text())
    resources['qm_cores_per_worker']=1
    resource_path=tmp_path/'resources.json'; resource_path.write_text(json.dumps(resources))
    monkeypatch.setattr(runner,'build_config',lambda *args,roles,**kwargs:roles)
    monkeypatch.setattr(parsl,'load',lambda configuration:phases.append(configuration))
    monkeypatch.setattr(parsl,'dfk',lambda:SimpleNamespace(cleanup=lambda:phases.append('cleanup')))
    monkeypatch.setattr(parsl,'clear',lambda:phases.append('clear'))
    def label(molecule,*args,**kwargs):
        assert args[0]['num_threads']==1
        submitted.append(molecule.get_metadata()['source_global_index'])
        return completed(successful(molecule,reference))
    def train(config,h5_dir,model_path,index,gpus,properties):
        trained.append(index)
        assert validate_training(directory,config)['after_filtering']==12
        assert gpus==4 and config['n_models']==4
        for member in range(4):
            path=Path(model_path.format(index))/f'model-{member:02d}'
            path.mkdir(); (path/'training_log.txt').write_text('mock Training complete')
        return completed(([not (fail_training and index==0)]*4,index))
    monkeypatch.setattr(qm,'gpu4pyscf_excited_state_task',label)
    monkeypatch.setattr(trainer,'train_excited_state_HIPPYNN_ensemble_task',train)
    args=['--input-h5',str(source[0]),'--run-dir',str(directory),'--resources',str(resource_path)]
    assert runner.main(args)==(3 if fail_training else 0)
    assert phases[:6]==[('QM',),'cleanup','clear',('ML',),'cleanup','clear']
    assert runner.main([*args,'--resume'])==0
    assert sorted(submitted)==list(range(12)) and len(submitted)==12
    assert trained==([0,1] if fail_training else [0])
    assert json.loads((directory/'training.json').read_text())['status']=='complete'
    if fail_training:
        assert (directory/'models/model-0000').is_dir()
    # Storage sorts atom identities and every aligned label together.
    with h5py.File(directory/'h5store/data-0000000000.h5') as handle:
        group=next(iter(handle.values()))
        order=np.argsort(source[1].numbers,kind='stable')
        assert group['topology_atom_ids'][0].tolist()==order.tolist()
        np.testing.assert_allclose(group['coordinates'][0],source[1].positions[order])
