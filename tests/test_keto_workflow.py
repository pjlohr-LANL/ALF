"""Frozen keto handoff/checkpoint contracts; no scheduler or production data."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
from ase.io import read


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'examples/keto_rdkit_alf_galf'
spec = importlib.util.spec_from_file_location('keto_example', EXAMPLE / 'prepare.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


@pytest.fixture
def source(tmp_path):
    run = tmp_path / 'alf'
    workflow.initialize(run)
    atoms = read(run / 'keto.xyz')
    order = np.argsort(atoms.numbers, kind='stable')
    master = workflow.read_json(run / 'master_config.json')
    # Write out of creation order, with a nonfinite row and alternating same-Z IDs.
    for shard, start, n in [(2, 20, 2), (0, 0, 3), (1, 10, 3), (10, 100, 2)]:
        with h5py.File(str(run / 'h5store/data-{:04d}.h5').format(shard), 'w') as handle:
            handle.attrs['source'] = 'test source'
            group = handle.create_group('keto')
            group.attrs['units'] = 'eV, Angstrom'
            permutations = np.tile(order, (n, 1))
            permutations[1, [0, 1]] = permutations[1, [1, 0]]
            xyz = np.stack([atoms.positions[p] + .001 * (start + i) for i,p in enumerate(permutations)])
            if shard == 0:
                xyz[1, 0, 0] = np.nan
            group['coordinates'] = xyz
            group['species'] = np.asarray(atoms.get_chemical_symbols(), dtype='S2')[order]
            group['topology_atom_ids'] = permutations
            group['_id'] = np.asarray([f'mol-boot-{i:010d}' if shard==0 else f'mol-{shard:04d}-{i:010d}' for i in range(n)], dtype='S32')
            metadata=np.arange(start,start+n,dtype=float); metadata[0]=np.nan
            group['extra_frame_property'] = metadata
            for state in range(4):
                group[f'sE{state}'] = np.arange(start, start+n, dtype=float)[:,None] + state*.05
                force = np.broadcast_to(permutations[:,:,None], (n,len(atoms),3)).copy().astype(float)
                group[f'F{state}'] = force + state
                group[f'F{state}'].attrs['state'] = state
    workflow.write_json(run / 'status.json', {'current_h5_id':3,'current_model_id':2})
    return run, master


def frame_keys(manifest):
    return [(r['sha256'], row) for r in manifest['sources'] for row in r['rows']]


def test_exact_handoff_preserves_rows_labels_identity_and_provenance(source, tmp_path):
    run, master = source
    target = tmp_path / 'galf'
    manifest = workflow.freeze(run / 'master_config.json', target, 4, handoff=True)
    assert manifest['frames'] == 4 and manifest['available_valid_frames'] == 7
    assert manifest['phase_counts'] == {'seed':2,'alf':2,'galf':0}
    assert manifest['nonfinite_frames'] == 1 and manifest['overshoot'] == 3
    assert [(r['shard'],r['rows']) for r in manifest['sources']] == [(0,[0,2]),(1,[0,1]),(2,[])]
    assert not (target / 'status.json').exists() and not list((target / 'models').iterdir())
    with h5py.File(target / 'h5store/data-0001.h5') as handle, h5py.File(run / 'h5store/data-0001.h5') as original:
        group, old = handle['keto'], original['keto']
        assert handle.attrs['source'] == 'test source' and group.attrs['units'] == 'eV, Angstrom'
        for key in old:
            expected = old[key][()] if key == 'species' else old[key][:2]
            np.testing.assert_array_equal(group[key][()], expected)
        for key,(lower,upper) in workflow.GAPS.items():
            np.testing.assert_allclose(group[key][:], (old[upper][:2] - old[lower][:2]).reshape(-1))
        assert group['F3'].attrs['state'] == 3
    for artifact in manifest['artifacts']:
        workflow.verify(artifact)
        assert artifact['derived_gap_properties'] == ['dE01','dE12','dE23']
    report = workflow.check(target / 'master_config.json')
    assert report['frames'] == 4 and report['stage'] == 'galf'
    with pytest.raises(FileExistsError):
        workflow.freeze(run / 'master_config.json', target, 4, handoff=True)
    with pytest.raises(ValueError, match='Need 8 valid'):
        workflow.freeze(run / 'master_config.json', tmp_path / 'too_many', 8, handoff=True)
    assert not (tmp_path / 'too_many').exists()


def test_nested_snapshots_and_hybrid_id_reset_use_lineage_not_generation(source, tmp_path):
    run, _ = source
    target = tmp_path / 'galf'
    workflow.freeze(run / 'master_config.json', target, 4, handoff=True)
    # New GALF generation IDs restart and overlap the independent ALF branch.
    with h5py.File(target / 'h5store/data-0001.h5') as handle, h5py.File(target / 'h5store/data-0002.h5', 'w') as new:
        handle.copy('keto', new)
        new['keto/_id'][:] = np.array(['mol-0000-0000000000','mol-0000-0000000001'],dtype='S32')
    workflow.write_json(target / 'status.json', {'current_h5_id':3,'current_model_id':1})
    small = workflow.freeze(target / 'master_config.json', tmp_path / 'snapshot5', 5)
    large = workflow.freeze(target / 'master_config.json', tmp_path / 'snapshot6', 6)
    assert frame_keys(small) == frame_keys(large)[:5]
    assert small['phase_counts'] == {'seed':2,'alf':2,'galf':1}
    assert large['phase_counts'] == {'seed':2,'alf':2,'galf':2}
    assert small['overshoot'] == 1 and large['overshoot'] == 0
    saved = workflow.read_json(tmp_path / 'snapshot6/manifest.json')
    for artifact in saved['artifacts']:
        workflow.verify(artifact)
    assert saved == large


@pytest.mark.parametrize('fault,message', [('missing_ids','Ambiguous'),('identity','Topology IDs'),
    ('force_shape','energy/force shape'),('gap','Inconsistent stored'),('extra_state','exactly four'),('qm','matching PySEQM')])
def test_invalid_sources_fail_without_publishing(source, tmp_path, fault, message):
    run, master = source
    path = run / 'h5store/data-0000.h5'
    with h5py.File(path, 'a') as handle:
        group = handle['keto']
        if fault == 'missing_ids':
            del group['topology_atom_ids']
        elif fault == 'identity':
            ids = group['topology_atom_ids'][:]
            ids[:,[0,-1]] = ids[:,[-1,0]]
            group['topology_atom_ids'][:] = ids
        elif fault == 'force_shape':
            del group['F0']
            group['F0'] = np.zeros((3,15,2))
        elif fault == 'gap':
            group['dE01'] = np.zeros(3)
        elif fault == 'qm':
            qm=workflow.read_json(run/'QM_config.json'); qm['method']='PM3'
            workflow.write_json(run/'QM_config.json',qm)
        else:
            master['properties_list']['sE4'] = ['sE4','system',1.]
            workflow.write_json(run / 'master_config.json', master)
    with pytest.raises(ValueError, match=message):
        workflow.freeze(run / 'master_config.json', tmp_path / 'bad', 2, handoff=True)
    assert not (tmp_path / 'bad').exists()


def test_fingerprints_boundaries_static_ids_and_initial_training_gate(source, tmp_path):
    run, _ = source
    manifest = workflow.select_frames(run / 'master_config.json', 2)
    with h5py.File(run / 'h5store/data-0000.h5', 'a') as handle:
        handle['keto'].attrs['changed'] = True
    with workflow.new_directory(tmp_path / 'changed') as (temporary, destination):
        (temporary / 'h5store').mkdir()
        with pytest.raises(RuntimeError, match='Fingerprint changed'):
            workflow.copy_selection(manifest, temporary, destination)
    (run / 'status.json').unlink()
    with pytest.raises(ValueError, match='No completed-shard status'):
        workflow.select_frames(run / 'master_config.json', 2)
    # Explicit boundary includes shard 10 after shard 2, never lexicographic order.
    full = workflow.select_frames(run / 'master_config.json', 9, closed_before=11)
    assert [r['shard'] for r in full['sources']] == [0,1,2,10]
    with h5py.File(run / 'h5store/data-0000.h5','a') as handle:
        group=handle['keto']; ids=group['topology_atom_ids'][0]
        del group['topology_atom_ids']; group['topology_atom_ids']=ids
    target=tmp_path/'galf'
    workflow.freeze(run/'master_config.json',target,2,handoff=True,closed_before=1)
    contamination=target/'h5store/data-9999.h5'
    contamination.write_bytes(b'unexpected training input')
    with pytest.raises(ValueError,match='only the frozen handoff'):
        workflow.check(target/'master_config.json')
    contamination.unlink()
    workflow.write_json(target/'status.json',{'current_h5_id':1,'current_model_id':-1})
    with pytest.raises(ValueError,match='not been promoted'):
        workflow.check(target/'master_config.json',require_trained=True)
    for member in range(4):
        directory=target/f'models/model-0000/model-{member:02d}'; directory.mkdir(parents=True)
        workflow.write_json(directory/'training_summary.json',{'data':{'n_structures':2,'n_structures_before_filtering':2},'gap_targets':{'enabled':False}})
        (directory/'training_log.txt').write_text('Training complete\n')
        for name in ('experiment_structure.pt','best_checkpoint.pt'):
            (directory/name).write_bytes(b'mock checkpoint')
    with pytest.raises(ValueError,match='training is running or incomplete'):
        workflow.check(target/'master_config.json')
    workflow.write_json(target/'status.json',{'current_h5_id':1,'current_model_id':0})
    assert len(workflow.check(target/'master_config.json',require_trained=True)['initial_training'])==4
    summary=target/'models/model-0000/model-03/training_summary.json'
    content=workflow.read_json(summary); content['data']['n_structures_before_filtering']=3
    workflow.write_json(summary,content)
    with pytest.raises(ValueError,match='did not read the frozen'):
        workflow.check(target/'master_config.json',require_trained=True)
    with h5py.File(target/'h5store/data-0000.h5','a') as handle:
        handle['keto/F0'][0,0,0]+=1
    with pytest.raises(RuntimeError,match='Fingerprint changed'):
        workflow.check(target/'master_config.json')


def test_cli_is_independent_of_working_directory_and_validates_seed_artifacts(tmp_path):
    target=tmp_path/'fresh'
    environment={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
    for arguments in [('init','--output-dir',str(target)),('check','--master',str(target/'master_config.json'))]:
        result=subprocess.run([sys.executable,str(EXAMPLE/'prepare.py'),*arguments],cwd=tmp_path,
                              env=environment,capture_output=True,text=True,timeout=60)
        assert result.returncode==0,result.stderr
    assert workflow.check(target/'master_config.json')['seed_generation_required']
    with pytest.raises(FileNotFoundError,match='Generate the seed'):
        workflow.check(target/'master_config.json',require_seeds=True)
    assert workflow.read_json(target/'master_config.json')['master_directory']==str(target)

    from alframework.tools.rdkit_initialization import prepare_rdkit_seed
    config=workflow.read_json(target/'seed_config.json')
    config.update(pool_size=12,seed_size=6)
    prepare_rdkit_seed(config,target/'seeds')
    master=workflow.read_json(target/'master_config.json')
    master['bootstrap_set']=6
    workflow.write_json(target/'master_config.json',master)
    assert workflow.check(target/'master_config.json',require_seeds=True)['selected_seeds']==6
    with h5py.File(target/'seeds/seed.h5','a') as handle:
        handle['molecule'].attrs['changed']=True
    with pytest.raises(RuntimeError,match='Seed preparation fingerprint changed'):
        workflow.check(target/'master_config.json',require_seeds=True)
