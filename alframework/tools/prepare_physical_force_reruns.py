"""Prepare independent physical-force-screen GALF controls; never submit jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np

from alframework.tools.topology_preflight import run_preflight


REPO = Path(__file__).resolve().parents[2]
SUFFIX = '_physical_force_screen'


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def differences(before, after, prefix=''):
    changes = {}
    for key in sorted(before.keys() | after.keys()):
        name = f'{prefix}.{key}' if prefix else key
        a, b = before.get(key), after.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            changes.update(differences(a, b, name))
        elif a != b:
            changes[name] = {'before': a, 'after': b}
    return changes


def prepare(tautomer, production):
    source = production / f'{tautomer.title()}_GALF_4_states_EF_300K_etkdg_kdg_40k'
    destination = source.with_name(source.name + SUFFIX)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    continuation = json.loads((source / 'galf_continuation_provenance.json').read_text())
    master = json.loads((source / 'master_config.json').read_text())
    sampler = json.loads((source / 'sampler_config.json').read_text())
    assert sampler['max_force_cutoff'] == 16.0
    assert sampler['gap_seeking']['sigma'] == 3.5
    assert sampler['gap_seeking']['alpha_eV'] == .05
    assert sampler['gap_seeking']['trigger_gap_threshold_eV'] == .1
    verified = []

    def copy_file(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = digest(src)
        shutil.copy2(src, dst)
        assert fingerprint == digest(dst) == digest(src)
        verified.append({'source':str(src), 'destination':str(dst), 'sha256':fingerprint})

    # Verify every inherited shard before creating the new run.
    for shard in continuation['copied_h5']:
        path = (source / master['h5_path']).parent / shard['file']
        assert digest(path, 'md5') == shard['md5'], path
    destination.mkdir()
    total = 0
    for shard in continuation['copied_h5']:
        src = (source / master['h5_path']).parent / shard['file']
        copy_file(src, destination / src.relative_to(source))
        with h5py.File(src) as handle:
            count = 0
            for group in handle.values():
                n = len(group['coordinates']); count += n
                assert group['coordinates'].shape[1:] == (15, 3)
                assert np.isfinite(group['coordinates'][:]).all()
                for state in range(4):
                    assert len(group[f'sE{state}']) == n and len(group[f'F{state}']) == n
                    assert group[f'sE{state}'].shape in {(n,), (n,1)}
                    assert group[f'F{state}'].shape == (n,15,3)
                    assert np.isfinite(group[f'sE{state}'][:]).all()
                    assert np.isfinite(group[f'F{state}'][:]).all()
            assert count == shard['frames']
            total += count
    assert total == continuation['total_frames']
    model = source / master['model_path'].format(0)
    original_model = Path(continuation['model_0000_source'])
    members = sorted(model.glob('model-*'))
    assert len(members) == 4
    for path in sorted(model.rglob('*')):
        if path.is_file():
            copy_file(path, destination / path.relative_to(source))
            # Inference artifacts must also match the original BALF ensemble.
            if path.name in {'best_model.pt','best_checkpoint.pt','experiment_structure.pt'}:
                assert digest(path) == digest(original_model / path.relative_to(model))
    for path in sorted((source / 'seed_dataset_scaled').rglob('*')):
        if path.is_file(): copy_file(path, destination / path.relative_to(source))
    topology = json.loads((source / 'topology_reference_state.json').read_text())
    reference = Path(topology['reference_path'])
    local_reference = destination / 'topology_reference_geometry.xyz'
    copy_file(reference, local_reference)
    assert digest(local_reference) == topology['reference_sha256']
    config_changes = {}
    for name in ['master_config.json','sampler_config.json','builder_config.json','hippynn_config.json',
                 'QM_config.json','topology_reference_state.json','seed_run.json','seed_finalize_report.json','reorder_report.json']:
        path = source / name
        if not path.exists(): continue
        before = json.loads(path.read_text())
        after = json.loads(path.read_text().replace(str(source), str(destination)))
        if name == 'sampler_config.json':
            after['topology_check']['reference_conformer_path'] = str(local_reference)
        if name == 'topology_reference_state.json':
            after['reference_path'] = str(local_reference)
        (destination / name).write_text(json.dumps(after,indent=2)+'\n')
        config_changes[name] = differences(before, after)
    for path in source.glob('*.py'):
        (destination / path.name).write_text(path.read_text().replace(str(source), str(destination)))
    for path in source.glob('*.slurm'):
        text = path.read_text().replace(source.name, destination.name)
        old_job = f'GALF_4STATE_EF_{tautomer.upper()}_ETKDGKDG_40K'
        text = text.replace(old_job, old_job + '_PHYSFORCE')
        text = text.replace('Starting ALF master;', 'Physical force screen: all state means, 16 eV/Angstrom. Starting ALF master;')
        target = destination / path.name
        target.write_text(text); target.chmod(path.stat().st_mode)
        subprocess.run(['bash','-n',str(target)],check=True)
    shutil.copy2(source / 'galf_continuation_provenance.json', destination / 'original_galf_continuation_provenance.json')
    result = run_preflight(destination, destination / 'master_config.json', require_model=True)
    (destination / 'physical_force_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    next_h5 = max(int(Path(s['file']).stem.split('-')[1]) for s in continuation['copied_h5']) + 1
    assert not (destination / master['status_path']).exists()
    assert not (destination / 'sampling').exists()
    assert not (destination / 'runinfo').exists()
    assert sorted(p.name for p in (destination / master['model_path']).parent.glob('model-*')) == ['model-0000']
    report = {'source_galf_run':str(source),'run':str(destination),'inherited_frames':total,
        'initial_model_source':str(original_model),'initial_model_trained_on':continuation['model_0000_trained_on'],
        'next_h5':f'data-{next_h5:04d}.h5','initial_model_id':0,'next_training_id':1,
        'comparison_frames':50000,'new_frames_for_comparison':50000-total,
        'force_screen_source':'physical_state_mean','force_screen_scope':'all_states',
        'sampler_metadata_schema_version':2,'submitted':False,
        'config_changes':config_changes,'verified_copies':verified,
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'code_sha256':{str(p.relative_to(REPO)):digest(p) for p in [
            REPO/'alframework/samplers/excited_state_sampling.py',Path(__file__)]}}
    patch = subprocess.check_output(['git','diff','--','alframework/samplers/excited_state_sampling.py',
                                   'alframework/tools/excited_state_iteration_plot_inputs.py'],cwd=REPO,text=True)
    (destination / 'physical_force_sampler.patch').write_text(patch)
    (destination / 'physical_force_rerun_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    (destination / 'README.physical_force_screen.md').write_text(
        f'# {tautomer.title()} GALF physical-force-screen control\n\n'
        f'Prepared only; not submitted. Inherits {total:,} accepted frames and the exact original handoff ensemble. '
        f'That model was trained on {continuation["model_0000_trained_on"]}.\n\n'
        'The physical force screen is the largest atom-force norm across ensemble-mean forces of all states, '
        'with the original 16 eV/Å cutoff. GALF propagation and all bias parameters remain unchanged. '
        'Metadata schema 2 retains actual rejected coordinates and distinguishes physical, propagation, and gap-bias forces.\n\n'
        f'With no inherited ALF status, the master discovers model-0000, skips bootstrap labeling, writes {report["next_h5"]}, '
        f'and trains model-0001 next. Compare the first 50,000 accepted frames including inheritance '
        f'({50000-total:,} new frames); the master has no automatic exact-50k stop.\n\n'
        'Only the original handoff H5 shards and model-0000 are inherited. Training records inside that frozen '
        'ensemble are provenance; no later GALF models, master status, sampling output, QM output, or run logs are copied.\n\n'
        'The provenance JSON records all copy hashes, configuration changes, and code fingerprints. '
        'The source patch is retained here for review. Submission is a separate step.\n')
    print(f'Prepared {destination}: {total} inherited frames, next {report["next_h5"]}; NOT submitted.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tautomer',choices=['enol','keto','both'],default='both')
    parser.add_argument('--production-root',type=Path,default=REPO/'production')
    args = parser.parse_args()
    for tautomer in (('enol','keto') if args.tautomer == 'both' else (args.tautomer,)):
        prepare(tautomer, args.production_root.resolve())
