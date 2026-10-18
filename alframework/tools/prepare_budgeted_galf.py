"""Prepare exact-40k ALF -> 20k GALF runs, without training or submission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from alframework.tools.budgeted_galf import (
    copy_selected_shard, fingerprint, freeze_selection, read_json, select_frames,
    verify_handoff_inputs, write_json,
)
from alframework.tools.prepare_physical_force_reruns import differences
from alframework.tools.topology_preflight import run_preflight

REPO = Path(__file__).resolve().parents[2]
RUN_TEMPLATE = '{title}_GALF_4_states_EF_300K_etkdg_kdg_alf40k_galf20k_physical_force'


def prepare(tautomer, production):
    title = tautomer.title()
    production = Path(production).resolve()
    baseline = production / f'{title}_BALF_4_states_EF_300K_etkdg_kdg_seed'
    template = production / f'{title}_GALF_4_states_EF_300K_etkdg_kdg_40k_physical_force_screen'
    destination = production / RUN_TEMPLATE.format(title=title)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    # Verify availability and input contracts before creating the run.
    handoff, selection = select_frames(baseline, 40000)
    pure, _ = select_frames(baseline, 60000)
    assert np.array_equal(handoff['ids'], pure['ids'][:40000])
    destination.mkdir()
    config_changes, copies = {}, []
    for name in ('master_config.json', 'sampler_config.json', 'builder_config.json', 'hippynn_config.json',
                 'QM_config.json', 'topology_reference_state.json', 'seed_run.json', 'seed_finalize_report.json', 'reorder_report.json'):
        source = template / name
        if not source.exists():
            continue
        before = read_json(source)
        after = json.loads(source.read_text().replace(str(template), str(destination)))
        if name == 'master_config.json':
            after['accepted_frame_budget'] = {'handoff_frames': 40000, 'target_frames': 60000,
                                               'checkpoints': [50000, 60000]}
        if name == 'sampler_config.json':
            after['topology_check']['reference_conformer_path'] = str(destination / 'topology_reference_geometry.xyz')
        if name == 'topology_reference_state.json':
            after['reference_path'] = str(destination / 'topology_reference_geometry.xyz')
        write_json(destination / name, after)
        config_changes[name] = differences(before, after)
    master = read_json(destination / 'master_config.json')
    sampler = read_json(destination / 'sampler_config.json')
    training = read_json(destination / 'hippynn_config.json')
    assert training['n_models'] == 4 and training['network_choice'] == 1
    assert training['train_forces'] and not training['gap_targets']['enabled']
    assert sampler['max_force_cutoff'] == 16.
    assert sampler['gap_seeking']['alpha_eV'] == .05 and sampler['gap_seeking']['sigma'] == 3.5
    assert sampler['gap_seeking']['trigger_gap_threshold_eV'] == .1
    copied_h5 = []
    for record in selection['sources']:
        if not record['rows']:
            continue
        target = destination / Path(master['h5_path']).parent / Path(record['path']).name
        copied = copy_selected_shard(record, target)
        copied.update(frames=len(record['rows']), source=record)
        copied_h5.append(copied)
    for source in sorted((template / 'seed_dataset_scaled').rglob('*')):
        if source.is_file():
            target = destination / source.relative_to(template)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source_record, target_record = fingerprint(source), fingerprint(target)
            assert source_record['sha256'] == target_record['sha256']
            copies.append({'source': source_record, 'copy': target_record})
    reference = template / 'topology_reference_geometry.xyz'
    shutil.copy2(reference, destination / reference.name)
    assert fingerprint(reference)['sha256'] == fingerprint(destination / reference.name)['sha256']
    for source in template.glob('*.py'):
        (destination / source.name).write_text(source.read_text().replace(str(template), str(destination)))
    for source in template.glob('*.slurm'):
        text = source.read_text().replace(template.name, destination.name)
        old_job = f'GALF_4STATE_EF_{tautomer.upper()}_ETKDGKDG_40K_PHYSFORCE'
        text = text.replace(old_job, f'GALF_{tautomer.upper()}_ALF40K_GALF20K_PHYSFORCE')
        text = text.replace('echo "End time: $(date)"', 'echo "End time: $(date)"\n    echo "Elapsed seconds: ${SECONDS}"')
        start = text.index('ALF two-node four-state')
        end = text.index('\n', start)
        text = text[:start] + (
            f'{title}: fresh four-member ensemble on exactly 40,000 ALF labels including seed; '
            'then GALF to 60,000 total retained labels. Checkpoints at 50k/60k; physical-force screen.'
        ) + text[end:]
        analysis = REPO.parent / 'molecule_smax/benchmarks/conformational_analysis/compare_budgeted_galf.py'
        text = text.replace('exit "${EXIT_STATUS}"',
            'if [[ "${EXIT_STATUS}" -eq 0 ]]; then\n'
            f'    "${{PYTHON_BIN}}" "{analysis}" --tautomer {tautomer} --stage update --production-root "{production}"\n'
            'fi\n\nexit "${EXIT_STATUS}"')
        target = destination / 'submit_2node_alf40k_galf20k.slurm'
        target.write_text(text)
        target.chmod(0o750)
        subprocess.run(['bash', '-n', str(target)], check=True)
    report = {'version': 1, 'tautomer': tautomer, 'run': str(destination), 'baseline_run': str(baseline),
              'configuration_template': str(template), 'handoff_frames': 40000, 'target_frames': 60000,
              'checkpoints': [50000, 60000], 'copied_h5': copied_h5, 'seed_copies': copies,
              'config_changes': config_changes, 'initial_model': 'fresh model-0000; no imported models',
              'force_screen_source': 'physical_state_mean', 'force_screen_scope': 'all_states',
              'submitted': False, 'source_selection': selection,
              'code': [fingerprint(path) for path in (
                  Path(__file__), REPO / 'alframework/tools/budgeted_galf.py',
                  REPO / 'alframework/__main__.py', REPO / 'alframework/samplers/excited_state_sampling.py',
                  REPO / 'alframework/ml_interfaces/excited_state_hippynn_interface.py')]}
    comparison_configs = ('master_config.json', 'sampler_config.json', 'QM_config.json', 'hippynn_config.json', 'builder_config.json')
    report['baseline_configuration_differences'] = {
        name: differences(read_json(baseline / name), read_json(destination / name)) for name in comparison_configs}
    report['configuration_snapshots'] = {
        label: {name: read_json(folder / name) for name in comparison_configs}
        for label, folder in (('baseline', baseline), ('hybrid', destination))}
    write_json(destination / 'budgeted_galf_provenance.json', report)
    verify_handoff_inputs(destination)
    copied, _ = select_frames(destination, 40000)
    for name in ('coordinates', 'energies', 'ids', 'numbers', 'stored_topology_atom_ids'):
        assert np.array_equal(copied[name], handoff[name]), name
    for count in (40000, 50000, 60000):
        freeze_selection(baseline, count, destination / 'baseline_checkpoints' / str(count))
    freeze_selection(destination, 40000, destination / 'budget_checkpoints' / '40000', handoff_frames=40000)
    write_json(destination / 'topology_preflight.json', run_preflight(destination, destination / 'master_config.json'))
    assert not (destination / master['status_path']).exists()
    assert not (destination / Path(master['model_path']).parent).exists()
    assert not (destination / 'sampling').exists()
    (destination / 'README.budgeted_galf.md').write_text(
        f'# {title}: exact 40k ALF + up to 20k GALF\n\n'
        'Prepared and validated; not submitted. No inherited model or master status.\n\n'
        'The normal ALF bootstrap-training path trains model-0000 on the frozen 40,000-frame pool '
        '(including seed), using the existing random train/validation/test defaults. All four members '
        'must complete, report 40,000 input frames, and pass the artifact/input fingerprint gate before sampling. '
        'An incomplete ensemble causes a failure before sampling; restarting an interrupted initial training '
        'requires explicit repair rather than treating a partial directory as a model.\n\n'
        'GALF retains alpha=0.05 eV, sigma=3.5, trigger=0.1 eV, normal E/F training and uncertainty acquisition, '
        'and all-state physical mean-force screening at 16 eV/Angstrom. Gap loss remains disabled.\n\n'
        'The opt-in accepted_frame_budget master setting freezes first-50k and first-60k checkpoints '
        'after completed H5 saves, then stops after reaching at least 60k. It does not truncate a completed batch. '
        'budget_progress.json records retained overshoot separately. Completed-but-unsaved or in-flight QM work '
        'is not part of the retained budget and must be counted separately in cost reporting.\n\n'
        'Pure-ALF 40k/50k/60k snapshots and the hybrid 40k snapshot are already frozen locally. '
        'The analysis CLI is benchmarks/conformational_analysis/compare_budgeted_galf.py in molecule_smax; '
        'use --tautomer ' + tautomer + ' --stage update to refresh from completed checkpoints. '
        'Plots load reduced caches and do not run SOAP/PCA/DensMAP.\n\n'
        'Submission script: submit_2node_alf40k_galf20k.slurm (existing two-node ml4chem allocation).\n')
    print(f'Prepared {destination}; exactly 40000 inherited frames; no models; NOT submitted.', flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tautomer', choices=('enol', 'keto', 'both'), default='both')
    parser.add_argument('--production-root', type=Path, default=REPO / 'production')
    args = parser.parse_args()
    for tautomer in ('enol', 'keto') if args.tautomer == 'both' else (args.tautomer,):
        prepare(tautomer, args.production_root)


if __name__ == '__main__':
    main()
