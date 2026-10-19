#!/usr/bin/env python
"""Explicit keto ALF/GALF preparation and frozen, seed-inclusive H5 snapshots.

This example utility never submits work or changes the ALF driver. Only shards
below a recorded status boundary (or an explicit --closed-before) are read.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import h5py
import numpy as np
from ase.data import atomic_numbers
from ase.io import read

from alframework.tools.atom_order import canonical_orders

HERE = Path(__file__).resolve().parent
SCHEMA = 'keto-rdkit-alf-galf-v1'
GAPS = {f'dE{i}{i+1}': (f'sE{i}', f'sE{i+1}') for i in range(3)}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size': path.stat().st_size}


def verify(record):
    if fingerprint(record['path']) != {key: record[key] for key in ('path', 'sha256', 'size')}:
        raise RuntimeError(f"Fingerprint changed: {record['path']}")


def run_config(master_path):
    path = Path(master_path).expanduser().resolve(strict=True)
    master = read_json(path)
    directory = master.get('master_directory')
    root = path.parent if directory in (None, 'pwd') else Path(directory).expanduser()
    if not root.is_absolute():
        root = path.parent / root
    return path, master, root.resolve()


def local_path(root, value):
    return (root / value).resolve()


@contextmanager
def new_directory(destination):
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing existing output: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=destination.parent))
    try:
        yield temporary, destination
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def configure_run(temporary, destination, stage):
    # Copy templates, never scientific data, status, or models from another run.
    for pattern in ('*.json', '*.xyz', '*.py', '*.sh', '*.md'):
        for path in HERE.glob(pattern):
            shutil.copy2(path, temporary / path.name)
    master = read_json(temporary / f'master_{stage}.json')
    master['master_directory'] = str(destination)
    for key in ('h5_path', 'model_path', 'status_path', 'QM_scratch_dir',
                'builder_config_path', 'sampler_config_path', 'QM_config_path', 'ML_config_path'):
        master[key] = str(destination / master[key])
    write_json(temporary / 'master_config.json', master)
    seed = read_json(temporary / 'seed_config.json')
    seed['reference_geometry'] = str(destination / 'keto.xyz')
    write_json(temporary / 'seed_config.json', seed)
    for name in ('sampler_alf.json', 'sampler_galf.json'):
        sampler = read_json(temporary / name)
        sampler['topology_check']['reference_conformer_path'] = str(destination / 'keto.xyz')
        sampler['force_screen_audit_dir'] = str(destination / 'sampling/force_screen_audit')
        write_json(temporary / name, sampler)
    builder = read_json(temporary / 'builder_bootstrap.json')
    builder['bootstrap_h5_path'] = str(destination / 'seeds/seed.h5')
    write_json(temporary / 'builder_bootstrap.json', builder)
    qm = read_json(temporary / 'QM_config.json')
    qm['pyseqm_log_dir'] = str(destination / 'pyseqm_logs')
    write_json(temporary / 'QM_config.json', qm)
    for name in ('h5store', 'models', 'sampling'):
        (temporary / name).mkdir()


def initialize(output):
    with new_directory(output) as (temporary, destination):
        configure_run(temporary, destination, 'alf')
        write_json(temporary / 'preparation.json', {'schema': SCHEMA, 'stage': 'alf',
                   'seed_generation': 'Run the RDKit initialization command before launching ALF.'})
    return {'run': str(destination), 'stage': 'alf', 'seed_generation_required': True}


def completed_shards(master_path, closed_before=None):
    path, master, root = run_config(master_path)
    status_path = local_path(root, master['status_path'])
    if closed_before is None:
        if status_path.exists():
            closed_before = int(read_json(status_path)['current_h5_id'])
        elif (root / 'handoff_manifest.json').exists():
            manifest = read_json(root / 'handoff_manifest.json')
            for artifact in manifest['artifacts']:
                verify(artifact)
            closed_before = len(manifest['artifacts'])
        else:
            raise ValueError('No completed-shard status; supply --closed-before only for a verified quiescent store.')
    if closed_before < 0:
        raise ValueError('--closed-before must be nonnegative.')
    pattern = str(local_path(root, master['h5_path']))
    # Accept the driver's one integer formatting field, not a hard-coded filename.
    sample = Path(pattern.format(0))
    match = re.fullmatch(r'(.*)\{(?::[^{}]*)?\}(.*)', Path(pattern).name)
    if match is None:
        raise ValueError('h5_path must contain exactly one integer formatting field.')
    expression = re.compile(re.escape(match[1]) + r'(\d+)' + re.escape(match[2]))
    paths = []
    for candidate in sample.parent.glob('*'):
        found = expression.fullmatch(candidate.name)
        if found and int(found[1]) < closed_before:
            paths.append((int(found[1]), candidate))
    return path, master, root, sorted(paths), closed_before


def select_frames(master_path, count=None, closed_before=None):
    if count is not None and count <= 0:
        raise ValueError('--frames must be positive.')
    path, master, root, paths, boundary = completed_shards(master_path, closed_before)
    schema = master['properties_list']
    for state in range(4):
        for key in (f'sE{state}', f'F{state}'):
            if schema.get(key) != [key, 'system' if key.startswith('sE') else 'atomic', 1.]:
                raise ValueError(f'Expected four-state, unscaled {key} property contract.')
    if {key for key in schema if re.fullmatch(r'sE\d+', key)} != {f'sE{i}' for i in range(4)}:
        raise ValueError('This example accepts exactly four electronic states.')
    sampler_path = local_path(root, master['sampler_config_path'])
    sampler = read_json(sampler_path)
    qm_path = local_path(root, master['QM_config_path'])
    reference_path = local_path(root, sampler['topology_check']['reference_conformer_path'])
    reference = read(reference_path)
    expected = read(HERE / 'keto.xyz').numbers
    from alframework.tools.molecular_topology import load_reference_topology
    topology_settings = {key: value for key, value in sampler['topology_check'].items()
                         if key in {'enabled', 'reference_format', 'reference_charge', 'bond_min_scale', 'bond_max_scale', 'connectivity_scale'}}
    topology_settings['reference_conformer_path'] = str(reference_path)
    source_topology = load_reference_topology({'topology_check': topology_settings})
    topology_settings['reference_conformer_path'] = str(HERE / 'keto.xyz')
    keto_topology = load_reference_topology({'topology_check': topology_settings})
    if source_topology.bond_set != keto_topology.bond_set:
        raise ValueError('Source reference graph is not the supplied keto topology.')
    if not np.array_equal(reference.numbers, expected):
        raise ValueError('Reference must use the supplied keto reference atom order.')
    inherited = read_json(root / 'handoff_manifest.json') if (root / 'handoff_manifest.json').exists() else None
    handoff_count = inherited['frames'] if inherited else None
    records, selected, valid_count, invalid_count, stored_count = [], 0, 0, 0, 0
    for shard, source in paths:
        record = fingerprint(source)
        with h5py.File(source) as handle:
            if len(handle) != 1 or not isinstance(next(iter(handle.values())), h5py.Group):
                raise ValueError(f'Expected one keto molecular group: {source}')
            group_name, group = next(iter(handle.items()))
            xyz = group['coordinates'][:]
            if xyz.ndim != 3 or xyz.shape[1:] != (len(expected), 3):
                raise ValueError(f'Invalid coordinates shape: {source}')
            n = len(xyz)
            species = group['species'][:]
            if species.shape != (len(expected),):
                raise ValueError(f'Expected one stored species vector: {source}')
            numbers = np.asarray([atomic_numbers[s.decode() if isinstance(s, bytes) else str(s)] for s in species])
            if 'topology_atom_ids' not in group:
                raise ValueError(f'Ambiguous atom identities: explicit topology_atom_ids required: {source}')
            identities = group['topology_atom_ids'][:]
            order = canonical_orders(identities, n, len(expected), context=str(source))
            if not np.equal(numbers[order], expected).all():
                raise ValueError(f'Topology IDs disagree with reference species: {source}')
            keep = np.isfinite(xyz).all((1, 2))
            energies = {}
            for state in range(4):
                energy, force = group[f'sE{state}'][:], group[f'F{state}'][:]
                if energy.shape not in ((n,), (n, 1)) or force.shape != xyz.shape:
                    raise ValueError(f'Invalid energy/force shape for state {state}: {source}')
                energies[f'sE{state}'] = energy.reshape(n)
                keep &= np.isfinite(energy.reshape(n)) & np.isfinite(force).all((1, 2))
            ids = group['_id'].asstr()[:]
            if ids.shape != (n,):
                raise ValueError(f'Invalid frame IDs: {source}')
            available = np.flatnonzero(keep)
            take = len(available) if count is None else min(len(available), count - selected)
            rows = available[:take]
            phases = []
            for offset, row in enumerate(rows):
                if handoff_count is not None and selected + offset >= handoff_count:
                    phases.append('galf')
                elif str(ids[row]).startswith('mol-boot-'):
                    phases.append('seed')
                else:
                    phases.append('alf')
            ids_by_frame = np.broadcast_to(identities, (n, len(expected))) if identities.ndim == 1 else identities
            permutations, indices = np.unique(ids_by_frame[rows], axis=0, return_inverse=True)
            for key, (lower, upper) in GAPS.items():
                if key in group:
                    gap = group[key][:]
                    if gap.shape not in ((n,), (n, 1)) or not np.allclose(
                            gap.reshape(n)[rows], (energies[upper] - energies[lower])[rows], atol=1e-6, rtol=1e-6):
                        raise ValueError(f'Inconsistent stored {key}: {source}')
            record.update(shard=shard, group=group_name, frames_available=n, rejected_nonfinite=int((~keep).sum()),
                          rows=rows.tolist(), ids=ids[rows].tolist(), phases=phases,
                          topology_permutations=permutations.tolist(), topology_permutation_indices=indices.tolist())
        verify(record)
        records.append(record)
        selected += len(rows)
        valid_count += int(keep.sum())
        invalid_count += int((~keep).sum())
        stored_count += n
    if count is not None and selected < count:
        raise ValueError(f'Need {count} valid frames; found {selected}.')
    if not records and count is not None:
        raise ValueError('No completed shards available.')
    phases = [phase for record in records for phase in record['phases']]
    manifest = {'schema': SCHEMA, 'source_master': fingerprint(path), 'source_sampler': fingerprint(sampler_path),
                'reference': fingerprint(reference_path), 'source_QM': fingerprint(qm_path), 'closed_before': boundary, 'frames': selected,
                'available_valid_frames': valid_count, 'stored_frames': stored_count, 'nonfinite_frames': invalid_count,
                'overshoot': valid_count - selected, 'phase_counts': {p: phases.count(p) for p in ('seed','alf','galf')},
                'selection': 'numeric completed shards; stored row order; finite coordinates and four energies/forces; no geometry filter',
                'sources': records}
    if inherited:
        manifest['lineage'] = {'handoff_frames': handoff_count, 'manifest': fingerprint(root / 'handoff_manifest.json')}
    return manifest


def sliced_data(dataset, key, rows, n_frames, n_atoms):
    if key == 'species' or dataset.ndim == 0 or (key == 'topology_atom_ids' and dataset.shape == (n_atoms,)):
        return dataset[()]
    if dataset.shape[0] != n_frames:
        raise ValueError(f'Cannot classify static versus frame dataset: {dataset.name}')
    return dataset[rows]


def copy_selection(manifest, temporary, destination):
    artifacts = []
    for record in manifest['sources']:
        if not record['rows']:
            continue
        verify(record)
        name = f'h5store/data-{len(artifacts):04d}.h5'
        rows = np.asarray(record['rows'], dtype=np.int64)
        with h5py.File(record['path']) as src, h5py.File(temporary / name, 'x') as dst:
            dst.attrs.update(src.attrs)
            group = src[record['group']]
            target = dst.create_group(record['group'])
            target.attrs.update(group.attrs)
            for key, dataset in group.items():
                data = sliced_data(dataset, key, rows, record['frames_available'], group['coordinates'].shape[1])
                copied = target.create_dataset(key, data=data, dtype=dataset.dtype)
                copied.attrs.update(dataset.attrs)
            derived = []
            for key, (lower, upper) in GAPS.items():
                if key not in target:
                    target[key] = target[upper][:].reshape(-1) - target[lower][:].reshape(-1)
                    target[key].attrs['derived_from'] = f'{upper} - {lower}'
                    derived.append(key)
        verify(record)
        # Compare every copied value to its original selected row before publishing.
        with h5py.File(record['path']) as src, h5py.File(temporary / name) as dst:
            for key, dataset in src[record['group']].items():
                expected = sliced_data(dataset, key, rows, record['frames_available'], src[record['group']]['coordinates'].shape[1])
                if not np.array_equal(expected, dst[record['group']][key][()], equal_nan=np.asarray(expected).dtype.kind in 'fc'):
                    raise RuntimeError(f'Copy mismatch: {record["path"]}:{key}')
        artifact = fingerprint(temporary / name)
        artifact.update(path=str(destination / name), derived_gap_properties=derived)
        artifacts.append(artifact)
    manifest['artifacts'] = artifacts
    return manifest


def freeze(master_path, output, count, *, handoff=False, closed_before=None):
    manifest = select_frames(master_path, count, closed_before)
    if handoff:
        _, master, root = run_config(master_path)
        qm = read_json(local_path(root, master['QM_config_path']))
        sampler = read_json(local_path(root, master['sampler_config_path']))
        reference_qm = read_json(HERE / 'QM_config.json')
        offset = qm.get('energy_offset_eV', sampler.get('energy_offset_eV', 0.))
        if (not master['QM_task'].startswith('alframework.qm_interfaces.pyseqm_interface.')
                or qm.get('method', 'AM1') != reference_qm['method']
                or offset != reference_qm['energy_offset_eV']
                or any(qm.get(key, reference_qm[key]) != reference_qm[key] for key in ('scf_eps','cis_tol'))):
            raise ValueError('Handoff requires matching PySEQM AM1/CIS settings and keto energy offset.')
        if sampler.get('gap_seeking', {}).get('enabled', False):
            raise ValueError('A GALF handoff must start from ordinary ALF, not an existing GALF continuation.')
    with new_directory(output) as (temporary, destination):
        if handoff:
            configure_run(temporary, destination, 'galf')
            # Preserve the verified source reference order and actual geometry.
            shutil.copy2(manifest['reference']['path'], temporary / 'keto.xyz')
        else:
            (temporary / 'h5store').mkdir()
        copy_selection(manifest, temporary, destination)
        for key in ('source_master', 'source_sampler', 'source_QM', 'reference'):
            verify(manifest[key])
        if handoff:
            ml = read_json(temporary / 'hippynn_config.json')
            manifest['initial_training'] = {'frames': count, 'members': ml['n_models'],
                'ML_config': fingerprint(temporary / 'hippynn_config.json')}
            manifest['initial_training']['ML_config']['path'] = str(destination / 'hippynn_config.json')
        write_json(temporary / ('handoff_manifest.json' if handoff else 'manifest.json'), manifest)
    return manifest


def verify_initial_training(root, master):
    manifest = read_json(root / 'handoff_manifest.json')
    for artifact in manifest['artifacts']:
        verify(artifact)
    verify(manifest['initial_training']['ML_config'])
    if str(local_path(root, master['ML_config_path'])) != manifest['initial_training']['ML_config']['path']:
        raise ValueError('Initial training must use the frozen handoff ML configuration.')
    ml = read_json(local_path(root, master['ML_config_path']))
    status = read_json(local_path(root, master['status_path']))
    if status['current_model_id'] < 0:
        raise ValueError('Initial ensemble has not been promoted; wait for all four members.')
    ensemble = Path(str(local_path(root, master['model_path'])).format(0))
    expected = {f'model-{i:02d}' for i in range(ml['n_models'])}
    if {p.name for p in ensemble.glob('model-*') if p.is_dir()} != expected:
        raise ValueError('Initial ensemble member directories are incomplete or unexpected.')
    summaries = []
    for name in sorted(expected):
        member = ensemble / name
        for artifact in ('training_summary.json', 'training_log.txt', 'experiment_structure.pt', 'best_checkpoint.pt'):
            if not (member / artifact).is_file() or not (member / artifact).stat().st_size:
                raise ValueError(f'Incomplete initial training artifact: {member / artifact}')
        if 'Training complete' not in (member / 'training_log.txt').read_text():
            raise ValueError(f'Initial member has not completed training: {name}')
        summary = read_json(member / 'training_summary.json')
        data = summary['data']
        if data.get('n_structures_before_filtering', data['n_structures']) != manifest['frames']:
            raise ValueError(f'Initial member did not read the frozen {manifest["frames"]}-frame pool: {name}')
        if summary['gap_targets']['enabled']:
            raise ValueError('Gap loss must remain disabled for this example.')
        summaries.append({'member': name, 'input_frames': manifest['frames'], 'training_pool_after_filters': data['n_structures']})
    return summaries


def check(master_path, *, closed_before=None, require_trained=False, require_seeds=False):
    from alframework.ml_interfaces.excited_state_hippynn_interface import validate_excited_state_training_config
    from alframework.tools.molecular_topology import load_reference_topology, validate_fixed_topology
    from alframework.tools.sampler_batching import validate_state_selection
    from alframework.tools.sampling_validation import configured_alchemi_gap_rows, configured_score
    from alframework.tools.tools import load_module_from_string

    _, master, root = run_config(master_path)
    configs = {role: read_json(local_path(root, master[f'{role}_config_path'])) for role in ('builder','sampler','QM','ML')}
    table = validate_excited_state_training_config(configs['ML'], master['properties_list'])
    if len(table) != 4:
        raise ValueError('Exactly four states are required.')
    for role in configs:
        load_module_from_string(master[f'{role}_task'])  # No Parsl configuration import or allocation.
    sampler = configs['sampler']
    topology = load_reference_topology(sampler, master_directory=str(root))
    if not validate_fixed_topology(read(topology.reference_path), topology).valid:
        raise ValueError('Reference geometry fails its topology preflight.')
    _, states = validate_state_selection(sampler, master['properties_list'])
    for state in states:
        configured_score(sampler, master['properties_list'], state)
        configured_alchemi_gap_rows(sampler, master['properties_list'], state)
    status_path = local_path(root, master['status_path'])
    h5_directory = local_path(root, master['h5_path']).parent
    fresh = not status_path.exists() and not list(h5_directory.glob('*.h5'))
    report = {'run': str(root), 'stage': 'galf' if sampler.get('gap_seeking', {}).get('enabled') else 'alf'}
    if fresh:
        report.update(frames=0, available_valid_frames=0, phase_counts={'seed':0,'alf':0,'galf':0})
    else:
        selection = select_frames(master_path, closed_before=closed_before)
        report.update({key: selection[key] for key in ('frames','available_valid_frames','nonfinite_frames','phase_counts','closed_before')})
    if (root / 'handoff_manifest.json').exists():
        handoff = read_json(root / 'handoff_manifest.json')
        for artifact in handoff['artifacts']:
            verify(artifact)
        verify(handoff['initial_training']['ML_config'])
        if (not status_path.exists() or read_json(status_path)['current_model_id'] < 0):
            actual = {str(path.resolve()) for path in h5_directory.iterdir() if path.suffix in ('.h5','.hdf5')}
            if actual != {record['path'] for record in handoff['artifacts']}:
                raise ValueError('Initial training store must contain only the frozen handoff shards.')
    if require_seeds or (fresh and configs['builder'].get('bootstrap_h5_path')):
        seed_path = local_path(root, configs['builder']['bootstrap_h5_path'])
        if not seed_path.exists():
            if require_seeds:
                raise FileNotFoundError(f'Generate the seed geometries first: {seed_path}')
            report['seed_generation_required'] = True
        else:
            from alframework.builders.h5_replay_builder import build_h5_replay_structures
            provenance = read_json(seed_path.parent / 'provenance.json')
            for name, digest in provenance['artifacts_sha256'].items():
                if fingerprint(seed_path.parent / name)['sha256'] != digest:
                    raise RuntimeError(f'Seed preparation fingerprint changed: {name}')
            if provenance['selected_count'] != master['bootstrap_set']:
                raise ValueError('bootstrap_set must match the number of selected seeds.')
            build_h5_replay_structures(moleculeids=['mol-boot-0000000000'], builder_config=configs['builder'],
                sampler_config=sampler, h5_path=master['h5_path'], current_h5_id=0, master_directory=str(root))
            report['selected_seeds'] = provenance['selected_count']
    if status_path.exists():
        status = read_json(status_path)
        if status['current_model_id'] < 0 and list(local_path(root, master['model_path']).parent.glob('model-*')):
            raise ValueError('Initial ensemble has not been promoted; training is running or incomplete. Inspect before restart.')
    if require_trained:
        report['initial_training'] = verify_initial_training(root, master)
    elif fresh and list(local_path(root, master['model_path']).parent.glob('model-*')):
        raise ValueError('Fresh run contains model directories; inspect incomplete training before restart.')
    if status_path.exists():
        status = read_json(status_path)
        report['last_dataset_screening'] = status.get('last_dataset_screening')
        report['failed_QM_tasks'] = status.get('lifetime_failed_QM_tasks', 0)
        report['current_model_id'] = status['current_model_id']
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='Prepare a fresh ALF run; seed generation remains an explicit command.')
    init.add_argument('--output-dir', required=True)
    for name in ('handoff','snapshot'):
        command = commands.add_parser(name, help='Freeze exact seed-inclusive rows from completed labeled shards.')
        command.add_argument('--source-master', required=True)
        command.add_argument('--output-dir', required=True)
        command.add_argument('--frames', type=int, **({'default':40000} if name=='handoff' else {'required':True}))
        command.add_argument('--closed-before', type=int)
    status = commands.add_parser('check', help='Validate configuration and report counts without starting Parsl.')
    status.add_argument('--master', required=True)
    status.add_argument('--closed-before', type=int)
    status.add_argument('--require-trained', action='store_true', help='Check all initial GALF ensemble artifacts and 40k input summaries.')
    status.add_argument('--require-seeds', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'init':
        result = initialize(args.output_dir)
    elif args.command in ('handoff','snapshot'):
        result = freeze(args.source_master, args.output_dir, args.frames, handoff=args.command=='handoff', closed_before=args.closed_before)
        result = {key: result[key] for key in ('frames','available_valid_frames','overshoot','phase_counts')}
    else:
        result = check(args.master, closed_before=args.closed_before, require_trained=args.require_trained, require_seeds=args.require_seeds)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
