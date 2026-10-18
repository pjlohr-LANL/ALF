"""Frozen, seed-inclusive label budgets for the exact-40k GALF experiment.

These helpers are deliberately independent of Parsl and model inference. Only
runs with ``accepted_frame_budget`` enabled use the master-loop hooks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(path):
    path = Path(path).resolve()
    before = path.stat()
    result = {'path': str(path), 'size': before.st_size, 'sha256': sha256(path)}
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Source changed while reading: {path}')
    return result


def verify_fingerprint(record):
    if fingerprint(record['path']) != {k: record[k] for k in ('path', 'size', 'sha256')}:
        raise RuntimeError(f"Frozen source changed: {record['path']}")


def run_path(run, value):
    return Path(run) / value


def select_frames(run, limit=None, *, closed_before=None, handoff_frames=None):
    """Select finite stored rows in numeric-shard/row order, without extra filters.

    Coordinates are returned in topology-ID order; source rows and stored atom
    permutations are retained. Stored files are never changed. The H5 copier
    below preserves their original atom order and every dataset/attribute.
    """
    run = Path(run).resolve()
    master = read_json(run / 'master_config.json')
    status_path = run_path(run, master['status_path'])
    if closed_before is None and status_path.exists():
        closed_before = read_json(status_path)['current_h5_id']
    paths = sorted(run_path(run, master['h5_path']).parent.glob('data-*.h5'),
                   key=lambda p: int(p.stem.split('-')[1]))
    pieces, records, count, canonical_numbers = [], [], 0, None
    atomic_numbers = {'H': 1, 'C': 6, 'O': 8}
    for path in paths:
        shard = int(path.stem.split('-')[1])
        if closed_before is not None and shard >= closed_before:
            continue
        if limit is not None and count >= limit:
            break
        record = fingerprint(path)
        with h5py.File(path) as handle:
            if len(handle) != 1:
                raise ValueError(f'Expected one molecular group: {path}')
            group_name, group = next(iter(handle.items()))
            coordinates = group['coordinates'][:]
            if coordinates.ndim != 3 or coordinates.shape[2] != 3:
                raise ValueError(f'Invalid coordinates shape: {path}')
            n, atoms, _ = coordinates.shape
            energies = np.column_stack([group[f'sE{i}'][:].reshape(n) for i in range(4)])
            keep = np.isfinite(coordinates).all((1, 2)) & np.isfinite(energies).all(1)
            for i in range(4):
                forces = group[f'F{i}'][:]
                if forces.shape != (n, atoms, 3):
                    raise ValueError(f'Invalid F{i} shape: {path}')
                keep &= np.isfinite(forces).all((1, 2))
            identities = group['topology_atom_ids'][:]
            if identities.shape != (n, atoms) or not np.equal(np.sort(identities, axis=1), np.arange(atoms)).all():
                raise ValueError(f'Invalid topology-ID permutation: {path}')
            species = group['species'][:]
            if species.shape != (atoms,):
                raise ValueError(f'Invalid species shape: {path}')
            numbers = np.array([atomic_numbers[x.decode() if isinstance(x, bytes) else str(x)] for x in species])
            order = np.argsort(identities, axis=1)
            ordered_numbers = np.take_along_axis(np.broadcast_to(numbers, (n, atoms)), order, axis=1)
            if n and (not np.equal(ordered_numbers, ordered_numbers[0]).all() or
                      (canonical_numbers is not None and not np.array_equal(ordered_numbers[0], canonical_numbers))):
                raise ValueError(f'Canonical species order changed: {path}')
            if n:
                canonical_numbers = ordered_numbers[0]
            ids = group['_id'].asstr()[:].astype(str)
            if ids.shape != (n,):
                raise ValueError(f'ID/coordinate alignment failed: {path}')
            rows = np.flatnonzero(keep)
            if limit is not None:
                rows = rows[:limit - count]
            record.update(group=group_name, shard=shard, frames_available=n,
                          rejected_nonfinite=int((~keep).sum()), rows=rows.tolist())
            records.append(record)
            if len(rows):
                canonical = np.take_along_axis(coordinates[rows], order[rows, :, None], axis=1)
                pieces.append({'coordinates': canonical, 'energies': energies[rows], 'ids': ids[rows],
                               'stored_topology_atom_ids': identities[rows],
                               'shards': np.full(len(rows), shard, dtype=np.int64), 'rows': rows})
                count += len(rows)
        verify_fingerprint(record)
    if limit is not None and count < limit:
        raise ValueError(f'{run.name}: needs {limit} finite frames, has {count}')
    if not pieces:
        raise ValueError(f'No finite stored frames in {run}')
    arrays = {name: np.concatenate([p[name] for p in pieces]) for name in pieces[0]}
    arrays['numbers'] = canonical_numbers
    generations = []
    for identifier in arrays['ids']:
        fields = identifier.split('-')
        if len(fields) < 3 or fields[0] != 'mol':
            raise ValueError(f'Unrecognized frame ID: {identifier}')
        generations.append(-1 if fields[1] == 'boot' else int(fields[1]))
    arrays['generations'] = np.array(generations, dtype=np.int64)
    arrays['phases'] = np.where(arrays['generations'] == -1, 'seed', 'alf').astype('U4')
    if handoff_frames is not None:
        arrays['phases'][handoff_frames:] = 'galf'
    manifest = {'version': 1, 'run': str(run), 'frames': count, 'sources': records,
                'selection': 'completed numeric shards; stored row order; finite coordinates and four E/F labels; no other filters',
                'handoff_frames': handoff_frames}
    return arrays, manifest


def freeze_selection(run, count, directory, *, closed_before=None, handoff_frames=None):
    arrays, manifest = select_frames(run, count, closed_before=closed_before, handoff_frames=handoff_frames)
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        old = read_json(manifest_path)
        if {k: v for k, v in old.items() if k != 'arrays_sha256'} != manifest:
            raise RuntimeError(f'Refusing to replace frozen selection: {directory}')
        cached = load_selection(directory)
        for name in arrays:
            if not np.array_equal(cached[name], arrays[name]):
                raise RuntimeError(f'Frozen array mismatch: {directory}/{name}')
        return cached, old
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / 'arrays.npz', **arrays)
    manifest['arrays_sha256'] = sha256(directory / 'arrays.npz')
    write_json(manifest_path, manifest)
    return arrays, manifest


def load_selection(directory, *, verify_sources=True):
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    if sha256(directory / 'arrays.npz') != manifest['arrays_sha256']:
        raise RuntimeError(f'Frozen cache changed: {directory}')
    if verify_sources:
        for record in manifest['sources']:
            verify_fingerprint(record)
    with np.load(directory / 'arrays.npz', allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def copy_selected_shard(record, destination):
    """Slice every frame dataset, preserving species, dtypes, and attributes."""
    verify_fingerprint(record)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    rows = np.asarray(record['rows'], dtype=np.int64)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if np.array_equal(rows, np.arange(record['frames_available'])):
        shutil.copy2(record['path'], destination)
        copied = fingerprint(destination)
        if copied['sha256'] != record['sha256']:
            raise RuntimeError(f'Copy mismatch: {destination}')
        verify_fingerprint(record)
        return copied
    with h5py.File(record['path']) as src, h5py.File(destination, 'x') as dst:
        dst.attrs.update(src.attrs)
        source = src[record['group']]
        target = dst.create_group(record['group'])
        target.attrs.update(source.attrs)
        for name, dataset in source.items():
            if name == 'species' or dataset.ndim == 0:
                data = dataset[()]
            elif dataset.shape[0] == record['frames_available']:
                data = dataset[rows]
            else:
                raise ValueError(f'Unclassified H5 dataset: {dataset.name}')
            copied = target.create_dataset(name, data=data, dtype=dataset.dtype)
            copied.attrs.update(dataset.attrs)
    verify_fingerprint(record)
    with h5py.File(record['path']) as src, h5py.File(destination) as dst:
        for name, dataset in src[record['group']].items():
            expected = dataset[()] if name == 'species' or dataset.ndim == 0 else dataset[rows]
            if not np.array_equal(expected, dst[record['group']][name][()]):
                raise RuntimeError(f'Copy mismatch: {destination}/{name}')
    return fingerprint(destination)


def verify_handoff_inputs(run):
    run = Path(run).resolve()
    provenance = read_json(run / 'budgeted_galf_provenance.json')
    for record in provenance['copied_h5']:
        verify_fingerprint(record)
    master = read_json(run / 'master_config.json')
    # PyAniDirectoryDB reads every *.h5 file, including nonstandard names.
    files = list(run_path(run, master['h5_path']).parent.glob('*.h5'))
    if len(files) != len(provenance['copied_h5']):
        raise RuntimeError('Initial training must see only the frozen handoff shards')
    arrays, _ = select_frames(run, provenance['handoff_frames'], closed_before=len(files))
    if len(arrays['ids']) != provenance['handoff_frames']:
        raise RuntimeError('Handoff frame count mismatch')
    return provenance


def verify_initial_ensemble(run):
    """Fail closed before sampling; incomplete model directories are insufficient."""
    run = Path(run).resolve()
    master = read_json(run / 'master_config.json')
    provenance = read_json(run / 'budgeted_galf_provenance.json')
    report_path = run / 'initial_ensemble_verified.json'
    if report_path.exists():
        report = read_json(report_path)
        for member in report['members']:
            for record in member['artifacts']:
                verify_fingerprint(record)
        for record in provenance['copied_h5']:
            verify_fingerprint(record)
        return report
    verify_handoff_inputs(run)
    config = read_json(run_path(run, master['ML_config_path']))
    model = run_path(run, master['model_path'].format(0))
    members = sorted(model.glob('model-*'))
    if [p.name for p in members] != [f'model-{i:02d}' for i in range(config['n_models'])]:
        raise RuntimeError('Initial ensemble is incomplete; all four fresh members are required')
    records = []
    for member in members:
        log = member / 'training_log.txt'
        summary = read_json(member / 'training_summary.json')
        if 'Training complete' not in log.read_text():
            raise RuntimeError(f'Initial member did not complete: {member}')
        if summary['dataset_frames'] != provenance['handoff_frames'] or sum(summary['split_frames'].values()) != provenance['handoff_frames']:
            raise RuntimeError(f'Initial member did not use the exact handoff pool: {member}')
        if summary['gap_target_mode'] != 'disabled' or not summary['train_forces']:
            raise RuntimeError(f'Initial member has unexpected training objective: {member}')
        artifacts = [fingerprint(member / name) for name in
                     ('training_summary.json', 'training_log.txt', 'best_model.pt', 'best_checkpoint.pt', 'experiment_structure.pt')]
        records.append({'member': member.name, 'dataset_frames': summary['dataset_frames'],
                        'split_frames': summary['split_frames'], 'artifacts': artifacts})
    report = {'handoff_frames': provenance['handoff_frames'], 'members': records,
              'input_files': provenance['copied_h5'], 'fresh_training': True}
    write_json(report_path, report)
    return report


def update_budget_checkpoints(run, status, queues=None):
    """Called at startup and after an H5 save, never on the normal polling path."""
    run = Path(run).resolve()
    master = read_json(run / 'master_config.json')
    budget = master.get('accepted_frame_budget')
    if not budget:
        return False
    arrays, _ = select_frames(run, closed_before=status['current_h5_id'], handoff_frames=budget['handoff_frames'])
    count = len(arrays['ids'])
    for threshold in budget['checkpoints']:
        directory = run / 'budget_checkpoints' / str(threshold)
        if count >= threshold:
            if directory.exists():
                load_selection(directory)
            else:
                freeze_selection(run, threshold, directory, closed_before=status['current_h5_id'],
                                 handoff_frames=budget['handoff_frames'])
    complete = count >= budget['target_frames']
    report = {'accepted_frames': count, 'target_frames': budget['target_frames'],
              'handoff_frames': budget['handoff_frames'], 'new_frames': count - budget['handoff_frames'],
              'overshoot_frames': max(0, count - budget['target_frames']), 'complete': complete,
              'status': status, 'budget_unit': 'retained finite QM-labeled frames including seed; not QM attempts'}
    if complete and queues is not None:
        report['unharvested_tasks_before_executor_shutdown'] = {
            name: {'total': queue.get_number(), 'running': queue.get_running_number(),
                   'queued': queue.get_queued_number(), 'completed': queue.get_completed_number()}
            for name, queue in queues.items()}
    write_json(run / 'budget_progress.json', report)
    return complete
