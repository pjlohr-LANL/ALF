"""Atomic local workflow records and property-schema-driven ALF snapshots."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import h5py
import numpy as np
from ase.data import atomic_numbers

from alframework.tools.atom_order import canonical_orders

ROW_KEY = '_alf_workflow_row'
PHASE_KEY = '_alf_workflow_phase'


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def sync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest(), 'size': path.stat().st_size}


def verify(record):
    if fingerprint(record['path']) != {key: record[key] for key in ('path','sha256','size')}:
        raise RuntimeError(f"Workflow fingerprint changed: {record['path']}")


class ExclusiveLock:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+')
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            raise RuntimeError(f'Another controller owns {self.path}') from None
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(str(os.getpid()) + '\n')
        self.handle.flush()

    def close(self):
        if not self.handle.closed:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@contextmanager
def publish_directory(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing existing output: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=destination.parent))
    try:
        yield temporary
        os.rename(temporary, destination)
        sync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def shards(pattern, closed_before=None):
    pattern = Path(pattern).expanduser().resolve()
    found = re.fullmatch(r'(.*)\{(?::[^{}]*)?\}(.*)', pattern.name)
    if not found:
        raise ValueError('H5 path needs exactly one integer formatting field.')
    expression = re.compile(re.escape(found[1]) + r'(\d+)' + re.escape(found[2]))
    paths = []
    for path in pattern.parent.glob('*'):
        match = expression.fullmatch(path.name)
        if path.is_file() and match and (closed_before is None or int(match[1]) < closed_before):
            paths.append((int(match[1]), path))
    return sorted(paths)


def group_valid_rows(group, properties):
    xyz = group['coordinates']
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or xyz.shape[1] < 1:
        raise ValueError(f'Malformed coordinates: {group.name}')
    count, atoms, _ = xyz.shape
    species = group['species'][:]
    if species.shape != (atoms,):
        raise ValueError(f'Malformed species vector: {group.name}')
    species_numbers(group)
    if count==0:
        return np.empty(0,dtype=np.int64)
    keep = np.isfinite(xyz[:]).all((1,2))
    if 'cell' in group:
        cell = group['cell'][:]
        if cell.shape != (count,3,3):
            raise ValueError(f'Malformed cell: {group.name}')
        keep &= np.isfinite(cell).all((1,2))
    for key, schema in properties.items():
        name, scope = schema[:2]
        data = group[name][:]
        if data.ndim < 1 or data.shape[0] != count:
            raise ValueError(f'Malformed frame axis for {key}: {group.name}')
        if scope.lower() == 'atomic' and (data.ndim < 2 or data.shape[1] != atoms):
            raise ValueError(f'Malformed atom axis for {key}: {group.name}')
        if scope.lower() == 'pair_atomic' and (data.ndim < 3 or data.shape[2] != atoms):
            raise ValueError(f'Malformed pair/atom axes for {key}: {group.name}')
        if data.dtype.kind not in 'biufc':
            raise ValueError(f'Configured scientific property must be numeric: {key}')
        keep &= np.isfinite(data).reshape(count,-1).all(1)
    if '_id' not in group or group['_id'].shape != (count,):
        raise ValueError(f'Missing or malformed frame IDs: {group.name}')
    if 'topology_atom_ids' in group:
        canonical_orders(group['topology_atom_ids'][:], count, atoms, context=group.name)
    return np.flatnonzero(keep)



def species_numbers(group):
    species=group['species'][:]
    if species.dtype.kind in 'iu':
        numbers=species.astype(np.int64)
    elif species.dtype.kind in 'SUO':
        numbers=np.asarray([atomic_numbers[value.decode() if isinstance(value,bytes) else str(value)] for value in species],dtype=np.int64)
    else:
        raise ValueError(f'Invalid species representation: {group.name}')
    if np.any(numbers<1) or np.any(numbers>118):
        raise ValueError(f'Invalid atomic numbers: {group.name}')
    return numbers


def validate_reference_layout(manifest,reference_numbers):
    expected=np.asarray(reference_numbers)
    for record in manifest['sources']:
        verify(record)
        with h5py.File(record['path']) as handle:
            for detail in record['groups']:
                group=handle[detail['group']]
                numbers=species_numbers(group)
                if len(numbers)!=len(expected) or not np.array_equal(np.sort(numbers),np.sort(expected)):
                    raise ValueError('Inherited data composition disagrees with the configured topology reference.')
                if 'topology_atom_ids' in group:
                    order=canonical_orders(group['topology_atom_ids'][:],detail['frames_available'],len(numbers))
                    if not np.equal(numbers[order[detail['rows']]],expected).all():
                        raise ValueError('Inherited topology IDs disagree with reference atom identities.')
                elif np.any(np.diff(numbers)<0):
                    raise ValueError('Untagged inherited data must retain upstream atomic-number ordering; identities are ambiguous.')
                # Untagged legacy rows stay untagged; no identity permutation is fabricated.
        verify(record)

def scan_store(pattern, properties, *, count=None, closed_before=None):
    if count is not None and (isinstance(count,bool) or not isinstance(count,int) or count < 1):
        raise ValueError('Snapshot count must be a positive integer.')
    selected, available, invalid, sources = 0, 0, 0, []
    for index, path in shards(pattern, closed_before):
        record = fingerprint(path)
        groups = []
        with h5py.File(path) as handle:
            # ALF writers use sorted formula groups; explicit ordering is reproducible.
            for name in sorted(handle):
                group = handle[name]
                if not isinstance(group,h5py.Group) or 'coordinates' not in group:
                    raise ValueError(f'Unexpected non-molecular H5 entry: {path}:{name}')
                rows = group_valid_rows(group, properties)
                n = len(group['coordinates'])
                available += len(rows)
                invalid += n-len(rows)
                take = len(rows) if count is None else min(len(rows),count-selected)
                chosen = rows[:take]
                if len(chosen):
                    ids = group['_id'].asstr()[chosen].tolist()
                    phases = group[PHASE_KEY].asstr()[chosen].tolist() if PHASE_KEY in group else [
                        'seed' if str(value).startswith('mol-boot-') else 'inherited' for value in ids]
                    detail = {'group':name,'frames_available':n,'rows':chosen.tolist(),'ids':ids,'phases':phases}
                    if 'topology_atom_ids' in group:
                        identities=group['topology_atom_ids'][:]
                        if identities.ndim==1:
                            identities=np.broadcast_to(identities,(n,len(identities)))
                        permutations, indices=np.unique(identities[chosen],axis=0,return_inverse=True)
                        detail.update(topology_permutations=permutations.tolist(),topology_permutation_indices=indices.tolist())
                    groups.append(detail)
                    selected += len(chosen)
        verify(record)
        if groups:
            record.update(shard=index,groups=groups)
            sources.append(record)
    if count is not None and selected < count:
        raise ValueError(f'Need {count} finite retained frames; found {selected}.')
    phases = [phase for record in sources for group in record['groups'] for phase in group['phases']]
    return {'schema':'alf.snapshot.v1','frames':selected,'available_frames':available,'nonfinite_frames':invalid,
            'overshoot':available-selected,'properties':properties,'sources':sources,
            'phase_counts':{phase:phases.count(phase) for phase in sorted(set(phases))}}


def frame_data(dataset, key, rows, n, atoms):
    # These are static metadata even when a dimension happens to equal n.
    if (key in {'species','nacr_pair_index','nac_pair_index','pair_index','nac_pairs'} or dataset.ndim==0
            or (key=='topology_atom_ids' and dataset.shape==(atoms,))):
        return dataset[()]
    if dataset.shape[0] != n:
        raise ValueError(f'Ambiguous static/frame dataset: {dataset.name}')
    return dataset[rows]


def copy_snapshot_data(manifest, temporary, destination):
    (temporary/'h5store').mkdir(exist_ok=True)
    artifacts=[]
    for record in manifest['sources']:
        verify(record)
        name=f'h5store/data-{len(artifacts):04d}.h5'
        with h5py.File(record['path']) as src, h5py.File(temporary/name,'x') as dst:
            dst.attrs.update(src.attrs)
            for detail in record['groups']:
                original=src[detail['group']]
                group=dst.create_group(detail['group'])
                group.attrs.update(original.attrs)
                rows=np.asarray(detail['rows'],dtype=int)
                atoms=original['coordinates'].shape[1]
                for key,dataset in original.items():
                    data=frame_data(dataset,key,rows,detail['frames_available'],atoms)
                    copied=group.create_dataset(key,data=data,dtype=dataset.dtype)
                    copied.attrs.update(dataset.attrs)
                    if not np.array_equal(data,copied[()],equal_nan=np.asarray(data).dtype.kind in 'fc'):
                        raise RuntimeError(f'Snapshot copy mismatch: {key}')
        with (temporary/name).open('rb') as saved:
            os.fsync(saved.fileno())
        verify(record)
        artifact=fingerprint(temporary/name)
        artifact['relative_path']=name
        artifact['path']=str(destination/name)
        artifacts.append(artifact)
    sync_directory(temporary/'h5store')
    return artifacts


def verify_snapshot(directory):
    manifest=read_json(Path(directory)/'manifest.json')
    for artifact in manifest['artifacts']:
        verify(artifact)
    for record in manifest['sources']:
        verify(record)
    return manifest


def freeze_snapshot(pattern, properties, count, destination, *, closed_before=None):
    destination=Path(destination).resolve()
    if destination.exists():
        manifest=verify_snapshot(destination)
        if manifest['frames']!=count or manifest['properties']!=properties:
            raise RuntimeError('Existing checkpoint has incompatible count or properties.')
        return manifest
    manifest=scan_store(pattern,properties,count=count,closed_before=closed_before)
    with publish_directory(destination) as temporary:
        manifest['artifacts']=copy_snapshot_data(manifest,temporary,destination)
        atomic_json(temporary/'manifest.json',manifest)
    return manifest
