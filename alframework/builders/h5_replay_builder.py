"""Deterministic fixed-topology replay from existing ALF HDF5 shards."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from parsl import python_app

from alframework.tools.molecular_topology import load_reference_topology
from alframework.tools.molecules_class import MoleculesObject


_TOPOLOGY_ATOM_IDS_KEY = "topology_atom_ids"


@dataclass(frozen=True)
class H5ReplayGroup:
    """One replayable fixed-composition group within an ALF shard."""

    shard_path: str
    group_path: str
    n_frames: int
    start: int
    stop: int


def _stored_atomic_numbers(raw_species: Any, frame_index: int) -> np.ndarray:
    species = np.asarray(raw_species)
    if species.ndim == 2:
        species = species[int(frame_index)]
    if species.ndim != 1:
        raise ValueError(
            "HDF5 replay species must have shape [atoms] or "
            f"[structures, atoms]; received {species.shape}."
        )
    if species.dtype.kind in {"S", "U", "O"}:
        symbols = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in species.tolist()
        ]
        try:
            numbers = [int(atomic_numbers[symbol]) for symbol in symbols]
        except KeyError as exc:
            raise ValueError(
                f"HDF5 replay encountered an unknown chemical symbol: {exc.args[0]!r}."
            ) from exc
        return np.asarray(numbers, dtype=np.int64)
    numbers = np.asarray(species, dtype=np.int64)
    if np.any(numbers <= 0):
        raise ValueError("HDF5 replay species must be positive atomic numbers.")
    return numbers


def build_h5_replay_manifest(
    h5_path: str,
    current_h5_id: int,
) -> list[H5ReplayGroup]:
    """Return a deterministic flat index over existing HDF5 groups."""

    groups: list[H5ReplayGroup] = []
    running_total = 0
    for shard_id in range(int(current_h5_id)):
        shard_path = Path(str(h5_path).format(shard_id)).expanduser().resolve()
        if not shard_path.is_file():
            continue
        with h5py.File(shard_path, "r") as handle:
            for group_path in sorted(handle):
                group = handle[group_path]
                if not isinstance(group, h5py.Group):
                    continue
                if "coordinates" not in group or "species" not in group:
                    continue
                coordinates = group["coordinates"]
                if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
                    raise ValueError(
                        f"{shard_path}:{group_path}/coordinates must have shape "
                        f"[structures, atoms, 3]; received {coordinates.shape}."
                    )
                n_frames = int(coordinates.shape[0])
                if n_frames < 1:
                    continue
                groups.append(
                    H5ReplayGroup(
                        shard_path=str(shard_path),
                        group_path=str(group_path),
                        n_frames=n_frames,
                        start=running_total,
                        stop=running_total + n_frames,
                    )
                )
                running_total += n_frames
    if not groups:
        raise FileNotFoundError(
            "HDF5 replay found no usable groups below current_h5_id="
            f"{int(current_h5_id)} for pattern {str(h5_path)!r}."
        )
    return groups


def _replay_global_index(
    molecule_id: str,
    *,
    current_h5_id: int,
    frame_count: int,
    selection_seed: int,
) -> int:
    payload = (
        f"{str(molecule_id)}|{int(current_h5_id)}|{int(selection_seed)}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % int(
        frame_count
    )


def _canonical_reorder(
    *,
    stored_numbers: np.ndarray,
    topology_atom_ids: np.ndarray | None,
    reference_numbers: np.ndarray,
    context: str,
) -> np.ndarray:
    n_atoms = int(reference_numbers.size)
    if stored_numbers.shape != (n_atoms,):
        raise ValueError(
            f"{context} contains {stored_numbers.size} atoms, while the topology "
            f"reference contains {n_atoms}."
        )
    if topology_atom_ids is not None:
        atom_ids = np.asarray(topology_atom_ids, dtype=np.int64).reshape(-1)
        if (
            atom_ids.shape != (n_atoms,)
            or np.any(atom_ids < 0)
            or np.any(atom_ids >= n_atoms)
            or len(set(atom_ids.tolist())) != n_atoms
        ):
            raise ValueError(
                f"{context}:{_TOPOLOGY_ATOM_IDS_KEY} must be a permutation of "
                f"0..{n_atoms - 1}; received {atom_ids.tolist()}."
            )
        reorder = np.argsort(atom_ids, kind="stable")
        if not np.array_equal(stored_numbers[reorder], reference_numbers):
            raise ValueError(
                f"{context}:{_TOPOLOGY_ATOM_IDS_KEY} does not restore the "
                "topology reference atomic-number order."
            )
        return reorder

    storage_order = np.argsort(reference_numbers, kind="stable")
    expected_stored = reference_numbers[storage_order]
    if not np.array_equal(stored_numbers, expected_stored):
        raise ValueError(
            f"{context}:species does not match ALF's stable atomic-number "
            f"storage order. Expected {expected_stored.tolist()}, received "
            f"{stored_numbers.tolist()}."
        )
    return np.argsort(storage_order, kind="stable")


def _load_replay_frame(
    entry: H5ReplayGroup,
    frame_index: int,
    *,
    sampler_config: dict[str, Any],
    master_directory: str | None,
) -> tuple[Atoms, dict[str, Any]]:
    context = f"{entry.shard_path}:{entry.group_path}[{int(frame_index)}]"
    with h5py.File(entry.shard_path, "r") as handle:
        group = handle[entry.group_path]
        positions = np.asarray(
            group["coordinates"][int(frame_index)], dtype=np.float64
        )
        stored_numbers = _stored_atomic_numbers(
            group["species"][()], int(frame_index)
        )
        topology = load_reference_topology(
            sampler_config,
            master_directory=master_directory,
        )
        canonicalized = topology is not None
        if topology is None:
            reorder = np.arange(stored_numbers.size, dtype=np.int64)
        else:
            raw_ids = None
            if _TOPOLOGY_ATOM_IDS_KEY in group:
                raw_ids = np.asarray(group[_TOPOLOGY_ATOM_IDS_KEY])
                if raw_ids.ndim == 2:
                    raw_ids = raw_ids[int(frame_index)]
            reorder = _canonical_reorder(
                stored_numbers=stored_numbers,
                topology_atom_ids=raw_ids,
                reference_numbers=np.asarray(
                    topology.atomic_numbers, dtype=np.int64
                ),
                context=context,
            )
        atoms = Atoms(
            numbers=stored_numbers[reorder],
            positions=positions[reorder],
            pbc=False,
        )
    return atoms, {
        "replay_source_kind": "h5",
        "replay_source_path": str(entry.shard_path),
        "replay_group_path": str(entry.group_path),
        "replay_frame_index": int(frame_index),
        "replay_canonicalized_to_topology": bool(canonicalized),
    }


def build_h5_replay_structures(
    *,
    moleculeids: list[str],
    builder_config: dict[str, Any],
    sampler_config: dict[str, Any],
    h5_path: str,
    current_h5_id: int,
    master_directory: str | None = None,
) -> list[MoleculesObject]:
    """Build deterministic replay structures for one driver submission."""

    ids = [str(value) for value in moleculeids]
    if not ids:
        raise ValueError("HDF5 replay requires at least one molecule ID.")
    config = dict(builder_config or {})
    source_priority = str(
        config.get("source_priority", "h5_only")
    ).strip().lower()
    if source_priority != "h5_only":
        raise ValueError(
            "The streamlined HDF5 replay builder supports only "
            "builder_config.source_priority='h5_only'."
        )
    selection_seed = int(config.get("selection_seed", 42))
    manifest = build_h5_replay_manifest(h5_path, int(current_h5_id))
    frame_count = int(manifest[-1].stop)

    outputs: list[MoleculesObject] = []
    for molecule_id in ids:
        global_index = _replay_global_index(
            molecule_id,
            current_h5_id=int(current_h5_id),
            frame_count=frame_count,
            selection_seed=selection_seed,
        )
        entry = next(
            item
            for item in manifest
            if int(item.start) <= global_index < int(item.stop)
        )
        local_index = int(global_index - int(entry.start))
        atoms, metadata = _load_replay_frame(
            entry,
            local_index,
            sampler_config=dict(sampler_config or {}),
            master_directory=master_directory,
        )
        metadata["replay_global_frame_index"] = int(global_index)
        molecule = MoleculesObject(atoms, molecule_id)
        molecule.update_metadata(metadata)
        outputs.append(molecule)
    return outputs


@python_app(executors=["alf_sampler_executor"])
def h5_replay_builder_task(
    moleculeid=None,
    moleculeids=None,
    builder_config=None,
    sampler_config=None,
    h5_path=None,
    current_h5_id=0,
    master_directory=None,
):
    """Parsl entry point for fixed-topology HDF5 geometry replay."""

    ids = (
        [str(moleculeid)]
        if moleculeids is None
        else [str(value) for value in moleculeids]
    )
    outputs = build_h5_replay_structures(
        moleculeids=ids,
        builder_config=dict(builder_config or {}),
        sampler_config=dict(sampler_config or {}),
        h5_path=str(h5_path),
        current_h5_id=int(current_h5_id),
        master_directory=master_directory,
    )
    return outputs[0] if moleculeids is None else outputs
