"""Annotate an existing ALF HDF5 shard with stable topology atom IDs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from alframework.tools.molecular_topology import (
    H5_TOPOLOGY_ATOM_IDS_KEY,
    load_fixed_topology,
    validate_fixed_topology,
)


def _numbers_from_species(values) -> np.ndarray:
    result = []
    for value in np.asarray(values).reshape(-1).tolist():
        text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        result.append(int(atomic_numbers[text]))
    return np.asarray(result, dtype=np.int64)


def annotate_h5(path: Path, sampler_config: dict, *, validate_all: bool = True) -> dict:
    topology = load_fixed_topology(sampler_config)
    if topology is None:
        raise ValueError("Topology checking must be enabled before annotating an HDF5 shard.")
    reference_numbers = np.asarray(topology.atomic_numbers, dtype=np.int64)
    legacy_sorted_ids = np.argsort(reference_numbers)
    summary = {"path": str(path), "groups": {}, "frames": 0}
    with h5py.File(path, "r+") as handle:
        for group_name, group in handle.items():
            if "coordinates" not in group or "species" not in group:
                continue
            stored_numbers = _numbers_from_species(group["species"][()])
            if np.array_equal(stored_numbers, reference_numbers):
                atom_ids = np.arange(topology.n_atoms, dtype=np.int64)
            elif np.array_equal(stored_numbers, reference_numbers[legacy_sorted_ids]):
                atom_ids = np.asarray(legacy_sorted_ids, dtype=np.int64)
            else:
                raise ValueError(
                    f"HDF5 group {group_name!r} species do not match the topology reference order "
                    "or ALF's legacy atomic-number sort order."
                )
            n_frames = int(group["coordinates"].shape[0])
            values = np.broadcast_to(atom_ids, (n_frames, topology.n_atoms)).copy()
            if H5_TOPOLOGY_ATOM_IDS_KEY in group:
                existing = np.asarray(group[H5_TOPOLOGY_ATOM_IDS_KEY])
                if not np.array_equal(existing, values):
                    raise ValueError(f"Existing topology atom IDs disagree in HDF5 group {group_name!r}.")
            else:
                group.create_dataset(
                    H5_TOPOLOGY_ATOM_IDS_KEY,
                    data=values,
                    compression="gzip",
                    compression_opts=6,
                )
            failures = 0
            if validate_all:
                for frame in np.asarray(group["coordinates"]):
                    atoms = Atoms(numbers=stored_numbers, positions=frame)
                    atoms.set_array(
                        str(sampler_config["topology_check"].get("atom_id_array", "alf_topology_atom_id")),
                        atom_ids,
                    )
                    if not validate_fixed_topology(atoms, sampler_config, topology).valid:
                        failures += 1
            summary["groups"][str(group_name)] = {
                "frames": n_frames,
                "topology_failures": failures,
                "atom_ids": atom_ids.tolist(),
            }
            summary["frames"] += n_frames
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="HDF5 shard to annotate in place")
    parser.add_argument("--sampler", required=True, help="Sampler configuration JSON")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    sampler_path = Path(args.sampler).expanduser().resolve()
    h5_path = Path(args.h5).expanduser().resolve()
    sampler_config = json.loads(sampler_path.read_text(encoding="utf-8"))
    previous = Path.cwd()
    try:
        os.chdir(sampler_path.parent)
        summary = annotate_h5(h5_path, sampler_config, validate_all=not args.skip_validation)
    finally:
        os.chdir(previous)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
