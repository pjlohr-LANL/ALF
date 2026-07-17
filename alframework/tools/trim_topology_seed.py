"""Create matched HDF5 and NumPy seed data using ALF's three hard screens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from alframework.tools.molecular_topology import (
    H5_TOPOLOGY_ATOM_IDS_KEY,
    load_fixed_topology,
    topology_atom_id_array,
    validate_fixed_topology,
)


_BOOTSTRAP_ID = re.compile(r"mol-boot-(\d+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numbers_from_species(values) -> np.ndarray:
    numbers = []
    for value in np.asarray(values).reshape(-1).tolist():
        symbol = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        numbers.append(int(atomic_numbers[symbol]))
    return np.asarray(numbers, dtype=np.int64)


def _decode_id(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _source_index(value: Any) -> int:
    molecule_id = _decode_id(value)
    match = _BOOTSTRAP_ID.fullmatch(molecule_id)
    if match is None:
        raise ValueError(
            f"Cannot map HDF5 molecule ID {molecule_id!r} to an original bootstrap frame."
        )
    return int(match.group(1))


def _minimum_distance(positions: np.ndarray) -> float:
    if not np.isfinite(positions).all():
        return float("nan")
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def _json_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _dataset_kwargs(dataset: h5py.Dataset) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if dataset.compression is not None:
        kwargs["compression"] = dataset.compression
        kwargs["compression_opts"] = dataset.compression_opts
    if dataset.shuffle:
        kwargs["shuffle"] = True
    if dataset.fletcher32:
        kwargs["fletcher32"] = True
    if dataset.scaleoffset is not None:
        kwargs["scaleoffset"] = dataset.scaleoffset
    return kwargs


def _copy_filtered_group(
    source: h5py.Group,
    destination: h5py.Group,
    selected_rows: np.ndarray,
) -> None:
    for key, value in source.attrs.items():
        destination.attrs[key] = value
    frame_count = int(source["coordinates"].shape[0])
    for key, item in source.items():
        if not isinstance(item, h5py.Dataset):
            source.copy(item, destination, name=key)
            continue
        if item.ndim > 0 and int(item.shape[0]) == frame_count:
            data = np.asarray(item)[selected_rows]
            copied = destination.create_dataset(key, data=data, **_dataset_kwargs(item))
            for attr_key, attr_value in item.attrs.items():
                copied.attrs[attr_key] = attr_value
        else:
            source.copy(item, destination, name=key)


def trim_seed_dataset(
    *,
    source_h5: Path,
    source_r: Path,
    source_z: Path,
    output_h5: Path,
    output_r: Path,
    output_z: Path,
    report_path: Path,
    sampler_config: dict[str, Any],
    expected_accepted: int | None = None,
) -> dict[str, Any]:
    outputs = (output_h5, output_r, output_z, report_path)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite topology-trim outputs: {existing}")

    topology = load_fixed_topology(sampler_config)
    if topology is None:
        raise ValueError("topology_check.enabled must be true when trimming a topology seed.")
    min_distance_cutoff = float(sampler_config["min_distance_cutoff"])
    max_force_cutoff = float(sampler_config["max_force_cutoff"])
    array_name = topology_atom_id_array(sampler_config)
    source_positions = np.load(source_r, mmap_mode="r")
    source_numbers = np.load(source_z, mmap_mode="r")

    accepted: list[tuple[int, int]] = []
    rejected: list[dict[str, Any]] = []
    with h5py.File(source_h5, "r") as handle:
        group_names = [
            name
            for name, group in handle.items()
            if isinstance(group, h5py.Group) and "coordinates" in group and "species" in group
        ]
        if len(group_names) != 1:
            raise ValueError(
                "Matched seed trimming currently requires exactly one molecular HDF5 group; "
                f"found {group_names}."
            )
        group_name = group_names[0]
        group = handle[group_name]
        required = {"coordinates", "species", "_id", H5_TOPOLOGY_ATOM_IDS_KEY}
        missing = sorted(required.difference(group.keys()))
        if missing:
            raise ValueError(f"HDF5 group {group_name!r} is missing required datasets: {missing}")
        coordinates = np.asarray(group["coordinates"], dtype=np.float64)
        molecule_ids = np.asarray(group["_id"])
        topology_ids = np.asarray(group[H5_TOPOLOGY_ATOM_IDS_KEY], dtype=np.int64)
        numbers = _numbers_from_species(group["species"][()])
        force_keys = sorted(
            key
            for key, dataset in group.items()
            if str(key).startswith("F")
            and isinstance(dataset, h5py.Dataset)
            and dataset.ndim == 3
            and dataset.shape[-1] == 3
        )
        if not force_keys:
            raise ValueError(f"HDF5 group {group_name!r} has no per-state force datasets.")
        forces = {key: np.asarray(group[key], dtype=np.float64) for key in force_keys}

        for row, (positions, molecule_id, atom_ids) in enumerate(
            zip(coordinates, molecule_ids, topology_ids)
        ):
            source_index = _source_index(molecule_id)
            if source_index < 0 or source_index >= int(source_positions.shape[0]):
                raise IndexError(
                    f"Source index {source_index} from {_decode_id(molecule_id)!r} is outside R/Z data."
                )
            atoms = Atoms(numbers=numbers, positions=positions)
            atoms.set_array(array_name, np.asarray(atom_ids, dtype=np.int64))
            topology_result = validate_fixed_topology(atoms, sampler_config, topology)
            minimum_distance = _minimum_distance(positions)
            maximum_force = max(
                float(np.linalg.norm(values[row], axis=-1).max())
                if np.isfinite(values[row]).all()
                else float("inf")
                for values in forces.values()
            )
            triggered: list[str] = []
            if np.isfinite(minimum_distance) and minimum_distance < min_distance_cutoff:
                triggered.append("min_distance")
            if not np.isfinite(maximum_force) or maximum_force > max_force_cutoff:
                triggered.append("max_force")
            if not topology_result.valid:
                triggered.append("topology")
            if triggered:
                rejected.append(
                    {
                        "molecule_id": _decode_id(molecule_id),
                        "source_index": source_index,
                        "primary_reason": triggered[0],
                        "triggered_screens": triggered,
                        "minimum_distance_A": _json_float(minimum_distance),
                        "maximum_force_eV_per_A": _json_float(maximum_force),
                        "topology_reason": topology_result.reason,
                        "topology_violations": [dict(item) for item in topology_result.violations],
                    }
                )
            else:
                accepted.append((source_index, row))

        accepted.sort(key=lambda item: item[0])
        source_indices = np.asarray([item[0] for item in accepted], dtype=np.int64)
        selected_rows = np.asarray([item[1] for item in accepted], dtype=np.int64)
        if len(set(source_indices.tolist())) != len(source_indices):
            raise ValueError("Accepted HDF5 molecule IDs do not map to unique source indices.")
        if expected_accepted is not None and len(accepted) != int(expected_accepted):
            raise ValueError(
                f"Expected {expected_accepted} accepted frames, found {len(accepted)}. "
                "No output files were written."
            )

        selected_r = np.asarray(source_positions[source_indices])
        selected_z = np.asarray(source_numbers[source_indices])
        for output in outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
        h5_tmp = output_h5.with_name(output_h5.name + ".tmp")
        with h5py.File(h5_tmp, "w") as destination:
            for key, value in handle.attrs.items():
                destination.attrs[key] = value
            destination_group = destination.create_group(group_name)
            _copy_filtered_group(group, destination_group, selected_rows)
        os.replace(h5_tmp, output_h5)

    for array, output in ((selected_r, output_r), (selected_z, output_z)):
        temp = output.with_name(output.name + ".tmp")
        with open(temp, "wb") as stream:
            np.save(stream, array)
        os.replace(temp, output)

    with h5py.File(output_h5, "r") as handle:
        output_group = handle[group_name]
        output_coordinates = np.asarray(output_group["coordinates"], dtype=np.float64)
        output_ids = np.asarray(output_group[H5_TOPOLOGY_ATOM_IDS_KEY], dtype=np.int64)
        for row, (positions, atom_ids) in enumerate(zip(output_coordinates, output_ids)):
            by_id = np.empty(topology.n_atoms, dtype=np.int64)
            by_id[atom_ids] = np.arange(topology.n_atoms, dtype=np.int64)
            if not np.array_equal(positions[by_id], selected_r[row]):
                raise ValueError(f"Filtered HDF5 and replay R coordinates disagree at row {row}.")

    reason_counts: dict[str, int] = {}
    for record in rejected:
        reason = str(record["primary_reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    report = {
        "source_h5": str(source_h5),
        "source_h5_sha256": _sha256(source_h5),
        "source_r": str(source_r),
        "source_z": str(source_z),
        "output_h5": str(output_h5),
        "output_h5_sha256": _sha256(output_h5),
        "output_r": str(output_r),
        "output_r_sha256": _sha256(output_r),
        "output_z": str(output_z),
        "output_z_sha256": _sha256(output_z),
        "source_frames": int(len(accepted) + len(rejected)),
        "accepted_frames": int(len(accepted)),
        "rejected_frames": int(len(rejected)),
        "primary_reason_counts": reason_counts,
        "thresholds": {
            "minimum_distance_A": min_distance_cutoff,
            "maximum_force_eV_per_A": max_force_cutoff,
            "bond_window_mode": sampler_config["topology_check"].get(
                "bond_window_mode", "scale"
            ),
            "bond_compression_tolerance_A": sampler_config["topology_check"].get(
                "bond_compression_tolerance_A"
            ),
            "bond_extension_tolerance_A": sampler_config["topology_check"].get(
                "bond_extension_tolerance_A"
            ),
            "nonbonded_min_scale": sampler_config["topology_check"]["nonbonded_min_scale"],
            "connectivity_scale": sampler_config["topology_check"]["connectivity_scale"],
        },
        "rejected": rejected,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--source-r", required=True)
    parser.add_argument("--source-z", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--output-r", required=True)
    parser.add_argument("--output-z", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--sampler", required=True)
    parser.add_argument("--expected-accepted", type=int, default=None)
    args = parser.parse_args()

    sampler_path = Path(args.sampler).expanduser().resolve()
    sampler_config = json.loads(sampler_path.read_text(encoding="utf-8"))
    previous_cwd = Path.cwd()
    try:
        os.chdir(sampler_path.parent)
        report = trim_seed_dataset(
            source_h5=Path(args.source_h5).expanduser().resolve(),
            source_r=Path(args.source_r).expanduser().resolve(),
            source_z=Path(args.source_z).expanduser().resolve(),
            output_h5=Path(args.output_h5).expanduser().resolve(),
            output_r=Path(args.output_r).expanduser().resolve(),
            output_z=Path(args.output_z).expanduser().resolve(),
            report_path=Path(args.report).expanduser().resolve(),
            sampler_config=sampler_config,
            expected_accepted=args.expected_accepted,
        )
    finally:
        os.chdir(previous_cwd)
    print(json.dumps({key: value for key, value in report.items() if key != "rejected"}, indent=2))


if __name__ == "__main__":
    main()
