"""Preflight an ALF run that uses configurable fixed molecular topology."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from alframework.tools.molecular_topology import (
    H5_TOPOLOGY_ATOM_IDS_KEY,
    load_fixed_topology,
    topology_atom_id_array,
    topology_config,
    validate_fixed_topology,
)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_run_path(run_dir: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = run_dir / path
    return path.resolve()


def _numbers_from_species(values) -> np.ndarray:
    numbers = []
    for value in np.asarray(values).reshape(-1).tolist():
        symbol = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        numbers.append(int(atomic_numbers[symbol]))
    return np.asarray(numbers, dtype=np.int64)


def _minimum_distance(positions: np.ndarray) -> float:
    delta = positions[:, :, None, :] - positions[:, None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    diagonal = np.arange(positions.shape[1])
    distances[:, diagonal, diagonal] = np.inf
    return float(np.min(distances))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_frames(
    positions: np.ndarray,
    numbers: np.ndarray,
    atom_ids: np.ndarray,
    sampler_config: dict[str, Any],
    topology,
) -> tuple[int, dict[str, float]]:
    failure_count = 0
    extrema = {
        "minimum_bond_ratio": np.inf,
        "maximum_bond_ratio": -np.inf,
        "minimum_bond_deviation_A": np.inf,
        "maximum_bond_deviation_A": -np.inf,
        "minimum_nonbonded_covalent_ratio": np.inf,
    }
    array_name = topology_atom_id_array(sampler_config)
    for coordinates in positions:
        atoms = Atoms(numbers=numbers, positions=coordinates)
        atoms.set_array(array_name, np.asarray(atom_ids, dtype=np.int64))
        result = validate_fixed_topology(atoms, sampler_config, topology)
        if not result.valid:
            failure_count += 1
        if result.metrics:
            extrema["minimum_bond_ratio"] = min(
                extrema["minimum_bond_ratio"],
                float(result.metrics["topology_min_bond_ratio"]),
            )
            extrema["maximum_bond_ratio"] = max(
                extrema["maximum_bond_ratio"],
                float(result.metrics["topology_max_bond_ratio"]),
            )
            extrema["minimum_bond_deviation_A"] = min(
                extrema["minimum_bond_deviation_A"],
                float(result.metrics["topology_min_bond_deviation_A"]),
            )
            extrema["maximum_bond_deviation_A"] = max(
                extrema["maximum_bond_deviation_A"],
                float(result.metrics["topology_max_bond_deviation_A"]),
            )
            extrema["minimum_nonbonded_covalent_ratio"] = min(
                extrema["minimum_nonbonded_covalent_ratio"],
                float(result.metrics["topology_min_nonbonded_covalent_ratio"]),
            )
    return int(failure_count), {key: float(value) for key, value in extrema.items()}


def _write_or_validate_state(path: Path, state: dict[str, Any]) -> None:
    if path.exists():
        current = _read_json(path)
        exact_keys = (
            "schema_version",
            "name",
            "reference_path",
            "reference_sha256",
            "reference_format",
            "reference_charge",
            "atomic_numbers",
            "atom_ids",
            "bonds",
        )
        exact_match = all(current.get(key) == state.get(key) for key in exact_keys)
        current_lengths = np.asarray(current.get("bond_references", []), dtype=np.float64)
        state_lengths = np.asarray(state.get("bond_references", []), dtype=np.float64)
        length_match = current_lengths.shape == state_lengths.shape and np.allclose(
            current_lengths,
            state_lengths,
            rtol=0.0,
            atol=1.0e-12,
        )
        if not exact_match or not length_match:
            raise ValueError(
                f"Serialized topology state disagrees with the configured reference: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
        handle.write("\n")


def run_preflight(
    run_dir: Path,
    master_path: Path,
    *,
    expected_atoms: int = 15,
    expected_bonds: int = 14,
    state_out: Path | None = None,
    require_model: bool = False,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    master_path = master_path.expanduser().resolve()
    master_config = _read_json(master_path)
    sampler_path = _resolve_run_path(run_dir, master_config["sampler_config_path"])
    builder_path = _resolve_run_path(run_dir, master_config["builder_config_path"])
    sampler_config = _read_json(sampler_path)
    builder_config = _read_json(builder_path)

    previous_cwd = Path.cwd()
    try:
        os.chdir(run_dir)
        topology = load_fixed_topology(sampler_config)
    finally:
        os.chdir(previous_cwd)
    if topology is None:
        raise ValueError("topology_check.enabled must be true for topology preflight.")
    if topology.n_atoms != int(expected_atoms) or len(topology.bonds) != int(expected_bonds):
        raise ValueError(
            f"Expected {expected_atoms} atoms and {expected_bonds} bonds, found "
            f"{topology.n_atoms} atoms and {len(topology.bonds)} bonds."
        )

    state_path = state_out or run_dir / "topology_reference_state.json"
    _write_or_validate_state(state_path.resolve(), topology.to_state())

    h5_path = _resolve_run_path(run_dir, str(master_config["h5_path"]).format(0))
    model_path = _resolve_run_path(run_dir, str(master_config["model_path"]).format(0))
    if not h5_path.is_file():
        raise FileNotFoundError(f"Seed HDF5 shard is missing: {h5_path}")
    model_present = model_path.is_dir()
    if require_model and not model_present:
        raise FileNotFoundError(f"Seed model directory is missing: {model_path}")

    all_shards = sorted(h5_path.parent.glob("data-*.h5"))
    later_shards = [path for path in all_shards if path.resolve() != h5_path]
    expected_shard_names = [f"data-{index:04d}.h5" for index in range(len(all_shards))]
    if [path.name for path in all_shards] != expected_shard_names:
        raise ValueError(
            "HDF5 shards must be sequential with no gaps: "
            f"expected {expected_shard_names}, found {[path.name for path in all_shards]}."
        )
    all_models = sorted(path for path in model_path.parent.glob("model-*") if path.is_dir())
    later_models = [path for path in all_models if path.resolve() != model_path]

    required_h5_keys = {
        "coordinates",
        "species",
        H5_TOPOLOGY_ATOM_IDS_KEY,
        *(str(spec[0]) for spec in master_config["properties_list"].values()),
    }
    for shard_path in all_shards:
        with h5py.File(shard_path, "r") as shard:
            for group_name, group in shard.items():
                if "coordinates" not in group or "species" not in group:
                    continue
                missing = sorted(required_h5_keys.difference(group.keys()))
                if missing:
                    raise ValueError(
                        f"HDF5 shard {shard_path.name}, group {group_name!r}, lacks required "
                        f"six-state datasets: {missing}."
                    )
                coordinates_shape = tuple(group["coordinates"].shape)
                topology_ids_shape = tuple(group[H5_TOPOLOGY_ATOM_IDS_KEY].shape)
                if topology_ids_shape != coordinates_shape[:2]:
                    raise ValueError(
                        f"HDF5 shard {shard_path.name} topology ID shape {topology_ids_shape} "
                        f"disagrees with coordinates {coordinates_shape}."
                    )

    h5_frames = 0
    h5_topology_failures = 0
    h5_minimum_distance = np.inf
    h5_maximum_force = -np.inf
    h5_extrema = {
        "minimum_bond_ratio": np.inf,
        "maximum_bond_ratio": -np.inf,
        "minimum_bond_deviation_A": np.inf,
        "maximum_bond_deviation_A": -np.inf,
        "minimum_nonbonded_covalent_ratio": np.inf,
    }
    with h5py.File(h5_path, "r") as handle:
        for group_name, group in handle.items():
            if "coordinates" not in group or "species" not in group:
                continue
            if H5_TOPOLOGY_ATOM_IDS_KEY not in group:
                raise ValueError(
                    f"HDF5 group {group_name!r} lacks stable topology IDs. "
                    "Run alframework.tools.prepare_topology_h5 first."
                )
            positions = np.asarray(group["coordinates"], dtype=np.float64)
            numbers = _numbers_from_species(group["species"][()])
            stored_ids = np.asarray(group[H5_TOPOLOGY_ATOM_IDS_KEY], dtype=np.int64)
            if stored_ids.shape != positions.shape[:2]:
                raise ValueError(
                    f"HDF5 topology ID shape {stored_ids.shape} disagrees with coordinates {positions.shape}."
                )
            if not np.all(stored_ids == stored_ids[0]):
                raise ValueError(f"Topology atom IDs change between frames in HDF5 group {group_name!r}.")
            failures, extrema = _validate_frames(
                positions,
                numbers,
                stored_ids[0],
                sampler_config,
                topology,
            )
            h5_topology_failures += failures
            h5_minimum_distance = min(h5_minimum_distance, _minimum_distance(positions))
            for key, value in extrema.items():
                if key.startswith("maximum"):
                    h5_extrema[key] = max(h5_extrema[key], value)
                else:
                    h5_extrema[key] = min(h5_extrema[key], value)
            for key, dataset in group.items():
                if not str(key).startswith("F"):
                    continue
                forces = np.asarray(dataset, dtype=np.float64)
                if forces.ndim == 3 and forces.shape[-1] == 3:
                    h5_maximum_force = max(
                        h5_maximum_force,
                        float(np.linalg.norm(forces, axis=-1).max()),
                    )
            h5_frames += int(positions.shape[0])

    min_distance_cutoff = float(sampler_config["min_distance_cutoff"])
    max_force_cutoff = float(sampler_config["max_force_cutoff"])
    if h5_topology_failures:
        raise ValueError(f"{h5_topology_failures} copied HDF5 seed frames fail fixed topology.")
    if h5_minimum_distance < min_distance_cutoff:
        raise ValueError(
            f"Copied HDF5 seed minimum distance {h5_minimum_distance:.8g} is below {min_distance_cutoff:.8g}."
        )
    if not np.isfinite(h5_maximum_force) or h5_maximum_force > max_force_cutoff:
        raise ValueError(
            f"Copied HDF5 seed maximum force {h5_maximum_force:.8g} exceeds {max_force_cutoff:.8g}."
        )

    seed_dir = Path(str(builder_config["seed_dataset_dir"])).expanduser().resolve()
    seed_names = dict(builder_config["seed_file_names"])
    seed_positions = np.asarray(np.load(seed_dir / seed_names["R"]), dtype=np.float64)
    seed_numbers_raw = np.asarray(np.load(seed_dir / seed_names["Z"]), dtype=np.int64)
    if seed_numbers_raw.ndim == 1:
        seed_numbers = seed_numbers_raw
    elif seed_numbers_raw.ndim == 2 and np.all(seed_numbers_raw == seed_numbers_raw[0]):
        seed_numbers = seed_numbers_raw[0]
    else:
        raise ValueError("Replay seed atomic numbers are not constant across frames.")
    if not np.array_equal(seed_numbers, np.asarray(topology.atomic_numbers, dtype=np.int64)):
        raise ValueError("Replay seed atom identities/species disagree with the topology reference order.")
    seed_ids = np.arange(topology.n_atoms, dtype=np.int64)
    seed_failures, seed_extrema = _validate_frames(
        seed_positions,
        seed_numbers,
        seed_ids,
        sampler_config,
        topology,
    )
    seed_minimum_distance = _minimum_distance(seed_positions)
    if seed_failures:
        raise ValueError(f"{seed_failures} replay source frames fail fixed topology.")
    if seed_minimum_distance < min_distance_cutoff:
        raise ValueError(
            f"Replay source minimum distance {seed_minimum_distance:.8g} is below {min_distance_cutoff:.8g}."
        )
    expected_seed_count = int(master_config.get("bootstrap_set", seed_positions.shape[0]))
    if h5_frames != expected_seed_count or seed_positions.shape[0] < expected_seed_count:
        raise ValueError(
            f"Expected {expected_seed_count} HDF5 seed frames and at least that many replay frames, "
            f"found replay={seed_positions.shape[0]}, HDF5={h5_frames}."
        )

    settings = topology_config(sampler_config)
    screening = dict(sampler_config.get("dataset_screening") or {})
    if not screening.get("enabled") or not all(
        bool(screening.get(name)) for name in ("min_distance", "force", "topology")
    ):
        raise ValueError("Dataset screening must enable min_distance, force, and topology.")
    if bool(sampler_config.get("max_nearest_neighbor_distance_check", False)):
        raise ValueError("max_nearest_neighbor_distance_check must be disabled for fixed-topology sampling.")

    try:
        import rdkit

        rdkit_version = str(rdkit.__version__)
    except ImportError as exc:  # pragma: no cover - topology loading fails first
        raise ImportError("RDKit is unavailable.") from exc

    summary = {
        "status": "passed",
        "rdkit_version": rdkit_version,
        "topology_name": topology.name,
        "reference_conformer_path": topology.reference_path,
        "reference_sha256": topology.reference_sha256,
        "reference_format": topology.reference_format,
        "reference_charge": topology.reference_charge,
        "atoms": topology.n_atoms,
        "fixed_bonds": len(topology.bonds),
        "topology_state_path": str(state_path.resolve()),
        "h5_path": str(h5_path),
        "h5_sha256": _sha256(h5_path),
        "model_path": str(model_path),
        "model_present": bool(model_present),
        "later_h5_shards": [str(path) for path in later_shards],
        "later_models": [str(path) for path in later_models],
        "replay_source_frames": int(seed_positions.shape[0]),
        "h5_frames": int(h5_frames),
        "minimum_distance_cutoff_A": min_distance_cutoff,
        "maximum_force_cutoff_eV_per_A": max_force_cutoff,
        "seed_minimum_distance_A": seed_minimum_distance,
        "h5_minimum_distance_A": float(h5_minimum_distance),
        "h5_maximum_force_eV_per_A": float(h5_maximum_force),
        "bond_window_mode": str(settings["bond_window_mode"]),
        "nonbonded_min_scale": float(settings["nonbonded_min_scale"]),
        "connectivity_scale": float(settings["connectivity_scale"]),
        "check_every_ncheck": bool(settings["check_every_ncheck"]),
        "Ncheck": int(sampler_config["Ncheck"]),
        "seed_topology_extrema": seed_extrema,
        "h5_topology_extrema": h5_extrema,
        "stable_h5_topology_ids": True,
    }
    if str(settings["bond_window_mode"]).strip().lower() == "linear":
        summary["bond_compression_tolerance_A"] = float(
            settings["bond_compression_tolerance_A"]
        )
        summary["bond_extension_tolerance_A"] = float(
            settings["bond_extension_tolerance_A"]
        )
    else:
        summary["bond_scale_window"] = [
            float(settings["bond_min_scale"]),
            float(settings["bond_max_scale"]),
        ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--master", required=True)
    parser.add_argument("--expected-atoms", type=int, default=15)
    parser.add_argument("--expected-bonds", type=int, default=14)
    parser.add_argument("--state-out", default=None)
    parser.add_argument(
        "--require-model",
        action="store_true",
        help="Fail if model-0000 is absent (normally omitted before clean --test_ml training).",
    )
    args = parser.parse_args()

    state_out = None if args.state_out is None else Path(args.state_out).expanduser().resolve()
    summary = run_preflight(
        Path(args.run_dir),
        Path(args.master),
        expected_atoms=args.expected_atoms,
        expected_bonds=args.expected_bonds,
        state_out=state_out,
        require_model=args.require_model,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
