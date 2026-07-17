from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from parsl import python_app

from alframework.tools.excited_state_tools import (
    derive_state_ids,
    empirical_formula_from_numbers,
    empirical_formula_from_species,
    select_excited_state,
    stable_uint32_seed,
    validate_gap_results,
)
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.molecular_topology import (
    H5_TOPOLOGY_ATOM_IDS_KEY,
    ensure_topology_atom_ids,
    load_fixed_topology,
    topology_atom_id_array,
    topology_enabled,
)


_MANIFEST_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def _normalize_formula_filter(formula_filter: str | list[str] | None) -> set[str] | None:
    if formula_filter is None:
        return None
    if isinstance(formula_filter, str):
        values = [formula_filter]
    else:
        values = [str(value) for value in formula_filter]
    normalized = {value.strip() for value in values if str(value).strip()}
    return normalized or None


def _seed_file_names(builder_config: dict[str, Any]) -> tuple[str, str]:
    names = dict(builder_config.get("seed_file_names") or {})
    return str(names.get("R", "R.npy")), str(names.get("Z", "Z.npy"))


def _prelabeled_seed_file_names(builder_config: dict[str, Any]) -> dict[str, str]:
    names = dict(builder_config.get("prelabeled_seed_file_names") or {})
    return {str(key): str(value) for key, value in names.items() if str(key).strip()}


def _manifest_cache_key(
    *,
    h5_path: str,
    current_h5_id: int,
    builder_config: dict[str, Any],
    formula_filter: set[str] | None,
) -> tuple[Any, ...]:
    seed_r, seed_z = _seed_file_names(builder_config)
    return (
        str(h5_path),
        int(current_h5_id),
        str(builder_config.get("seed_dataset_dir")),
        str(builder_config.get("source_priority", "h5_then_seed")),
        seed_r,
        seed_z,
        tuple(sorted(_prelabeled_seed_file_names(builder_config).items())),
        tuple(sorted(formula_filter)) if formula_filter else None,
    )


def _manifest_cache_path(
    *,
    current_h5_id: int,
    builder_config: dict[str, Any],
    formula_filter: set[str] | None,
) -> Path | None:
    cache_dir = builder_config.get("manifest_cache_dir")
    if not cache_dir:
        return None
    path = Path(str(cache_dir)).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(
        repr(
            (
                int(current_h5_id),
                str(builder_config.get("seed_dataset_dir")),
                _seed_file_names(builder_config),
                tuple(sorted(_prelabeled_seed_file_names(builder_config).items())),
                tuple(sorted(formula_filter)) if formula_filter else None,
            )
        ).encode("utf-8")
    ).hexdigest()[:12]
    return path / f"replay_manifest_{int(current_h5_id):04d}_{digest}.pkl"


def _load_manifest_from_cache(
    *,
    h5_path: str,
    current_h5_id: int,
    builder_config: dict[str, Any],
    formula_filter: set[str] | None,
) -> dict[str, Any] | None:
    cache_key = _manifest_cache_key(
        h5_path=h5_path,
        current_h5_id=current_h5_id,
        builder_config=builder_config,
        formula_filter=formula_filter,
    )
    if cache_key in _MANIFEST_CACHE:
        return _MANIFEST_CACHE[cache_key]

    cache_path = _manifest_cache_path(
        current_h5_id=current_h5_id,
        builder_config=builder_config,
        formula_filter=formula_filter,
    )
    if cache_path is not None and cache_path.exists():
        try:
            with open(cache_path, "rb") as handle:
                manifest = pickle.load(handle)
        except (EOFError, pickle.UnpicklingError, OSError):
            return None
        if not isinstance(manifest, dict) or "h5_groups" not in manifest or "seed" not in manifest:
            return None
        _MANIFEST_CACHE[cache_key] = manifest
        return manifest
    return None


def _store_manifest_in_cache(
    *,
    manifest: dict[str, Any],
    h5_path: str,
    current_h5_id: int,
    builder_config: dict[str, Any],
    formula_filter: set[str] | None,
) -> None:
    cache_key = _manifest_cache_key(
        h5_path=h5_path,
        current_h5_id=current_h5_id,
        builder_config=builder_config,
        formula_filter=formula_filter,
    )
    _MANIFEST_CACHE[cache_key] = manifest
    cache_path = _manifest_cache_path(
        current_h5_id=current_h5_id,
        builder_config=builder_config,
        formula_filter=formula_filter,
    )
    if cache_path is not None:
        with tempfile.NamedTemporaryFile("wb", dir=cache_path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            pickle.dump(manifest, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, cache_path)


def _build_h5_manifest(
    *,
    h5_path: str,
    current_h5_id: int,
    formula_filter: set[str] | None,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    running_total = 0
    for shard_id in range(int(current_h5_id)):
        shard_path = Path(h5_path.format(shard_id)).expanduser().resolve()
        if not shard_path.exists():
            continue
        with h5py.File(shard_path, "r") as handle:
            for group_name, group in handle.items():
                if "coordinates" not in group or "species" not in group:
                    continue
                species = np.asarray(group["species"][()])
                formula = empirical_formula_from_species(species)
                if formula_filter and formula not in formula_filter:
                    continue
                n_frames = int(group["coordinates"].shape[0])
                if n_frames <= 0:
                    continue
                groups.append(
                    {
                        "kind": "h5",
                        "path": str(shard_path),
                        "group_path": str(group_name),
                        "formula": formula,
                        "n_frames": n_frames,
                        "start": running_total,
                        "stop": running_total + n_frames,
                    }
                )
                running_total += n_frames
    return groups


def _build_seed_manifest(
    *,
    builder_config: dict[str, Any],
    formula_filter: set[str] | None,
) -> dict[str, Any] | None:
    seed_dataset_dir = builder_config.get("seed_dataset_dir")
    if not seed_dataset_dir:
        return None

    seed_root = Path(str(seed_dataset_dir)).expanduser().resolve()
    r_name, z_name = _seed_file_names(builder_config)
    r_path = seed_root / r_name
    z_path = seed_root / z_name
    if not r_path.exists() or not z_path.exists():
        return None

    R = np.load(r_path, mmap_mode="r")
    Z = np.load(z_path, mmap_mode="r")
    if R.shape[0] != Z.shape[0]:
        raise ValueError(f"Seed dataset R/Z frame counts do not match: {R.shape} vs {Z.shape}")

    prelabeled_paths: dict[str, str] = {}
    for prop_key, filename in _prelabeled_seed_file_names(builder_config).items():
        label_path = seed_root / filename
        if not label_path.exists():
            raise FileNotFoundError(f"Pre-labeled seed file for {prop_key!r} does not exist: {label_path}")
        label_array = np.load(label_path, mmap_mode="r")
        if int(label_array.shape[0]) != int(R.shape[0]):
            raise ValueError(
                f"Pre-labeled seed file {label_path} has {label_array.shape[0]} frames, "
                f"but seed R/Z have {R.shape[0]} frames."
            )
        prelabeled_paths[prop_key] = str(label_path)

    eligible_indices: np.ndarray | None = None
    if formula_filter:
        selected: list[int] = []
        for frame_idx in range(int(Z.shape[0])):
            formula = empirical_formula_from_numbers(np.asarray(Z[frame_idx]))
            if formula in formula_filter:
                selected.append(frame_idx)
        if selected:
            eligible_indices = np.asarray(selected, dtype=np.int64)
        else:
            eligible_indices = np.asarray([], dtype=np.int64)

    return {
        "kind": "seed",
        "root": str(seed_root),
        "R_path": str(r_path),
        "Z_path": str(z_path),
        "n_frames": int(R.shape[0]),
        "eligible_indices": eligible_indices,
        "prelabeled_paths": prelabeled_paths,
    }


def build_replay_manifest(
    *,
    h5_path: str,
    current_h5_id: int,
    builder_config: dict[str, Any],
) -> dict[str, Any]:
    formula_filter = _normalize_formula_filter(builder_config.get("empirical_formula_filter"))
    cached = _load_manifest_from_cache(
        h5_path=h5_path,
        current_h5_id=current_h5_id,
        builder_config=builder_config,
        formula_filter=formula_filter,
    )
    if cached is not None:
        return cached

    manifest = {
        "h5_groups": _build_h5_manifest(
            h5_path=h5_path,
            current_h5_id=current_h5_id,
            formula_filter=formula_filter,
        ),
        "seed": _build_seed_manifest(builder_config=builder_config, formula_filter=formula_filter),
    }
    _store_manifest_in_cache(
        manifest=manifest,
        h5_path=h5_path,
        current_h5_id=current_h5_id,
        builder_config=builder_config,
        formula_filter=formula_filter,
    )
    return manifest


def _choose_source(manifest: dict[str, Any], source_priority: str) -> str:
    source_priority = str(source_priority).strip().lower()
    has_h5 = bool(manifest.get("h5_groups"))
    seed_manifest = manifest.get("seed")
    has_seed = bool(seed_manifest) and (
        seed_manifest.get("eligible_indices") is None or len(seed_manifest["eligible_indices"]) > 0
    )

    if source_priority == "seed_all_once":
        if has_h5:
            return "h5"
        if has_seed:
            return "seed"
        raise RuntimeError("No replay source is available for seed_all_once. Check the seed dataset path.")

    if source_priority == "h5_only":
        if not has_h5:
            raise RuntimeError("No HDF5 replay frames are available for excited-state replay.")
        return "h5"
    if source_priority == "seed_only":
        if not has_seed:
            raise RuntimeError("No seed-dataset replay frames are available for excited-state replay.")
        return "seed"
    if source_priority == "seed_then_h5":
        if has_seed:
            return "seed"
        if has_h5:
            return "h5"
    else:
        if has_h5:
            return "h5"
        if has_seed:
            return "seed"

    raise RuntimeError("No replay source is available. Check the seed dataset path and existing HDF5 shards.")


def _seed_frame_to_atoms(seed_manifest: dict[str, Any], frame_idx: int) -> tuple[Atoms, dict[str, Any]]:
    R = np.load(seed_manifest["R_path"], mmap_mode="r")
    Z = np.load(seed_manifest["Z_path"], mmap_mode="r")
    numbers = np.asarray(Z[int(frame_idx)], dtype=np.int64).reshape(-1)
    coords = np.asarray(R[int(frame_idx)], dtype=np.float64).reshape(-1, 3)
    mask = numbers > 0
    numbers = numbers[mask]
    coords = coords[mask]
    atoms = Atoms(numbers=numbers.tolist(), positions=coords)
    return atoms, {
        "replay_source_kind": "seed",
        "replay_source_path": str(seed_manifest["R_path"]),
        "replay_frame_index": int(frame_idx),
        "replay_formula": empirical_formula_from_numbers(numbers),
    }


def _load_prelabeled_seed_results(
    seed_manifest: dict[str, Any],
    frame_idx: int,
    properties_list: dict[str, list[Any]],
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    prelabeled_paths = dict(seed_manifest.get("prelabeled_paths") or {})
    if not prelabeled_paths:
        return {}, False, {"prelabeled_seed_configured": False}

    Z = np.load(seed_manifest["Z_path"], mmap_mode="r")
    numbers = np.asarray(Z[int(frame_idx)], dtype=np.int64).reshape(-1)
    atom_mask = numbers > 0
    n_atoms = int(np.sum(atom_mask))
    results: dict[str, Any] = {}
    errors: list[str] = []

    for prop_key, schema in properties_list.items():
        if prop_key not in prelabeled_paths:
            errors.append(f"missing file for {prop_key}")
            continue
        prop_kind = str(schema[1]).lower() if len(schema) > 1 else ""
        try:
            label_array = np.load(prelabeled_paths[prop_key], mmap_mode="r")
            raw_value = np.asarray(label_array[int(frame_idx)], dtype=np.float64)
            if prop_kind == "system":
                flat_value = raw_value.reshape(-1)
                if flat_value.size != 1 or not np.all(np.isfinite(flat_value)):
                    errors.append(f"{prop_key} has invalid system value shape {raw_value.shape}")
                    continue
                results[prop_key] = float(flat_value[0])
            elif prop_kind == "atomic":
                if raw_value.ndim != 2 or raw_value.shape[-1] != 3:
                    errors.append(f"{prop_key} has invalid atomic value shape {raw_value.shape}")
                    continue
                if raw_value.shape[0] == atom_mask.shape[0]:
                    value = raw_value[atom_mask]
                elif raw_value.shape[0] == n_atoms:
                    value = raw_value
                else:
                    errors.append(
                        f"{prop_key} atom count {raw_value.shape[0]} does not match seed atom count {n_atoms}"
                    )
                    continue
                if value.shape != (n_atoms, 3) or not np.all(np.isfinite(value)):
                    errors.append(f"{prop_key} has non-finite or invalid force shape {value.shape}")
                    continue
                results[prop_key] = np.asarray(value, dtype=np.float64)
            else:
                errors.append(f"{prop_key} has unsupported property kind {prop_kind!r}")
        except Exception as exc:
            errors.append(f"{prop_key}: {type(exc).__name__}: {exc}")

    if not errors:
        errors.extend(validate_gap_results(results, properties_list))

    valid = not errors and set(results) == set(properties_list)
    metadata = {
        "prelabeled_seed_configured": True,
        "prelabeled_seed_valid": bool(valid),
        "prelabeled_seed_keys": sorted(results),
    }
    if errors:
        metadata["prelabeled_seed_errors"] = errors
    return (results if valid else {}), bool(valid), metadata


def _seed_all_once_frame_index(seed_manifest: dict[str, Any], moleculeid: str) -> int:
    eligible = seed_manifest.get("eligible_indices")
    if eligible is None:
        available = np.arange(int(seed_manifest["n_frames"]), dtype=np.int64)
    else:
        available = np.asarray(eligible, dtype=np.int64)

    if available.size == 0:
        raise RuntimeError("Seed replay manifest contains no eligible frames.")

    match = os.path.basename(str(moleculeid))
    trailing_digits = ""
    for char in reversed(match):
        if char.isdigit():
            trailing_digits = char + trailing_digits
        elif trailing_digits:
            break
    if trailing_digits:
        cycle_index = int(trailing_digits)
    else:
        cycle_index = stable_uint32_seed(str(moleculeid))

    return int(available[cycle_index % int(available.size)])


def _sample_seed_frame(seed_manifest: dict[str, Any], rng: np.random.Generator) -> tuple[Atoms, dict[str, Any]]:
    eligible = seed_manifest.get("eligible_indices")
    if eligible is None:
        frame_idx = int(rng.integers(int(seed_manifest["n_frames"])))
    else:
        if len(eligible) == 0:
            raise RuntimeError("Seed replay manifest contains no eligible frames.")
        frame_idx = int(rng.choice(eligible))
    return _seed_frame_to_atoms(seed_manifest, frame_idx)


def _sample_h5_frame(
    h5_groups: list[dict[str, Any]],
    rng: np.random.Generator,
    *,
    sampler_config: dict[str, Any] | None = None,
) -> tuple[Atoms, dict[str, Any]]:
    if not h5_groups:
        raise RuntimeError("HDF5 replay manifest contains no eligible groups.")
    total_frames = int(h5_groups[-1]["stop"])
    selected = int(rng.integers(total_frames))
    group_entry = h5_groups[-1]
    for entry in h5_groups:
        if int(entry["start"]) <= selected < int(entry["stop"]):
            group_entry = entry
            break
    local_index = int(selected - int(group_entry["start"]))
    with h5py.File(group_entry["path"], "r") as handle:
        group = handle[group_entry["group_path"]]
        coords = np.asarray(group["coordinates"][local_index], dtype=np.float64)
        species = np.asarray(group["species"][()])
        symbols = [chemical_symbols[int(z)] for z in []]
        if species.dtype.kind in {"S", "O", "U"}:
            symbols = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in species.tolist()]
            numbers = np.asarray([chemical_symbols.index(symbol) for symbol in symbols], dtype=np.int64)
        else:
            numbers = np.asarray(species, dtype=np.int64)
            symbols = [chemical_symbols[int(z)] for z in numbers.tolist()]
        if "cell" in group:
            cell = np.asarray(group["cell"][local_index], dtype=np.float64)
            atoms = Atoms(symbols=symbols, positions=coords, cell=cell, pbc=True)
        else:
            atoms = Atoms(symbols=symbols, positions=coords)
        if H5_TOPOLOGY_ATOM_IDS_KEY in group:
            stored_ids = np.asarray(group[H5_TOPOLOGY_ATOM_IDS_KEY])
            if stored_ids.ndim == 2:
                stored_ids = stored_ids[local_index]
            atoms.set_array(
                topology_atom_id_array(sampler_config),
                np.asarray(stored_ids, dtype=np.int64).reshape(-1),
            )
    return atoms, {
        "replay_source_kind": "h5",
        "replay_source_path": str(group_entry["path"]),
        "replay_group_path": str(group_entry["group_path"]),
        "replay_frame_index": int(local_index),
        "replay_formula": str(group_entry["formula"]),
    }


def build_excited_state_replay_structures(
    *,
    moleculeids: list[str],
    builder_config: dict[str, Any],
    properties_list: dict[str, list[Any]],
    h5_path: str,
    current_h5_id: int,
    sampler_config: dict[str, Any] | None = None,
) -> list[MoleculesObject]:
    manifest = build_replay_manifest(
        h5_path=h5_path,
        current_h5_id=current_h5_id,
        builder_config=builder_config,
    )
    state_ids = derive_state_ids(properties_list)
    selection_config = dict(builder_config.get("state_selection") or {})
    source_priority = str(builder_config.get("source_priority", "h5_then_seed"))
    outputs: list[MoleculesObject] = []

    for moleculeid in moleculeids:
        seed_value = stable_uint32_seed(str(moleculeid), int(current_h5_id))
        rng = np.random.default_rng(seed_value)
        source_kind = _choose_source(manifest, source_priority)
        if source_kind == "h5":
            atoms, source_meta = _sample_h5_frame(
                manifest["h5_groups"],
                rng,
                sampler_config=sampler_config,
            )
        else:
            if source_priority.strip().lower() == "seed_all_once":
                frame_idx = _seed_all_once_frame_index(manifest["seed"], str(moleculeid))
                atoms, source_meta = _seed_frame_to_atoms(manifest["seed"], frame_idx)
            else:
                atoms, source_meta = _sample_seed_frame(manifest["seed"], rng)

        if topology_enabled(sampler_config):
            ensure_topology_atom_ids(
                atoms,
                sampler_config,
                load_fixed_topology(sampler_config),
                allow_inference=True,
            )
        molecule = MoleculesObject(atoms, str(moleculeid))
        excited_state = select_excited_state(
            moleculeid=str(moleculeid),
            available_states=state_ids,
            metadata=source_meta,
            selection_config=selection_config,
            rng=rng,
        )
        source_meta.update(
            {
                "excited_state": int(excited_state),
                "excited_state_selection_mode": str(selection_config.get("mode", "cyclic")).lower(),
            }
        )
        molecule.update_metadata(source_meta)
        if source_kind == "seed":
            results, converged, label_meta = _load_prelabeled_seed_results(
                manifest["seed"],
                int(source_meta["replay_frame_index"]),
                properties_list,
            )
            molecule.update_metadata(label_meta)
            if label_meta.get("prelabeled_seed_configured"):
                if results:
                    molecule.store_results(results)
                molecule.set_converged_flag(bool(converged))
        outputs.append(molecule)

    return outputs


@python_app(executors=["alf_sampler_executor"])
def excited_state_replay_builder_task(
    moleculeid=None,
    moleculeids=None,
    builder_config=None,
    sampler_config=None,
    properties_list=None,
    h5_path=None,
    current_h5_id=0,
):
    ids = [str(moleculeid)] if moleculeids is None else [str(value) for value in moleculeids]
    outputs = build_excited_state_replay_structures(
        moleculeids=ids,
        builder_config=dict(builder_config or {}),
        properties_list=dict(properties_list or {}),
        h5_path=str(h5_path),
        current_h5_id=int(current_h5_id),
        sampler_config=dict(sampler_config or {}),
    )
    if moleculeids is None:
        return outputs[0]
    return outputs


@python_app(executors=["alf_gpu_executor"])
def excited_state_replay_builder_gpu_task(
    moleculeid=None,
    moleculeids=None,
    builder_config=None,
    sampler_config=None,
    properties_list=None,
    h5_path=None,
    current_h5_id=0,
):
    ids = [str(moleculeid)] if moleculeids is None else [str(value) for value in moleculeids]
    outputs = build_excited_state_replay_structures(
        moleculeids=ids,
        builder_config=dict(builder_config or {}),
        properties_list=dict(properties_list or {}),
        h5_path=str(h5_path),
        current_h5_id=int(current_h5_id),
        sampler_config=dict(sampler_config or {}),
    )
    if moleculeids is None:
        return outputs[0]
    return outputs
