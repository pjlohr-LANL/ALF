"""Isolated multi-state HIPPYNN training for excited-state ALF workflows.

This module deliberately does not modify the ground-state HIPPYNN interface.
Each ensemble member has one shared HipNN or HipHopNN trunk and one energy
head (plus its force gradient) for every contiguous ``sE#``/``F#`` property
configured in ALF.
"""

from __future__ import annotations

import glob
import json
import math
import multiprocessing
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from parsl import python_app

from alframework.tools.excited_state_tools import (
    derive_gap_property_table,
    derive_state_property_table,
)
from alframework.tools.pyanitools import anidataloader


def validate_excited_state_training_properties(
    properties_list: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Validate flattened state targets and their HDF5 storage schemas."""

    state_table = derive_state_property_table(
        dict(properties_list or {}),
        require_forces=True,
    )
    database_names: list[str] = []
    for row in state_table:
        energy_schema = properties_list[str(row["energy_key"])]
        force_schema = properties_list[str(row["force_key"])]
        if len(energy_schema) < 2 or str(energy_schema[1]).lower() != "system":
            raise ValueError(
                f"{row['energy_key']} must be a system property in "
                "properties_list."
            )
        if len(force_schema) < 2 or str(force_schema[1]).lower() != "atomic":
            raise ValueError(
                f"{row['force_key']} must be an atomic property in "
                "properties_list."
            )
        database_names.extend(
            [str(row["energy_db_name"]), str(row["force_db_name"])]
        )
    gap_table = derive_gap_property_table(properties_list)
    database_names.extend(str(row["gap_db_name"]) for row in gap_table)
    if any(not name for name in database_names):
        raise ValueError("Excited-state HDF5 database names cannot be empty.")
    if len(database_names) != len(set(database_names)):
        raise ValueError(
            "Every excited-state energy, force, and gap requires a unique "
            "HDF5 database name."
        )
    return state_table


def validate_excited_state_training_config(
    ML_config: dict[str, Any],
    properties_list: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Validate the supported multi-state training contract."""

    config = dict(ML_config or {})
    state_table = validate_excited_state_training_properties(properties_list)
    n_models = int(config.get("n_models", 0))
    if n_models < 1:
        raise ValueError("n_models must be a positive integer.")

    cell_key = config.get("cell_key")
    if cell_key not in {None, "None"}:
        raise ValueError(
            "Excited-state HIPPYNN training currently supports only "
            "nonperiodic data; cell_key must be null."
        )
    if not bool(config.get("train_forces", True)):
        raise ValueError(
            "Excited-state HIPPYNN training requires train_forces=true."
        )
    if not bool(config.get("export_force_gradients", True)):
        raise ValueError(
            "Excited-state checkpoints require export_force_gradients=true "
            "so every F# node is available to sampling."
        )

    gap_config = config.get("gap_targets")
    if gap_config is not None and not isinstance(gap_config, dict):
        raise TypeError("gap_targets must be a dictionary when provided.")
    gap_options = dict(gap_config or {})
    gap_enabled = bool(gap_options.get("enabled", False))
    gap_mode = str(gap_options.get("mode", "derived")).strip().lower()
    if gap_mode != "derived":
        raise ValueError(
            "gap_targets.mode must be 'derived'; independent gap heads are "
            "not supported."
        )
    gap_weight = float(gap_options.get("weight", 1.0))
    if not np.isfinite(gap_weight) or gap_weight < 0:
        raise ValueError("gap_targets.weight must be finite and nonnegative.")
    configured_gaps = derive_gap_property_table(
        properties_list,
        gap_config=gap_options,
        require_properties=gap_enabled,
    )
    if gap_enabled and not configured_gaps:
        raise ValueError(
            "Enabled gap targets require at least one dE# property in "
            "properties_list."
        )

    exports = config.get("exports")
    if exports is not None:
        if not isinstance(exports, dict):
            raise TypeError("exports must be a dictionary when provided.")
        if bool(exports.get("csv", False)):
            raise ValueError(
                "CSV prediction export is not supported by the isolated "
                "excited-state trainer; PDF and PNG plots are supported."
            )

    network_choice = int(config.get("network_choice", 1))
    if network_choice not in {0, 1}:
        raise ValueError(
            "network_choice must be 0 (HipNN) or 1 (HipHopNN)."
        )
    if str(config.get("optimizer", "AdamW")).lower() != "adamw":
        raise ValueError(
            "The isolated excited-state trainer currently supports "
            "optimizer='AdamW' only."
        )
    if config.get("n_atoms") is not None and int(config["n_atoms"]) < 1:
        raise ValueError("n_atoms must be positive when configured.")

    for key in (
        "batch_size",
        "eval_batch_size",
        "max_epochs",
        "termination_patience",
        "plot_frequency",
    ):
        if int(config.get(key, 1)) < 1:
            raise ValueError(f"{key} must be a positive integer.")

    valid_size = float(config.get("valid_size", 0.1))
    test_size = float(config.get("test_size", 0.1))
    if (
        not np.isfinite(valid_size)
        or not np.isfinite(test_size)
        or valid_size <= 0
        or test_size <= 0
        or valid_size + test_size >= 1
    ):
        raise ValueError(
            "valid_size and test_size must be finite, positive, and sum to "
            "less than one."
        )

    for key in (
        "energy_weight",
        "force_weight",
        "l2_weight",
        "learning_rate",
    ):
        value = float(config.get(key, 1.0 if key != "l2_weight" else 2.0e-5))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    if float(config.get("learning_rate", 5.0e-4)) <= 0:
        raise ValueError("learning_rate must be positive.")
    for key in (
        "remove_high_energy_cut",
        "remove_high_energy_std",
        "remove_high_forces_cut",
        "remove_high_forces_std",
    ):
        if config.get(key) is None:
            continue
        value = float(config[key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative or null.")

    return state_table


def _atomic_number_sequence(species: Any, *, context: str) -> np.ndarray:
    """Convert one ALF species row to atomic numbers."""

    from ase.data import atomic_numbers

    values = np.asarray(species).reshape(-1).tolist()
    numbers: list[int] = []
    for value in values:
        if isinstance(value, (int, np.integer)):
            number = int(value)
        else:
            if isinstance(value, (bytes, np.bytes_)):
                value = value.decode("utf-8")
            text = str(value)
            if text not in atomic_numbers:
                raise ValueError(
                    f"{context} contains an unknown chemical symbol {text!r}."
                )
            number = int(atomic_numbers[text])
        if number <= 0:
            raise ValueError(
                f"{context} contains a nonpositive atomic number {number}."
            )
        numbers.append(number)
    if not numbers:
        raise ValueError(f"{context} contains no atoms.")
    return np.asarray(numbers, dtype=np.int64)


def _group_species_rows(
    species: Any,
    *,
    n_systems: int,
    n_atoms: int,
    context: str,
) -> np.ndarray:
    values = np.asarray(species)
    if values.ndim == 1:
        sequence = _atomic_number_sequence(values, context=context)
        if sequence.shape != (n_atoms,):
            raise ValueError(
                f"{context} has {sequence.size} atoms but coordinates have "
                f"{n_atoms}."
            )
        return np.broadcast_to(sequence, (n_systems, n_atoms)).copy()
    if values.ndim == 2 and values.shape == (n_systems, n_atoms):
        return np.stack(
            [
                _atomic_number_sequence(row, context=f"{context} row {index}")
                for index, row in enumerate(values)
            ],
            axis=0,
        )
    raise ValueError(
        f"{context} must have shape ({n_atoms},) or "
        f"({n_systems}, {n_atoms}); received {values.shape}."
    )


def _normalize_energy(
    values: Any,
    *,
    n_systems: int,
    context: str,
    output_dtype: Any = np.float32,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 and n_systems == 1:
        array = array.reshape(1)
    elif array.shape == (n_systems, 1):
        array = array[:, 0]
    if array.shape != (n_systems,):
        raise ValueError(
            f"{context} must contain one energy per structure with shape "
            f"({n_systems},) or ({n_systems}, 1); received {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{context} contains nonfinite values.")
    return array.astype(output_dtype, copy=False)


def _normalize_forces(
    values: Any,
    *,
    n_systems: int,
    n_atoms: int,
    context: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    expected = (n_systems, n_atoms, 3)
    if array.shape != expected:
        raise ValueError(
            f"{context} must have shape {expected}; received {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{context} contains nonfinite values.")
    return array.astype(np.float32, copy=False)


def _h5_shards(h5_train_dir: str) -> list[str]:
    root = Path(h5_train_dir).expanduser()
    if root.is_file():
        paths = [str(root)]
    else:
        paths = sorted(
            set(
                glob.glob(str(root / "*.h5"))
                + glob.glob(str(root / "*.hdf5"))
            )
        )
    if not paths:
        raise FileNotFoundError(
            f"No ALF HDF5 shards were found in {str(root)!r}."
        )
    return paths


def load_excited_state_h5_arrays(
    h5_train_dir: str,
    properties_list: dict[str, list[Any]],
    *,
    coordinates_key: str = "coordinates",
    species_key: str = "species",
    configured_n_atoms: int | None = None,
    configured_possible_species: list[int] | tuple[int, ...] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read and validate fixed-composition ALF HDF5 shards explicitly.

    Only coordinates, species, and the configured state energy/force database
    names are loaded. This avoids HIPPYNN's PyANI field auto-detection path,
    which is unreliable for some small molecules.
    """

    state_table = validate_excited_state_training_properties(properties_list)
    gap_table = derive_gap_property_table(properties_list)
    chunks: dict[str, list[np.ndarray]] = {
        str(coordinates_key): [],
        str(species_key): [],
    }
    for row in state_table:
        chunks[str(row["energy_db_name"])] = []
        chunks[str(row["force_db_name"])] = []
    for row in gap_table:
        chunks[str(row["gap_db_name"])] = []

    reference_species: np.ndarray | None = None
    group_count = 0
    for shard_path in _h5_shards(h5_train_dir):
        loader = anidataloader(shard_path)
        try:
            for group in loader:
                group_count += 1
                context = f"{shard_path}:{group.get('path', '<unknown>')}"
                missing = [
                    key
                    for key in (str(coordinates_key), str(species_key))
                    if key not in group
                ]
                for row in state_table:
                    for key in (
                        str(row["energy_db_name"]),
                        str(row["force_db_name"]),
                    ):
                        if key not in group:
                            missing.append(key)
                for row in gap_table:
                    gap_name = str(row["gap_db_name"])
                    if gap_name not in group:
                        missing.append(gap_name)
                if missing:
                    raise KeyError(
                        f"{context} is missing required datasets: "
                        + ", ".join(sorted(set(missing)))
                    )

                coordinates = np.asarray(
                    group[str(coordinates_key)],
                    dtype=np.float64,
                )
                if coordinates.ndim == 2 and coordinates.shape[-1] == 3:
                    coordinates = coordinates[None, :, :]
                if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
                    raise ValueError(
                        f"{context}:{coordinates_key} must have shape "
                        "[structures, atoms, 3]; received "
                        f"{coordinates.shape}."
                    )
                if not np.all(np.isfinite(coordinates)):
                    raise ValueError(
                        f"{context}:{coordinates_key} contains nonfinite values."
                    )
                n_systems, n_atoms, _ = coordinates.shape
                if n_systems < 1 or n_atoms < 1:
                    raise ValueError(f"{context} contains an empty structure set.")

                species_rows = _group_species_rows(
                    group[str(species_key)],
                    n_systems=n_systems,
                    n_atoms=n_atoms,
                    context=f"{context}:{species_key}",
                )
                for row_index, sequence in enumerate(species_rows):
                    if reference_species is None:
                        reference_species = sequence.copy()
                    elif not np.array_equal(sequence, reference_species):
                        raise ValueError(
                            "Excited-state HIPPYNN training requires one exact "
                            "atomic-number sequence; "
                            f"{context} row {row_index} has "
                            f"{sequence.tolist()}, expected "
                            f"{reference_species.tolist()}."
                        )

                chunks[str(coordinates_key)].append(
                    coordinates.astype(np.float32, copy=False)
                )
                chunks[str(species_key)].append(species_rows)
                group_state_energies: dict[int, np.ndarray] = {}
                for row in state_table:
                    energy_name = str(row["energy_db_name"])
                    force_name = str(row["force_db_name"])
                    energy_values = _normalize_energy(
                        group[energy_name],
                        n_systems=n_systems,
                        context=f"{context}:{energy_name}",
                        output_dtype=np.float64,
                    )
                    group_state_energies[int(row["state"])] = energy_values
                    chunks[energy_name].append(
                        energy_values.astype(np.float32, copy=False)
                    )
                    chunks[force_name].append(
                        _normalize_forces(
                            group[force_name],
                            n_systems=n_systems,
                            n_atoms=n_atoms,
                            context=f"{context}:{force_name}",
                        )
                    )
                for row in gap_table:
                    gap_name = str(row["gap_db_name"])
                    gap_values = _normalize_energy(
                        group[gap_name],
                        n_systems=n_systems,
                        context=f"{context}:{gap_name}",
                        output_dtype=np.float64,
                    )
                    expected_gap = (
                        group_state_energies[int(row["upper_state"])]
                        - group_state_energies[int(row["lower_state"])]
                    )
                    if not np.allclose(
                        gap_values,
                        expected_gap,
                        rtol=1.0e-7,
                        atol=1.0e-8,
                    ):
                        max_error = float(
                            np.max(np.abs(gap_values - expected_gap))
                        )
                        raise ValueError(
                            f"HDF5 gap target {gap_name!r} is inconsistent "
                            f"with {row['upper_energy_key']} - "
                            f"{row['lower_energy_key']}; maximum absolute "
                            f"error is {max_error:.6g}."
                        )
                    chunks[gap_name].append(
                        gap_values.astype(np.float32, copy=False)
                    )
        finally:
            loader.cleanup()

    if group_count == 0 or reference_species is None:
        raise ValueError("ALF HDF5 shards contain no molecular data groups.")

    inferred_n_atoms = int(reference_species.size)
    if configured_n_atoms is not None:
        configured_atoms = int(configured_n_atoms)
        if configured_atoms != inferred_n_atoms:
            raise ValueError(
                f"Configured n_atoms={configured_atoms} does not match the "
                f"HDF5 atom count {inferred_n_atoms}."
            )

    inferred_species = sorted(set(int(value) for value in reference_species))
    if configured_possible_species is not None:
        configured_species = [int(value) for value in configured_possible_species]
        if not configured_species or configured_species[0] != 0:
            raise ValueError(
                "network_params.possible_species must begin with padding "
                "species 0."
            )
        if len(configured_species) != len(set(configured_species)):
            raise ValueError(
                "network_params.possible_species must not contain duplicates."
            )
        if any(value < 0 for value in configured_species):
            raise ValueError(
                "network_params.possible_species cannot contain negative "
                "atomic numbers."
            )
        missing_species = sorted(
            set(inferred_species).difference(configured_species)
        )
        if missing_species:
            raise ValueError(
                "network_params.possible_species is missing atomic numbers "
                f"present in HDF5: {missing_species}."
            )
        possible_species = configured_species
    else:
        possible_species = [0, *inferred_species]

    arrays = {
        key: np.concatenate(values, axis=0)
        for key, values in chunks.items()
    }
    summary = {
        "n_structures": int(arrays[str(coordinates_key)].shape[0]),
        "n_atoms": inferred_n_atoms,
        "atomic_numbers": reference_species.tolist(),
        "possible_species": possible_species,
        "state_table": state_table,
        "gap_table": gap_table,
    }
    return arrays, summary


def _minimum_structure_distances(coordinates: np.ndarray) -> np.ndarray:
    """Return the shortest nonperiodic pair distance in every structure."""

    count, atom_count, _ = coordinates.shape
    if atom_count < 2:
        return np.full(count, np.inf, dtype=np.float64)
    minima = np.full(count, np.inf, dtype=np.float64)
    for first in range(atom_count - 1):
        distances = np.linalg.norm(
            coordinates[:, first + 1 :, :]
            - coordinates[:, first : first + 1, :],
            axis=-1,
        )
        minima = np.minimum(minima, np.min(distances, axis=1))
    return minima


def _property_outlier_mask(
    values: np.ndarray,
    active: np.ndarray,
    threshold: float | None,
    *,
    standardized: bool,
) -> np.ndarray:
    """Apply production HIPPYNN's centered componentwise outlier formula."""

    failures = np.zeros(active.shape, dtype=bool)
    if threshold is None or not np.any(active):
        return failures
    selected = np.asarray(values, dtype=np.float64)[active]
    mean = float(np.mean(selected))
    if standardized:
        flattened = selected.reshape(-1)
        if flattened.size < 2:
            return failures
        deviation_scale = float(np.std(flattened, ddof=1))
        if not np.isfinite(deviation_scale) or deviation_scale == 0.0:
            return failures
        element_failures = (
            np.abs(selected - mean) / deviation_scale > float(threshold)
        )
    else:
        element_failures = np.abs(selected - mean) > float(threshold)
    structure_failures = np.any(
        element_failures.reshape(selected.shape[0], -1),
        axis=1,
    )
    failures[np.flatnonzero(active)] = structure_failures
    return failures


def _mask_training_arrays(
    arrays: dict[str, np.ndarray],
    keep: np.ndarray,
) -> dict[str, np.ndarray]:
    count = int(keep.shape[0])
    filtered: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.shape[0] != count:
            raise ValueError(
                f"Training array {key!r} has {array.shape[0]} structures; "
                f"expected {count}."
            )
        filtered[key] = array[keep]
    return filtered


def filter_excited_state_training_arrays(
    arrays: dict[str, np.ndarray],
    state_table: list[dict[str, Any]],
    *,
    coordinates_key: str,
    dist_soft_min: float,
    remove_high_energy_cut: float | None = None,
    remove_high_energy_std: float | None = None,
    remove_high_forces_cut: float | None = None,
    remove_high_forces_std: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Filter fixed-composition multi-state arrays before data splitting.

    Multi-state masks are computed from a common survivor pool and unioned,
    making the result independent of state order. The one-state path preserves
    production's force-then-energy cut/std sequence exactly.
    """

    coordinates = np.asarray(arrays[coordinates_key], dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"{coordinates_key} must have shape [structures, atoms, 3]."
        )
    count = int(coordinates.shape[0])
    minimum_distance = float(dist_soft_min)
    if not np.isfinite(minimum_distance) or minimum_distance < 0:
        raise ValueError("network_params.dist_soft_min must be nonnegative.")
    options = {
        "remove_high_energy_cut": remove_high_energy_cut,
        "remove_high_energy_std": remove_high_energy_std,
        "remove_high_forces_cut": remove_high_forces_cut,
        "remove_high_forces_std": remove_high_forces_std,
    }
    for key, value in options.items():
        if value is not None and (
            not np.isfinite(float(value)) or float(value) < 0
        ):
            raise ValueError(f"{key} must be finite and nonnegative or null.")

    active = _minimum_structure_distances(coordinates) > minimum_distance
    summary: dict[str, Any] = {
        "initial_structures": count,
        "dist_soft_min": minimum_distance,
        "removed_min_distance": int(count - np.count_nonzero(active)),
        "static_cut_failures": {},
        "standard_deviation_failures": {},
    }
    energy_names = [
        str(row["energy_db_name"]) for row in state_table
    ]
    force_names = [
        str(row["force_db_name"]) for row in state_table
    ]
    removed_static = 0
    removed_standardized = 0

    def apply_one(
        property_name: str,
        threshold: float | None,
        *,
        standardized: bool,
        summary_key: str,
    ) -> None:
        nonlocal active, removed_static, removed_standardized
        failures = _property_outlier_mask(
            arrays[property_name],
            active,
            threshold,
            standardized=standardized,
        )
        failure_count = int(np.count_nonzero(failures))
        summary[summary_key][property_name] = failure_count
        if standardized:
            removed_standardized += failure_count
        else:
            removed_static += failure_count
        active &= ~failures

    if len(state_table) == 1:
        # Production ground-state HIPPYNN filters force cut/std first, then
        # energy cut/std, recomputing statistics after every removal.
        apply_one(
            force_names[0],
            remove_high_forces_cut,
            standardized=False,
            summary_key="static_cut_failures",
        )
        apply_one(
            force_names[0],
            remove_high_forces_std,
            standardized=True,
            summary_key="standard_deviation_failures",
        )
        apply_one(
            energy_names[0],
            remove_high_energy_cut,
            standardized=False,
            summary_key="static_cut_failures",
        )
        apply_one(
            energy_names[0],
            remove_high_energy_std,
            standardized=True,
            summary_key="standard_deviation_failures",
        )
    else:
        static_failures = np.zeros(count, dtype=bool)
        for property_name, threshold in (
            *(
                (name, remove_high_forces_cut)
                for name in force_names
            ),
            *(
                (name, remove_high_energy_cut)
                for name in energy_names
            ),
        ):
            failures = _property_outlier_mask(
                arrays[property_name],
                active,
                threshold,
                standardized=False,
            )
            summary["static_cut_failures"][property_name] = int(
                np.count_nonzero(failures)
            )
            static_failures |= failures
        removed_static = int(
            np.count_nonzero(static_failures & active)
        )
        active &= ~static_failures

        standard_failures = np.zeros(count, dtype=bool)
        for property_name, threshold in (
            *(
                (name, remove_high_forces_std)
                for name in force_names
            ),
            *(
                (name, remove_high_energy_std)
                for name in energy_names
            ),
        ):
            failures = _property_outlier_mask(
                arrays[property_name],
                active,
                threshold,
                standardized=True,
            )
            summary["standard_deviation_failures"][property_name] = int(
                np.count_nonzero(failures)
            )
            standard_failures |= failures
        removed_standardized = int(
            np.count_nonzero(standard_failures & active)
        )
        active &= ~standard_failures

    summary["removed_static_cut_union"] = int(removed_static)
    summary["removed_standard_deviation_union"] = int(
        removed_standardized
    )
    summary["final_structures"] = int(np.count_nonzero(active))
    summary["per_state_failures"] = {
        str(int(row["state"])): {
            "energy": {
                "static_cut": int(
                    summary["static_cut_failures"].get(
                        str(row["energy_db_name"]),
                        0,
                    )
                ),
                "standard_deviation": int(
                    summary["standard_deviation_failures"].get(
                        str(row["energy_db_name"]),
                        0,
                    )
                ),
            },
            "forces": {
                "static_cut": int(
                    summary["static_cut_failures"].get(
                        str(row["force_db_name"]),
                        0,
                    )
                ),
                "standard_deviation": int(
                    summary["standard_deviation_failures"].get(
                        str(row["force_db_name"]),
                        0,
                    )
                ),
            },
        }
        for row in state_table
    }
    filtered = _mask_training_arrays(arrays, active)
    return filtered, summary


def validate_filtered_split_capacity(
    structure_count: int,
    *,
    test_size: float,
    valid_size: float,
) -> None:
    """Fail before HIPPYNN attempts to create an empty data split."""

    count = int(structure_count)
    test_count = int(float(test_size) * count)
    remaining = count - test_count
    adjusted_valid = float(valid_size) / (1.0 - float(test_size))
    valid_count = int(adjusted_valid * remaining)
    train_count = remaining - valid_count
    if min(test_count, valid_count, train_count) < 1:
        raise ValueError(
            "Excited-state filtering left too few structures for nonempty "
            f"train/valid/test splits: {count} survivors produce "
            f"train={train_count}, valid={valid_count}, test={test_count}."
        )


def compose_excited_state_loss(
    energy_error_terms: list[tuple[Any, Any]],
    force_error_terms: list[tuple[Any, Any]],
    l2_term: Any,
    *,
    gap_error_terms: list[tuple[Any, Any]] | None = None,
    n_atoms: int,
    energy_weight: float,
    force_weight: float,
    gap_weight: float = 1.0,
    l2_weight: float,
) -> Any:
    """Compose the fork-compatible multi-state E/F/gap/L2 objective."""

    if not energy_error_terms:
        raise ValueError("At least one state-energy loss term is required.")
    atom_count = int(n_atoms)
    if atom_count < 1:
        raise ValueError("n_atoms must be positive.")
    total = sum(
        float(energy_weight) * (rmse + mae)
        for rmse, mae in energy_error_terms
    )
    force_normalizer = math.sqrt(3.0 * float(atom_count))
    total = total + sum(
        float(force_weight) * (rmse + mae) / force_normalizer
        for rmse, mae in force_error_terms
    )
    total = total + sum(
        float(gap_weight) * (rmse + mae)
        for rmse, mae in (gap_error_terms or [])
    )
    return total + float(l2_weight) * l2_term


def _configure_cuda_visible_devices(
    device_string: str,
    from_multiprocessing_nGPU: int | None,
) -> tuple[str | None, str]:
    requested = str(device_string).strip().lower()
    if requested == "cpu":
        return None, "<cpu>"
    if requested == "from_multiprocessing":
        gpu_count = int(from_multiprocessing_nGPU or 0)
        if gpu_count < 1:
            return None, "<cpu>"
        identity = getattr(multiprocessing.current_process(), "_identity", ())
        worker_number = int(identity[-1]) if identity else 1
        visible = str((worker_number - 1) % gpu_count)
    else:
        visible = str(device_string)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return visible, visible


def _resolve_torch_device(requested_device: str, visible_device: str | None):
    import torch

    if (
        str(requested_device).strip().lower() == "cpu"
        or visible_device is None
        or not torch.cuda.is_available()
    ):
        return torch.device("cpu"), "<cpu>"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return device, str(visible_device)


def _build_shared_network(
    species_node: Any,
    positions_node: Any,
    *,
    network_choice: int,
    network_params: dict[str, Any],
) -> Any:
    from hippynn.graphs import networks

    parents = (species_node, positions_node)
    if int(network_choice) == 0:
        return networks.Hipnn(
            "excited_state_HipNN",
            parents,
            module_kwargs=dict(network_params),
            periodic=False,
        )
    if int(network_choice) == 1:
        return networks.HipHopnn(
            "excited_state_HipHopNN",
            parents,
            module_kwargs=dict(network_params),
            periodic=False,
        )
    raise ValueError("network_choice must be 0 (HipNN) or 1 (HipHopNN).")


def build_excited_state_training_graph(
    network: Any,
    positions_node: Any,
    state_table: list[dict[str, Any]],
    *,
    gap_table: list[dict[str, Any]] | None = None,
    n_atoms: int,
    energy_weight: float,
    force_weight: float,
    gap_weight: float = 1.0,
    l2_weight: float,
    first_is_interacting: bool = False,
) -> tuple[
    list[tuple[dict[str, Any], Any]],
    list[tuple[dict[str, Any], Any]],
    Any,
    dict[str, Any],
]:
    """Attach state heads/gradients and construct the shared training loss."""

    from hippynn.graphs import loss, physics, targets

    energy_outputs: list[tuple[dict[str, Any], Any]] = []
    force_outputs: list[tuple[dict[str, Any], Any]] = []
    gap_outputs: list[tuple[dict[str, Any], Any]] = []
    energy_error_terms: list[tuple[Any, Any]] = []
    force_error_terms: list[tuple[Any, Any]] = []
    gap_error_terms: list[tuple[Any, Any]] = []
    validation_losses: dict[str, Any] = {}
    for row in state_table:
        state = int(row["state"])
        energy_head = targets.HEnergyNode(
            f"state_{state}_energy_head",
            network,
            first_is_interacting=bool(first_is_interacting),
        )
        energy_output = energy_head.mol_energy
        energy_output.db_name = str(row["energy_db_name"])
        force_output = physics.GradientNode(
            f"state_{state}_force_gradient",
            (energy_output, positions_node),
            sign=-1,
            db_name=str(row["force_db_name"]),
        )
        energy_outputs.append((row, energy_output))
        force_outputs.append((row, force_output))

        energy_rmse = loss.MSELoss.of_node(energy_output) ** 0.5
        energy_mae = loss.MAELoss.of_node(energy_output)
        force_rmse = loss.MSELoss.of_node(force_output) ** 0.5
        force_mae = loss.MAELoss.of_node(force_output)
        energy_error_terms.append((energy_rmse, energy_mae))
        force_error_terms.append((force_rmse, force_mae))
        validation_losses[f"sE{state}_RMSE"] = energy_rmse
        validation_losses[f"sE{state}_MAE"] = energy_mae
        validation_losses[f"F{state}_RMSE"] = force_rmse
        validation_losses[f"F{state}_MAE"] = force_mae

    energy_by_state = {
        int(row["state"]): output for row, output in energy_outputs
    }
    for row in gap_table or []:
        gap_key = str(row["gap_key"])
        gap_output = (
            energy_by_state[int(row["upper_state"])]
            - energy_by_state[int(row["lower_state"])]
        )
        gap_output.name = f"derived_{gap_key}"
        gap_output.db_name = str(row["gap_db_name"])
        gap_outputs.append((row, gap_output))
        gap_rmse = loss.MSELoss.of_node(gap_output) ** 0.5
        gap_mae = loss.MAELoss.of_node(gap_output)
        gap_error_terms.append((gap_rmse, gap_mae))
        validation_losses[f"{gap_key}_RMSE"] = gap_rmse
        validation_losses[f"{gap_key}_MAE"] = gap_mae

    l2_term = loss.l2reg(network)
    total_loss = compose_excited_state_loss(
        energy_error_terms,
        force_error_terms,
        l2_term,
        gap_error_terms=gap_error_terms,
        n_atoms=n_atoms,
        energy_weight=energy_weight,
        force_weight=force_weight,
        gap_weight=gap_weight,
        l2_weight=l2_weight,
    )
    validation_losses["L2"] = l2_term
    validation_losses["Loss"] = total_loss
    return (
        energy_outputs,
        force_outputs,
        total_loss,
        validation_losses,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def model_member_seed(ML_config: dict[str, Any], model_id: int) -> int:
    """Return the deterministic seed assigned to an ensemble member."""

    return int(ML_config.get("random_seed", 114514)) + int(model_id)


def train_single_excited_state_model(
    *,
    model_id: int,
    model_dir: str,
    h5_train_dir: str,
    properties_list: dict[str, list[Any]],
    ML_config: dict[str, Any],
    from_multiprocessing_nGPU: int | None = None,
) -> dict[str, Any]:
    """Train one shared-trunk multi-state HIPPYNN model member."""

    config = dict(ML_config or {})
    state_table = validate_excited_state_training_config(
        config,
        properties_list,
    )
    gap_options = dict(config.get("gap_targets") or {})
    gap_table = (
        derive_gap_property_table(
            properties_list,
            gap_config=gap_options,
            require_properties=True,
        )
        if bool(gap_options.get("enabled", False))
        else []
    )
    visible_device, recorded_device = _configure_cuda_visible_devices(
        str(config.get("device_string", "from_multiprocessing")),
        from_multiprocessing_nGPU,
    )

    # Torch and HIPPYNN must be imported only after the child selects its GPU.
    import torch
    import hippynn
    from hippynn import plotting
    from hippynn.databases.database import Database
    from hippynn.experiment import SetupParams, setup_training, train_model
    from hippynn.experiment.controllers import (
        PatienceController,
        RaiseBatchSizeOnPlateau,
    )
    from hippynn.graphs import inputs

    device, recorded_device = _resolve_torch_device(
        str(config.get("device_string", "from_multiprocessing")),
        visible_device,
    )
    seed = model_member_seed(config, model_id)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_default_dtype(torch.float32)
    hippynn.settings.WARN_LOW_DISTANCES = False
    hippynn.settings.TRANSPARENT_PLOT = True

    coordinates_key = str(config.get("coordinates_key", "coordinates"))
    species_key = str(config.get("species_key", "species"))
    network_params = dict(config.get("network_params") or {})
    configured_n_atoms = config.get("n_atoms")
    configured_species = network_params.get("possible_species")
    arrays, data_summary = load_excited_state_h5_arrays(
        h5_train_dir,
        properties_list,
        coordinates_key=coordinates_key,
        species_key=species_key,
        configured_n_atoms=(
            None if configured_n_atoms is None else int(configured_n_atoms)
        ),
        configured_possible_species=configured_species,
    )
    if "dist_soft_min" not in network_params:
        raise ValueError(
            "network_params.dist_soft_min is required for production-style "
            "excited-state training filtering."
        )
    arrays, filtering_summary = filter_excited_state_training_arrays(
        arrays,
        state_table,
        coordinates_key=coordinates_key,
        dist_soft_min=float(network_params["dist_soft_min"]),
        remove_high_energy_cut=config.get("remove_high_energy_cut"),
        remove_high_energy_std=config.get("remove_high_energy_std"),
        remove_high_forces_cut=config.get("remove_high_forces_cut"),
        remove_high_forces_std=config.get("remove_high_forces_std"),
    )
    validate_filtered_split_capacity(
        int(arrays[coordinates_key].shape[0]),
        test_size=float(config.get("test_size", 0.1)),
        valid_size=float(config.get("valid_size", 0.1)),
    )
    data_summary["n_structures_before_filtering"] = int(
        data_summary["n_structures"]
    )
    data_summary["n_structures"] = int(
        arrays[coordinates_key].shape[0]
    )
    data_summary["filtering"] = filtering_summary
    network_params["possible_species"] = data_summary["possible_species"]
    n_atoms = int(data_summary["n_atoms"])

    model_root = Path(model_dir).expanduser().resolve()
    if model_root.exists() and any(model_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite nonempty model directory {model_root}."
        )
    model_root.mkdir(parents=True, exist_ok=True)

    with hippynn.tools.active_directory(str(model_root)):
        with hippynn.tools.log_terminal("training_log.txt", "wt"):
            print(f"Model ID: {int(model_id)}")
            print(f"Model seed: {seed}")
            print(f"CUDA_VISIBLE_DEVICES: {recorded_device}")
            print(f"Training device: {device}")
            print("Training filters:")
            print(json.dumps(filtering_summary, indent=2))

            species_node = inputs.SpeciesNode(db_name=species_key)
            positions_node = inputs.PositionsNode(db_name=coordinates_key)
            positions_node.requires_grad = True
            network = _build_shared_network(
                species_node,
                positions_node,
                network_choice=int(config.get("network_choice", 1)),
                network_params=network_params,
            )

            plotters: list[Any] = []
            exports = dict(config.get("exports") or {})
            export_pdf = bool(exports.get("pdf", True))
            export_png = bool(exports.get("png", False))
            (
                energy_outputs,
                force_outputs,
                total_loss,
                validation_losses,
            ) = build_excited_state_training_graph(
                network,
                positions_node,
                state_table,
                gap_table=gap_table,
                n_atoms=n_atoms,
                energy_weight=float(config.get("energy_weight", 1.0)),
                force_weight=float(config.get("force_weight", 1.0)),
                gap_weight=float(gap_options.get("weight", 1.0)),
                l2_weight=float(config.get("l2_weight", 2.0e-5)),
                first_is_interacting=bool(
                    config.get("first_is_interacting", False)
                ),
            )

            plot_dir = model_root / str(
                exports.get("output_subdir", "plots")
            )
            if export_pdf or export_png:
                plot_dir.mkdir(parents=True, exist_ok=True)
            for row, output in energy_outputs:
                if export_pdf:
                    plotters.append(
                        plotting.Hist2D.compare(
                            output,
                            saved=str(
                                plot_dir / f"{str(row['energy_key'])}.pdf"
                            ),
                            shown=False,
                        )
                    )
                if export_png:
                    plotters.append(
                        plotting.Hist2D.compare(
                            output,
                            saved=str(
                                plot_dir / f"{str(row['energy_key'])}.png"
                            ),
                            shown=False,
                        )
                    )
            for row, output in force_outputs:
                if export_pdf:
                    plotters.append(
                        plotting.Hist2D.compare(
                            output,
                            saved=str(
                                plot_dir / f"{str(row['force_key'])}.pdf"
                            ),
                            shown=False,
                        )
                    )
                if export_png:
                    plotters.append(
                        plotting.Hist2D.compare(
                            output,
                            saved=str(
                                plot_dir / f"{str(row['force_key'])}.png"
                            ),
                            shown=False,
                        )
                    )
            plot_maker = (
                plotting.PlotMaker(
                    *plotters,
                    plot_every=int(config.get("plot_frequency", 50)),
                )
                if plotters
                else None
            )

            training_modules, db_info = hippynn.experiment.assemble_for_training(
                total_loss,
                validation_losses,
                plot_maker=plot_maker,
            )
            database = Database(
                arrays,
                inputs=db_info["inputs"],
                targets=db_info["targets"],
                seed=seed,
                test_size=float(config.get("test_size", 0.1)),
                valid_size=float(config.get("valid_size", 0.1)),
                num_workers=0,
                pin_memory=device.type == "cuda",
                allow_unfound=False,
                quiet=bool(config.get("quiet_database", False)),
            )

            optimizer = torch.optim.AdamW(
                training_modules.model.parameters(),
                lr=float(config.get("learning_rate", 5.0e-4)),
            )
            scheduler = RaiseBatchSizeOnPlateau(
                optimizer=optimizer,
                **dict(
                    config.get("scheduler_options")
                    or {
                        "max_batch_size": 128,
                        "patience": 15,
                        "factor": 0.5,
                    }
                ),
            )
            controller_options = dict(config.get("controller_options") or {})
            controller_options.setdefault(
                "batch_size",
                int(config.get("batch_size", 64)),
            )
            controller_options.setdefault(
                "eval_batch_size",
                int(config.get("eval_batch_size", 128)),
            )
            controller_options.setdefault(
                "max_epochs",
                int(config.get("max_epochs", 1000)),
            )
            controller_options.setdefault(
                "termination_patience",
                int(config.get("termination_patience", 55)),
            )
            controller_options.setdefault(
                "fraction_train_eval",
                float(config.get("fraction_train_eval", 0.1)),
            )
            controller = PatienceController(
                optimizer=optimizer,
                scheduler=scheduler,
                stopping_key="Loss",
                **controller_options,
            )
            setup_params = SetupParams(
                controller=controller,
                device=device,
            )
            training_modules, controller, metric_tracker = setup_training(
                training_modules=training_modules,
                setup_params=setup_params,
            )
            metric_tracker = train_model(
                training_modules,
                database,
                controller,
                metric_tracker,
                callbacks=None,
                batch_callbacks=None,
            )

            summary = {
                "model_id": int(model_id),
                "seed": seed,
                "device": str(device),
                "cuda_visible_devices": recorded_device,
                "state_table": state_table,
                "gap_table": gap_table,
                "gap_targets": {
                    "enabled": bool(gap_table),
                    "weight": float(gap_options.get("weight", 1.0)),
                    "mode": "derived",
                },
                "data": data_summary,
                "metric": _json_safe(metric_tracker.best_metric_values),
                "average_epoch_time": float(
                    np.average(metric_tracker.epoch_times)
                ),
            }
            Path("training_summary.json").write_text(
                json.dumps(_json_safe(summary), indent=2),
                encoding="utf-8",
            )

    return {
        "model_id": int(model_id),
        "model_dir": str(model_root),
        "device": recorded_device,
        "seed": seed,
    }


def _excited_state_training_worker(
    arg_dict: dict[str, Any],
) -> dict[str, Any]:
    try:
        payload = train_single_excited_state_model(**arg_dict)
        return {"ok": True, "payload": payload}
    except Exception as exc:
        return {
            "ok": False,
            "payload": {
                "model_id": int(arg_dict["model_id"]),
                "model_dir": str(arg_dict["model_dir"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        }


def _training_complete(model_dir: str) -> bool:
    log_path = Path(model_dir) / "training_log.txt"
    return (
        log_path.exists()
        and "Training complete" in log_path.read_text(encoding="utf-8")
    )


def _validate_completed_ensemble(
    ensemble_dir: str,
    state_table: list[dict[str, Any]],
    gap_table: list[dict[str, Any]] | None = None,
) -> None:
    """Ensure saved checkpoints expose every target used by ALCHEMI."""

    import hippynn

    expected_targets = []
    for row in state_table:
        expected_targets.extend(
            [str(row["energy_db_name"]), str(row["force_db_name"])]
        )
    expected_targets.extend(
        str(row["gap_db_name"]) for row in (gap_table or [])
    )
    _, (_, output_info) = hippynn.graphs.make_ensemble(
        os.path.join(str(ensemble_dir), "model-*"),
        targets=expected_targets,
        quiet=True,
    )
    member_count = len(glob.glob(os.path.join(str(ensemble_dir), "model-*")))
    if member_count < 1:
        raise RuntimeError("No completed HIPPYNN model members were found.")
    missing = [
        target
        for target in expected_targets
        if int(output_info.get(target, 0)) != member_count
    ]
    if missing:
        raise RuntimeError(
            "Completed checkpoints are missing required excited-state "
            "outputs: " + ", ".join(missing)
        )


def _worker_count(
    ML_config: dict[str, Any],
    *,
    n_models: int,
    gpus_per_node: int,
) -> int:
    if str(ML_config.get("device_string", "from_multiprocessing")).lower() == "cpu":
        return 1
    gpu_count = int(gpus_per_node)
    return max(1, min(gpu_count if gpu_count > 0 else 1, int(n_models)))


def train_excited_state_ensemble(
    *,
    ML_config: dict[str, Any],
    h5_dir: str,
    model_path: str,
    current_training_id: int,
    gpus_per_node: int,
    properties_list: dict[str, list[Any]],
) -> tuple[list[bool], int]:
    """Train all model members and preserve ALF's promotion contract."""

    config = dict(ML_config or {})
    state_table = validate_excited_state_training_config(
        config,
        properties_list,
    )
    gap_options = dict(config.get("gap_targets") or {})
    gap_table = (
        derive_gap_property_table(
            properties_list,
            gap_config=gap_options,
            require_properties=True,
        )
        if bool(gap_options.get("enabled", False))
        else []
    )
    n_models = int(config["n_models"])
    ensemble_dir = model_path.format(int(current_training_id))
    params_list = [
        {
            "model_id": model_id,
            "model_dir": os.path.join(
                ensemble_dir,
                f"model-{model_id:02d}",
            ),
            "h5_train_dir": str(h5_dir),
            "properties_list": dict(properties_list),
            "ML_config": config,
            "from_multiprocessing_nGPU": (
                int(gpus_per_node) if int(gpus_per_node) > 0 else None
            ),
        }
        for model_id in range(n_models)
    ]

    context = multiprocessing.get_context("spawn")
    pool = context.Pool(
        processes=_worker_count(
            config,
            n_models=n_models,
            gpus_per_node=int(gpus_per_node),
        )
    )
    try:
        raw_results = pool.map(_excited_state_training_worker, params_list)
    finally:
        pool.close()
        pool.join()

    completed: list[bool] = []
    for result in raw_results:
        payload = result["payload"]
        model_dir = Path(payload["model_dir"]).expanduser().resolve()
        if bool(result["ok"]):
            member_complete = _training_complete(str(model_dir))
            completed.append(member_complete)
            if not member_complete:
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "training_error.json").write_text(
                    json.dumps(
                        {
                            "model_id": int(payload["model_id"]),
                            "model_dir": str(model_dir),
                            "error_type": "IncompleteTrainingError",
                            "error": (
                                "HIPPYNN returned without a single "
                                "'Training complete' marker."
                            ),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        else:
            completed.append(False)
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "training_error.json").write_text(
                json.dumps(_json_safe(payload), indent=2),
                encoding="utf-8",
            )

    if all(completed):
        try:
            if gap_table:
                _validate_completed_ensemble(
                    ensemble_dir,
                    state_table,
                    gap_table,
                )
            else:
                _validate_completed_ensemble(ensemble_dir, state_table)
        except Exception as exc:
            completed = [False for _ in completed]
            ensemble_root = Path(ensemble_dir).expanduser().resolve()
            ensemble_root.mkdir(parents=True, exist_ok=True)
            (ensemble_root / "training_error.json").write_text(
                json.dumps(
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    return completed, int(current_training_id)


def validate_excited_state_task_options(
    *,
    remove_existing: bool,
    h5_test_dir: str | None,
) -> None:
    """Reject task options that would delete or silently ignore data."""

    if bool(remove_existing):
        raise ValueError(
            "remove_existing=True is not supported because this trainer will "
            "not delete or overwrite model data."
        )
    if h5_test_dir is not None:
        raise ValueError(
            "h5_test_dir is not supported by the isolated excited-state "
            "trainer; use valid_size and test_size splits."
        )


@python_app(executors=["alf_ML_executor"])
def train_excited_state_HIPPYNN_ensemble_task(
    ML_config,
    h5_dir,
    model_path,
    current_training_id,
    gpus_per_node,
    properties_list,
    remove_existing=False,
    h5_test_dir=None,
):
    """Parsl entry point for isolated multi-state HIPPYNN ensembles."""

    validate_excited_state_task_options(
        remove_existing=remove_existing,
        h5_test_dir=h5_test_dir,
    )
    return train_excited_state_ensemble(
        ML_config=dict(ML_config or {}),
        h5_dir=str(h5_dir),
        model_path=str(model_path),
        current_training_id=int(current_training_id),
        gpus_per_node=int(gpus_per_node),
        properties_list=dict(properties_list or {}),
    )
