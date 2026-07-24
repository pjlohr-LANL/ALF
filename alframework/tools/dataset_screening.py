"""Hard acceptance screens applied before ALF stores new QM labels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from alframework.tools.molecular_topology import (
    load_reference_topology,
    validate_fixed_topology,
)
from alframework.tools.molecules_class import MoleculesObject


_SCREEN_KEYS = frozenset({"enabled", "force", "min_distance", "topology"})


def dataset_screening_options(
    sampler_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate and normalize the optional production screening contract."""

    config = dict(sampler_config or {})
    raw = config.get("dataset_screening", False)
    if isinstance(raw, bool):
        enabled = raw
        settings: dict[str, Any] = {}
    elif isinstance(raw, dict):
        unknown = sorted(set(raw) - _SCREEN_KEYS)
        if unknown:
            raise ValueError(
                "Unknown dataset_screening options: " + ", ".join(unknown)
            )
        enabled = bool(raw.get("enabled", False))
        settings = dict(raw)
    else:
        raise TypeError("dataset_screening must be a Boolean or dictionary.")

    max_force = float(config.get("max_force_cutoff", 10.0))
    min_distance = float(
        config.get(
            "min_distance_cutoff",
            config.get("distcut", 0.3),
        )
    )
    if not np.isfinite(max_force) or max_force <= 0:
        raise ValueError("max_force_cutoff must be finite and positive.")
    if not np.isfinite(min_distance) or min_distance <= 0:
        raise ValueError("min_distance_cutoff must be finite and positive.")
    return {
        "enabled": enabled,
        "force": bool(settings.get("force", True)),
        "min_distance": bool(settings.get("min_distance", True)),
        "topology": bool(settings.get("topology", False)),
        "max_force_cutoff": max_force,
        "min_distance_cutoff": min_distance,
    }


def _force_property_keys(
    properties_list: dict[str, list[Any]],
    results: dict[str, Any],
) -> list[str]:
    keys: list[str] = []
    for property_key, schema in dict(properties_list or {}).items():
        if (
            len(schema) >= 2
            and str(schema[1]).strip().lower() == "atomic"
            and str(property_key).startswith("F")
        ):
            keys.append(str(property_key))
    missing = [key for key in keys if key not in results]
    if missing:
        raise ValueError(
            "Dataset screening requires every configured force result; "
            "missing " + ", ".join(missing) + "."
        )
    return keys


def dataset_screening_metrics(
    molecule: MoleculesObject,
    properties_list: dict[str, list[Any]],
    sampler_config: dict[str, Any],
    *,
    master_directory: str | None = None,
) -> dict[str, Any]:
    """Return force, distance, and fixed-topology acceptance metrics."""

    if not isinstance(molecule, MoleculesObject):
        raise TypeError("Dataset screening requires MoleculesObject inputs.")
    atoms = molecule.get_atoms()
    if atoms is None:
        raise ValueError("Dataset screening cannot inspect missing atoms.")
    results = molecule.get_results()
    force_maxima: list[float] = []
    for force_key in _force_property_keys(properties_list, results):
        values = np.asarray(results[force_key], dtype=np.float64)
        if values.shape != (len(atoms), 3):
            raise ValueError(
                f"Dataset screening expected {force_key} to have shape "
                f"{(len(atoms), 3)}; received {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            force_maxima.append(float("inf"))
        else:
            force_maxima.append(
                float(np.max(np.linalg.vector_norm(values, axis=1)))
            )

    if len(atoms) > 1:
        distances = np.asarray(
            atoms.get_all_distances(mic=bool(np.any(atoms.get_pbc()))),
            dtype=np.float64,
        )
        np.fill_diagonal(distances, np.inf)
        minimum_distance = float(np.min(distances))
    else:
        minimum_distance = float("inf")

    topology = load_reference_topology(
        sampler_config,
        master_directory=master_directory,
    )
    topology_result = validate_fixed_topology(atoms, topology)
    return {
        "max_force_norm": (
            max(force_maxima) if force_maxima else None
        ),
        "min_distance": minimum_distance,
        "topology_valid": bool(topology_result.valid),
        "topology_reject_reason": topology_result.reason,
        "topology_violations": [
            dict(violation) for violation in topology_result.violations
        ],
    }


def filter_dataset_screening(
    molecules: list[MoleculesObject],
    properties_list: dict[str, list[Any]],
    sampler_config: dict[str, Any],
    *,
    master_directory: str | None = None,
) -> tuple[list[MoleculesObject], dict[str, Any]]:
    """Reject converged labels that violate enabled production screens."""

    options = dataset_screening_options(sampler_config)
    summary = {
        **options,
        "total": len(molecules),
        "kept": 0,
        "rejected_force": 0,
        "rejected_min_distance": 0,
        "rejected_topology": 0,
        "rejected_unusable": 0,
    }
    if not options["enabled"]:
        summary["kept"] = len(molecules)
        return list(molecules), summary

    kept: list[MoleculesObject] = []
    for molecule in molecules:
        if (
            not isinstance(molecule, MoleculesObject)
            or molecule.get_atoms() is None
            or not molecule.check_convergence()
        ):
            summary["rejected_unusable"] += 1
            continue
        metrics = dataset_screening_metrics(
            molecule,
            properties_list,
            sampler_config,
            master_directory=master_directory,
        )
        reject_force = bool(
            options["force"]
            and metrics["max_force_norm"] is not None
            and float(metrics["max_force_norm"])
            > float(options["max_force_cutoff"])
        )
        reject_distance = bool(
            options["min_distance"]
            and float(metrics["min_distance"])
            < float(options["min_distance_cutoff"])
        )
        reject_topology = bool(
            options["topology"] and not metrics["topology_valid"]
        )
        summary["rejected_force"] += int(reject_force)
        summary["rejected_min_distance"] += int(reject_distance)
        summary["rejected_topology"] += int(reject_topology)
        molecule.update_metadata(
            {
                "dataset_screening": {
                    **metrics,
                    "rejected": bool(
                        reject_force or reject_distance or reject_topology
                    ),
                }
            }
        )
        if not (reject_force or reject_distance or reject_topology):
            kept.append(molecule)
    summary["kept"] = len(kept)
    return kept, summary


def screen_and_store_dataset(
    h5_path: str,
    molecules: list[MoleculesObject],
    properties_list: dict[str, list[Any]],
    sampler_config: dict[str, Any],
    *,
    master_directory: str | None = None,
    store_function: Callable[..., Any] | None = None,
) -> tuple[bool, list[MoleculesObject], dict[str, Any]]:
    """Screen a completed QM batch and create a shard only when data remain."""

    kept, summary = filter_dataset_screening(
        molecules,
        properties_list,
        sampler_config,
        master_directory=master_directory,
    )
    if not kept:
        return False, kept, summary
    if store_function is None:
        from alframework.tools.tools import store_current_data

        store_function = store_current_data
    store_function(h5_path, kept, properties_list)
    return True, kept, summary


def print_dataset_screening_summary(summary: dict[str, Any]) -> None:
    """Print one compact screening summary for driver logs."""

    if not bool(summary.get("enabled", False)):
        return
    print("Dataset screening summary")
    print(f"Total QM results: {int(summary['total'])}")
    print(f"Kept results: {int(summary['kept'])}")
    print(f"Rejected by force: {int(summary['rejected_force'])}")
    print(
        "Rejected by minimum distance: "
        f"{int(summary['rejected_min_distance'])}"
    )
    print(f"Rejected by topology: {int(summary['rejected_topology'])}")
    print(f"Rejected as unusable: {int(summary['rejected_unusable'])}")
