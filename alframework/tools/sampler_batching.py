from __future__ import annotations

from typing import Any

from alframework.tools.molecules_class import MoleculesObject


def _normalized_dynamics_backend(sampler_config: dict[str, Any]) -> str:
    backend = str(sampler_config.get("dynamics_backend", "ase")).strip().lower()
    return {
        "ase_langevin": "ase",
        "alchemi": "alchemi_baoab",
        "nvalchemi": "alchemi_baoab",
        "nvalchemi_langevin": "alchemi_baoab",
    }.get(backend, backend)


def alchemi_sampler_batch_size(sampler_config: dict[str, Any]) -> int:
    if _normalized_dynamics_backend(sampler_config) != "alchemi_baoab":
        return 1
    return max(1, int(dict(sampler_config.get("alchemi_baoab") or {}).get("batch_size", 1)))


def alchemi_allow_partial_batches(sampler_config: dict[str, Any]) -> bool:
    return bool(dict(sampler_config.get("alchemi_baoab") or {}).get("allow_partial_batches", False))


def selected_state_buffer_key(molecule: MoleculesObject) -> int:
    metadata = molecule.get_metadata()
    if "excited_state" in metadata:
        return int(metadata["excited_state"])
    if "selected_state" in metadata:
        return int(metadata["selected_state"])
    raise ValueError(
        "Batched ALCHEMI sampling requires builder metadata to include "
        "'excited_state' or 'selected_state'."
    )


def add_molecule_to_same_state_buffer(
    buffers: dict[int, list[MoleculesObject]],
    molecule: MoleculesObject,
) -> None:
    buffers.setdefault(selected_state_buffer_key(molecule), []).append(molecule)


def pop_ready_same_state_batches(
    buffers: dict[int, list[MoleculesObject]],
    *,
    batch_size: int,
    allow_partial: bool = False,
    force_partial: bool = False,
) -> list[list[MoleculesObject]]:
    ready: list[list[MoleculesObject]] = []
    size = max(1, int(batch_size))
    for state in sorted(list(buffers)):
        bucket = buffers[state]
        while len(bucket) >= size:
            ready.append(bucket[:size])
            del bucket[:size]
        if allow_partial and force_partial and bucket:
            ready.append(list(bucket))
            bucket.clear()
        if not bucket:
            del buffers[state]
    return ready
