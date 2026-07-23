"""Batching helpers for sampler tasks.

The active-learning driver remains agnostic to the sampler implementation.  A
sampler opts into batching by defining an ``alchemi_baoab`` configuration
block; existing samplers continue to receive one ``MoleculesObject`` at a
time.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from alframework.tools.molecules_class import MoleculesObject


def sampler_uses_batches(sampler_config: dict[str, Any]) -> bool:
    """Return whether the configured sampler consumes molecule lists."""

    return isinstance(sampler_config.get("alchemi_baoab"), dict)


def sampler_batch_size(sampler_config: dict[str, Any]) -> int:
    """Return and validate the requested sampler batch size."""

    if not sampler_uses_batches(sampler_config):
        return 1
    raw = sampler_config["alchemi_baoab"]
    size = int(raw.get("batch_size", 1))
    if size < 1:
        raise ValueError("alchemi_baoab.batch_size must be at least one.")
    policy = str(raw.get("partial_policy", "full_only")).strip().lower()
    if policy != "full_only":
        raise ValueError(
            "Only alchemi_baoab.partial_policy='full_only' is currently supported."
        )
    return size


def selected_state(molecule: MoleculesObject) -> int:
    """Return the selected excited state stored on a molecule."""

    metadata = molecule.get_metadata()
    if "selected_state" in metadata:
        return int(metadata["selected_state"])
    if "excited_state" in metadata:
        return int(metadata["excited_state"])
    raise ValueError(
        "Excited-state ALCHEMI batching requires molecule metadata to contain "
        "'selected_state'."
    )


def sampler_batch_key(
    molecule: MoleculesObject,
    sampler_config: dict[str, Any],
) -> tuple[tuple[int, ...], int | None]:
    """Build the compatibility key for one batched sampler input.

    Exact atom order is deliberately retained because HIPPYNN's padded batch
    inputs and state-specific dynamics must have the same layout in a batch.
    """

    if not isinstance(molecule, MoleculesObject):
        raise TypeError("Sampler batches may contain only MoleculesObject instances.")
    atoms = molecule.get_atoms()
    if atoms is None:
        raise ValueError("Cannot batch a MoleculesObject whose atoms are None.")
    atomic_numbers = tuple(int(value) for value in atoms.get_atomic_numbers())
    mode = str(sampler_config.get("model_mode", "ground_state")).strip().lower()
    if mode == "ground_state":
        state = None
    elif mode == "excited_state":
        state = selected_state(molecule)
    else:
        raise ValueError(
            "model_mode must be either 'ground_state' or 'excited_state'."
        )
    return atomic_numbers, state


def flatten_molecule_output(output: Any) -> list[MoleculesObject]:
    """Normalize nested single/list sampler or builder output."""

    if output is None:
        return []
    if isinstance(output, MoleculesObject):
        return [output]
    if isinstance(output, (list, tuple)):
        flattened: list[MoleculesObject] = []
        for item in output:
            flattened.extend(flatten_molecule_output(item))
        return flattened
    raise TypeError(
        "Sampler output must be a MoleculesObject, a nested list/tuple of "
        f"MoleculesObject instances, or None; got {type(output).__name__}."
    )


@dataclass
class _BufferedMolecule:
    molecule: MoleculesObject
    buffered_at: float


class SamplerBatchBuffer:
    """In-memory strict-full batching buffer used by the ALF driver."""

    def __init__(self) -> None:
        self._buckets: dict[
            tuple[tuple[int, ...], int | None], list[_BufferedMolecule]
        ] = defaultdict(list)

    def __len__(self) -> int:
        return sum(len(bucket) for bucket in self._buckets.values())

    def add(
        self,
        molecule: MoleculesObject,
        sampler_config: dict[str, Any],
        *,
        buffered_at: float | None = None,
    ) -> None:
        key = sampler_batch_key(molecule, sampler_config)
        self._buckets[key].append(
            _BufferedMolecule(
                molecule=molecule,
                buffered_at=time.time() if buffered_at is None else float(buffered_at),
            )
        )

    def pop_ready(self, batch_size: int) -> list[list[MoleculesObject]]:
        """Remove and return every complete compatibility batch."""

        size = int(batch_size)
        if size < 1:
            raise ValueError("batch_size must be at least one.")
        ready: list[list[MoleculesObject]] = []
        for key in sorted(self._buckets, key=repr):
            bucket = self._buckets[key]
            while len(bucket) >= size:
                ready.append([entry.molecule for entry in bucket[:size]])
                del bucket[:size]
        self._buckets = defaultdict(
            list, {key: bucket for key, bucket in self._buckets.items() if bucket}
        )
        return ready

    def status(self, *, now: float | None = None) -> dict[str, Any]:
        """Return JSON-safe visibility into incomplete strict-full batches."""

        current_time = time.time() if now is None else float(now)
        buckets = []
        oldest_timestamp: float | None = None
        for (atomic_numbers, state), entries in sorted(
            self._buckets.items(), key=lambda item: repr(item[0])
        ):
            if not entries:
                continue
            oldest = min(entry.buffered_at for entry in entries)
            oldest_timestamp = oldest if oldest_timestamp is None else min(oldest_timestamp, oldest)
            buckets.append(
                {
                    "atomic_numbers": list(atomic_numbers),
                    "selected_state": state,
                    "count": len(entries),
                    "oldest_age_seconds": max(0.0, current_time - oldest),
                }
            )
        return {
            "buffered_structures": len(self),
            "oldest_age_seconds": (
                None
                if oldest_timestamp is None
                else max(0.0, current_time - oldest_timestamp)
            ),
            "buckets": buckets,
        }

    def molecules(self) -> Iterable[MoleculesObject]:
        """Iterate buffered molecules without removing them (primarily for tests)."""

        for bucket in self._buckets.values():
            for entry in bucket:
                yield entry.molecule
