"""Batching helpers for sampler tasks.

The active-learning driver remains agnostic to the sampler implementation.  A
sampler opts into batching by defining an ``alchemi_baoab`` configuration
block; existing samplers continue to receive one ``MoleculesObject`` at a
time.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from alframework.tools.molecules_class import MoleculesObject


_MOLECULE_ID_INTEGER_RE = re.compile(r"(\d+)(?!.*\d)")
_MISSING = object()


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


def _nonnegative_integer(value: Any, *, name: str) -> int:
    """Return one validated nonnegative queue or capacity count."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer.")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be a nonnegative integer.")
    if normalized < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return normalized


def _state_index(value: Any, *, name: str) -> int:
    """Return one validated, nonnegative electronic-state index."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer, not {value!r}.")
    try:
        state = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a nonnegative integer; received {value!r}."
        ) from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(
            f"{name} must be a nonnegative integer; received {value!r}."
        )
    if state < 0:
        raise ValueError(f"{name} must be nonnegative; received {state}.")
    return state


def _configured_state_selection(
    sampler_config: dict[str, Any],
) -> tuple[str, tuple[int, ...]] | None:
    """Validate and normalize the optional sampler-level state policy."""

    raw_selection = sampler_config.get("state_selection")
    if raw_selection is None:
        if "selected_state" not in sampler_config:
            return None
        state = _state_index(
            sampler_config["selected_state"],
            name="sampler_config.selected_state",
        )
        return "fixed", (state,)
    if not isinstance(raw_selection, dict):
        raise ValueError("state_selection must be a dictionary when provided.")

    mode = str(raw_selection.get("mode", "")).strip().lower()
    if mode == "fixed":
        raw_state = raw_selection.get(
            "state",
            raw_selection.get(
                "selected_state",
                sampler_config.get("selected_state", _MISSING),
            ),
        )
        if raw_state is _MISSING:
            raise ValueError(
                "state_selection.mode='fixed' requires state_selection.state."
            )
        return "fixed", (
            _state_index(raw_state, name="state_selection.state"),
        )

    if mode == "batch_cycle":
        raw_states = raw_selection.get("states")
        if not isinstance(raw_states, (list, tuple)) or not raw_states:
            raise ValueError(
                "state_selection.mode='batch_cycle' requires a non-empty "
                "state_selection.states list."
            )
        states = tuple(
            _state_index(value, name=f"state_selection.states[{index}]")
            for index, value in enumerate(raw_states)
        )
        if len(set(states)) != len(states):
            raise ValueError(
                "state_selection.states must not contain duplicate states."
            )
        return "batch_cycle", states

    raise ValueError(
        "state_selection.mode must be either 'fixed' or 'batch_cycle'."
    )


def validate_state_selection(
    sampler_config: dict[str, Any],
    properties_list: dict[str, Any] | None = None,
) -> tuple[str, tuple[int, ...]] | None:
    """Validate state selection and optionally check configured properties."""

    mode = str(sampler_config.get("model_mode", "ground_state")).strip().lower()
    if mode not in {"ground_state", "excited_state"}:
        raise ValueError(
            "model_mode must be either 'ground_state' or 'excited_state'."
        )
    selection = _configured_state_selection(sampler_config)
    if mode == "ground_state":
        return selection
    if selection is None or properties_list is None:
        return selection

    from alframework.tools.excited_state_tools import derive_state_property_table

    available = {
        int(row["state"])
        for row in derive_state_property_table(properties_list)
    }
    unavailable = sorted(set(selection[1]) - available)
    if unavailable:
        raise ValueError(
            f"Configured selected states {unavailable} are not present in "
            f"properties_list; available states are {sorted(available)}."
        )
    return selection


def sampler_batching_signature(
    master_config: dict[str, Any],
    sampler_config: dict[str, Any],
) -> dict[str, Any]:
    """Return the batching-critical portion of the live ALF configuration."""

    batched = sampler_uses_batches(sampler_config)
    mode = str(
        sampler_config.get("model_mode", "ground_state")
    ).strip().lower()
    if mode not in {"ground_state", "excited_state"}:
        raise ValueError(
            "model_mode must be either 'ground_state' or 'excited_state'."
        )
    if batched:
        batch_size = sampler_batch_size(sampler_config)
        partial_policy = str(
            sampler_config["alchemi_baoab"].get(
                "partial_policy", "full_only"
            )
        ).strip().lower()
        selection = _configured_state_selection(sampler_config)
        state_selection = (
            None
            if selection is None
            else {
                "mode": selection[0],
                "states": list(selection[1]),
            }
        )
    else:
        batch_size = 1
        partial_policy = None
        state_selection = None
    return {
        "sampler_task": str(master_config.get("sampler_task", "")),
        "batched": bool(batched),
        "batch_size": int(batch_size),
        "partial_policy": partial_policy,
        "model_mode": mode,
        "state_selection": state_selection,
    }


def changed_batching_signature_fields(
    current_master_config: dict[str, Any],
    current_sampler_config: dict[str, Any],
    new_master_config: dict[str, Any],
    new_sampler_config: dict[str, Any],
) -> list[str]:
    """Return batching-critical fields changed by a proposed hot reload."""

    current = sampler_batching_signature(
        current_master_config, current_sampler_config
    )
    proposed = sampler_batching_signature(
        new_master_config, new_sampler_config
    )
    return sorted(
        key for key in current if current[key] != proposed[key]
    )


def validate_batching_config_reload(
    current_master_config: dict[str, Any],
    current_sampler_config: dict[str, Any],
    new_master_config: dict[str, Any],
    new_sampler_config: dict[str, Any],
    sampler_batch_buffer: "SamplerBatchBuffer",
) -> list[str]:
    """Reject incompatible hot reloads while strict-full inputs are waiting.

    The proposed signature is always validated.  A critical change is safe
    once no molecule remains in the strict-full input buffer.  Already
    submitted tasks retain their recorded replica widths separately.
    """

    changed = changed_batching_signature_fields(
        current_master_config,
        current_sampler_config,
        new_master_config,
        new_sampler_config,
    )
    buffered_count = len(sampler_batch_buffer)
    if changed and buffered_count:
        raise ValueError(
            "Cannot apply batching-critical configuration changes while "
            f"{buffered_count} sampler structure(s) are buffered. "
            "Changed fields: "
            + ", ".join(changed)
            + ". ALF retained the current configuration and buffer; retry "
            "after the buffer is empty."
        )
    return changed


def sampler_task_replicas(
    tasks: Iterable[Any],
    task_replica_counts: dict[int, Any],
    *,
    completed_only: bool = False,
) -> int:
    """Count input replicas represented by registered sampler tasks.

    Replica widths are recorded when each task is submitted.  This keeps
    accounting correct when a batching-critical reload is accepted after the
    input buffer empties but tasks from the previous batch width are still
    running.
    """

    total = 0
    for task in tasks:
        if completed_only and not task.done():
            continue
        task_key = id(task)
        if task_key not in task_replica_counts:
            raise RuntimeError(
                "Sampler task is missing its input-replica accounting entry."
            )
        total += _nonnegative_integer(
            task_replica_counts[task_key],
            name="sampler task replica count",
        )
    return total


def sampler_capacity_status(
    *,
    parallel_sampler_limit: Any,
    submitted_sampler_replicas: Any,
    builder_task_count: Any,
    maximum_builder_structures: Any,
    buffered_structure_count: Any,
) -> dict[str, int]:
    """Return replica-based accounting for ALF's sampler capacity."""

    limit = _nonnegative_integer(
        parallel_sampler_limit,
        name="parallel_sampler_limit",
    )
    submitted = _nonnegative_integer(
        submitted_sampler_replicas,
        name="submitted_sampler_replicas",
    )
    builder_tasks = _nonnegative_integer(
        builder_task_count,
        name="builder_task_count",
    )
    builder_width = _nonnegative_integer(
        maximum_builder_structures,
        name="maximum_builder_structures",
    )
    if builder_width < 1:
        raise ValueError("maximum_builder_structures must be at least one.")
    buffered = _nonnegative_integer(
        buffered_structure_count,
        name="buffered_structure_count",
    )
    builder_capacity = builder_tasks * builder_width
    total = submitted + builder_capacity + buffered
    return {
        "parallel_sampler_limit": limit,
        "submitted_sampler_replicas": submitted,
        "pending_builder_replicas": builder_capacity,
        "buffered_replicas": buffered,
        "accounted_replicas": total,
        "available_replica_slots": max(0, limit - total),
    }


def sampler_has_builder_capacity(
    capacity_status: dict[str, Any],
    maximum_builder_structures: Any,
) -> bool:
    """Return whether one more builder fits without exceeding the limit."""

    builder_width = _nonnegative_integer(
        maximum_builder_structures,
        name="maximum_builder_structures",
    )
    if builder_width < 1:
        raise ValueError("maximum_builder_structures must be at least one.")
    available = _nonnegative_integer(
        capacity_status.get("available_replica_slots"),
        name="sampler_capacity.available_replica_slots",
    )
    return available >= builder_width


def configured_sampler_calculator_status(
    sampler_config: dict[str, Any],
) -> dict[str, str] | None:
    """Describe the configured calculator for status output."""

    if not sampler_uses_batches(sampler_config):
        return None
    native_loader = sampler_config.get("alchemi_calculator")
    if native_loader:
        return {"interface": "native", "loader": str(native_loader)}
    ase_loader = sampler_config.get("ase_calculator")
    if ase_loader:
        return {"interface": "ase_fallback", "loader": str(ase_loader)}
    return {"interface": "unconfigured", "loader": ""}


def _molecule_sequence_index(molecule: MoleculesObject) -> int:
    """Derive a stable nonnegative sequence index from an ALF molecule ID."""

    molecule_id = str(molecule.get_moleculeid())
    match = _MOLECULE_ID_INTEGER_RE.search(molecule_id)
    if match is not None:
        return int(match.group(1))
    digest = hashlib.sha256(molecule_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def selected_state(
    molecule: MoleculesObject,
    sampler_config: dict[str, Any] | None = None,
) -> int:
    """Resolve and record the selected excited state for one molecule."""

    if not isinstance(molecule, MoleculesObject):
        raise TypeError("selected_state requires a MoleculesObject instance.")

    metadata = molecule.get_metadata()
    if "selected_state" in metadata:
        state = _state_index(
            metadata["selected_state"], name="molecule metadata selected_state"
        )
        if "selected_state_source" not in metadata:
            molecule.update_metadata(
                {"selected_state_source": "metadata.selected_state"}
            )
        return state
    if "excited_state" in metadata:
        state = _state_index(
            metadata["excited_state"], name="molecule metadata excited_state"
        )
        molecule.update_metadata(
            {
                "selected_state": state,
                "selected_state_source": "metadata.excited_state",
            }
        )
        return state

    selection = _configured_state_selection(dict(sampler_config or {}))
    if selection is not None:
        mode, states = selection
        if mode == "fixed":
            state = int(states[0])
            selection_metadata = {
                "selected_state": state,
                "selected_state_source": "state_selection.fixed",
            }
        else:
            sequence_index = _molecule_sequence_index(molecule)
            batch_size = sampler_batch_size(dict(sampler_config or {}))
            batch_ordinal = sequence_index // batch_size
            cycle_index = batch_ordinal % len(states)
            state = int(states[cycle_index])
            selection_metadata = {
                "selected_state": state,
                "selected_state_source": "state_selection.batch_cycle",
                "selected_state_sequence_index": int(sequence_index),
                "selected_state_batch_ordinal": int(batch_ordinal),
                "selected_state_cycle_index": int(cycle_index),
            }
        molecule.update_metadata(selection_metadata)
        return state

    raise ValueError(
        "Excited-state ALCHEMI batching requires molecule metadata to contain "
        "'selected_state' or sampler_config.state_selection to define a "
        "fixed or batch-cycle policy."
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
        state = selected_state(molecule, sampler_config)
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


def sampler_task_feed(
    molecule_objects: list[MoleculesObject],
    sampler_config: dict[str, Any],
) -> dict[str, Any]:
    """Build the task feed for a legacy or strict-full sampler."""

    molecules = list(molecule_objects)
    if sampler_uses_batches(sampler_config):
        expected = sampler_batch_size(sampler_config)
        if len(molecules) != expected:
            raise ValueError(
                "Strict-full sampler tasks require exactly "
                f"{expected} molecules; received {len(molecules)}."
            )
        return {
            "molecule_objects": molecules,
            "sampler_config": sampler_config,
        }
    if len(molecules) != 1:
        raise ValueError("Legacy sampler tasks accept exactly one molecule.")
    return {
        "molecule_object": molecules[0],
        "sampler_config": sampler_config,
    }


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


def sampler_submission_groups(
    molecule: MoleculesObject,
    sampler_config: dict[str, Any],
    sampler_batch_buffer: SamplerBatchBuffer,
) -> list[list[MoleculesObject]]:
    """Route one builder result into legacy or strict-full submission groups."""

    if not isinstance(molecule, MoleculesObject):
        raise TypeError(
            "Sampler routing requires a MoleculesObject instance."
        )
    if not sampler_uses_batches(sampler_config):
        return [[molecule]]
    sampler_batch_buffer.add(molecule, sampler_config)
    return sampler_batch_buffer.pop_ready(
        sampler_batch_size(sampler_config)
    )
