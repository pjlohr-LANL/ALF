"""Opaque model handoffs without assumptions about checkpoint families."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelReference:
    artifact_location: str
    generation: int | str
    adapter: str
    interface: str = "native"
    options: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.artifact_location or not isinstance(self.adapter, str) or not self.adapter.strip():
            raise ValueError("A model reference requires an artifact location and inference adapter.")
        if self.interface not in {"native", "ase"}:
            raise ValueError("Model reference interface must be 'native' or 'ase'.")
        object.__setattr__(self, "artifact_location", str(self.artifact_location))
        object.__setattr__(self, "options", deepcopy(dict(self.options)))
        object.__setattr__(self, "capabilities", deepcopy(dict(self.capabilities)))
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        values = dict(value)
        if "loader" in values and "adapter" not in values:
            values["adapter"] = values.pop("loader")
        if "location" in values and "artifact_location" not in values:
            values["artifact_location"] = values.pop("location")
        return cls(**values)

    def as_dict(self):
        return deepcopy({"artifact_location": self.artifact_location,
                         "generation": self.generation, "adapter": self.adapter,
                         "interface": self.interface, "options": self.options,
                         "capabilities": self.capabilities, "metadata": self.metadata})


def legacy_training_reference(result, *, model_path, inference, capabilities=None, metadata=None):
    """Adapt upstream's ``(member_success, generation)`` training return value.

    Incomplete ensembles return ``None`` and are not promoted. Only the selected
    adapter knows what files exist at the resulting opaque artifact location.
    """
    success, generation = result
    success = list(success)
    if not success or not all(success):
        return None
    generation = int(generation)
    if generation < 0:
        raise ValueError("A completed model generation must be nonnegative.")
    adapter = inference.get("loader", inference.get("adapter"))
    return ModelReference(str(model_path).format(generation), generation, adapter,
                          str(inference.get("interface", "native")),
                          deepcopy(inference.get("options") or {}),
                          deepcopy(capabilities or {}), deepcopy(metadata or {}))
