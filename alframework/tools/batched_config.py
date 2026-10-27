"""Dependency-free normalization of portable and legacy sampler settings."""

from __future__ import annotations

import math
from typing import Any


_BATCH_KEYS = frozenset({"batch_size", "partial_policy", "max_wait_seconds"})


def portable_batching(config: dict[str, Any]) -> bool:
    """Return whether the caller opted into the portable sampler interface."""

    return "batching" in config or "dynamics" in config


def _block(config, name):
    value = config.get(name, {})
    if value is None and name.endswith("_options"):
        return {}
    if value is None and name.endswith("_options"):
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a configuration dictionary.")
    return dict(value)


def _merge_options(legacy, current, name):
    for key in legacy.keys() & current.keys():
        if legacy[key] != current[key]:
            raise ValueError(f"Conflicting legacy and {name} settings for {key}.")
    return {**legacy, **current}


def batching_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize batching without importing a dynamics engine or tensor stack.

    Portable configurations default to a bounded wait. Legacy configurations
    keep strict-full batching, including when read back from a durable journal.
    Explicit duplicate legacy and portable options must agree.
    """

    portable = portable_batching(config)
    legacy = _block(config, "alchemi_baoab")
    current = _block(config, "batching")
    raw = _merge_options(
        {key: value for key, value in legacy.items() if key in _BATCH_KEYS},
        current,
        "batching",
    )
    source = "batching" if portable else "alchemi_baoab"
    value = raw.get("batch_size", 1)
    try:
        size = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{source}.batch_size must be a positive integer.") from exc
    if isinstance(value, bool) or size < 1 or (isinstance(value, float) and value != size):
        raise ValueError(f"{source}.batch_size must be a positive integer and at least one.")
    policy = str(raw.get("partial_policy", "timeout" if portable else "full_only")).strip().lower()
    allowed = {"full_only", "timeout"} if portable else {"full_only"}
    if policy not in allowed:
        if not portable:
            raise ValueError("Only alchemi_baoab.partial_policy='full_only' is currently supported.")
        raise ValueError("batching.partial_policy must be 'full_only' or 'timeout'.")
    wait = raw.get("max_wait_seconds", 5.0)
    try:
        seconds = float(wait)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{source}.max_wait_seconds must be finite and nonnegative.") from exc
    if isinstance(wait, bool) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{source}.max_wait_seconds must be finite and nonnegative.")
    return {"batch_size": size, "partial_policy": policy,
            "max_wait_seconds": seconds, "portable": portable}


def dynamics_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Return backend options, requiring explicit selection for portable use."""

    legacy = {key: value for key, value in _block(config, "alchemi_baoab").items()
              if key not in _BATCH_KEYS}
    current = _block(config, "dynamics")
    result = _merge_options(legacy, current, "dynamics")
    if portable_batching(config) and "backend" not in current:
        raise ValueError("Portable sampling requires dynamics.backend='alchemi' or 'torchsim'.")
    backend = str(result.get("backend", "alchemi")).strip().lower()
    if backend not in {"alchemi", "torchsim"}:
        raise ValueError("dynamics.backend must be 'alchemi' or 'torchsim'.")
    result["backend"] = backend
    return result


def configured_native_loader(config: dict[str, Any]) -> str | None:
    """Resolve a native loader while rejecting contradictory legacy settings."""

    legacy = config.get("alchemi_calculator")
    current = config.get("batched_calculator")
    if legacy and current and legacy != current:
        raise ValueError("Conflicting alchemi_calculator and batched_calculator loaders.")
    _merge_options(_block(config, "alchemi_calculator_options"),
                   _block(config, "batched_calculator_options"), "batched_calculator_options")
    selected = current or legacy
    return str(selected) if selected else None


def configured_native_options(config: dict[str, Any]) -> dict[str, Any]:
    """Return compatible native-loader options for either configuration style."""

    configured_native_loader(config)
    return _merge_options(_block(config, "alchemi_calculator_options"),
                          _block(config, "batched_calculator_options"),
                          "batched_calculator_options")
