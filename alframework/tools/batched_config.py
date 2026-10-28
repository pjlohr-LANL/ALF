"""Dependency-free normalization of portable and legacy sampler settings."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


_BATCH_KEYS = frozenset({"batch_size", "partial_policy", "max_wait_seconds"})


def unified_sampling(config: dict[str, Any]) -> bool:
    """Return whether a configuration selects the independent sampler API."""

    return any(name in config for name in ("sampling", "acquisition", "inference")) or (
        isinstance(config.get("dynamics"), dict) and "driver" in config["dynamics"]
    )


def portable_batching(config: dict[str, Any]) -> bool:
    """Return whether portable geometry/policy rules apply.

    This is deliberately distinct from execution cardinality. Unified ASE
    sampling uses these rules but still submits one system at a time.
    """

    return unified_sampling(config) or "batching" in config or "dynamics" in config


def _block(config, name):
    value = config.get(name, {})
    if value is None and name.endswith("_options"):
        return {}
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
    if unified_sampling(config) and dynamics_settings(config)["driver"] == "ase":
        return {"batch_size": 1, "partial_policy": "full_only",
                "max_wait_seconds": 0.0, "portable": True}
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

    current = _block(config, "dynamics")
    if unified_sampling(config):
        driver = str(current.get("driver", "ase")).strip().lower()
        if driver not in {"ase", "batched"}:
            raise ValueError("dynamics.driver must be 'ase' or 'batched'.")
        if driver == "ase":
            # Retain inactive batching/backend fields in the caller's config
            # so changing just the driver remains sufficient.
            return {**current, "driver": "ase", "backend": "ase"}
    else:
        driver = "batched"
    legacy = {key: value for key, value in _block(config, "alchemi_baoab").items()
              if key not in _BATCH_KEYS}
    result = _merge_options(legacy, current, "dynamics")
    if portable_batching(config) and "backend" not in current:
        raise ValueError("Portable sampling requires dynamics.backend='alchemi' or 'torchsim'.")
    backend = str(result.get("backend", "alchemi")).strip().lower()
    if backend not in {"alchemi", "torchsim"}:
        raise ValueError("dynamics.backend must be 'alchemi' or 'torchsim'.")
    result["backend"] = backend
    if unified_sampling(config):
        result["driver"] = driver
    return result


def configured_native_loader(config: dict[str, Any]) -> str | None:
    """Resolve a native loader while rejecting contradictory legacy settings."""

    if "inference" in config:
        settings = inference_settings(config)
        return settings["loader"] if settings["interface"] == "native" else None
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

    if "inference" in config:
        settings = inference_settings(config)
        return settings["options"] if settings["interface"] == "native" else {}
    configured_native_loader(config)
    return _merge_options(_block(config, "alchemi_calculator_options"),
                          _block(config, "batched_calculator_options"),
                          "batched_calculator_options")


def _named_settings(config, name, default):
    settings = _block(config, name)
    method = settings.get("method", default)
    if not isinstance(method, str) or not method.strip():
        raise ValueError(f"{name}.method must be a nonempty built-in name or import path.")
    method = method.strip()
    if "." not in method:
        method = method.lower()
    options = settings.get("options", {})
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError(f"{name}.options must be a configuration dictionary.")
    return {**settings, "method": method, "options": deepcopy(options)}


def method_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve a method without importing its implementation."""

    return _named_settings(config, "sampling", "mlmd")


def acquisition_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve acquisition independently from sampling and execution."""

    settings = _named_settings(config, "acquisition", "uncertainty")
    legacy = config.get("uncertainty_policy")
    current = settings.get("policy")
    if legacy is not None and current is not None and str(legacy).strip().lower() != str(current).strip().lower():
        raise ValueError("Conflicting uncertainty_policy and acquisition.policy settings.")
    policy = str(current if current is not None else legacy if legacy is not None else "stop").strip().lower()
    if settings["method"] == "uncertainty" and policy not in {"stop", "continue"}:
        raise ValueError("acquisition.policy must be 'stop' or 'continue'.")
    settings["policy"] = policy
    return settings


def inference_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve model inference independently of the execution driver.

    A missing loader is reported as ``None`` so config/status inspection and
    custom methods need not construct or import a model. Sampling validates
    the required evaluator before starting its program.
    """

    legacy_native = config.get("alchemi_calculator")
    current_native = config.get("batched_calculator")
    if legacy_native and current_native and legacy_native != current_native:
        raise ValueError("Conflicting alchemi_calculator and batched_calculator loaders.")
    native = current_native or legacy_native
    native_options = _merge_options(_block(config, "alchemi_calculator_options"),
                                    _block(config, "batched_calculator_options"),
                                    "batched_calculator_options")
    ase = config.get("ase_calculator")
    ase_options = _block(config, "ase_calculator_options")
    current = _block(config, "inference")
    interface = str(current.get("interface", "native" if native else "ase")).strip().lower()
    if interface not in {"native", "ase"}:
        raise ValueError("inference.interface must be 'native' or 'ase'.")
    if "inference" in config and ((interface == "ase" and native) or (interface == "native" and ase and not native)):
        raise ValueError("Conflicting inference.interface and legacy calculator settings.")
    alias = native if interface == "native" else ase
    loader = current.get("loader", alias)
    if alias and loader and alias != loader:
        raise ValueError("Conflicting inference.loader and legacy calculator loader.")
    if loader is not None and (not isinstance(loader, str) or not loader.strip()):
        raise ValueError("inference.loader must be a nonempty import path.")
    options = current.get("options", {})
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("inference.options must be a configuration dictionary.")
    merged = _merge_options(native_options if interface == "native" else ase_options,
                            options, "inference.options")
    return {"interface": interface, "loader": None if loader is None else loader.strip(),
            "options": deepcopy(merged)}


def normalize_sampling_config(config: dict[str, Any]) -> dict[str, Any]:
    """Snapshot a configuration for the unified task without rewriting files."""

    result = deepcopy(config)
    result["sampling"] = method_settings(config)
    result["acquisition"] = acquisition_settings(config)
    result["uncertainty_policy"] = result["acquisition"]["policy"]
    result["dynamics"] = dynamics_settings(result)
    result["inference"] = inference_settings(config)
    batching_settings(result)
    if (result["sampling"]["method"] == "udd" and
            bool(_block(result, "gap_seeking").get("enabled", False))):
        raise ValueError("UDD and LCM gap seeking cannot be combined in this milestone.")
    return result
