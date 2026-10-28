"""Sampling and acquisition extension contracts, independent of numerical engines.

Plugin import paths name zero-argument factories (or classes). A method creates
a program with a ``run()`` method; programs may advance MD, optimization, or
other algorithms and can request grouped model evaluations through the context.
No simulation clock or optional tensor stack is required by these contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Protocol

from alframework.tools.tools import load_module_from_string


@dataclass
class SamplingObservation:
    system_index: int
    molecule: Any
    atoms: Any
    diagnostics: dict[str, Any]
    step: int
    progress: dict[str, Any] = field(default_factory=dict)


@dataclass
class AcquisitionDecision:
    capture: bool
    stop: bool
    flags: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AcquisitionResult:
    candidate: dict[str, Any] | None
    stop: bool
    rejection: str | None = None


@dataclass
class SamplingContext:
    molecule_objects: list[Any]
    config: dict[str, Any]
    model: Any
    acquisition: Any
    device: Any = None
    properties_list: dict[str, Any] | None = None
    master_directory: str | None = None
    runner_factory: Callable[..., Any] | None = None
    capabilities: frozenset[str] = frozenset()
    state: dict[str, Any] = field(default_factory=dict)

    def evaluate_group(self, atoms_list):
        """Evaluate proposed configurations without requiring a dynamics step."""
        from alframework.ml_interfaces.sampling_inference import evaluate_model
        return evaluate_model(self.model, atoms_list, device=self.device)

    def acquire(self, observation: SamplingObservation) -> AcquisitionResult:
        """Apply common scientific screening and acquisition to a proposal."""
        from alframework.samplers.sampling import acquire_observation
        return acquire_observation(observation, self)

    def finalize(self, candidates):
        """Select proposals and restore normal ALF molecule/provenance outputs."""
        from alframework.samplers.sampling import finalize_candidates
        return finalize_candidates(self.acquisition.select(candidates, self, final=True), self)

    def run_md(self):
        from alframework.samplers.sampling import _run_md_sampling
        return _run_md_sampling(
            self.molecule_objects, self.config, self.model,
            properties_list=self.properties_list, device=self.device,
            runner_factory=self.runner_factory, master_directory=self.master_directory,
            method_context=self,
        )


class SamplingProgram(Protocol):
    def run(self) -> list[Any]: ...


class SamplingMethod(Protocol):
    required_capabilities: frozenset[str]

    def create_program(self, context: SamplingContext) -> SamplingProgram: ...


class AcquisitionPolicy(Protocol):
    def decide(self, observation: SamplingObservation, context: SamplingContext) -> AcquisitionDecision: ...

    def select(self, candidates: list[dict[str, Any]], context: SamplingContext,
               *, final: bool = False) -> list[dict[str, Any]]: ...


class _MDProgram:
    def __init__(self, context):
        self.context = context

    def run(self):
        return self.context.run_md()


class MLMDMethod:
    required_capabilities = frozenset({"dynamics"})

    def create_program(self, context):
        if hasattr(context.model, "configure_sampling_method"):
            context.model.configure_sampling_method("mlmd", {})
        return _MDProgram(context)


class UDDMethod:
    required_capabilities = frozenset({"dynamics", "member_contributions"})

    def create_program(self, context):
        config = context.config
        if dict(config.get("gap_seeking") or {}).get("enabled", False):
            raise ValueError("UDD and LCM switching cannot be combined in this milestone.")
        options = dict(config["sampling"].get("options") or {})
        for key, default in (("udd_bias_weight", None), ("udd_bias_eps", 1.e-12)):
            if key in options and key in config and options[key] != config[key]:
                raise ValueError(f"Conflicting sampling.options.{key} and {key}.")
            options.setdefault(key, config.get(key, default))
        if options["udd_bias_weight"] is None:
            raise ValueError("UDD sampling requires udd_bias_weight.")
        for key in ("udd_bias_weight", "udd_bias_eps"):
            value = float(options[key])
            if not math.isfinite(value) or (key == "udd_bias_eps" and value < 0):
                raise ValueError(f"{key} must be finite" + (" and nonnegative." if key.endswith("eps") else "."))
            options[key] = value
        configure = getattr(context.model, "configure_sampling_method", None)
        if configure is None:
            raise ValueError("The selected inference adapter cannot apply UDD member-force controls.")
        configure("udd", options)
        return _MDProgram(context)


class UncertaintyAcquisition:
    """Current portable ALF threshold, stop/continue, and ranking semantics."""

    def decide(self, observation, context):
        from alframework.samplers.sampling import uncertainty_flags, candidate_score
        from alframework.tools.sampling_validation import configured_score
        config = context.config
        diagnostics = observation.diagnostics
        flags = uncertainty_flags(diagnostics, Escut=config["Escut"], Fscut=config["Fscut"])
        score = configured_score(config, context.properties_list, context.state.get("selected_state"))
        metadata = candidate_score(diagnostics, flags, score,
                                   Escut=config["Escut"], Fscut=config["Fscut"])
        eligible = float(observation.progress.get("time_ps", 0.)) >= float(config.get("min_time", 0.))
        capture = bool(eligible and flags["uncertain"])
        return AcquisitionDecision(capture, capture and config["acquisition"]["policy"] == "stop",
                                   flags, metadata)

    def select(self, candidates, context, *, final=False):
        from alframework.samplers.sampling import _prune_candidates
        from alframework.tools.sampling_validation import (
            _configured_return_top_k, _configured_replica_candidate_limit,
        )
        if context.config["acquisition"]["policy"] == "stop":
            return sorted(candidates, key=lambda row: int(row["record"]["batch_index"])) if final else candidates
        return _prune_candidates(candidates, _configured_return_top_k(context.config),
                                 _configured_replica_candidate_limit(context.config))


def _plugin(name, builtins, required):
    if name in builtins:
        plugin = builtins[name]()
    else:
        factory = load_module_from_string(name)
        plugin = factory() if callable(factory) else factory
    for member in required:
        if not callable(getattr(plugin, member, None)):
            raise TypeError(f"Sampling plugin {name!r} must implement {member}().")
    return plugin


def sampling_method(name):
    return _plugin(name, {"mlmd": MLMDMethod, "udd": UDDMethod}, ("create_program",))


def acquisition_policy(name):
    return _plugin(name, {"uncertainty": UncertaintyAcquisition}, ("decide", "select"))
