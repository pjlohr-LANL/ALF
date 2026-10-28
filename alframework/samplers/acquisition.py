"""Callable acquisition strategies, independent of propagation and inference."""
from __future__ import annotations

from alframework.tools.batched_config import canonical_sampling


class UncertaintyAcquisition:
    required_signals = frozenset({"Es", "Fs", "Fsmax"})
    retention = "scalar_top_k"
    ranking_mode = "max"

    def decide(self, observation, context):
        from alframework.samplers.sampling import uncertainty_flags, candidate_score
        from alframework.samplers.sampling_methods import AcquisitionDecision
        from alframework.tools.sampling_validation import configured_score
        config = context.config
        missing = self.required_signals - observation.diagnostics.keys()
        if missing:
            raise ValueError(f"Acquisition requires missing physical signals: {sorted(missing)}.")
        flags = uncertainty_flags(observation.diagnostics, Escut=config["Escut"], Fscut=config["Fscut"])
        scored = config
        if canonical_sampling(config):
            scored = {**config, "score": {**dict(config.get("score") or {}), "mode": self.ranking_mode}}
        score = configured_score(scored, context.properties_list, context.state.get("selected_state"))
        metadata = candidate_score(observation.diagnostics, flags, score,
                                   Escut=config["Escut"], Fscut=config["Fscut"])
        eligible = float(observation.progress.get("time_ps", 0.)) >= float(config.get("min_time", 0.))
        capture = bool(eligible and flags["uncertain"])
        return AcquisitionDecision(capture, capture and config["acquisition"]["policy"] == "stop", flags, metadata)

    def select(self, candidates, context, *, final=False):
        from alframework.samplers.sampling import _candidate_sort_key, _prune_candidates
        from alframework.tools.sampling_validation import _configured_return_top_k, _configured_replica_candidate_limit
        config = context.config
        if canonical_sampling(config):
            selection = config["acquisition"]["selection"]
            if selection["mode"] == "cohort_top_k":
                # Each worker may discard only frames that cannot enter the
                # global scalar top-K. This is not a per-trajectory quota.
                return sorted(candidates, key=_candidate_sort_key)[:selection["k"]]
            return sorted(candidates, key=_candidate_sort_key) if final else candidates
        if config["acquisition"]["policy"] == "stop":
            return sorted(candidates, key=lambda row: int(row["record"]["batch_index"])) if final else candidates
        return _prune_candidates(candidates, _configured_return_top_k(config),
                                 _configured_replica_candidate_limit(config))


class SummedAcquisition(UncertaintyAcquisition):
    """Use the ordinary uncertainty trigger and the existing weighted ranking."""
    required_signals = UncertaintyAcquisition.required_signals | {"Fsrms"}
    ranking_mode = "sum"
