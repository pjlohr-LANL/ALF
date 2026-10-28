"""Independent method/driver settings and legacy-safe task routing."""

from copy import deepcopy
import pickle

from ase import Atoms
import pytest

from alframework.tools.batched_config import (
    acquisition_settings, batching_settings, configured_native_loader,
    configured_native_options, dynamics_settings, inference_settings,
    method_settings, normalize_sampling_config, unified_sampling,
)
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampler_batching import (
    SamplerBatchBuffer, changed_batching_signature_fields,
    configured_sampler_calculator_status, sampler_batch_size,
    sampler_ready_groups, sampler_submission_groups, sampler_task_feed,
    sampler_uses_batches, validate_batching_config_reload,
)


def molecule(name="m0", formula="H2"):
    return MoleculesObject(Atoms(formula), name)


def batched_config():
    return {"sampling": {"method": "udd"},
            "udd_bias_weight": .45,
            "acquisition": {"method": "uncertainty", "policy": "continue"},
            "dynamics": {"driver": "batched", "backend": "torchsim"},
            "batching": {"batch_size": 4, "max_wait_seconds": 5.},
            "inference": {"interface": "native", "loader": "example.load",
                          "options": {"key": [1, 2]}}}


def test_unified_defaults_and_ase_driver_ignore_inactive_batching():
    raw = {"dt": .1, "sampling": {},
           "dynamics": {"backend": "torchsim"},
           "batching": {"batch_size": -4, "partial_policy": "invalid"}}
    normalized = normalize_sampling_config(raw)
    assert normalized["sampling"]["method"] == "mlmd"
    assert normalized["acquisition"]["policy"] == "stop"
    assert normalized["dynamics"]["driver"] == "ase"
    assert normalized["dynamics"]["backend"] == "ase"
    assert normalized["dt"] == .1
    assert not sampler_uses_batches(normalized)
    assert sampler_batch_size(normalized) == 1
    assert batching_settings(normalized)["batch_size"] == 1
    assert normalize_sampling_config({})["dynamics"]["driver"] == "ase"
    assert raw["dynamics"]["backend"] == "torchsim"


def test_changing_only_driver_selects_batched_execution():
    config = batched_config()
    assert sampler_uses_batches(config)
    assert sampler_batch_size(config) == 4
    config["dynamics"]["driver"] = "ase"
    assert not sampler_uses_batches(config)
    assert sampler_batch_size(config) == 1
    config["dynamics"]["driver"] = "batched"
    assert dynamics_settings(config)["backend"] == "torchsim"
    assert sampler_batch_size(config) == 4


def test_unified_method_and_inference_options_are_snapshot_copies():
    config = batched_config()
    config["sampling"]["options"] = {"custom": [3]}
    feed = sampler_task_feed([molecule()], config)
    config["sampling"]["options"]["custom"].append(4)
    config["inference"]["options"]["key"].append(3)
    assert feed["sampler_config"]["sampling"]["options"] == {"custom": [3]}
    assert feed["sampler_config"]["inference"]["options"] == {"key": [1, 2]}
    assert feed["sampler_config"]["uncertainty_policy"] == "continue"


def test_ase_unified_uses_one_element_list_without_buffering():
    config = batched_config()
    config["dynamics"]["driver"] = "ase"
    first = molecule()
    buffer = SamplerBatchBuffer()
    assert sampler_submission_groups(first, config, buffer) == [[first]]
    assert len(buffer) == 0
    assert sampler_task_feed([first], config)["molecule_objects"] == [first]
    with pytest.raises(ValueError, match="exactly one"):
        sampler_task_feed([first, molecule("m1")], config)


def test_new_batched_partial_feed_and_buffer_pickle_preserve_actual_width():
    config = batched_config()
    config["batching"]["max_wait_seconds"] = 0.
    buffer = SamplerBatchBuffer()
    first = molecule()
    buffer.add(first, config, buffered_at=0.)
    restored = pickle.loads(pickle.dumps(buffer))
    groups = sampler_ready_groups(restored, config, now=1.)
    assert len(groups) == 1 and len(groups[0]) == 1
    assert sampler_task_feed(groups[0], config)["molecule_objects"][0].get_moleculeid() == "m0"
    assert len(restored) == 0


@pytest.mark.parametrize("alias,interface", [
    ("batched_calculator", "native"), ("alchemi_calculator", "native"),
    ("ase_calculator", "ase"),
])
def test_inference_aliases_merge_matching_settings(alias, interface):
    config = {alias: "example.load", alias + "_options": {"a": 1},
              "inference": {"interface": interface, "loader": "example.load",
                            "options": {"b": 2}}}
    assert inference_settings(config) == {"interface": interface, "loader": "example.load",
                                          "options": {"a": 1, "b": 2}}
    if interface == "native":
        assert configured_native_loader(config) == "example.load"
        assert configured_native_options(config) == {"a": 1, "b": 2}
    else:
        assert configured_native_loader(config) is None


@pytest.mark.parametrize("extra", [
    {"batched_calculator": "other.load"},
    {"batched_calculator_options": {"key": 9}},
    {"ase_calculator": "other.load"},
])
def test_contradictory_inference_aliases_fail(extra):
    config = batched_config()
    config.update(extra)
    with pytest.raises(ValueError, match="Conflicting"):
        inference_settings(config)


def test_policy_alias_agrees_or_fails_without_silently_overriding():
    config = {"uncertainty_policy": "continue", "acquisition": {"policy": "continue"}}
    assert acquisition_settings(config)["policy"] == "continue"
    config["acquisition"]["policy"] = "stop"
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_sampling_config(config)


def test_custom_method_and_acquisition_paths_do_not_import_plugins():
    config = {"sampling": {"method": "missing.package.CustomMethod"},
              "acquisition": {"method": "missing.package.CustomAcquisition"}}
    assert method_settings(config)["method"] == "missing.package.CustomMethod"
    assert acquisition_settings(config)["method"] == "missing.package.CustomAcquisition"
    assert normalize_sampling_config(config)["dynamics"]["driver"] == "ase"


@pytest.mark.parametrize("change", [
    {"sampling": {"method": "mlmd"}},
    {"sampling": {"method": "udd", "options": {"extra": 1}}},
    {"acquisition": {"policy": "stop"}},
    {"dynamics": {"driver": "ase"}},
    {"inference": {"interface": "native", "loader": "other.load"}},
    {"inference": {"interface": "native", "loader": "example.load", "options": {"key": 2}}},
    {"udd_bias_weight": .6},
])
def test_reload_tracks_resolved_scientific_and_execution_identity(change):
    config = batched_config()
    proposed = deepcopy(config)
    proposed.update(change)
    master = {"sampler_task": "alframework.samplers.sampling.sampling_task"}
    buffer = SamplerBatchBuffer()
    buffer.add(molecule(), config)
    assert changed_batching_signature_fields(master, config, master, proposed)
    with pytest.raises(ValueError, match="buffered"):
        validate_batching_config_reload(master, config, master, proposed, buffer)
    assert validate_batching_config_reload(master, config, master, proposed, SamplerBatchBuffer())


def test_status_reports_independent_selections_for_ase_execution():
    config = batched_config()
    config["dynamics"]["driver"] = "ase"
    status = configured_sampler_calculator_status(config)
    assert status["driver"] == "ase" and status["backend"] == "ase"
    assert status["sampling_method"] == "udd"
    assert status["acquisition_policy"] == "continue"
    assert status["interface"] == "native"


def test_udd_lcm_and_missing_batched_engine_fail_before_worker_launch():
    config = batched_config()
    config["gap_seeking"] = {"enabled": True}
    with pytest.raises(ValueError, match="UDD and LCM"):
        normalize_sampling_config(config)
    with pytest.raises(ValueError, match="backend"):
        normalize_sampling_config({"sampling": {}, "dynamics": {"driver": "batched"}})
    assert not unified_sampling({"dynamics": {"backend": "torchsim"}})
