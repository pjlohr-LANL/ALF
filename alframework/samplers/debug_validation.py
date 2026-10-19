"""Sampler-owned assertions for the driver's optional --test_sampler path.

No optional GPU or excited-state dependencies are imported by this helper.
The core driver only supplies the selected output and sampler configuration.
"""


def validate_sampler_debug_output(molecule, sampler_config):
    metadata = molecule.get_metadata()
    backend = dict(sampler_config.get("alchemi_baoab", {}))
    if backend.get("strict_gpu", False) and not backend.get("allow_cpu_debug", False):
        assert str(metadata.get("sampler_device", "")).startswith("cuda:"), (
            "Strict-GPU sampler testing must return CUDA device metadata.")
    if sampler_config.get("model_mode") == "excited_state":
        selection = dict(sampler_config.get("state_selection", {}))
        states = ({selection.get("state")} if selection.get("mode") == "fixed"
                  else set(selection.get("states", [])))
        assert metadata.get("selected_state") in states, (
            "Excited-state sampler testing must return configured selected_state metadata.")
    if sampler_config.get("topology_check", {}).get("enabled", False):
        assert metadata.get("topology_valid") is True, (
            "Topology-enabled sampler testing must return topology_valid=true.")
