"""The same ALF sampling policies exercised against both actual MD engines."""
import importlib.util
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from alframework.ml_interfaces.batched_inference import PackedBatch
from alframework.samplers.batched_sampling import run_batched_sampling
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampling_validation import configured_gap_seeking
from tests.test_batched_inference import HarmonicStates, _hippynn_model, _systems


@pytest.fixture(params=["alchemi", "torchsim"])
def backend(request):
    if request.param == "alchemi":
        pytest.importorskip("nvalchemi")
        from alframework.samplers.alchemi_sampling import AlchemiDynamicsRunner
        return request.param, AlchemiDynamicsRunner
    if importlib.util.find_spec("torch_sim") is None:
        pytest.skip("Install the separate TorchSim environment for this conformance case.")
    from alframework.samplers.torchsim_sampling import TorchSimDynamicsRunner
    return request.param, TorchSimDynamicsRunner


def _excited_case(backend_name, *, cutoff=None, audit=None):
    config = {
        "model_mode": "excited_state", "state_selection": {"mode": "fixed", "state": 0},
        "batching": {"batch_size": 2, "partial_policy": "full_only"},
        "dynamics": {"backend": backend_name, "allow_cpu_debug": True, "random_seed": 7},
        "uncertainty_policy": "continue", "return_top_k": 10,
        "dt": .1, "maxt": .0003, "Ncheck": 1,
        "Escut": .001, "Fscut": .001, "distcut": .2, "friction_per_fs": 0.,
        "srt_temp": [0., 0.], "end_temp": [0., 0.], "amp_temp": [0., 0.], "per_temp": [1., 1.],
        "gap_seeking": {"enabled": True, "mode": "levine_coe_martinez_switch",
                        "switch_policy": "stay_fixed", "candidate_pairs": "adjacent",
                        "trigger_gap_threshold_eV": 1., "sigma": 3.5, "alpha_eV": .05},
    }
    if cutoff is not None:
        config["max_force_cutoff"] = cutoff
    if audit is not None:
        config["force_screen_audit_dir"] = str(audit)
    properties = {"sE0": ["sE0", "system", 1.], "F0": ["F0", "atomic", 1.],
                  "sE1": ["sE1", "system", 1.], "F1": ["F1", "atomic", 1.],
                  "dE01": ["dE01", "system", 1.]}
    seeking = configured_gap_seeking(config, properties, 0)
    model = HarmonicStates(physical_force_states=[0, 1], gap_rows=seeking["rows"],
                           gap_seeking_settings=seeking)
    systems = _systems()
    for atoms in systems:
        atoms.set_velocities(np.zeros((len(atoms), 3)))
    molecules = [MoleculesObject(atoms, name) for atoms, name in zip(systems, ("small", "large"))]
    return config, properties, model, molecules


def test_actual_engine_ragged_excited_sampling_switches_only_the_near_gap_replica(backend):
    name, runner = backend
    config, properties, model, molecules = _excited_case(name)
    output = run_batched_sampling(molecules, config, model, properties_list=properties,
                                  device="cpu", runner_factory=runner)
    rows = {parent: sorted([m.get_metadata() for m in output if m.get_metadata()["parent_molecule_id"] == parent],
                           key=lambda row: row["step"])
            for parent in ("small", "large")}
    assert len(rows["small"]) == len(rows["large"]) == 3
    assert rows["small"][0]["gap_seeking_current_mode"] == "direct"
    assert [row["gap_seeking_current_mode"] for row in rows["small"]][1:] == ["lcm", "lcm"]
    assert all(row["gap_seeking_current_mode"] == "direct" for row in rows["large"])
    assert len(rows["small"][1]["gap_seeking_switch_events"]) == 1
    assert rows["small"][1]["lcm_bias_Fmax"] > 0.
    for molecule in output:
        metadata = molecule.get_metadata()
        assert metadata["selected_state"] == 0
        assert metadata["dynamics_backend"] == f"{name}_baoab"
        assert metadata["gap_model_counts"]["dE01"] == 2
        assert len(molecule.get_atoms()) == (2 if metadata["parent_molecule_id"] == "small" else 3)


def test_actual_engine_uses_physical_all_state_force_screen_after_lcm(backend, tmp_path):
    name, runner = backend
    config, properties, model, molecules = _excited_case(name, cutoff=2.3, audit=tmp_path)
    output = run_batched_sampling(molecules, config, model, properties_list=properties,
                                  device="cpu", runner_factory=runner)
    assert output
    assert {m.get_metadata()["parent_molecule_id"] for m in output} == {"small"}
    lcm = [m.get_metadata() for m in output if m.get_metadata()["gap_seeking_current_mode"] == "lcm"]
    assert lcm and all(row["Fmeanmax"] <= 2.3 < row["propagation_Fmax"] for row in lcm)
    data = json.loads(next(tmp_path.glob("force-screen-*.json")).read_text())
    rejection = data["replicas"][1]["rejected"]
    assert rejection["reason"] == "physical_force_cutoff"
    assert len(rejection["physical_forces"]["F1"]) == 3
    assert rejection["cell"] == [[0., 0., 0.]] * 3
    assert rejection["pbc"] == [False] * 3


def test_actual_engine_periodic_hippynn_force_cache_matches_direct_inference_after_cell_update(backend):
    _, runner_type = backend
    model = _hippynn_model(periodic=True)
    initial = _systems(periodic=True)
    runner = runner_type(model=model, atoms_list=initial, dt_fs=.1,
                         temperature_K=[0., 0.], friction_per_fs=0., random_seed=3, device="cpu")
    runner.run(2)
    updated = [atoms.copy() for atoms in initial]
    for index, atoms in enumerate(updated):
        runner.sync_graph_to_atoms(index, atoms)
    runner.set_cell(1, updated[1].cell.array * 1.17)
    runner.sync_graph_to_atoms(1, updated[1])
    direct = model(PackedBatch.from_atoms(updated))
    for index, atoms in enumerate(updated):
        diagnostics = runner.diagnostics_for_graph(index)
        direct_diagnostics = model.diagnostics_for_graph(index)
        np.testing.assert_allclose(diagnostics["F0"], direct_diagnostics["F0"], rtol=1.e-5, atol=1.e-6)
        assert diagnostics["sE0"] == pytest.approx(direct["energy"][index].item(), rel=1.e-5, abs=1.e-6)
        assert atoms.pbc.tolist() == [True] * 3
