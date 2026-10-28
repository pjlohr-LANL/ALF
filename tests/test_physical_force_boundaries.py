"""Model-family independent physical inference and sampling composition."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from alframework.ml_interfaces.model_reference import ModelReference, legacy_training_reference
from alframework.ml_interfaces.sampling_inference import adapt_physical_evaluator, load_physical_evaluator
from alframework.samplers.force_controls import build_sampling_model
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator


class ArrayEvaluator:
    """A different model family: no ALF, Torch, or HIPPYNN superclass."""
    array_backend = "numpy"
    capabilities = {"supports_pbc": True, "supports_member_contributions": True}

    def predict(self, atoms_list):
        energies = np.asarray([.5 * (atoms.positions**2).sum() for atoms in atoms_list])
        positions = np.concatenate([atoms.positions for atoms in atoms_list])
        return {"energy_contributions": np.stack([.8 * energies, 1.2 * energies]),
                "force_contributions": np.stack([-.8 * positions, -1.2 * positions])}


class TensorArrayEvaluator:
    array_backend = "torch"
    capabilities = {"supports_member_contributions": True}

    def predict(self, batch):
        import torch
        energy = batch.positions.new_zeros(batch.num_graphs)
        energy.index_add_(0, batch.system_idx, .5 * batch.positions.square().sum(-1))
        return {"energy_contributions": torch.stack([.8 * energy, 1.2 * energy]),
                "force_contributions": torch.stack([-.8 * batch.positions, -1.2 * batch.positions])}


class GeometryBias:
    """Import-path toy control using only the packed physical state contract."""
    required_properties = ("energy_contributions", "force_contributions")

    def apply(self, state, physical, propagation, options):
        stiffness, factor = float(options["stiffness"]), float(options.get("factor", 1.))
        if hasattr(state.positions, "detach"):
            delta = state.positions.new_zeros(state.num_graphs)
            delta.index_add_(0, state.system_idx, .5 * stiffness * state.positions.square().sum(-1))
        else:
            delta = np.zeros(state.num_graphs)
            np.add.at(delta, state.system_idx, .5 * stiffness * (state.positions**2).sum(-1))
        return {"energy": factor * propagation["energy"] + delta.reshape(-1, 1),
                "forces": factor * propagation["forces"] - stiffness * state.positions,
                "diagnostics": {"geometry_bias_energy": delta}}


def load_array_model(model_reference, *, requested_properties, device=None):
    model = ArrayEvaluator()
    model.received_reference, model.received_request = model_reference, requested_properties
    return model


def load_legacy_array_model(directory, *, model_mode, selected_state_value, ML_config,
                            properties_list, sampler_config, device=None):
    assert sampler_config == {"model_mode": "ground_state"}
    assert directory == "/opaque/artifact.bundle"
    return ArrayEvaluator()


def systems():
    return [Atoms("H2", positions=[[.1, .2, .3], [1., .1, .2]]),
            Atoms("OH2", positions=[[.3, .2, .1], [1., .2, .1], [.3, 1., .2]])]


def config(engine="ase", udd=True, well=None):
    result = {"driver": {"strategy": "single" if engine == "ase" else "batched", "engine": engine},
              "sampling": {"propagator": "langevin", "biases": [
                  {"strategy": "udd", "weight": .45, "epsilon": 1.e-12}] if udd else []}}
    if well is not None:
        result["sampling"]["biases"].append({"strategy": "confinement", **well})
    return result


class ReferenceMember(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, strength):
        super().__init__()
        self.strength = strength

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": .5 * self.strength * (atoms.positions**2).sum(),
                        "forces": -self.strength * atoms.positions.copy()}


@pytest.mark.parametrize("well", [None, {"r_start": .2, "force": .03, "mass_weighted": False}])
def test_structural_evaluator_and_force_pipeline_match_existing_science(well):
    evaluator = ArrayEvaluator()
    atoms = systems()
    before = deepcopy(evaluator.predict(atoms))
    pipeline = build_sampling_model(evaluator, config(well=well), device="cpu")
    actual = pipeline.evaluate_atoms(atoms)
    offset = 0
    for index, system in enumerate(atoms):
        reference = MLMD_calculator([ReferenceMember(.8), ReferenceMember(1.2)],
            udd_bias_weight=.45, well_params=None if well is None else dict(well))
        reference.calculate(system, ["energy", "forces"])
        np.testing.assert_allclose(actual["energy"][index, 0], reference.results["energy"], atol=1.e-12)
        np.testing.assert_allclose(actual["forces"][offset:offset+len(system)], reference.results["forces"], atol=1.e-12)
        assert pipeline.diagnostics_for_graph(index)["Fmeanmax"] == pytest.approx(np.linalg.norm(system.positions, axis=-1).max())
        offset += len(system)
    for key, value in evaluator.predict(atoms).items():
        np.testing.assert_array_equal(value, before[key])
    assert not any(hasattr(evaluator, name) for name in ("udd_bias_weight", "well_params", "_lcm_controls"))


def test_new_loader_and_training_reference_have_no_checkpoint_family_assumptions():
    reference = legacy_training_reference(([True, True], 7), model_path="/opaque/model-{:04d}.bundle",
        inference={"interface": "native", "loader": __name__ + ".load_array_model"},
        capabilities={"supported_properties": ["energy", "forces"]})
    assert reference.artifact_location == "/opaque/model-0007.bundle"
    assert legacy_training_reference(([True, False], 8), model_path="/unused/{}", inference={}) is None
    loaded = load_physical_evaluator(reference, properties_list={"energy": ["energy", "system", 1.]}, device="cpu")
    assert loaded.received_reference is reference
    assert "sampler_config" not in loaded.received_request
    assert not {"udd_bias_weight", "Escut", "Fscut", "score"} & loaded.received_request.keys()
    assert loaded.predict(systems())["energy_contributions"].shape == (2, 2)
    assert ModelReference.from_value(reference.as_dict()) == reference


def test_legacy_loader_receives_only_prediction_configuration():
    reference = ModelReference("/opaque/artifact.bundle", 0, __name__ + ".load_legacy_array_model")
    evaluator = load_physical_evaluator(reference, device="cpu")
    assert evaluator.predict(systems())["force_contributions"].shape == (2, 5, 3)


def test_model_reference_preserves_adapter_owned_prediction_configuration():
    pinned = {"ML_config": {"species_key": "old_species"},
              "properties_list": {"old_energy": ["old_energy", "system", 1.]}}
    reference = ModelReference("/opaque/old.bundle", 1, __name__ + ".load_array_model", metadata=pinned)
    pinned["ML_config"]["species_key"] = "mutated"
    model = load_physical_evaluator(reference, ML_config={"species_key": "new_species"},
        properties_list={"new_energy": ["new_energy", "system", 1.]})
    assert model.received_request["model_config"]["species_key"] == "old_species"
    assert set(model.received_request["properties_list"]) == {"old_energy"}


def test_plain_tensor_evaluator_needs_no_scientific_controller_superclass():
    torch = pytest.importorskip("torch")
    from alframework.ml_interfaces.batched_inference import PackedBatch

    evaluator = TensorArrayEvaluator()
    native = build_sampling_model(evaluator, config("torchsim"), device="cpu")
    numpy = build_sampling_model(ArrayEvaluator(), config(), device="cpu")
    actual, expected = native(PackedBatch.from_atoms(systems())), numpy.evaluate_atoms(systems())
    np.testing.assert_allclose(actual["energy"].detach(), expected["energy"], rtol=1.e-5, atol=1.e-5)
    np.testing.assert_allclose(actual["forces"].detach(), expected["forces"], rtol=1.e-5, atol=1.e-5)
    assert adapt_physical_evaluator(native) is evaluator
    assert not hasattr(evaluator, "configure_sampling_method")


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_import_path_geometry_bias_runs_before_confinement_without_engine_changes(backend):
    well = {"r_start": .2, "force": .03, "mass_weighted": False}
    cfg = config("ase" if backend == "numpy" else "torchsim", udd=False, well=well)
    cfg["sampling"]["biases"].insert(0, {"strategy": __name__ + ".GeometryBias",
                                        "stiffness": .3, "factor": 2.})
    if backend == "numpy":
        evaluator = ArrayEvaluator()
    else:
        pytest.importorskip("torch")
        evaluator = TensorArrayEvaluator()
    model = build_sampling_model(evaluator, cfg, device="cpu")
    if backend == "numpy":
        actual = model.evaluate_atoms(systems())
    else:
        from alframework.ml_interfaces.batched_inference import PackedBatch
        actual = {key: value.detach().cpu().numpy() for key, value in model(PackedBatch.from_atoms(systems())).items()}
    offset = 0
    for index, atoms in enumerate(systems()):
        reference = MLMD_calculator([ReferenceMember(2.3), ReferenceMember(2.3)], well_params=dict(well))
        reference.calculate(atoms, ["energy", "forces"])
        np.testing.assert_allclose(actual["energy"][index, 0], reference.results["energy"], rtol=1.e-5, atol=1.e-5)
        np.testing.assert_allclose(actual["forces"][offset:offset+len(atoms)], reference.results["forces"], rtol=1.e-5, atol=1.e-5)
        diagnostic = model.diagnostics_for_graph(index)
        assert diagnostic["Fmeanmax"] == pytest.approx(np.linalg.norm(atoms.positions, axis=-1).max(), abs=1.e-6)
        assert diagnostic["geometry_bias_energy"] > 0
        offset += len(atoms)


def test_alchemi_compatibility_adapter_keeps_import_path_bias_controls():
    pytest.importorskip("nvalchemi")
    from ase import units
    from alframework.samplers.alchemi_sampling import AlchemiDynamicsRunner
    cfg = config("alchemi", udd=False)
    cfg["sampling"]["biases"] = [{"strategy": __name__ + ".GeometryBias", "stiffness": .3}]
    model = build_sampling_model(TensorArrayEvaluator(), cfg, device="cpu")
    atoms = systems()[0]
    atoms.set_velocities(np.full((len(atoms), 3), .01))
    runner = AlchemiDynamicsRunner(model=model, atoms_list=[atoms], dt_fs=.1,
        temperature_K=[0.], friction_per_fs=0., random_seed=3, device="cpu")
    runner.run(1)
    actual = atoms.copy()
    runner.sync_graph_to_atoms(0, actual)
    h = .1 * units.fs
    expected = atoms.positions + h * atoms.get_velocities() - .5 * h*h * 1.3 * atoms.positions / atoms.get_masses()[:, None]
    np.testing.assert_allclose(actual.positions, expected, atol=1.e-6)
    assert runner.diagnostics_for_graph(0)["geometry_bias_energy"] > 0


def test_numpy_force_pipeline_import_and_evaluation_stay_torch_free():
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','hippynn','torch_sim','nvalchemi','nvalchemiops','warp'}:
   raise ImportError('optional runtime blocked: '+fullname)
sys.meta_path.insert(0,Block())
from tests.test_physical_force_boundaries import ArrayEvaluator, config, systems
from alframework.samplers.force_controls import build_sampling_model
cfg=config()
cfg['sampling']['biases'].append({'strategy':'tests.test_physical_force_boundaries.GeometryBias','stiffness':.1})
model=build_sampling_model(ArrayEvaluator(),cfg,device='cpu')
assert model.evaluate_atoms(systems())['energy'].shape==(2,1)
assert 'torch' not in sys.modules
'''
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
