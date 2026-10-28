"""Scientific conformance of inference and UDD across array backends."""

import subprocess
import sys
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from alframework.ml_interfaces.sampling_inference import NumpyASEEnsembleModel, evaluate_model
from alframework.samplers.ASE_ensemble_constructor import MLMD_calculator


class Quadratic(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, stiffness=1., shift=0.):
        super().__init__()
        self.stiffness, self.shift = stiffness, shift

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": .5 * self.stiffness * np.sum(atoms.positions**2) + self.shift,
                        "forces": -self.stiffness * atoms.positions.copy()}


def structures():
    return [Atoms("H2", positions=[[.1, .2, .3], [.7, .2, .1]]),
            Atoms("OH2", positions=[[.3, .1, .4], [.8, .3, .2], [.2, .6, .1]])]


@pytest.mark.parametrize("well", [None, {"r_start": .2, "force": .03, "mass_weighted": False}])
@pytest.mark.parametrize("weight", [-.45, 0., .45])
def test_numpy_udd_matches_existing_ase_calculator_per_system(well, weight):
    model = NumpyASEEnsembleModel([Quadratic(1., .2), Quadratic(1.4, -.1)], well_params=well)
    model.configure_sampling_method("udd", {"udd_bias_weight": weight})
    atoms = structures()
    result = model.evaluate_atoms(atoms)
    offset = 0
    for index, system in enumerate(atoms):
        reference = MLMD_calculator([Quadratic(1., .2), Quadratic(1.4, -.1)],
                                   well_params=None if well is None else dict(well), udd_bias_weight=weight)
        reference.calculate(system, ["energy", "forces"])
        np.testing.assert_allclose(result["energy"][index, 0], reference.results["energy"], atol=1e-12)
        np.testing.assert_allclose(result["forces"][offset:offset+len(system)], reference.results["forces"], atol=1e-12)
        diagnostic = model.diagnostics_for_graph(index)
        np.testing.assert_allclose(diagnostic["E_en_bias"], reference.results["E_en_bias"])
        np.testing.assert_allclose(diagnostic["F_en_bias"], reference.results["F_en_bias"])
        physical = -.5 * (1. + 1.4) * system.positions
        assert diagnostic["Fmeanmax"] == pytest.approx(np.linalg.norm(physical, axis=1).max())
        offset += len(system)


def test_torch_udd_matches_numpy_for_mixed_sizes_and_reset():
    torch = pytest.importorskip("torch")
    from alframework.ml_interfaces.batched_inference import ASEEnsembleModel, PackedBatch
    atoms = structures()
    tensor = ASEEnsembleModel([Quadratic(1., .2), Quadratic(1.4, -.1)],
                             model_mode="ground_state", selected_state_value=0)
    numpy = NumpyASEEnsembleModel([Quadratic(1., .2), Quadratic(1.4, -.1)])
    for model in (tensor, numpy):
        model.configure_sampling_method("udd", {"udd_bias_weight": .45})
    expected, actual = numpy.evaluate_atoms(atoms), tensor(PackedBatch.from_atoms(atoms))
    np.testing.assert_allclose(actual["energy"].detach(), expected["energy"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual["forces"].detach(), expected["forces"], rtol=1e-5, atol=1e-5)
    for i in range(len(atoms)):
        for key in ("Es", "Fs", "Fsmax", "Fmeanmax", "udd_bias_Fmax"):
            assert tensor.diagnostics_for_graph(i)[key] == pytest.approx(numpy.diagnostics_for_graph(i)[key], abs=1e-6)
    tensor.configure_sampling_method("mlmd", {})
    assert tensor.udd_bias_weight is None
    assert "energy_contributions" in evaluate_model(tensor, atoms)


@pytest.mark.parametrize("epsilon", [0., 1.e-12, 10.])
def test_zero_disagreement_and_epsilon_keep_force_bias_finite(epsilon):
    model = NumpyASEEnsembleModel([Quadratic(), Quadratic()])
    model.configure_sampling_method("udd", {"udd_bias_weight": .5, "udd_bias_eps": epsilon})
    result = model.evaluate_atoms(structures())
    for row in result["diagnostics"]:
        np.testing.assert_array_equal(row["F_en_bias"], 0.)
        assert row["E_en_bias"] == 0


def test_aggregate_uncertainty_rejected_for_udd_but_ensemble_members_supported():
    class Aggregated(Quadratic):
        implemented_properties = Quadratic.implemented_properties + ["energy_stdev", "forces_stdev_mean", "forces_stdev_max"]

        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            self.results.update(energy_stdev=.3, forces_stdev_mean=.2, forces_stdev_max=.4)

    model = NumpyASEEnsembleModel(Aggregated())
    assert model.evaluate_atoms(structures())["diagnostics"][0]["Es"] == .3
    model.configure_sampling_method("udd", {"udd_bias_weight": .5})
    with pytest.raises(ValueError, match="aggregate uncertainty"):
        model.evaluate_atoms(structures())
    reference = MLMD_calculator([Quadratic(), Quadratic(1.2)])
    model = NumpyASEEnsembleModel(reference)
    model.configure_sampling_method("udd", {"udd_bias_weight": .5})
    assert model.evaluate_atoms(structures())["energy_contributions"].shape == (2, 2)


def test_numpy_import_and_ase_execution_need_no_torch():
    script = '''
import builtins
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torch_sim', 'nvalchemi', 'nvalchemiops', 'warp'}:
        raise AssertionError('Unexpected optional dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from ase import Atoms
from ase.calculators.emt import EMT
from alframework.ml_interfaces.sampling_inference import NumpyASEEnsembleModel
from alframework.samplers.ase_sampling import ASEDynamicsRunner
model = NumpyASEEnsembleModel([EMT(), EMT()])
model.configure_sampling_method('udd', {'udd_bias_weight': .5})
runner = ASEDynamicsRunner(model=model, atoms_list=[Atoms('Cu2',positions=[[0,0,0],[2.5,0,0]])],
    dt_fs=.1, temperature_K=[300.], friction_per_fs=.01, random_seed=4)
runner.run(2)
assert runner.nsteps == 2
'''
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_method_bias_rejects_nonfinite_settings_and_lcm_combination():
    model = NumpyASEEnsembleModel([Quadratic(), Quadratic(1.2)], gap_seeking_settings={"enabled": True})
    with pytest.raises(ValueError, match="combined with LCM"):
        model.configure_sampling_method("udd", {"udd_bias_weight": .5})
    with pytest.raises(ValueError, match="finite"):
        model.configure_sampling_method("udd", {"udd_bias_weight": float("nan")})


def test_wrapped_ase_ensemble_excludes_its_well_from_member_statistics():
    well = {"r_start": .2, "force": .03, "mass_weighted": False}
    wrapped = MLMD_calculator([Quadratic(1., .2), Quadratic(1.4, -.1)], well_params=dict(well))
    model = NumpyASEEnsembleModel(wrapped)
    model.configure_sampling_method("udd", {"udd_bias_weight": .45})
    atoms = structures()
    actual = model.evaluate_atoms(atoms)
    expected_model = NumpyASEEnsembleModel([Quadratic(1., .2), Quadratic(1.4, -.1)], well_params=well)
    expected_model.configure_sampling_method("udd", {"udd_bias_weight": .45})
    expected = expected_model.evaluate_atoms(atoms)
    assert actual["energy_contributions"].shape == (2, 2)
    np.testing.assert_allclose(actual["energy"], expected["energy"])
    np.testing.assert_allclose(actual["forces"], expected["forces"])
    assert actual["diagnostics"][0]["Es"] == pytest.approx(expected["diagnostics"][0]["Es"])
    pytest.importorskip("torch")
    from alframework.ml_interfaces.batched_inference import ASEEnsembleModel, PackedBatch
    tensor = ASEEnsembleModel(MLMD_calculator([Quadratic(1., .2), Quadratic(1.4, -.1)], well_params=dict(well)),
                             model_mode="ground_state", selected_state_value=0)
    tensor.configure_sampling_method("udd", {"udd_bias_weight": .45})
    tensor_result = tensor(PackedBatch.from_atoms(atoms))
    np.testing.assert_allclose(tensor_result["energy"].detach(), expected["energy"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(tensor_result["forces"].detach(), expected["forces"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("adapter", ["numpy", "torch"])
def test_new_program_resets_replica_local_lcm_controls(adapter):
    kwargs = dict(model_mode="ground_state", selected_state_value=0,
                  gap_seeking_settings={"enabled": True, "rows": [{"lower_state": 0, "upper_state": 1}]})
    if adapter == "numpy":
        model = NumpyASEEnsembleModel([Quadratic()], **kwargs)
    else:
        pytest.importorskip("torch")
        from alframework.ml_interfaces.batched_inference import ASEEnsembleModel
        model = ASEEnsembleModel([Quadratic()], **kwargs)
    model.set_lcm_mode(0, pair=[0, 1], sigma=3.5, alpha_eV=.05)
    assert model._lcm_controls
    model.configure_sampling_method("mlmd", {})
    assert model._lcm_controls == {}


@pytest.mark.parametrize("method", ["udd", "lcm"])
def test_selected_excited_state_controls_match_numpy_and_tensor(method):
    pytest.importorskip("torch")
    from alframework.ml_interfaces.batched_inference import ASEEnsembleModel, PackedBatch

    class Surfaces(Quadratic):
        implemented_properties = ["sE0", "sE1", "F0", "F1"]

        def calculate(self, atoms=None, properties=("sE0",), system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            value = .5 * self.stiffness * np.sum(atoms.positions**2)
            self.results = {"sE0": value + self.shift, "sE1": 1.3 * value + .02,
                            "F0": -self.stiffness * atoms.positions.copy(),
                            "F1": -1.3 * self.stiffness * atoms.positions.copy()}

    gap = {"lower_state": 0, "upper_state": 1, "gap_key": "dE01"}
    kwargs = dict(model_mode="excited_state", selected_state_value=1,
                  physical_force_states=[0, 1], gap_rows=[gap],
                  gap_seeking_settings={"enabled": method == "lcm", "rows": [gap] if method == "lcm" else []})
    numpy = NumpyASEEnsembleModel([Surfaces(1., .1), Surfaces(1.4, .2)], **kwargs)
    tensor = ASEEnsembleModel([Surfaces(1., .1), Surfaces(1.4, .2)], **kwargs)
    for model in (numpy, tensor):
        if method == "udd":
            model.configure_sampling_method("udd", {"udd_bias_weight": .45})
        else:
            model.set_lcm_mode(0, pair=[0, 1], sigma=3.5, alpha_eV=.05)
    expected = numpy.evaluate_atoms(structures())
    actual = tensor(PackedBatch.from_atoms(structures()))
    np.testing.assert_allclose(actual["energy"].detach(), expected["energy"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual["forces"].detach(), expected["forces"], rtol=1e-5, atol=1e-5)
    for index in range(2):
        for key in ("Fmeanmax", "physical_F0_max", "physical_F1_max", "dE01", "dE01_stdev", "lcm_bias_Fmax"):
            assert tensor.diagnostics_for_graph(index)[key] == pytest.approx(numpy.diagnostics_for_graph(index)[key], abs=1e-6)
