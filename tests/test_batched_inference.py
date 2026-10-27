"""Conformance of packed ensemble inference across systems and model interfaces."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

torch = pytest.importorskip("torch")

from alframework.ml_interfaces.batched_inference import (
    ASEEnsembleModel, BatchedEnsembleCalculator, HippynnEnsembleModel,
    NativeEnsembleModel, PackedBatch, batch_slices,
)


def _systems(periodic=False):
    systems = [Atoms("H2", positions=[[.1, .2, .3], [.1, .2, 1.05]]),
               Atoms("OH2", positions=[[.2, .3, .4], [1., .3, .4], [.1, 1.1, .4]])]
    for index, atoms in enumerate(systems):
        atoms.set_velocities(np.arange(len(atoms) * 3).reshape(-1, 3) * .01)
        atoms.set_masses(atoms.get_masses() * (index + 1))
        if periodic:
            atoms.set_cell([[5. + index, .3, .1], [0., 4.8, .2], [0., 0., 5.1]])
            atoms.set_pbc(True)
    return systems


class HarmonicStates(BatchedEnsembleCalculator):
    def ensemble_forward(self, batch):
        energy = torch.stack([batch.positions[s].square().sum() / 2 for s in batch_slices(batch)])
        states = {}
        for state in self.physical_force_states:
            factor = state + 1
            states[state] = {
                "energy_contributions": torch.stack([.5 * factor * energy, 1.5 * factor * energy]),
                "force_contributions": torch.stack([-.5 * factor * batch.positions, -1.5 * factor * batch.positions]),
            }
        return {**states[self.selected_state_value], "state_contributions": states}


@pytest.mark.parametrize("periodic", [False, True])
def test_packed_state_preserves_ragged_geometry_and_ase_velocities(periodic):
    systems = _systems(periodic)
    batch = PackedBatch.from_atoms(systems, dtype=torch.float64)
    assert batch.num_nodes_per_graph.tolist() == [2, 3]
    assert batch.batch_ptr.tolist() == [0, 2, 5]
    assert batch.system_idx.tolist() == [0, 0, 1, 1, 1]
    for index, atom_slice in enumerate(batch_slices(batch)):
        atoms = systems[index]
        np.testing.assert_array_equal(batch.atomic_numbers[atom_slice], atoms.numbers)
        np.testing.assert_allclose(batch.positions[atom_slice], atoms.positions)
        np.testing.assert_allclose(batch.velocities[atom_slice], atoms.get_velocities())
        np.testing.assert_allclose(batch.atomic_masses[atom_slice], atoms.get_masses())
        np.testing.assert_allclose(batch.cell[index], atoms.cell.array)
        np.testing.assert_array_equal(batch.pbc[index], atoms.pbc)


@pytest.mark.parametrize("well", [False, True])
def test_segmented_uncertainty_and_physical_forces_match_individual_systems(well):
    systems = _systems()
    options = {"physical_force_states": [0, 1]}
    if well:
        options["well_params"] = {"r_start": .5, "force": 2., "mass_weighted": True}
    model = HarmonicStates(**options)
    output = model(PackedBatch.from_atoms(systems))
    diagnostics = [model.diagnostics_for_graph(index) for index in range(2)]
    for index, atoms in enumerate(systems):
        single = HarmonicStates(**options)
        expected = single(PackedBatch.from_atoms([atoms]))
        result = single.diagnostics_for_graph(0)
        atom_slice = slice(0, 2) if index == 0 else slice(2, 5)
        torch.testing.assert_close(output["forces"][atom_slice], expected["forces"])
        torch.testing.assert_close(output["energy"][index], expected["energy"][0])
        for key in result:
            if isinstance(result[key], str):
                assert diagnostics[index][key] == result[key]
            else:
                np.testing.assert_allclose(diagnostics[index][key], result[key], rtol=1.e-6)
        assert diagnostics[index]["F0"].shape == (len(atoms), 3)
        assert diagnostics[index]["F1_stdev"].shape == (len(atoms), 3)
        assert diagnostics[index]["Fs"] == pytest.approx(np.abs(atoms.positions * .5).mean())


def test_lcm_switch_only_changes_the_selected_ragged_system_and_preserves_physical_screen():
    row = {"lower_state": 0, "upper_state": 1, "gap_key": "dE01"}
    model = HarmonicStates(physical_force_states=[0, 1], gap_rows=[row],
                           gap_seeking_settings={"enabled": True, "rows": [row]})
    batch = PackedBatch.from_atoms(_systems())
    before = model(batch)
    physical = model.diagnostics_for_graph(1)["Fmeanmax"]
    model.set_lcm_mode(1, pair=[0, 1], sigma=3.5, alpha_eV=.05)
    after = model(batch)
    torch.testing.assert_close(before["forces"][:2], after["forces"][:2])
    assert not torch.allclose(before["forces"][2:], after["forces"][2:])
    assert model.diagnostics_for_graph(1)["Fmeanmax"] == physical
    assert model.diagnostics_for_graph(0)["lcm_bias_Fmax"] == 0.
    assert model.diagnostics_for_graph(1)["lcm_bias_Fmax"] > 0.


class CellAwareCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        self.seen = []

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.seen.append(atoms.copy())
        self.results = {"energy": self.factor * (np.square(atoms.positions).sum() + atoms.get_volume()),
                        "forces": -2 * self.factor * atoms.positions}


def test_ase_fallback_preserves_each_periodic_cell_species_masses_and_force_width():
    systems = _systems(True)
    calculators = [CellAwareCalculator(1.), CellAwareCalculator(3.)]
    model = ASEEnsembleModel(calculators, model_mode="ground_state", selected_state_value=None)
    result = model(PackedBatch.from_atoms(systems))
    np.testing.assert_allclose(result["forces"], -4 * np.concatenate([atoms.positions for atoms in systems]))
    for calculator in calculators:
        for seen, original in zip(calculator.seen, systems):
            np.testing.assert_allclose(seen.cell.array, original.cell.array)
            np.testing.assert_array_equal(seen.pbc, original.pbc)
            np.testing.assert_array_equal(seen.numbers, original.numbers)
            np.testing.assert_allclose(seen.get_masses(), original.get_masses())
    assert model.calculator_interface == "ase_fallback"
    assert model.diagnostics_for_graph(1)["F0"].shape == (3, 3)


def _hippynn_model(periodic=False, directory=None):
    hippynn = pytest.importorskip("hippynn")
    from hippynn.graphs import inputs, networks, targets, physics
    hippynn.custom_kernels.set_custom_kernels("pytorch")
    models = []
    for seed in (41, 42):
        torch.manual_seed(seed)
        species = inputs.SpeciesNode(db_name="species")
        positions = inputs.PositionsNode(db_name="coordinates")
        parents = (species, positions)
        if periodic:
            parents += (inputs.CellNode(db_name="cell"),)
        network = networks.Hipnn("network", parents, periodic=periodic, module_kwargs={
            "possible_species": [0, 1, 8], "n_features": 4, "n_sensitivities": 4,
            "dist_soft_min": .5, "dist_soft_max": 2., "dist_hard_max": 2.5,
            "n_interaction_layers": 1, "n_atom_layers": 1,
            "sensitivity_type": "inverse", "resnet": True})
        energy = targets.HEnergyNode("energy_head", network).mol_energy
        energy.db_name = "energy"
        forces = physics.GradientNode("force_gradient", (energy, positions), sign=-1, db_name="forces")
        model = hippynn.graphs.GraphModule(parents, [energy, forces])
        models.append(model)
        if directory is not None:
            from types import SimpleNamespace
            member = Path(directory) / f"model-{len(models) - 1:02d}"
            member.mkdir(parents=True, exist_ok=True)
            torch.save({"training_modules": SimpleNamespace(model=model)}, member / "experiment_structure.pt")
            torch.save(model.state_dict(), member / "best_model.pt")
    graph, (_, info) = hippynn.graphs.make_ensemble(models, quiet=True)
    return HippynnEnsembleModel(
        ensemble_graph=graph,
        state_nodes=[{"state": 0, "energy_all_node": graph.node_from_name("ensemble_energy").all,
                      "force_all_node": graph.node_from_name("ensemble_forces").all,
                      "energy_model_count": info["energy"], "force_model_count": info["forces"]}],
        selected_state_value=0, species_key="species", coordinates_key="coordinates",
        cell_key="cell" if periodic else None, device="cpu")


@pytest.mark.parametrize("periodic", [False, True])
def test_real_hippynn_ragged_predictions_and_statistics_match_individual_evaluations(periodic):
    model = _hippynn_model(periodic)
    systems = _systems(periodic)
    batch = PackedBatch.from_atoms(systems)
    batched = model(batch)
    diagnostic_rows = [model.diagnostics_for_graph(index) for index in range(2)]
    for index, atom_slice in enumerate(batch_slices(batch)):
        single = model(PackedBatch.from_atoms([systems[index]]))
        torch.testing.assert_close(batched["energy"][index], single["energy"][0], rtol=1.e-5, atol=1.e-6)
        torch.testing.assert_close(batched["forces"][atom_slice], single["forces"], rtol=1.e-5, atol=1.e-6)
        for key in ("Es", "Fs", "Fsmax", "Fsrms", "Fmeanmax"):
            assert diagnostic_rows[index][key] == pytest.approx(model.diagnostics_for_graph(0)[key], rel=1.e-5, abs=1.e-6)
    # Reordering atoms changes the returned force order, not the physical prediction.
    reordered = systems[1][[2, 0, 1]]
    permuted = model(PackedBatch.from_atoms([reordered]))
    reference = model(PackedBatch.from_atoms([systems[1]]))
    torch.testing.assert_close(permuted["energy"], reference["energy"], rtol=1.e-5, atol=1.e-6)
    torch.testing.assert_close(permuted["forces"], reference["forces"][[2, 0, 1]], rtol=1.e-5, atol=1.e-6)


def test_molecular_hippynn_rejects_periodic_inputs_and_untrained_species():
    model = _hippynn_model()
    with pytest.raises(ValueError, match="molecular.*periodic"):
        model(PackedBatch.from_atoms(_systems(True)))
    with pytest.raises(ValueError, match="atomic numbers.*6"):
        model(PackedBatch.from_atoms([Atoms("CH", positions=[[0, 0, 0], [0, 0, 1.]])]))


@pytest.mark.parametrize("periodic", [False, True])
def test_checkpoint_loader_roundtrip_matches_in_memory_inference(tmp_path, periodic):
    from alframework.ml_interfaces.batched_inference import load_hippynn_batched_model
    reference = _hippynn_model(periodic, directory=tmp_path)
    loaded = load_hippynn_batched_model(
        str(tmp_path), model_mode="ground_state", selected_state_value=None,
        ML_config={"species_key": "species", "coordinates_key": "coordinates",
                   "energy_key": "energy", "force_key": "forces",
                   "cell_key": "cell" if periodic else None},
        properties_list={"energy": ["energy", "system", 1.], "forces": ["forces", "atomic", 1.]},
        sampler_config={}, device="cpu")
    batch = PackedBatch.from_atoms(_systems(periodic))
    expected, actual = reference(batch), loaded(batch)
    torch.testing.assert_close(actual["energy"], expected["energy"])
    torch.testing.assert_close(actual["forces"], expected["forces"])
    assert loaded.supports_pbc == periodic
    for index in range(batch.num_graphs):
        for key in ("Es", "Fs", "Fsmax", "Fsrms", "Fmeanmax"):
            assert loaded.diagnostics_for_graph(index)[key] == pytest.approx(reference.diagnostics_for_graph(index)[key])


def test_shared_inference_import_does_not_load_nvidia_stack(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = '''
import builtins,sys
original = builtins.__import__
def checked(name,*args,**kwargs):
    if name.split('.')[0] in {'nvalchemi','nvalchemiops','warp'}:
        raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__ = checked
from alframework.ml_interfaces.batched_inference import PackedBatch, HippynnEnsembleModel
assert not any(name.split('.')[0] in {'nvalchemi','nvalchemiops','warp'} for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                            env=dict(os.environ, PYTHONPATH=str(root)), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
