"""Optional engine conformance against the actual pinned TorchSim integrator."""

import builtins
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms, units

from alframework.samplers.torchsim_sampling import (
    TorchSimDynamicsRunner,
    ensure_torchsim_available,
)


@pytest.fixture
def torchsim():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_sim")
    return torch


class HarmonicModel:
    """Analytic independent-system model; no mocked engine functions."""

    def set_device(self, device):
        self.device = device

    def __call__(self, batch):
        import torch

        energies = torch.zeros(batch.num_graphs, dtype=batch.positions.dtype, device=batch.device)
        energies.index_add_(0, batch.system_idx, .5 * batch.positions.square().sum(dim=1))
        self.last_batch = batch
        self.last_energy = energies
        return {"energy": energies, "forces": -batch.positions}

    def diagnostics_for_graph(self, index):
        return {"energy": float(self.last_energy[index])}

    def set_lcm_mode(self, graph_index, **kwargs):
        self.last_lcm = (graph_index, kwargs)


def systems(periodic=False, speed=.05):
    cell = np.array([[5., 0., 0.], [.6, 6., 0.], [.2, .7, 7.]])
    molecules = [
        Atoms("H2", positions=[[.1, .2, .3], [.5, .3, .2]], cell=cell, pbc=periodic),
        Atoms("H2O", positions=[[.2, .1, .3], [.7, .2, .1], [.3, .5, .2]], cell=cell * 1.2, pbc=periodic),
    ]
    for atoms in molecules:
        atoms.set_velocities(np.full((len(atoms), 3), speed))
    return molecules


def runner_for(atoms, *, temperature=(0., 0.), friction=0., **kwargs):
    return TorchSimDynamicsRunner(
        model=HarmonicModel(), atoms_list=atoms, dt_fs=.1,
        temperature_K=temperature, friction_per_fs=friction,
        random_seed=4, device="cpu", **kwargs,
    )


def test_runner_module_import_does_not_load_optional_engines():
    script = """
import builtins, sys
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torch_sim', 'nvalchemi', 'nvalchemiops', 'warp'}:
        raise AssertionError('Unexpected optional engine import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
import alframework.samplers.torchsim_sampling
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_missing_optional_engine_has_installation_guidance(monkeypatch):
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        if name.startswith("torch_sim"):
            raise ModuleNotFoundError("Deliberately missing engine")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)
    with pytest.raises(ImportError, match="torchsim_dynamics extra"):
        ensure_torchsim_available()


@pytest.mark.parametrize("periodic", [False, True], ids=["molecular", "triclinic-periodic"])
@pytest.mark.parametrize("speed", [0., .05], ids=["zero-velocities", "nonzero-velocities"])
def test_actual_torchsim_ragged_one_step_and_velocity_roundtrip(torchsim, periodic, speed):
    atoms = systems(periodic, speed)
    runner = runner_for(atoms)
    np.testing.assert_allclose(runner.state.momenta.detach().numpy(),
                               np.concatenate([a.get_momenta() for a in atoms]), atol=1e-7)
    assert runner.batch.num_nodes_per_graph.tolist() == [2, 3]
    for index, initial in enumerate(atoms):
        restored = initial.copy()
        runner.sync_graph_to_atoms(index, restored)
        np.testing.assert_allclose(restored.get_velocities(), initial.get_velocities(), atol=1e-8)
        np.testing.assert_allclose(restored.cell, initial.cell, atol=1e-6)
        assert restored.get_kinetic_energy() == pytest.approx(initial.get_kinetic_energy(), abs=1e-8)
        assert restored.pbc.tolist() == initial.pbc.tolist()
    runner.run(1)
    assert runner.nsteps == 1
    for index, initial in enumerate(atoms):
        restored = initial.copy()
        runner.sync_graph_to_atoms(index, restored)
        h = .1 * units.fs
        acceleration = -initial.positions / initial.get_masses()[:, None]
        expected_positions = initial.positions + h * initial.get_velocities() + .5 * h * h * acceleration
        expected_velocities = initial.get_velocities() - .5 * h * (
            initial.positions + expected_positions
        ) / initial.get_masses()[:, None]
        np.testing.assert_allclose(restored.positions, expected_positions, atol=1e-7)
        np.testing.assert_allclose(restored.get_velocities(), expected_velocities, atol=1e-7)


def test_actual_torchsim_freezes_one_replica_through_temperature_and_cell_changes(torchsim):
    atoms = systems(periodic=True)
    runner = runner_for(atoms, temperature=[10., 20.], friction=.01)
    runner.run(1)
    frozen = atoms[0].copy()
    runner.sync_graph_to_atoms(0, frozen)
    runner.freeze_graph(0)
    runner.set_temperature([200., 300.])
    runner.set_cell(0, frozen.cell.array * 2)
    runner.set_cell(1, atoms[1].cell.array * 1.1)
    runner.run(5)
    first, second = [a.copy() for a in atoms]
    runner.sync_graph_to_atoms(0, first)
    runner.sync_graph_to_atoms(1, second)
    np.testing.assert_array_equal(first.positions, frozen.positions)
    np.testing.assert_array_equal(first.cell.array, frozen.cell.array)
    np.testing.assert_array_equal(first.get_velocities(), 0.)
    assert not np.allclose(second.positions, atoms[1].positions * 1.1)
    np.testing.assert_allclose(second.cell.array, atoms[1].cell.array * 1.1, atol=1e-6)
    assert runner._temperature.tolist() == [200., 300.]


def test_actual_torchsim_cell_scaling_preserves_fractional_positions_and_kinetic_energy(torchsim):
    atoms = systems(periodic=True)
    runner = runner_for(atoms)
    updated_cell = atoms[1].cell.array * 1.3
    runner.set_cell(1, updated_cell)
    restored = atoms[1].copy()
    runner.sync_graph_to_atoms(1, restored)
    np.testing.assert_allclose(restored.get_scaled_positions(), atoms[1].get_scaled_positions(), atol=1e-7)
    np.testing.assert_allclose(restored.get_velocities(), atoms[1].get_velocities(), atol=1e-8)
    assert restored.get_kinetic_energy() == pytest.approx(atoms[1].get_kinetic_energy(), abs=1e-8)
    assert runner.diagnostics_for_graph(1)["energy"] == pytest.approx(.5 * np.sum(restored.positions**2), rel=1e-6)
    runner.set_lcm_mode(1, pair=[0, 1], sigma=2., alpha_eV=.1)
    assert runner.model.last_lcm == (1, {"pair": [0, 1], "sigma": 2., "alpha_eV": .1})


def test_actual_torchsim_rejects_invalid_per_system_controls(torchsim):
    runner = runner_for(systems())
    with pytest.raises(ValueError, match="one value per system"):
        runner.set_temperature([1., 2., 3.])
    with pytest.raises(ValueError, match="nonnegative"):
        runner.set_temperature([-1., 2.])
    with pytest.raises(ValueError, match="positive volume"):
        runner.set_cell(0, np.zeros((3, 3)))
    with pytest.raises(IndexError, match="outside"):
        runner.freeze_graph(2)


def test_actual_torchsim_seed_reproduces_thermal_trajectory(torchsim):
    def trajectory(seed):
        runner = TorchSimDynamicsRunner(
            model=HarmonicModel(), atoms_list=systems(), dt_fs=.1,
            temperature_K=[200., 300.], friction_per_fs=.01,
            random_seed=seed, device="cpu",
        )
        runner.run(4)
        return runner.state.positions.detach().clone()

    first, repeated, different = trajectory(42), trajectory(42), trajectory(43)
    assert torchsim.equal(first, repeated)
    assert not torchsim.equal(first, different)


def test_actual_torchsim_frozen_geometry_never_reaches_inference_after_displacement(torchsim):
    class RecordingModel(HarmonicModel):
        frozen_coordinates = None

        def __call__(self, batch):
            if self.frozen_coordinates is not None:
                torchsim.testing.assert_close(batch.positions[:2], self.frozen_coordinates, rtol=0., atol=0.)
            return super().__call__(batch)

    model = RecordingModel()
    runner = TorchSimDynamicsRunner(
        model=model, atoms_list=systems(), dt_fs=.1, temperature_K=[1000., 2000.],
        friction_per_fs=.1, random_seed=42, device="cpu",
    )
    runner.run(1)
    runner.freeze_graph(0)
    model.frozen_coordinates = runner.state.positions[:2].clone()
    runner.run(5)
