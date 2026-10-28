"""ASE runner mechanics behind the shared controller protocol."""

import numpy as np
import pytest
from ase import units
from ase.constraints import FixAtoms

from alframework.ml_interfaces.sampling_inference import NumpyASEEnsembleModel
from alframework.samplers.ase_sampling import ASEDynamicsRunner
from tests.test_sampling_inference import Quadratic, structures


def runner_for(atoms, temperature=(0., 0.), friction=0.):
    return ASEDynamicsRunner(model=NumpyASEEnsembleModel([Quadratic(), Quadratic()]),
        atoms_list=atoms, dt_fs=.1, temperature_K=temperature, friction_per_fs=friction,
        random_seed=42, device="cpu")


@pytest.mark.parametrize("speed", [0., .05])
def test_ase_one_step_initial_velocity_and_kinetic_energy_roundtrip(speed):
    atoms = structures()
    for system in atoms:
        system.set_velocities(np.full((len(system), 3), speed))
    runner = runner_for(atoms)
    for i, initial in enumerate(atoms):
        restored = initial.copy(); runner.sync_graph_to_atoms(i, restored)
        assert restored.get_kinetic_energy() == pytest.approx(initial.get_kinetic_energy())
    runner.run(1)
    for i, initial in enumerate(atoms):
        restored = initial.copy(); runner.sync_graph_to_atoms(i, restored)
        h = .1 * units.fs
        expected_positions = initial.positions + h * initial.get_velocities() - .5 * h*h * initial.positions / initial.get_masses()[:, None]
        expected_velocities = initial.get_velocities() - .5*h * (initial.positions + expected_positions) / initial.get_masses()[:, None]
        np.testing.assert_allclose(restored.positions, expected_positions, atol=1e-12)
        np.testing.assert_allclose(restored.get_velocities(), expected_velocities, atol=1e-12)


def test_ase_independent_freeze_cell_temperature_and_constraints():
    atoms = structures()
    for system in atoms:
        system.set_cell([[5., 0, 0], [.2, 6., 0], [.1, .4, 7.]])
        system.set_pbc([True, True, False])
        system.set_velocities(np.full((len(system), 3), .01))
    atoms[1].set_constraint(FixAtoms(indices=[0]))
    runner = runner_for(atoms, temperature=[300., 600.], friction=.01)
    runner.run(1)
    frozen = atoms[0].copy(); runner.sync_graph_to_atoms(0, frozen)
    runner.freeze_graph(0)
    runner.set_temperature([600., 300.])
    runner.set_cell(0, frozen.cell.array * 2.)
    runner.run(4)
    restored = frozen.copy(); runner.sync_graph_to_atoms(0, restored)
    np.testing.assert_array_equal(restored.positions, frozen.positions)
    np.testing.assert_array_equal(restored.cell.array, frozen.cell.array)
    np.testing.assert_array_equal(restored.get_velocities(), 0.)
    np.testing.assert_array_equal(runner.atoms_list[1].positions[0], atoms[1].positions[0])
    runner.close()


def test_ase_density_scaling_preserves_velocities_and_refreshes_forces():
    atoms = structures()
    for system in atoms:
        system.set_cell(np.eye(3)*5.)
        system.set_pbc(True)
        system.set_velocities(np.full((len(system), 3), .03))
    runner = runner_for(atoms)
    runner.set_cell(1, atoms[1].cell.array * 1.2)
    restored = atoms[1].copy(); runner.sync_graph_to_atoms(1, restored)
    np.testing.assert_allclose(restored.positions, atoms[1].positions*1.2)
    np.testing.assert_allclose(restored.get_velocities(), atoms[1].get_velocities())
    assert restored.get_kinetic_energy() == pytest.approx(atoms[1].get_kinetic_energy())
    assert runner.diagnostics_for_graph(1)["sE0"] == pytest.approx(.5*np.sum(restored.positions**2))


def test_ase_seeded_thermal_trajectory_is_repeatable():
    first = runner_for(structures(), temperature=[300., 600.], friction=.01)
    second = runner_for(structures(), temperature=[300., 600.], friction=.01)
    first.run(4); second.run(4)
    for left, right in zip(first.atoms_list, second.atoms_list):
        np.testing.assert_array_equal(left.positions, right.positions)
