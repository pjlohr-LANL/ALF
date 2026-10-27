"""Shared sampling policies and actual ALCHEMI adapter conformance."""
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms, units

from alframework.samplers.batched_sampling import (
    _minimum_distance, run_batched_sampling, sampler_device,
)
from alframework.tools.molecules_class import MoleculesObject
from tests.helpers.fakes import _config


def config(policy="continue"):
    result = _config(policy=policy, batch_size=4, return_top_k=8)
    result.pop("alchemi_baoab")
    result.update(batching={"batch_size": 4},
                  dynamics={"backend": "alchemi", "allow_cpu_debug": True},
                  dt=.1, maxt=.0003, Ncheck=1, friction_per_fs=0.,
                  srt_temp=[0., 0.], end_temp=[0., 0.])
    return result


def systems(periodic=True):
    cell = np.array([[5., 0., 0.], [.6, 6., 0.], [.2, .7, 7.]])
    atoms = [Atoms("H2", positions=[[.1, .2, .3], [.7, .3, .2]], cell=cell, pbc=periodic),
             Atoms("H2O", positions=[[.2, .1, .3], [.8, .2, .1], [.3, .7, .2]],
                   cell=cell * 1.2, pbc=periodic)]
    for item in atoms:
        item.set_velocities(np.full((len(item), 3), .05))
    return atoms


def harmonic_model():
    torch = pytest.importorskip("torch")
    from alframework.ml_interfaces.batched_inference import BatchedEnsembleCalculator, batch_slices

    class Harmonic(BatchedEnsembleCalculator):
        def __init__(self):
            super().__init__(device="cpu")
            self.supports_pbc = True

        def ensemble_forward(self, batch):
            energy = torch.stack([.5 * batch.positions[part].square().sum()
                                  for part in batch_slices(batch)])
            return {"energy_contributions": torch.stack((energy, energy + 4.)),
                    "force_contributions": torch.stack((-batch.positions, -batch.positions))}
    return Harmonic()


@pytest.mark.parametrize("periodic", [False, True])
def test_actual_alchemi_ragged_analytic_step_and_cell_updates(periodic):
    pytest.importorskip("nvalchemi")
    from alframework.samplers.alchemi_sampling import AlchemiDynamicsRunner
    atoms = systems(periodic)
    runner = AlchemiDynamicsRunner(model=harmonic_model(), atoms_list=atoms, dt_fs=.1,
                                   temperature_K=np.zeros(2), friction_per_fs=0.,
                                   random_seed=4, device="cpu")
    runner.run(1)
    for index, initial in enumerate(atoms):
        result = initial.copy()
        runner.sync_graph_to_atoms(index, result)
        step = .1 * units.fs
        expected = initial.positions + step * initial.get_velocities() - .5 * step**2 * initial.positions / initial.get_masses()[:, None]
        velocity = initial.get_velocities() - .5 * step * (initial.positions + expected) / initial.get_masses()[:, None]
        np.testing.assert_allclose(result.positions, expected, atol=1e-7)
        np.testing.assert_allclose(result.get_velocities(), velocity, atol=1e-7)
        np.testing.assert_allclose(result.cell, initial.cell)
        assert runner.diagnostics_for_graph(index)["F0"].shape == (len(initial), 3)
    if periodic:
        result = atoms[1].copy()
        runner.sync_graph_to_atoms(1, result)
        previous = result.copy()
        runner.set_cell(1, np.asarray(result.cell) * 1.1)
        runner.sync_graph_to_atoms(1, result)
        np.testing.assert_allclose(result.positions, previous.positions * 1.1, atol=1e-7)
        np.testing.assert_allclose(result.get_velocities(), previous.get_velocities(), atol=1e-7)
        runner.freeze_graph(1)
        frozen = result.copy()
        runner.set_cell(1, np.asarray(result.cell) * 1.2)
        runner.run(2)
        runner.sync_graph_to_atoms(1, result)
        np.testing.assert_array_equal(result.positions, frozen.positions)
        np.testing.assert_array_equal(result.cell, frozen.cell)


@pytest.mark.parametrize("backend", ["alchemi", "torchsim"])
def test_shared_density_schedule_returns_consistent_ragged_periodic_candidates(backend):
    pytest.importorskip("nvalchemi" if backend == "alchemi" else "torch_sim")
    cfg = config()
    cfg["dynamics"]["backend"] = backend
    cfg.update(end_dens=[.02, .02], amp_dens=[0., 0.], per_dens=[1., 1.])
    molecules = [MoleculesObject(atoms, f"source-{i}") for i, atoms in enumerate(systems())]
    out = run_batched_sampling(molecules, cfg, harmonic_model(), device="cpu")
    assert len(out) == 6
    for item in out:
        metadata = item.get_metadata()
        atoms = item.get_atoms()
        index = int(metadata["batch_index"])
        assert len(atoms) == (2, 3)[index]
        assert metadata["parent_molecule_id"] == f"source-{index}"
        actual_density = 1.66054 * atoms.get_masses().sum() / atoms.get_volume()
        expected = metadata["denss"][-1] * 1.66054 if metadata["denss"] else metadata["Rsrt"]
        assert actual_density == pytest.approx(expected, rel=2e-6)
        assert len(metadata["denss"]) == len(metadata["temps"])
        assert metadata["dynamics_backend"] == f"{backend}_baoab"
        assert metadata["force_screen_source"] == "ensemble_mean_physical_all_states"
        assert atoms.pbc.all()


def test_periodic_distance_checks_images_and_exact_overlaps():
    assert _minimum_distance(Atoms("H", positions=[[0, 0, 0]], cell=[.5, 4, 4], pbc=True)) == pytest.approx(.5)
    assert _minimum_distance(Atoms("H2", positions=[[0, 0, 0], [4.9, 0, 0]], cell=[5, 5, 5], pbc=True)) == pytest.approx(.1)
    assert _minimum_distance(Atoms("H2", positions=np.zeros((2, 3)))) == 0.


def test_calculator_periodicity_is_validated_before_dynamics():
    model = harmonic_model()
    model.supports_pbc = False
    with pytest.raises(ValueError, match="calculator does not support periodic"):
        run_batched_sampling([MoleculesObject(systems()[0], "periodic")], config(), model)


@pytest.mark.parametrize("vendor,mask", [("amd", "ROCR_VISIBLE_DEVICES"), ("nvidia", "CUDA_VISIBLE_DEVICES")])
def test_device_selection_honors_scheduler_visibility(monkeypatch, vendor, mask):
    monkeypatch.setenv(mask, "4,7")
    monkeypatch.setenv("PARSL_WORKER_RANK", "3")
    selected = []
    fake = SimpleNamespace(version=SimpleNamespace(hip="7.0" if vendor == "amd" else None),
                           cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                                                set_device=selected.append), device=lambda value: value)
    cfg = config()
    cfg["dynamics"]["backend"] = "torchsim"
    device, metadata = sampler_device(cfg, 8, torch_module=fake)
    assert device == "cuda:1"
    assert selected == ["cuda:1"]
    assert metadata["sampler_visible_device"] == 7
    assert metadata["sampler_gpu_vendor"] == vendor
    assert __import__("os").environ[mask] == "4,7"


def test_alchemi_rejects_rocm_before_selecting_a_device():
    fake = SimpleNamespace(version=SimpleNamespace(hip="7.0"))
    with pytest.raises(RuntimeError, match="select dynamics.backend='torchsim'"):
        sampler_device(config(), 1, torch_module=fake)
