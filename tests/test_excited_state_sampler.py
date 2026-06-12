from __future__ import annotations

import json
import sys
import types
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from alframework.samplers import excited_state_sampling as sampler_mod
from alframework.tools.molecules_class import MoleculesObject


def _properties_list() -> dict[str, list[object]]:
    return {
        "sE0": ["sE0", "system", 1.0],
        "F0": ["F0", "atomic", 1.0],
        "sE1": ["sE1", "system", 1.0],
        "F1": ["F1", "atomic", 1.0],
    }


def _snapshot(*, energy, forces, e0, e1, u0, u1, f0, f1):
    return {
        "energy": float(energy),
        "forces": np.asarray(forces, dtype=np.float64),
        "E_mean_S0": float(e0),
        "E_std_S0": float(u0),
        "F_std_S0": np.asarray(f0, dtype=np.float64),
        "E_mean_S1": float(e1),
        "E_std_S1": float(u1),
        "F_std_S1": np.asarray(f1, dtype=np.float64),
    }


def test_selected_state_uncertainty_is_default_for_gate_and_score():
    atoms = Atoms(symbols=["O", "H", "H"], positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    metrics = sampler_mod._results_to_metrics(
        atoms,
        _snapshot(
            energy=0.0,
            forces=np.zeros((3, 3)),
            e0=0.00,
            e1=0.03,
            u0=0.20,
            u1=2.00,
            f0=np.full((3, 3), 0.05),
            f1=np.full((3, 3), 3.00),
        ),
        [
            {"state": 0},
            {"state": 1},
        ],
        [],
        selected_state=0,
        forces_override=np.zeros((3, 3)),
    )
    sampler_config = {
        "uncertainty": {"enabled": True, "min_uE": 0.15, "min_uF": 0.10, "logic": "either"},
        "score": {"w_energy": 2.0, "w_force": 3.0, "w_gap": 0.0},
    }

    assert metrics["uncertainty_state"] == 0
    assert metrics["uE_selected"] == pytest.approx(0.20)
    assert metrics["uF_selected"] == pytest.approx(0.05)
    assert metrics["uE_max"] == pytest.approx(2.00)
    assert metrics["uF_max"] == pytest.approx(3.00)
    assert sampler_mod._passes_uncertainty_gate(metrics, sampler_config)
    components = sampler_mod.compute_excited_state_score_components(metrics, sampler_config)
    assert components["energy_uncertainty"] == pytest.approx(0.40)
    assert components["force_uncertainty"] == pytest.approx(0.15)


def test_nonselected_state_uncertainty_does_not_pass_selected_state_gate():
    atoms = Atoms(symbols=["O", "H", "H"], positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    metrics = sampler_mod._results_to_metrics(
        atoms,
        _snapshot(
            energy=0.0,
            forces=np.zeros((3, 3)),
            e0=0.00,
            e1=0.03,
            u0=0.01,
            u1=2.00,
            f0=np.full((3, 3), 0.01),
            f1=np.full((3, 3), 3.00),
        ),
        [
            {"state": 0},
            {"state": 1},
        ],
        [],
        selected_state=0,
        forces_override=np.zeros((3, 3)),
    )
    sampler_config = {
        "uncertainty": {"enabled": True, "min_uE": 0.15, "min_uF": 0.10, "logic": "either"},
        "score": {"w_energy": 1.0, "w_force": 1.0, "w_gap": 0.0},
    }

    assert not sampler_mod._passes_uncertainty_gate(metrics, sampler_config)
    components = sampler_mod.compute_excited_state_score_components(metrics, sampler_config)
    assert components["energy_uncertainty"] == pytest.approx(0.01)
    assert components["force_uncertainty"] == pytest.approx(0.01)


def test_all_state_uncertainty_scope_preserves_legacy_aggregate_behavior():
    metrics = {
        "selected_state": 0,
        "uncertainty_state": 0,
        "uE_selected": 0.01,
        "uF_selected": 0.02,
        "uE_max": 1.0,
        "uF_max": 2.0,
        "uE_rms": 0.5,
        "uF_rms": 0.75,
        "min_gap": 0.03,
        "gap_stds": {},
    }
    sampler_config = {
        "score": {
            "w_energy": 1.0,
            "w_force": 1.0,
            "w_gap": 0.0,
            "uncertainty_scope": "all_states",
            "uncertainty_aggregate": "rms",
        }
    }

    components = sampler_mod.compute_excited_state_score_components(metrics, sampler_config)
    assert components["energy_uncertainty"] == pytest.approx(0.5)
    assert components["force_uncertainty"] == pytest.approx(0.75)


def _install_fake_sampling_runtime(monkeypatch, snapshots):
    fake_snapshots = [deepcopy(s) for s in snapshots]

    class FakeHippynnCalculator(Calculator):
        implemented_properties = ["energy", "forces"]
        snapshots = fake_snapshots

        def __init__(self, energy=None, charges=None, skin=1.0, extra_properties=None, en_unit=None, dist_unit=None, species_set=None, indexer=None, offset=None):
            del energy, charges, skin, extra_properties, en_unit, dist_unit, species_set, indexer, offset
            super().__init__()
            self._index = 0
            self.results = {}

        def advance(self):
            idx = min(self._index, len(self.snapshots) - 1)
            self.results = deepcopy(self.snapshots[idx])
            self._index += 1

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            del atoms, properties, system_changes
            if not self.results:
                self.advance()

        def to(self, *args, **kwargs):
            del args, kwargs
            return self

    class FakeNode:
        def __init__(self, name):
            self.mean = f"{name}.mean"
            self.std = f"{name}.std"

    class FakeGraph:
        def node_from_name(self, name):
            return FakeNode(str(name))

    class FakeLangevin:
        instances = []

        def __init__(self, atoms, timestep, friction, temperature_K):
            del timestep, friction
            self.atoms = atoms
            self.temperature_history = [float(temperature_K)]
            self.run_calls = []
            self.__class__.instances.append(self)

        def set_temperature(self, temperature_K):
            self.temperature_history.append(float(temperature_K))

        def run(self, steps):
            self.run_calls.append(int(steps))
            self.atoms.calc.advance()

    def fake_load_excited_state_ensemble(ensemble_directory, properties_list, device="cpu"):
        del ensemble_directory, properties_list, device
        return (
            FakeGraph(),
            [
                {
                    "state": 0,
                    "energy_node_base": "ensemble_sE0",
                    "force_node_base": "ensemble_F0",
                    "energy_db_name": "sE0",
                    "force_db_name": "F0",
                },
                {
                    "state": 1,
                    "energy_node_base": "ensemble_sE1",
                    "force_node_base": "ensemble_F1",
                    "energy_db_name": "sE1",
                    "force_db_name": "F1",
                },
            ],
            [],
        )

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.device = lambda name: name
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, set_device=lambda device: None)

    fake_hippynn = types.ModuleType("hippynn")
    fake_interfaces = types.ModuleType("hippynn.interfaces")
    fake_ase_interface = types.ModuleType("hippynn.interfaces.ase_interface")
    fake_ase_interface.HippynnCalculator = FakeHippynnCalculator

    monkeypatch.setattr(sampler_mod, "load_excited_state_ensemble", fake_load_excited_state_ensemble)
    monkeypatch.setattr(sampler_mod, "Langevin", FakeLangevin)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "hippynn", fake_hippynn)
    monkeypatch.setitem(sys.modules, "hippynn.interfaces", fake_interfaces)
    monkeypatch.setitem(sys.modules, "hippynn.interfaces.ase_interface", fake_ase_interface)

    return FakeLangevin


def test_run_excited_state_sampling_selects_highest_scoring_valid_frame(monkeypatch, tmp_path: Path):
    snapshots = [
        _snapshot(
            energy=0.0,
            forces=np.zeros((3, 3)),
            e0=0.00,
            e1=0.04,
            u0=0.00,
            u1=0.00,
            f0=np.zeros((3, 3)),
            f1=np.zeros((3, 3)),
        ),
        _snapshot(
            energy=0.1,
            forces=np.zeros((3, 3)),
            e0=0.00,
            e1=0.03,
            u0=0.20,
            u1=0.10,
            f0=np.full((3, 3), 0.10),
            f1=np.full((3, 3), 0.05),
        ),
        _snapshot(
            energy=0.2,
            forces=np.zeros((3, 3)),
            e0=0.00,
            e1=0.015,
            u0=0.40,
            u1=0.20,
            f0=np.full((3, 3), 0.20),
            f1=np.full((3, 3), 0.10),
        ),
    ]
    fake_langevin = _install_fake_sampling_runtime(monkeypatch, snapshots)

    schedule_calls = []

    def fake_annealing_schedule(t, tmax, amp, per, srt, end):
        schedule_calls.append((float(t), float(tmax), float(amp), float(per), float(srt), float(end)))
        return 150.0 + float(t)

    monkeypatch.setattr(sampler_mod, "annealing_schedule", fake_annealing_schedule)

    molecule = MoleculesObject(
        Atoms(symbols=["O", "H", "H"], positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
        "traj_0000",
    )
    result = sampler_mod.run_excited_state_sampling(
        molecule,
        sampler_config={
            "dt": 1.0,
            "maxt": 0.002,
            "Ncheck": 1,
            "min_time": 0.0,
            "friction": 0.02,
            "srt_temp": [100.0, 100.0],
            "end_temp": [200.0, 200.0],
            "amp_temp": [0.0, 0.0],
            "per_temp": [5.0, 5.0],
            "amp_dens": None,
            "per_dens": None,
            "end_dens": None,
            "meta_dir": str(tmp_path / "meta"),
            "metadata_format": "json",
            "write_traj_binary": False,
            "write_traj_xyz": False,
            "trajectory_frequency": 0.0,
            "trajectory_interval": 1,
            "min_distance_cutoff": 0.3,
            "max_force_cutoff": 10.0,
            "state_selection": {"mode": "fixed", "fixed_state": 0},
            "uncertainty": {"enabled": True, "min_uE": 0.05, "min_uF": 0.05, "logic": "either"},
            "score": {
                "w_energy": 1.0,
                "w_force": 1.0,
                "w_gap": 1.0,
                "gap_threshold_eV": 0.02,
                "gap_mode": "linear",
                "uncertainty_aggregate": "max",
            },
        },
        model_path=str(tmp_path / "models" / "model-{:04d}"),
        current_model_id=0,
        gpus_per_node=0,
        properties_list=_properties_list(),
    )

    assert result.get_atoms() is not None
    assert result.get_metadata()["selected_state"] == 0
    assert result.get_metadata()["best_candidate"]["step"] == 1
    assert result.get_metadata()["best_candidate"]["score"] > 0.6
    best_candidate = result.get_metadata()["best_candidate"]
    assert best_candidate["score_components"]["energy_uncertainty"] == pytest.approx(best_candidate["uE_selected"])
    assert best_candidate["score_components"]["force_uncertainty"] == pytest.approx(best_candidate["uF_selected"])
    assert schedule_calls[0] == (0.0, 0.002, 0.0, 5.0, 100.0, 200.0)
    assert fake_langevin.instances[0].run_calls == [1, 1, 1]
    meta_path = tmp_path / "meta" / "metadata-traj_0000.json"
    assert meta_path.exists()
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["best_candidate"]["score"] > 0.0
    assert "timing" not in payload


def test_run_excited_state_sampling_records_opt_in_ase_timing(monkeypatch, tmp_path: Path):
    snapshots = [
        _snapshot(
            energy=0.0,
            forces=np.zeros((3, 3)),
            e0=0.00,
            e1=0.04,
            u0=0.00,
            u1=0.00,
            f0=np.zeros((3, 3)),
            f1=np.zeros((3, 3)),
        ),
        _snapshot(
            energy=0.1,
            forces=np.zeros((3, 3)),
            e0=0.00,
            e1=0.03,
            u0=0.20,
            u1=0.10,
            f0=np.full((3, 3), 0.10),
            f1=np.full((3, 3), 0.05),
        ),
    ]
    _install_fake_sampling_runtime(monkeypatch, snapshots)
    monkeypatch.setattr(sampler_mod, "annealing_schedule", lambda *args: 100.0)

    molecule = MoleculesObject(
        Atoms(symbols=["O", "H", "H"], positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
        "traj_timing_ase",
    )
    result = sampler_mod.run_excited_state_sampling(
        molecule,
        sampler_config={
            "dt": 1.0,
            "maxt": 0.001,
            "Ncheck": 1,
            "min_time": 0.0,
            "friction": 0.02,
            "srt_temp": [100.0, 100.0],
            "end_temp": [100.0, 100.0],
            "amp_temp": [0.0, 0.0],
            "per_temp": [5.0, 5.0],
            "amp_dens": None,
            "per_dens": None,
            "end_dens": None,
            "meta_dir": str(tmp_path / "meta"),
            "metadata_format": "json",
            "write_traj_binary": False,
            "write_traj_xyz": False,
            "trajectory_frequency": 0.0,
            "trajectory_interval": 1,
            "min_distance_cutoff": 0.3,
            "max_force_cutoff": 10.0,
            "timing": {"enabled": True, "cuda_events": True, "record_chunk_timings": True},
            "state_selection": {"mode": "fixed", "fixed_state": 0},
            "uncertainty": {"enabled": True, "min_uE": 0.05, "min_uF": 0.05, "logic": "either"},
            "score": {"w_energy": 1.0, "w_force": 1.0, "w_gap": 0.0, "uncertainty_aggregate": "max"},
        },
        model_path=str(tmp_path / "models" / "model-{:04d}"),
        current_model_id=0,
        gpus_per_node=0,
        properties_list=_properties_list(),
    )

    timing = result.get_metadata()["timing"]
    assert result.get_metadata()["realtime_simulation"] >= 0.0
    assert timing["backend"] == "ase"
    assert timing["cuda_events_enabled"] is False
    assert timing["num_md_steps_completed"] == 2
    assert timing["num_md_chunks"] == 2
    assert timing["md_run_wall_s"] >= 0.0
    assert timing["model_eval_wall_s"] >= 0.0
    assert len(timing["chunk_timings"]) == 1
    assert timing["chunk_timings"][0]["chunk_steps"] == 1


def test_run_excited_state_sampling_returns_none_when_no_candidate_passes(monkeypatch, tmp_path: Path):
    snapshots = [
        _snapshot(
            energy=0.0,
            forces=np.zeros((3, 3)),
            e0=0.00,
            e1=0.04,
            u0=0.00,
            u1=0.00,
            f0=np.zeros((3, 3)),
            f1=np.zeros((3, 3)),
        ),
        _snapshot(
            energy=0.1,
            forces=np.zeros((3, 3)),
            e0=0.00,
            e1=0.03,
            u0=0.05,
            u1=0.04,
            f0=np.full((3, 3), 0.02),
            f1=np.full((3, 3), 0.01),
        ),
    ]
    _install_fake_sampling_runtime(monkeypatch, snapshots)
    monkeypatch.setattr(sampler_mod, "annealing_schedule", lambda *args: 100.0)

    molecule = MoleculesObject(
        Atoms(symbols=["O", "H", "H"], positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
        "traj_0001",
    )
    result = sampler_mod.run_excited_state_sampling(
        molecule,
        sampler_config={
            "dt": 1.0,
            "maxt": 0.001,
            "Ncheck": 1,
            "min_time": 0.0,
            "friction": 0.02,
            "srt_temp": [100.0, 100.0],
            "end_temp": [100.0, 100.0],
            "amp_temp": [0.0, 0.0],
            "per_temp": [5.0, 5.0],
            "amp_dens": None,
            "per_dens": None,
            "end_dens": None,
            "meta_dir": str(tmp_path / "meta"),
            "metadata_format": "json",
            "write_traj_binary": False,
            "write_traj_xyz": False,
            "trajectory_frequency": 0.0,
            "trajectory_interval": 1,
            "min_distance_cutoff": 0.3,
            "max_force_cutoff": 10.0,
            "state_selection": {"mode": "fixed", "fixed_state": 0},
            "uncertainty": {"enabled": True, "min_uE": 1.0, "min_uF": 1.0, "logic": "both"},
            "score": {"w_energy": 1.0, "w_force": 1.0, "w_gap": 0.0, "uncertainty_aggregate": "max"},
        },
        model_path=str(tmp_path / "models" / "model-{:04d}"),
        current_model_id=0,
        gpus_per_node=0,
        properties_list=_properties_list(),
    )

    assert result.get_atoms() is None
    assert result.get_metadata()["best_candidate"] is None


def test_alchemi_backend_requires_optional_dependency(monkeypatch, tmp_path: Path):
    def raise_missing_alchemi():
        raise ImportError("install nvalchemi-toolkit")

    fake_backend = types.ModuleType("alframework.samplers.alchemi_baoab_dynamics")
    fake_backend.ALFExcitedStateAlchemiModel = object
    fake_backend.AlchemiBaoabRunner = object
    fake_backend.alchemi_config_from_sampler = lambda *args, **kwargs: None
    fake_backend.build_alchemi_batch = lambda *args, **kwargs: None
    fake_backend.validate_alchemi_sampler_support = lambda *args, **kwargs: None
    fake_backend.ensure_alchemi_available = raise_missing_alchemi

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.device = lambda name: type("FakeDevice", (), {"type": str(name), "__str__": lambda self: str(name)})()
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, set_device=lambda device: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "alframework.samplers.alchemi_baoab_dynamics", fake_backend)
    monkeypatch.delitem(sys.modules, "nvalchemi", raising=False)

    molecule = MoleculesObject(
        Atoms(symbols=["O", "H", "H"], positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
        "traj_alchemi_missing",
    )
    with pytest.raises(ImportError, match="nvalchemi-toolkit"):
        sampler_mod.run_excited_state_sampling(
            molecule,
            sampler_config={
                "dynamics_backend": "alchemi_baoab",
                "dt": 1.0,
                "maxt": 0.001,
                "Ncheck": 1,
                "srt_temp": [100.0, 100.0],
                "end_temp": [100.0, 100.0],
                "amp_temp": [0.0, 0.0],
                "per_temp": [5.0, 5.0],
                "amp_dens": None,
                "per_dens": None,
                "end_dens": None,
                "alchemi_baoab": {"allow_cpu_debug": True},
            },
            model_path=str(tmp_path / "models" / "model-{:04d}"),
            current_model_id=0,
            gpus_per_node=0,
            properties_list=_properties_list(),
        )


def test_alchemi_backend_uses_fake_runner_not_ase_langevin(monkeypatch, tmp_path: Path):
    snapshots = [
        _snapshot(
            energy=0.0,
            forces=np.zeros((3, 3)),
            e0=0.00,
            e1=0.04,
            u0=0.00,
            u1=0.00,
            f0=np.zeros((3, 3)),
            f1=np.zeros((3, 3)),
        ),
        _snapshot(
            energy=0.2,
            forces=np.zeros((3, 3)),
            e0=0.00,
            e1=0.015,
            u0=0.40,
            u1=0.20,
            f0=np.full((3, 3), 0.20),
            f1=np.full((3, 3), 0.10),
        ),
    ]

    class FakeNode:
        def __init__(self, name):
            self.mean = f"{name}.mean"
            self.std = f"{name}.std"

    class FakeGraph:
        def node_from_name(self, name):
            return FakeNode(str(name))

    def fake_load_excited_state_ensemble(ensemble_directory, properties_list, device="cpu"):
        del ensemble_directory, properties_list, device
        return (
            FakeGraph(),
            [
                {"state": 0, "energy_node_base": "ensemble_sE0", "force_node_base": "ensemble_F0"},
                {"state": 1, "energy_node_base": "ensemble_sE1", "force_node_base": "ensemble_F1"},
            ],
            [],
        )

    class FakeAlchemiModel:
        init_count = 0

        def __init__(self, **kwargs):
            del kwargs
            self.__class__.init_count += 1
            self.energy_nodes = []

        def set_energy_node(self, energy_node, **kwargs):
            del kwargs
            self.energy_nodes.append(energy_node)

        def results_from_batch(self, batch):
            return deepcopy(snapshots[min(batch.index, len(snapshots) - 1)])

    class FakeAlchemiRunner:
        instances = []

        def __init__(self, *, model, batch, dt_fs, temperature_K, friction_per_fs, random_seed, device):
            del dt_fs, temperature_K, friction_per_fs, random_seed, device
            self.model = model
            self.batch = batch
            self.nsteps = 0
            self.callbacks = []
            self.initialized_forces = True
            self.results = self.model.results_from_batch(self.batch)
            self.__class__.instances.append(self)

        def attach(self, callback, interval=1):
            self.callbacks.append((callback, int(interval)))

        def set_temperature(self, *, temperature_K):
            self.last_temperature = float(temperature_K)

        def run(self, steps):
            for _ in range(int(steps)):
                self.nsteps += 1
                self.batch.index = min(self.batch.index + 1, len(snapshots) - 1)
                for callback, interval in self.callbacks:
                    if self.nsteps % interval == 0:
                        callback()

        def evaluate_results(self):
            self.results = self.model.results_from_batch(self.batch)
            return deepcopy(self.results)

        def sync_to_atoms(self, atoms):
            atoms.set_positions(atoms.get_positions() + 0.01)

        def kinetic_energy_eV(self):
            return 0.0

        def temperature_K(self):
            return 100.0

    fake_backend = types.ModuleType("alframework.samplers.alchemi_baoab_dynamics")
    fake_backend.ALFExcitedStateAlchemiModel = FakeAlchemiModel
    fake_backend.AlchemiBaoabRunner = FakeAlchemiRunner
    fake_backend.ensure_alchemi_available = lambda: None
    fake_backend.alchemi_config_from_sampler = lambda sampler_config, default_seed: types.SimpleNamespace(
        random_seed=default_seed,
        allow_cpu_debug=True,
        strict_gpu=False,
        species_key="species",
        coordinates_key="coordinates",
    )
    fake_backend.validate_alchemi_sampler_support = lambda atoms, feed, device, config: None
    fake_backend.build_alchemi_batch = lambda atoms, device: types.SimpleNamespace(index=0)

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.device = lambda name: type("FakeDevice", (), {"type": str(name), "__str__": lambda self: str(name)})()
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, set_device=lambda device: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "alframework.samplers.alchemi_baoab_dynamics", fake_backend)
    monkeypatch.setattr(sampler_mod, "load_excited_state_ensemble", fake_load_excited_state_ensemble)
    monkeypatch.setattr(sampler_mod, "annealing_schedule", lambda *args: 100.0)
    monkeypatch.setattr(
        sampler_mod,
        "Langevin",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ASE Langevin should not run")),
    )

    molecule = MoleculesObject(
        Atoms(symbols=["O", "H", "H"], positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
        "traj_alchemi_fake",
    )
    result = sampler_mod.run_excited_state_sampling(
        molecule,
        sampler_config={
            "dynamics_backend": "alchemi_baoab",
            "dt": 1.0,
            "maxt": 0.001,
            "Ncheck": 1,
            "min_time": 0.0,
            "friction": 0.02,
            "srt_temp": [100.0, 100.0],
            "end_temp": [100.0, 100.0],
            "amp_temp": [0.0, 0.0],
            "per_temp": [5.0, 5.0],
            "amp_dens": None,
            "per_dens": None,
            "end_dens": None,
            "meta_dir": str(tmp_path / "meta"),
            "metadata_format": "json",
            "write_traj_binary": False,
            "write_traj_xyz": False,
            "trajectory_frequency": 0.0,
            "trajectory_interval": 1,
            "timing": {"enabled": True},
            "min_distance_cutoff": 0.3,
            "max_force_cutoff": 10.0,
            "alchemi_baoab": {"allow_cpu_debug": True},
            "state_selection": {"mode": "fixed", "fixed_state": 0},
            "uncertainty": {"enabled": True, "min_uE": 0.05, "min_uF": 0.05, "logic": "either"},
            "score": {"w_energy": 1.0, "w_force": 1.0, "w_gap": 0.0, "uncertainty_aggregate": "max"},
        },
        model_path=str(tmp_path / "models" / "model-{:04d}"),
        current_model_id=0,
        gpus_per_node=0,
        properties_list=_properties_list(),
    )

    assert result.get_atoms() is not None
    assert FakeAlchemiModel.init_count == 1
    assert FakeAlchemiRunner.instances[0].initialized_forces is True
    assert result.get_metadata()["best_candidate"]["step"] == 1
    timing = result.get_metadata()["timing"]
    assert timing["backend"] == "alchemi_baoab"
    assert timing["num_md_steps_completed"] == 2
    assert timing["num_md_chunks"] == 2
    assert timing["full_sync_count"] == 2
    assert timing["scalar_sync_count"] >= 3
    assert timing["md_run_wall_s"] >= 0.0


def test_batched_alchemi_backend_scores_multiple_molecules(monkeypatch, tmp_path: Path):
    class FakeNode:
        def __init__(self, name):
            self.mean = f"{name}.mean"
            self.std = f"{name}.std"

    class FakeGraph:
        def node_from_name(self, name):
            return FakeNode(str(name))

    def fake_load_excited_state_ensemble(ensemble_directory, properties_list, device="cpu"):
        del ensemble_directory, properties_list, device
        return (
            FakeGraph(),
            [
                {"state": 0, "energy_node_base": "ensemble_sE0", "force_node_base": "ensemble_F0"},
                {"state": 1, "energy_node_base": "ensemble_sE1", "force_node_base": "ensemble_F1"},
            ],
            [],
        )

    class FakeAlchemiModel:
        def __init__(self, **kwargs):
            del kwargs

        def set_energy_node(self, *args, **kwargs):
            del args, kwargs

    class FakeBatch:
        def __init__(self, atoms_list):
            self.atoms_list = [atoms.copy() for atoms in atoms_list]
            self.num_graphs = len(atoms_list)
            self.num_nodes = sum(len(atoms) for atoms in atoms_list)

    class FakeAlchemiRunner:
        def __init__(self, *, model, batch, dt_fs, temperature_K, friction_per_fs, random_seed, device):
            del model, dt_fs, temperature_K, friction_per_fs, random_seed, device
            self.batch = batch
            self.nsteps = 0
            self.callbacks = []

        def attach(self, callback, interval=1):
            self.callbacks.append((callback, int(interval)))

        def set_temperature(self, *, temperature_K):
            self.temperature_K_target = temperature_K

        def run(self, steps):
            for _ in range(int(steps)):
                self.nsteps += 1
                for atoms in self.batch.atoms_list:
                    atoms.set_positions(atoms.get_positions() + 0.01)
                for callback, interval in self.callbacks:
                    if self.nsteps % interval == 0:
                        callback()

        def evaluate_results(self):
            return {"energy": np.zeros((self.batch.num_graphs, 1)), "forces": np.zeros((self.batch.num_nodes, 3))}

        def results_for_graph(self, graph_index):
            uncertainty = 0.20 + 0.10 * int(graph_index)
            forces = np.zeros((3, 3), dtype=float)
            return {
                "energy": float(graph_index),
                "forces": forces,
                "E_mean_S0": 0.0,
                "E_std_S0": uncertainty,
                "F_std_S0": np.full((3, 3), uncertainty),
                "E_mean_S1": 0.03,
                "E_std_S1": 10.0,
                "F_std_S1": np.full((3, 3), 10.0),
            }

        def sync_graph_to_atoms(self, graph_index, atoms):
            atoms.set_positions(self.batch.atoms_list[int(graph_index)].get_positions())

        def kinetic_energy_eV(self):
            return np.zeros(self.batch.num_graphs)

        def temperature_K(self):
            return np.full(self.batch.num_graphs, 100.0)

    fake_backend = types.ModuleType("alframework.samplers.alchemi_baoab_dynamics")
    fake_backend.ALFExcitedStateAlchemiModel = FakeAlchemiModel
    fake_backend.AlchemiBaoabRunner = FakeAlchemiRunner
    fake_backend.ensure_alchemi_available = lambda: None
    fake_backend.alchemi_config_from_sampler = lambda sampler_config, default_seed: types.SimpleNamespace(
        random_seed=default_seed,
        allow_cpu_debug=True,
        strict_gpu=False,
        scalar_sync_policy="control_only",
        species_key="species",
        coordinates_key="coordinates",
        batch_size=2,
        allow_partial_batches=False,
        require_same_selected_state=True,
        batched_gap_switch_policy="global_lowest_gap_trigger",
    )
    fake_backend.validate_alchemi_sampler_support = lambda atoms, feed, device, config: None
    fake_backend.build_alchemi_batch_from_atoms_list = lambda atoms_list, device: FakeBatch(atoms_list)

    monkeypatch.setitem(sys.modules, "alframework.samplers.alchemi_baoab_dynamics", fake_backend)
    monkeypatch.setattr(sampler_mod, "load_excited_state_ensemble", fake_load_excited_state_ensemble)
    monkeypatch.setattr(sampler_mod, "annealing_schedule", lambda *args: 100.0)

    molecules = []
    for idx, shift in enumerate([0.0, 0.2]):
        molecule = MoleculesObject(
            Atoms(symbols=["O", "H", "H"], positions=[[shift, 0, 0], [0.96 + shift, 0, 0], [-0.24 + shift, 0.93, 0]]),
            f"batch_{idx}",
        )
        molecule.update_metadata({"excited_state": 0})
        molecules.append(molecule)

    result = sampler_mod.run_excited_state_sampling_batch(
        molecules,
        sampler_config={
            "dynamics_backend": "alchemi_baoab",
            "dt": 1.0,
            "maxt": 0.001,
            "Ncheck": 1,
            "min_time": 0.0,
            "friction": 0.02,
            "srt_temp": [100.0, 100.0],
            "end_temp": [100.0, 100.0],
            "amp_temp": [0.0, 0.0],
            "per_temp": [5.0, 5.0],
            "amp_dens": None,
            "per_dens": None,
            "end_dens": None,
            "meta_dir": str(tmp_path / "meta"),
            "metadata_format": "json",
            "write_traj_binary": False,
            "write_traj_xyz": False,
            "trajectory_frequency": 0.0,
            "trajectory_interval": 1,
            "timing": {"enabled": True},
            "return_top_n": 2,
            "min_distance_cutoff": 0.3,
            "max_force_cutoff": 10.0,
            "alchemi_baoab": {"allow_cpu_debug": True, "batch_size": 2},
            "state_selection": {"mode": "fixed", "fixed_state": 0},
            "uncertainty": {"enabled": True, "min_uE": 0.05, "min_uF": 0.05, "logic": "either"},
            "gap_seeking": {"enabled": False},
            "udd": {"enabled": False},
            "score": {"w_energy": 1.0, "w_force": 1.0, "w_gap": 0.0, "uncertainty_aggregate": "max"},
        },
        model_path=str(tmp_path / "models" / "model-{:04d}"),
        current_model_id=0,
        gpus_per_node=0,
        properties_list=_properties_list(),
    )

    assert len(result) == 2
    assert {item.get_metadata()["parent_molecule_id"] for item in result} == {"batch_0", "batch_1"}
    assert all(item.get_metadata()["batch_size"] == 2 for item in result)
    assert all(
        item.get_metadata()["score_components"]["energy_uncertainty"] == pytest.approx(item.get_metadata()["uE_selected"])
        for item in result
    )
    assert all(
        item.get_metadata()["score_components"]["force_uncertainty"] == pytest.approx(item.get_metadata()["uF_selected"])
        for item in result
    )
    payload = json.loads((tmp_path / "meta" / "metadata-batch_0.json").read_text(encoding="utf-8"))
    assert payload["batch_size"] == 2
    assert payload["timing"]["batch_size"] == 2
