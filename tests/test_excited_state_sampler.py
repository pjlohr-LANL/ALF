from __future__ import annotations

import json
import sys
import types
from copy import deepcopy
from pathlib import Path

import numpy as np
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
    assert result.get_metadata()["best_candidate"]["step"] == 2
    assert schedule_calls[0] == (0.0, 0.002, 0.0, 5.0, 100.0, 200.0)
    assert fake_langevin.instances[0].run_calls == [1, 1, 1]
    meta_path = tmp_path / "meta" / "metadata-traj_0000.json"
    assert meta_path.exists()
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["best_candidate"]["score"] > 0.0


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
