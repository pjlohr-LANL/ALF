from __future__ import annotations

import sys
import types
from copy import deepcopy
from pathlib import Path

import numpy as np
import parsl
import pytest
from ase.calculators.calculator import Calculator, all_changes
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor

from alframework.builders.excited_state_builder import excited_state_replay_builder_task
from alframework.ml_interfaces import excited_state_hippynn_interface as ml_mod
from alframework.ml_interfaces.excited_state_hippynn_interface import (
    train_excited_state_HIPPYNN_ensemble_task,
)
from alframework.qm_interfaces import pyseqm_interface
from alframework.qm_interfaces.pyseqm_interface import pyseqm_excited_state_task
from alframework.samplers import excited_state_sampling as sampler_mod
from alframework.samplers.excited_state_sampling import excited_state_sampling_task
from alframework.tools.molecule_payloads import flatten_molecule_output
from alframework.tools.tools import parsl_task_queue, store_current_data
from tests.test_excited_state_builder import _write_seed_dataset


def _properties_list() -> dict[str, list[object]]:
    return {
        "sE0": ["sE0", "system", 1.0],
        "F0": ["F0", "atomic", 1.0],
        "sE1": ["sE1", "system", 1.0],
        "F1": ["F1", "atomic", 1.0],
    }


@pytest.fixture
def local_parsl():
    try:
        parsl.dfk().cleanup()
    except Exception:
        pass
    try:
        parsl.clear()
    except Exception:
        pass
    parsl.load(
        Config(
            executors=[
                ThreadPoolExecutor(label="alf_sampler_executor", max_threads=4),
                ThreadPoolExecutor(label="alf_QM_executor", max_threads=2),
                ThreadPoolExecutor(label="alf_ML_executor", max_threads=2),
            ],
            strategy=None,
        )
    )
    yield
    try:
        parsl.dfk().cleanup()
    finally:
        parsl.clear()


def test_async_queue_handoff_builder_sampler_qm_h5_ml(monkeypatch, tmp_path: Path, local_parsl):
    seed_dir = tmp_path / "seed"
    _write_seed_dataset(seed_dir)
    properties_list = _properties_list()

    snapshots = [
        {
            "energy": 0.0,
            "forces": [[[0.0, 0.0, 0.0]] * 3],
            "E_mean_S0": 0.0,
            "E_std_S0": 0.0,
            "F_std_S0": [[[0.0, 0.0, 0.0]] * 3],
            "E_mean_S1": 0.04,
            "E_std_S1": 0.0,
            "F_std_S1": [[[0.0, 0.0, 0.0]] * 3],
        },
        {
            "energy": 0.1,
            "forces": [[[0.0, 0.0, 0.0]] * 3],
            "E_mean_S0": 0.0,
            "E_std_S0": 0.4,
            "F_std_S0": [[[0.2, 0.0, 0.0]] * 3],
            "E_mean_S1": 0.015,
            "E_std_S1": 0.2,
            "F_std_S1": [[[0.1, 0.0, 0.0]] * 3],
        },
    ]
    fake_snapshots = [deepcopy(s) for s in snapshots]

    class FakeHippynnCalculator(Calculator):
        implemented_properties = ["energy", "forces"]
        snapshots = fake_snapshots

        def __init__(self, *args, **kwargs):
            del args, kwargs
            super().__init__()
            self._index = 0
            self.results = {}

        def advance(self):
            idx = min(self._index, len(self.snapshots) - 1)
            current = deepcopy(self.snapshots[idx])
            current["forces"] = np.asarray(current["forces"], dtype=float)
            current["F_std_S0"] = np.asarray(current["F_std_S0"], dtype=float)
            current["F_std_S1"] = np.asarray(current["F_std_S1"], dtype=float)
            self.results = current
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
            return FakeNode(name)

    class FakeLangevin:
        def __init__(self, atoms, timestep, friction, temperature_K):
            del timestep, friction, temperature_K
            self.atoms = atoms

        def set_temperature(self, temperature_K):
            del temperature_K

        def run(self, steps):
            del steps
            self.atoms.calc.advance()

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

    def fake_train_excited_state_ensemble(**kwargs):
        model_dir = Path(str(kwargs["model_path"]).format(int(kwargs["current_training_id"]))) / "model-00"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "training_log.txt").write_text("Training complete\n", encoding="utf-8")
        return [True], int(kwargs["current_training_id"])

    monkeypatch.setattr(sampler_mod, "load_excited_state_ensemble", fake_load_excited_state_ensemble)
    monkeypatch.setattr(sampler_mod, "Langevin", FakeLangevin)
    monkeypatch.setattr(sampler_mod, "annealing_schedule", lambda *args: 100.0)
    monkeypatch.setattr(ml_mod, "train_excited_state_ensemble", fake_train_excited_state_ensemble)
    monkeypatch.setattr(
        pyseqm_interface,
        "run_pyseqm_batch",
        lambda **kwargs: (
            np.array([[-90.0, -89.5]], dtype=np.float64),
            np.zeros((1, 2, 3, 3), dtype=np.float64),
        ),
    )

    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.device = lambda name: name
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, set_device=lambda device: None)
    fake_hippynn = types.ModuleType("hippynn")
    fake_interfaces = types.ModuleType("hippynn.interfaces")
    fake_ase_interface = types.ModuleType("hippynn.interfaces.ase_interface")
    fake_ase_interface.HippynnCalculator = FakeHippynnCalculator
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "hippynn", fake_hippynn)
    monkeypatch.setitem(sys.modules, "hippynn.interfaces", fake_interfaces)
    monkeypatch.setitem(sys.modules, "hippynn.interfaces.ase_interface", fake_ase_interface)

    builder_queue = parsl_task_queue()
    sampler_queue = parsl_task_queue()
    qm_queue = parsl_task_queue()
    ml_queue = parsl_task_queue()

    builder_queue.add_task(
        excited_state_replay_builder_task(
            moleculeid="traj_0000",
            builder_config={
                "seed_dataset_dir": str(seed_dir),
                "source_priority": "seed_only",
                "state_selection": {"mode": "fixed", "fixed_state": 0},
            },
            properties_list=properties_list,
            h5_path=str(tmp_path / "h5store" / "data-{:04d}.h5"),
            current_h5_id=0,
        )
    )
    builder_queue.task_list[0].result()
    builder_results, _ = builder_queue.get_task_results()
    molecule = builder_results[0]

    sampler_queue.add_task(
        excited_state_sampling_task(
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
                "meta_dir": str(tmp_path / "sampling"),
                "metadata_format": "json",
                "write_traj_binary": False,
                "write_traj_xyz": False,
                "trajectory_frequency": 0.0,
                "trajectory_interval": 1,
                "min_distance_cutoff": 0.3,
                "max_force_cutoff": 10.0,
                "state_selection": {"mode": "fixed", "fixed_state": 0},
                "uncertainty": {"enabled": True, "min_uE": 0.05, "min_uF": 0.05, "logic": "either"},
                "score": {"w_energy": 1.0, "w_force": 1.0, "w_gap": 0.0, "uncertainty_aggregate": "max"},
            },
            model_path=str(tmp_path / "models" / "model-{:04d}"),
            current_model_id=0,
            gpus_per_node=0,
            properties_list=properties_list,
        )
    )
    sampler_queue.task_list[0].result()
    sampler_results, _ = sampler_queue.get_task_results()
    sampled = flatten_molecule_output(sampler_results[0])[0]
    assert sampled.get_atoms() is not None

    qm_queue.add_task(
        pyseqm_excited_state_task(
            sampled,
            QM_config={"method": "AM1"},
            properties_list=properties_list,
            sampler_config={"energy_offset_eV": -100.0},
            gpus_per_node=0,
        )
    )
    qm_queue.task_list[0].result()
    qm_results, _ = qm_queue.get_task_results()
    labeled = qm_results[0]
    assert labeled.check_convergence() is True

    h5_path = tmp_path / "h5store" / "data-0000.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    store_current_data(str(h5_path), [labeled], properties_list)
    assert h5_path.exists()

    ml_queue.add_task(
        train_excited_state_HIPPYNN_ensemble_task(
            ML_config={"n_models": 1},
            h5_dir=str(h5_path.parent),
            model_path=str(tmp_path / "models" / "model-{:04d}"),
            current_training_id=1,
            gpus_per_node=0,
            properties_list=properties_list,
        )
    )
    ml_queue.task_list[0].result()
    ml_results, _ = ml_queue.get_task_results()
    completed, training_id = ml_results[0]
    assert completed == [True]
    assert training_id == 1
    assert (tmp_path / "models" / "model-0001" / "model-00" / "training_log.txt").exists()
