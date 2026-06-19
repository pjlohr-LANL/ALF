from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from alframework.ml_interfaces import excited_state_hippynn_interface as ml_mod
from alframework.qm_interfaces import pyseqm_interface
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.qm_batching import pyseqm_batch_size_from_qm_config


def _properties_list() -> dict[str, list[object]]:
    return {
        "sE0": ["sE0", "system", 1.0],
        "F0": ["F0", "atomic", 1.0],
        "sE1": ["sE1", "system", 1.0],
        "F1": ["F1", "atomic", 1.0],
    }


def test_pyseqm_batch_size_comes_from_qm_config_only():
    def batch_capable_task(molecule_objects=None):
        return molecule_objects

    def single_molecule_task(molecule_object=None):
        return molecule_object

    assert pyseqm_batch_size_from_qm_config({"pyseqm_batch_size": 16}, batch_capable_task) == 16
    assert pyseqm_batch_size_from_qm_config({}, batch_capable_task) == 1
    assert pyseqm_batch_size_from_qm_config({"batch_size": 16}, batch_capable_task) == 1
    assert pyseqm_batch_size_from_qm_config({"pyseqm_batch_size": 16}, single_molecule_task) == 1


def test_label_excited_state_molecule_flattens_outputs_and_applies_offset(monkeypatch):
    atoms = Atoms(symbols=["O", "H", "H"], positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    molecule = MoleculesObject(atoms, "traj_0000")
    energies = np.array([[-90.0, -89.5]], dtype=np.float64)
    forces = np.arange(18, dtype=np.float64).reshape(1, 2, 3, 3)

    def fake_run_pyseqm_batch(**kwargs):
        assert kwargs["n_states"] == 2
        return energies.copy(), forces.copy()

    monkeypatch.setattr(pyseqm_interface, "run_pyseqm_batch", fake_run_pyseqm_batch)
    monkeypatch.setattr(
        pyseqm_interface.multiprocessing,
        "get_context",
        lambda method: (_ for _ in ()).throw(AssertionError(f"timeout context should not be used: {method}")),
    )

    labeled = pyseqm_interface.label_excited_state_molecule(
        molecule,
        QM_config={"method": "AM1", "max_solve_time_seconds": 0},
        properties_list=_properties_list(),
        sampler_config={"energy_offset_eV": -100.0},
        gpus_per_node=0,
    )

    assert labeled.check_convergence() is True
    assert labeled.get_metadata()["qm_backend"] == "pyseqm"
    np.testing.assert_allclose(labeled.get_results()["sE0"], 10.0)
    np.testing.assert_allclose(labeled.get_results()["sE1"], 10.5)
    np.testing.assert_allclose(labeled.get_results()["F0"], forces[0, 0])
    np.testing.assert_allclose(labeled.get_results()["F1"], forces[0, 1])


def test_run_pyseqm_batch_sorts_species_descending_and_restores_force_order(monkeypatch):
    captured = {}

    class FakeTensor:
        def __init__(self, array):
            self.array = np.asarray(array)
            self.device = None

        def pin_memory(self):
            return self

        def to(self, device, non_blocking=False):
            del non_blocking
            self.device = device
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self.array)

        def unsqueeze(self, axis):
            return FakeTensor(np.expand_dims(self.array, axis))

        def __getitem__(self, item):
            return FakeTensor(self.array[item])

        def __setitem__(self, item, value):
            if isinstance(value, FakeTensor):
                value = value.array
            self.array[item] = value

        def __add__(self, other):
            if isinstance(other, FakeTensor):
                other = other.array
            return FakeTensor(self.array + other)

    class FakeTorch:
        float64 = "float64"

        @staticmethod
        def set_default_dtype(dtype):
            captured["dtype"] = dtype

        @staticmethod
        def from_numpy(array):
            return FakeTensor(np.asarray(array))

        @staticmethod
        def empty(shape, dtype=None, device=None):
            del dtype, device
            return FakeTensor(np.empty(shape, dtype=np.float64))

        @staticmethod
        def device(name):
            return type("FakeDevice", (), {"type": str(name).split(":")[0], "__str__": lambda self: str(name)})()

        class cuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def empty_cache():
                return None

    class FakeConstants:
        def to(self, device):
            captured["constants_device"] = device
            return self

    class FakeMolecule:
        def __init__(self, const, params, coords, species):
            del const, params
            captured["coords_seen_by_pyseqm"] = np.asarray(coords.array)
            captured["species_seen_by_pyseqm"] = np.asarray(species.array)
            self.nmol = coords.array.shape[0]
            self.Etot = FakeTensor(np.array([-10.0], dtype=np.float64))
            self.cis_energies = FakeTensor(np.array([[0.5]], dtype=np.float64))
            self.all_forces = FakeTensor(
                np.array(
                    [
                        [
                            [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
                            [[11.0, 0.0, 0.0], [21.0, 0.0, 0.0], [31.0, 0.0, 0.0]],
                        ]
                    ],
                    dtype=np.float64,
                )
            )

        def to(self, device):
            captured["molecule_device"] = device
            return self

    class FakeElectronicStructure:
        def __init__(self, params):
            captured["driver_params"] = params

        def to(self, device):
            captured["driver_device"] = device
            return self

        def __call__(self, molecule):
            captured["driver_called"] = molecule.nmol

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "seqm.seqm_functions.constants", type("M", (), {"Constants": FakeConstants}))
    monkeypatch.setitem(sys.modules, "seqm.Molecule", type("M", (), {"Molecule": FakeMolecule}))
    monkeypatch.setitem(
        sys.modules,
        "seqm.ElectronicStructure",
        type("M", (), {"Electronic_Structure": FakeElectronicStructure}),
    )

    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=np.float64)
    species = np.array([[1, 8, 6]], dtype=np.int64)

    energies, forces = pyseqm_interface.run_pyseqm_batch(
        coords_np=coords,
        species_np=species,
        n_states=2,
        method="AM1",
        scf_eps=1e-10,
        cis_tol=1e-8,
        device=FakeTorch.device("cpu"),
    )

    np.testing.assert_array_equal(captured["species_seen_by_pyseqm"], np.array([[8, 6, 1]], dtype=np.int64))
    np.testing.assert_allclose(
        captured["coords_seen_by_pyseqm"],
        np.array([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float64),
    )
    np.testing.assert_allclose(energies, np.array([[-10.0, -9.5]], dtype=np.float64))
    np.testing.assert_allclose(
        forces,
        np.array(
            [
                [
                    [[30.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                    [[31.0, 0.0, 0.0], [11.0, 0.0, 0.0], [21.0, 0.0, 0.0]],
                ]
            ],
            dtype=np.float64,
        ),
    )


def test_label_excited_state_molecule_handles_backend_failure(monkeypatch):
    atoms = Atoms(symbols=["H", "H"], positions=[[0, 0, 0], [0, 0, 0.75]])
    molecule = MoleculesObject(atoms, "traj_0001")

    def fake_run_pyseqm_batch(**kwargs):
        del kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(pyseqm_interface, "run_pyseqm_batch", fake_run_pyseqm_batch)

    labeled = pyseqm_interface.label_excited_state_molecule(
        molecule,
        QM_config={"method": "AM1"},
        properties_list=_properties_list(),
        sampler_config={"energy_offset_eV": 0.0},
        gpus_per_node=0,
    )

    assert labeled.check_convergence() is False
    assert "qm_error" in labeled.get_metadata()
    assert labeled.get_results() == {}


def test_label_excited_state_molecules_batches_and_splits_outputs(monkeypatch):
    molecules = [
        MoleculesObject(Atoms(symbols=["O", "H", "H"], positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]), "traj_0000"),
        MoleculesObject(Atoms(symbols=["O", "H", "H"], positions=[[0, 0, 0], [1.1, 0, 0], [0, 1.1, 0]]), "traj_0001"),
    ]
    energies = np.array([[-90.0, -89.5], [-91.0, -90.25]], dtype=np.float64)
    forces = np.arange(36, dtype=np.float64).reshape(2, 2, 3, 3)
    calls = []

    def fake_run_pyseqm_batch(**kwargs):
        calls.append((kwargs["coords_np"].shape, kwargs["species_np"].shape))
        assert kwargs["n_states"] == 2
        return energies.copy(), forces.copy()

    monkeypatch.setattr(pyseqm_interface, "run_pyseqm_batch", fake_run_pyseqm_batch)
    monkeypatch.setattr(pyseqm_interface, "_pyseqm_device", lambda gpus_per_node: None)

    labeled = pyseqm_interface.label_excited_state_molecules(
        molecules,
        QM_config={"method": "AM1"},
        properties_list=_properties_list(),
        sampler_config={"energy_offset_eV": -100.0},
        gpus_per_node=0,
    )

    assert calls == [((2, 3, 3), (2, 3))]
    assert [item.check_convergence() for item in labeled] == [True, True]
    assert [item.get_metadata()["pyseqm_batch_size"] for item in labeled] == [2, 2]
    np.testing.assert_allclose(labeled[0].get_results()["sE0"], 10.0)
    np.testing.assert_allclose(labeled[1].get_results()["sE1"], 9.75)
    np.testing.assert_allclose(labeled[0].get_results()["F0"], forces[0, 0])
    np.testing.assert_allclose(labeled[1].get_results()["F1"], forces[1, 1])


def test_label_excited_state_molecules_retries_individually_after_batch_failure(monkeypatch):
    molecules = [
        MoleculesObject(Atoms(symbols=["O", "H", "H"], positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]), "traj_0000"),
        MoleculesObject(Atoms(symbols=["O", "H", "H"], positions=[[0, 0, 0], [1.1, 0, 0], [0, 1.1, 0]]), "traj_0001"),
    ]
    calls = []

    def fake_run_pyseqm_batch(**kwargs):
        batch_size = int(kwargs["coords_np"].shape[0])
        calls.append(batch_size)
        if batch_size > 1:
            raise RuntimeError("synthetic batch failure")
        return np.array([[-90.0, -89.5]], dtype=np.float64), np.zeros((1, 2, 3, 3), dtype=np.float64)

    monkeypatch.setattr(pyseqm_interface, "run_pyseqm_batch", fake_run_pyseqm_batch)
    monkeypatch.setattr(pyseqm_interface, "_pyseqm_device", lambda gpus_per_node: None)

    labeled = pyseqm_interface.label_excited_state_molecules(
        molecules,
        QM_config={"method": "AM1"},
        properties_list=_properties_list(),
        sampler_config={"energy_offset_eV": -100.0},
        gpus_per_node=0,
    )

    assert calls == [2, 1, 1]
    assert [item.check_convergence() for item in labeled] == [True, True]
    assert all("pyseqm_batch_size" not in item.get_metadata() for item in labeled)


def test_label_excited_state_molecule_uses_timeout_child_and_succeeds(monkeypatch, tmp_path: Path):
    atoms = Atoms(symbols=["O", "H", "H"], positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    molecule = MoleculesObject(atoms, "traj_timeout_success")
    energies = np.array([[-90.0, -89.5]], dtype=np.float64)
    forces = np.zeros((1, 2, 3, 3), dtype=np.float64)

    class FakeQueue:
        def __init__(self, maxsize=0):
            del maxsize
            self.payload = None

        def put(self, payload):
            self.payload = payload

        def get_nowait(self):
            if self.payload is None:
                raise pyseqm_interface.queue.Empty
            return self.payload

    class InlineProcess:
        def __init__(self, *, target, args):
            self.target = target
            self.args = args
            self.exitcode = None

        def start(self):
            self.target(*self.args)
            self.exitcode = 0

        def join(self, timeout=None):
            self.join_timeout = timeout

        def is_alive(self):
            return False

        def terminate(self):
            raise AssertionError("successful child should not be terminated")

        def kill(self):
            raise AssertionError("successful child should not be killed")

    class InlineContext:
        Queue = FakeQueue
        Process = InlineProcess

    def fake_run_pyseqm_batch(**kwargs):
        assert kwargs["log_handle"] is not None
        assert kwargs["n_states"] == 2
        return energies.copy(), forces.copy()

    monkeypatch.setattr(pyseqm_interface, "run_pyseqm_batch", fake_run_pyseqm_batch)
    monkeypatch.setattr(pyseqm_interface, "_pyseqm_device", lambda gpus_per_node: None)
    monkeypatch.setattr(pyseqm_interface.multiprocessing, "get_context", lambda method: InlineContext())

    labeled = pyseqm_interface.label_excited_state_molecule(
        molecule,
        QM_config={
            "method": "AM1",
            "max_solve_time_seconds": 60,
            "capture_pyseqm_logs": True,
            "pyseqm_log_dir": str(tmp_path),
        },
        properties_list=_properties_list(),
        sampler_config={"energy_offset_eV": -100.0},
        gpus_per_node=0,
    )

    assert labeled.check_convergence() is True
    assert labeled.get_metadata()["qm_backend"] == "pyseqm"
    assert "qm_timeout" not in labeled.get_metadata()
    log_text = next(tmp_path.glob("traj_timeout_success*.log")).read_text(encoding="utf-8")
    assert "status: success" in log_text


def test_label_excited_state_molecule_times_out_child(monkeypatch, tmp_path: Path):
    atoms = Atoms(symbols=["H", "H"], positions=[[0, 0, 0], [0, 0, 0.75]])
    molecule = MoleculesObject(atoms, "traj_timeout_failure")

    class EmptyQueue:
        def __init__(self, maxsize=0):
            del maxsize

        def get_nowait(self):
            raise pyseqm_interface.queue.Empty

    class HangingProcess:
        terminated = False

        def __init__(self, *, target, args):
            del target, args
            self.exitcode = None
            self._alive = True

        def start(self):
            self._alive = True

        def join(self, timeout=None):
            self.join_timeout = timeout

        def is_alive(self):
            return self._alive

        def terminate(self):
            self.__class__.terminated = True
            self._alive = False
            self.exitcode = -15

        def kill(self):
            self._alive = False
            self.exitcode = -9

    class HangingContext:
        Queue = EmptyQueue
        Process = HangingProcess

    monkeypatch.setattr(pyseqm_interface, "_pyseqm_device", lambda gpus_per_node: None)
    monkeypatch.setattr(pyseqm_interface.multiprocessing, "get_context", lambda method: HangingContext())

    labeled = pyseqm_interface.label_excited_state_molecule(
        molecule,
        QM_config={
            "method": "AM1",
            "max_solve_time_seconds": 0.01,
            "capture_pyseqm_logs": True,
            "pyseqm_log_dir": str(tmp_path),
        },
        properties_list=_properties_list(),
        sampler_config={"energy_offset_eV": 0.0},
        gpus_per_node=0,
    )

    assert HangingProcess.terminated is True
    assert labeled.check_convergence() is False
    assert labeled.get_metadata()["qm_timeout"] is True
    assert "exceeded" in labeled.get_metadata()["qm_error"]
    assert labeled.get_metadata()["max_solve_time_seconds"] == pytest.approx(0.01)
    log_text = next(tmp_path.glob("traj_timeout_failure*.log")).read_text(encoding="utf-8")
    assert "status: timeout" in log_text


def test_train_excited_state_ensemble_returns_contract_and_writes_errors(monkeypatch, tmp_path: Path):
    class FakePool:
        def __init__(self, processes):
            self.processes = processes

        def map(self, func, params_list):
            return [func(params) for params in params_list]

        def close(self):
            pass

        def join(self):
            pass

    def fake_worker(arg_dict):
        model_dir = Path(arg_dict["model_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        if int(arg_dict["model_id"]) == 0:
            (model_dir / "training_log.txt").write_text("Training complete\n", encoding="utf-8")
            return {"ok": True, "payload": {"model_id": 0, "model_dir": str(model_dir)}}
        return {
            "ok": False,
            "payload": {"model_id": 1, "model_dir": str(model_dir), "error": "synthetic failure"},
        }

    monkeypatch.setattr(ml_mod.multiprocessing, "Pool", FakePool)
    monkeypatch.setattr(ml_mod, "_excited_state_training_worker", fake_worker)

    completed, training_id = ml_mod.train_excited_state_ensemble(
        ML_config={"n_models": 2},
        h5_dir=str(tmp_path / "h5store"),
        model_path=str(tmp_path / "models" / "model-{:04d}"),
        current_training_id=3,
        gpus_per_node=0,
        properties_list=_properties_list(),
    )

    assert completed == [True, False]
    assert training_id == 3
    assert (tmp_path / "models" / "model-0003" / "model-00" / "training_log.txt").exists()
    error_path = tmp_path / "models" / "model-0003" / "model-01" / "training_error.json"
    assert error_path.exists()
    assert "synthetic failure" in error_path.read_text(encoding="utf-8")
