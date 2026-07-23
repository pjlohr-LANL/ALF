import builtins
import queue
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from ase import Atoms

import alframework.qm_interfaces.pyseqm_interface as pyseqm_module
from alframework.qm_interfaces.pyseqm_interface import (
    PySEQMConvergenceError,
    PySEQMTimeoutError,
    _prepare_pyseqm_inputs,
    _require_scf_convergence,
    _restore_pyseqm_force_order,
    _run_pyseqm_with_timeout,
    label_excited_state_molecule,
    pyseqm_excited_state_gpu_task,
    pyseqm_excited_state_task,
    run_pyseqm_batch,
)
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import build_input_dict, store_current_data


def _properties(include_forces=True):
    properties = {
        "sE0": ["state_0_energy", "system", 1.0],
        "sE1": ["state_1_energy", "system", 1.0],
    }
    if include_forces:
        properties.update(
            {
                "F0": ["state_0_forces", "atomic", 1.0],
                "F1": ["state_1_forces", "atomic", 1.0],
            }
        )
    return properties


def _molecule(molecule_id="water"):
    return MoleculesObject(
        Atoms(
            "OH2",
            positions=[
                [0.0, 0.0, 0.0],
                [0.9572, 0.0, 0.0],
                [-0.2390, 0.9266, 0.0],
            ],
        ),
        molecule_id,
    )


def _predictions():
    energies = np.asarray([[-90.0, -89.5]], dtype=np.float64)
    forces = np.arange(18, dtype=np.float64).reshape(1, 2, 3, 3)
    return energies, forces


def test_pyseqm_input_sorting_and_force_order_restoration():
    coordinates = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
    )
    species = np.asarray([[1, 8, 6]])

    sorted_coords, sorted_species, sort_indices = _prepare_pyseqm_inputs(
        coordinates, species
    )

    np.testing.assert_array_equal(sorted_species, [[8, 6, 1]])
    np.testing.assert_allclose(sorted_coords[..., 0], [[1.0, 2.0, 0.0]])
    np.testing.assert_array_equal(sort_indices, [[1, 2, 0]])

    sorted_forces = np.asarray(
        [
            [
                [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
                [[11.0, 0.0, 0.0], [21.0, 0.0, 0.0], [31.0, 0.0, 0.0]],
            ]
        ]
    )
    restored = _restore_pyseqm_force_order(sorted_forces, sort_indices)
    np.testing.assert_allclose(
        restored,
        [
            [
                [[30.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                [[31.0, 0.0, 0.0], [11.0, 0.0, 0.0], [21.0, 0.0, 0.0]],
            ]
        ],
    )


def test_pyseqm_scf_convergence_contract():
    _require_scf_convergence(
        SimpleNamespace(notconverged=np.asarray([False, False], dtype=bool)),
        2,
    )

    with pytest.raises(PySEQMConvergenceError) as exc_info:
        _require_scf_convergence(
            SimpleNamespace(notconverged=np.asarray([False, True], dtype=bool)),
            2,
        )
    assert exc_info.value.failed_indices == (1,)


@pytest.mark.parametrize(
    ("driver", "message"),
    [
        (SimpleNamespace(), "did not expose"),
        (
            SimpleNamespace(notconverged=np.asarray([[False]], dtype=bool)),
            "shape",
        ),
        (
            SimpleNamespace(notconverged=np.asarray([0], dtype=int)),
            "Boolean",
        ),
    ],
)
def test_pyseqm_rejects_missing_or_malformed_scf_flags(
    driver, message
):
    with pytest.raises(PySEQMConvergenceError, match=message) as exc_info:
        _require_scf_convergence(driver, 1)
    assert exc_info.value.failed_indices == (0,)


def test_missing_pyseqm_dependency_has_actionable_error(monkeypatch):
    original_import = builtins.__import__

    def block_seqm(name, *args, **kwargs):
        if name == "seqm" or name.startswith("seqm."):
            raise ModuleNotFoundError("synthetic missing PYSEQM")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_seqm)
    with pytest.raises(ImportError, match="working PYSEQM and Torch installation"):
        run_pyseqm_batch(
            np.zeros((1, 2, 3)),
            np.ones((1, 2), dtype=int),
            1,
        )


def test_single_molecule_labeling_maps_flattened_results_and_offset(monkeypatch):
    energies, forces = _predictions()
    calls = []

    def fake_run(coordinates, species, state_count, **kwargs):
        calls.append((coordinates.shape, species.shape, state_count, kwargs))
        return energies.copy(), forces.copy()

    monkeypatch.setattr(pyseqm_module, "run_pyseqm_batch", fake_run)
    monkeypatch.setattr(pyseqm_module, "_pyseqm_device", lambda count: "cpu")
    molecule = _molecule()
    molecule.update_metadata({"selected_state": 1})

    labeled = label_excited_state_molecule(
        molecule,
        QM_config={
            "method": "AM1",
            "energy_offset_eV": -100.0,
            "max_solve_time_seconds": 0,
        },
        properties_list=_properties(),
        sampler_config={"energy_offset_eV": -50.0},
        gpus_per_node=0,
    )

    assert calls[0][:3] == ((1, 3, 3), (1, 3), 2)
    assert labeled.check_convergence() is True
    assert labeled.get_results()["sE0"] == pytest.approx(10.0)
    assert labeled.get_results()["sE1"] == pytest.approx(10.5)
    np.testing.assert_allclose(labeled.get_results()["F0"], forces[0, 0])
    np.testing.assert_allclose(labeled.get_results()["F1"], forces[0, 1])
    metadata = labeled.get_metadata()
    assert metadata["qm_backend"] == "pyseqm"
    assert metadata["energy_offset_eV"] == -100.0
    assert metadata["energy_offset_source"] == "QM_config"
    assert metadata["n_excited_states"] == 2
    assert metadata["qm_scf_converged"] is True


def test_energy_only_contract_and_sampler_offset_fallback(monkeypatch):
    energies, forces = _predictions()
    monkeypatch.setattr(
        pyseqm_module,
        "run_pyseqm_batch",
        lambda *args, **kwargs: (energies.copy(), forces.copy()),
    )
    monkeypatch.setattr(pyseqm_module, "_pyseqm_device", lambda count: "cpu")

    labeled = label_excited_state_molecule(
        _molecule(),
        QM_config={"max_solve_time_seconds": 0},
        properties_list=_properties(include_forces=False),
        sampler_config={"energy_offset_eV": -100.0},
    )

    assert set(labeled.get_results()) == {"sE0", "sE1"}
    assert labeled.get_metadata()["energy_offset_source"] == "sampler_config"


def test_labeled_results_use_existing_hdf5_storage_path(monkeypatch, tmp_path):
    energies, forces = _predictions()
    monkeypatch.setattr(
        pyseqm_module,
        "run_pyseqm_batch",
        lambda *args, **kwargs: (energies.copy(), forces.copy()),
    )
    monkeypatch.setattr(pyseqm_module, "_pyseqm_device", lambda count: "cpu")
    properties = _properties()
    labeled = label_excited_state_molecule(
        _molecule("labeled-water"),
        QM_config={"max_solve_time_seconds": 0},
        properties_list=properties,
    )
    h5_path = tmp_path / "excited-state-labels.h5"

    store_current_data(str(h5_path), [labeled], properties)

    with h5py.File(h5_path, "r") as handle:
        group = handle["H02_O01"]
        assert group["state_0_energy"].shape == (1,)
        assert group["state_1_energy"].shape == (1,)
        assert group["state_0_forces"].shape == (1, 3, 3)
        assert group["state_1_forces"].shape == (1, 3, 3)
        assert group["state_0_energy"][0] == pytest.approx(-90.0)
        assert group["state_1_energy"][0] == pytest.approx(-89.5)


def test_existing_qm_task_input_contract_routes_without_driver_changes(monkeypatch):
    energies, forces = _predictions()
    monkeypatch.setattr(
        pyseqm_module,
        "run_pyseqm_batch",
        lambda *args, **kwargs: (energies.copy(), forces.copy()),
    )
    monkeypatch.setattr(pyseqm_module, "_pyseqm_device", lambda count: "cpu")
    molecule = _molecule("qm-task-contract")
    task_input = build_input_dict(
        pyseqm_excited_state_task.func,
        [
            {
                "molecule_object": molecule,
                "QM_config": {"max_solve_time_seconds": 0},
            },
            {"properties_list": _properties()},
            {"sampler_config": {}},
            {"gpus_per_node": 0},
        ],
        raise_on_fail=True,
    )

    labeled = pyseqm_excited_state_task.func(**task_input)

    assert labeled.check_convergence() is True
    assert set(labeled.get_results()) == {"sE0", "F0", "sE1", "F1"}


@pytest.mark.parametrize(
    ("energies", "forces", "message"),
    [
        (np.zeros((1, 1)), np.zeros((1, 2, 3, 3)), "malformed energies"),
        (np.zeros((1, 2)), np.zeros((1, 1, 3, 3)), "malformed forces"),
        (
            np.asarray([[0.0, np.nan]]),
            np.zeros((1, 2, 3, 3)),
            "non-finite energies",
        ),
        (
            np.zeros((1, 2)),
            np.full((1, 2, 3, 3), np.inf),
            "non-finite forces",
        ),
    ],
)
def test_malformed_backend_results_are_nonconverged(
    monkeypatch, energies, forces, message
):
    monkeypatch.setattr(
        pyseqm_module,
        "run_pyseqm_batch",
        lambda *args, **kwargs: (energies, forces),
    )
    monkeypatch.setattr(pyseqm_module, "_pyseqm_device", lambda count: "cpu")

    labeled = label_excited_state_molecule(
        _molecule(),
        QM_config={"max_solve_time_seconds": 0},
        properties_list=_properties(),
    )

    assert labeled.check_convergence() is False
    assert message in labeled.get_metadata()["qm_error"]
    assert labeled.get_results() == {}


def test_unsupported_properties_and_periodic_molecules_fail_cleanly(monkeypatch):
    monkeypatch.setattr(pyseqm_module, "_pyseqm_device", lambda count: "cpu")
    unsupported = _properties()
    unsupported["dE01"] = ["gap_01", "system", 1.0]
    labeled = label_excited_state_molecule(
        _molecule(),
        QM_config={},
        properties_list=unsupported,
    )
    assert labeled.check_convergence() is False
    assert "unsupported keys" in labeled.get_metadata()["qm_error"]

    periodic = _molecule("periodic")
    periodic.get_atoms().set_cell(np.eye(3) * 10.0)
    periodic.get_atoms().set_pbc(True)
    labeled = label_excited_state_molecule(
        periodic,
        QM_config={},
        properties_list=_properties(),
    )
    assert labeled.check_convergence() is False
    assert "nonperiodic" in labeled.get_metadata()["qm_error"]


def test_backend_failure_and_timeout_return_diagnostics(monkeypatch):
    monkeypatch.setattr(pyseqm_module, "_pyseqm_device", lambda count: "cpu")

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(pyseqm_module, "run_pyseqm_batch", fail)
    failed = label_excited_state_molecule(
        _molecule("failure"),
        QM_config={"max_solve_time_seconds": 0},
        properties_list=_properties(),
    )
    assert failed.check_convergence() is False
    assert failed.get_metadata()["qm_error_type"] == "RuntimeError"
    assert "synthetic solver failure" in failed.get_metadata()["qm_error"]

    def time_out(**kwargs):
        raise PySEQMTimeoutError("synthetic timeout")

    monkeypatch.setattr(pyseqm_module, "_run_pyseqm_with_timeout", time_out)
    timed_out = label_excited_state_molecule(
        _molecule("timeout"),
        QM_config={"max_solve_time_seconds": 0.01},
        properties_list=_properties(),
    )
    assert timed_out.check_convergence() is False
    assert timed_out.get_metadata()["qm_timeout"] is True
    assert timed_out.get_metadata()["max_solve_time_seconds"] == pytest.approx(
        0.01
    )


def test_scf_failure_is_nonconverged_and_removes_state_labels(monkeypatch):
    monkeypatch.setattr(pyseqm_module, "_pyseqm_device", lambda count: "cpu")

    def fail_scf(*args, **kwargs):
        raise PySEQMConvergenceError([0])

    monkeypatch.setattr(pyseqm_module, "run_pyseqm_batch", fail_scf)
    molecule = _molecule("scf-failure")
    molecule.store_results({"sE0": -1.0, "F0": np.ones((3, 3)), "other": 7})

    failed = label_excited_state_molecule(
        molecule,
        QM_config={"max_solve_time_seconds": 0},
        properties_list=_properties(),
    )

    assert failed.check_convergence() is False
    assert failed.get_results() == {"other": 7}
    metadata = failed.get_metadata()
    assert metadata["qm_error_type"] == "PySEQMConvergenceError"
    assert metadata["qm_scf_converged"] is False
    assert metadata["qm_scf_notconverged_indices"] == [0]


def test_timeout_helper_preserves_scf_convergence_error(monkeypatch):
    class ResultQueue:
        def get(self, timeout=None):
            del timeout
            return {
                "ok": False,
                "error": "synthetic SCF failure",
                "error_type": "PySEQMConvergenceError",
                "failed_indices": [0],
                "traceback": "",
            }

    class CompletedProcess:
        def __init__(self, *, target, args):
            del target, args
            self.exitcode = 0

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            del timeout

    class CompletedContext:
        def Queue(self, maxsize=0):
            del maxsize
            return ResultQueue()

        Process = CompletedProcess

    monkeypatch.setattr(
        pyseqm_module.multiprocessing,
        "get_context",
        lambda method: CompletedContext(),
    )

    with pytest.raises(PySEQMConvergenceError) as exc_info:
        _run_pyseqm_with_timeout(
            coordinates=np.zeros((1, 2, 3)),
            species=np.ones((1, 2), dtype=int),
            state_count=2,
            method="AM1",
            scf_eps=1.0e-10,
            cis_tol=1.0e-8,
            device="cpu",
            log_path=None,
            max_solve_time_seconds=1.0,
        )
    assert exc_info.value.failed_indices == (0,)


def test_timeout_helper_terminates_a_hanging_child(monkeypatch):
    class EmptyQueue:
        def get(self, timeout=None):
            del timeout
            raise queue.Empty

    class HangingProcess:
        terminated = False

        def __init__(self, *, target, args):
            del target, args
            self._alive = True
            self.exitcode = None

        def start(self):
            pass

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            del timeout

        def terminate(self):
            self.__class__.terminated = True
            self._alive = False
            self.exitcode = -15

        def kill(self):
            self._alive = False
            self.exitcode = -9

    class HangingContext:
        def Queue(self, maxsize=0):
            del maxsize
            return EmptyQueue()

        Process = HangingProcess

    monkeypatch.setattr(
        pyseqm_module.multiprocessing,
        "get_context",
        lambda method: HangingContext(),
    )

    with pytest.raises(PySEQMTimeoutError, match="exceeded"):
        _run_pyseqm_with_timeout(
            coordinates=np.zeros((1, 2, 3)),
            species=np.ones((1, 2), dtype=int),
            state_count=2,
            method="AM1",
            scf_eps=1.0e-10,
            cis_tol=1.0e-8,
            device="cpu",
            log_path=None,
            max_solve_time_seconds=0.001,
        )
    assert HangingProcess.terminated is True


def test_optional_log_records_success_and_candidate_context(monkeypatch, tmp_path):
    energies, forces = _predictions()

    def fake_run(*args, log_handle=None, **kwargs):
        print("synthetic PySEQM output", file=log_handle)
        return energies.copy(), forces.copy()

    monkeypatch.setattr(pyseqm_module, "run_pyseqm_batch", fake_run)
    monkeypatch.setattr(pyseqm_module, "_pyseqm_device", lambda count: "cpu")
    molecule = _molecule("candidate/one")
    molecule.update_metadata(
        {
            "parent_molecule_id": "parent",
            "candidate_rank": 0,
            "uncertainty_score": 2.5,
            "selected_state": 1,
            "time_ps": 0.25,
        }
    )

    labeled = label_excited_state_molecule(
        molecule,
        QM_config={
            "capture_pyseqm_logs": True,
            "pyseqm_log_dir": str(tmp_path),
            "max_solve_time_seconds": 0,
        },
        properties_list=_properties(),
    )

    assert labeled.check_convergence() is True
    log_path = next(tmp_path.glob("candidate_one*.log"))
    log_text = log_path.read_text(encoding="utf-8")
    assert "selected_state: 1" in log_text
    assert "uncertainty_score: 2.5" in log_text
    assert "synthetic PySEQM output" in log_text
    assert "status: success" in log_text


def test_device_selection_and_task_executor_contract(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 4,
        ),
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("PARSL_WORKER_RANK", "3")

    assert pyseqm_module._pyseqm_device(2) == "cuda:1"
    assert pyseqm_excited_state_task.executors == ["alf_QM_executor"]
    assert pyseqm_excited_state_gpu_task.executors == ["alf_gpu_executor"]


def test_installed_pyseqm_cpu_smoke():
    pytest.importorskip("seqm")
    torch = pytest.importorskip("torch")
    molecule = _molecule()
    atoms = molecule.get_atoms()

    energies, forces = run_pyseqm_batch(
        atoms.get_positions()[None, ...],
        atoms.get_atomic_numbers()[None, ...],
        2,
        method="AM1",
        device=torch.device("cpu"),
    )

    assert energies.shape == (1, 2)
    assert forces.shape == (1, 2, 3, 3)
    assert np.all(np.isfinite(energies))
    assert np.all(np.isfinite(forces))


def test_installed_pyseqm_timeout_process_cpu_smoke(monkeypatch):
    pytest.importorskip("seqm")
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        pyseqm_module,
        "_pyseqm_device",
        lambda count: torch.device("cpu"),
    )

    labeled = label_excited_state_molecule(
        _molecule("timeout-process-smoke"),
        QM_config={"method": "AM1", "max_solve_time_seconds": 60},
        properties_list=_properties(),
        gpus_per_node=0,
    )

    assert labeled.check_convergence() is True
    assert set(labeled.get_results()) == {"sE0", "F0", "sE1", "F1"}
