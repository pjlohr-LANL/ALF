"""Optional PySEQM labeling for flattened excited-state properties.

The public Parsl tasks in this module intentionally label one molecule at a
time. PySEQM itself receives a leading batch dimension of one, but ALF's main
driver remains unchanged and continues to submit ordinary ``QM_task`` calls.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import queue
import socket
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from parsl import python_app

from alframework.tools.excited_state_tools import (
    derive_gap_property_table,
    derive_state_property_table,
)
from alframework.tools.molecules_class import MoleculesObject


class PySEQMTimeoutError(TimeoutError):
    """Raised when a PySEQM solve exceeds its configured wall-time limit."""


class PySEQMConvergenceError(RuntimeError):
    """Raised when PySEQM reports one or more unconverged SCF solutions."""

    def __init__(
        self,
        failed_indices: list[int] | tuple[int, ...],
        message: str | None = None,
    ) -> None:
        self.failed_indices = tuple(int(index) for index in failed_indices)
        if message is None:
            message = (
                "PySEQM SCF did not converge for batch indices "
                f"{list(self.failed_indices)}."
            )
        super().__init__(message)


def _require_scf_convergence(driver: Any, batch_size: int) -> None:
    """Validate PySEQM's per-molecule SCF convergence result."""

    rejected_indices = list(range(int(batch_size)))
    raw_flags = getattr(driver, "notconverged", None)
    if raw_flags is None:
        raise PySEQMConvergenceError(
            rejected_indices,
            message=(
                "PySEQM did not expose driver.notconverged after the SCF "
                "solve; ALF cannot safely accept these labels."
            ),
        )
    if hasattr(raw_flags, "detach"):
        raw_flags = raw_flags.detach()
    if hasattr(raw_flags, "cpu"):
        raw_flags = raw_flags.cpu()
    if hasattr(raw_flags, "numpy"):
        raw_flags = raw_flags.numpy()
    flags = np.asarray(raw_flags)
    expected_shape = (int(batch_size),)
    if flags.shape != expected_shape:
        raise PySEQMConvergenceError(
            rejected_indices,
            message=(
                "PySEQM driver.notconverged must provide one Boolean flag per "
                f"molecule with shape {expected_shape}; received "
                f"{flags.shape}."
            ),
        )
    if not np.issubdtype(flags.dtype, np.bool_):
        raise PySEQMConvergenceError(
            rejected_indices,
            message=(
                "PySEQM driver.notconverged must contain Boolean values; "
                f"received dtype {flags.dtype}."
            ),
        )
    failed_indices = np.flatnonzero(flags).astype(int).tolist()
    if failed_indices:
        raise PySEQMConvergenceError(failed_indices)


def _prepare_pyseqm_inputs(
    coordinates: np.ndarray,
    species: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate inputs and stably sort atoms as required by PySEQM."""

    coords = np.asarray(coordinates, dtype=np.float64)
    numbers = np.asarray(species, dtype=np.int64)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(
            "PySEQM coordinates must have shape [batch, atoms, 3]; "
            f"received {coords.shape}."
        )
    if numbers.ndim != 2 or coords.shape[:2] != numbers.shape:
        raise ValueError(
            "PySEQM species must have shape [batch, atoms] matching coordinates; "
            f"received coordinates {coords.shape} and species {numbers.shape}."
        )
    if coords.shape[0] < 1 or coords.shape[1] < 1:
        raise ValueError("PySEQM requires at least one molecule and one atom.")
    if not np.all(np.isfinite(coords)):
        raise ValueError("PySEQM coordinates must be finite.")
    if np.any(numbers <= 0):
        raise ValueError("PySEQM atomic numbers must be positive integers.")

    sort_indices = np.empty(numbers.shape, dtype=np.int64)
    sorted_numbers = np.empty_like(numbers)
    sorted_coords = np.empty_like(coords)
    for batch_index in range(numbers.shape[0]):
        order = np.argsort(-numbers[batch_index], kind="stable")
        sort_indices[batch_index] = order
        sorted_numbers[batch_index] = numbers[batch_index, order]
        sorted_coords[batch_index] = coords[batch_index, order]
    return sorted_coords, sorted_numbers, sort_indices


def _restore_pyseqm_force_order(
    forces: np.ndarray,
    sort_indices: np.ndarray,
) -> np.ndarray:
    """Restore PySEQM forces to the input atom order."""

    force_array = np.asarray(forces, dtype=np.float64)
    indices = np.asarray(sort_indices, dtype=np.int64)
    if force_array.ndim != 4 or force_array.shape[-1] != 3:
        raise ValueError(
            "PySEQM forces must have shape [batch, states, atoms, 3]; "
            f"received {force_array.shape}."
        )
    if (
        indices.ndim != 2
        or indices.shape[0] != force_array.shape[0]
        or indices.shape[1] != force_array.shape[2]
    ):
        raise ValueError(
            f"Sort-index shape {indices.shape} is incompatible with force shape "
            f"{force_array.shape}."
        )

    restored = np.empty_like(force_array)
    for batch_index in range(force_array.shape[0]):
        inverse = np.empty_like(indices[batch_index])
        inverse[indices[batch_index]] = np.arange(
            indices.shape[1], dtype=np.int64
        )
        restored[batch_index] = force_array[batch_index][:, inverse, :]
    return restored


def _validate_pyseqm_outputs(
    energies: Any,
    forces: Any,
    *,
    batch_size: int,
    state_count: int,
    atom_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    energy_array = np.asarray(energies, dtype=np.float64)
    force_array = np.asarray(forces, dtype=np.float64)
    expected_energy_shape = (int(batch_size), int(state_count))
    expected_force_shape = (
        int(batch_size),
        int(state_count),
        int(atom_count),
        3,
    )
    if energy_array.shape != expected_energy_shape:
        raise ValueError(
            "PySEQM returned malformed energies: expected "
            f"{expected_energy_shape}, received {energy_array.shape}."
        )
    if force_array.shape != expected_force_shape:
        raise ValueError(
            "PySEQM returned malformed forces: expected "
            f"{expected_force_shape}, received {force_array.shape}."
        )
    if not np.all(np.isfinite(energy_array)):
        raise ValueError("PySEQM returned non-finite energies.")
    if not np.all(np.isfinite(force_array)):
        raise ValueError("PySEQM returned non-finite forces.")
    return energy_array, force_array


def run_pyseqm_batch(
    coordinates: np.ndarray,
    species: np.ndarray,
    state_count: int,
    *,
    method: str = "AM1",
    scf_eps: float = 1.0e-10,
    cis_tol: float = 1.0e-8,
    device: Any = None,
    log_handle: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run PySEQM for one or more compatible molecules.

    This internal batch-shaped function reflects PySEQM's API. The public ALF
    tasks in this milestone always call it with a batch size of one.
    """

    try:
        import torch
        from seqm.ElectronicStructure import Electronic_Structure
        from seqm.Molecule import Molecule
        from seqm.seqm_functions.constants import Constants
    except Exception as exc:
        raise ImportError(
            "PySEQM excited-state labeling requires a working PYSEQM and Torch "
            "installation on the selected QM worker."
        ) from exc

    n_states = int(state_count)
    if n_states < 1:
        raise ValueError("state_count must be at least one.")
    scf_tolerance = float(scf_eps)
    cis_tolerance = float(cis_tol)
    if not np.isfinite(scf_tolerance) or scf_tolerance <= 0:
        raise ValueError("scf_eps must be a finite positive value.")
    if not np.isfinite(cis_tolerance) or cis_tolerance <= 0:
        raise ValueError("cis_tol must be a finite positive value.")

    coords, numbers, sort_indices = _prepare_pyseqm_inputs(
        coordinates, species
    )
    coords = np.ascontiguousarray(coords)
    numbers = np.ascontiguousarray(numbers)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        coords_tensor = torch.from_numpy(coords)
        species_tensor = torch.from_numpy(numbers)
        if device.type == "cuda":
            coords_tensor = coords_tensor.pin_memory()
            species_tensor = species_tensor.pin_memory()
        coords_tensor = coords_tensor.to(
            device, non_blocking=(device.type == "cuda")
        )
        species_tensor = species_tensor.to(
            device, non_blocking=(device.type == "cuda")
        )

        parameters = {
            "method": str(method),
            "scf_eps": scf_tolerance,
            "scf_converger": [2],
            "excited_states": {
                "n_states": max(0, n_states - 1),
                "cis_tol": cis_tolerance,
            },
            "analytical_gradient": [True],
            "do_all_forces": True,
        }
        constants = Constants().to(device)
        molecule = Molecule(
            constants, parameters, coords_tensor, species_tensor
        ).to(device)
        driver = Electronic_Structure(parameters).to(device)
        driver(molecule)
        _require_scf_convergence(driver, molecule.nmol)

        all_energies = torch.empty(
            (molecule.nmol, n_states), dtype=torch.float64, device=device
        )
        all_energies[:, 0] = molecule.Etot
        if n_states > 1:
            all_energies[:, 1:] = (
                molecule.Etot.unsqueeze(1) + molecule.cis_energies
            )
        energy_array = all_energies.detach().cpu().numpy()
        force_array = _restore_pyseqm_force_order(
            molecule.all_forces.detach().cpu().numpy(), sort_indices
        )
        energy_array, force_array = _validate_pyseqm_outputs(
            energy_array,
            force_array,
            batch_size=coords.shape[0],
            state_count=n_states,
            atom_count=coords.shape[1],
        )
        if log_handle is not None:
            print(
                "PySEQM solve complete: "
                f"batch={coords.shape[0]} states={n_states} atoms={coords.shape[1]}",
                file=log_handle,
                flush=True,
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return energy_array, force_array
    finally:
        torch.set_default_dtype(previous_dtype)


def _pyseqm_timeout_seconds(QM_config: dict[str, Any]) -> float | None:
    raw_timeout = QM_config.get("max_solve_time_seconds")
    if raw_timeout is None:
        return None
    timeout = float(raw_timeout)
    if not np.isfinite(timeout):
        raise ValueError("max_solve_time_seconds must be finite when provided.")
    return None if timeout <= 0 else timeout


def _pyseqm_device(gpus_per_node: int | None = None):
    try:
        import torch
    except Exception as exc:
        raise ImportError(
            "PySEQM excited-state labeling requires Torch on the QM worker."
        ) from exc

    if not torch.cuda.is_available() or int(torch.cuda.device_count()) < 1:
        return torch.device("cpu")
    visible_count = int(torch.cuda.device_count())
    usable_count = visible_count
    if gpus_per_node is not None and int(gpus_per_node) > 0:
        usable_count = min(visible_count, int(gpus_per_node))
    worker_rank = int(os.environ.get("PARSL_WORKER_RANK", "0"))
    return torch.device(f"cuda:{worker_rank % usable_count}")


def _pyseqm_child(
    result_queue: Any,
    coordinates: np.ndarray,
    species: np.ndarray,
    state_count: int,
    method: str,
    scf_eps: float,
    cis_tol: float,
    device_name: str,
    log_path: str | None,
) -> None:
    try:
        import torch

        device = torch.device(device_name)
        with contextlib.ExitStack() as stack:
            log_handle = None
            if log_path is not None:
                log_handle = stack.enter_context(
                    open(log_path, "a", encoding="utf-8", buffering=1)
                )
                stack.enter_context(contextlib.redirect_stdout(log_handle))
                stack.enter_context(contextlib.redirect_stderr(log_handle))
                stack.enter_context(warnings.catch_warnings())
                warnings.simplefilter("always")
            energies, forces = run_pyseqm_batch(
                coordinates,
                species,
                state_count,
                method=method,
                scf_eps=scf_eps,
                cis_tol=cis_tol,
                device=device,
                log_handle=log_handle,
            )
        result_queue.put(
            {"ok": True, "energies": energies, "forces": forces}
        )
    except BaseException as exc:
        payload = {
            "ok": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
        if isinstance(exc, PySEQMConvergenceError):
            payload["failed_indices"] = list(exc.failed_indices)
        result_queue.put(payload)


def _stop_child_process(process: Any) -> None:
    if not process.is_alive():
        return
    process.terminate()
    process.join(5.0)
    if process.is_alive():
        process.kill()
        process.join(5.0)


def _run_pyseqm_with_timeout(
    *,
    coordinates: np.ndarray,
    species: np.ndarray,
    state_count: int,
    method: str,
    scf_eps: float,
    cis_tol: float,
    device: Any,
    log_path: Path | None,
    max_solve_time_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_pyseqm_child,
        args=(
            result_queue,
            np.ascontiguousarray(coordinates).copy(),
            np.ascontiguousarray(species).copy(),
            int(state_count),
            str(method),
            float(scf_eps),
            float(cis_tol),
            str(device),
            None if log_path is None else str(log_path),
        ),
    )
    process.start()
    deadline = time.monotonic() + float(max_solve_time_seconds)
    payload = None
    while payload is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _stop_child_process(process)
            raise PySEQMTimeoutError(
                "PySEQM solve exceeded "
                f"{float(max_solve_time_seconds):.6g} seconds."
            )
        try:
            payload = result_queue.get(timeout=min(0.1, remaining))
        except queue.Empty:
            if not process.is_alive():
                process.join()
                raise RuntimeError(
                    "PySEQM child process exited without returning a result; "
                    f"exitcode={process.exitcode}."
                )

    process.join(5.0)
    _stop_child_process(process)
    if not bool(payload.get("ok", False)):
        message = str(payload.get("error", "unknown PySEQM child error"))
        child_traceback = str(payload.get("traceback", "")).strip()
        if child_traceback:
            message = f"{message}\n{child_traceback}"
        if payload.get("error_type") == "PySEQMConvergenceError":
            raise PySEQMConvergenceError(
                payload.get("failed_indices", []),
                message=message,
            )
        raise RuntimeError(message)
    return _validate_pyseqm_outputs(
        payload["energies"],
        payload["forces"],
        batch_size=int(np.asarray(coordinates).shape[0]),
        state_count=int(state_count),
        atom_count=int(np.asarray(coordinates).shape[1]),
    )


def _safe_log_component(value: Any) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(value)
    )


def _pyseqm_log_path(
    QM_config: dict[str, Any], molecule: MoleculesObject
) -> Path | None:
    if not bool(QM_config.get("capture_pyseqm_logs", False)):
        return None
    log_dir = Path(
        str(QM_config.get("pyseqm_log_dir", "pyseqm_logs"))
    ).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    molecule_id = _safe_log_component(molecule.get_moleculeid())
    return log_dir / (
        f"{molecule_id}.pid-{os.getpid()}.{time.time_ns()}.log"
    )


def _write_log_header(
    handle: Any,
    molecule: MoleculesObject,
    *,
    device: Any,
    method: str,
    scf_eps: float,
    cis_tol: float,
    state_count: int,
) -> None:
    atoms = molecule.get_atoms()
    metadata = molecule.get_metadata()
    print("=== PySEQM excited-state QM task start ===", file=handle)
    print(f"timestamp_unix: {time.time():.6f}", file=handle)
    print(f"hostname: {socket.gethostname()}", file=handle)
    print(f"pid: {os.getpid()}", file=handle)
    print(f"molecule_id: {molecule.get_moleculeid()}", file=handle)
    print(f"parent_molecule_id: {metadata.get('parent_molecule_id')}", file=handle)
    print(f"candidate_rank: {metadata.get('candidate_rank')}", file=handle)
    print(f"uncertainty_score: {metadata.get('uncertainty_score')}", file=handle)
    print(f"selected_state: {metadata.get('selected_state')}", file=handle)
    print(f"time_ps: {metadata.get('time_ps')}", file=handle)
    print(f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}", file=handle)
    print(f"parsl_worker_rank: {os.environ.get('PARSL_WORKER_RANK')}", file=handle)
    print(f"device: {device}", file=handle)
    print(f"method: {method}", file=handle)
    print(f"scf_eps: {scf_eps}", file=handle)
    print(f"cis_tol: {cis_tol}", file=handle)
    print(f"state_count: {state_count}", file=handle)
    print(f"atom_count: {len(atoms)}", file=handle)
    print(f"atomic_numbers: {atoms.get_atomic_numbers().tolist()}", file=handle)
    print("=== PySEQM stdout/stderr follows ===", file=handle, flush=True)


def _write_log_footer(
    handle: Any,
    status: str,
    elapsed_seconds: float,
    error: BaseException | None = None,
) -> None:
    print("=== PySEQM excited-state QM task end ===", file=handle)
    print(f"status: {status}", file=handle)
    print(f"elapsed_seconds: {elapsed_seconds:.6f}", file=handle)
    if error is not None:
        print(f"error: {error!r}", file=handle)
    handle.flush()


def _finish_log(
    log_handle: Any,
    log_path: Path | None,
    *,
    status: str,
    elapsed_seconds: float,
    error: BaseException | None = None,
) -> None:
    if log_handle is not None and not log_handle.closed:
        _write_log_footer(
            log_handle, status, elapsed_seconds, error=error
        )
        log_handle.close()
    elif log_path is not None:
        with open(log_path, "a", encoding="utf-8", buffering=1) as handle:
            _write_log_footer(handle, status, elapsed_seconds, error=error)


def _energy_offset(
    QM_config: dict[str, Any], sampler_config: dict[str, Any]
) -> tuple[float, str]:
    if "energy_offset_eV" in QM_config:
        raw_offset = QM_config["energy_offset_eV"]
        source = "QM_config"
    else:
        raw_offset = sampler_config.get("energy_offset_eV", 0.0)
        source = "sampler_config" if "energy_offset_eV" in sampler_config else "default"
    offset = float(raw_offset)
    if not np.isfinite(offset):
        raise ValueError("energy_offset_eV must be finite.")
    return offset, source


def label_excited_state_molecule(
    molecule_object: MoleculesObject,
    *,
    QM_config: dict[str, Any],
    properties_list: dict[str, list[Any]],
    sampler_config: dict[str, Any] | None = None,
    gpus_per_node: int | None = None,
) -> MoleculesObject:
    """Label one molecule with PySEQM state energies and optional forces."""

    if not isinstance(molecule_object, MoleculesObject):
        raise TypeError("molecule_object must be a MoleculesObject instance.")

    qm_options = dict(QM_config or {})
    sampler_options = dict(sampler_config or {})
    start_time = time.time()
    log_path: Path | None = None
    log_handle = None
    timeout: float | None = None
    try:
        state_table = derive_state_property_table(
            properties_list, require_forces=False
        )
        gap_table = derive_gap_property_table(properties_list)
        supported_keys = {
            key
            for row in state_table
            for key in (row["energy_key"], row["force_key"])
            if key is not None
        }
        supported_keys.update(str(row["gap_key"]) for row in gap_table)
        unsupported = sorted(set(properties_list) - supported_keys)
        if unsupported:
            raise ValueError(
                "The PySEQM interface currently supports only flattened "
                f"sE#/F#/dE# properties; unsupported keys: {unsupported}."
            )

        atoms = molecule_object.get_atoms()
        if atoms is None or len(atoms) < 1:
            raise ValueError("PySEQM cannot label a molecule without atoms.")
        if bool(np.any(atoms.get_pbc())):
            raise NotImplementedError(
                "PySEQM excited-state labeling currently supports only "
                "nonperiodic molecules."
            )
        coordinates = np.asarray(
            atoms.get_positions(), dtype=np.float64
        )[None, ...]
        species = np.asarray(
            atoms.get_atomic_numbers(), dtype=np.int64
        )[None, ...]
        offset, offset_source = _energy_offset(qm_options, sampler_options)
        method = str(qm_options.get("method", "AM1"))
        scf_eps = float(qm_options.get("scf_eps", 1.0e-10))
        cis_tol = float(qm_options.get("cis_tol", 1.0e-8))
        timeout = _pyseqm_timeout_seconds(qm_options)
        device = _pyseqm_device(gpus_per_node)
        log_path = _pyseqm_log_path(qm_options, molecule_object)
        if log_path is not None:
            log_handle = open(
                log_path, "w", encoding="utf-8", buffering=1
            )
            _write_log_header(
                log_handle,
                molecule_object,
                device=device,
                method=method,
                scf_eps=scf_eps,
                cis_tol=cis_tol,
                state_count=len(state_table),
            )

        if timeout is not None:
            if log_handle is not None:
                log_handle.close()
                log_handle = None
            energies, forces = _run_pyseqm_with_timeout(
                coordinates=coordinates,
                species=species,
                state_count=len(state_table),
                method=method,
                scf_eps=scf_eps,
                cis_tol=cis_tol,
                device=device,
                log_path=log_path,
                max_solve_time_seconds=timeout,
            )
        else:
            with contextlib.ExitStack() as stack:
                if log_handle is not None:
                    stack.enter_context(contextlib.redirect_stdout(log_handle))
                    stack.enter_context(contextlib.redirect_stderr(log_handle))
                    stack.enter_context(warnings.catch_warnings())
                    warnings.simplefilter("always")
                energies, forces = run_pyseqm_batch(
                    coordinates,
                    species,
                    len(state_table),
                    method=method,
                    scf_eps=scf_eps,
                    cis_tol=cis_tol,
                    device=device,
                    log_handle=log_handle,
                )
        energies, forces = _validate_pyseqm_outputs(
            energies,
            forces,
            batch_size=1,
            state_count=len(state_table),
            atom_count=len(atoms),
        )
    except Exception as exc:
        elapsed = time.time() - start_time
        _finish_log(
            log_handle,
            log_path,
            status=("timeout" if isinstance(exc, PySEQMTimeoutError) else "error"),
            elapsed_seconds=elapsed,
            error=exc,
        )
        error_metadata = {
            "qm_backend": "pyseqm",
            "qm_error": str(exc),
            "qm_error_type": type(exc).__name__,
            "qm_elapsed_seconds": float(elapsed),
        }
        if isinstance(exc, PySEQMConvergenceError):
            error_metadata.update(
                {
                    "qm_scf_converged": False,
                    "qm_scf_notconverged_indices": list(exc.failed_indices),
                }
            )
        if isinstance(exc, PySEQMTimeoutError):
            error_metadata["qm_timeout"] = True
        if timeout is not None:
            error_metadata["max_solve_time_seconds"] = float(timeout)
        if log_path is not None:
            error_metadata["pyseqm_log_path"] = str(log_path)
        for property_key in dict(properties_list or {}):
            molecule_object.get_results().pop(str(property_key), None)
        molecule_object.update_metadata(error_metadata)
        molecule_object.set_converged_flag(False)
        return molecule_object

    results: dict[str, Any] = {}
    for row in state_table:
        state_index = int(row["state"])
        results[str(row["energy_key"])] = float(
            energies[0, state_index] - offset
        )
        if row["force_key"] is not None:
            results[str(row["force_key"])] = np.asarray(
                forces[0, state_index], dtype=np.float64
            )
    for row in gap_table:
        results[str(row["gap_key"])] = float(
            results[str(row["upper_energy_key"])]
            - results[str(row["lower_energy_key"])]
        )
    elapsed = time.time() - start_time
    _finish_log(
        log_handle,
        log_path,
        status="success",
        elapsed_seconds=elapsed,
    )
    molecule_object.store_results(results)
    molecule_object.update_metadata(
        {
            "qm_backend": "pyseqm",
            "qm_device": str(device),
            "qm_elapsed_seconds": float(elapsed),
            "qm_scf_converged": True,
            "energy_offset_eV": float(offset),
            "energy_offset_source": offset_source,
            "n_excited_states": len(state_table),
            "gap_properties": [
                str(row["gap_key"]) for row in gap_table
            ],
            **(
                {"pyseqm_log_path": str(log_path)}
                if log_path is not None
                else {}
            ),
        }
    )
    molecule_object.set_converged_flag(True)
    return molecule_object


@python_app(executors=["alf_QM_executor"])
def pyseqm_excited_state_task(
    molecule_object,
    QM_config,
    properties_list,
    sampler_config=None,
    gpus_per_node=None,
):
    """Label one molecule on ALF's ordinary QM executor."""

    return label_excited_state_molecule(
        molecule_object,
        QM_config=dict(QM_config or {}),
        properties_list=dict(properties_list or {}),
        sampler_config=dict(sampler_config or {}),
        gpus_per_node=(
            None if gpus_per_node is None else int(gpus_per_node)
        ),
    )


@python_app(executors=["alf_gpu_executor"])
def pyseqm_excited_state_gpu_task(
    molecule_object,
    QM_config,
    properties_list,
    sampler_config=None,
    gpus_per_node=None,
):
    """Label one molecule on ALF's shared GPU executor."""

    return label_excited_state_molecule(
        molecule_object,
        QM_config=dict(QM_config or {}),
        properties_list=dict(properties_list or {}),
        sampler_config=dict(sampler_config or {}),
        gpus_per_node=(
            None if gpus_per_node is None else int(gpus_per_node)
        ),
    )
