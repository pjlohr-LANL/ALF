from __future__ import annotations

import contextlib
import os
import socket
import time
from typing import Any
import warnings
from pathlib import Path

import numpy as np
from parsl import python_app

from alframework.tools.excited_state_tools import derive_gap_property_table, derive_state_property_table
from alframework.tools.molecules_class import MoleculesObject


def _prepare_pyseqm_inputs(
    coords_np: np.ndarray,
    species_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(coords_np, dtype=np.float64)
    species = np.asarray(species_np, dtype=np.int64)
    if coords.ndim != 3 or species.ndim != 2:
        raise ValueError(
            f"pyseqm inputs must have shapes (B, A, 3) and (B, A). Got {coords.shape} and {species.shape}."
        )
    if coords.shape[:2] != species.shape:
        raise ValueError(f"Coordinate/species batch shapes do not match: {coords.shape} vs {species.shape}.")

    batch_size, n_atoms = species.shape
    sort_idx = np.empty((batch_size, n_atoms), dtype=np.int64)
    sorted_species = np.empty_like(species)
    sorted_coords = np.empty_like(coords)
    for batch_index in range(batch_size):
        # pyseqm expects atomic numbers in decreasing order. Use a stable sort so
        # equal-Z atoms preserve their relative geometry ordering.
        order = np.argsort(-species[batch_index], kind="stable")
        sort_idx[batch_index] = order
        sorted_species[batch_index] = species[batch_index, order]
        sorted_coords[batch_index] = coords[batch_index, order]

    return sorted_coords, sorted_species, sort_idx


def _restore_pyseqm_force_order(forces_np: np.ndarray, sort_idx: np.ndarray) -> np.ndarray:
    forces = np.asarray(forces_np, dtype=np.float64)
    sort_idx = np.asarray(sort_idx, dtype=np.int64)
    if forces.ndim != 4:
        raise ValueError(f"pyseqm forces must have shape (B, S, A, 3). Got {forces.shape}.")
    if sort_idx.shape[0] != forces.shape[0] or sort_idx.shape[1] != forces.shape[2]:
        raise ValueError(f"Sort index shape {sort_idx.shape} is incompatible with force shape {forces.shape}.")

    restored = np.empty_like(forces)
    for batch_index in range(forces.shape[0]):
        inverse = np.empty_like(sort_idx[batch_index])
        inverse[sort_idx[batch_index]] = np.arange(sort_idx.shape[1], dtype=np.int64)
        restored[batch_index] = forces[batch_index][:, inverse, :]
    return restored


def _log_pyseqm_stage(handle, stage: str, *, start_time: float | None = None, **fields: Any) -> None:
    if handle is None:
        return
    parts = [f"[pyseqm-stage] {stage}"]
    if start_time is not None:
        parts.append(f"elapsed_seconds={time.time() - start_time:.6f}")
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    print(" | ".join(parts), file=handle, flush=True)


def run_pyseqm_batch(
    coords_np: np.ndarray,
    species_np: np.ndarray,
    n_states: int,
    *,
    method: str = "AM1",
    scf_eps: float = 1e-10,
    cis_tol: float = 1e-8,
    device=None,
    log_handle=None,
) -> tuple[np.ndarray, np.ndarray]:
    stage_start = time.time()
    _log_pyseqm_stage(log_handle, "import torch/seqm start", start_time=stage_start)
    import torch

    from seqm.ElectronicStructure import Electronic_Structure
    from seqm.Molecule import Molecule
    from seqm.seqm_functions.constants import Constants
    _log_pyseqm_stage(log_handle, "import torch/seqm done", start_time=stage_start)

    torch.set_default_dtype(torch.float64)
    _log_pyseqm_stage(log_handle, "torch default dtype set", start_time=stage_start, dtype="float64")

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    _log_pyseqm_stage(
        log_handle,
        "device selected",
        start_time=stage_start,
        device=device,
        cuda_available=torch.cuda.is_available(),
        cuda_device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
    )

    _log_pyseqm_stage(
        log_handle,
        "numpy copy/sort start",
        start_time=stage_start,
        coords_shape=np.asarray(coords_np).shape,
        species_shape=np.asarray(species_np).shape,
    )
    coords_np = np.ascontiguousarray(coords_np).copy()
    species_np = np.ascontiguousarray(species_np).copy()
    coords_np, species_np, sort_idx = _prepare_pyseqm_inputs(coords_np, species_np)
    _log_pyseqm_stage(
        log_handle,
        "numpy copy/sort done",
        start_time=stage_start,
        sorted_coords_shape=coords_np.shape,
        sorted_species_shape=species_np.shape,
    )

    _log_pyseqm_stage(log_handle, "torch tensor creation start", start_time=stage_start)
    coords = torch.from_numpy(coords_np).pin_memory() if device.type == "cuda" else torch.from_numpy(coords_np)
    species = torch.from_numpy(species_np).pin_memory() if device.type == "cuda" else torch.from_numpy(species_np)
    _log_pyseqm_stage(
        log_handle,
        "torch tensor creation done",
        start_time=stage_start,
        coords_dtype=getattr(coords, "dtype", None),
        species_dtype=getattr(species, "dtype", None),
    )

    _log_pyseqm_stage(log_handle, "tensor transfer to device start", start_time=stage_start, device=device)
    coords = coords.to(device, non_blocking=(device.type == "cuda"))
    species = species.to(device, non_blocking=(device.type == "cuda"))
    _log_pyseqm_stage(log_handle, "tensor transfer to device done", start_time=stage_start, device=device)

    _log_pyseqm_stage(log_handle, "pyseqm params construction start", start_time=stage_start)
    params = {
        "method": method,
        "scf_eps": float(scf_eps),
        "scf_converger": [2],
        "excited_states": {"n_states": max(0, int(n_states) - 1), "cis_tol": float(cis_tol)},
        "analytical_gradient": [True],
        "do_all_forces": True,
    }
    _log_pyseqm_stage(
        log_handle,
        "pyseqm params construction done",
        start_time=stage_start,
        method=method,
        n_states=n_states,
        scf_eps=scf_eps,
        cis_tol=cis_tol,
    )

    _log_pyseqm_stage(log_handle, "Constants().to(device) start", start_time=stage_start, device=device)
    const = Constants().to(device)
    _log_pyseqm_stage(log_handle, "Constants().to(device) done", start_time=stage_start, device=device)

    _log_pyseqm_stage(log_handle, "Molecule(...).to(device) start", start_time=stage_start, device=device)
    molecule = Molecule(const, params, coords, species).to(device)
    _log_pyseqm_stage(
        log_handle,
        "Molecule(...).to(device) done",
        start_time=stage_start,
        nmol=getattr(molecule, "nmol", None),
        molsize=getattr(molecule, "molsize", None),
    )

    _log_pyseqm_stage(log_handle, "Electronic_Structure(...).to(device) start", start_time=stage_start, device=device)
    driver = Electronic_Structure(params).to(device)
    _log_pyseqm_stage(log_handle, "Electronic_Structure(...).to(device) done", start_time=stage_start, device=device)

    _log_pyseqm_stage(log_handle, "driver(molecule) start", start_time=stage_start)
    driver(molecule)
    _log_pyseqm_stage(
        log_handle,
        "driver(molecule) done",
        start_time=stage_start,
        notconverged=getattr(driver, "notconverged", None),
    )

    _log_pyseqm_stage(log_handle, "energy extraction start", start_time=stage_start)
    batch = molecule.nmol
    all_energies = torch.empty((batch, int(n_states)), dtype=torch.float64, device=device)
    all_energies[:, 0] = molecule.Etot
    if int(n_states) > 1:
        all_energies[:, 1:] = molecule.Etot.unsqueeze(1) + molecule.cis_energies

    energies = all_energies.detach().cpu().numpy()
    _log_pyseqm_stage(log_handle, "energy extraction done", start_time=stage_start, energies_shape=energies.shape)

    _log_pyseqm_stage(log_handle, "force extraction/unpermutation start", start_time=stage_start)
    forces = _restore_pyseqm_force_order(molecule.all_forces.detach().cpu().numpy(), sort_idx)
    _log_pyseqm_stage(log_handle, "force extraction/unpermutation done", start_time=stage_start, forces_shape=forces.shape)

    _log_pyseqm_stage(log_handle, "cleanup start", start_time=stage_start)
    del coords, species, const, molecule, driver, all_energies
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _log_pyseqm_stage(log_handle, "cleanup done", start_time=stage_start)

    return energies, forces


def _pyseqm_device(gpus_per_node: int | None = None):
    import torch

    if not torch.cuda.is_available():
        return torch.device("cpu")
    visible_device_count = int(torch.cuda.device_count())
    if visible_device_count <= 0:
        return torch.device("cpu")
    if visible_device_count == 1:
        return torch.device("cuda:0")
    worker_rank = int(os.environ.get("PARSL_WORKER_RANK", "0"))
    return torch.device(f"cuda:{worker_rank % visible_device_count}")


def _has_complete_prelabeled_results(
    molecule_object: MoleculesObject,
    properties_list: dict[str, list[Any]],
) -> bool:
    if molecule_object.check_convergence() is not True:
        return False

    atoms = molecule_object.get_atoms()
    if atoms is None:
        return False

    n_atoms = len(atoms)
    results = molecule_object.get_results()
    for prop_key, schema in properties_list.items():
        if prop_key not in results:
            return False
        prop_kind = str(schema[1]).lower() if len(schema) > 1 else ""
        value = np.asarray(results[prop_key], dtype=np.float64)
        if prop_kind == "system":
            if value.reshape(-1).size != 1 or not np.all(np.isfinite(value)):
                return False
        elif prop_kind == "atomic":
            if value.shape != (n_atoms, 3) or not np.all(np.isfinite(value)):
                return False
        else:
            return False
    return True


def _safe_log_component(value: Any) -> str:
    raw = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in raw)


def _minimum_distance(atoms) -> float | None:
    if atoms is None or len(atoms) < 2:
        return None
    distances = np.asarray(atoms.get_all_distances(mic=True), dtype=np.float64)
    np.fill_diagonal(distances, np.inf)
    min_distance = float(np.min(distances))
    if not np.isfinite(min_distance):
        return None
    return min_distance


def _pyseqm_log_path(QM_config: dict[str, Any], molecule_object: MoleculesObject) -> Path | None:
    if not bool(QM_config.get("capture_pyseqm_logs", False)):
        return None
    log_dir = Path(str(QM_config.get("pyseqm_log_dir", "pyseqm_logs"))).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    molecule_id = _safe_log_component(molecule_object.get_moleculeid())
    return log_dir / f"{molecule_id}.pid-{os.getpid()}.{time.time_ns()}.log"


def _write_pyseqm_log_header(
    handle,
    molecule_object: MoleculesObject,
    atoms,
    device,
    method: str,
    scf_eps: float,
    cis_tol: float,
    n_states: int,
) -> None:
    metadata = molecule_object.get_metadata()
    print("=== PySEQM excited-state QM task start ===", file=handle, flush=True)
    print(f"timestamp_unix: {time.time():.6f}", file=handle, flush=True)
    print(f"hostname: {socket.gethostname()}", file=handle, flush=True)
    print(f"pid: {os.getpid()}", file=handle, flush=True)
    print(f"molecule_id: {molecule_object.get_moleculeid()}", file=handle, flush=True)
    print(f"parent_molecule_id: {metadata.get('parent_molecule_id')}", file=handle, flush=True)
    print(f"candidate_rank: {metadata.get('candidate_rank')}", file=handle, flush=True)
    print(f"candidate_score: {metadata.get('candidate_score')}", file=handle, flush=True)
    print(f"selected_state: {metadata.get('selected_state')}", file=handle, flush=True)
    print(f"candidate_time_ps: {metadata.get('candidate_time_ps')}", file=handle, flush=True)
    print(f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}", file=handle, flush=True)
    print(f"parsl_worker_rank: {os.environ.get('PARSL_WORKER_RANK')}", file=handle, flush=True)
    print(f"device: {device}", file=handle, flush=True)
    print(f"method: {method}", file=handle, flush=True)
    print(f"scf_eps: {scf_eps}", file=handle, flush=True)
    print(f"cis_tol: {cis_tol}", file=handle, flush=True)
    print(f"n_states: {n_states}", file=handle, flush=True)
    print(f"n_atoms: {len(atoms)}", file=handle, flush=True)
    print(f"atomic_numbers: {atoms.get_atomic_numbers().tolist()}", file=handle, flush=True)
    print(f"min_distance_A: {_minimum_distance(atoms)}", file=handle, flush=True)
    print("=== PySEQM stdout/stderr follows ===", file=handle, flush=True)


def _write_pyseqm_log_footer(handle, status: str, elapsed_seconds: float, error: Exception | None = None) -> None:
    print("=== PySEQM excited-state QM task end ===", file=handle, flush=True)
    print(f"status: {status}", file=handle, flush=True)
    print(f"elapsed_seconds: {elapsed_seconds:.6f}", file=handle, flush=True)
    if error is not None:
        print(f"error: {error!r}", file=handle, flush=True)


def label_excited_state_molecule(
    molecule_object: MoleculesObject,
    *,
    QM_config: dict[str, Any],
    properties_list: dict[str, list[Any]],
    sampler_config: dict[str, Any] | None = None,
    gpus_per_node: int | None = None,
) -> MoleculesObject:
    if not isinstance(molecule_object, MoleculesObject):
        raise TypeError("molecule_object must be a MoleculesObject instance.")

    if bool(QM_config.get("accept_prelabeled", False)) and _has_complete_prelabeled_results(
        molecule_object,
        properties_list,
    ):
        molecule_object.update_metadata({"qm_backend": "prelabeled_seed", "qm_skipped": True})
        return molecule_object

    state_table = derive_state_property_table(properties_list, require_forces=False)
    atoms = molecule_object.get_atoms()
    coords = np.asarray(atoms.get_positions(), dtype=np.float64)[None, ...]
    species = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)[None, ...]
    offset = float(
        QM_config.get(
            "energy_offset_eV",
            (sampler_config or {}).get("energy_offset_eV", 0.0),
        )
    )
    method = str(QM_config.get("method", "AM1"))
    scf_eps = float(QM_config.get("scf_eps", 1e-10))
    cis_tol = float(QM_config.get("cis_tol", 1e-8))
    device = _pyseqm_device(gpus_per_node)
    log_path = _pyseqm_log_path(QM_config, molecule_object)
    log_handle = None
    start_time = time.time()
    try:
        if log_path is not None:
            log_handle = open(log_path, "w", encoding="utf-8", buffering=1)
            _write_pyseqm_log_header(log_handle, molecule_object, atoms, device, method, scf_eps, cis_tol, len(state_table))
        with contextlib.ExitStack() as stack:
            if log_handle is not None:
                stack.enter_context(contextlib.redirect_stdout(log_handle))
                stack.enter_context(contextlib.redirect_stderr(log_handle))
                stack.enter_context(warnings.catch_warnings())
                warnings.simplefilter("always")
            energies, forces = run_pyseqm_batch(
                coords_np=coords,
                species_np=species,
                n_states=len(state_table),
                method=method,
                scf_eps=scf_eps,
                cis_tol=cis_tol,
                device=device,
                log_handle=log_handle,
            )
    except Exception as exc:
        if log_handle is not None:
            _write_pyseqm_log_footer(log_handle, "error", time.time() - start_time, error=exc)
            log_handle.close()
        error_metadata = {"qm_backend": "pyseqm", "qm_error": repr(exc)}
        if log_path is not None:
            error_metadata["pyseqm_log_path"] = str(log_path)
        molecule_object.update_metadata(error_metadata)
        molecule_object.set_converged_flag(False)
        return molecule_object
    finally:
        if log_handle is not None and not log_handle.closed:
            _write_pyseqm_log_footer(log_handle, "success", time.time() - start_time)
            log_handle.close()

    results: dict[str, Any] = {}
    for row in state_table:
        state_index = int(row["state"])
        results[row["energy_key"]] = float(energies[0, state_index] - offset)
        if row["force_key"] is not None:
            results[row["force_key"]] = np.asarray(forces[0, state_index], dtype=np.float64)
    for row in derive_gap_property_table(properties_list):
        lower_key = f"sE{int(row['lower_state'])}"
        upper_key = f"sE{int(row['upper_state'])}"
        results[row["gap_key"]] = float(results[upper_key] - results[lower_key])

    molecule_object.store_results(results)
    molecule_object.update_metadata(
        {
            "qm_backend": "pyseqm",
            "energy_offset_eV": float(offset),
            "n_excited_states": int(len(state_table)),
            **({"pyseqm_log_path": str(log_path)} if log_path is not None else {}),
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
    return label_excited_state_molecule(
        molecule_object=molecule_object,
        QM_config=dict(QM_config or {}),
        properties_list=dict(properties_list or {}),
        sampler_config=dict(sampler_config or {}),
        gpus_per_node=None if gpus_per_node is None else int(gpus_per_node),
    )


@python_app(executors=["alf_gpu_executor"])
def pyseqm_excited_state_gpu_task(
    molecule_object,
    QM_config,
    properties_list,
    sampler_config=None,
    gpus_per_node=None,
):
    return label_excited_state_molecule(
        molecule_object=molecule_object,
        QM_config=dict(QM_config or {}),
        properties_list=dict(properties_list or {}),
        sampler_config=dict(sampler_config or {}),
        gpus_per_node=None if gpus_per_node is None else int(gpus_per_node),
    )
