from __future__ import annotations

import os
from typing import Any

import numpy as np
from parsl import python_app

from alframework.tools.excited_state_tools import derive_state_property_table
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


def run_pyseqm_batch(
    coords_np: np.ndarray,
    species_np: np.ndarray,
    n_states: int,
    *,
    method: str = "AM1",
    scf_eps: float = 1e-10,
    cis_tol: float = 1e-8,
    device=None,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    from seqm.ElectronicStructure import Electronic_Structure
    from seqm.Molecule import Molecule
    from seqm.seqm_functions.constants import Constants

    torch.set_default_dtype(torch.float64)

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

    coords_np = np.ascontiguousarray(coords_np).copy()
    species_np = np.ascontiguousarray(species_np).copy()
    coords_np, species_np, sort_idx = _prepare_pyseqm_inputs(coords_np, species_np)

    coords = torch.from_numpy(coords_np).pin_memory() if device.type == "cuda" else torch.from_numpy(coords_np)
    species = torch.from_numpy(species_np).pin_memory() if device.type == "cuda" else torch.from_numpy(species_np)
    coords = coords.to(device, non_blocking=(device.type == "cuda"))
    species = species.to(device, non_blocking=(device.type == "cuda"))

    params = {
        "method": method,
        "scf_eps": float(scf_eps),
        "scf_converger": [2],
        "excited_states": {"n_states": max(0, int(n_states) - 1), "cis_tol": float(cis_tol)},
        "analytical_gradient": [True],
        "do_all_forces": True,
    }

    const = Constants().to(device)
    molecule = Molecule(const, params, coords, species).to(device)
    driver = Electronic_Structure(params).to(device)
    driver(molecule)

    batch = molecule.nmol
    all_energies = torch.empty((batch, int(n_states)), dtype=torch.float64, device=device)
    all_energies[:, 0] = molecule.Etot
    if int(n_states) > 1:
        all_energies[:, 1:] = molecule.Etot.unsqueeze(1) + molecule.cis_energies

    energies = all_energies.detach().cpu().numpy()
    forces = _restore_pyseqm_force_order(molecule.all_forces.detach().cpu().numpy(), sort_idx)

    del coords, species, const, molecule, driver, all_energies
    if device.type == "cuda":
        torch.cuda.empty_cache()

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

    state_table = derive_state_property_table(properties_list, require_forces=False)
    atoms = molecule_object.get_atoms()
    coords = np.asarray(atoms.get_positions(), dtype=np.float64)[None, ...]
    species = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)[None, ...]
    offset = float((sampler_config or {}).get("energy_offset_eV", 0.0))
    try:
        energies, forces = run_pyseqm_batch(
            coords_np=coords,
            species_np=species,
            n_states=len(state_table),
            method=str(QM_config.get("method", "AM1")),
            scf_eps=float(QM_config.get("scf_eps", 1e-10)),
            cis_tol=float(QM_config.get("cis_tol", 1e-8)),
            device=_pyseqm_device(gpus_per_node),
        )
    except Exception as exc:
        molecule_object.update_metadata({"qm_backend": "pyseqm", "qm_error": repr(exc)})
        molecule_object.set_converged_flag(False)
        return molecule_object

    results: dict[str, Any] = {}
    for row in state_table:
        state_index = int(row["state"])
        results[row["energy_key"]] = float(energies[0, state_index] - offset)
        if row["force_key"] is not None:
            results[row["force_key"]] = np.asarray(forces[0, state_index], dtype=np.float64)

    molecule_object.store_results(results)
    molecule_object.update_metadata(
        {
            "qm_backend": "pyseqm",
            "energy_offset_eV": float(offset),
            "n_excited_states": int(len(state_table)),
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
