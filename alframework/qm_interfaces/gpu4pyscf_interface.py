"""GPU4PySCF labeling for flattened excited-state energies and forces.

The public task in this module labels one nonperiodic ``MoleculesObject`` at a
time on ALF's ordinary QM executor.  It intentionally calls PySCF/GPU4PySCF
directly: the upstream ASE adapter exposes ordinary ground-state properties,
but not ALF's flattened ``sE#``/``F#`` multi-state contract.
"""

from __future__ import annotations

import importlib
import math
import numbers
import os
import time
from typing import Any

import numpy as np
from ase.data import chemical_symbols
from parsl import python_app

from alframework.tools.excited_state_tools import (
    derive_gap_property_table,
    derive_state_property_table,
)
from alframework.tools.molecules_class import MoleculesObject


DEFAULT_GPU4PYSCF_CONFIG = {
    "xc": "cam-b3lyp",
    "basis": "6-31g*",
    "charge": 0,
    "multiplicity": 1,
    "density_fit": True,
    "auxbasis": None,
    "grids_level": 3,
    "nroots": 5,
    "scf_conv_tol": 1.0e-10,
    "scf_max_cycle": 100,
    "tda_conv_tol": 1.0e-8,
    "tda_max_cycle": 100,
    "num_threads": 8,
    "max_memory_mb": None,
    "verbosity": 0,
    "energy_offset_eV": 0.0,
}


class GPU4PySCFError(RuntimeError):
    """Base class carrying the failed calculation stage and states."""

    stage = "calculation"

    def __init__(
        self,
        message: str,
        *,
        failed_states: list[int] | tuple[int, ...] = (),
        stage: str | None = None,
    ) -> None:
        self.failed_states = tuple(int(state) for state in failed_states)
        if stage is not None:
            self.stage = str(stage)
        super().__init__(message)


class GPU4PySCFDependencyError(GPU4PySCFError):
    """Raised when the CUDA-specific PySCF stack is unavailable."""

    stage = "dependency"


class GPU4PySCFDeviceError(GPU4PySCFError):
    """Raised when the QM worker cannot select a CUDA device."""

    stage = "device"


class GPU4PySCFSCFError(GPU4PySCFError):
    """Raised when the ground-state RKS calculation fails."""

    stage = "scf"


class GPU4PySCFTDAError(GPU4PySCFError):
    """Raised when one or more TDA roots fail."""

    stage = "tda"


class GPU4PySCFGradientError(GPU4PySCFError):
    """Raised when a ground- or excited-state gradient fails."""

    stage = "gradient"


class GPU4PySCFOutputError(GPU4PySCFError):
    """Raised when converged backend output is incomplete or malformed."""

    stage = "output"


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a finite positive number; received {value!r}."
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return parsed


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be finite; received {value!r}."
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed


def _integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer; received {value!r}.")
    parsed = int(value)
    if minimum is not None and parsed < int(minimum):
        raise ValueError(f"{name} must be at least {minimum}; received {parsed}.")
    return parsed


def validate_gpu4pyscf_config(QM_config: dict[str, Any] | None) -> dict[str, Any]:
    """Return a validated GPU4PySCF configuration with public defaults."""

    if QM_config is not None and not isinstance(QM_config, dict):
        raise TypeError("QM_config must be a dictionary.")
    options = dict(DEFAULT_GPU4PYSCF_CONFIG)
    options.update(dict(QM_config or {}))

    for key in ("xc", "basis"):
        value = options.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty string.")
        options[key] = value.strip()

    options["charge"] = _integer(options.get("charge"), name="charge")
    options["multiplicity"] = _integer(
        options.get("multiplicity"),
        name="multiplicity",
        minimum=1,
    )
    if options["multiplicity"] != 1:
        raise ValueError(
            "gpu4pyscf_excited_state_task currently supports only singlet "
            "RKS/TDA calculations (multiplicity=1)."
        )

    if not isinstance(options.get("density_fit"), bool):
        raise ValueError("density_fit must be a Boolean.")
    auxbasis = options.get("auxbasis")
    if auxbasis is not None:
        if not isinstance(auxbasis, str) or not auxbasis.strip():
            raise ValueError("auxbasis must be null or a nonempty string.")
        if not options["density_fit"]:
            raise ValueError("auxbasis requires density_fit=true.")
        options["auxbasis"] = auxbasis.strip()

    options["grids_level"] = _integer(
        options.get("grids_level"), name="grids_level", minimum=0
    )
    options["nroots"] = _integer(
        options.get("nroots"), name="nroots", minimum=1
    )
    options["scf_conv_tol"] = _positive_float(
        options.get("scf_conv_tol"), name="scf_conv_tol"
    )
    options["tda_conv_tol"] = _positive_float(
        options.get("tda_conv_tol"), name="tda_conv_tol"
    )
    for key in ("scf_max_cycle", "tda_max_cycle", "num_threads"):
        options[key] = _integer(options.get(key), name=key, minimum=1)
    options["verbosity"] = _integer(
        options.get("verbosity"), name="verbosity", minimum=0
    )

    memory = options.get("max_memory_mb")
    options["max_memory_mb"] = (
        None
        if memory is None
        else _positive_float(memory, name="max_memory_mb")
    )
    options["energy_offset_eV"] = _finite_float(
        options.get("energy_offset_eV"), name="energy_offset_eV"
    )
    return options


def validate_gpu4pyscf_properties(
    properties_list: dict[str, list[Any]],
    *,
    nroots: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate ALF's complete energy/force contract for ``nroots``."""

    state_table = derive_state_property_table(
        properties_list, require_forces=True
    )
    gap_table = derive_gap_property_table(properties_list)
    expected_states = int(nroots) + 1
    if len(state_table) != expected_states:
        raise ValueError(
            "GPU4PySCF property count must equal nroots + 1: "
            f"nroots={nroots} requires states 0 through {nroots}, but "
            f"properties_list defines {len(state_table)} states."
        )

    supported_keys: set[str] = set()
    for row in state_table:
        energy_key = str(row["energy_key"])
        force_key = str(row["force_key"])
        energy_schema = properties_list[energy_key]
        force_schema = properties_list[force_key]
        if len(energy_schema) < 2 or str(energy_schema[1]).lower() != "system":
            raise ValueError(
                f"Energy property {energy_key!r} must be a system property."
            )
        if len(force_schema) < 2 or str(force_schema[1]).lower() != "atomic":
            raise ValueError(
                f"Force property {force_key!r} must be an atomic property."
            )
        if not str(energy_schema[0]) or not str(force_schema[0]):
            raise ValueError(
                f"Properties {energy_key!r}/{force_key!r} require nonempty "
                "database names."
            )
        supported_keys.update((energy_key, force_key))
    supported_keys.update(str(row["gap_key"]) for row in gap_table)
    unsupported = sorted(set(properties_list) - supported_keys)
    if unsupported:
        raise ValueError(
            "The GPU4PySCF interface supports only flattened sE#/F#/dE# "
            f"properties; unsupported keys: {unsupported}."
        )
    return state_table, gap_table


def _energy_offset(
    QM_config: dict[str, Any],
    sampler_config: dict[str, Any],
) -> tuple[float, str]:
    if "energy_offset_eV" in QM_config:
        raw_offset = QM_config["energy_offset_eV"]
        source = "QM_config"
    elif "energy_offset_eV" in sampler_config:
        raw_offset = sampler_config["energy_offset_eV"]
        source = "sampler_config"
    else:
        raw_offset = 0.0
        source = "default"
    offset = _finite_float(raw_offset, name="energy_offset_eV")
    return offset, source


def _import_cupy():
    try:
        return importlib.import_module("cupy")
    except Exception as exc:
        raise GPU4PySCFDependencyError(
            "GPU4PySCF excited-state labeling requires a working CuPy "
            "installation matching the worker's CUDA runtime."
        ) from exc


def select_gpu4pyscf_device(
    gpus_per_node: int | None = None,
) -> dict[str, Any]:
    """Select one real CUDA device using ALF's Parsl worker-rank convention."""

    cupy = _import_cupy()
    try:
        visible_count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise GPU4PySCFDeviceError(
            "CuPy could not enumerate CUDA devices on the QM worker."
        ) from exc
    if visible_count < 1:
        raise GPU4PySCFDeviceError(
            "gpu4pyscf_excited_state_task requires at least one visible CUDA "
            "device; CPU fallback is intentionally disabled."
        )

    usable_count = visible_count
    if gpus_per_node is not None:
        configured_count = _integer(
            gpus_per_node, name="gpus_per_node", minimum=1
        )
        usable_count = min(visible_count, configured_count)
    try:
        worker_rank = int(os.environ.get("PARSL_WORKER_RANK", "0"))
    except (TypeError, ValueError) as exc:
        raise GPU4PySCFDeviceError(
            "PARSL_WORKER_RANK must be an integer when provided."
        ) from exc
    if worker_rank < 0:
        raise GPU4PySCFDeviceError("PARSL_WORKER_RANK cannot be negative.")

    device_index = worker_rank % usable_count
    try:
        cupy.cuda.Device(device_index).use()
    except Exception as exc:
        raise GPU4PySCFDeviceError(
            f"CuPy could not activate CUDA device {device_index}."
        ) from exc
    return {
        "index": device_index,
        "label": f"cuda:{device_index}",
        "visible_count": visible_count,
        "worker_rank": worker_rank,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _to_numpy(value: Any, cupy_module: Any | None = None) -> np.ndarray:
    """Explicitly move a possible CuPy value to host memory."""

    if cupy_module is not None:
        array_type = getattr(cupy_module, "ndarray", None)
        if array_type is not None and isinstance(value, array_type):
            return np.asarray(cupy_module.asnumpy(value))
        if type(value).__module__.split(".", 1)[0] == "cupy":
            return np.asarray(cupy_module.asnumpy(value))
    if hasattr(value, "get") and callable(value.get):
        return np.asarray(value.get())
    return np.asarray(value)


def _require_scf_convergence(mf: Any, cupy_module: Any | None = None) -> None:
    raw_flag = getattr(mf, "converged", None)
    if raw_flag is None:
        raise GPU4PySCFSCFError(
            "GPU4PySCF did not expose mf.converged after the RKS solve.",
            failed_states=(0,),
        )
    flag = _to_numpy(raw_flag, cupy_module)
    if flag.shape != () or not np.issubdtype(flag.dtype, np.bool_):
        raise GPU4PySCFSCFError(
            "GPU4PySCF mf.converged must be one Boolean scalar; "
            f"received shape {flag.shape} and dtype {flag.dtype}.",
            failed_states=(0,),
        )
    if not bool(flag):
        raise GPU4PySCFSCFError(
            "GPU4PySCF RKS did not converge.", failed_states=(0,)
        )


def _require_tda_convergence(
    td: Any,
    nroots: int,
    cupy_module: Any | None = None,
) -> None:
    root_states = tuple(range(1, int(nroots) + 1))
    raw_flags = getattr(td, "converged", None)
    if raw_flags is None:
        raise GPU4PySCFTDAError(
            "GPU4PySCF did not expose one TDA convergence flag per root.",
            failed_states=root_states,
        )
    flags = _to_numpy(raw_flags, cupy_module)
    if flags.shape != (int(nroots),) or not np.issubdtype(
        flags.dtype, np.bool_
    ):
        raise GPU4PySCFTDAError(
            "GPU4PySCF td.converged must contain one Boolean per root with "
            f"shape {(int(nroots),)}; received shape {flags.shape} and "
            f"dtype {flags.dtype}.",
            failed_states=root_states,
        )
    failed_states = (np.flatnonzero(~flags) + 1).astype(int).tolist()
    if failed_states:
        raise GPU4PySCFTDAError(
            "GPU4PySCF TDA did not converge for excited states "
            f"{failed_states}.",
            failed_states=failed_states,
        )


def convert_gpu4pyscf_atomic_units(
    energies_hartree: Any,
    gradients_hartree_per_bohr: Any,
    *,
    state_count: int,
    atom_count: int,
    hartree_to_ev: float,
    bohr_to_angstrom: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate raw state output and convert it to ALF energy/force units."""

    energies = np.asarray(energies_hartree, dtype=np.float64)
    gradients = np.asarray(
        gradients_hartree_per_bohr, dtype=np.float64
    )
    expected_energies = (int(state_count),)
    expected_gradients = (int(state_count), int(atom_count), 3)
    if energies.shape != expected_energies:
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned malformed energies: expected "
            f"{expected_energies}, received {energies.shape}."
        )
    if gradients.shape != expected_gradients:
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned malformed gradients: expected "
            f"{expected_gradients}, received {gradients.shape}."
        )
    if not np.all(np.isfinite(energies)):
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned non-finite state energies."
        )
    if not np.all(np.isfinite(gradients)):
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned non-finite state gradients."
        )
    energy_factor = _positive_float(
        hartree_to_ev, name="Hartree-to-eV conversion"
    )
    length_factor = _positive_float(
        bohr_to_angstrom, name="Bohr-to-Angstrom conversion"
    )
    return energies * energy_factor, -gradients * (
        energy_factor / length_factor
    )


def _validated_calculated_outputs(
    energies: Any,
    forces: Any,
    *,
    state_count: int,
    atom_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    energy_array = np.asarray(energies, dtype=np.float64)
    force_array = np.asarray(forces, dtype=np.float64)
    expected_energies = (int(state_count),)
    expected_forces = (int(state_count), int(atom_count), 3)
    if energy_array.shape != expected_energies:
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned malformed converted energies: expected "
            f"{expected_energies}, received {energy_array.shape}."
        )
    if force_array.shape != expected_forces:
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned malformed converted forces: expected "
            f"{expected_forces}, received {force_array.shape}."
        )
    if not np.all(np.isfinite(energy_array)):
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned non-finite converted energies."
        )
    if not np.all(np.isfinite(force_array)):
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned non-finite converted forces."
        )
    return energy_array, force_array


def run_gpu4pyscf_states(
    atoms,
    options: dict[str, Any],
    *,
    device_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Run one GPU4PySCF RKS/TDA energy-and-gradient calculation."""

    try:
        pyscf = importlib.import_module("pyscf")
        gpu4pyscf = importlib.import_module("gpu4pyscf")
        gpu_dft = importlib.import_module("gpu4pyscf.dft")
        cupy = importlib.import_module("cupy")
        nist = importlib.import_module("pyscf.data.nist")
    except Exception as exc:
        raise GPU4PySCFDependencyError(
            "GPU4PySCF excited-state labeling requires compatible PySCF, "
            "GPU4PySCF, and CuPy installations."
        ) from exc

    try:
        cupy.cuda.Device(int(device_index)).use()
        pyscf.lib.num_threads(int(options["num_threads"]))
        atom_spec = [
            (
                chemical_symbols[int(number)],
                tuple(float(value) for value in position),
            )
            for number, position in zip(
                atoms.get_atomic_numbers(), atoms.get_positions()
            )
        ]
        molecule_kwargs = {
            "atom": atom_spec,
            "basis": options["basis"],
            "charge": int(options["charge"]),
            "spin": 0,
            "unit": "Angstrom",
            "verbose": int(options["verbosity"]),
        }
        if options["max_memory_mb"] is not None:
            molecule_kwargs["max_memory"] = float(options["max_memory_mb"])
        mol = pyscf.M(**molecule_kwargs)
        mf = gpu_dft.RKS(mol, xc=options["xc"])
        if options["density_fit"]:
            if options["auxbasis"] is None:
                mf = mf.density_fit()
            else:
                mf = mf.density_fit(auxbasis=options["auxbasis"])
        to_gpu = getattr(mf, "to_gpu", None)
        if callable(to_gpu):
            mf = to_gpu()
        mf.grids.level = int(options["grids_level"])
        mf.conv_tol = float(options["scf_conv_tol"])
        mf.max_cycle = int(options["scf_max_cycle"])
        mf.kernel()
    except GPU4PySCFError:
        raise
    except Exception as exc:
        raise GPU4PySCFSCFError(
            f"GPU4PySCF RKS failed: {type(exc).__name__}: {exc}",
            failed_states=(0,),
        ) from exc
    _require_scf_convergence(mf, cupy)

    nroots = int(options["nroots"])
    root_states = tuple(range(1, nroots + 1))
    try:
        td = mf.TDA()
        td.nstates = nroots
        td.conv_tol = float(options["tda_conv_tol"])
        td.max_cycle = int(options["tda_max_cycle"])
        td.kernel()
    except GPU4PySCFError:
        raise
    except Exception as exc:
        raise GPU4PySCFTDAError(
            f"GPU4PySCF TDA failed: {type(exc).__name__}: {exc}",
            failed_states=root_states,
        ) from exc
    _require_tda_convergence(td, nroots, cupy)

    try:
        excitation_energies = np.asarray(
            _to_numpy(getattr(td, "e", None), cupy),
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned TDA roots that could not be converted to "
            "floating-point values.",
            failed_states=root_states,
        ) from exc
    if excitation_energies.shape != (nroots,):
        raise GPU4PySCFOutputError(
            "GPU4PySCF returned malformed TDA roots: expected "
            f"{(nroots,)}, received {excitation_energies.shape}.",
            failed_states=root_states,
        )
    try:
        ground_energy = np.asarray(
            _to_numpy(getattr(mf, "e_tot", None), cupy),
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise GPU4PySCFOutputError(
            "GPU4PySCF mf.e_tot could not be converted to a floating-point "
            "value.",
            failed_states=(0,),
        ) from exc
    if ground_energy.shape != ():
        raise GPU4PySCFOutputError(
            "GPU4PySCF mf.e_tot must be one scalar; received "
            f"shape {ground_energy.shape}.",
            failed_states=(0,),
        )
    energies_hartree = np.empty(nroots + 1, dtype=np.float64)
    energies_hartree[0] = float(ground_energy)
    energies_hartree[1:] = (
        float(ground_energy)
        + np.asarray(excitation_energies, dtype=np.float64)
    )

    atom_count = len(atoms)
    gradients = np.empty((nroots + 1, atom_count, 3), dtype=np.float64)
    try:
        gradients[0] = _to_numpy(
            mf.nuc_grad_method().kernel(), cupy
        ).astype(np.float64, copy=False)
    except Exception as exc:
        raise GPU4PySCFGradientError(
            f"GPU4PySCF ground-state gradient failed: "
            f"{type(exc).__name__}: {exc}",
            failed_states=(0,),
        ) from exc
    try:
        excited_gradient = td.nuc_grad_method()
    except Exception as exc:
        raise GPU4PySCFGradientError(
            "GPU4PySCF could not construct the TDA gradient method: "
            f"{type(exc).__name__}: {exc}",
            failed_states=root_states,
        ) from exc
    for state in range(1, nroots + 1):
        try:
            gradients[state] = _to_numpy(
                excited_gradient.kernel(state=state), cupy
            ).astype(np.float64, copy=False)
        except Exception as exc:
            raise GPU4PySCFGradientError(
                f"GPU4PySCF gradient failed for state S{state}: "
                f"{type(exc).__name__}: {exc}",
                failed_states=(state,),
            ) from exc

    energies_ev, forces_ev_per_angstrom = convert_gpu4pyscf_atomic_units(
        energies_hartree,
        gradients,
        state_count=nroots + 1,
        atom_count=atom_count,
        hartree_to_ev=float(nist.HARTREE2EV),
        bohr_to_angstrom=float(nist.BOHR),
    )
    versions = {
        "pyscf_version": str(getattr(pyscf, "__version__", "unknown")),
        "gpu4pyscf_version": str(
            getattr(gpu4pyscf, "__version__", "unknown")
        ),
        "cupy_version": str(getattr(cupy, "__version__", "unknown")),
    }
    return energies_ev, forces_ev_per_angstrom, versions


def _convergence_metadata_for_failure(stage: str) -> dict[str, bool]:
    return {
        "qm_scf_converged": stage in {"tda", "gradient", "output"},
        "qm_tda_converged": stage in {"gradient", "output"},
    }


def label_gpu4pyscf_excited_state_molecule(
    molecule_object: MoleculesObject,
    *,
    QM_config: dict[str, Any],
    properties_list: dict[str, list[Any]],
    sampler_config: dict[str, Any] | None = None,
    gpus_per_node: int | None = None,
) -> MoleculesObject:
    """Label one molecule with GPU4PySCF state energies and forces."""

    if not isinstance(molecule_object, MoleculesObject):
        raise TypeError("molecule_object must be a MoleculesObject instance.")

    qm_options = dict(QM_config or {})
    sampler_options = dict(sampler_config or {})
    start_time = time.time()
    options: dict[str, Any] = {}
    device: dict[str, Any] = {}
    try:
        options = validate_gpu4pyscf_config(qm_options)
        state_table, gap_table = validate_gpu4pyscf_properties(
            properties_list, nroots=options["nroots"]
        )
        atoms = molecule_object.get_atoms()
        if atoms is None or len(atoms) < 1:
            raise ValueError(
                "GPU4PySCF cannot label a molecule without atoms."
            )
        if bool(np.any(atoms.get_pbc())):
            raise NotImplementedError(
                "GPU4PySCF excited-state labeling currently supports only "
                "nonperiodic molecules."
            )
        coordinates = np.asarray(atoms.get_positions(), dtype=np.float64)
        atomic_numbers = np.asarray(
            atoms.get_atomic_numbers(), dtype=np.int64
        )
        if coordinates.shape != (len(atoms), 3) or not np.all(
            np.isfinite(coordinates)
        ):
            raise ValueError(
                "GPU4PySCF coordinates must have shape [atoms, 3] and be "
                "finite."
            )
        if np.any(atomic_numbers <= 0) or np.any(
            atomic_numbers >= len(chemical_symbols)
        ):
            raise ValueError(
                "GPU4PySCF atomic numbers must identify valid elements."
            )
        offset, offset_source = _energy_offset(
            qm_options, sampler_options
        )
        device = select_gpu4pyscf_device(gpus_per_node)
        energies, forces, backend_metadata = run_gpu4pyscf_states(
            atoms,
            options,
            device_index=int(device["index"]),
        )
        energies, forces = _validated_calculated_outputs(
            energies,
            forces,
            state_count=len(state_table),
            atom_count=len(atoms),
        )
    except Exception as exc:
        elapsed = time.time() - start_time
        stage = str(getattr(exc, "stage", "configuration"))
        failed_states = [
            int(state) for state in getattr(exc, "failed_states", ())
        ]
        error_metadata: dict[str, Any] = {
            "qm_backend": "gpu4pyscf",
            "qm_error": str(exc),
            "qm_error_type": type(exc).__name__,
            "qm_failed_stage": stage,
            "qm_elapsed_seconds": float(elapsed),
            **_convergence_metadata_for_failure(stage),
        }
        if failed_states:
            error_metadata["qm_failed_states"] = failed_states
        if device:
            error_metadata.update(
                {
                    "qm_device": str(device["label"]),
                    "qm_device_index": int(device["index"]),
                    "qm_visible_gpu_count": int(device["visible_count"]),
                    "parsl_worker_rank": int(device["worker_rank"]),
                    "cuda_visible_devices": device[
                        "cuda_visible_devices"
                    ],
                }
            )
        for property_key in dict(properties_list or {}):
            molecule_object.get_results().pop(str(property_key), None)
        molecule_object.update_metadata(error_metadata)
        molecule_object.set_converged_flag(False)
        return molecule_object

    results: dict[str, Any] = {}
    for row in state_table:
        state = int(row["state"])
        results[str(row["energy_key"])] = float(energies[state] - offset)
        results[str(row["force_key"])] = np.asarray(
            forces[state], dtype=np.float64
        )
    for row in gap_table:
        results[str(row["gap_key"])] = float(
            results[str(row["upper_energy_key"])]
            - results[str(row["lower_energy_key"])]
        )

    elapsed = time.time() - start_time
    molecule_object.store_results(results)
    molecule_object.update_metadata(
        {
            "qm_backend": "gpu4pyscf",
            "qm_device": str(device["label"]),
            "qm_device_index": int(device["index"]),
            "qm_visible_gpu_count": int(device["visible_count"]),
            "parsl_worker_rank": int(device["worker_rank"]),
            "cuda_visible_devices": device["cuda_visible_devices"],
            "qm_elapsed_seconds": float(elapsed),
            "qm_scf_converged": True,
            "qm_tda_converged": True,
            "xc": str(options["xc"]),
            "basis": str(options["basis"]),
            "charge": int(options["charge"]),
            "multiplicity": int(options["multiplicity"]),
            "density_fit": bool(options["density_fit"]),
            "auxbasis": options["auxbasis"],
            "grids_level": int(options["grids_level"]),
            "nroots": int(options["nroots"]),
            "n_states": len(state_table),
            "scf_conv_tol": float(options["scf_conv_tol"]),
            "scf_max_cycle": int(options["scf_max_cycle"]),
            "tda_conv_tol": float(options["tda_conv_tol"]),
            "tda_max_cycle": int(options["tda_max_cycle"]),
            "energy_offset_eV": float(offset),
            "energy_offset_source": offset_source,
            "gap_properties": [
                str(row["gap_key"]) for row in gap_table
            ],
            **dict(backend_metadata),
        }
    )
    molecule_object.set_converged_flag(True)
    return molecule_object


@python_app(executors=["alf_QM_executor"])
def gpu4pyscf_excited_state_task(
    molecule_object,
    QM_config,
    properties_list,
    sampler_config=None,
    gpus_per_node=None,
):
    """Label one molecule using GPU4PySCF on ALF's QM executor."""

    return label_gpu4pyscf_excited_state_molecule(
        molecule_object,
        QM_config=dict(QM_config or {}),
        properties_list=dict(properties_list or {}),
        sampler_config=dict(sampler_config or {}),
        gpus_per_node=gpus_per_node,
    )
