"""Example-local CPP9 optical extension of ALF's GPU4PySCF interface.

Adapted from cpp9_trial_001; see inputs/provenance.json and LICENSE_ALF.
Shared validation, errors, device selection, and unit conversion come from ALF.
The calculation and label assembly below preserve the trial's optical/NAC path;
this example does not change ALF's public QM implementation.
"""
from __future__ import annotations
import importlib
import time
from typing import Any

import numpy as np
from ase.data import chemical_symbols
from parsl import python_app

from alframework.tools.excited_state_tools import derive_gap_property_table, derive_state_property_table
from alframework.tools.molecules_class import MoleculesObject
from alframework.qm_interfaces.gpu4pyscf_interface import (
    DEFAULT_GPU4PYSCF_CONFIG,
    NACR_PROPERTY_KEY,
    NAC_SCOPE,
    NACR_UNITS,
    excited_excited_nac_pairs,
    GPU4PySCFError,
    GPU4PySCFDependencyError,
    GPU4PySCFDeviceError,
    GPU4PySCFSCFError,
    GPU4PySCFTDAError,
    GPU4PySCFNACError,
    GPU4PySCFGradientError,
    GPU4PySCFOutputError,
    _positive_float,
    _finite_float,
    _integer,
    validate_gpu4pyscf_config,
    _energy_offset,
    _import_cupy,
    select_gpu4pyscf_device,
    _to_numpy,
    _require_scf_convergence,
    _require_tda_convergence,
    convert_gpu4pyscf_atomic_units,
    _validated_calculated_outputs,
    _validated_nac_outputs,
    _convergence_metadata_for_failure,
)


def validate_gpu4pyscf_properties(
    properties_list: dict[str, list[Any]],
    *,
    nroots: int,
    compute_nacr: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate ALF's complete energy/force contract for ``nroots``.

    When ``compute_nacr`` is set, ``properties_list`` must also declare the
    ``nacr`` pair_atomic property, and vice versa. Requiring both together
    prevents a configuration that computes coupling with nowhere to store it, or
    that declares storage for data that is never produced.
    """

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

    nacr_declared = NACR_PROPERTY_KEY in properties_list
    if bool(compute_nacr) and not nacr_declared:
        raise ValueError(
            f"compute_nacr requires properties_list to declare the "
            f"{NACR_PROPERTY_KEY!r} pair_atomic property so coupling vectors "
            "can be stored."
        )
    if nacr_declared and not bool(compute_nacr):
        raise ValueError(
            f"properties_list declares {NACR_PROPERTY_KEY!r} but "
            "QM_config.compute_nacr is false, so no coupling vectors would be "
            "calculated."
        )
    if nacr_declared:
        nacr_schema = properties_list[NACR_PROPERTY_KEY]
        if len(nacr_schema) < 2 or str(nacr_schema[1]).lower() != "pair_atomic":
            raise ValueError(
                f"Property {NACR_PROPERTY_KEY!r} must use the 'pair_atomic' "
                "scope."
            )
        if not str(nacr_schema[0]):
            raise ValueError(
                f"Property {NACR_PROPERTY_KEY!r} requires a nonempty database "
                "name."
            )
        supported_keys.add(NACR_PROPERTY_KEY)

    optical_keys = {"transition_dipoles", "oscillator_strengths", "absolute_state_energies"}
    for key in optical_keys:
        if key not in properties_list or properties_list[key][1] != "system":
            raise ValueError(f"CPP trial requires system property {key!r}.")
    supported_keys.update(optical_keys)
    unsupported = sorted(set(properties_list) - supported_keys)
    if unsupported:
        raise ValueError(
            "The GPU4PySCF interface supports only flattened sE#/F#/dE# "
            f"properties and {NACR_PROPERTY_KEY!r}; unsupported keys: "
            f"{unsupported}."
        )
    return state_table, gap_table



def _validated_optical_outputs(dipoles, strengths, *, nroots):
    dipoles = np.asarray(dipoles, dtype=np.float64)
    strengths = np.asarray(strengths, dtype=np.float64)
    if dipoles.shape != (nroots, 3) or strengths.shape != (nroots,):
        raise GPU4PySCFOutputError("Malformed transition dipoles/oscillator strengths.")
    if not np.all(np.isfinite(dipoles)) or not np.all(np.isfinite(strengths)):
        raise GPU4PySCFOutputError("Nonfinite transition dipoles/oscillator strengths.")
    if np.any(strengths < 0):
        raise GPU4PySCFOutputError("Negative oscillator strength.")
    return dipoles, strengths



def transition_dipoles_staged(td, cupy):
    """Two-stage AO->occupied->virtual contraction used by the CPP benchmark.

    Avoid the large intermediate in a single three-operand contraction.
    Length-gauge moments are in atomic units; retain the solver's phase.
    """
    from pyscf.tdscf.rhf import _charge_center
    mol = td.mol
    with mol.with_common_orig(_charge_center(mol)):
        ao = cupy.asarray(mol.intor_symmetric("int1e_r", comp=3))
    coefficients = cupy.asarray(td._scf.mo_coeff)
    occupations = cupy.asarray(td._scf.mo_occ)
    occupied = coefficients[:, occupations == 2]
    virtual = coefficients[:, occupations == 0]
    temporary = cupy.einsum("xpq,pi->xiq", ao, occupied.conj(), optimize=True)
    ov = cupy.einsum("xiq,qj->xij", temporary, virtual, optimize=True)
    del temporary, ao
    xs = cupy.asarray([x for x, _ in td.xy])
    moments = cupy.einsum("xij,nij->nx", ov, xs, optimize=True) * 2.0
    if isinstance(td.xy[0][1], (np.ndarray, cupy.ndarray)):
        ys = cupy.asarray([y for _, y in td.xy])
        moments += cupy.einsum("xij,nij->nx", ov, ys, optimize=True) * 2.0
    return _to_numpy(moments, cupy)



def run_gpu4pyscf_states(
    atoms,
    options: dict[str, Any],
    *,
    device_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Run one GPU4PySCF RKS/TDA energy-and-gradient calculation.

    Returns state energies (eV), forces (eV/Angstrom), NACR (Angstrom^-1, empty
    unless ``compute_nacr``), the NACR pair index, and backend metadata.
    """

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

    scf_start = time.time()
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
    scf_seconds = time.time() - scf_start
    tda_start = time.time()

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
    tda_seconds = time.time() - tda_start

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

    optical_start = time.time()
    try:
        transition_dipoles = transition_dipoles_staged(td, cupy)
        oscillator_strengths = (2.0 / 3.0) * excitation_energies * np.einsum(
            "si,si->s", transition_dipoles, transition_dipoles
        )
        transition_dipoles, oscillator_strengths = _validated_optical_outputs(
            transition_dipoles, oscillator_strengths, nroots=nroots
        )
    except Exception as exc:
        error = GPU4PySCFOutputError(f"Optical properties failed: {type(exc).__name__}: {exc}")
        error.stage = "optical"
        raise error from exc
    optical_seconds = time.time() - optical_start
    gradient_start = time.time()
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
    compute_nacr = bool(options.get("compute_nacr", False))
    expected_pairs = (
        excited_excited_nac_pairs(nroots) if compute_nacr else ()
    )
    nac_pairs = np.asarray(expected_pairs, dtype=np.int64).reshape(-1, 2)
    nacr_bohr_inverse = np.empty((0, atom_count, 3), dtype=np.float64)
    nac_elapsed: float | None = None
    # The NAC solver returns the S1 gradient as a byproduct of grad_state=1, so
    # the direct TDA gradient loop starts at S2 when coupling is requested.
    first_direct_state = 1
    if compute_nacr:
        nac_start = time.time()
        try:
            nac_solver = td.nac_gradient_method()
            nac_solver.states = list(root_states)
            nac_solver.grad_state = 1
            nac_solver.cphf_conv_tol = float(options["cphf_conv_tol"])
            nac_results = nac_solver.kernel()
        except Exception as exc:
            raise GPU4PySCFNACError(
                f"GPU4PySCF nonadiabatic coupling failed: "
                f"{type(exc).__name__}: {exc}",
                failed_states=root_states,
            ) from exc
        nac_elapsed = time.time() - nac_start
        try:
            gradients[1] = np.asarray(
                _to_numpy(nac_solver.grad_result, cupy), dtype=np.float64
            )
        except Exception as exc:
            raise GPU4PySCFNACError(
                "GPU4PySCF NAC solver did not return a usable S1 gradient: "
                f"{type(exc).__name__}: {exc}",
                failed_states=(1,),
            ) from exc
        returned_pairs = sorted(
            tuple(int(state) for state in pair) for pair in nac_results
        )
        if returned_pairs != sorted(expected_pairs):
            raise GPU4PySCFNACError(
                f"GPU4PySCF returned NAC pairs {returned_pairs}; expected "
                f"{sorted(expected_pairs)}.",
                failed_states=root_states,
            )
        try:
            nacr_bohr_inverse = np.stack(
                [
                    np.asarray(
                        _to_numpy(nac_results[pair]["de_etf_scaled"], cupy),
                        dtype=np.float64,
                    )
                    for pair in expected_pairs
                ]
            )
        except Exception as exc:
            raise GPU4PySCFNACError(
                "GPU4PySCF NAC results could not be converted to arrays: "
                f"{type(exc).__name__}: {exc}",
                failed_states=root_states,
            ) from exc
        first_direct_state = 2

    try:
        excited_gradient = td.nuc_grad_method()
        excited_gradient.cphf_conv_tol = float(options["cphf_conv_tol"])
    except Exception as exc:
        raise GPU4PySCFGradientError(
            "GPU4PySCF could not construct the TDA gradient method: "
            f"{type(exc).__name__}: {exc}",
            failed_states=root_states,
        ) from exc
    for state in range(first_direct_state, nroots + 1):
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

    # de_etf_scaled is bohr^-1; ALF stores angstrom^-1 alongside its
    # angstrom-based coordinates and eV/angstrom forces.
    nacr_angstrom_inverse = nacr_bohr_inverse / float(nist.BOHR)

    energies_ev, forces_ev_per_angstrom = convert_gpu4pyscf_atomic_units(
        energies_hartree,
        gradients,
        state_count=nroots + 1,
        atom_count=atom_count,
        hartree_to_ev=float(nist.HARTREE2EV),
        bohr_to_angstrom=float(nist.BOHR),
    )
    versions = {
        "_transition_dipoles": transition_dipoles,
        "_oscillator_strengths": oscillator_strengths,
        "qm_scf_seconds": scf_seconds,
        "qm_tda_seconds": tda_seconds,
        "qm_optical_seconds": optical_seconds,
        "qm_gradient_seconds": time.time() - gradient_start - (nac_elapsed or 0.0),
        "pyscf_version": str(getattr(pyscf, "__version__", "unknown")),
        "gpu4pyscf_version": str(
            getattr(gpu4pyscf, "__version__", "unknown")
        ),
        "cupy_version": str(getattr(cupy, "__version__", "unknown")),
        "s1_gradient_source": (
            "nac_solver" if compute_nacr else "tda_gradient"
        ),
    }
    if nac_elapsed is not None:
        versions["qm_nac_seconds"] = float(nac_elapsed)
    return (
        energies_ev,
        forces_ev_per_angstrom,
        nacr_angstrom_inverse,
        nac_pairs,
        versions,
    )



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
            properties_list,
            nroots=options["nroots"],
            compute_nacr=bool(options["compute_nacr"]),
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
        (
            energies,
            forces,
            nacr,
            nac_pairs,
            backend_metadata,
        ) = run_gpu4pyscf_states(
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
        if options["compute_nacr"]:
            nacr, nac_pairs = _validated_nac_outputs(
                nacr,
                nac_pairs,
                nroots=int(options["nroots"]),
                atom_count=len(atoms),
            )
        backend_metadata = dict(backend_metadata)
        try:
            dipoles, strengths = _validated_optical_outputs(
                backend_metadata.pop("_transition_dipoles"),
                backend_metadata.pop("_oscillator_strengths"),
                nroots=int(options["nroots"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GPU4PySCFOutputError("Missing or invalid optical output.") from exc
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

    results: dict[str, Any] = {
        "absolute_state_energies": np.asarray(energies, dtype=np.float64).copy(),
        "transition_dipoles": dipoles,
        "oscillator_strengths": strengths,
    }
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
    if options["compute_nacr"]:
        results[NACR_PROPERTY_KEY] = np.asarray(nacr, dtype=np.float64)

    elapsed = time.time() - start_time
    nac_metadata: dict[str, Any] = {
        "compute_nacr": bool(options["compute_nacr"]),
        "qm_nac_converged": True,
    }
    if options["compute_nacr"]:
        nac_metadata.update(
            {
                # store_pair_index reads nac_pairs to write the shard's pair
                # index dataset, so this key is load-bearing, not just record.
                "nac_pairs": [
                    [int(pair[0]), int(pair[1])] for pair in nac_pairs
                ],
                "nac_scope": NAC_SCOPE,
                "nacr_units": NACR_UNITS,
                "nac_pair_count": int(len(nac_pairs)),
                # NACR carries an arbitrary global sign per pair; nothing is
                # trained on it, so the raw solver sign is stored unaligned.
                "nacr_sign_convention": "raw_solver_arbitrary_global_sign",
            }
        )
    molecule_object.store_results(results)
    molecule_object.update_metadata(
        {
            **nac_metadata,
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
