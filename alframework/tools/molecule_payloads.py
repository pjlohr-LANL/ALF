from __future__ import annotations

import pickle
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from alframework.tools.molecules_class import MoleculesObject


MOLECULE_PAYLOAD_MARKER = "__alframework_molecule_payload__"
MOLECULE_PAYLOAD_VERSION = 2
SAMPLER_RESULT_REF_MARKER = "__alframework_sampler_result_ref__"


def plain_value(value: Any, _seen: set[int] | None = None) -> Any:
    """Convert metadata/results to primitives safe for Parsl serialization."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)

    seen = set() if _seen is None else _seen
    value_id = id(value)
    if value_id in seen:
        return str(value)
    seen.add(value_id)

    if isinstance(value, np.ndarray):
        return plain_value(value.tolist(), seen)
    if hasattr(value, "detach"):
        tensor = value.detach()
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        return plain_value(np.asarray(tensor), seen)
    if isinstance(value, dict):
        return {str(plain_value(key, seen)): plain_value(item, seen) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain_value(item, seen) for item in value]
    if hasattr(value, "tolist"):
        return plain_value(value.tolist(), seen)
    return str(value)


def plain_metadata_dict(metadata: dict[str, Any]) -> dict[str, Any]:
    return {str(key): plain_value(value) for key, value in dict(metadata or {}).items()}


def clean_atoms(atoms) -> Atoms:
    """Build a calculator-free Atoms object with only portable array state."""
    clean = Atoms(
        numbers=np.asarray(atoms.get_atomic_numbers(), dtype=int),
        positions=np.asarray(atoms.get_positions(), dtype=float),
        cell=np.asarray(atoms.get_cell(), dtype=float),
        pbc=np.asarray(atoms.get_pbc(), dtype=bool),
    )
    if atoms.has("momenta"):
        clean.set_momenta(np.asarray(atoms.get_momenta(), dtype=float))
    for name, value in atoms.arrays.items():
        if name in {"numbers", "positions", "momenta"}:
            continue
        array = np.asarray(value)
        if array.dtype.kind in "biufc" and array.shape[0] == len(atoms):
            clean.set_array(str(name), array.copy())
    return clean


def _atoms_to_payload(atoms) -> dict[str, Any] | None:
    if atoms is None:
        return None
    payload = {
        "numbers": plain_value(np.asarray(atoms.get_atomic_numbers(), dtype=int)),
        "positions": plain_value(np.asarray(atoms.get_positions(), dtype=float)),
        "cell": plain_value(np.asarray(atoms.get_cell(), dtype=float)),
        "pbc": plain_value(np.asarray(atoms.get_pbc(), dtype=bool)),
        "momenta": None,
        "arrays": {},
    }
    if atoms.has("momenta"):
        payload["momenta"] = plain_value(np.asarray(atoms.get_momenta(), dtype=float))
    for name, value in atoms.arrays.items():
        if name in {"numbers", "positions", "momenta"}:
            continue
        array = np.asarray(value)
        if array.dtype.kind in "biufc" and array.shape[0] == len(atoms):
            payload["arrays"][str(name)] = plain_value(array)
    return payload


def _atoms_from_payload(payload: dict[str, Any] | None) -> Atoms | None:
    if payload is None:
        return None
    atoms = Atoms(
        numbers=np.asarray(payload["numbers"], dtype=int),
        positions=np.asarray(payload["positions"], dtype=float),
        cell=np.asarray(payload["cell"], dtype=float),
        pbc=np.asarray(payload["pbc"], dtype=bool),
    )
    momenta = payload.get("momenta")
    if momenta is not None:
        atoms.set_momenta(np.asarray(momenta, dtype=float))
    for name, value in dict(payload.get("arrays") or {}).items():
        atoms.set_array(str(name), np.asarray(value))
    return atoms


def molecule_to_payload(molecule: MoleculesObject) -> dict[str, Any]:
    if not isinstance(molecule, MoleculesObject):
        raise TypeError("molecule_to_payload expects a MoleculesObject.")
    return {
        MOLECULE_PAYLOAD_MARKER: True,
        "payload_version": MOLECULE_PAYLOAD_VERSION,
        "moleculeid": molecule.get_moleculeid(),
        "atoms": _atoms_to_payload(molecule.get_atoms()),
        "metadata": plain_metadata_dict(molecule.get_metadata()),
        "qm_results": plain_metadata_dict(molecule.get_results()),
        "converged": molecule.check_convergence(),
    }


def is_molecule_payload(value: Any) -> bool:
    return isinstance(value, dict) and value.get(MOLECULE_PAYLOAD_MARKER) is True


def payload_to_molecule(payload: dict[str, Any]) -> MoleculesObject:
    if not is_molecule_payload(payload):
        raise TypeError("payload_to_molecule expects an ALF molecule payload.")
    atoms = _atoms_from_payload(payload.get("atoms"))
    molecule = MoleculesObject(atoms if atoms is not None else Atoms(), str(payload["moleculeid"]))
    if atoms is None:
        molecule.update_atoms(None)
    metadata = payload.get("metadata") or {}
    if metadata:
        molecule.update_metadata(dict(metadata))
    qm_results = payload.get("qm_results") or {}
    if qm_results:
        molecule.store_results(dict(qm_results))
    converged = payload.get("converged")
    if converged is not None:
        molecule.set_converged_flag(bool(converged))
    return molecule


def molecule_output_to_payload(output: Any) -> Any:
    if isinstance(output, MoleculesObject):
        return molecule_to_payload(output)
    if isinstance(output, list):
        return [molecule_output_to_payload(item) for item in output]
    if isinstance(output, tuple):
        return [molecule_output_to_payload(item) for item in output]
    if is_molecule_payload(output):
        return output
    return output


def molecule_output_from_payload(output: Any) -> Any:
    if is_sampler_result_ref(output):
        output = load_sampler_result_ref(output)
    if is_molecule_payload(output):
        return payload_to_molecule(output)
    if isinstance(output, list):
        return [molecule_output_from_payload(item) for item in output]
    if isinstance(output, tuple):
        return [molecule_output_from_payload(item) for item in output]
    return output


def flatten_molecule_output(output: Any) -> list[MoleculesObject]:
    output = molecule_output_from_payload(output)
    if isinstance(output, MoleculesObject):
        return [output]
    if isinstance(output, list):
        flattened: list[MoleculesObject] = []
        for item in output:
            flattened.extend(flatten_molecule_output(item))
        return flattened
    raise TypeError("output must be a MoleculesObject, molecule payload, or nested list of either.")


def _sampler_result_dir(sampler_config: dict[str, Any] | None) -> Path:
    config = dict(sampler_config or {})
    output_dir = config.get("sampler_result_payload_dir")
    if output_dir is None:
        output_dir = Path(str(config.get("meta_dir", "sampling"))) / "task_results"
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_sampler_result_ref(value: Any) -> bool:
    return isinstance(value, dict) and value.get(SAMPLER_RESULT_REF_MARKER) is True


def write_sampler_result_ref(output: Any, sampler_config: dict[str, Any] | None) -> dict[str, Any]:
    payload = molecule_output_to_payload(output)
    output_dir = _sampler_result_dir(sampler_config)
    output_path = output_dir / f"sampler-result-{uuid.uuid4().hex}.pkl"
    with open(output_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        SAMPLER_RESULT_REF_MARKER: True,
        "status": "ok",
        "path": str(output_path),
    }


def write_sampler_error_ref(exc: BaseException, sampler_config: dict[str, Any] | None) -> dict[str, Any]:
    output_dir = _sampler_result_dir(sampler_config)
    output_path = output_dir / f"sampler-error-{uuid.uuid4().hex}.txt"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return {
        SAMPLER_RESULT_REF_MARKER: True,
        "status": "error",
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback_path": str(output_path),
    }


def load_sampler_result_ref(ref: dict[str, Any]) -> Any:
    if not is_sampler_result_ref(ref):
        return ref
    status = str(ref.get("status", "error"))
    if status != "ok":
        message = str(ref.get("message", "unknown sampler error"))
        traceback_path = str(ref.get("traceback_path", ""))
        raise RuntimeError(f"Sampler task failed inside worker: {message}. Traceback: {traceback_path}")
    with open(str(ref["path"]), "rb") as handle:
        return pickle.load(handle)
