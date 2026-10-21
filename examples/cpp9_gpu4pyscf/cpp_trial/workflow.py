"""Finite, resumable CPP9 labeling. All checkpoint writes belong to the driver."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np
from ase.io import read
from alframework.builders.h5_replay_builder import (
    _load_replay_frame, _stored_atomic_numbers, build_h5_replay_manifest,
)
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import store_current_data


PROPERTIES = {key: [key, scope, 1.0] for state in range(4)
              for key, scope in ((f"sE{state}", "system"), (f"F{state}", "atomic"))}
PROPERTIES.update({key: [key, "system", 1.0] for key in (
    "transition_dipoles", "oscillator_strengths", "absolute_state_energies")})
PROPERTIES["nacr"] = ["nacr", "pair_atomic", 1.0]
PAIRS = np.array([[1, 2], [1, 3], [2, 3]], dtype=np.int64)
UNITS = {"coordinates": "angstrom", "state_energies": "eV",
         "forces": "eV/angstrom", "transition_dipoles": "e*bohr",
         "oscillator_strengths": "dimensionless", "nacr": "angstrom^-1"}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reference(root, qm):
    root = Path(root)
    reference = json.loads((root / "reference.json").read_text())
    for name, info in reference["sources"].items():
        if file_hash(root / name) != info["sha256"]:
            raise ValueError(f"Reference snapshot hash mismatch: {name}")
    result = json.loads((root / "result.json").read_text())
    settings = json.loads((root / "effective_config.json").read_text())
    if not (result["success"] and result["converged"] and result["selected_state"] == 0
            and result["ring"] == "cpp9" and result["basis"].lower() == "6-31g"):
        raise ValueError("Reference is not a converged CPP9/6-31G S0 optimization.")
    for key in ("xc", "charge", "multiplicity", "density_fit", "auxbasis", "grids_level",
                "scf_conv_tol", "scf_max_cycle"):
        if qm[key] != settings[key]:
            raise ValueError(f"QM setting {key} differs from the energy reference.")
    if qm["basis"].lower() != "6-31g" or qm["nroots"] != 3 or not qm["compute_nacr"]:
        raise ValueError("Trial requires 6-31G, S0-S3, and compute_nacr=true.")
    if qm["cphf_conv_tol"] != settings["cphf_conv_tol"]:
        raise ValueError("Response tolerance differs from benchmark settings.")
    from pyscf.data.nist import HARTREE2EV
    energy = float(result["ground_energy_hartree"]) * float(HARTREE2EV)
    if reference["energy_hartree"] != result["ground_energy_hartree"]:
        raise ValueError("Reference energy disagrees with benchmark result.")
    if not np.isclose(qm["energy_offset_eV"], energy, rtol=0, atol=1e-8):
        raise ValueError("QM energy offset must equal the optimized S0 energy.")
    reference.update(energy_eV=energy, hartree_to_eV=float(HARTREE2EV))
    atoms = read(root / "optimized.xyz")
    if atoms.get_chemical_formula() != "C54H36" or len(atoms) != 90:
        raise ValueError("Reference geometry is not CPP9.")
    checkpoint = json.loads((root / "checkpoint.json").read_text())
    if not np.allclose(atoms.positions, checkpoint["coords_angstrom"], rtol=0, atol=1e-10):
        raise ValueError("Reference XYZ differs from the full-precision checkpoint.")
    return reference, atoms.numbers


@dataclass(frozen=True)
class Frame:
    index: int
    group: object
    local_index: int
    source_id: str


def input_frames(path, reference_numbers, limit=None):
    path = Path(path).expanduser().resolve()
    groups = build_h5_replay_manifest(str(path), 1)
    frames, seen = [], set()
    for entry in groups:
        with h5py.File(entry.shard_path, "r") as handle:
            group = handle[entry.group_path]
            if group["coordinates"].shape[1:] != (90, 3):
                raise ValueError("CPP9 coordinates must have shape [frames,90,3].")
            if "_id" not in group or group["_id"].shape != (entry.n_frames,):
                raise ValueError("Every source frame requires one unique _id.")
            for local in range(entry.n_frames):
                numbers = _stored_atomic_numbers(group["species"][()], local)
                if not np.array_equal(numbers, reference_numbers):
                    raise ValueError("CPP9 species/order differs from the benchmark reference.")
                if "topology_atom_ids" in group:
                    ids = group["topology_atom_ids"][()]
                    ids = ids[local] if ids.ndim == 2 else ids
                    if not np.array_equal(ids, np.arange(90)):
                        raise ValueError("Input must use canonical reference topology_atom_ids.")
                raw = group["_id"][local]
                source_id = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                if not source_id or source_id in seen:
                    raise ValueError("Source _id values must be nonempty and unique.")
                seen.add(source_id)
                frames.append(Frame(len(frames), entry, local, source_id))
    if limit is not None:
        if limit < 1 or limit > len(frames):
            raise ValueError("--limit must be between 1 and the source frame count.")
        frames = frames[:limit]
    return frames


def build_molecule(frame):
    atoms, metadata = _load_replay_frame(frame.group, frame.local_index,
                                       sampler_config={}, master_directory=None)
    molecule = MoleculesObject(atoms, f"cpp9-frame-{frame.index:010d}")
    molecule.update_metadata(metadata | {"source_id": frame.source_id,
                                         "source_global_index": frame.index})
    return molecule


def validate_labels(molecule, reference):
    if not isinstance(molecule, MoleculesObject) or not molecule.check_convergence():
        raise ValueError("Unconverged or invalid molecule.")
    results = molecule.get_results()
    shapes = {**{f"sE{i}": () for i in range(4)},
              **{f"F{i}": (90, 3) for i in range(4)},
              "absolute_state_energies": (4,), "transition_dipoles": (3, 3),
              "oscillator_strengths": (3,), "nacr": (3, 90, 3)}
    for key, shape in shapes.items():
        values = np.asarray(results.get(key), dtype=np.float64)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"Missing, malformed, or nonfinite {key}.")
    if np.any(np.asarray(results["oscillator_strengths"]) < 0):
        raise ValueError("Negative oscillator strength.")
    if not np.array_equal(molecule.get_metadata().get("nac_pairs"), PAIRS):
        raise ValueError("Unexpected NAC pair ordering.")
    energies = np.array([results[f"sE{i}"] for i in range(4)], dtype=np.float64)
    absolute = np.asarray(results["absolute_state_energies"], dtype=np.float64)
    if not np.allclose(energies, absolute - reference["energy_eV"], rtol=0, atol=1e-8):
        raise ValueError("Energy reference was not applied exactly once.")


def label_metrics(molecule):
    r = molecule.get_results()
    energies = np.array([r[f"sE{i}"] for i in range(4)])
    return {"energies_eV": energies, "excitation_energies_eV": energies[1:] - energies[0],
            "force_max_eV_per_angstrom": [float(np.linalg.norm(r[f"F{i}"], axis=1).max()) for i in range(4)],
            "transition_dipole_norm_au": np.linalg.norm(r["transition_dipoles"], axis=1),
            "oscillator_strengths": r["oscillator_strengths"],
            "nac_l2_norm_angstrom_inverse": np.linalg.norm(np.asarray(r["nacr"]).reshape(3, -1), axis=1)}


def persist_success(directory, frame, molecule, record, reference):
    path = directory / "h5store" / f"data-{frame.index:010d}.h5"
    temporary = path.with_suffix(".h5.partial")
    # A previous process may have stopped before publishing this frame. ALF's
    # writer creates files exclusively; discard only this unpublished partial.
    temporary.unlink(missing_ok=True)
    store_current_data(str(temporary), [molecule], PROPERTIES)
    with h5py.File(temporary, "a") as handle:
        handle.attrs["outcome_json"] = json.dumps(json_safe(record), allow_nan=False)
        handle.attrs["energy_reference_json"] = json.dumps(reference)
        handle.attrs["units_json"] = json.dumps(UNITS)
        for group in handle.values():
            group.attrs["source_id"] = frame.source_id
            group.attrs["source_group"] = frame.group.group_path
            group.attrs["source_frame_index"] = frame.local_index
        handle.flush()
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def summarize(records, total, elapsed, workers=8):
    accepted = [r for r in records.values() if r["status"] == "accepted"]
    rejected = [r for r in records.values() if r["status"] == "rejected"]
    times = [r["metadata"].get("qm_elapsed_seconds") for r in records.values()]
    times = [t for t in times if t is not None]
    statistics = {}
    if accepted:
        for key in accepted[0]["metrics"]:
            values = np.array([r["metrics"][key] for r in accepted])
            statistics[key] = {"min": values.min(axis=0), "mean": values.mean(axis=0),
                               "max": values.max(axis=0), "std": values.std(axis=0)}
    stages = {}
    for key in ("qm_scf_seconds", "qm_tda_seconds", "qm_optical_seconds", "qm_gradient_seconds", "qm_nac_seconds"):
        values = [r["metadata"][key] for r in records.values() if key in r["metadata"]]
        if values:
            stages[key] = {"mean": float(np.mean(values)), "median": float(np.median(values))}
    return json_safe({"total": total, "completed": len(records), "accepted": len(accepted),
                      "rejected": len(rejected), "elapsed_seconds_this_session": elapsed,
                      "failure_stages": dict(Counter(r["metadata"].get("qm_failed_stage", "unknown") for r in rejected)),
                      "median_calculation_seconds": float(np.median(times)) if times else None,
                      "estimated_remaining_seconds": (total - len(records)) * float(np.median(times)) / workers if times else None,
                      "estimated_frames_per_hour": workers * 3600 / float(np.mean(times)) if times and np.mean(times) > 0 else None,
                      "stage_times": stages, "label_statistics": statistics})


def label_all(*, frames, submit, directory, request, reference, resume=False,
              max_pending=8, stop=None, allow_created=False):
    """submit(molecule) -> Future; terminal outcome count controls completion."""
    directory = Path(directory)
    if directory.exists() and not resume and not (
        allow_created and all(p.name == 'runinfo' for p in directory.iterdir())
    ):
        raise FileExistsError("Run directory exists; use --resume or a new --run-dir.")
    if resume and not (directory / "request.json").exists():
        raise FileNotFoundError("Cannot resume without request.json.")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("outcomes", "h5store"):
        (directory / name).mkdir(exist_ok=True)
    if resume:
        if json.loads((directory / "request.json").read_text()) != json_safe(request):
            raise ValueError("Resume input/settings/reference do not match the original run.")
    else:
        atomic_json(directory / "request.json", request)
    records = {}
    for frame in frames:
        outcome = directory / "outcomes" / f"{frame.index:010d}.json"
        shard = directory / "h5store" / f"data-{frame.index:010d}.h5"
        if outcome.exists():
            record = json.loads(outcome.read_text())
            if record["status"] == "accepted" and not shard.exists():
                raise RuntimeError("Accepted outcome is missing its published shard.")
        elif shard.exists():
            # Recover a crash between atomic shard publication and outcome write.
            with h5py.File(shard, "r") as handle:
                record = json.loads(handle.attrs["outcome_json"])
            atomic_json(outcome, record)
        else:
            continue
        if record["source_id"] != frame.source_id or record["index"] != frame.index:
            raise ValueError("Checkpoint source identity mismatch.")
        records[frame.index] = record
    started = time.monotonic()

    def report():
        progress = summarize(records, len(frames), time.monotonic() - started, workers=max_pending)
        atomic_json(directory / "progress.json", progress)
        print(f"Labeling {progress['completed']}/{progress['total']}: "
              f"accepted={progress['accepted']} rejected={progress['rejected']} "
              f"ETA_seconds={progress['estimated_remaining_seconds']}", flush=True)

    def finish(frame, molecule=None, metadata=None):
        metadata = dict(metadata or molecule.get_metadata())
        record = {"index": frame.index, "source_id": frame.source_id, "metadata": metadata,
                  "status": "rejected"}
        if molecule is not None and molecule.check_convergence():
            try:
                validate_labels(molecule, reference)
            except (ValueError, TypeError) as exc:
                metadata.update(qm_failed_stage="output", qm_error=str(exc))
            else:
                record.update(status="accepted", metrics=label_metrics(molecule))
                persist_success(directory, frame, molecule, record, reference)
        if metadata.get("qm_failed_stage") in {"configuration", "dependency", "device"}:
            atomic_json(directory / "fatal_error.json", record)
            raise RuntimeError(f"Systematic QM failure: {metadata.get('qm_error')}")
        atomic_json(directory / "outcomes" / f"{frame.index:010d}.json", record)
        records[frame.index] = json_safe(record)
        report()

    report()
    remaining = iter(frame for frame in frames if frame.index not in records)
    pending, exhausted = {}, False
    while pending or not exhausted:
        if stop is not None and stop.is_set():
            report()
            raise InterruptedError("Driver stopped; completed labels are preserved. Use --resume.")
        while not exhausted and len(pending) < max_pending and (stop is None or not stop.is_set()):
            try:
                frame = next(remaining)
            except StopIteration:
                exhausted = True
                break
            molecule = build_molecule(frame)
            if not np.all(np.isfinite(molecule.get_atoms().positions)):
                finish(frame, metadata={"qm_failed_stage": "input", "qm_error": "Nonfinite coordinates."})
                continue
            pending[submit(molecule)] = frame
        if pending:
            completed, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in completed:
                frame = pending.pop(future)
                try:
                    molecule = future.result()
                except Exception as exc:
                    # Executor/process failures are not evidence of bad geometry.
                    atomic_json(directory / "fatal_error.json", {
                        "index": frame.index, "source_id": frame.source_id,
                        "stage": "executor", "error": f"{type(exc).__name__}: {exc}"})
                    raise RuntimeError("QM executor failed; pending frames remain resumable.") from exc
                finish(frame, molecule)
    return summarize(records, len(frames), time.monotonic() - started, workers=max_pending)
