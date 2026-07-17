"""Configurable fixed molecular topology derived once with RDKit.

RDKit is used only while loading a reference conformer or, as a legacy
fallback, while assigning missing stable atom IDs.  Dynamics-time validation
uses only NumPy/ASE distance calculations against the cached fixed graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from itertools import combinations
import math
from pathlib import Path
from typing import Any

import numpy as np
from ase.data import chemical_symbols, covalent_radii


DEFAULT_TOPOLOGY_ATOM_ID_ARRAY = "alf_topology_atom_id"
H5_TOPOLOGY_ATOM_IDS_KEY = "topology_atom_ids"
SUPPORTED_BOND_WINDOW_MODES = frozenset({"scale", "linear"})


@dataclass(frozen=True)
class FixedMolecularTopology:
    name: str
    reference_path: str
    reference_sha256: str
    reference_format: str
    reference_charge: int
    atomic_numbers: tuple[int, ...]
    bonds: tuple[tuple[int, int], ...]
    bond_references: tuple[float, ...]

    @property
    def n_atoms(self) -> int:
        return len(self.atomic_numbers)

    @property
    def bond_set(self) -> set[tuple[int, int]]:
        return set(self.bonds)

    def to_state(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "reference_path": self.reference_path,
            "reference_sha256": self.reference_sha256,
            "reference_format": self.reference_format,
            "reference_charge": self.reference_charge,
            "atomic_numbers": list(self.atomic_numbers),
            "atom_ids": list(range(self.n_atoms)),
            "bonds": [list(pair) for pair in self.bonds],
            "bond_references": list(self.bond_references),
        }


@dataclass(frozen=True)
class TopologyValidationResult:
    valid: bool
    reason: str | None
    reason_detail: str | None
    metrics: dict[str, Any]
    violations: tuple[dict[str, Any], ...]


def topology_config(config: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict((config or {}).get("topology_check") or {})
    raw.setdefault("enabled", False)
    raw.setdefault("name", "fixed_molecular_topology")
    raw.setdefault("reference_format", "auto")
    raw.setdefault("reference_charge", 0)
    raw.setdefault("atom_id_array", DEFAULT_TOPOLOGY_ATOM_ID_ARRAY)
    raw.setdefault("bond_window_mode", "scale")
    raw.setdefault("bond_min_scale", 0.70)
    raw.setdefault("bond_max_scale", 1.35)
    raw.setdefault("nonbonded_min_scale", 0.65)
    raw.setdefault("connectivity_scale", 1.25)
    raw.setdefault("check_every_ncheck", True)
    return raw


def topology_enabled(config: dict[str, Any] | None) -> bool:
    return bool(topology_config(config).get("enabled", False))


def topology_atom_id_array(config: dict[str, Any] | None) -> str:
    return str(topology_config(config).get("atom_id_array", DEFAULT_TOPOLOGY_ATOM_ID_ARRAY))


def _normalized_reference_path(raw_path: str) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Topology reference conformer does not exist: {path}")
    return path


def _load_rdkit_reference(path: Path, reference_format: str, charge: int):
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except ImportError as exc:  # pragma: no cover - depends on deployment environment
        raise ImportError(
            "RDKit is required when topology_check.enabled=true. Install ALF's topology dependency."
        ) from exc

    selected_format = str(reference_format).strip().lower()
    if selected_format == "auto":
        selected_format = path.suffix.lower().lstrip(".")
    if selected_format == "xyz":
        molecule = Chem.MolFromXYZBlock(path.read_text(encoding="utf-8"))
        if molecule is None:
            raise ValueError(f"RDKit could not parse XYZ topology reference: {path}")
        rdDetermineBonds.DetermineBonds(molecule, charge=int(charge))
    elif selected_format in {"mol", "mdl"}:
        molecule = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
    elif selected_format == "sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
        molecule = next((item for item in supplier if item is not None), None)
    else:
        raise ValueError(
            f"Unsupported topology reference format {selected_format!r}. Use XYZ, MOL, or SDF."
        )
    if molecule is None:
        raise ValueError(f"RDKit could not parse topology reference conformer: {path}")
    if molecule.GetNumConformers() < 1:
        raise ValueError(f"Topology reference conformer has no coordinates: {path}")
    return molecule, selected_format


@lru_cache(maxsize=16)
def _load_fixed_topology_cached(
    reference_path: str,
    reference_format: str,
    reference_charge: int,
    name: str,
) -> FixedMolecularTopology:
    path = Path(reference_path)
    molecule, selected_format = _load_rdkit_reference(path, reference_format, reference_charge)
    conformer = molecule.GetConformer()
    positions = np.asarray(conformer.GetPositions(), dtype=np.float64)
    atomic_numbers = tuple(int(atom.GetAtomicNum()) for atom in molecule.GetAtoms())
    if len(atomic_numbers) < 2:
        raise ValueError("Fixed molecular topology requires at least two atoms.")
    if positions.shape != (len(atomic_numbers), 3) or not np.isfinite(positions).all():
        raise ValueError("Topology reference coordinates must be finite with shape (n_atoms, 3).")
    bonds = tuple(
        sorted(
            (min(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())),
             max(int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())))
            for bond in molecule.GetBonds()
        )
    )
    if not bonds:
        raise ValueError("RDKit did not find any bonds in the topology reference conformer.")
    bond_references = tuple(float(np.linalg.norm(positions[i] - positions[j])) for i, j in bonds)
    return FixedMolecularTopology(
        name=str(name),
        reference_path=str(path),
        reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        reference_format=selected_format,
        reference_charge=int(reference_charge),
        atomic_numbers=atomic_numbers,
        bonds=bonds,
        bond_references=bond_references,
    )


def load_fixed_topology(config: dict[str, Any] | None) -> FixedMolecularTopology | None:
    settings = topology_config(config)
    if not bool(settings["enabled"]):
        return None
    raw_path = settings.get("reference_conformer_path")
    if not raw_path:
        raise ValueError("topology_check.reference_conformer_path is required when topology checking is enabled.")
    mode = str(settings["bond_window_mode"]).strip().lower()
    if mode not in SUPPORTED_BOND_WINDOW_MODES:
        raise ValueError(
            "topology_check.bond_window_mode must be one of "
            f"{sorted(SUPPORTED_BOND_WINDOW_MODES)}, found {mode!r}."
        )
    if mode == "scale":
        bond_min = float(settings["bond_min_scale"])
        bond_max = float(settings["bond_max_scale"])
        if not math.isfinite(bond_min) or not math.isfinite(bond_max):
            raise ValueError("Topology bond scales must be finite.")
        if bond_min <= 0.0 or bond_max < bond_min:
            raise ValueError("Topology bond scales must satisfy 0 < bond_min_scale <= bond_max_scale.")
    else:
        missing = [
            key
            for key in ("bond_compression_tolerance_A", "bond_extension_tolerance_A")
            if key not in settings
        ]
        if missing:
            raise ValueError(
                "Linear topology bond windows require configuration keys: " + ", ".join(missing)
            )
        compression = float(settings["bond_compression_tolerance_A"])
        extension = float(settings["bond_extension_tolerance_A"])
        if not math.isfinite(compression) or not math.isfinite(extension):
            raise ValueError("Linear topology bond tolerances must be finite.")
        if compression < 0.0 or extension < 0.0:
            raise ValueError("Linear topology bond tolerances must be non-negative.")
    nonbonded_min = float(settings["nonbonded_min_scale"])
    connectivity = float(settings["connectivity_scale"])
    if not math.isfinite(nonbonded_min) or not math.isfinite(connectivity):
        raise ValueError("Topology nonbonded and connectivity scales must be finite.")
    if nonbonded_min <= 0.0 or connectivity <= 0.0:
        raise ValueError("Topology nonbonded and connectivity scales must be positive.")
    path = _normalized_reference_path(str(raw_path))
    topology = _load_fixed_topology_cached(
        str(path),
        str(settings["reference_format"]),
        int(settings["reference_charge"]),
        str(settings["name"]),
    )
    if mode == "linear" and compression >= min(topology.bond_references):
        raise ValueError(
            "bond_compression_tolerance_A must be smaller than every reference bond length "
            "so that all linear bond-window lower bounds remain positive."
        )
    return topology


def _reference_query(topology: FixedMolecularTopology):
    from rdkit import Chem

    editable = Chem.RWMol()
    for number in topology.atomic_numbers:
        editable.AddAtom(Chem.Atom(int(number)))
    for first, second in topology.bonds:
        editable.AddBond(int(first), int(second), Chem.BondType.SINGLE)
    return editable.GetMol()


def _infer_atom_ids_with_rdkit(atoms, topology: FixedMolecularTopology) -> np.ndarray:
    """Infer a one-time reference-ID mapping for legacy structures lacking IDs."""

    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    lines = [str(len(atoms)), ""]
    for number, (x, y, z) in zip(atoms.get_atomic_numbers(), atoms.get_positions()):
        lines.append(f"{chemical_symbols[int(number)]} {x:.12f} {y:.12f} {z:.12f}")
    molecule = Chem.MolFromXYZBlock("\n".join(lines) + "\n")
    if molecule is None:
        raise ValueError("RDKit could not parse the starting structure while assigning topology atom IDs.")
    rdDetermineBonds.DetermineConnectivity(molecule)
    query = _reference_query(topology)
    matches = molecule.GetSubstructMatches(query, uniquify=False, maxMatches=100000)
    exact_matches = [match for match in matches if len(match) == topology.n_atoms]
    if not exact_matches:
        raise ValueError("Starting structure connectivity does not match the fixed reference topology.")

    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    refs = np.asarray(topology.bond_references, dtype=np.float64)

    def match_cost(match: tuple[int, ...]) -> float:
        mapped = np.asarray(match, dtype=np.int64)
        distances = np.asarray(
            [np.linalg.norm(positions[mapped[first]] - positions[mapped[second]]) for first, second in topology.bonds],
            dtype=np.float64,
        )
        return float(np.square((distances - refs) / refs).sum())

    match = min(exact_matches, key=match_cost)
    atom_ids = np.empty(topology.n_atoms, dtype=np.int64)
    for reference_id, current_index in enumerate(match):
        atom_ids[int(current_index)] = int(reference_id)
    return atom_ids


def ensure_topology_atom_ids(
    atoms,
    config: dict[str, Any] | None,
    topology: FixedMolecularTopology | None = None,
    *,
    allow_inference: bool = True,
) -> np.ndarray | None:
    topology = topology if topology is not None else load_fixed_topology(config)
    if topology is None:
        return None
    array_name = topology_atom_id_array(config)
    if atoms.has(array_name):
        atom_ids = np.asarray(atoms.get_array(array_name), dtype=np.int64).reshape(-1)
    else:
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        reference_numbers = np.asarray(topology.atomic_numbers, dtype=np.int64)
        if np.array_equal(numbers, reference_numbers):
            atom_ids = np.arange(topology.n_atoms, dtype=np.int64)
        elif allow_inference:
            atom_ids = _infer_atom_ids_with_rdkit(atoms, topology)
        else:
            raise ValueError(f"Atoms are missing required topology ID array {array_name!r}.")
        atoms.set_array(array_name, atom_ids)
    if atom_ids.shape != (topology.n_atoms,) or len(np.unique(atom_ids)) != topology.n_atoms:
        raise ValueError("Topology atom IDs must be unique and contain one value per reference atom.")
    if set(atom_ids.tolist()) != set(range(topology.n_atoms)):
        raise ValueError("Topology atom IDs do not match the reference atom-ID set.")
    expected = np.asarray(topology.atomic_numbers, dtype=np.int64)
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    for current_index, atom_id in enumerate(atom_ids):
        if int(numbers[current_index]) != int(expected[int(atom_id)]):
            raise ValueError(
                f"Atomic species changed for topology atom ID {int(atom_id)}: "
                f"expected Z={int(expected[int(atom_id)])}, found Z={int(numbers[current_index])}."
            )
    return atom_ids


def _invalid_result(reason: str, detail: str) -> TopologyValidationResult:
    violation = {"kind": str(reason), "detail": str(detail)}
    return TopologyValidationResult(False, str(reason), str(detail), {}, (violation,))


def validate_fixed_topology(
    atoms,
    config: dict[str, Any] | None,
    topology: FixedMolecularTopology | None = None,
    *,
    assign_missing_ids: bool = False,
) -> TopologyValidationResult:
    topology = topology if topology is not None else load_fixed_topology(config)
    if topology is None:
        return TopologyValidationResult(True, None, None, {}, ())
    settings = topology_config(config)
    try:
        atom_ids = ensure_topology_atom_ids(
            atoms,
            config,
            topology,
            allow_inference=bool(assign_missing_ids),
        )
    except (ImportError, TypeError, ValueError) as exc:
        return _invalid_result("topology_identity", str(exc))

    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    if positions.shape != (topology.n_atoms, 3) or not np.isfinite(positions).all():
        return _invalid_result("topology_nonfinite", "Topology coordinates are non-finite or have the wrong shape.")
    current_by_id = np.empty(topology.n_atoms, dtype=np.int64)
    for current_index, atom_id in enumerate(np.asarray(atom_ids, dtype=np.int64)):
        current_by_id[int(atom_id)] = int(current_index)
    positions_by_id = positions[current_by_id]
    numbers = np.asarray(topology.atomic_numbers, dtype=np.int64)
    distances = np.linalg.norm(positions_by_id[:, None, :] - positions_by_id[None, :, :], axis=-1)

    bond_window_mode = str(settings["bond_window_mode"]).strip().lower()
    bond_min = float(settings["bond_min_scale"])
    bond_max = float(settings["bond_max_scale"])
    compression_tolerance = (
        float(settings["bond_compression_tolerance_A"])
        if bond_window_mode == "linear"
        else None
    )
    extension_tolerance = (
        float(settings["bond_extension_tolerance_A"])
        if bond_window_mode == "linear"
        else None
    )
    nonbond_min = float(settings["nonbonded_min_scale"])
    connectivity = float(settings["connectivity_scale"])
    bond_set = topology.bond_set
    violations: list[dict[str, Any]] = []
    bond_ratios: list[float] = []
    bond_deviations: list[float] = []
    for (first, second), reference in zip(topology.bonds, topology.bond_references):
        distance = float(distances[first, second])
        ratio = distance / float(reference)
        deviation = distance - float(reference)
        bond_ratios.append(ratio)
        bond_deviations.append(deviation)
        if bond_window_mode == "linear":
            allowed_min = float(reference) - float(compression_tolerance)
            allowed_max = float(reference) + float(extension_tolerance)
            outside_window = distance < allowed_min or distance > allowed_max
        else:
            allowed_min = float(reference) * bond_min
            allowed_max = float(reference) * bond_max
            outside_window = ratio < bond_min or ratio > bond_max
        if outside_window:
            violation = {
                "kind": "topology_bond_window",
                "bond_window_mode": bond_window_mode,
                "atom_ids": [int(first), int(second)],
                "distance_A": distance,
                "reference_distance_A": float(reference),
                "deviation_A": deviation,
                "ratio": ratio,
                "allowed_distance_A": [allowed_min, allowed_max],
            }
            if bond_window_mode == "linear":
                violation["allowed_deviation_A"] = [
                    -float(compression_tolerance),
                    float(extension_tolerance),
                ]
            else:
                violation["allowed_ratio"] = [bond_min, bond_max]
            violations.append(violation)

    nonbond_scaled: list[float] = []
    for first, second in combinations(range(topology.n_atoms), 2):
        if (first, second) in bond_set:
            continue
        radius_sum = float(covalent_radii[numbers[first]] + covalent_radii[numbers[second]])
        distance = float(distances[first, second])
        scaled = distance / radius_sum
        nonbond_scaled.append(scaled)
        if scaled < nonbond_min:
            violations.append(
                {
                    "kind": "topology_nonbonded_close",
                    "atom_ids": [int(first), int(second)],
                    "distance_A": distance,
                    "covalent_radius_ratio": scaled,
                    "minimum_ratio": nonbond_min,
                }
            )
        if scaled <= connectivity:
            violations.append(
                {
                    "kind": "topology_extra_bond",
                    "atom_ids": [int(first), int(second)],
                    "distance_A": distance,
                    "covalent_radius_ratio": scaled,
                    "connectivity_scale": connectivity,
                }
            )

    metrics = {
        "topology_name": topology.name,
        "topology_reference_sha256": topology.reference_sha256,
        "topology_bond_window_mode": bond_window_mode,
        "topology_min_bond_ratio": float(min(bond_ratios)),
        "topology_max_bond_ratio": float(max(bond_ratios)),
        "topology_min_bond_deviation_A": float(min(bond_deviations)),
        "topology_max_bond_deviation_A": float(max(bond_deviations)),
        "topology_min_nonbonded_covalent_ratio": float(min(nonbond_scaled)),
        "topology_violation_count": int(len(violations)),
    }
    if not violations:
        return TopologyValidationResult(True, None, None, metrics, ())
    first = violations[0]
    kind = str(first["kind"])
    atom_pair = first.get("atom_ids")
    detail = f"{kind} for topology atom IDs {atom_pair}"
    return TopologyValidationResult(False, kind, detail, metrics, tuple(violations))


def topology_metadata(result: TopologyValidationResult) -> dict[str, Any]:
    return {
        "topology_valid": bool(result.valid),
        "topology_reject_reason": result.reason,
        "topology_reject_detail": result.reason_detail,
        "topology_violations": [dict(item) for item in result.violations],
        **dict(result.metrics),
    }
