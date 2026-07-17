from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from alframework.tools.molecular_topology import (
    FixedMolecularTopology,
    load_fixed_topology,
    topology_config,
    validate_fixed_topology,
)


def _topology() -> FixedMolecularTopology:
    return FixedMolecularTopology(
        name="three-carbon-chain",
        reference_path="synthetic.mol",
        reference_sha256="synthetic",
        reference_format="mol",
        reference_charge=0,
        atomic_numbers=(6, 6, 6),
        bonds=((0, 1), (1, 2)),
        bond_references=(1.5, 1.5),
    )


def _atoms(first_distance: float = 1.5, second_distance: float = 1.5) -> Atoms:
    atoms = Atoms(
        numbers=[6, 6, 6],
        positions=[[0.0, 0.0, 0.0], [first_distance, 0.0, 0.0],
                   [first_distance + second_distance, 0.0, 0.0]],
    )
    atoms.set_array("alf_topology_atom_id", np.arange(3, dtype=np.int64))
    return atoms


def _linear_config(**overrides):
    topology = {
        "enabled": True,
        "bond_window_mode": "linear",
        "bond_compression_tolerance_A": 0.15,
        "bond_extension_tolerance_A": 0.25,
        "nonbonded_min_scale": 0.65,
        "connectivity_scale": 1.25,
    }
    topology.update(overrides)
    return {"topology_check": topology}


def test_scale_mode_remains_the_default():
    config = {
        "topology_check": {
            "enabled": True,
            "bond_min_scale": 0.9,
            "bond_max_scale": 1.1,
        }
    }

    assert topology_config(config)["bond_window_mode"] == "scale"
    assert validate_fixed_topology(_atoms(1.35, 1.65), config, _topology()).valid


def test_linear_window_accepts_exact_boundaries():
    result = validate_fixed_topology(_atoms(1.35, 1.75), _linear_config(), _topology())

    assert result.valid
    assert result.metrics["topology_bond_window_mode"] == "linear"
    assert result.metrics["topology_min_bond_deviation_A"] == pytest.approx(-0.15)
    assert result.metrics["topology_max_bond_deviation_A"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("first_distance", "expected_deviation"),
    [(1.349, -0.151), (1.751, 0.251)],
)
def test_linear_window_rejects_outside_boundaries(first_distance, expected_deviation):
    result = validate_fixed_topology(
        _atoms(first_distance, 1.5),
        _linear_config(),
        _topology(),
    )

    assert not result.valid
    violation = next(item for item in result.violations if item["kind"] == "topology_bond_window")
    assert violation["bond_window_mode"] == "linear"
    assert violation["deviation_A"] == pytest.approx(expected_deviation)
    assert violation["allowed_distance_A"] == pytest.approx([1.35, 1.75])
    assert violation["allowed_deviation_A"] == pytest.approx([-0.15, 0.25])


def test_linear_mode_preserves_nonbonded_extra_bond_check():
    atoms = _atoms()
    atoms.positions[2] = [0.5, 0.0, 0.0]

    result = validate_fixed_topology(atoms, _linear_config(), _topology())

    assert not result.valid
    assert "topology_extra_bond" in {item["kind"] for item in result.violations}


def test_nonfinite_coordinates_are_rejected():
    atoms = _atoms()
    atoms.positions[0, 0] = np.nan

    result = validate_fixed_topology(atoms, _linear_config(), _topology())

    assert not result.valid
    assert result.reason == "topology_nonfinite"


def _write_mol(path: Path, *, sdf: bool = False) -> None:
    suffix = "$$$$\n" if sdf else ""
    path.write_text(
        "three carbon chain\n"
        "  ALF\n"
        "\n"
        "  3  2  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    3.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "  1  2  1  0  0  0  0\n"
        "  2  3  1  0  0  0  0\n"
        "M  END\n"
        + suffix,
        encoding="utf-8",
    )


@pytest.mark.parametrize(("extension", "reference_format"), [("mol", "auto"), ("sdf", "auto")])
def test_linear_config_loads_absolute_mol_and_sdf_references(tmp_path, extension, reference_format):
    pytest.importorskip("rdkit")
    path = tmp_path / f"reference.{extension}"
    _write_mol(path, sdf=extension == "sdf")
    config = _linear_config(
        reference_conformer_path=str(path),
        reference_format=reference_format,
    )

    topology = load_fixed_topology(config)

    assert topology is not None
    assert topology.n_atoms == 3
    assert len(topology.bonds) == 2


def test_linear_config_loads_relative_reference(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    path = tmp_path / "reference.mol"
    _write_mol(path)
    monkeypatch.chdir(tmp_path)

    topology = load_fixed_topology(
        _linear_config(reference_conformer_path="reference.mol", reference_format="auto")
    )

    assert topology is not None
    assert topology.reference_path == str(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"bond_window_mode": "unknown"},
        {"bond_compression_tolerance_A": -0.01},
        {"bond_extension_tolerance_A": -0.01},
        {"bond_compression_tolerance_A": float("nan")},
        {"bond_extension_tolerance_A": float("inf")},
        {"bond_compression_tolerance_A": 1.5},
    ],
)
def test_invalid_linear_configuration_fails_preflight(tmp_path, overrides):
    pytest.importorskip("rdkit")
    path = tmp_path / "reference.mol"
    _write_mol(path)
    config = _linear_config(reference_conformer_path=str(path), **overrides)

    with pytest.raises(ValueError):
        load_fixed_topology(config)


@pytest.mark.parametrize("missing_key", ["bond_compression_tolerance_A", "bond_extension_tolerance_A"])
def test_linear_configuration_requires_both_tolerances(tmp_path, missing_key):
    pytest.importorskip("rdkit")
    path = tmp_path / "reference.mol"
    _write_mol(path)
    config = _linear_config(reference_conformer_path=str(path))
    del config["topology_check"][missing_key]

    with pytest.raises(ValueError, match="require configuration keys"):
        load_fixed_topology(config)


def test_invalid_reference_file_fails(tmp_path):
    pytest.importorskip("rdkit")
    path = tmp_path / "invalid.mol"
    path.write_text("not a molecule\n", encoding="utf-8")

    with pytest.raises(ValueError, match="could not parse"):
        load_fixed_topology(_linear_config(reference_conformer_path=str(path)))
