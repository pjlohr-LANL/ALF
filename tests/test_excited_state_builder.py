from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from alframework.builders.excited_state_builder import (
    build_excited_state_replay_structures,
    build_replay_manifest,
)


def _write_seed_dataset(seed_dir: Path) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    coords = np.array(
        [
            [
                [0.0000, 0.0000, 0.0000],
                [0.9572, 0.0000, 0.0000],
                [-0.2390, 0.9266, 0.0000],
                [0.0000, 0.0000, 0.0000],
            ],
            [
                [0.0000, 0.0000, 0.0000],
                [0.6291, 0.6291, 0.6291],
                [-0.6291, -0.6291, 0.6291],
                [-0.6291, 0.6291, -0.6291],
            ],
            [
                [0.1000, 0.0000, 0.0000],
                [1.0572, 0.0000, 0.0000],
                [-0.1390, 0.9266, 0.0000],
                [0.0000, 0.0000, 0.0000],
            ],
        ],
        dtype=np.float64,
    )
    numbers = np.array(
        [
            [8, 1, 1, 0],
            [6, 1, 1, 1],
            [8, 1, 1, 0],
        ],
        dtype=np.int64,
    )
    np.save(seed_dir / "R.npy", coords)
    np.save(seed_dir / "Z.npy", numbers)


def _properties_list() -> dict[str, list[object]]:
    return {
        "sE0": ["sE0", "system", 1.0],
        "F0": ["F0", "atomic", 1.0],
        "sE1": ["sE1", "system", 1.0],
        "F1": ["F1", "atomic", 1.0],
    }


def test_seed_replay_builder_returns_metadata_and_selected_state(tmp_path: Path):
    seed_dir = tmp_path / "seed"
    _write_seed_dataset(seed_dir)
    builder_config = {
        "seed_dataset_dir": str(seed_dir),
        "source_priority": "seed_only",
        "empirical_formula_filter": "H02_O01",
        "state_selection": {"mode": "fixed", "fixed_state": 1},
    }

    molecules = build_excited_state_replay_structures(
        moleculeids=["traj_0001"],
        builder_config=builder_config,
        properties_list=_properties_list(),
        h5_path=str(tmp_path / "h5store" / "data-{:04d}.h5"),
        current_h5_id=0,
    )

    mol = molecules[0]
    assert mol.get_metadata()["replay_source_kind"] == "seed"
    assert mol.get_metadata()["excited_state"] == 1
    assert mol.get_metadata()["replay_formula"] == "H02_O01"
    assert len(mol.get_atoms()) == 3


def test_h5_replay_builder_prefers_h5_and_caches_manifest(tmp_path: Path):
    seed_dir = tmp_path / "seed"
    _write_seed_dataset(seed_dir)
    h5_dir = tmp_path / "h5store"
    h5_dir.mkdir()
    h5_path = h5_dir / "data-0000.h5"
    with h5py.File(h5_path, "w") as handle:
        grp = handle.create_group("H02_O01")
        grp.create_dataset(
            "coordinates",
            data=np.array(
                [[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]],
                dtype=np.float64,
            ),
        )
        grp.create_dataset("species", data=np.array([b"O", b"H", b"H"]))

    cache_dir = tmp_path / "manifest_cache"
    builder_config = {
        "seed_dataset_dir": str(seed_dir),
        "source_priority": "h5_then_seed",
        "manifest_cache_dir": str(cache_dir),
        "empirical_formula_filter": "H02_O01",
        "state_selection": {"mode": "fixed", "fixed_state": 0},
    }

    manifest = build_replay_manifest(
        h5_path=str(h5_dir / "data-{:04d}.h5"),
        current_h5_id=1,
        builder_config=builder_config,
    )
    assert manifest["h5_groups"]
    assert any(cache_dir.glob("replay_manifest_*.pkl"))

    molecules = build_excited_state_replay_structures(
        moleculeids=["traj_0000"],
        builder_config=builder_config,
        properties_list=_properties_list(),
        h5_path=str(h5_dir / "data-{:04d}.h5"),
        current_h5_id=1,
    )

    mol = molecules[0]
    assert mol.get_metadata()["replay_source_kind"] == "h5"
    assert mol.get_metadata()["replay_group_path"] == "H02_O01"
    assert mol.get_metadata()["excited_state"] == 0
    assert len(mol.get_atoms()) == 3


def test_seed_all_once_bootstrap_walks_eligible_seed_frames(tmp_path: Path):
    seed_dir = tmp_path / "seed"
    _write_seed_dataset(seed_dir)
    builder_config = {
        "seed_dataset_dir": str(seed_dir),
        "source_priority": "seed_all_once",
        "empirical_formula_filter": "H02_O01",
        "state_selection": {"mode": "fixed", "fixed_state": 0},
    }

    molecules = build_excited_state_replay_structures(
        moleculeids=["mol-boot-0000000000", "mol-boot-0000000001"],
        builder_config=builder_config,
        properties_list=_properties_list(),
        h5_path=str(tmp_path / "h5store" / "data-{:04d}.h5"),
        current_h5_id=0,
    )

    frame_indices = [int(mol.get_metadata()["replay_frame_index"]) for mol in molecules]
    assert frame_indices == [0, 2]

    coords = [mol.get_atoms().get_positions() for mol in molecules]
    assert np.allclose(coords[0][0], np.array([0.0, 0.0, 0.0]))
    assert np.allclose(coords[1][0], np.array([0.1, 0.0, 0.0]))
