from pathlib import Path

import h5py
import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from alframework.builders.h5_replay_builder import (
    build_h5_replay_manifest,
    build_h5_replay_structures,
)


REFERENCE_NUMBERS = np.asarray([8, 6, 1, 1], dtype=np.int64)
STORAGE_ORDER = np.argsort(REFERENCE_NUMBERS, kind="stable")
STORED_NUMBERS = REFERENCE_NUMBERS[STORAGE_ORDER]
CANONICAL_POSITIONS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [1.7, 0.9, 0.0],
        [1.7, -0.9, 0.0],
    ],
    dtype=np.float64,
)


def _topology_config(tmp_path):
    reference = tmp_path / "reference.xyz"
    write(
        reference,
        Atoms(numbers=REFERENCE_NUMBERS, positions=CANONICAL_POSITIONS),
    )
    return {
        "topology_check": {
            "enabled": True,
            "reference_conformer_path": str(reference),
            "reference_format": "xyz",
            "reference_charge": 0,
            "bond_min_scale": 0.70,
            "bond_max_scale": 1.50,
            "connectivity_scale": 1.25,
        }
    }


def _write_shard(path, *, frame_offset=0.0, topology_ids=True, species=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.stack(
        [
            CANONICAL_POSITIONS + [frame_offset, 0.0, 0.0],
            CANONICAL_POSITIONS + [frame_offset + 0.1, 0.0, 0.0],
        ]
    )
    stored_positions = positions[:, STORAGE_ORDER]
    stored_species = (
        STORED_NUMBERS if species is None else np.asarray(species)
    )
    with h5py.File(path, "w") as handle:
        group = handle.create_group("C01_H02_O01")
        group.create_dataset("coordinates", data=stored_positions)
        group.create_dataset("species", data=stored_species)
        if topology_ids:
            group.create_dataset(
                "topology_atom_ids",
                data=np.tile(STORAGE_ORDER, (2, 1)),
            )


def _build(tmp_path, moleculeids, *, current_h5_id=1):
    return build_h5_replay_structures(
        moleculeids=moleculeids,
        builder_config={
            "source_priority": "h5_only",
            "selection_seed": 17,
        },
        sampler_config=_topology_config(tmp_path),
        h5_path=str(tmp_path / "h5store" / "data-{:04d}.h5"),
        current_h5_id=current_h5_id,
        master_directory=str(tmp_path),
    )


@pytest.mark.parametrize("topology_ids", [True, False])
def test_h5_replay_restores_reference_order(tmp_path, topology_ids):
    _write_shard(
        tmp_path / "h5store" / "data-0000.h5",
        topology_ids=topology_ids,
    )

    first = _build(tmp_path, ["mol-0000-0000000042"])[0]
    second = _build(tmp_path, ["mol-0000-0000000042"])[0]

    np.testing.assert_array_equal(
        first.get_atoms().get_atomic_numbers(), REFERENCE_NUMBERS
    )
    np.testing.assert_allclose(
        first.get_atoms().get_positions(),
        second.get_atoms().get_positions(),
    )
    assert first.get_metadata() == second.get_metadata()
    assert first.get_metadata()["replay_canonicalized_to_topology"] is True


def test_h5_replay_manifest_and_batch_span_multiple_shards(tmp_path):
    _write_shard(tmp_path / "h5store" / "data-0000.h5")
    _write_shard(
        tmp_path / "h5store" / "data-0001.h5",
        frame_offset=10.0,
        topology_ids=False,
    )

    manifest = build_h5_replay_manifest(
        str(tmp_path / "h5store" / "data-{:04d}.h5"),
        current_h5_id=2,
    )
    molecules = _build(
        tmp_path,
        [f"mol-0000-{index:010d}" for index in range(50)],
        current_h5_id=2,
    )

    assert [(item.start, item.stop) for item in manifest] == [(0, 2), (2, 4)]
    assert len(molecules) == 50
    assert {
        Path(molecule.get_metadata()["replay_source_path"]).name
        for molecule in molecules
    } == {"data-0000.h5", "data-0001.h5"}
    assert all(
        molecule.get_atoms().get_atomic_numbers().tolist()
        == REFERENCE_NUMBERS.tolist()
        for molecule in molecules
    )


def test_h5_replay_rejects_species_outside_stable_storage_order(tmp_path):
    _write_shard(
        tmp_path / "h5store" / "data-0000.h5",
        topology_ids=False,
        species=[1, 6, 1, 8],
    )

    with pytest.raises(ValueError, match="stable atomic-number storage order"):
        _build(tmp_path, ["mol-0000-0000000000"])


def test_h5_replay_rejects_unsupported_source_priority(tmp_path):
    _write_shard(tmp_path / "h5store" / "data-0000.h5")

    with pytest.raises(ValueError, match="source_priority='h5_only'"):
        build_h5_replay_structures(
            moleculeids=["example"],
            builder_config={"source_priority": "seed_only"},
            sampler_config=_topology_config(tmp_path),
            h5_path=str(tmp_path / "h5store" / "data-{:04d}.h5"),
            current_h5_id=1,
            master_directory=str(tmp_path),
        )
