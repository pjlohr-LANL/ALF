from pathlib import Path

import h5py
import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from alframework.builders.h5_replay_builder import (
    build_h5_replay_manifest,
    build_h5_replay_structures,
    h5_replay_builder_local_task,
    select_h5_replay_indices,
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


def test_sequential_selection_covers_4990_frames_once_in_50_batches():
    selected = []
    batch_lengths = []
    for start in range(0, 5000, 50):
        batch = select_h5_replay_indices(
            [
                f"mol-boot-{index:010d}"
                for index in range(start, start + 50)
            ],
            current_h5_id=1,
            frame_count=4990,
            selection_seed=42,
            selection_mode="sequential",
        )
        batch_lengths.append(len(batch))
        selected.extend(index for _, index in batch)

    assert batch_lengths[:-1] == [50] * 99
    assert batch_lengths[-1] == 40
    assert selected == list(range(4990))


def test_external_bootstrap_uses_only_source_geometry_and_then_switches(
    tmp_path,
):
    source = tmp_path / "source" / "data-0000.h5"
    output = tmp_path / "output" / "data-0000.h5"
    _write_shard(source)
    _write_shard(output, frame_offset=10.0)
    with h5py.File(source, "a") as handle:
        group = handle["C01_H02_O01"]
        group.create_dataset("sE0", data=[-100.0, -99.0])
        group.create_dataset("F0", data=np.ones((2, 4, 3)))
        group.create_dataset("_id", data=[b"source-0", b"source-1"])

    config = {
        "source_priority": "h5_only",
        "selection_seed": 17,
        "bootstrap_h5_path": "source/data-{:04d}.h5",
        "bootstrap_h5_count": 1,
        "bootstrap_selection_mode": "sequential",
    }
    common = {
        "builder_config": config,
        "sampler_config": _topology_config(tmp_path),
        "h5_path": str(tmp_path / "output" / "data-{:04d}.h5"),
        "master_directory": str(tmp_path),
    }
    bootstrap = build_h5_replay_structures(
        moleculeids=[
            "mol-boot-0000000000",
            "mol-boot-0000000001",
            "mol-boot-0000000002",
        ],
        current_h5_id=0,
        **common,
    )

    assert len(bootstrap) == 2
    assert [
        molecule.get_metadata()["replay_global_frame_index"]
        for molecule in bootstrap
    ] == [0, 1]
    assert [
        molecule.get_metadata()["replay_source_id"]
        for molecule in bootstrap
    ] == ["source-0", "source-1"]
    assert all(
        molecule.get_metadata()["replay_external_bootstrap"] is True
        for molecule in bootstrap
    )
    assert all(molecule.get_results() == {} for molecule in bootstrap)
    assert all(
        Path(molecule.get_metadata()["replay_source_path"]) == source
        for molecule in bootstrap
    )

    ordinary = build_h5_replay_structures(
        moleculeids=["mol-0000-0000000000"],
        current_h5_id=1,
        **common,
    )[0]
    assert Path(ordinary.get_metadata()["replay_source_path"]) == output
    assert ordinary.get_metadata()["replay_external_bootstrap"] is False
    assert ordinary.get_metadata()["replay_selection_mode"] == (
        "deterministic_hash"
    )


def test_local_h5_replay_entry_point_targets_builder_executor():
    assert h5_replay_builder_local_task.executors == [
        "alf_builder_executor"
    ]
