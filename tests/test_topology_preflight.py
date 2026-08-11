from pathlib import Path

import h5py
import numpy as np
import pytest

from alframework.tools.topology_preflight import (
    _validate_bootstrap_frame_count,
    _validate_h5_shards,
)


def _bootstrap_ids(count: int) -> list[str]:
    return [f"mol-boot-{index:010d}" for index in range(count)]


def test_runtime_labeled_bootstrap_accepts_valid_subset():
    ids = _bootstrap_ids(5)
    ids.pop(2)

    summary = _validate_bootstrap_frame_count(
        ids,
        expected_frames=5,
        runtime_labeled=True,
    )

    assert summary == {
        "bootstrap_labeling_mode": "runtime_labeled",
        "requested_bootstrap_frames": 5,
        "accepted_bootstrap_frames": 4,
        "rejected_bootstrap_frames": 1,
    }


def test_prelabeled_bootstrap_requires_exact_count():
    summary = _validate_bootstrap_frame_count(
        _bootstrap_ids(5),
        expected_frames=5,
        runtime_labeled=False,
    )
    assert summary["accepted_bootstrap_frames"] == 5
    assert summary["rejected_bootstrap_frames"] == 0

    with pytest.raises(ValueError, match="Expected 5 pre-labeled"):
        _validate_bootstrap_frame_count(
            _bootstrap_ids(4),
            expected_frames=5,
            runtime_labeled=False,
        )


@pytest.mark.parametrize(
    "ids, message",
    [
        (["mol-boot-0000000000", "mol-boot-0000000000"], "duplicate"),
        (["mol-boot-0000000000", "mol-boot-0000000005"], "out-of-range"),
        (["not-a-bootstrap-id"], "invalid"),
    ],
)
def test_runtime_labeled_bootstrap_rejects_bad_ids(ids, message):
    with pytest.raises(ValueError, match=message):
        _validate_bootstrap_frame_count(
            ids,
            expected_frames=5,
            runtime_labeled=True,
        )


def test_h5_schema_requires_topology_atom_ids(tmp_path: Path):
    shard_path = tmp_path / "data-0000.h5"
    with h5py.File(shard_path, "w") as handle:
        group = handle.create_group("C01_H01")
        group.create_dataset("coordinates", data=np.zeros((1, 2, 3)))
        group.create_dataset("species", data=np.asarray([b"C", b"H"]))
        group.create_dataset("_id", data=np.asarray([b"mol-boot-0000000000"]))
        group.create_dataset("sE0", data=np.zeros(1))

    with pytest.raises(ValueError, match="topology_atom_ids"):
        _validate_h5_shards(
            [shard_path],
            {"coordinates", "species", "_id", "sE0", "topology_atom_ids"},
        )
