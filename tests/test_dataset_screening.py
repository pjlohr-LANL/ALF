import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from alframework.tools.dataset_screening import (
    dataset_screening_metrics,
    dataset_screening_options,
    filter_dataset_screening,
    screen_and_store_dataset,
)
from alframework.tools.molecules_class import MoleculesObject


PROPERTIES = {
    "sE0": ["sE0", "system", 1.0],
    "F0": ["F0", "atomic", 1.0],
    "sE1": ["sE1", "system", 1.0],
    "F1": ["F1", "atomic", 1.0],
}


def _sampler_config(tmp_path):
    reference = tmp_path / "water.xyz"
    write(
        reference,
        Atoms(
            "OH2",
            positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ],
        ),
    )
    return {
        "distcut": 0.70,
        "min_distance_cutoff": 0.70,
        "max_force_cutoff": 16.0,
        "dataset_screening": {
            "enabled": True,
            "force": True,
            "min_distance": True,
            "topology": True,
        },
        "topology_check": {
            "enabled": True,
            "reference_conformer_path": str(reference),
            "reference_format": "xyz",
            "reference_charge": 0,
            "bond_min_scale": 0.70,
            "bond_max_scale": 1.35,
            "connectivity_scale": 1.25,
        },
    }


def _molecule(
    molecule_id,
    *,
    positions=None,
    force_value=1.0,
):
    if positions is None:
        positions = [
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ]
    molecule = MoleculesObject(Atoms("OH2", positions=positions), molecule_id)
    molecule.store_results(
        {
            "sE0": 0.0,
            "F0": np.full((3, 3), force_value),
            "sE1": 1.0,
            "F1": np.full((3, 3), force_value),
        }
    )
    molecule.set_converged_flag(True)
    return molecule


def test_dataset_screening_keeps_valid_six_force_contract(tmp_path):
    molecule = _molecule("valid")
    kept, summary = filter_dataset_screening(
        [molecule],
        PROPERTIES,
        _sampler_config(tmp_path),
        master_directory=str(tmp_path),
    )

    assert kept == [molecule]
    assert summary["kept"] == 1
    assert molecule.get_metadata()["dataset_screening"]["rejected"] is False


def test_dataset_screening_rejects_force_distance_and_topology(tmp_path):
    force = _molecule("force", force_value=20.0)
    distance = _molecule(
        "distance",
        positions=[
            [0.0, 0.0, 0.0],
            [0.50, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
    )
    topology = _molecule(
        "topology",
        positions=[
            [0.0, 0.0, 0.0],
            [1.50, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
    )

    kept, summary = filter_dataset_screening(
        [force, distance, topology],
        PROPERTIES,
        _sampler_config(tmp_path),
        master_directory=str(tmp_path),
    )

    assert kept == []
    assert summary["rejected_force"] == 1
    assert summary["rejected_min_distance"] == 1
    assert summary["rejected_topology"] == 2


def test_dataset_screening_uses_all_configured_force_states(tmp_path):
    molecule = _molecule("state-one")
    molecule.get_results()["F0"][:] = 0.0
    molecule.get_results()["F1"][:] = 17.0

    metrics = dataset_screening_metrics(
        molecule,
        PROPERTIES,
        _sampler_config(tmp_path),
        master_directory=str(tmp_path),
    )

    assert metrics["max_force_norm"] == pytest.approx(
        np.sqrt(3.0) * 17.0
    )


def test_dataset_screening_rejects_missing_force_state(tmp_path):
    molecule = _molecule("missing-state")
    del molecule.get_results()["F1"]

    with pytest.raises(ValueError, match="missing F1"):
        dataset_screening_metrics(
            molecule,
            PROPERTIES,
            _sampler_config(tmp_path),
            master_directory=str(tmp_path),
        )


def test_all_rejected_batch_creates_no_new_shard(tmp_path):
    destination = tmp_path / "data-0001.h5"
    rejected = _molecule("rejected", force_value=20.0)

    stored, kept, summary = screen_and_store_dataset(
        str(destination),
        [rejected],
        PROPERTIES,
        _sampler_config(tmp_path),
        master_directory=str(tmp_path),
    )

    assert stored is False
    assert kept == []
    assert summary["kept"] == 0
    assert not destination.exists()


def test_dataset_screening_validates_options():
    with pytest.raises(ValueError, match="Unknown dataset_screening"):
        dataset_screening_options(
            {"dataset_screening": {"enabled": True, "mystery": True}}
        )
    with pytest.raises(ValueError, match="max_force_cutoff"):
        dataset_screening_options(
            {
                "dataset_screening": True,
                "max_force_cutoff": 0.0,
            }
        )
