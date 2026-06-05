from __future__ import annotations

import numpy as np
from ase import Atoms

from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import dataset_screening_metrics, filter_dataset_screening


def _properties_list() -> dict[str, list[object]]:
    return {
        "sE0": ["sE0", "system", 1.0],
        "F0": ["F0", "atomic", 1.0],
        "sE1": ["sE1", "system", 1.0],
        "F1": ["F1", "atomic", 1.0],
    }


def _sampler_config(**overrides):
    config = {
        "max_force_cutoff": 10.0,
        "min_distance_cutoff": 0.7,
        "dataset_screening": {
            "enabled": True,
            "force": True,
            "min_distance": True,
        },
    }
    config.update(overrides)
    return config


def _molecule(
    molecule_id: str,
    *,
    positions=None,
    force0=None,
    force1=None,
) -> MoleculesObject:
    atoms = Atoms(
        symbols=["H", "H"],
        positions=positions if positions is not None else [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
    )
    molecule = MoleculesObject(atoms, molecule_id)
    molecule.store_results(
        {
            "sE0": 0.0,
            "sE1": 1.0,
            "F0": np.asarray(force0 if force0 is not None else np.zeros((2, 3)), dtype=float),
            "F1": np.asarray(force1 if force1 is not None else np.zeros((2, 3)), dtype=float),
        }
    )
    molecule.set_converged_flag(True)
    return molecule


def test_dataset_screening_keeps_molecule_below_thresholds():
    molecule = _molecule("keep", force0=[[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])

    kept, summary = filter_dataset_screening([molecule], _properties_list(), _sampler_config())

    assert kept == [molecule]
    assert summary["kept"] == 1
    assert summary["rejected_force"] == 0
    assert summary["rejected_min_distance"] == 0


def test_dataset_screening_rejects_by_per_atom_force_norm():
    molecule = _molecule("force", force0=[[8.0, 8.0, 0.0], [0.0, 0.0, 0.0]])

    metrics = dataset_screening_metrics(molecule, _properties_list())
    kept, summary = filter_dataset_screening([molecule], _properties_list(), _sampler_config())

    assert max(abs(component) for component in molecule.get_results()["F0"][0]) < 10.0
    assert metrics["max_force_norm"] > 10.0
    assert kept == []
    assert summary["rejected_force"] == 1
    assert summary["rejected_min_distance"] == 0


def test_dataset_screening_rejects_by_any_labeled_force_state():
    molecule = _molecule("force-state-1", force1=[[0.0, 0.0, 0.0], [11.0, 0.0, 0.0]])

    kept, summary = filter_dataset_screening([molecule], _properties_list(), _sampler_config())

    assert kept == []
    assert summary["rejected_force"] == 1


def test_dataset_screening_rejects_by_min_distance():
    molecule = _molecule("close", positions=[[0.0, 0.0, 0.0], [0.69, 0.0, 0.0]])

    kept, summary = filter_dataset_screening([molecule], _properties_list(), _sampler_config())

    assert kept == []
    assert summary["rejected_force"] == 0
    assert summary["rejected_min_distance"] == 1


def test_dataset_screening_all_rejected_batch_returns_empty_kept_list():
    high_force = _molecule("force", force0=[[11.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    close_contact = _molecule("close", positions=[[0.0, 0.0, 0.0], [0.69, 0.0, 0.0]])

    kept, summary = filter_dataset_screening([high_force, close_contact], _properties_list(), _sampler_config())

    assert kept == []
    assert summary["total"] == 2
    assert summary["kept"] == 0
    assert summary["rejected_force"] == 1
    assert summary["rejected_min_distance"] == 1


def test_dataset_screening_disabled_keeps_everything():
    molecule = _molecule("force", force0=[[99.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    kept, summary = filter_dataset_screening(
        [molecule],
        _properties_list(),
        _sampler_config(dataset_screening={"enabled": False}),
    )

    assert kept == [molecule]
    assert summary["enabled"] is False
    assert summary["kept"] == 1
