import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from alframework.ml_interfaces import excited_state_hippynn_interface as ml


def _properties(with_gap=False):
    properties = {
        "sE0": ["sE0", "system", 1.0],
        "F0": ["F0", "atomic", 1.0],
        "sE1": ["sE1", "system", 1.0],
        "F1": ["F1", "atomic", 1.0],
    }
    if with_gap:
        properties["dE01"] = ["dE01", "system", 1.0]
    return properties


def _config(**overrides):
    config = {
        "n_models": 2,
        "cell_key": None,
        "train_forces": True,
        "network_choice": 0,
        "valid_size": 0.2,
        "test_size": 0.2,
        "device_string": "cpu",
        "exports": {"pdf": False, "png": False, "csv": False},
    }
    config.update(overrides)
    return config


def _write_group(
    path: Path,
    name: str,
    *,
    species=("H", "O"),
    count=3,
    missing=None,
    nonfinite=None,
):
    mode = "a" if path.exists() else "w"
    with h5py.File(path, mode) as store:
        group = store.create_group(name)
        values = {
            "species": np.asarray(species, dtype="S2"),
            "coordinates": np.arange(
                count * len(species) * 3,
                dtype=float,
            ).reshape(count, len(species), 3),
            "sE0": np.linspace(-2.0, -1.0, count),
            "F0": np.zeros((count, len(species), 3)),
            "sE1": np.linspace(-1.5, -0.5, count),
            "F1": np.ones((count, len(species), 3)),
            "dE01": np.full(count, 0.5),
        }
        if nonfinite is not None:
            values[nonfinite] = np.asarray(values[nonfinite]).copy()
            values[nonfinite].reshape(-1)[0] = np.nan
        for key, value in values.items():
            if key != missing:
                group.create_dataset(key, data=value)


def _filter_arrays(count=10, with_gap=True):
    coordinates = np.zeros((count, 2, 3), dtype=np.float32)
    coordinates[:, 1, 0] = 1.0
    arrays = {
        "coordinates": coordinates,
        "species": np.tile(np.asarray([[1, 1]], dtype=np.int64), (count, 1)),
        "sE0": np.zeros(count, dtype=np.float32),
        "F0": np.zeros((count, 2, 3), dtype=np.float32),
        "sE1": np.zeros(count, dtype=np.float32),
        "F1": np.zeros((count, 2, 3), dtype=np.float32),
        "indices": np.arange(count, dtype=np.int64),
    }
    if with_gap:
        arrays["dE01"] = arrays["sE1"] - arrays["sE0"]
    return arrays


def test_explicit_h5_loader_reads_two_states_and_small_molecules(tmp_path):
    shard = tmp_path / "data-0000.h5"
    _write_group(shard, "HO", count=4)

    arrays, summary = ml.load_excited_state_h5_arrays(
        str(tmp_path),
        _properties(),
        configured_n_atoms=2,
        configured_possible_species=[0, 1, 8],
    )

    assert set(arrays) == {"coordinates", "species", "sE0", "F0", "sE1", "F1"}
    assert arrays["coordinates"].shape == (4, 2, 3)
    assert arrays["species"].shape == (4, 2)
    assert arrays["species"].dtype == np.int64
    assert arrays["sE0"].dtype == np.float32
    assert arrays["F1"].shape == (4, 2, 3)
    assert summary["n_structures"] == 4
    assert summary["n_atoms"] == 2
    assert summary["atomic_numbers"] == [1, 8]
    assert [row["energy_db_name"] for row in summary["state_table"]] == [
        "sE0",
        "sE1",
    ]


def test_explicit_h5_loader_uses_configured_database_names(tmp_path):
    properties = {
        "sE0": ["state_zero_energy", "system", 1.0],
        "F0": ["state_zero_force", "atomic", 1.0],
    }
    shard = tmp_path / "data.h5"
    with h5py.File(shard, "w") as store:
        group = store.create_group("H2")
        group["species"] = np.asarray(["H", "H"], dtype="S2")
        group["coordinates"] = np.zeros((3, 2, 3))
        group["state_zero_energy"] = np.arange(3, dtype=float)
        group["state_zero_force"] = np.zeros((3, 2, 3))

    arrays, summary = ml.load_excited_state_h5_arrays(
        str(shard),
        properties,
    )

    assert set(arrays) == {
        "coordinates",
        "species",
        "state_zero_energy",
        "state_zero_force",
    }
    assert summary["state_table"][0]["energy_key"] == "sE0"
    assert summary["state_table"][0]["energy_db_name"] == "state_zero_energy"


def test_explicit_h5_loader_reads_and_validates_gap_targets(tmp_path):
    _write_group(tmp_path / "data.h5", "HO")

    arrays, summary = ml.load_excited_state_h5_arrays(
        str(tmp_path),
        _properties(with_gap=True),
    )

    np.testing.assert_allclose(arrays["dE01"], arrays["sE1"] - arrays["sE0"])
    assert summary["gap_table"][0]["gap_key"] == "dE01"


def test_explicit_h5_loader_rejects_inconsistent_gap_targets(tmp_path):
    path = tmp_path / "data.h5"
    _write_group(path, "HO")
    with h5py.File(path, "a") as store:
        store["HO"]["dE01"][:] = 0.25

    with pytest.raises(ValueError, match="inconsistent"):
        ml.load_excited_state_h5_arrays(
            str(tmp_path),
            _properties(with_gap=True),
        )


@pytest.mark.parametrize(
    ("missing", "nonfinite", "message"),
    [
        ("dE01", None, "missing required datasets"),
        (None, "dE01", "nonfinite"),
    ],
)
def test_explicit_h5_loader_rejects_missing_or_nonfinite_gap(
    tmp_path,
    missing,
    nonfinite,
    message,
):
    _write_group(
        tmp_path / "data.h5",
        "HO",
        missing=missing,
        nonfinite=nonfinite,
    )

    with pytest.raises((KeyError, ValueError), match=message):
        ml.load_excited_state_h5_arrays(
            str(tmp_path),
            _properties(with_gap=True),
        )


def test_explicit_h5_loader_rejects_missing_target(tmp_path):
    _write_group(tmp_path / "data.h5", "HO", missing="F1")

    with pytest.raises(KeyError, match="F1"):
        ml.load_excited_state_h5_arrays(str(tmp_path), _properties())


def test_explicit_h5_loader_rejects_mixed_atomic_ordering(tmp_path):
    shard = tmp_path / "data.h5"
    _write_group(shard, "first", species=("H", "O"))
    _write_group(shard, "second", species=("O", "H"))

    with pytest.raises(ValueError, match="exact atomic-number sequence"):
        ml.load_excited_state_h5_arrays(str(tmp_path), _properties())


@pytest.mark.parametrize("field", ["coordinates", "sE0", "F1"])
def test_explicit_h5_loader_rejects_nonfinite_data(tmp_path, field):
    _write_group(tmp_path / "data.h5", "HO", nonfinite=field)

    with pytest.raises(ValueError, match="nonfinite"):
        ml.load_excited_state_h5_arrays(str(tmp_path), _properties())


def test_explicit_h5_loader_validates_configured_shape_and_species(tmp_path):
    _write_group(tmp_path / "data.h5", "HO")

    with pytest.raises(ValueError, match="n_atoms=3"):
        ml.load_excited_state_h5_arrays(
            str(tmp_path),
            _properties(),
            configured_n_atoms=3,
        )
    with pytest.raises(ValueError, match="missing atomic numbers"):
        ml.load_excited_state_h5_arrays(
            str(tmp_path),
            _properties(),
            configured_possible_species=[0, 1],
        )
    with pytest.raises(ValueError, match="begin with padding"):
        ml.load_excited_state_h5_arrays(
            str(tmp_path),
            _properties(),
            configured_possible_species=[1, 8],
        )


def test_multistate_filter_unions_state_outliers_and_aligns_all_arrays():
    arrays = _filter_arrays()
    arrays["sE1"][8] = 10.0
    arrays["F0"][9, 0, 2] = 10.0
    arrays["dE01"] = arrays["sE1"] - arrays["sE0"]
    state_table = ml.validate_excited_state_training_properties(
        _properties(with_gap=True)
    )

    filtered, summary = ml.filter_excited_state_training_arrays(
        arrays,
        state_table,
        coordinates_key="coordinates",
        dist_soft_min=0.5,
        remove_high_energy_cut=5.0,
        remove_high_forces_cut=5.0,
    )

    assert filtered["indices"].tolist() == list(range(8))
    np.testing.assert_allclose(
        filtered["dE01"],
        filtered["sE1"] - filtered["sE0"],
    )
    assert summary["removed_static_cut_union"] == 2
    assert summary["removed_standard_deviation_union"] == 0
    assert summary["per_state_failures"]["0"]["forces"]["static_cut"] == 1
    assert summary["per_state_failures"]["1"]["energy"]["static_cut"] == 1
    assert summary["final_structures"] == 8


def test_multistate_filter_is_state_order_independent():
    arrays = _filter_arrays()
    arrays["sE0"][7] = 10.0
    arrays["F1"][8, 1, 0] = 10.0
    state_table = ml.validate_excited_state_training_properties(_properties())

    forward, _ = ml.filter_excited_state_training_arrays(
        arrays,
        state_table,
        coordinates_key="coordinates",
        dist_soft_min=0.5,
        remove_high_energy_cut=5.0,
        remove_high_forces_cut=5.0,
    )
    reverse, _ = ml.filter_excited_state_training_arrays(
        arrays,
        list(reversed(state_table)),
        coordinates_key="coordinates",
        dist_soft_min=0.5,
        remove_high_energy_cut=5.0,
        remove_high_forces_cut=5.0,
    )

    np.testing.assert_array_equal(forward["indices"], reverse["indices"])


def test_multistate_filter_removes_minimum_distance_before_statistics():
    arrays = _filter_arrays()
    arrays["coordinates"][0, 1, 0] = 0.4
    arrays["sE1"][0] = 1000.0
    state_table = ml.validate_excited_state_training_properties(_properties())

    filtered, summary = ml.filter_excited_state_training_arrays(
        arrays,
        state_table,
        coordinates_key="coordinates",
        dist_soft_min=0.5,
        remove_high_energy_std=2.0,
    )

    assert filtered["indices"].tolist() == list(range(1, 10))
    assert summary["removed_min_distance"] == 1
    assert summary["removed_standard_deviation_union"] == 0


def test_multistate_filter_unions_standardized_energy_and_force_outliers():
    arrays = _filter_arrays()
    arrays["sE1"][8] = 10.0
    arrays["F0"][9, 0, 2] = 10.0
    state_table = ml.validate_excited_state_training_properties(_properties())

    filtered, summary = ml.filter_excited_state_training_arrays(
        arrays,
        state_table,
        coordinates_key="coordinates",
        dist_soft_min=0.5,
        remove_high_energy_std=2.0,
        remove_high_forces_std=2.0,
    )

    assert filtered["indices"].tolist() == list(range(8))
    assert summary["removed_standard_deviation_union"] == 2
    assert (
        summary["per_state_failures"]["0"]["forces"][
            "standard_deviation"
        ]
        == 1
    )
    assert (
        summary["per_state_failures"]["1"]["energy"][
            "standard_deviation"
        ]
        == 1
    )


def test_multistate_filter_handles_zero_variance_and_disabled_outlier_filters():
    arrays = _filter_arrays()
    state_table = ml.validate_excited_state_training_properties(_properties())

    filtered, summary = ml.filter_excited_state_training_arrays(
        arrays,
        state_table,
        coordinates_key="coordinates",
        dist_soft_min=0.0,
        remove_high_energy_std=2.0,
        remove_high_forces_std=2.0,
    )

    assert filtered["indices"].tolist() == list(range(10))
    assert summary["final_structures"] == 10


def test_disabled_outlier_filters_retain_large_finite_values():
    arrays = _filter_arrays()
    arrays["sE1"][8] = 1000.0
    arrays["F0"][9, 0, 0] = 1000.0
    state_table = ml.validate_excited_state_training_properties(_properties())

    filtered, summary = ml.filter_excited_state_training_arrays(
        arrays,
        state_table,
        coordinates_key="coordinates",
        dist_soft_min=0.0,
    )

    assert filtered["indices"].tolist() == list(range(10))
    assert summary["removed_static_cut_union"] == 0
    assert summary["removed_standard_deviation_union"] == 0


def test_single_state_filter_matches_production_hippynn():
    torch = pytest.importorskip("torch")
    database_module = pytest.importorskip("hippynn.databases.database")
    Database = database_module.Database
    arrays = _filter_arrays(with_gap=False)
    arrays = {
        key: value
        for key, value in arrays.items()
        if key not in {"sE1", "F1"}
    }
    arrays["F0"][7, 0, 0] = 8.0
    arrays["F0"][8, 0, 1] = 3.0
    arrays["sE0"][9] = 8.0
    properties = {
        "sE0": ["sE0", "system", 1.0],
        "F0": ["F0", "atomic", 1.0],
    }
    state_table = ml.validate_excited_state_training_properties(properties)
    filtered, _ = ml.filter_excited_state_training_arrays(
        arrays,
        state_table,
        coordinates_key="coordinates",
        dist_soft_min=0.5,
        remove_high_energy_cut=4.0,
        remove_high_energy_std=2.0,
        remove_high_forces_cut=4.0,
        remove_high_forces_std=2.0,
    )

    production_arrays = {
        key: torch.as_tensor(value) for key, value in arrays.items()
    }
    production_arrays["sE0"] = production_arrays["sE0"].reshape(-1, 1)
    production = Database(
        production_arrays,
        inputs=["species", "coordinates"],
        targets=["sE0", "F0"],
        seed=7,
        quiet=True,
    )
    production.remove_high_property(
        "F0",
        True,
        species_key="species",
        cut=4.0,
        std_factor=2.0,
    )
    production.remove_high_property(
        "sE0",
        False,
        species_key="species",
        cut=4.0,
        std_factor=2.0,
    )

    np.testing.assert_array_equal(
        filtered["indices"],
        production.arr_dict["indices"].cpu().numpy(),
    )


@pytest.mark.parametrize("count", [0, 2, 4])
def test_filtered_split_capacity_rejects_insufficient_survivors(count):
    with pytest.raises(ValueError, match="too few structures"):
        ml.validate_filtered_split_capacity(
            count,
            test_size=0.2,
            valid_size=0.2,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"cell_key": "cell"}, "nonperiodic"),
        ({"train_forces": False}, "train_forces=true"),
        ({"export_force_gradients": False}, "export_force_gradients=true"),
        ({"network_choice": 2}, "network_choice"),
        ({"optimizer": "SGD"}, "AdamW"),
        ({"gap_targets": {"enabled": True}}, "gap targets"),
        ({"exports": {"csv": True}}, "CSV"),
        ({"remove_high_energy_cut": -1.0}, "remove_high_energy_cut"),
        ({"remove_high_forces_std": np.inf}, "remove_high_forces_std"),
    ],
)
def test_training_config_rejects_unsupported_features(override, message):
    with pytest.raises((ValueError, TypeError), match=message):
        ml.validate_excited_state_training_config(
            _config(**override),
            _properties(),
        )


def test_training_properties_require_energy_force_kinds_and_unique_names():
    wrong_kind = _properties()
    wrong_kind["F0"] = ["F0", "system", 1.0]
    with pytest.raises(ValueError, match="atomic property"):
        ml.validate_excited_state_training_properties(wrong_kind)

    duplicate_name = _properties()
    duplicate_name["F0"] = ["sE0", "atomic", 1.0]
    with pytest.raises(ValueError, match="unique HDF5"):
        ml.validate_excited_state_training_properties(duplicate_name)


def test_loss_matches_fork_formula_exactly():
    actual = ml.compose_excited_state_loss(
        [(2.0, 1.0), (4.0, 3.0)],
        [(6.0, 5.0), (8.0, 7.0)],
        9.0,
        n_atoms=2,
        energy_weight=2.0,
        force_weight=3.0,
        l2_weight=0.5,
    )
    expected = (
        2.0 * ((2.0 + 1.0) + (4.0 + 3.0))
        + 3.0 * ((6.0 + 5.0) + (8.0 + 7.0)) / np.sqrt(6.0)
        + 0.5 * 9.0
    )
    assert actual == pytest.approx(expected)


def test_loss_adds_weighted_derived_gap_terms():
    actual = ml.compose_excited_state_loss(
        [(2.0, 1.0)],
        [(6.0, 5.0)],
        9.0,
        gap_error_terms=[(4.0, 3.0)],
        n_atoms=2,
        energy_weight=2.0,
        force_weight=3.0,
        gap_weight=0.25,
        l2_weight=0.5,
    )
    expected = (
        2.0 * (2.0 + 1.0)
        + 3.0 * (6.0 + 5.0) / np.sqrt(6.0)
        + 0.25 * (4.0 + 3.0)
        + 0.5 * 9.0
    )
    assert actual == pytest.approx(expected)


def test_graph_uses_one_shared_trunk_and_all_state_database_names():
    hippynn = pytest.importorskip("hippynn")
    from hippynn.graphs import inputs

    species = inputs.SpeciesNode(db_name="species")
    positions = inputs.PositionsNode(db_name="coordinates")
    positions.requires_grad = True
    network = ml._build_shared_network(
        species,
        positions,
        network_choice=0,
        network_params={
            "possible_species": [0, 1],
            "n_features": 4,
            "n_sensitivities": 4,
            "dist_soft_min": 0.5,
            "dist_soft_max": 2.0,
            "dist_hard_max": 2.5,
            "n_interaction_layers": 1,
            "n_atom_layers": 1,
            "sensitivity_type": "inverse",
            "resnet": True,
        },
    )
    table = ml.validate_excited_state_training_config(
        _config(),
        _properties(),
    )
    energy_outputs, force_outputs, total_loss, validation = (
        ml.build_excited_state_training_graph(
            network,
            positions,
            table,
            n_atoms=2,
            energy_weight=1.0,
            force_weight=1.0,
            l2_weight=2.0e-5,
        )
    )
    _, db_info = hippynn.experiment.assemble_for_training(
        total_loss,
        validation,
    )

    assert [output.db_name for _, output in energy_outputs] == ["sE0", "sE1"]
    assert [output.db_name for _, output in force_outputs] == ["F0", "F1"]
    assert set(db_info["inputs"]) == {"species", "coordinates"}
    assert {"sE0", "F0", "sE1", "F1"}.issubset(db_info["targets"])
    assert all(
        network in output.parents[0].parents
        for _, output in energy_outputs
    )


def test_graph_exposes_derived_gap_target_without_independent_head():
    hippynn = pytest.importorskip("hippynn")
    from hippynn.graphs import inputs
    from alframework.tools.excited_state_tools import derive_gap_property_table

    species = inputs.SpeciesNode(db_name="species")
    positions = inputs.PositionsNode(db_name="coordinates")
    positions.requires_grad = True
    network = ml._build_shared_network(
        species,
        positions,
        network_choice=0,
        network_params={
            "possible_species": [0, 1],
            "n_features": 4,
            "n_sensitivities": 4,
            "dist_soft_min": 0.5,
            "dist_soft_max": 2.0,
            "dist_hard_max": 2.5,
            "n_interaction_layers": 1,
            "n_atom_layers": 1,
            "sensitivity_type": "inverse",
            "resnet": True,
        },
    )
    properties = _properties(with_gap=True)
    state_table = ml.validate_excited_state_training_config(
        _config(
            gap_targets={
                "enabled": True,
                "pairs": [[0, 1]],
                "weight": 0.5,
            }
        ),
        properties,
    )
    gap_table = derive_gap_property_table(
        properties,
        gap_config={"pairs": [[0, 1]]},
        require_properties=True,
    )
    _, _, total_loss, validation = ml.build_excited_state_training_graph(
        network,
        positions,
        state_table,
        gap_table=gap_table,
        n_atoms=2,
        energy_weight=1.0,
        force_weight=1.0,
        gap_weight=0.5,
        l2_weight=2.0e-5,
    )
    _, db_info = hippynn.experiment.assemble_for_training(
        total_loss,
        validation,
    )

    assert "dE01" in db_info["targets"]
    assert {"dE01_RMSE", "dE01_MAE"}.issubset(validation)


def test_gap_training_config_rejects_independent_heads_and_bad_weight():
    properties = _properties(with_gap=True)
    with pytest.raises(ValueError, match="derived"):
        ml.validate_excited_state_training_config(
            _config(
                gap_targets={
                    "enabled": True,
                    "mode": "head",
                    "pairs": [[0, 1]],
                }
            ),
            properties,
        )
    with pytest.raises(ValueError, match="weight"):
        ml.validate_excited_state_training_config(
            _config(
                gap_targets={
                    "enabled": True,
                    "pairs": [[0, 1]],
                    "weight": -1.0,
                }
            ),
            properties,
        )


def test_model_member_seeds_are_deterministic():
    config = {"random_seed": 42}
    assert [ml.model_member_seed(config, index) for index in range(3)] == [
        42,
        43,
        44,
    ]


def test_gpu_assignment_uses_worker_rank_not_state(monkeypatch):
    class Worker:
        _identity = (6,)

    monkeypatch.setattr(ml.multiprocessing, "current_process", lambda: Worker())
    visible, recorded = ml._configure_cuda_visible_devices(
        "from_multiprocessing",
        4,
    )

    assert visible == "1"
    assert recorded == "1"
    assert ml._worker_count(
        {"device_string": "from_multiprocessing"},
        n_models=7,
        gpus_per_node=4,
    ) == 4
    assert ml._worker_count(
        {"device_string": "cpu"},
        n_models=7,
        gpus_per_node=4,
    ) == 1


def test_train_ensemble_reports_partial_failure_and_writes_error(
    tmp_path,
    monkeypatch,
):
    captured = {}

    class FakePool:
        def __init__(self, processes):
            captured["processes"] = processes

        def map(self, function, params):
            captured["function"] = function
            captured["params"] = params
            outputs = []
            for index, item in enumerate(params):
                model_dir = Path(item["model_dir"])
                model_dir.mkdir(parents=True)
                if index == 0:
                    (model_dir / "training_log.txt").write_text(
                        "Training complete\n",
                        encoding="utf-8",
                    )
                    outputs.append(
                        {
                            "ok": True,
                            "payload": {
                                "model_id": index,
                                "model_dir": str(model_dir),
                            },
                        }
                    )
                else:
                    outputs.append(
                        {
                            "ok": False,
                            "payload": {
                                "model_id": index,
                                "model_dir": str(model_dir),
                                "error_type": "RuntimeError",
                                "error": "synthetic failure",
                            },
                        }
                    )
            return outputs

        def close(self):
            captured["closed"] = True

        def join(self):
            captured["joined"] = True

    class FakeContext:
        Pool = FakePool

    monkeypatch.setattr(
        ml.multiprocessing,
        "get_context",
        lambda method: FakeContext(),
    )

    completed, training_id = ml.train_excited_state_ensemble(
        ML_config=_config(),
        h5_dir=str(tmp_path / "h5store"),
        model_path=str(tmp_path / "models" / "model-{:04d}"),
        current_training_id=3,
        gpus_per_node=4,
        properties_list=_properties(),
    )

    assert completed == [True, False]
    assert training_id == 3
    assert captured["processes"] == 1
    assert captured["function"] is ml._excited_state_training_worker
    assert captured["closed"] and captured["joined"]
    assert [item["model_id"] for item in captured["params"]] == [0, 1]
    error_path = (
        tmp_path
        / "models"
        / "model-0003"
        / "model-01"
        / "training_error.json"
    )
    assert json.loads(error_path.read_text())["error"] == "synthetic failure"


def test_train_ensemble_validates_all_completed_checkpoints(
    tmp_path,
    monkeypatch,
):
    validated = {}

    class FakePool:
        def __init__(self, processes):
            pass

        def map(self, function, params):
            outputs = []
            for item in params:
                model_dir = Path(item["model_dir"])
                model_dir.mkdir(parents=True)
                (model_dir / "training_log.txt").write_text(
                    "Training complete\n",
                    encoding="utf-8",
                )
                outputs.append(
                    {
                        "ok": True,
                        "payload": {
                            "model_id": item["model_id"],
                            "model_dir": str(model_dir),
                        },
                    }
                )
            return outputs

        def close(self):
            pass

        def join(self):
            pass

    class FakeContext:
        Pool = FakePool

    monkeypatch.setattr(
        ml.multiprocessing,
        "get_context",
        lambda method: FakeContext(),
    )
    monkeypatch.setattr(
        ml,
        "_validate_completed_ensemble",
        lambda ensemble_dir, state_table: validated.update(
            {"ensemble_dir": ensemble_dir, "state_table": state_table}
        ),
    )

    completed, _ = ml.train_excited_state_ensemble(
        ML_config=_config(),
        h5_dir="h5store",
        model_path=str(tmp_path / "model-{:04d}"),
        current_training_id=4,
        gpus_per_node=0,
        properties_list=_properties(),
    )

    assert completed == [True, True]
    assert validated["ensemble_dir"].endswith("model-0004")
    assert [row["state"] for row in validated["state_table"]] == [0, 1]


def test_completed_ensemble_requires_every_state_output_from_every_member(
    monkeypatch,
):
    expected = ["sE0", "F0", "sE1", "F1"]
    calls = {}

    def make_ensemble(pattern, targets, quiet):
        calls.update(
            {"pattern": pattern, "targets": targets, "quiet": quiet}
        )
        return object(), ({}, {key: 2 for key in expected})

    fake_hippynn = SimpleNamespace(
        graphs=SimpleNamespace(make_ensemble=make_ensemble)
    )
    monkeypatch.setitem(sys.modules, "hippynn", fake_hippynn)
    monkeypatch.setattr(ml.glob, "glob", lambda pattern: ["model-00", "model-01"])
    table = ml.validate_excited_state_training_config(
        _config(),
        _properties(),
    )

    ml._validate_completed_ensemble("ensemble", table)

    assert calls["targets"] == expected
    assert calls["quiet"] is True

    def incomplete_ensemble(pattern, targets, quiet):
        return object(), ({}, {"sE0": 2, "F0": 2, "sE1": 2, "F1": 1})

    fake_hippynn.graphs.make_ensemble = incomplete_ensemble
    with pytest.raises(RuntimeError, match="F1"):
        ml._validate_completed_ensemble("ensemble", table)


def test_completed_ensemble_requires_derived_gap_output(monkeypatch):
    expected = ["sE0", "F0", "sE1", "F1", "dE01"]
    calls = {}

    def make_ensemble(pattern, targets, quiet):
        calls["targets"] = targets
        return object(), ({}, {key: 2 for key in expected})

    fake_hippynn = SimpleNamespace(
        graphs=SimpleNamespace(make_ensemble=make_ensemble)
    )
    monkeypatch.setitem(sys.modules, "hippynn", fake_hippynn)
    monkeypatch.setattr(
        ml.glob,
        "glob",
        lambda pattern: ["model-00", "model-01"],
    )
    properties = _properties(with_gap=True)
    state_table = ml.validate_excited_state_training_config(
        _config(
            gap_targets={"enabled": True, "pairs": [[0, 1]]}
        ),
        properties,
    )
    gap_table = ml.derive_gap_property_table(
        properties,
        gap_config={"pairs": [[0, 1]]},
        require_properties=True,
    )

    ml._validate_completed_ensemble(
        "ensemble",
        state_table,
        gap_table,
    )

    assert calls["targets"] == expected


def test_task_signature_and_destructive_options(tmp_path, monkeypatch):
    assert ml.train_excited_state_HIPPYNN_ensemble_task.executors == [
        "alf_ML_executor"
    ]
    with pytest.raises(ValueError, match="remove_existing=True"):
        ml.validate_excited_state_task_options(
            remove_existing=True,
            h5_test_dir=None,
        )
    with pytest.raises(ValueError, match="h5_test_dir"):
        ml.validate_excited_state_task_options(
            remove_existing=False,
            h5_test_dir="separate-test-data",
        )
