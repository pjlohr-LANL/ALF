import importlib.util
import json
from pathlib import Path

import pytest
from ase.io import read

from alframework.ml_interfaces.excited_state_hippynn_interface import (
    validate_excited_state_training_config,
)
from alframework.samplers.alchemi_sampling import (
    configured_alchemi_gap_rows,
)
from alframework.tools.molecular_topology import (
    load_reference_topology,
    validate_fixed_topology,
)
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampler_batching import sampler_submission_groups
from alframework.tools.tools import (
    build_input_dict,
    load_config_file,
    load_module_from_string,
)


EXAMPLE_DIR = (
    Path(__file__).resolve().parents[1] / "examples" / "excited_state_pyseqm"
)
MASTER_FILENAMES = (
    "master_config.json",
    "master_config_existing_h5.json",
    "master_config_debug.json",
)
EXPECTED_PROPERTIES = {
    key: [key, kind, 1.0]
    for state in range(6)
    for key, kind in (
        (f"sE{state}", "system"),
        (f"F{state}", "atomic"),
    )
}
EXPECTED_TASKS = {
    "builder_task": (
        "alframework.builders.h5_replay_builder.h5_replay_builder_task"
    ),
    "sampler_task": (
        "alframework.samplers.alchemi_sampling.alchemi_sampling_task"
    ),
    "QM_task": (
        "alframework.qm_interfaces.pyseqm_interface."
        "pyseqm_excited_state_task"
    ),
    "ML_task": (
        "alframework.ml_interfaces.excited_state_hippynn_interface."
        "train_excited_state_HIPPYNN_ensemble_task"
    ),
}


def _read_json(filename):
    return json.loads((EXAMPLE_DIR / filename).read_text(encoding="utf-8"))


def _load_example_configs(monkeypatch, master_filename):
    monkeypatch.chdir(EXAMPLE_DIR)
    master = load_config_file(str(EXAMPLE_DIR / master_filename))
    builder = load_config_file(
        master["builder_config_path"],
        master["master_directory"],
    )
    sampler = load_config_file(
        master["sampler_config_path"],
        master["master_directory"],
    )
    qm = load_config_file(
        master["QM_config_path"],
        master["master_directory"],
    )
    ml = load_config_file(
        master["ML_config_path"],
        master["master_directory"],
    )
    return master, builder, sampler, qm, ml


def test_all_example_json_files_parse():
    json_files = sorted(EXAMPLE_DIR.glob("*.json"))

    assert json_files
    for path in json_files:
        assert isinstance(
            json.loads(path.read_text(encoding="utf-8")),
            dict,
        ), path


@pytest.mark.parametrize("master_filename", MASTER_FILENAMES)
def test_master_configs_use_complete_six_state_contract(master_filename):
    master = _read_json(master_filename)

    assert {
        key: master[key] for key in EXPECTED_TASKS
    } == EXPECTED_TASKS
    assert master["properties_list"] == EXPECTED_PROPERTIES
    assert master["h5_path"] == "h5store/data-{:04d}.h5"
    assert master["gpus_per_node"] == 4


def test_production_and_debug_histories_are_isolated():
    production = _read_json("master_config.json")
    compatibility = _read_json("master_config_existing_h5.json")
    debug = _read_json("master_config_debug.json")

    assert compatibility == production
    assert production["status_path"] == "status.txt"
    assert production["model_path"] == "models/model-{:04d}"
    assert production["maximum_builder_structures"] == 50
    assert production["parallel_samplers"] == 100
    assert production["target_queued_QM"] == 6
    assert production["save_h5_threshold"] == 2000
    assert debug["status_path"] == "status_debug.txt"
    assert debug["model_path"] == "models_debug/model-{:04d}"
    assert debug["ML_config_path"] == "hippynn_config_debug.json"
    assert debug["maximum_builder_structures"] == 1


def test_production_and_debug_sampler_settings():
    production = _read_json("sampler_config.json")
    debug = _read_json("sampler_config_debug.json")

    for sampler in (production, debug):
        assert sampler["model_mode"] == "excited_state"
        assert sampler["state_selection"] == {
            "mode": "batch_cycle",
            "states": [0, 1, 2, 3, 4, 5],
        }
        assert sampler["uncertainty_policy"] == "stop"
        assert sampler["gap_diagnostics"]["enabled"] is False
        assert sampler["gap_seeking"]["enabled"] is False
        assert sampler["dataset_screening"] == {
            "enabled": True,
            "force": True,
            "min_distance": True,
            "topology": True,
        }
        assert sampler["distcut"] == pytest.approx(0.7)
        assert sampler["min_distance_cutoff"] == pytest.approx(0.7)
        assert sampler["max_force_cutoff"] == pytest.approx(16.0)
        assert sampler["friction_per_fs"] == pytest.approx(0.1)
        assert sampler["topology_check"]["reference_conformer_path"] == (
            "keto_form_coords.xyz"
        )

    assert production["alchemi_baoab"]["batch_size"] == 50
    assert production["dt"] == pytest.approx(0.1)
    assert production["maxt"] == pytest.approx(2.0)
    assert production["Escut"] == pytest.approx(0.001)
    assert production["Fscut"] == pytest.approx(0.01)
    assert debug["alchemi_baoab"]["batch_size"] == 1


def test_training_schema_has_six_states_and_no_gap_targets():
    master = _read_json("master_config.json")
    ml = _read_json("hippynn_config.json")
    sampler = _read_json("sampler_config.json")

    state_table = validate_excited_state_training_config(
        ml,
        master["properties_list"],
    )
    assert [row["state"] for row in state_table] == list(range(6))
    assert [row["energy_db_name"] for row in state_table] == [
        f"sE{state}" for state in range(6)
    ]
    assert ml["n_atoms"] == 15
    assert ml["network_params"]["possible_species"] == [0, 1, 6, 8]
    assert ml["n_models"] == 4
    assert ml["max_epochs"] == 200
    for state in range(6):
        diagnostics, seeking = configured_alchemi_gap_rows(
            sampler,
            master["properties_list"],
            state,
        )
        assert diagnostics == []
        assert seeking == {"enabled": False, "rows": []}


def test_keto_reference_topology_is_valid_and_stable():
    sampler = _read_json("sampler_config.json")
    atoms = read(EXAMPLE_DIR / "keto_form_coords.xyz")
    topology = load_reference_topology(
        sampler,
        master_directory=str(EXAMPLE_DIR),
    )
    result = validate_fixed_topology(atoms, topology)

    assert result.valid
    assert topology.atomic_numbers == (
        8,
        8,
        6,
        6,
        6,
        6,
        6,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    assert len(topology.bonds) == 14
    assert topology.reference_sha256 == (
        "254d4cd1674cd353606d12c6b4185d1f9b5e29d323500c9d292f2c66c37d211a"
    )


@pytest.mark.parametrize("master_filename", MASTER_FILENAMES)
def test_example_task_signatures_receive_every_required_input(
    monkeypatch,
    master_filename,
):
    master, builder, sampler, qm, ml = _load_example_configs(
        monkeypatch,
        master_filename,
    )
    all_configs = [master, builder, sampler, qm, ml]
    molecule = MoleculesObject(
        read(EXAMPLE_DIR / "keto_form_coords.xyz"),
        "mol-0000-0000000000",
    )
    status = {
        "current_model_id": 0,
        "current_training_id": 0,
        "current_h5_id": 1,
    }

    builder_task = load_module_from_string(master["builder_task"])
    sampler_task = load_module_from_string(master["sampler_task"])
    qm_task = load_module_from_string(master["QM_task"])
    ml_task = load_module_from_string(master["ML_task"])

    builder_input = build_input_dict(
        builder_task.func,
        [{"moleculeid": "example", "builder_config": builder,
          "sampler_config": sampler},
         *all_configs, status],
        raise_on_fail=True,
    )
    sampler_input = build_input_dict(
        sampler_task.func,
        [
            {
                "molecule_objects": [molecule],
                "sampler_config": sampler,
            },
            {"ML_config": ml},
            *all_configs,
            status,
        ],
        raise_on_fail=True,
    )
    qm_input = build_input_dict(
        qm_task.func,
        [{"molecule_object": molecule, "QM_config": qm},
         *all_configs, status],
        raise_on_fail=True,
    )
    ml_input = build_input_dict(
        ml_task.func,
        [{"ML_config": ml}, *all_configs, status],
        raise_on_fail=True,
    )

    assert set(builder_input) == {
        "moleculeid",
        "builder_config",
        "sampler_config",
        "h5_path",
        "current_h5_id",
        "master_directory",
    }
    assert sampler_input["ML_config"] is ml
    assert sampler_input["molecule_objects"] == [molecule]
    assert set(qm_input) >= {
        "molecule_object",
        "QM_config",
        "properties_list",
    }
    assert set(ml_input) >= {
        "ML_config",
        "h5_dir",
        "model_path",
        "current_training_id",
        "gpus_per_node",
        "properties_list",
    }


def test_six_state_cycle_forms_complete_state_specific_batches():
    sampler = _read_json("sampler_config.json")
    atoms = read(EXAMPLE_DIR / "keto_form_coords.xyz")
    submissions = []
    from alframework.tools.sampler_batching import SamplerBatchBuffer

    buffer = SamplerBatchBuffer()
    for index in range(300):
        molecule = MoleculesObject(
            atoms.copy(),
            f"mol-0000-{index:010d}",
        )
        submissions.extend(
            sampler_submission_groups(molecule, sampler, buffer)
        )

    assert len(buffer) == 0
    assert len(submissions) == 6
    assert [
        {item.get_metadata()["selected_state"] for item in batch}
        for batch in submissions
    ] == [{state} for state in range(6)]
    assert [len(batch) for batch in submissions] == [50] * 6


def test_darwin_parsl_resources_are_independent_and_dynamic():
    path = EXAMPLE_DIR / "parsl_configs.py"
    spec = importlib.util.spec_from_file_location(
        "excited_state_example_parsl_configs",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    executors = {
        executor.label: executor
        for executor in module.config_darwin.executors
    }
    assert set(executors) == {
        "alf_ML_executor",
        "alf_sampler_executor",
        "alf_QM_executor",
    }
    assert executors["alf_ML_executor"].provider.partition == "ml4chem"
    assert executors["alf_ML_executor"].provider.qos == "long"
    assert (
        executors["alf_sampler_executor"].provider.partition
        == "shared-gpu-ampere"
    )
    assert (
        executors["alf_QM_executor"].provider.partition
        == "shared-gpu-ampere"
    )
    assert executors["alf_ML_executor"].max_workers_per_node == 1
    assert executors["alf_sampler_executor"].max_workers_per_node == 4
    assert executors["alf_QM_executor"].max_workers_per_node == 4
    assert executors["alf_ML_executor"].available_accelerators == []
    assert len(executors["alf_sampler_executor"].available_accelerators) == 4
    assert len(executors["alf_QM_executor"].available_accelerators) == 4
    assert all(
        executor.provider.init_blocks == 0
        and executor.provider.min_blocks == 0
        for executor in executors.values()
    )
    assert executors["alf_ML_executor"].provider.max_blocks == 1
    assert (
        executors["alf_sampler_executor"].provider.max_blocks
        == executors["alf_QM_executor"].provider.max_blocks
        == 2
    )
    assert (
        executors["alf_sampler_executor"].provider
        is not executors["alf_QM_executor"].provider
    )
    assert module.DARWIN_ACCOUNT == "y2020-bf"
    assert "atomistic" in module.WORKER_INIT
    assert "WARP_CACHE_PATH" in module.WORKER_INIT
    assert "MPLCONFIGDIR" in module.WORKER_INIT
