import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from alframework.builders.builders import simple_cfg_loader_task
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
    "sE0": ["state_0_energy", "system", 1.0],
    "F0": ["state_0_forces", "atomic", 1.0],
    "sE1": ["state_1_energy", "system", 1.0],
    "F1": ["state_1_forces", "atomic", 1.0],
    "dE01": ["gap_01", "system", 1.0],
}
EXPECTED_TASKS = {
    "builder_task": (
        "alframework.builders.builders.simple_cfg_loader_task"
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
        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict), path


@pytest.mark.parametrize("master_filename", MASTER_FILENAMES)
def test_master_configs_use_complete_excited_state_contract(master_filename):
    master = _read_json(master_filename)

    assert {
        key: master[key] for key in EXPECTED_TASKS
    } == EXPECTED_TASKS
    assert master["properties_list"] == EXPECTED_PROPERTIES
    assert master["maximum_builder_structures"] == 1
    assert master["gpus_per_node"] == 4


def test_startup_modes_use_the_expected_cfg_and_hdf5_paths():
    bootstrap = _read_json("master_config.json")
    existing = _read_json("master_config_existing_h5.json")
    bootstrap_builder = _read_json("builder_config_bootstrap.json")
    existing_builder = _read_json("builder_config_existing_h5.json")

    assert bootstrap["h5_path"] == "h5store_bootstrap/data-{:04d}.h5"
    assert (
        bootstrap["builder_config_path"]
        == "builder_config_bootstrap.json"
    )
    assert bootstrap_builder["shake"] > 0.0
    assert bootstrap_builder["pbc"] is False

    assert existing["h5_path"] == "h5store/data-{:04d}.h5"
    assert (
        existing["builder_config_path"]
        == "builder_config_existing_h5.json"
    )
    assert existing_builder["shake"] == 0.0
    assert existing_builder["pbc"] is False


def test_production_and_debug_sampler_settings():
    production = _read_json("sampler_config.json")
    debug = _read_json("sampler_config_debug.json")

    for sampler in (production, debug):
        assert sampler["model_mode"] == "excited_state"
        assert sampler["state_selection"] == {
            "mode": "batch_cycle",
            "states": [0, 1],
        }
        assert sampler["uncertainty_policy"] == "stop"
        assert sampler["gap_diagnostics"]["enabled"] is True
        assert sampler["gap_seeking"]["enabled"] is False
        assert (
            sampler["gap_seeking"]["switch_policy"] == "stay_fixed"
        )
        assert sampler["topology_check"]["enabled"] is True
        assert sampler["distcut"] < 0.957
        assert sampler["friction_per_fs"] == pytest.approx(0.0019645)
        assert sampler["MLMD_calculator_options"]["well_params"]

    assert production["alchemi_baoab"]["batch_size"] == 50
    assert debug["alchemi_baoab"]["batch_size"] == 1


def test_training_and_gap_schemas_validate_for_both_sampling_states():
    master = _read_json("master_config.json")
    ml = _read_json("hippynn_config.json")
    sampler = _read_json("sampler_config.json")

    state_table = validate_excited_state_training_config(
        ml,
        master["properties_list"],
    )
    assert [row["state"] for row in state_table] == [0, 1]
    assert [row["energy_db_name"] for row in state_table] == [
        "state_0_energy",
        "state_1_energy",
    ]
    for state in (0, 1):
        diagnostic_rows, seeking = configured_alchemi_gap_rows(
            sampler,
            master["properties_list"],
            state,
        )
        assert [row["gap_db_name"] for row in diagnostic_rows] == ["gap_01"]
        assert seeking == {"enabled": False, "rows": []}


def test_cfg_loader_override_matches_nonperiodic_xyz_reference(monkeypatch):
    _, builder, sampler, _, _ = _load_example_configs(
        monkeypatch,
        "master_config_existing_h5.json",
    )
    molecule = simple_cfg_loader_task.func(
        moleculeid="example-water",
        builder_config=builder,
        shake=0.0,
    )
    cfg_atoms = molecule.get_atoms()
    xyz_atoms = read(EXAMPLE_DIR / "fragment_library" / "water.xyz")

    assert cfg_atoms.get_atomic_numbers().tolist() == [8, 1, 1]
    assert xyz_atoms.get_atomic_numbers().tolist() == [8, 1, 1]
    assert not cfg_atoms.get_pbc().any()
    np.testing.assert_allclose(
        cfg_atoms.get_positions(),
        xyz_atoms.get_positions(),
        atol=1.0e-8,
    )

    pytest.importorskip("rdkit")
    topology = load_reference_topology(
        sampler,
        master_directory=str(EXAMPLE_DIR),
    )
    result = validate_fixed_topology(cfg_atoms, topology)
    assert result.valid
    assert topology.atomic_numbers == (8, 1, 1)
    assert topology.bonds == ((0, 1), (0, 2))


def test_cfg_loader_omitting_pbc_override_preserves_ase_behavior(monkeypatch):
    _, builder, _, _, _ = _load_example_configs(
        monkeypatch,
        "master_config_existing_h5.json",
    )
    builder.pop("pbc")

    molecule = simple_cfg_loader_task.func(
        moleculeid="legacy-cfg",
        builder_config=builder,
        shake=0.0,
    )

    assert molecule.get_atoms().get_pbc().all()


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
    molecule = simple_cfg_loader_task.func(
        moleculeid="mol-0000-0000000000",
        builder_config=builder,
        shake=0.0,
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
        [
            {
                "moleculeid": "example",
                "builder_config": builder,
            },
            *all_configs,
            status,
        ],
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
        [
            {
                "molecule_object": molecule,
                "QM_config": qm,
            },
            *all_configs,
            status,
        ],
        raise_on_fail=True,
    )
    ml_input = build_input_dict(
        ml_task.func,
        [{"ML_config": ml}, *all_configs, status],
        raise_on_fail=True,
    )

    assert set(builder_input) == {"moleculeid", "builder_config", "shake"}
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


def test_two_state_cycle_forms_complete_state_specific_batches(monkeypatch):
    _, builder, sampler, _, _ = _load_example_configs(
        monkeypatch,
        "master_config_existing_h5.json",
    )
    submissions = []
    from alframework.tools.sampler_batching import SamplerBatchBuffer

    buffer = SamplerBatchBuffer()
    for index in range(100):
        molecule = simple_cfg_loader_task.func(
            moleculeid=f"mol-0000-{index:010d}",
            builder_config=builder,
            shake=0.0,
        )
        submissions.extend(
            sampler_submission_groups(molecule, sampler, buffer)
        )

    assert len(buffer) == 0
    assert len(submissions) == 2
    assert [
        {item.get_metadata()["selected_state"] for item in batch}
        for batch in submissions
    ] == [{0}, {1}]
    assert [len(batch) for batch in submissions] == [50, 50]


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
    assert all(
        executor.provider.init_blocks == 0
        and executor.provider.min_blocks == 0
        for executor in executors.values()
    )
    assert executors["alf_ML_executor"].provider.max_blocks == 1
    assert (
        executors["alf_sampler_executor"].provider
        is not executors["alf_QM_executor"].provider
    )
    assert "WARP_CACHE_PATH" in module.WORKER_INIT
    assert "MPLCONFIGDIR" in module.WORKER_INIT
    assert hasattr(module, "DARWIN_CACHE_ROOT")
