import importlib.util
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from ase.io import read

from alframework.ml_interfaces.excited_state_hippynn_interface import (
    validate_excited_state_training_config,
)
from alframework.qm_interfaces.gpu4pyscf_interface import (
    validate_gpu4pyscf_config,
    validate_gpu4pyscf_properties,
)
from alframework.tools.molecular_topology import (
    load_reference_topology,
    validate_fixed_topology,
)
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import (
    build_input_dict,
    load_config_file,
    load_module_from_string,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPOSITORY_ROOT / "examples" / "excited_state_gpu4pyscf"
PYSEQM_EXAMPLE_DIR = REPOSITORY_ROOT / "examples" / "excited_state_pyseqm"
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
EXPECTED_COMMON_TASKS = {
    "sampler_task": (
        "alframework.samplers.alchemi_sampling.alchemi_sampling_task"
    ),
    "QM_task": (
        "alframework.qm_interfaces.gpu4pyscf_interface."
        "gpu4pyscf_excited_state_task"
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


def _load_parsl_module(monkeypatch):
    monkeypatch.setenv(
        "ALF_DARWIN_ENV_ACTIVATION",
        "activate atomistic-test-environment",
    )
    monkeypatch.setenv(
        "ALF_DARWIN_GPU4PYSCF_ENV_ACTIVATION",
        "activate gpu4pyscf-test-environment",
    )
    monkeypatch.setenv(
        "ALF_DARWIN_GPU4PYSCF_PYTHONPATH",
        "/gpu4pyscf/source",
    )
    path = EXAMPLE_DIR / "parsl_configs.py"
    spec = importlib.util.spec_from_file_location(
        "gpu4pyscf_example_parsl_configs",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_all_gpu4pyscf_example_json_files_parse():
    json_files = sorted(EXAMPLE_DIR.glob("*.json"))

    assert json_files
    for path in json_files:
        assert isinstance(
            json.loads(path.read_text(encoding="utf-8")),
            dict,
        ), path


@pytest.mark.parametrize("master_filename", MASTER_FILENAMES)
def test_masters_use_the_validated_six_surface_contract(master_filename):
    master = _read_json(master_filename)

    assert {
        key: master[key] for key in EXPECTED_COMMON_TASKS
    } == EXPECTED_COMMON_TASKS
    assert master["properties_list"] == EXPECTED_PROPERTIES
    assert master["gpus_per_node"] == 4

    options = validate_gpu4pyscf_config(_read_json("QM_config.json"))
    state_table, gap_table = validate_gpu4pyscf_properties(
        master["properties_list"],
        nroots=options["nroots"],
    )
    assert options["xc"] == "cam-b3lyp"
    assert options["basis"] == "6-31g*"
    assert options["density_fit"] is True
    assert options["grids_level"] == 3
    assert options["nroots"] == 5
    assert options["tda_conv_tol"] == pytest.approx(1.0e-6)
    assert options["tda_max_cycle"] == 300
    assert options["energy_offset_eV"] == pytest.approx(-9405.0)
    assert [row["state"] for row in state_table] == list(range(6))
    assert gap_table == []


def test_startup_modes_and_output_histories_are_independent():
    bootstrap = _read_json("master_config.json")
    existing = _read_json("master_config_existing_h5.json")
    debug = _read_json("master_config_debug.json")

    assert bootstrap["builder_task"] == (
        "alframework.builders.h5_replay_builder."
        "h5_replay_builder_local_task"
    )
    assert bootstrap["builder_config_path"] == (
        "builder_config_bootstrap.json"
    )
    assert bootstrap["bootstrap_set"] == 4990
    assert bootstrap["maximum_builder_structures"] == 50
    assert bootstrap["save_h5_threshold"] == 50
    assert existing["builder_task"] == (
        "alframework.builders.h5_replay_builder."
        "h5_replay_builder_task"
    )
    assert existing["builder_config_path"] == (
        "builder_config_existing_h5.json"
    )
    assert debug["builder_task"] == bootstrap["builder_task"]
    assert debug["builder_config_path"] == bootstrap["builder_config_path"]
    assert bootstrap["h5_path"] == existing["h5_path"]
    assert debug["h5_path"] == "h5store_debug/data-{:04d}.h5"
    assert debug["model_path"] == "models_debug/model-{:04d}"
    assert debug["status_path"] == "status_debug.txt"
    assert bootstrap["target_queued_QM"] == 6
    assert debug["target_queued_QM"] == 4

    bootstrap_builder = _read_json("builder_config_bootstrap.json")
    assert bootstrap_builder == {
        "source_priority": "h5_only",
        "selection_seed": 42,
        "bootstrap_h5_path": (
            "../excited_state_pyseqm/h5store/data-{:04d}.h5"
        ),
        "bootstrap_h5_count": 1,
        "bootstrap_selection_mode": "sequential",
    }
    assert not (
        PYSEQM_EXAMPLE_DIR / "QM_config_gpu4pyscf.json"
    ).exists()
    assert not (
        PYSEQM_EXAMPLE_DIR / "master_config_gpu4pyscf_debug.json"
    ).exists()


def test_cfg_and_reference_preserve_keto_order_and_topology():
    cfg_atoms = read(
        EXAMPLE_DIR / "fragment_library" / "keto_acetylacetone.cfg"
    )
    reference = read(EXAMPLE_DIR / "keto_form_coords.xyz")
    cfg_atoms.set_pbc(False)

    np.testing.assert_array_equal(
        cfg_atoms.get_atomic_numbers(),
        reference.get_atomic_numbers(),
    )
    translation = (
        cfg_atoms.get_positions()[0] - reference.get_positions()[0]
    )
    np.testing.assert_allclose(
        cfg_atoms.get_positions(),
        reference.get_positions() + translation,
        atol=2.0e-5,
    )

    sampler = _read_json("sampler_config.json")
    topology = load_reference_topology(
        sampler,
        master_directory=str(EXAMPLE_DIR),
    )
    assert validate_fixed_topology(cfg_atoms, topology).valid
    assert topology.atomic_numbers == tuple(
        reference.get_atomic_numbers().tolist()
    )


def test_bootstrap_replay_uses_source_geometry_without_pyseqm_labels(
    monkeypatch,
):
    master, builder, sampler, _, _ = _load_example_configs(
        monkeypatch,
        "master_config_debug.json",
    )
    task = load_module_from_string(master["builder_task"])
    task_input = build_input_dict(
        task.func,
        [
            {
                "moleculeid": "mol-boot-0000000000",
                "builder_config": builder,
                "sampler_config": sampler,
            },
            master,
            builder,
            sampler,
            {"current_h5_id": 0},
        ],
        raise_on_fail=True,
    )
    molecule = task.func(**task_input)
    atoms = molecule.get_atoms()
    source_path = (
        PYSEQM_EXAMPLE_DIR / "h5store" / "data-0000.h5"
    )
    with h5py.File(source_path, "r") as handle:
        group = handle["C05_H08_O02"]
        reorder = np.argsort(
            np.asarray(group["topology_atom_ids"][0]),
            kind="stable",
        )
        expected_positions = np.asarray(group["coordinates"][0])[reorder]

    assert isinstance(molecule, MoleculesObject)
    assert not np.any(atoms.get_pbc())
    np.testing.assert_allclose(atoms.get_positions(), expected_positions)
    assert molecule.get_results() == {}
    assert molecule.get_metadata()["replay_source_id"]
    assert molecule.get_metadata()["replay_global_frame_index"] == 0
    assert molecule.get_metadata()["replay_external_bootstrap"] is True


def test_pyseqm_source_shard_is_read_only_geometry_input():
    source = PYSEQM_EXAMPLE_DIR / "h5store" / "data-0000.h5"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    assert digest == (
        "f47183504f50b191bb4dbee5ef1df13c2"
        "b83e76771d7fe30aaa9f774b83a9714"
    )
    with h5py.File(source, "r") as handle:
        group = handle["C05_H08_O02"]
        assert group["coordinates"].shape == (4990, 15, 3)
        assert group["topology_atom_ids"].shape == (4990, 15)
        assert all(
            name in group
            for state in range(6)
            for name in (f"sE{state}", f"F{state}")
        )


@pytest.mark.parametrize("master_filename", MASTER_FILENAMES)
def test_example_task_signatures_receive_required_inputs(
    monkeypatch,
    master_filename,
):
    master, builder, sampler, qm, ml = _load_example_configs(
        monkeypatch,
        master_filename,
    )
    all_configs = [master, builder, sampler, qm, ml]
    status = {
        "current_model_id": 0,
        "current_training_id": 0,
        "current_h5_id": 1,
    }
    molecule = MoleculesObject(
        read(EXAMPLE_DIR / "keto_form_coords.xyz"),
        "gpu4pyscf-example",
    )

    builder_task = load_module_from_string(master["builder_task"])
    sampler_task = load_module_from_string(master["sampler_task"])
    qm_task = load_module_from_string(master["QM_task"])
    ml_task = load_module_from_string(master["ML_task"])

    builder_input = build_input_dict(
        builder_task.func,
        [
            {
                "moleculeid": "gpu4pyscf-example",
                "builder_config": builder,
                "sampler_config": sampler,
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
        [{"molecule_object": molecule, "QM_config": qm},
         *all_configs, status],
        raise_on_fail=True,
    )
    ml_input = build_input_dict(
        ml_task.func,
        [{"ML_config": ml}, *all_configs, status],
        raise_on_fail=True,
    )

    assert builder_input["builder_config"] is builder
    assert sampler_input["molecule_objects"] == [molecule]
    assert set(qm_input) == {
        "molecule_object",
        "QM_config",
        "properties_list",
        "gpus_per_node",
    }
    assert set(ml_input) >= {
        "ML_config",
        "h5_dir",
        "model_path",
        "current_training_id",
        "gpus_per_node",
        "properties_list",
    }


def test_training_and_sampling_configs_cover_all_six_states():
    master = _read_json("master_config.json")
    ml = _read_json("hippynn_config.json")
    sampler = _read_json("sampler_config.json")

    state_table = validate_excited_state_training_config(
        ml,
        master["properties_list"],
    )
    assert [row["state"] for row in state_table] == list(range(6))
    assert ml["n_models"] == 4
    assert ml["n_atoms"] == 15
    assert sampler["state_selection"] == {
        "mode": "batch_cycle",
        "states": [0, 1, 2, 3, 4, 5],
    }
    assert sampler["alchemi_baoab"]["batch_size"] == 50
    assert sampler["topology_check"]["reference_conformer_path"] == (
        "keto_form_coords.xyz"
    )


def test_darwin_qm_executor_runs_one_molecule_per_gpu(monkeypatch):
    module = _load_parsl_module(monkeypatch)
    executors = {
        executor.label: executor
        for executor in module.config_darwin.executors
    }
    debug_executors = {
        executor.label: executor
        for executor in module.config_darwin_debug.executors
    }

    assert set(executors) == {
        "alf_builder_executor",
        "alf_ML_executor",
        "alf_sampler_executor",
        "alf_QM_executor",
    }
    builder = executors["alf_builder_executor"]
    ml = executors["alf_ML_executor"]
    sampler = executors["alf_sampler_executor"]
    qm = executors["alf_QM_executor"]
    assert builder.max_threads == 4
    assert ml.provider.partition == "ml4chem"
    assert sampler.provider.partition == "shared-gpu-ampere"
    assert qm.provider.partition == "shared-gpu-ampere"
    assert ml.max_workers_per_node == 1
    assert sampler.max_workers_per_node == 4
    assert qm.max_workers_per_node == 4
    assert ml.available_accelerators == []
    assert len(sampler.available_accelerators) == 4
    assert len(qm.available_accelerators) == 4
    assert qm.prefetch_capacity == 0
    assert qm.cores_per_worker == pytest.approx(8.0)
    assert ml.provider is not sampler.provider
    assert sampler.provider is not qm.provider
    assert all(
        executor.provider.init_blocks == 0
        and executor.provider.min_blocks == 0
        for label, executor in executors.items()
        if label != "alf_builder_executor"
    )
    assert ml.provider.max_blocks == 1
    assert sampler.provider.max_blocks == 2
    assert qm.provider.max_blocks == 2
    assert sampler.provider.qos == "long"
    assert sampler.provider.walltime == "12:00:00"
    assert qm.provider.qos == "long"
    assert qm.provider.walltime == "08:00:00"
    assert qm.drain_period == 28200
    assert debug_executors["alf_QM_executor"].provider.max_blocks == 1
    assert "atomistic-test-environment" in module.ATOMISTIC_WORKER_INIT
    assert "gpu4pyscf-test-environment" not in (
        module.ATOMISTIC_WORKER_INIT
    )
    assert "gpu4pyscf-test-environment" in (
        module.GPU4PYSCF_WORKER_INIT
    )
    assert "/gpu4pyscf/source" in module.GPU4PYSCF_WORKER_INIT
    assert "CUPY_CACHE_DIR" in module.GPU4PYSCF_WORKER_INIT
    assert (
        load_module_from_string(
            _read_json("master_config.json")["builder_task"]
        ).executors
        == ["alf_builder_executor"]
    )


def test_headnode_launcher_uses_validated_darwin_runtime():
    launcher = (
        EXAMPLE_DIR / "launch_headnode.sh"
    ).read_text(encoding="utf-8")

    assert "/vast/home/pjlohr/.conda/envs/alf_env" in launcher
    assert "PYTHONNOUSERSITE=1" in launcher
    assert launcher.count("#SBATCH --constraint=gpu_count:4") == 2
    assert "ALF_DARWIN_SAMPLER_QOS" in launcher
    assert "ALF_DARWIN_SAMPLER_WALLTIME" in launcher
    assert "12:00:00" in launcher
    assert "ALF_DARWIN_GPU4PYSCF_QOS" in launcher
    assert "ALF_DARWIN_GPU4PYSCF_WALLTIME" in launcher
    assert "08:00:00" in launcher
    assert "ALF_DARWIN_GPU4PYSCF_DRAIN_PERIOD" in launcher
    assert "28200" in launcher
    assert 'MASTER_CONFIG="${ALF_GPU4PYSCF_MASTER:-master_config.json}"' in (
        launcher
    )
    assert '-m alframework --master "${MASTER_CONFIG}"' in launcher
    assert "\nsbatch " not in launcher


def test_persistent_driver_uses_ml4chem_for_two_days_without_a_gpu():
    submission = (
        EXAMPLE_DIR / "submit_darwin.slurm"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --partition=ml4chem" in submission
    assert "#SBATCH --account=y2020-bf" in submission
    assert "#SBATCH --qos=long" in submission
    assert "#SBATCH --time=2-00:00:00" in submission
    assert "#SBATCH --ntasks=1" in submission
    assert "#SBATCH --gres" not in submission
    assert "#SBATCH --gpus" not in submission


def test_readme_documents_current_darwin_launch_and_recovery_contract():
    readme = (EXAMPLE_DIR / "README.md").read_text(encoding="utf-8")
    master = _read_json("master_config.json")
    builder = _read_json("builder_config_bootstrap.json")
    qm = _read_json("QM_config.json")

    assert "each frontend has its own local tmux server" in readme
    assert "ssh darwin-fe1" in readme
    assert "tmux attach -t alf_gpu4pyscf" in readme
    assert "CPU-only driver task on `ml4chem`" in readme
    assert "`qos=long`" in readme
    assert "two-day walltime" in readme
    assert "sbatch submit_darwin.slurm" in readme
    assert "Cancel only the explicit job IDs" in readme
    assert "Never use a user-wide cancellation" in readme
    assert (
        "Preserve\n`status.txt`, every HDF5 shard, promoted models"
        in readme
    )
    assert "queues exist only in the driver process" in readme
    assert "a crash during model training requires manual" in readme
    assert "current_training_id" in readme
    assert f"{master['bootstrap_set']:,}" in readme
    assert builder["bootstrap_h5_path"].format(0) in readme
    assert f"`{qm['energy_offset_eV']} eV`" in readme
    assert "50 serially generated GPU4PySCF labels" not in readme
    assert "persistent `general`-partition allocation" not in readme
