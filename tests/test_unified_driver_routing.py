"""Exercise the real driver with a unified task and default ASE execution."""

import json
from pathlib import Path
import runpy
import sys

from ase import Atoms
import parsl
from parsl import python_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
import pytest

from alframework.tools.molecules_class import MoleculesObject
from alframework.tools import tools as tool_module


@pytest.mark.parametrize("explicit_ase", [False, True], ids=["task-default", "explicit-ase"])
def test_component_debug_routes_unified_ase_as_one_element_list(tmp_path, monkeypatch, explicit_ase):
    try:
        parsl.dfk().cleanup()
    except parsl.errors.NoDataFlowKernelError:
        pass
    parsl.clear()
    monkeypatch.chdir(tmp_path)
    model_pattern = str(tmp_path / "models/model-{:04d}")
    Path(model_pattern.format(0)).mkdir(parents=True)
    calls = []

    @python_app(executors=["alf_builder_executor"])
    def builder(moleculeid):
        return MoleculesObject(Atoms("H2", positions=[[0, 0, 0], [0, 0, .74]]), moleculeid)

    @python_app(executors=["alf_sampler_executor"])
    def sampler(molecule_objects, sampler_config, current_model_id):
        assert current_model_id == 0
        assert len(molecule_objects) == 1
        assert sampler_config["dynamics"]["driver"] == "ase"
        assert sampler_config["sampling"]["method"] == "mlmd"
        calls.append(molecule_objects[0].get_moleculeid())
        return molecule_objects

    @python_app(executors=["alf_QM_executor"])
    def qm(molecule_object):
        return molecule_object

    @python_app(executors=["alf_ML_executor"])
    def ml():
        return [True], 0

    task_path = "alframework.samplers.sampling.sampling_task"
    configuration = Config(
        executors=[ThreadPoolExecutor(label=f"alf_{role}_executor", max_threads=1)
                   for role in ("builder", "sampler", "QM", "ML")],
        strategy="none", run_dir=str(tmp_path / "runinfo"),
    )
    modules = {"test.config": configuration, "test.builder": builder,
               task_path: sampler, "test.qm": qm, "test.ml": ml}
    monkeypatch.setattr(tool_module, "load_module_from_string", modules.__getitem__)
    master = {
        "master_directory": "pwd", "builder_task": "test.builder",
        "sampler_task": task_path, "QM_task": "test.qm", "ML_task": "test.ml",
        "parsl_configuration": "test.config", "builder_config_path": "builder.json",
        "sampler_config_path": "sampler.json", "QM_config_path": "qm.json",
        "ML_config_path": "ml.json", "status_path": "status.json",
        "model_path": model_pattern, "h5_path": str(tmp_path / "h5/data-{:04d}.h5"),
        "properties_list": {"energy": ["energy", "system", 1.],
                            "forces": ["forces", "atomic", 1.]},
        "parallel_samplers": 2, "maximum_builder_structures": 1,
        "bootstrap_set": 1, "target_queued_QM": 2, "minimum_QM": 0,
        "save_h5_threshold": 2, "gpus_per_node": 0,
    }
    sampler_config = {}
    if explicit_ase:
        # Neither this inactive width nor the old batched backend may turn
        # ASE execution into a full-batch task or require a GPU stack.
        sampler_config = {"dynamics": {"driver": "ase", "backend": "torchsim"},
                          "batching": {"batch_size": 50, "partial_policy": "full_only"}}
    for name, value in [("master", master), ("sampler", sampler_config),
                        ("builder", {}), ("qm", {}), ("ml", {})]:
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    monkeypatch.setattr(sys, "argv", ["alframework", "--master", str(tmp_path / "master.json"),
                                     "--test_sampler"])
    try:
        with pytest.raises(SystemExit):
            runpy.run_module("alframework", run_name="__main__")
    finally:
        parsl.dfk().cleanup()
        parsl.clear()
    assert calls == ["test_builder"]
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["sampler_execution"]["driver"] == "ase"
    assert status["sampler_execution"]["sampling"]["method"] == "mlmd"
    assert status["sampler_batching"]["buffered_structures"] == 0
    assert status["sampler_batching"]["batch_size"] == 1
    assert json.loads((tmp_path / "sampler.json").read_text()) == sampler_config
