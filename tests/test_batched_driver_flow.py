import parsl
import pytest
from ase import Atoms
from ase.io import write
from parsl import python_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor

from alframework.builders.builders import simple_cfg_loader_task
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampler_batching import (
    SamplerBatchBuffer,
    flatten_molecule_output,
    sampler_submission_groups,
    sampler_task_feed,
)
from alframework.tools.tools import parsl_task_queue


@python_app(executors=["alf_sampler_executor"])
def _list_returning_sampler(molecule_objects, sampler_config):
    del sampler_config
    for molecule in molecule_objects:
        molecule.update_metadata({"driver_flow_sampled": True})
    return [molecule_objects[0], [molecule_objects[1]]]


@python_app(executors=["alf_QM_executor"])
def _single_molecule_qm(molecule_object):
    molecule_object.update_metadata({"driver_flow_qm": True})
    return molecule_object


@pytest.fixture
def local_driver_parsl():
    try:
        parsl.dfk().cleanup()
    except Exception:
        pass
    try:
        parsl.clear()
    except Exception:
        pass
    parsl.load(
        Config(
            executors=[
                ThreadPoolExecutor(
                    label="alf_sampler_executor",
                    max_threads=4,
                ),
                ThreadPoolExecutor(
                    label="alf_QM_executor",
                    max_threads=2,
                ),
            ],
            strategy="none",
        )
    )
    yield
    try:
        parsl.dfk().cleanup()
    finally:
        parsl.clear()


def test_cfg_builder_strict_batch_list_sampler_to_individual_qm(
    tmp_path,
    local_driver_parsl,
):
    library = tmp_path / "cfg_library"
    library.mkdir()
    write(
        library / "water.cfg",
        Atoms(
            "OH2",
            positions=[
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ],
        ),
        format="cfg",
    )
    builder_config = {"molecule_library_dir": str(library)}
    sampler_config = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "fixed", "state": 0},
        "alchemi_baoab": {"batch_size": 2},
    }

    builder_queue = parsl_task_queue()
    for index in range(2):
        builder_queue.add_task(
            simple_cfg_loader_task(
                moleculeid=f"mol-0000-{index:010d}",
                builder_config=builder_config,
                shake=0.0,
            )
        )
    for task in builder_queue.task_list:
        task.result()
    builder_outputs, builder_failures = builder_queue.get_task_results()
    assert builder_failures == 0

    buffer = SamplerBatchBuffer()
    submission_groups = []
    for output in builder_outputs:
        for molecule in flatten_molecule_output(output):
            submission_groups.extend(
                sampler_submission_groups(
                    molecule,
                    sampler_config,
                    buffer,
                )
            )
    assert len(buffer) == 0
    assert len(submission_groups) == 1
    assert len(submission_groups[0]) == 2

    sampler_queue = parsl_task_queue()
    sampler_queue.add_task(
        _list_returning_sampler(
            **sampler_task_feed(
                submission_groups[0],
                sampler_config,
            )
        )
    )
    sampler_queue.task_list[0].result()
    sampler_outputs, sampler_failures = sampler_queue.get_task_results()
    assert sampler_failures == 0

    qm_queue = parsl_task_queue()
    for molecule in flatten_molecule_output(sampler_outputs):
        assert isinstance(molecule, MoleculesObject)
        qm_queue.add_task(_single_molecule_qm(molecule))
    for task in qm_queue.task_list:
        task.result()
    qm_outputs, qm_failures = qm_queue.get_task_results()

    assert qm_failures == 0
    assert len(qm_outputs) == 2
    assert {
        molecule.get_moleculeid() for molecule in qm_outputs
    } == {
        "mol-0000-0000000000",
        "mol-0000-0000000001",
    }
    assert all(
        molecule.get_metadata()["selected_state"] == 0
        and molecule.get_metadata()["driver_flow_sampled"] is True
        and molecule.get_metadata()["driver_flow_qm"] is True
        for molecule in qm_outputs
    )
