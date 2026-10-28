"""Real unified sampling tasks inside budgeted ALF workflows.

Only QM labels and training artifacts are local analytic fixtures. The actual
sampler, acquisition controller, driver, journal, storage, and final-training
handoff execute through the normal staged subprocess workflow.
"""

from pathlib import Path
import pickle
import sqlite3
import threading
import time

from ase.calculators.calculator import Calculator, all_changes
import numpy as np
import pytest

from alframework.tools.staged_workflow import prepare_workflow, WorkflowRunner
from alframework.tools.workflow_io import atomic_json, read_json, verify_snapshot
from tests.test_staged_workflow import fixture_workflow


class WorkflowHarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, offset):
        super().__init__()
        self.offset = float(offset)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        spring = .1 + .01 * self.offset
        self.results = {"energy": .5 * spring * np.square(atoms.positions).sum() + self.offset,
                        "forces": -spring * atoms.positions}


def load_workflow_ensemble(directory, *, device="cpu", empty_first=False,
                           fail_first=False, delay_seconds=0.):
    directory = Path(directory)
    assert (directory / "input_frames.json").is_file(), "Sampling must use a completed trained generation."
    stage = directory.parent.parent
    (stage / "unified_sampling_seen").touch()
    if delay_seconds:
        time.sleep(delay_seconds)
    if fail_first:
        try:
            with (stage / "unified_failure_seen").open("x") as stream:
                stream.write("injected inference failure")
        except FileExistsError:
            pass
        else:
            raise RuntimeError("Injected unified inference failure")
    quiet = False
    if empty_first:
        try:
            with (stage / "unified_empty_seen").open("x") as stream:
                stream.write("one complete trajectory returns no candidates")
        except FileExistsError:
            pass
        else:
            quiet = True
    offsets = np.zeros(4) if quiet else np.linspace(-.03, .03, 4)
    return [WorkflowHarmonicCalculator(offset) for offset in offsets]


def unified_workflow(tmp_path, *, method="mlmd", driver="ase", **inference_options):
    path = fixture_workflow(tmp_path, stages=1)
    master_path = path.parent / "master0.json"
    master = read_json(master_path)
    master["sampler_task"] = "alframework.samplers.sampling.sampling_task"
    atomic_json(master_path, master)
    sampler_path = path.parent / "sampler0.json"
    sampler = read_json(sampler_path)
    sampler.pop("uncertainty_policy")
    sampler.update(
        sampling={"method": method},
        acquisition={"method": "uncertainty", "policy": "continue"},
        dynamics={"driver": driver, "backend": "alchemi", "strict_gpu": False,
                  "allow_cpu_debug": True, "device": "cpu", "random_seed": 17},
        batching={"batch_size": 4, "partial_policy": "timeout", "max_wait_seconds": .001},
        inference={"interface": "ase", "loader": "tests.test_unified_workflow.load_workflow_ensemble",
                   "options": inference_options},
        udd_bias_weight=.45, max_candidates_per_replica=1,
    )
    atomic_json(sampler_path, sampler)
    return path


def journal_rows(stage, role):
    with sqlite3.connect(stage / "control/journal.sqlite") as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute("SELECT * FROM tasks WHERE role=? ORDER BY created", (role,))]


@pytest.mark.parametrize("method", ["mlmd", "udd"])
@pytest.mark.parametrize("driver", ["ase", "batched"])
def test_unified_full_workflow_empty_results_storage_retraining_and_capacity(tmp_path, method, driver):
    if driver == "batched":
        pytest.importorskip("nvalchemi")
    path = unified_workflow(tmp_path, method=method, driver=driver, empty_first=True)
    frozen_config = read_json(path.parent / "sampler0.json")
    root = tmp_path / "run"
    prepare_workflow(path, root)
    assert WorkflowRunner(root).run() == 0
    stage = root / "stages/00-alf0"
    assert read_json(root / "workflow_status.json")["state"] == "completed"
    assert verify_snapshot(stage / "checkpoints/3")["frames"] == 3
    assert read_json(root / "final_training/models/model-0000/input_frames.json")["frames"] == 3
    samplers = journal_rows(stage, "sampler")
    assert len(samplers) >= 2
    assert any(pickle.loads(row["output"]) == [] for row in samplers if row["output"] is not None)
    for row in samplers:
        inputs = pickle.loads(row["input"])
        width = len(inputs["molecule_objects"])
        assert width == 1 if driver == "ase" else 1 <= width < 4
        assert inputs["sampler_config"]["sampling"]["method"] == method
        assert inputs["sampler_config"]["dynamics"]["driver"] == driver
    acquired = [pickle.loads(row["input"])["molecule_object"] for row in journal_rows(stage, "QM")
                if "boot" not in pickle.loads(row["input"])["molecule_object"].get_moleculeid()]
    assert acquired
    for molecule in acquired:
        metadata = molecule.get_metadata()
        assert metadata["sampling_method"] == method
        assert metadata["execution_driver"] == driver
        assert metadata["acquisition_method"] == "uncertainty"
        assert metadata["calculator_interface"] == "ase_fallback"
        assert metadata["parent_molecule_id"]
        assert metadata["force_screen_source"] == "ensemble_mean_physical_all_states"
    status = read_json(stage / "status.json")
    assert status["sampler_capacity"]["accounted_replicas"] <= 2
    assert status["lifetime_failed_sampler_tasks"] == 0
    assert status["sampler_execution"]["driver"] == driver
    assert read_json(path.parent / "sampler0.json") == frozen_config


def test_unified_failed_sampler_resume_reuses_seed_work_and_width(tmp_path):
    path = unified_workflow(tmp_path, method="udd", driver="ase", fail_first=True)
    root = tmp_path / "run"
    prepare_workflow(path, root)
    with pytest.raises(RuntimeError, match="Stage alf0 failed"):
        WorkflowRunner(root).run()
    stage = root / "stages/00-alf0"
    seeds = {row["key"]: row["attempts"] for row in journal_rows(stage, "QM")
             if row["phase"] == "seed"}
    assert any(row["state"] == "failed" for row in journal_rows(stage, "sampler"))
    assert WorkflowRunner(root).run() == 0
    assert seeds == {row["key"]: row["attempts"] for row in journal_rows(stage, "QM")
                     if row["phase"] == "seed"}
    assert any(row["attempts"] > 1 for row in journal_rows(stage, "sampler"))
    assert all(len(pickle.loads(row["input"])["molecule_objects"]) == 1
               for row in journal_rows(stage, "sampler"))


def test_unified_pause_drains_and_resume_preserves_committed_qm(tmp_path):
    path = unified_workflow(tmp_path, method="udd", driver="ase", delay_seconds=.1)
    root = tmp_path / "run"
    prepare_workflow(path, root)
    runner = WorkflowRunner(root)
    marker = root / "stages/00-alf0/unified_sampling_seen"

    def interrupt_after_sampler_started():
        deadline = time.monotonic() + 30
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.002)
        if marker.exists():
            runner.interrupt()

    thread = threading.Thread(target=interrupt_after_sampler_started, daemon=True)
    thread.start()
    assert runner.run() == 130
    thread.join(timeout=1)
    stage = root / "stages/00-alf0"
    assert read_json(stage / "control/progress.json")["state"] == "paused"
    committed = {row["key"]: row["attempts"] for row in journal_rows(stage, "QM")
                 if row["state"] == "consumed"}
    assert committed
    assert WorkflowRunner(root).run() == 0
    after = {row["key"]: row["attempts"] for row in journal_rows(stage, "QM")}
    assert all(after[key] == attempts for key, attempts in committed.items())
    assert read_json(root / "workflow_status.json")["state"] == "completed"
