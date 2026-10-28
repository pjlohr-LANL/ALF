"""Real ALF workflows using coordinator-owned acquisition cohorts."""
from pathlib import Path
import os
import pickle
import sqlite3
import subprocess
import sys
import threading
import time

import h5py
import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes

from alframework.tools.staged_workflow import prepare_workflow, WorkflowRunner
from alframework.tools.workflow_io import atomic_json, read_json, verify_snapshot
from tests.test_staged_workflow import fixture_workflow
from tests.test_unified_workflow import journal_rows, load_workflow_ensemble


ROOT = Path(__file__).resolve().parents[1]


class ExcitedCohortMember(Calculator):
    implemented_properties = [key for state in range(4) for key in (f"sE{state}", f"F{state}")]

    def __init__(self, offset):
        super().__init__()
        self.offset = float(offset)

    def calculate(self, atoms=None, properties=("sE0",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {}
        for state in range(4):
            spring = (.1 + .01 * self.offset) * (1 + .1 * state)
            self.results[f"sE{state}"] = .5 * spring * np.square(atoms.positions).sum() + self.offset + .1 * state
            self.results[f"F{state}"] = -spring * atoms.positions


def load_excited_cohort_ensemble(directory, **options):
    return [ExcitedCohortMember(member.offset) for member in load_workflow_ensemble(directory, **options)]


def cohort_workflow(tmp_path, *, method="mlmd", driver="single", trajectory_policy="continue", **options):
    path = fixture_workflow(tmp_path, stages=1)
    master_path = path.parent / "master0.json"
    master = read_json(master_path)
    master["sampler_task"] = "alframework.samplers.sampling.sampling_task"
    atomic_json(master_path, master)
    sampler_path = path.parent / "sampler0.json"
    sampler = read_json(sampler_path)
    for name in ("uncertainty_policy", "return_top_k", "max_candidates_per_replica", "udd_bias_weight", "dynamics"):
        sampler.pop(name, None)
    sampler.update(
        driver={"strategy": driver, "engine": "ase" if driver == "single" else "alchemi",
                "strict_gpu": False, "allow_cpu_debug": True, "device": "cpu", "random_seed": 17},
        sampling={"propagator": "langevin", "biases": [] if method == "mlmd" else [{"strategy": "udd", "weight": .45}]},
        acquisition={"strategy": "uncertainty" if method == "mlmd" else "summed",
                     "trajectory_policy": trajectory_policy,
                     "selection": {"mode": "cohort_top_k", "cohort_size": 3, "k": 2}},
        batching={"batch_size": 4, "partial_policy": "timeout", "max_wait_seconds": .001},
        inference={"interface": "ase", "loader": "tests.test_unified_workflow.load_workflow_ensemble",
                   "options": options},
    )
    if method == "mlmd":
        sampler["score"] = {"mode": "max"}
    atomic_json(sampler_path, sampler)
    return path


def pool_path(stage):
    return stage / "status.cohorts.sqlite"


def pool_rows(stage, table):
    with sqlite3.connect(pool_path(stage)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY rowid")]


@pytest.mark.parametrize("method,driver,policy", [
    ("mlmd", "single", "stop"), ("udd", "single", "continue"),
    ("mlmd", "batched", "continue"), ("udd", "batched", "stop"),
])
def test_actual_cohort_selection_empty_trajectory_qm_storage_and_training(tmp_path, method, driver, policy):
    if driver == "batched":
        pytest.importorskip("nvalchemi")
    path = cohort_workflow(tmp_path, method=method, driver=driver, trajectory_policy=policy, empty_first=True)
    original = read_json(path.parent / "sampler0.json")
    root = tmp_path / "run"
    prepare_workflow(path, root)
    assert WorkflowRunner(root).run() == 0
    stage = root / "stages/00-alf0"
    assert verify_snapshot(stage / "checkpoints/3")["frames"] == 3
    assert read_json(root / "final_training/models/model-0000/input_frames.json")["frames"] == 3
    cohorts = pool_rows(stage, "cohorts")
    selected = [row for row in cohorts if row["state"] == "selected"]
    assert selected and all(row["admitted"] == row["completed"] == row["target"] == 3 for row in selected)
    sampler_rows = journal_rows(stage, "sampler")
    assert any(pickle.loads(row["output"]) == [] for row in sampler_rows if row["output"] is not None)
    completed_ids = set()
    for row in sampler_rows:
        inputs = pickle.loads(row["input"])
        molecules = inputs["molecule_objects"]
        assert len(molecules) == 1 if driver == "single" else 1 <= len(molecules) < 4
        for molecule in molecules:
            metadata = molecule.get_metadata()
            assert metadata["acquisition_cohort_id"]
            assert metadata["acquisition_trajectory_id"]
            if row["state"] == "delivered":
                completed_ids.add(metadata["acquisition_trajectory_id"])
    acquired = [pickle.loads(row["input"])["molecule_object"] for row in journal_rows(stage, "QM")
                if row["phase"] != "seed"]
    assert acquired
    for molecule in acquired:
        metadata = molecule.get_metadata()
        cohort = next(row for row in cohorts if row["id"] == metadata["acquisition_cohort_id"])
        assert cohort["completed"] == cohort["target"]
        assert metadata["acquisition_candidate_id"] == molecule.get_moleculeid()
        assert metadata["_workflow_origin"] == molecule.get_moleculeid()
        assert metadata["candidate_rank"] < 2
        assert metadata["acquisition_trajectory_id"] in completed_ids
    assert read_json(stage / "status.json")["sampler_capacity"]["accounted_replicas"] <= 2
    assert read_json(path.parent / "sampler0.json") == original


def test_controlled_restart_preserves_incomplete_cohort_and_completed_trajectories(tmp_path):
    path = cohort_workflow(tmp_path, method="udd", driver="single", delay_seconds=.2)
    root = tmp_path / "run"
    prepare_workflow(path, root)
    runner = WorkflowRunner(root)
    marker = root / "stages/00-alf0/unified_sampling_seen"

    def request_shutdown_after_first_sampler():
        deadline = time.monotonic() + 45
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.002)
        if marker.exists():
            runner.interrupt()

    thread = threading.Thread(target=request_shutdown_after_first_sampler, daemon=True)
    thread.start()
    assert runner.run() == 130
    thread.join(timeout=1)
    stage = root / "stages/00-alf0"
    partial = pool_rows(stage, "cohorts")
    assert partial and partial[0]["state"] == "open"
    assert 0 < partial[0]["admitted"] < partial[0]["target"]
    assert partial[0]["completed"] == partial[0]["admitted"]
    assert pool_rows(stage, "candidates")
    assert not [row for row in journal_rows(stage, "QM") if row["phase"] != "seed"]
    completed_before = {row["id"]: row["attempts"] for row in pool_rows(stage, "trajectories")
                        if row["state"] == "completed"}
    assert completed_before
    assert WorkflowRunner(root).run() == 0
    final = pool_rows(stage, "cohorts")
    original_cohort = next(row for row in final if row["id"] == partial[0]["id"])
    assert original_cohort["state"] == "selected"
    assert original_cohort["completed"] == original_cohort["target"] == 3
    attempts_after = {row["id"]: row["attempts"] for row in pool_rows(stage, "trajectories")}
    assert all(attempts_after[key] == attempts for key, attempts in completed_before.items())
    assert verify_snapshot(stage / "checkpoints/3")["frames"] == 3


def _run_bounded(runner, timeout=40):
    finished = threading.Event()
    expired = []

    def stop_hung_driver():
        if not finished.wait(timeout):
            expired.append(True)
            if runner.child is not None and runner.child.poll() is None:
                runner.child.kill()

    watchdog = threading.Thread(target=stop_hung_driver, daemon=True)
    watchdog.start()
    try:
        try:
            outcome = runner.run()
        except BaseException as exc:
            if expired:
                raise AssertionError("The cohort workflow did not reach a failure/recovery boundary.") from exc
            raise
        assert not expired
        return outcome
    finally:
        finished.set()
        watchdog.join(timeout=1)


def test_failed_cohort_sampler_retries_same_logical_inputs_without_repeating_seed_labels(tmp_path):
    path = cohort_workflow(tmp_path, method="udd", driver="single", fail_first=True)
    root = tmp_path / "run"
    prepare_workflow(path, root)
    with pytest.raises(RuntimeError, match="Stage alf0 failed"):
        _run_bounded(WorkflowRunner(root))
    stage = root / "stages/00-alf0"
    failed = [row for row in pool_rows(stage, "trajectories") if row["state"] == "failed"]
    assert failed
    original_inputs = {row["id"]: pickle.loads(row["input"]).get_metadata()["trajectory_seed"] for row in failed}
    seeds = {row["key"]: row["attempts"] for row in journal_rows(stage, "QM") if row["phase"] == "seed"}
    assert _run_bounded(WorkflowRunner(root)) == 0
    after = {row["id"]: row for row in pool_rows(stage, "trajectories")}
    for identifier, seed in original_inputs.items():
        assert after[identifier]["state"] == "completed"
        assert pickle.loads(after[identifier]["input"]).get_metadata()["trajectory_seed"] == seed
    assert seeds == {row["key"]: row["attempts"] for row in journal_rows(stage, "QM") if row["phase"] == "seed"}
    assert any(row["attempts"] > 1 for row in journal_rows(stage, "sampler"))
    assert verify_snapshot(stage / "checkpoints/3")["frames"] == 3


def test_actual_excited_state_cycle_follows_logical_cohorts_across_training_generations(tmp_path):
    path = fixture_workflow(tmp_path, excited=True, stages=1)
    master_path = path.parent / "master0.json"
    master = read_json(master_path)
    master["sampler_task"] = "alframework.samplers.sampling.sampling_task"
    atomic_json(master_path, master)
    sampler_path = path.parent / "sampler0.json"
    sampler = read_json(sampler_path)
    for key in ("alchemi_baoab", "uncertainty_policy", "return_top_k"):
        sampler.pop(key, None)
    sampler.update(
        driver={"strategy": "single", "engine": "ase", "device": "cpu", "random_seed": 17},
        sampling={"propagator": "langevin", "biases": []},
        acquisition={"strategy": "summed", "trajectory_policy": "stop",
                     "selection": {"mode": "cohort_top_k", "cohort_size": 3, "k": 1}},
        state_selection={"mode": "batch_cycle", "states": [0, 2]},
        inference={"interface": "ase", "loader": "tests.test_cohort_workflow.load_excited_cohort_ensemble", "options": {}},
    )
    atomic_json(sampler_path, sampler)
    definition = read_json(path)
    definition["stages"][0]["target_total_frames"] = 5
    atomic_json(path, definition)
    root = tmp_path / "run"
    prepare_workflow(path, root)
    assert WorkflowRunner(root).run() == 0
    stage = root / "stages/00-alf0"
    selected = [row for row in pool_rows(stage, "cohorts") if row["state"] == "selected"]
    assert len(selected) >= 4
    assert {row["selected_state"] for row in selected} == {0, 2}
    for row in selected:
        assert row["selected_state"] == [0, 2][row["ordinal"] % 2]
    for row in journal_rows(stage, "sampler"):
        inputs = pickle.loads(row["input"])
        for molecule in inputs["molecule_objects"]:
            metadata = molecule.get_metadata()
            assert metadata["selected_state"] == [0, 2][metadata["acquisition_cohort_ordinal"] % 2]
    assert verify_snapshot(stage / "checkpoints/5")["frames"] == 5


def _wait_for(predicate, process, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"ALF exited before the checkpoint condition: {process.returncode}")
        try:
            value = predicate()
        except (FileNotFoundError, sqlite3.OperationalError, OSError):
            value = False
        if value:
            return value
        time.sleep(.02)
    raise AssertionError("Timed out waiting for the ALF checkpoint condition")


@pytest.mark.parametrize("crash_point", ["sampling", "h5_commit"])
def test_ordinary_coordinator_restart_recovers_original_cohort_without_staged_workflow(tmp_path, crash_point):
    path = cohort_workflow(tmp_path, method="udd", driver="single", delay_seconds=.2)
    directory = path.parent
    master_path = directory / "master0.json"
    master = read_json(master_path)
    master.pop("run_control", None)
    atomic_json(master_path, master)
    # Ordinary upstream-style bootstrap may have builders in flight beyond
    # its requested seed count. Use an inexhaustible replay selection here;
    # exact sequential seed admission is covered by the controlled tests.
    builder_path = directory / "builder.json"
    builder = read_json(builder_path)
    builder["bootstrap_selection_mode"] = "deterministic_hash"
    atomic_json(builder_path, builder)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    # Exercise the ordinary coordinator without waiting its legacy 60-second
    # polling cadence. Scientific steps and subsecond worker delays are intact.
    launcher = '''
import runpy, sys, time
sleep = time.sleep
time.sleep = lambda seconds: sleep(.01 if seconds >= 1 else seconds)
if '--crash-after-h5' in sys.argv:
    import alframework.tools.dataset_screening as screening
    from pathlib import Path
    original_store = screening.screen_and_store_dataset
    def crash_boundary(*args, **kwargs):
        result = original_store(*args, **kwargs)
        if result[0] and any(m.get_metadata().get('acquisition_candidate_id') for m in args[1]):
            Path('h5-committed-before-status').touch()
            while True:
                sleep(.1)
        return result
    screening.screen_and_store_dataset = crash_boundary
sys.argv = ['alframework', '--master', sys.argv[1]]
runpy.run_module('alframework', run_name='__main__')
'''
    command = [sys.executable, "-u", "-c", launcher, str(master_path)]
    first_log = directory / "first-driver.log"
    second_log = directory / "resumed-driver.log"
    first = second = None
    try:
        with first_log.open("w") as stream:
            first_command = command + (["--crash-after-h5"] if crash_point == "h5_commit" else [])
            first = subprocess.Popen(first_command, cwd=directory, env=env, stdout=stream, stderr=subprocess.STDOUT)
            marker = "h5-committed-before-status" if crash_point == "h5_commit" else "unified_sampling_seen"
            _wait_for(lambda: (directory / marker).exists() and pool_rows(directory, "trajectories"), first)
            before = pool_rows(directory, "cohorts")[0]
            originals = {row["id"] for row in pool_rows(directory, "trajectories")}
            committed_shards = len(list((directory / "h5store").glob("*.h5")))
            first.kill()
            first.wait(timeout=15)
        assert before["admitted"] > 0
        with second_log.open("w") as stream:
            second = subprocess.Popen(command, cwd=directory, env=env, stdout=stream, stderr=subprocess.STDOUT)

            def original_cohort_is_selected_and_labels_stored():
                cohorts = pool_rows(directory, "cohorts")
                original = next(row for row in cohorts if row["id"] == before["id"])
                if original["state"] != "selected":
                    return False
                if crash_point == "h5_commit" and len(list((directory / "h5store").glob("*.h5"))) <= committed_shards:
                    return False
                stored = []
                for shard in (directory / "h5store").glob("*.h5"):
                    with h5py.File(shard) as handle:
                        for group in handle.values():
                            stored.extend(group["_id"].asstr()[:].tolist())
                return stored if len(stored) >= 3 else False

            stored = _wait_for(original_cohort_is_selected_and_labels_stored, second)
            assert len(stored) == len(set(stored))
            assert originals <= {row["id"] for row in pool_rows(directory, "trajectories")}
            original = next(row for row in pool_rows(directory, "cohorts") if row["id"] == before["id"])
            assert original["admitted"] == original["completed"] == original["target"] == 3
    except BaseException as exc:
        details = "\n".join(f"{p.name}:\n{p.read_text()[-7000:]}" for p in (first_log, second_log) if p.exists())
        raise AssertionError(details) from exc
    finally:
        for process in (first, second):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=15)
