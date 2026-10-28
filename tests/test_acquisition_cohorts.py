"""Logical selection is invariant to physical batches and survives handoffs."""
from copy import deepcopy
import sqlite3

import pytest
from ase import Atoms

from alframework.tools.acquisition_cohorts import AcquisitionCohorts
from alframework.tools.molecules_class import MoleculesObject


def config():
    return {"driver": {"strategy": "batched", "engine": "torchsim"},
            "sampling": {"propagator": "langevin", "biases": []},
            "acquisition": {"strategy": "uncertainty", "trajectory_policy": "continue"},
            "dt": .1, "maxt": 1., "Escut": .1, "Fscut": .2}


def molecule(index):
    return MoleculesObject(Atoms("H2", positions=[[0., 0., 0.], [1.+index*.001, 0., 0.]]), f"input-{index}")


def candidates(parent, *, scores=(1., 2., 3.)):
    result = []
    for step, score in enumerate(scores, 1):
        item = MoleculesObject(parent.get_atoms().copy(), f"unstable-physical-rank-{step}")
        item.update_metadata({**parent.get_metadata(), "parent_molecule_id": parent.get_moleculeid(),
                              "ranking_score": score, "step": step})
        result.append(item)
    return result


def ids(values):
    return [value.get_metadata()["acquisition_trajectory_id"] for value in values]


def test_ordinary_status_reconciles_committed_labels_and_reserved_allocators(tmp_path):
    import h5py
    from alframework.tools.acquisition_driver import CohortRuntime
    cfg = config()
    cfg["acquisition"]["selection"] = {"mode": "cohort_top_k", "cohort_size": 1, "k": 1}
    master = {"status_path": str(tmp_path / "status.json"),
              "h5_path": str(tmp_path / "data-{:04d}.h5"),
              "model_path": str(tmp_path / "model-{:04d}")}
    runtime = CohortRuntime(master, cfg)
    parent = molecule(0)
    parent._moleculeid = "mol-0003-0000000009"
    admitted = runtime.prepare(parent, {"generation": 3}, cfg)
    runtime.pool.accept_results(ids([admitted]), candidates(admitted))
    selected = runtime.pool.ready_qm()[0]
    runtime.pool.mark_qm_submitted(selected.get_moleculeid())
    runtime.pool.record_qm_result(selected)
    runtime.reserve_builder(3, 20)
    # The shard and training reservation committed; status/receipts did not.
    with h5py.File(master["h5_path"].format(0), "w") as handle:
        handle.create_group("H2").create_dataset("_id", data=[selected.get_moleculeid()],
                                                dtype=h5py.string_dtype())
    runtime.pool.close()
    runtime = CohortRuntime(master, cfg)
    try:
        status = {"current_model_id": 3, "current_training_id": 4,
                  "current_h5_id": 0, "current_molecule_id": 0,
                  "training_handoffs": {"4": {}}}
        runtime.reconcile_status(master, status)
        assert status["current_h5_id"] == 1
        assert status["current_training_id"] == 5
        assert status["current_molecule_id"] == 20
        assert runtime.pool.ready_qm_results() == []
        assert runtime.pool.status()["candidates"]["stored"] == 1
        assert runtime.prepare(parent, {"generation": 3}, cfg) is None
    finally:
        runtime.pool.close()


@pytest.mark.parametrize("width", [1, 2, 8, 32, 50])
def test_global_selection_invariant_to_physical_width_and_arrival_order(tmp_path, width):
    size, limit = 103, 13
    expected = []
    with AcquisitionCohorts(tmp_path / "journal.sqlite", cohort_size=size, k=limit, namespace="same-campaign") as pool:
        inputs = [pool.admit(molecule(i), {"generation": 3, "artifact": "/opaque/checkpoint"}, config()) for i in range(size)]
        all_candidates = []
        for index, parent in enumerate(inputs):
            values = [] if index % 7 == 0 else candidates(parent, scores=[index % 13, index % 13 + .25, index % 13 + .5])
            all_candidates.append(values)
            expected.extend((item.get_metadata()["ranking_score"], index, item.get_metadata()["step"]) for item in values)
        groups = [(inputs[i:i+width], all_candidates[i:i+width]) for i in range(0, size, width)]
        for number, (members, outputs) in enumerate(reversed(groups)):
            pool.mark_submitted(ids(members))
            pool.accept_results(ids(members), list(reversed([item for group in outputs for item in group])))
            assert pool.status()["retained_candidates"] <= limit
            if number != len(groups)-1:
                assert pool.ready_qm() == []
        chosen = pool.ready_qm()
        actual = [(row.get_metadata()["ranking_score"], row.get_metadata()["acquisition_trajectory_ordinal"],
                   row.get_metadata()["step"]) for row in chosen]
        reference = sorted(expected, key=lambda value: (-value[0], value[1], value[2]))[:limit]
        assert actual == reference
        assert len({row.get_moleculeid() for row in chosen}) == limit
        assert all(len(row.get_moleculeid()) == 32 for row in chosen)
        assert all(row.get_metadata()["_workflow_origin"] == row.get_moleculeid() for row in chosen)
        assert pool.status()["trajectories"] == {"completed": size}
        assert pool.status()["eligible_candidates"] == len(expected)
        # A delivery replay must neither alter the choice nor count arrivals twice.
        before = pool.status()
        pool.accept_results(ids(inputs), [item for group in all_candidates for item in group])
        assert pool.status() == before
        assert [row.get_moleculeid() for row in pool.ready_qm()] == [row.get_moleculeid() for row in chosen]


def test_exact_frame_ids_do_not_depend_on_physical_grouping(tmp_path):
    selections = []
    for width in (1, 2, 8):
        with AcquisitionCohorts(tmp_path / f"width-{width}.sqlite", cohort_size=8, k=5, namespace="one-science-run") as pool:
            inputs = [pool.admit(molecule(i), {"generation": 0}, config()) for i in range(8)]
            for start in range(0, 8, width):
                group = inputs[start:start+width]
                pool.accept_results(ids(group), [item for parent in reversed(group) for item in candidates(parent)])
            selections.append([item.get_moleculeid() for item in pool.ready_qm()])
    assert selections[0] == selections[1] == selections[2]


def test_large_cohort_executes_incrementally_without_waiting_for_all_admissions(tmp_path):
    with AcquisitionCohorts(tmp_path / "incremental.sqlite", cohort_size=50, k=4) as pool:
        for start in range(0, 50, 2):
            group = [pool.admit(molecule(i), {"generation": 1}, config()) for i in range(start, start+2)]
            pool.mark_submitted(ids(group))
            pool.accept_results(ids(group), [item for parent in group for item in candidates(parent)])
            assert pool.pending_trajectories() == []
            if start < 48:
                assert pool.ready_qm() == []
                assert pool.has_open_work()
        assert len(pool.ready_qm()) == 4


def test_no_per_trajectory_quota_and_fewer_than_k_candidates(tmp_path):
    with AcquisitionCohorts(tmp_path / "quota.sqlite", cohort_size=3, k=4) as pool:
        group = [pool.admit(molecule(i), {"generation": 0}, config()) for i in range(3)]
        pool.accept_results(ids(group[:2]), candidates(group[0], scores=[1., 2., 3.]))
        assert pool.ready_qm() == []
        pool.accept_results(ids(group[2:]), [])
        chosen = pool.ready_qm()
        assert len(chosen) == 3
        assert {m.get_metadata()["parent_molecule_id"] for m in chosen} == {"input-0"}
        assert [m.get_metadata()["ranking_score"] for m in chosen] == [3., 2., 1.]


def test_empty_full_cohort_selects_nothing_and_releases_all_inputs(tmp_path):
    with AcquisitionCohorts(tmp_path / "empty.sqlite", cohort_size=2, k=1) as pool:
        group = [pool.admit(molecule(i), {"generation": 0}, config()) for i in range(2)]
        pool.accept_results(ids(group), [])
        assert pool.ready_qm() == []
        assert not pool.has_open_work()
        assert pool.status()["cohorts"] == {"selected": 1}


def test_pinned_generation_state_cycle_and_execution_change_survive_restart(tmp_path):
    path = tmp_path / "pinned.sqlite"
    cfg = config()
    cfg.update(model_mode="excited_state", state_selection={"mode": "batch_cycle", "states": [0, 2]})
    with AcquisitionCohorts(path, cohort_size=3, k=2) as pool:
        first = pool.admit(molecule(0), {"generation": 0}, cfg)
        second = pool.admit(molecule(1), {"generation": 1}, cfg)
        assert first.get_metadata()["selected_state"] == second.get_metadata()["selected_state"] == 0
        assert second.get_metadata()["acquisition_model_reference"] == {"generation": 0}
        pool.accept_results(ids([first]), candidates(first))
        assert pool.ready_qm() == []
        old_id = first.get_metadata()["acquisition_cohort_id"]
    cfg["driver"] = {"strategy": "single", "engine": "ase"}
    with AcquisitionCohorts(path, cohort_size=3, k=2) as pool:
        third = pool.admit(molecule(2), {"generation": 2}, cfg)
        assert third.get_metadata()["acquisition_cohort_id"] == old_id
        assert third.get_metadata()["acquisition_model_reference"] == {"generation": 0}
        pool.accept_results(ids([second, third]), candidates(second) + candidates(third))
        fourth = pool.admit(molecule(3), {"generation": 2}, cfg)
        assert fourth.get_metadata()["selected_state"] == 2
        assert fourth.get_metadata()["acquisition_model_reference"] == {"generation": 2}
        assert pool.cohort_for_trajectory(ids([fourth])[0])["required_trajectory_count"] == 3
        assert pool.cohort_for_trajectory(ids([third])[0])["sampler_config"]["driver"]["strategy"] == "batched"


def test_incomplete_cohort_waits_across_restarts_with_no_admission_timeout(tmp_path):
    path = tmp_path / "incomplete.sqlite"
    with AcquisitionCohorts(path, cohort_size=3, k=1) as pool:
        first = pool.admit(molecule(0), {"generation": 0}, config())
        pool.accept_results(ids([first]), candidates(first))
    with AcquisitionCohorts(path, cohort_size=3, k=1) as pool:
        assert pool.ready_qm() == []
        assert pool.has_open_work()
        assert pool.status()["cohorts"] == {"open": 1}
        assert pool.admit(molecule(0), {"generation": 9}, config()).get_metadata() == first.get_metadata()


def test_failure_is_retryable_with_same_input_seed_and_no_premature_selection(tmp_path):
    path = tmp_path / "failure.sqlite"
    with AcquisitionCohorts(path, cohort_size=2, k=2) as pool:
        inputs = [pool.admit(molecule(i), {"generation": 0}, config()) for i in range(2)]
        pool.mark_submitted(ids(inputs))
        pool.accept_results(ids(inputs[:1]), candidates(inputs[0]))
        pool.mark_failed(ids(inputs[1:]), "worker crashed")
        assert pool.ready_qm() == []
    with AcquisitionCohorts(path, cohort_size=2, k=2) as pool:
        retry = pool.retryable_trajectories()
        assert ids(retry) == ids(inputs[1:])
        assert retry[0].get_metadata()["trajectory_seed"] == inputs[1].get_metadata()["trajectory_seed"]
        pool.mark_submitted(ids(retry))
        pool.accept_results(ids(retry), candidates(retry[0]))
        assert len(pool.ready_qm()) == 2
        assert pool.trajectory(ids(retry)[0])["attempts"] == 2


@pytest.mark.parametrize("boundary", ["completion", "selection"])
def test_ingestion_transaction_rolls_back_before_delivery_acknowledgement(tmp_path, boundary):
    with AcquisitionCohorts(tmp_path / f"atomic-{boundary}.sqlite", cohort_size=1, k=2) as pool:
        parent = pool.admit(molecule(0), {"generation": 0}, config())
        if boundary == "completion":
            sql = "CREATE TRIGGER injected BEFORE UPDATE OF state ON trajectories WHEN NEW.state='completed' BEGIN SELECT RAISE(ABORT,'injected'); END"
        else:
            sql = "CREATE TRIGGER injected BEFORE UPDATE OF state ON candidates WHEN NEW.state='pending' BEGIN SELECT RAISE(ABORT,'injected'); END"
        pool.db.execute(sql)
        pool.db.commit()
        with pytest.raises(sqlite3.IntegrityError, match="injected"):
            pool.accept_results(ids([parent]), candidates(parent))
        assert pool.status()["retained_candidates"] == 0
        assert pool.trajectory(ids([parent])[0])["state"] == "pending"
        assert pool.status()["cohorts"] == {"open": 1}
        pool.db.execute("DROP TRIGGER injected")
        pool.db.commit()
        pool.accept_results(ids([parent]), candidates(parent))
        assert len(pool.ready_qm()) == 2


def test_qm_handoff_result_and_storage_receipts_are_restart_safe(tmp_path):
    path = tmp_path / "handoff.sqlite"
    with AcquisitionCohorts(path, cohort_size=1, k=1) as pool:
        parent = pool.admit(molecule(0), {"generation": 0}, config())
        pool.accept_results(ids([parent]), candidates(parent))
        selected = pool.ready_qm()[0]
    with AcquisitionCohorts(path, cohort_size=1, k=1) as pool:
        assert pool.ready_qm()[0].get_moleculeid() == selected.get_moleculeid()
        pool.mark_qm_submitted(selected, {"task_key": "durable-QM-key"})
    with AcquisitionCohorts(path, cohort_size=1, k=1) as pool:
        assert pool.ready_qm() == []
        assert len(pool.ready_qm(include_submitted=True)) == 1
        selected.store_results({"energy": -1.})
        pool.record_qm_result(selected)
        pool.mark_qm_failed(selected, "late duplicate error")
    with AcquisitionCohorts(path, cohort_size=1, k=1) as pool:
        assert pool.ready_qm() == []
        assert pool.ready_qm_results()[0].get_results()["energy"] == -1.
        pool.mark_stored([selected.get_moleculeid()])
    with AcquisitionCohorts(path, cohort_size=1, k=1) as pool:
        assert not pool.has_open_work()
        assert pool.ready_qm_results() == []
        assert pool.ready_qm(include_submitted=True, include_failed=True) == []


def test_scientific_configuration_is_pinned_and_input_is_copied(tmp_path):
    cfg = config()
    original = molecule(0)
    with AcquisitionCohorts(tmp_path / "immutable.sqlite", cohort_size=2, k=1) as pool:
        admitted = pool.admit(original, {"generation": 0}, cfg)
        original.get_atoms().positions[:] = 99.
        cfg["dt"] = .5
        assert pool.trajectory(ids([admitted])[0])["input"].get_atoms().positions[0, 0] == 0.
        assert pool.cohort_for_trajectory(ids([admitted])[0])["sampler_config"]["dt"] == .1
        with pytest.raises(ValueError, match="Cannot change"):
            pool.admit(molecule(1), {"generation": 1}, cfg)


@pytest.mark.parametrize("fault", ["score", "step", "trajectory"])
def test_invalid_candidate_cannot_complete_or_partially_ingest_a_trajectory(tmp_path, fault):
    with AcquisitionCohorts(tmp_path / f"invalid-{fault}.sqlite", cohort_size=1, k=1) as pool:
        parent = pool.admit(molecule(0), {"generation": 0}, config())
        output = candidates(parent)
        if fault == "score":
            output[-1].update_metadata({"ranking_score": float("nan")})
        elif fault == "step":
            del output[-1].get_metadata()["step"]
        else:
            output[-1].update_metadata({"acquisition_trajectory_id": "not-a-member"})
        with pytest.raises(ValueError):
            pool.accept_results(ids([parent]), output)
        assert pool.status()["retained_candidates"] == 0
        assert pool.status()["trajectories"] == {"pending": 1}


def test_non_scalar_selector_cannot_use_scalar_top_k_retention(tmp_path):
    with pytest.raises(ValueError, match="different selector"):
        AcquisitionCohorts(tmp_path / "unsupported.sqlite", cohort_size=2, k=1, selection_id="structural_diversity")


def test_only_one_controller_can_own_a_cohort_journal(tmp_path):
    path = tmp_path / "locked.sqlite"
    with AcquisitionCohorts(path, cohort_size=1, k=1):
        with pytest.raises(RuntimeError, match="Another controller"):
            AcquisitionCohorts(path, cohort_size=1, k=1)


def test_proposal_names_can_repeat_at_distinct_iteration_steps(tmp_path):
    with AcquisitionCohorts(tmp_path / "proposals.sqlite", cohort_size=1, k=2) as pool:
        parent = pool.admit(molecule(0), {"generation": 0}, config())
        output = candidates(parent, scores=[1., 2.])
        for item in output:
            item.update_metadata({"proposal_id": "line-search-trial"})
        pool.accept_results(ids([parent]), output)
        selected = pool.ready_qm()
        assert len(selected) == len({item.get_moleculeid() for item in selected}) == 2
        assert [item.get_metadata()["step"] for item in selected] == [2, 1]


def test_interleaved_explicit_states_fill_separate_compatible_cohorts(tmp_path):
    cfg = config()
    cfg.update(model_mode="excited_state", state_selection={"mode": "fixed", "state": 0})
    with AcquisitionCohorts(tmp_path / "interleaved.sqlite", cohort_size=2, k=1) as pool:
        inputs = []
        for index, state in enumerate((0, 2, 2, 0)):
            item = molecule(index)
            item.update_metadata({"selected_state": state})
            inputs.append(pool.admit(item, {"generation": index // 2}, cfg, selected_state=state))
        assert inputs[0].get_metadata()["cohort_id"] == inputs[3].get_metadata()["cohort_id"]
        assert inputs[1].get_metadata()["cohort_id"] == inputs[2].get_metadata()["cohort_id"]
        assert inputs[0].get_metadata()["cohort_id"] != inputs[1].get_metadata()["cohort_id"]
        assert [item.get_metadata()["selected_state"] for item in inputs] == [0, 2, 2, 0]
        assert inputs[2].get_metadata()["acquisition_model_reference"] == {"generation": 0}
        pool.accept_results(ids(inputs), [item for parent in inputs for item in candidates(parent)])
        assert len(pool.ready_qm()) == 2
        assert pool.status()["cohorts"] == {"selected": 2}
