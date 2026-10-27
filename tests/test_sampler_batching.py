import numpy as np
import pytest


from ase import Atoms
from ase.io import write

from alframework.builders.builders import simple_cfg_loader_task
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampler_batching import (
    SamplerBatchBuffer,
    changed_batching_signature_fields,
    configured_sampler_calculator_status,
    flatten_molecule_output,
    sampler_batch_key,
    sampler_batch_size,
    sampler_batching_signature,
    sampler_capacity_status,
    sampler_has_builder_capacity,
    sampler_submission_groups,
    sampler_task_feed,
    sampler_task_replicas,
    sampler_ready_groups,
    sampler_loop_delay,
    selected_state,
    validate_batching_config_reload,
    validate_state_selection,
)
from alframework.tools.batched_config import (
    batching_settings, dynamics_settings, configured_native_loader, configured_native_options,
)
from alframework.tools.tools import parsl_task_queue
from tests.helpers.fakes import FakeTask


def _molecule(molecule_id, symbols="H2", state=None):
    atoms = Atoms(symbols, positions=np.zeros((len(symbols), 3)))
    molecule = MoleculesObject(atoms, molecule_id)
    if state is not None:
        molecule.update_metadata({"selected_state": state})
    return molecule


def test_strict_full_buffer_groups_exact_atom_order():
    config = {"model_mode": "ground_state", "alchemi_baoab": {"batch_size": 2}}
    buffer = SamplerBatchBuffer()
    first = _molecule("first", "OH")
    second = _molecule("second", "OH")
    different_order = _molecule("different", "HO")

    buffer.add(first, config, buffered_at=10.0)
    buffer.add(different_order, config, buffered_at=20.0)
    assert buffer.pop_ready(2) == []
    assert len(buffer) == 2

    buffer.add(second, config, buffered_at=30.0)
    batches = buffer.pop_ready(2)
    assert [[item.get_moleculeid() for item in batch] for batch in batches] == [
        ["first", "second"]
    ]
    assert [item.get_moleculeid() for item in buffer.molecules()] == ["different"]


def test_excited_state_is_part_of_batch_key():
    config = {"model_mode": "excited_state", "alchemi_baoab": {"batch_size": 2}}
    state_zero = _molecule("zero", state=0)
    state_one = _molecule("one", state=1)

    assert sampler_batch_key(state_zero, config) != sampler_batch_key(state_one, config)


def test_batch_cycle_assigns_complete_state_blocks_and_repeats():
    config = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "batch_cycle", "states": [0, 1]},
        "alchemi_baoab": {"batch_size": 2},
    }
    buffer = SamplerBatchBuffer()

    for index in range(8):
        buffer.add(_molecule(f"mol-0003-{index:010d}"), config)

    batches = buffer.pop_ready(2)
    assert [
        [item.get_metadata()["selected_state"] for item in batch]
        for batch in batches
    ] == [[0, 0], [0, 0], [1, 1], [1, 1]]
    assigned_by_id = {
        item.get_moleculeid(): item.get_metadata()["selected_state"]
        for batch in batches
        for item in batch
    }
    assert [assigned_by_id[f"mol-0003-{index:010d}"] for index in range(8)] == [
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
    ]


@pytest.mark.parametrize(
    "sequence_index,expected_state",
    [
        (0, 0),
        (49, 0),
        (50, 1),
        (199, 3),
        (200, 0),
    ],
)
def test_batch_cycle_matches_documented_fifty_replica_boundaries(
    sequence_index, expected_state
):
    config = {
        "model_mode": "excited_state",
        "state_selection": {
            "mode": "batch_cycle",
            "states": [0, 1, 2, 3],
        },
        "alchemi_baoab": {"batch_size": 50},
    }
    molecule = _molecule(f"mol-0000-{sequence_index:010d}")

    assert selected_state(molecule, config) == expected_state


def test_state_metadata_overrides_cycle_and_legacy_key_is_normalized():
    config = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "batch_cycle", "states": [0, 1]},
        "alchemi_baoab": {"batch_size": 2},
    }
    explicit = _molecule("mol-0000000000", state=3)
    legacy = _molecule("mol-0000000000")
    legacy.update_metadata({"excited_state": 2})

    assert selected_state(explicit, config) == 3
    assert explicit.get_metadata()["selected_state_source"] == "metadata.selected_state"
    assert selected_state(legacy, config) == 2
    assert legacy.get_metadata()["selected_state"] == 2
    assert legacy.get_metadata()["selected_state_source"] == "metadata.excited_state"


def test_fixed_state_and_non_numeric_ids_are_deterministic():
    fixed_config = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "fixed", "state": 2},
        "alchemi_baoab": {"batch_size": 4},
    }
    assert selected_state(_molecule("without-digits"), fixed_config) == 2

    cycle_config = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "batch_cycle", "states": [0, 1, 2]},
        "alchemi_baoab": {"batch_size": 1},
    }
    first = selected_state(_molecule("same-nonnumeric-id"), cycle_config)
    second = selected_state(_molecule("same-nonnumeric-id"), cycle_config)
    assert first == second


@pytest.mark.parametrize(
    "selection,error",
    [
        ({"mode": "batch_cycle", "states": []}, "non-empty"),
        ({"mode": "batch_cycle", "states": [0, 0]}, "duplicate"),
        ({"mode": "batch_cycle", "states": [0, -1]}, "nonnegative"),
        ({"mode": "fixed"}, "requires"),
        ({"mode": "random", "states": [0, 1]}, "fixed.*batch_cycle"),
    ],
)
def test_invalid_state_selection_is_rejected(selection, error):
    config = {
        "model_mode": "excited_state",
        "state_selection": selection,
        "alchemi_baoab": {"batch_size": 2},
    }
    with pytest.raises(ValueError, match=error):
        validate_state_selection(config)


def test_configured_states_are_validated_against_properties():
    config = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "batch_cycle", "states": [0, 2]},
        "alchemi_baoab": {"batch_size": 2},
    }
    properties = {
        "sE0": ["state_0_energy", "system", 1.0],
        "F0": ["state_0_forces", "atomic", 1.0],
        "sE1": ["state_1_energy", "system", 1.0],
        "F1": ["state_1_forces", "atomic", 1.0],
    }
    with pytest.raises(ValueError, match=r"states \[2\].*available states.*\[0, 1\]"):
        validate_state_selection(config, properties)


def test_cfg_loader_hands_structures_to_state_specific_batches(tmp_path):
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
    config = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "batch_cycle", "states": [0, 1]},
        "alchemi_baoab": {"batch_size": 2},
    }
    buffer = SamplerBatchBuffer()

    for index in range(4):
        molecule = simple_cfg_loader_task.func(
            moleculeid=f"mol-0000-{index:010d}",
            builder_config={"molecule_library_dir": str(library)},
            shake=0.0,
        )
        buffer.add(molecule, config)

    batches = buffer.pop_ready(2)
    assert len(batches) == 2
    assert [batch[0].get_metadata()["selected_state"] for batch in batches] == [0, 1]
    assert all(
        item.get_metadata()["selected_state_source"]
        == "state_selection.batch_cycle"
        for batch in batches
        for item in batch
    )


def test_incomplete_batches_remain_visible_without_being_dropped():
    config = {"model_mode": "ground_state", "alchemi_baoab": {"batch_size": 4}}
    buffer = SamplerBatchBuffer()
    buffer.add(_molecule("waiting"), config, buffered_at=100.0)

    assert buffer.pop_ready(4) == []
    summary = buffer.status(now=160.0)
    assert summary["buffered_structures"] == 1
    assert summary["oldest_age_seconds"] == 60.0
    assert summary["buckets"][0]["count"] == 1


def test_buffer_status_reports_cycle_assigned_state():
    config = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "batch_cycle", "states": [0, 1]},
        "alchemi_baoab": {"batch_size": 2},
    }
    buffer = SamplerBatchBuffer()
    buffer.add(_molecule("mol-0000000002"), config, buffered_at=100.0)

    summary = buffer.status(now=110.0)
    assert summary["buckets"][0]["selected_state"] == 1
    assert next(buffer.molecules()).get_metadata()["selected_state"] == 1


def test_flatten_molecule_output_handles_single_nested_and_empty():
    first = _molecule("first")
    second = _molecule("second")

    assert flatten_molecule_output(first) == [first]
    assert flatten_molecule_output([first, (second,)]) == [first, second]
    assert flatten_molecule_output(None) == []
    with pytest.raises(TypeError, match="Sampler output"):
        flatten_molecule_output("not-a-molecule")


def test_batch_configuration_accepts_only_full_only_policy():
    assert sampler_batch_size({"alchemi_baoab": {"batch_size": 3}}) == 3
    with pytest.raises(ValueError, match="full_only"):
        sampler_batch_size(
            {"alchemi_baoab": {"batch_size": 3, "partial_policy": "timeout"}}
        )


def test_replica_capacity_counts_tasks_builders_and_incomplete_buffer():
    status = sampler_capacity_status(
        parallel_sampler_limit=20,
        submitted_sampler_replicas=8,
        builder_task_count=1,
        maximum_builder_structures=3,
        buffered_structure_count=2,
    )

    assert status == {
        "parallel_sampler_limit": 20,
        "submitted_sampler_replicas": 8,
        "pending_builder_replicas": 3,
        "buffered_replicas": 2,
        "accounted_replicas": 13,
        "available_replica_slots": 7,
    }


def test_replica_capacity_preserves_legacy_task_units():
    status = sampler_capacity_status(
        parallel_sampler_limit=10,
        submitted_sampler_replicas=3,
        builder_task_count=2,
        maximum_builder_structures=1,
        buffered_structure_count=0,
    )

    assert status["submitted_sampler_replicas"] == 3
    assert status["pending_builder_replicas"] == 2
    assert status["accounted_replicas"] == 5
    tasks = [
        FakeTask("exec_done", result=[], done=True),
        FakeTask("running", done=False),
    ]
    replica_counts = {id(tasks[0]): 1, id(tasks[1]): 1}
    assert sampler_task_replicas(tasks, replica_counts) == 2
    assert sampler_task_replicas(
        tasks, replica_counts, completed_only=True
    ) == 1


def test_completed_batches_count_input_replicas_even_without_candidates():
    queue = parsl_task_queue()
    queue.add_task(FakeTask("exec_done", result=[], done=True))
    queue.add_task(FakeTask("failed", done=True))

    assert queue.get_completed_number() == 2
    replica_counts = {id(task): 4 for task in queue.task_list}
    assert sampler_task_replicas(
        queue.task_list,
        replica_counts,
        completed_only=True,
    ) == 8
    results, failed = queue.get_task_results()
    assert results == [[]]
    assert failed == 1
    assert queue.get_number() == 0


def test_buffered_replicas_close_parallel_sampler_capacity():
    config = {"alchemi_baoab": {"batch_size": 4}}
    buffer = SamplerBatchBuffer()
    for index in range(3):
        buffer.add(_molecule(f"buffered-{index}"), config)

    status = sampler_capacity_status(
        parallel_sampler_limit=4,
        submitted_sampler_replicas=0,
        builder_task_count=1,
        maximum_builder_structures=1,
        buffered_structure_count=len(buffer),
    )

    assert status["accounted_replicas"] == 4
    assert status["available_replica_slots"] == 0
    assert not sampler_has_builder_capacity(status, 1)


def test_sampler_submission_groups_and_feeds_preserve_legacy_behavior():
    molecule = _molecule("legacy")
    buffer = SamplerBatchBuffer()

    groups = sampler_submission_groups(molecule, {}, buffer)

    assert groups == [[molecule]]
    assert len(buffer) == 0
    assert sampler_task_feed(groups[0], {}) == {
        "molecule_object": molecule,
        "sampler_config": {},
    }


def test_sampler_submission_groups_emit_only_complete_batches():
    config = {"alchemi_baoab": {"batch_size": 2}}
    buffer = SamplerBatchBuffer()
    first = _molecule("first")
    second = _molecule("second")

    assert sampler_submission_groups(first, config, buffer) == []
    assert len(buffer) == 1
    groups = sampler_submission_groups(second, config, buffer)

    assert groups == [[first, second]]
    assert len(buffer) == 0
    assert sampler_task_feed(groups[0], config) == {
        "molecule_objects": [first, second],
        "sampler_config": config,
    }


def test_batching_signature_normalizes_state_selection_and_calculator():
    master = {"sampler_task": "package.alchemi_task"}
    config = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "fixed", "selected_state": 1},
        "alchemi_calculator": "package.native_loader",
        "alchemi_baoab": {
            "batch_size": 4,
            "partial_policy": "full_only",
        },
    }

    signature = sampler_batching_signature(master, config)

    assert signature == {
        "sampler_task": "package.alchemi_task",
        "batched": True,
        "batch_size": 4,
        "partial_policy": "full_only",
        "model_mode": "excited_state",
        "state_selection": {"mode": "fixed", "states": [1]},
        "portable": False,
        "backend": "alchemi",
        "calculator_loader": "package.native_loader",
        "max_wait_seconds": None,
    }
    assert configured_sampler_calculator_status(config) == {
        "interface": "native",
        "loader": "package.native_loader",
    }
    assert configured_sampler_calculator_status({}) is None


def test_noncritical_reload_is_accepted_without_reassigning_buffered_state():
    master = {"sampler_task": "package.alchemi_task"}
    current = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "fixed", "state": 1},
        "Escut": 1.0,
        "alchemi_baoab": {"batch_size": 2},
    }
    proposed = {
        **current,
        "Escut": 2.0,
        "gap_diagnostics": {"enabled": True},
    }
    buffer = SamplerBatchBuffer()
    molecule = _molecule("waiting")
    buffer.add(molecule, current)

    assert validate_batching_config_reload(
        master,
        current,
        master,
        proposed,
        buffer,
    ) == []
    assert next(buffer.molecules()) is molecule
    assert molecule.get_metadata()["selected_state"] == 1


@pytest.mark.parametrize(
    "master_update,config_update,changed_field",
    [
        ({"sampler_task": "package.other_task"}, {}, "sampler_task"),
        ({}, {"alchemi_baoab": None}, "batched"),
        ({}, {"alchemi_baoab": {"batch_size": 3}}, "batch_size"),
        ({}, {"model_mode": "ground_state"}, "model_mode"),
        (
            {},
            {"state_selection": {"mode": "fixed", "state": 0}},
            "state_selection",
        ),
    ],
)
def test_critical_reload_is_rejected_during_an_active_batching_epoch(
    master_update,
    config_update,
    changed_field,
):
    master = {"sampler_task": "package.alchemi_task"}
    current = {
        "model_mode": "excited_state",
        "state_selection": {"mode": "fixed", "state": 1},
        "alchemi_baoab": {"batch_size": 2},
    }
    proposed_master = {**master, **master_update}
    proposed_config = {**current, **config_update}
    buffer = SamplerBatchBuffer()
    waiting = _molecule("waiting")
    buffer.add(waiting, current)

    assert changed_field in changed_batching_signature_fields(
        master,
        current,
        proposed_master,
        proposed_config,
    )
    with pytest.raises(ValueError, match=changed_field):
        validate_batching_config_reload(
            master,
            current,
            proposed_master,
            proposed_config,
            buffer,
        )
    assert list(buffer.molecules()) == [waiting]


def test_critical_reload_is_accepted_after_buffer_drains():
    master = {"sampler_task": "package.alchemi_task"}
    current = {"alchemi_baoab": {"batch_size": 2}}
    proposed = {"alchemi_baoab": {"batch_size": 4}}
    buffer = SamplerBatchBuffer()

    assert validate_batching_config_reload(
        master,
        current,
        master,
        proposed,
        buffer,
    ) == ["batch_size"]


def test_mixed_batch_width_tasks_keep_their_submitted_replica_counts():
    old_batch = FakeTask("exec_done", result=[], done=True)
    new_legacy = FakeTask("exec_done", result=[], done=True)
    tasks = [old_batch, new_legacy]
    replica_counts = {id(old_batch): 4, id(new_legacy): 1}

    assert sampler_task_replicas(tasks, replica_counts) == 5
    assert sampler_task_replicas(
        tasks,
        replica_counts,
        completed_only=True,
    ) == 5


def test_builder_submission_requires_capacity_for_its_complete_width():
    status = sampler_capacity_status(
        parallel_sampler_limit=5,
        submitted_sampler_replicas=4,
        builder_task_count=0,
        maximum_builder_structures=2,
        buffered_structure_count=0,
    )

    assert status["available_replica_slots"] == 1
    assert not sampler_has_builder_capacity(status, 2)
    assert sampler_has_builder_capacity(status, 1)


@pytest.mark.parametrize(
    "current,proposed",
    [
        (
            {"alchemi_baoab": {"batch_size": 2}},
            {},
        ),
        (
            {},
            {"alchemi_baoab": {"batch_size": 2}},
        ),
        (
            {
                "model_mode": "excited_state",
                "state_selection": {"mode": "fixed", "state": 0},
                "alchemi_baoab": {"batch_size": 2},
            },
            {
                "model_mode": "excited_state",
                "state_selection": {
                    "mode": "batch_cycle",
                    "states": [0, 1],
                },
                "alchemi_baoab": {"batch_size": 2},
            },
        ),
    ],
)
def test_batching_mode_and_state_changes_are_accepted_when_epoch_is_empty(
    current,
    proposed,
):
    master = {"sampler_task": "package.sampler_task"}

    changed = validate_batching_config_reload(
        master,
        current,
        master,
        proposed,
        SamplerBatchBuffer(),
    )

    assert changed


def _portable_config(**settings):
    return {"model_mode": "ground_state", "dynamics": {"backend": "torchsim"},
            "batching": {"batch_size": 3, **settings}}


def test_portable_configuration_defaults_and_legacy_conflicts():
    config = {"dynamics": {"backend": "torchsim"}, "batching": {}}
    assert batching_settings(config) == {"batch_size": 1, "partial_policy": "timeout",
                                         "max_wait_seconds": 5., "portable": True}
    assert dynamics_settings(config) == {"backend": "torchsim"}
    assert dynamics_settings({"alchemi_baoab": {"random_seed": 8}}) == {"random_seed": 8, "backend": "alchemi"}
    with pytest.raises(ValueError, match="dynamics.backend"):
        dynamics_settings({"batching": {}})
    with pytest.raises(ValueError, match="Conflicting.*batch_size"):
        batching_settings({**config, "batching": {"batch_size": 2}, "alchemi_baoab": {"batch_size": 3}})
    with pytest.raises(ValueError, match="Conflicting.*friction"):
        dynamics_settings({"dynamics": {"backend": "alchemi", "friction_per_fs": 1.},
                           "alchemi_baoab": {"friction_per_fs": 2.}})
    with pytest.raises(ValueError, match="Conflicting.*loaders"):
        configured_native_loader({"batched_calculator": "new.load", "alchemi_calculator": "old.load"})
    with pytest.raises(ValueError, match="Conflicting.*options"):
        configured_native_options({"batched_calculator_options": {"key": 1},
                                   "alchemi_calculator_options": {"key": 2}})
    assert configured_native_options({"batched_calculator_options": {"new": 1},
                                      "alchemi_calculator_options": {"old": 2}}) == {"new": 1, "old": 2}


@pytest.mark.parametrize('settings', [
    {"batch_size": True}, {"batch_size": 1.5}, {"batch_size": 0},
    {"max_wait_seconds": float('nan')}, {"max_wait_seconds": -1},
    {"max_wait_seconds": True}, {"partial_policy": "unsupported"},
])
def test_portable_batching_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        batching_settings(_portable_config(**settings))


def test_portable_batch_packs_sizes_orders_and_compositions_but_separates_pbc():
    config = _portable_config()
    buffer = SamplerBatchBuffer()
    first, second, third = (_molecule("first", "OH"), _molecule("second", "HO"), _molecule("third", "H"))
    periodic = _molecule("periodic", "H")
    periodic.get_atoms().set_cell(np.eye(3) * 5.)
    periodic.get_atoms().set_pbc(True)
    for molecule in (first, periodic, second, third):
        buffer.add(molecule, config)
    assert sampler_ready_groups(buffer, config) == [[first, second, third]]
    summary = buffer.status()
    assert summary['buffered_structures'] == 1
    assert summary['buckets'][0]['pbc'] == [True] * 3
    assert summary['buckets'][0]['atom_counts'] == [1]
    np.testing.assert_array_equal(second.get_atoms().numbers, [1, 8])


def test_partial_timeout_is_polled_without_new_builder_results_and_caps_sleep():
    config = _portable_config()
    buffer = SamplerBatchBuffer()
    waiting = _molecule("waiting")
    buffer.add(waiting, config, buffered_at=10.)
    assert sampler_ready_groups(buffer, config, now=14.9) == []
    assert sampler_loop_delay(config, 60., buffer, now=13.) == pytest.approx(2.)
    assert sampler_ready_groups(buffer, config, now=15.) == [[waiting]]
    assert sampler_loop_delay(config, 60., buffer, now=15.) == 60.
    assert sampler_task_feed([waiting], config)['molecule_objects'] == [waiting]
    with pytest.raises(ValueError, match="between one"):
        sampler_task_feed([], config)


def test_complete_batch_resets_remaining_members_timeout():
    config = _portable_config(batch_size=2)
    buffer = SamplerBatchBuffer()
    molecules = [_molecule(str(i)) for i in range(3)]
    for molecule, timestamp in zip(molecules, (1., 2., 9.)):
        buffer.add(molecule, config, buffered_at=timestamp)
    assert sampler_ready_groups(buffer, config, now=10.) == [molecules[:2]]
    assert sampler_ready_groups(buffer, config, now=14.) == [molecules[2:]]


@pytest.mark.parametrize('field,update', [
    ('backend', {"dynamics": {"backend": "alchemi"}}),
    ('calculator_loader', {"batched_calculator": "other.loader"}),
    ('max_wait_seconds', {"batching": {"batch_size": 3, "max_wait_seconds": 10.}}),
])
def test_portable_backend_loader_and_deadline_reload_preserve_waiting_input(field, update):
    current = _portable_config()
    proposed = {**current, **update}
    buffer = SamplerBatchBuffer()
    waiting = _molecule('waiting')
    buffer.add(waiting, current)
    with pytest.raises(ValueError, match=field):
        validate_batching_config_reload({}, current, {}, proposed, buffer)
    assert list(buffer.molecules()) == [waiting]


@pytest.mark.parametrize('portable', [False, True], ids=['legacy-buffer', 'portable-buffer'])
def test_pickled_buffers_keep_members_order_and_original_deadlines(portable):
    import pickle
    config = _portable_config() if portable else {'alchemi_baoab': {'batch_size': 3}}
    buffer = SamplerBatchBuffer()
    buffer.add(_molecule('saved'), config, buffered_at=10.)
    restored = pickle.loads(pickle.dumps(buffer))
    assert restored.status(now=20.) == buffer.status(now=20.)
    assert [m.get_moleculeid() for m in restored.molecules()] == ['saved']
    assert len(sampler_ready_groups(restored, config, now=20.)) == int(portable)


def test_null_native_loader_options_retain_legacy_compatibility():
    from alframework.tools.batched_config import configured_native_loader, configured_native_options
    config = {"alchemi_calculator": "example.loader", "alchemi_calculator_options": None}
    assert configured_native_loader(config) == "example.loader"
    assert configured_native_options(config) == {}
    config.update(batched_calculator="example.loader", batched_calculator_options={"device_option": 2})
    assert configured_native_options(config) == {"device_option": 2}
