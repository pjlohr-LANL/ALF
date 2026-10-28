"""Coordinator integration for logical acquisition, without optional engines."""
from __future__ import annotations

import atexit
from copy import deepcopy, copy
import hashlib
import json
from pathlib import Path
import pickle

from alframework.tools.batched_config import inference_settings, normalize_sampling_config
from alframework.tools.sampler_batching import flatten_molecule_output
from alframework.tools.tools import is_final_task


def current_model_reference(master, status, config, ML_config=None):
    """Interpret upstream's generation/path convention, not its artifact layout."""
    inference = inference_settings(config)
    previous = status.get("current_model_reference")
    if previous is not None and int(previous["generation"]) == int(status["current_model_id"]):
        reference = deepcopy(previous)
        reference.update(adapter=inference["loader"], interface=inference["interface"], options=inference["options"])
        return reference
    return {"artifact_location": master["model_path"].format(int(status["current_model_id"])),
            "generation": int(status["current_model_id"]), "adapter": inference["loader"],
            "interface": inference["interface"], "options": inference["options"], "capabilities": {},
            "metadata": {"ML_config": deepcopy(ML_config or {}),
                         "properties_list": deepcopy(master["properties_list"])}}


def first_trigger_input(molecule, reference, config):
    """Assign a driver-independent trajectory identity without a selection pool."""
    original = molecule
    molecule = copy(original)
    molecule.atoms = original.get_atoms().copy()
    molecule.metadata = deepcopy(original.get_metadata())
    molecule.qm_results = deepcopy(original.get_results())
    metadata = molecule.get_metadata()
    if "acquisition_trajectory_id" not in metadata:
        origin = metadata.get("_workflow_origin", molecule.get_moleculeid())
        identity = json.dumps([reference, origin], sort_keys=True, default=str)
        identifier = "t" + hashlib.sha256(identity.encode()).hexdigest()[:31]
        from alframework.tools.sampler_batching import _molecule_sequence_index
        ordinal = _molecule_sequence_index(molecule)
        seed = dict(config.get("driver") or {}).get("random_seed", dict(config.get("dynamics") or {}).get("random_seed", 42))
        seed = int.from_bytes(hashlib.sha256(f"{seed}:{identifier}".encode()).digest()[:4], "big")
        molecule.update_metadata({"acquisition_trajectory_id": identifier,
            "acquisition_trajectory_ordinal": ordinal, "trajectory_seed": seed,
            "acquisition_model_reference": deepcopy(reference)})
    return molecule


class CohortRuntime:
    """Small bridge from Parsl task queues to the durable selection journal."""

    def __init__(self, master, config):
        from alframework.tools.acquisition_cohorts import AcquisitionCohorts
        selection = config["acquisition"]["selection"]
        path = selection.get("journal_path")
        if path is None:
            path = Path(master["status_path"]).with_suffix(".cohorts.sqlite")
        path = Path(path)
        if not path.is_absolute():
            path = Path(master.get("master_directory") or ".") / path
        self.pool = AcquisitionCohorts(path, cohort_size=selection["cohort_size"], k=selection["k"])
        self.sampler_inputs, self.qm_inputs = {}, {}
        atexit.register(self.pool.close)
        self.reconcile_storage(master["h5_path"])

    def _inputs(self, future, mapping):
        if id(future) in mapping:
            return mapping[id(future)]
        if hasattr(future, "control") and hasattr(future, "key"):
            row = future.control.task(future.key)
            arguments = pickle.loads(row["input"])
            return arguments.get("molecule_objects", [arguments.get("molecule_object")])
        return []

    def prepare(self, molecule, reference, config):
        if molecule.get_metadata().get("acquisition_trajectory_id"):
            return molecule
        metadata = molecule.get_metadata()
        selected = metadata.get("selected_state", metadata.get("excited_state"))
        admitted = self.pool.admit(molecule, reference, config, selected_state=selected)
        identifier = admitted.get_metadata()["acquisition_trajectory_id"]
        if self.pool.trajectory(identifier)["state"] in {"completed", "abandoned"}:
            return None
        return admitted

    def reconcile_status(self, master, status):
        """Recover ordinary-run allocators from durable records and artifacts."""
        from alframework.tools.tools import find_empty_directory
        from alframework.tools.workflow_io import shards
        paths = shards(master["h5_path"])
        status["current_h5_id"] = max(status.get("current_h5_id", 0),
                                      max((index + 1 for index, _ in paths), default=0))
        training_ids = [int(value) + 1 for value in status.get("training_handoffs", {})]
        status["current_training_id"] = max(status.get("current_training_id", 0),
                                           find_empty_directory(master["model_path"]), *training_ids)
        generation = status["current_model_id"]
        prefix = "mol-boot-" if generation < 0 else f"mol-{generation:04d}-"
        indices = [status.get("current_molecule_id", 0), self.pool.builder_index(generation)]
        # Compatibility with journals written before dispatch reservations.
        for row in self.pool.db.execute("SELECT input FROM trajectories"):
            molecule = pickle.loads(row["input"])
            name = molecule.get_moleculeid()
            suffix = name[len(prefix):] if name.startswith(prefix) else ""
            if suffix.isdecimal():
                indices.append(int(suffix) + 1)
        status["current_molecule_id"] = max(indices)

    def reserve_builder(self, generation, next_index):
        self.pool.reserve_builder_index(generation, next_index)

    def task_config(self, molecules, current):
        identifiers = [m.get_metadata()["acquisition_trajectory_id"] for m in molecules]
        cohort = self.pool.cohort_for_trajectory(identifiers[0])
        if any(self.pool.trajectory(identifier)["cohort_id"] != cohort["id"] for identifier in identifiers):
            raise ValueError("A physical sampler task cannot mix selection-cohort configurations.")
        pinned = deepcopy(cohort["sampler_config"])
        operational = {"strategy", "engine", "device", "strict_gpu", "allow_cpu_debug", "threads"}
        pinned["driver"] = {**dict(pinned.get("driver") or {}),
                            **{key: value for key, value in current["driver"].items() if key in operational}}
        pinned["batching"] = deepcopy(current.get("batching", {}))
        # The normalized dynamics block is a compatibility view, not another
        # source of strategy choices when replaying under a different driver.
        dynamics = dict(pinned.get("dynamics") or {})
        for key in operational | {"driver", "backend"}:
            dynamics.pop(key, None)
        pinned["dynamics"] = dynamics
        return normalize_sampling_config(pinned)

    def submitted_sampler(self, future, molecules):
        self.sampler_inputs[id(future)] = list(molecules)

    def begin_sampler(self, molecules):
        self.pool.mark_submitted([m.get_metadata()["acquisition_trajectory_id"] for m in molecules])

    def release_finished(self, sampler_queue, qm_queue):
        live_sampler = {id(future) for future in sampler_queue.task_list}
        live_qm = {id(future) for future in qm_queue.task_list}
        self.sampler_inputs = {key: value for key, value in self.sampler_inputs.items() if key in live_sampler}
        self.qm_inputs = {key: value for key, value in self.qm_inputs.items() if key in live_qm}

    def recover_inputs(self, sampler_queue, buffer):
        live = set()
        for future in sampler_queue.task_list:
            live.update(m.get_metadata().get("acquisition_trajectory_id")
                        for m in self._inputs(future, self.sampler_inputs) if m is not None)
        for bucket in buffer._buckets.values():
            live.update(entry.molecule.get_metadata().get("acquisition_trajectory_id") for entry in bucket)
        return [molecule for molecule in self.pool.pending_trajectories(include_submitted=True)
                if molecule.get_metadata()["acquisition_trajectory_id"] not in live]

    def ingest_samplers(self, queue, *, lifecycle=None):
        for future in list(queue.task_list):
            if not is_final_task(future):
                continue
            molecules = [m for m in self._inputs(future, self.sampler_inputs) if m is not None]
            identifiers = [m.get_metadata()["acquisition_trajectory_id"] for m in molecules]
            if not identifiers:
                raise RuntimeError("A cohort sampler result has no durable logical input identity.")
            try:
                output = future.result()
            except Exception as error:
                self.pool.mark_failed(identifiers, error)
                if lifecycle is None:
                    raise
                continue
            try:
                self.pool.accept_results(identifiers, [molecule for molecule in flatten_molecule_output(output)
                                                       if molecule.get_atoms() is not None])
            except BaseException:
                # Delivery acknowledgement must never precede durable ingress.
                if lifecycle is not None and hasattr(future, "key"):
                    lifecycle.deliveries.discard(future.key)
                raise

    def qm_ready(self, queue, *, recover=False):
        live = {m.get_moleculeid() for future in queue.task_list
                for m in self._inputs(future, self.qm_inputs) if m is not None}
        return [m for m in self.pool.ready_qm(include_submitted=recover, include_failed=recover)
                if m.get_moleculeid() not in live]

    def begin_qm(self, molecule):
        self.pool.mark_qm_submitted(molecule.get_moleculeid())

    def submitted_qm(self, future, molecule):
        self.qm_inputs[id(future)] = [molecule]
        receipt = getattr(future, "key", None)
        if not isinstance(receipt, (str, int, float, bool, type(None))):
            receipt = getattr(future, "tid", None)
        if not isinstance(receipt, (str, int, float, bool, type(None))):
            receipt = None
        self.pool.mark_qm_submitted(molecule.get_moleculeid(), receipt=receipt)

    def ingest_qm(self, queue, *, lifecycle=None):
        for future in list(queue.task_list):
            if not is_final_task(future):
                continue
            inputs = [m for m in self._inputs(future, self.qm_inputs) if m is not None]
            if not inputs or not inputs[0].get_metadata().get("acquisition_candidate_id"):
                continue
            original = inputs[0]
            try:
                output = flatten_molecule_output(future.result())
            except Exception as error:
                self.pool.mark_qm_failed(original.get_moleculeid(), error)
                if lifecycle is None:
                    raise
                continue
            if len(output) != 1:
                raise ValueError("A cohort QM task must return one labeled molecule per selected frame.")
            molecule = output[0]
            molecule._moleculeid = original.get_moleculeid()
            molecule.update_metadata({**original.get_metadata(), **molecule.get_metadata(),
                                      "_workflow_origin": original.get_moleculeid()})
            self.pool.record_qm_result(molecule)

    def reconcile_storage(self, pattern):
        """Reconcile a crash after H5 commit but before pool acknowledgement."""
        import h5py
        from alframework.tools.workflow_io import shards
        known = {row[0] for row in self.pool.db.execute("SELECT id FROM candidates WHERE state NOT IN ('stored','rejected','abandoned')")}
        if not known:
            return
        stored = set()
        for _, path in shards(pattern):
            with h5py.File(path) as handle:
                for group in handle.values():
                    if "_id" in group:
                        stored.update(known.intersection(group["_id"].asstr()[:].tolist()))
        self.pool.mark_stored(stored)

    def storage_results(self, current_results):
        results = {m.get_moleculeid(): m for m in self.pool.ready_qm_results()}
        results.update({m.get_moleculeid(): m for m in current_results})
        return list(results.values())

    def stored(self, offered, kept):
        known = {row[0] for row in self.pool.db.execute("SELECT id FROM candidates WHERE state='completed'")}
        offered_ids = known.intersection(m.get_moleculeid() for m in offered)
        kept_ids = offered_ids.intersection(m.get_moleculeid() for m in kept)
        self.pool.mark_stored(kept_ids, rejected_ids=offered_ids - kept_ids)

    def validate_reload(self, proposed):
        from alframework.tools.acquisition_cohorts import _science_config
        from alframework.tools.batched_config import cohort_selection
        if self.pool.has_open_work() and not cohort_selection(proposed):
            raise ValueError("Cannot replace an acquisition coordinator with unfinished work.")
        if self.pool.has_open_work() and cohort_selection(proposed):
            selection = proposed['acquisition']['selection']
            if (selection['cohort_size'], selection['k']) != (self.pool.cohort_size, self.pool.k):
                raise ValueError("Cannot change logical cohort size/K with unfinished acquisition work.")
        identity = json.dumps(_science_config(proposed), sort_keys=True, separators=(",", ":"), default=str)
        for row in self.pool.db.execute("SELECT config_identity FROM cohorts WHERE state='open'"):
            if row['config_identity'] != identity:
                raise ValueError("Cannot change acquisition/sampling science while logical cohorts are unfinished.")
