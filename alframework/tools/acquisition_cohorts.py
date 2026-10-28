"""Durable logical acquisition cohorts, independent of physical sampler tasks.

Payloads are trusted local pickle records, as in ALF's existing task journal.
The coordinator is a single writer. Each result ingestion commits candidates,
trajectory completion, and (when possible) the final selection together before
the caller acknowledges the physical sampler task. No GPU stack is imported.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import pickle
import sqlite3
import time
import uuid

from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.workflow_io import ExclusiveLock
from alframework.tools.acquisition_identity import candidate_identity, proposal_key


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _dump(value):
    return sqlite3.Binary(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _load(value):
    return pickle.loads(value)


def _molecule_copy(molecule):
    result = copy.copy(molecule)
    result.atoms = None if molecule.get_atoms() is None else molecule.get_atoms().copy()
    if result.atoms is not None:
        result.atoms.calc = None
    result.metadata = copy.deepcopy(molecule.get_metadata())
    result.qm_results = copy.deepcopy(molecule.get_results())
    return result


def _identity(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _science_config(config):
    """Physical batching/device choices cannot change cohort science identity."""
    result = copy.deepcopy(config)
    result.pop("batching", None)
    driver = dict(result.pop("driver", {}) or {})
    for key in ("strategy", "engine", "device", "strict_gpu", "allow_cpu_debug", "threads"):
        driver.pop(key, None)
    if driver:
        result["driver"] = driver
    for block in ("dynamics", "alchemi_baoab"):
        options = dict(result.pop(block, {}) or {})
        for key in ("driver", "backend", "engine", "strategy", "device", "strict_gpu",
                    "allow_cpu_debug", "batch_size", "partial_policy", "max_wait_seconds"):
            options.pop(key, None)
        if options:
            result[block] = options
    return result


def _candidate_id(trajectory_id, metadata):
    return candidate_identity(trajectory_id, metadata)


class AcquisitionCohorts:
    """Persist full, fixed-membership cohorts and exact scalar top-K selection.

    ``admit`` keeps filling the current logical cohort with its pinned model,
    even if a newer model is offered. The next cohort uses the new reference.
    An empty or rejected trajectory is complete; a failed trajectory is not.
    Pending and submitted work remain distinguishable across restarts.
    """

    def __init__(self, path, *, cohort_size, k, selection_id="scalar_top_k", namespace=None):
        self.cohort_size = _positive(cohort_size, "cohort_size")
        self.k = _positive(k, "k")
        if selection_id != "scalar_top_k":
            raise ValueError("Bounded scalar top-K retention cannot implement a different selector.")
        self.selection_id = selection_id
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = ExclusiveLock(str(self.path) + ".lock")
        self.db = sqlite3.connect(str(self.path), timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS cohorts(
                id TEXT PRIMARY KEY,ordinal INTEGER UNIQUE NOT NULL,target INTEGER NOT NULL,
                k INTEGER NOT NULL,model_reference BLOB NOT NULL,sampler_config BLOB NOT NULL,
                config_identity TEXT NOT NULL,selected_state INTEGER,state TEXT NOT NULL,
                admitted INTEGER NOT NULL DEFAULT 0,completed INTEGER NOT NULL DEFAULT 0,
                eligible INTEGER NOT NULL DEFAULT 0,reason TEXT,created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS trajectories(
                id TEXT PRIMARY KEY,source_identity TEXT UNIQUE NOT NULL,
                cohort_id TEXT NOT NULL REFERENCES cohorts(id),ordinal INTEGER UNIQUE NOT NULL,
                member_index INTEGER NOT NULL,input BLOB NOT NULL,state TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,error TEXT);
            CREATE TABLE IF NOT EXISTS candidates(
                id TEXT PRIMARY KEY,cohort_id TEXT NOT NULL REFERENCES cohorts(id),
                trajectory_id TEXT NOT NULL REFERENCES trajectories(id),score REAL NOT NULL,
                trajectory_ordinal INTEGER NOT NULL,step INTEGER NOT NULL,proposal TEXT NOT NULL,
                payload BLOB NOT NULL,state TEXT NOT NULL,rank INTEGER,receipt BLOB,error TEXT,
                result BLOB);
            CREATE INDEX IF NOT EXISTS cohort_trajectory_idx ON trajectories(cohort_id,state);
            CREATE INDEX IF NOT EXISTS cohort_candidate_idx ON candidates(cohort_id,score);
        """)
        row = self.db.execute("SELECT value FROM metadata WHERE key='namespace'").fetchone()
        if row is None:
            self.namespace = uuid.uuid4().hex if namespace is None else str(namespace)
            with self.db:
                self.db.execute("INSERT INTO metadata VALUES ('namespace',?)", (_dump(self.namespace),))
                self.db.execute("INSERT INTO metadata VALUES ('schema',?)", (_dump(1),))
        else:
            self.namespace = _load(row["value"])
            if namespace is not None and str(namespace) != self.namespace:
                self.db.close()
                self._lock.close()
                raise ValueError("The acquisition journal belongs to another namespace.")
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def close(self):
        if not self.closed:
            self.db.close()
            self._lock.close()
            self.closed = True

    def flush(self):
        self.db.commit()

    def reserve_builder_index(self, generation, next_index):
        """Reserve input identities before dispatch, independently of status JSON."""
        key = f"builder_index:{generation}"
        next_index = max(int(next_index), self.builder_index(generation))
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, _dump(next_index)))

    def builder_index(self, generation):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (f"builder_index:{generation}",)).fetchone()
        return 0 if row is None else int(_load(row["value"]))

    def _id(self, kind, ordinal):
        return kind + hashlib.sha256(f"{self.namespace}:{kind}:{ordinal}".encode()).hexdigest()[:31]

    def _selected_state(self, config, ordinal, supplied):
        if supplied is not None:
            return int(supplied)
        if config.get("model_mode", "ground_state") != "excited_state":
            return None
        policy = dict(config.get("state_selection") or {})
        mode = policy.get("mode", "fixed")
        if mode == "fixed":
            value = policy.get("state", policy.get("selected_state", config.get("selected_state")))
            if value is None:
                raise ValueError("Excited-state cohorts require an explicit state policy.")
            return int(value)
        if mode not in {"batch_cycle", "cohort_cycle", "cycle"}:
            raise ValueError("Unsupported cohort state_selection mode.")
        states = policy.get("states")
        if not isinstance(states, (list, tuple)) or not states:
            raise ValueError("Cohort state cycling requires a nonempty states list.")
        return int(states[ordinal % len(states)])

    def admit(self, molecule, model_reference, sampler_config, selected_state=None, *, source_identity=None):
        if not isinstance(molecule, MoleculesObject) or molecule.get_atoms() is None:
            raise ValueError("Cohort admission requires a molecule with atoms.")
        metadata = molecule.get_metadata()
        source = str(source_identity if source_identity is not None else
                     metadata.get("_workflow_origin", molecule.get_moleculeid()))
        previous = self.db.execute("SELECT input FROM trajectories WHERE source_identity=?", (source,)).fetchone()
        if previous is not None:
            return _molecule_copy(_load(previous["input"]))
        reference = asdict(model_reference) if is_dataclass(model_reference) else copy.deepcopy(model_reference)
        config = copy.deepcopy(sampler_config)
        identity = _identity(_science_config(config))
        with self.db:
            if selected_state is None:
                cohort = self.db.execute(
                    "SELECT * FROM cohorts WHERE state='open' AND admitted<target ORDER BY ordinal LIMIT 1").fetchone()
            else:
                # Explicit builder state identities must never be overwritten
                # simply because a different-state cohort is admitted first.
                cohort = self.db.execute(
                    "SELECT * FROM cohorts WHERE state='open' AND admitted<target AND selected_state=? ORDER BY ordinal LIMIT 1",
                    (int(selected_state),)).fetchone()
            if cohort is not None:
                if cohort["target"] != self.cohort_size or cohort["k"] != self.k or cohort["config_identity"] != identity:
                    raise ValueError("Cannot change acquisition science or cohort size while a cohort is being admitted.")
            else:
                ordinal = self.db.execute("SELECT COALESCE(MAX(ordinal),-1)+1 FROM cohorts").fetchone()[0]
                state = self._selected_state(config, ordinal, selected_state)
                cohort_id = self._id("w", ordinal)
                self.db.execute("INSERT INTO cohorts(id,ordinal,target,k,model_reference,sampler_config,config_identity,selected_state,state,created) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (cohort_id, ordinal, self.cohort_size, self.k, _dump(reference), _dump(config), identity, state, "open", time.time()))
                cohort = self.db.execute("SELECT * FROM cohorts WHERE id=?", (cohort_id,)).fetchone()
            ordinal = self.db.execute("SELECT COALESCE(MAX(ordinal),-1)+1 FROM trajectories").fetchone()[0]
            trajectory_id = self._id("t", ordinal)
            snapshot = _molecule_copy(molecule)
            base_seed = dict(config.get("driver") or {}).get("random_seed",
                        dict(config.get("dynamics") or {}).get("random_seed", config.get("random_seed", 42)))
            seed = int.from_bytes(hashlib.sha256(f"{base_seed}:{ordinal}".encode()).digest()[:4], "big")
            pinned_reference = _load(cohort["model_reference"])
            tags = {"acquisition_cohort_id": cohort["id"], "cohort_id": cohort["id"],
                    "acquisition_cohort_ordinal": cohort["ordinal"],
                    "acquisition_trajectory_id": trajectory_id, "trajectory_id": trajectory_id,
                    "acquisition_trajectory_ordinal": ordinal, "trajectory_seed": seed,
                    "acquisition_model_reference": pinned_reference}
            if cohort["selected_state"] is not None:
                tags.update(selected_state=cohort["selected_state"], selected_state_source="state_selection.logical_cohort")
            snapshot.update_metadata(tags)
            self.db.execute("INSERT INTO trajectories(id,source_identity,cohort_id,ordinal,member_index,input,state) VALUES (?,?,?,?,?,?,?)",
                (trajectory_id, source, cohort["id"], ordinal, cohort["admitted"], _dump(snapshot), "pending"))
            self.db.execute("UPDATE cohorts SET admitted=admitted+1 WHERE id=?", (cohort["id"],))
        return snapshot

    def trajectory(self, trajectory_id):
        row = self.db.execute("SELECT * FROM trajectories WHERE id=?", (str(trajectory_id),)).fetchone()
        if row is None:
            raise KeyError(trajectory_id)
        result = dict(row)
        result["input"] = _molecule_copy(_load(row["input"]))
        return result

    def cohort(self, cohort_id):
        row = self.db.execute("SELECT * FROM cohorts WHERE id=?", (str(cohort_id),)).fetchone()
        if row is None:
            raise KeyError(cohort_id)
        result = dict(row)
        for key in ("model_reference", "sampler_config"):
            result[key] = _load(row[key])
        result["required_trajectory_count"] = row["target"]
        return result

    def cohort_for_trajectory(self, trajectory_id):
        return self.cohort(self.trajectory(trajectory_id)["cohort_id"])

    def pending_trajectories(self, *, include_submitted=False):
        states = ("pending", "failed", "submitted") if include_submitted else ("pending", "failed")
        rows = self.db.execute(f"SELECT input FROM trajectories WHERE state IN ({','.join('?' for _ in states)}) ORDER BY ordinal", states)
        return [_molecule_copy(_load(row["input"])) for row in rows]

    def waiting_ids(self, *, include_submitted=True):
        return [m.get_metadata()["acquisition_trajectory_id"] for m in
                self.pending_trajectories(include_submitted=include_submitted)]

    def retryable_trajectories(self):
        return self.pending_trajectories()

    def mark_submitted(self, trajectory_ids):
        with self.db:
            for identifier in trajectory_ids:
                row = self.trajectory(identifier)
                if row["state"] == "completed":
                    continue
                if row["state"] == "abandoned":
                    raise ValueError("Cannot submit an abandoned trajectory.")
                self.db.execute("UPDATE trajectories SET state='submitted',attempts=attempts+1,error=NULL WHERE id=?", (identifier,))

    def mark_failed(self, trajectory_ids, error):
        with self.db:
            for identifier in trajectory_ids:
                self.trajectory(identifier)
                self.db.execute("UPDATE trajectories SET state='failed',error=? WHERE id=? AND state!='completed'",
                                (str(error), identifier))

    def _candidate(self, candidate, trajectories):
        if not isinstance(candidate, MoleculesObject) or candidate.get_atoms() is None:
            raise ValueError("Cohort candidates must be molecule objects with atoms.")
        metadata = copy.deepcopy(candidate.get_metadata())
        identifier = metadata.get("acquisition_trajectory_id", metadata.get("trajectory_id"))
        if identifier is None:
            parent = metadata.get("parent_molecule_id")
            matches = [key for key, row in trajectories.items()
                       if row["input"].get_moleculeid() == parent]
            if len(matches) != 1:
                raise ValueError("Candidate must identify its logical trajectory unambiguously.")
            identifier = matches[0]
        if identifier not in trajectories:
            raise ValueError("Candidate belongs to a trajectory outside this result.")
        row = trajectories[identifier]
        score = float(metadata.get("ranking_score", metadata.get("uncertainty_score", float("nan"))))
        if not math.isfinite(score):
            raise ValueError("Cohort ranking scores must be finite.")
        step = metadata.get("step", 0)
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
            raise ValueError("Candidate step must be a nonnegative integer.")
        step = int(step)
        if "step" in metadata:
            metadata["step"] = step
        candidate_id = _candidate_id(identifier, metadata)
        result = _molecule_copy(candidate)
        result._moleculeid = candidate_id
        parent = row["input"]
        pinned = parent.get_metadata()
        if ("selected_state" in metadata and "selected_state" in pinned
                and metadata["selected_state"] != pinned["selected_state"]):
            raise ValueError("A candidate cannot change its cohort's selected state.")
        if ("acquisition_model_reference" in metadata
                and metadata["acquisition_model_reference"] != pinned["acquisition_model_reference"]):
            raise ValueError("A candidate cannot change its cohort's pinned model.")
        result.update_metadata({**parent.get_metadata(), **metadata,
            "candidate_id": candidate_id, "acquisition_candidate_id": candidate_id,
            "acquisition_trajectory_id": identifier, "trajectory_id": identifier,
            "acquisition_cohort_id": row["cohort_id"], "cohort_id": row["cohort_id"],
            "acquisition_trajectory_ordinal": row["ordinal"], "ranking_score": score,
            "acquisition_model_reference": pinned["acquisition_model_reference"],
            "parent_molecule_id": parent.get_moleculeid(), "_workflow_origin": candidate_id})
        return result, score, step, proposal_key(metadata)

    def accept_results(self, trajectory_ids, candidates):
        """Atomically ingest complete trajectory results before task acknowledgement."""
        identifiers = list(dict.fromkeys(str(value) for value in trajectory_ids))
        if not identifiers:
            raise ValueError("A sampler result must name at least one trajectory.")
        trajectories = {identifier: self.trajectory(identifier) for identifier in identifiers}
        unfinished = {key: row for key, row in trajectories.items() if row["state"] != "completed"}
        if not unfinished:
            return []
        if any(row["state"] == "abandoned" for row in unfinished.values()):
            raise ValueError("Cannot ingest results for abandoned acquisition work.")
        prepared = []
        seen = set()
        for candidate in candidates:
            result, score, step, proposal = self._candidate(candidate, trajectories)
            identifier = result.get_metadata()["acquisition_trajectory_id"]
            if identifier not in unfinished:
                continue
            candidate_id = result.get_moleculeid()
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            prepared.append((result, score, step, proposal, unfinished[identifier]))
        cohort_ids = sorted({row["cohort_id"] for row in unfinished.values()})
        newly_selected = []
        with self.db:
            for result, score, step, proposal, trajectory in prepared:
                self.db.execute("INSERT OR IGNORE INTO candidates(id,cohort_id,trajectory_id,score,trajectory_ordinal,step,proposal,payload,state) VALUES (?,?,?,?,?,?,?,?,?)",
                    (result.get_moleculeid(), trajectory["cohort_id"], trajectory["id"], score,
                     trajectory["ordinal"], step, proposal, _dump(result), "retained"))
                self.db.execute("UPDATE cohorts SET eligible=eligible+1 WHERE id=?", (trajectory["cohort_id"],))
            for identifier, row in unfinished.items():
                self.db.execute("UPDATE trajectories SET state='completed',error=NULL WHERE id=?", (identifier,))
                self.db.execute("UPDATE cohorts SET completed=completed+1 WHERE id=?", (row["cohort_id"],))
            for cohort_id in cohort_ids:
                cohort = self.cohort(cohort_id)
                order = "score DESC,trajectory_ordinal,step,proposal,id"
                retained = self.db.execute(f"SELECT id FROM candidates WHERE cohort_id=? ORDER BY {order}", (cohort_id,)).fetchall()
                for discarded in retained[cohort["k"]:]:
                    self.db.execute("DELETE FROM candidates WHERE id=?", (discarded["id"],))
                if cohort["admitted"] == cohort["target"] and cohort["completed"] == cohort["target"]:
                    selected = self.db.execute(f"SELECT id,payload FROM candidates WHERE cohort_id=? ORDER BY {order}", (cohort_id,)).fetchall()
                    for rank, item in enumerate(selected):
                        molecule = _load(item["payload"])
                        molecule.update_metadata({"candidate_rank": rank, "candidate_count": len(selected),
                                                  "acquisition_selection": "cohort_top_k"})
                        self.db.execute("UPDATE candidates SET state='pending',rank=?,payload=? WHERE id=?",
                                        (rank, _dump(molecule), item["id"]))
                        newly_selected.append(molecule)
                    self.db.execute("UPDATE cohorts SET state='selected' WHERE id=?", (cohort_id,))
        return newly_selected

    def ready_qm(self, *, include_submitted=False, include_failed=False):
        states = ["pending"] + (["submitted"] if include_submitted else []) + (["failed"] if include_failed else [])
        rows = self.db.execute(f"SELECT c.payload FROM candidates c JOIN cohorts w ON w.id=c.cohort_id WHERE c.state IN ({','.join('?' for _ in states)}) ORDER BY w.ordinal,c.rank", states)
        return [_molecule_copy(_load(row["payload"])) for row in rows]

    def _transition(self, candidate_id, state, *, receipt=None, error=None, result=None):
        identifier = candidate_id.get_moleculeid() if isinstance(candidate_id, MoleculesObject) else str(candidate_id)
        row = self.db.execute("SELECT state FROM candidates WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError(identifier)
        if row["state"] == "retained":
            raise ValueError("Cannot submit a candidate before its complete cohort is selected.")
        if row["state"] in {"stored", "rejected", "abandoned"} and row["state"] != state:
            raise ValueError("A terminal candidate handoff cannot be changed.")
        if row["state"] == "completed" and state in {"submitted", "failed"}:
            return
        with self.db:
            self.db.execute("UPDATE candidates SET state=?,receipt=COALESCE(?,receipt),error=?,result=COALESCE(?,result) WHERE id=?",
                            (state, None if receipt is None else _dump(receipt), error,
                             None if result is None else _dump(_molecule_copy(result)), identifier))

    def mark_qm_submitted(self, candidate_id, receipt=None):
        self._transition(candidate_id, "submitted", receipt=receipt)

    def mark_qm_completed(self, candidate_id):
        self._transition(candidate_id, "completed")

    def mark_qm_failed(self, candidate_id, error):
        self._transition(candidate_id, "failed", error=str(error))

    def record_qm_result(self, molecule):
        self._transition(molecule, "completed", result=molecule)

    def ready_qm_results(self):
        rows = self.db.execute("SELECT result FROM candidates WHERE state='completed' AND result IS NOT NULL ORDER BY cohort_id,rank")
        return [_molecule_copy(_load(row["result"])) for row in rows]

    def mark_stored(self, candidate_ids, *, rejected_ids=()):
        for identifier in candidate_ids:
            self._transition(identifier, "stored")
        for identifier in rejected_ids:
            self._transition(identifier, "rejected")

    def abandon(self, reason):
        with self.db:
            self.db.execute("UPDATE cohorts SET state='abandoned',reason=? WHERE state='open'", (str(reason),))
            self.db.execute("UPDATE trajectories SET state='abandoned',error=? WHERE state IN ('pending','submitted','failed')", (str(reason),))
            self.db.execute("UPDATE candidates SET state='abandoned',error=? WHERE state IN ('retained','pending','failed')", (str(reason),))

    def has_open_work(self):
        return bool(self.db.execute("SELECT COUNT(*) FROM cohorts WHERE state='open'").fetchone()[0]
                    or self.db.execute("SELECT COUNT(*) FROM candidates WHERE state IN ('pending','submitted','failed','completed')").fetchone()[0])

    def status(self):
        def counts(table):
            return {row[0]: row[1] for row in self.db.execute(f"SELECT state,COUNT(*) FROM {table} GROUP BY state")}
        candidate_counts = counts("candidates")
        ingested = self.db.execute("SELECT COALESCE(SUM(eligible),0) FROM cohorts").fetchone()[0]
        return {"cohorts": counts("cohorts"), "trajectories": counts("trajectories"),
                "candidates": candidate_counts,
                "retained_candidates": sum(candidate_counts.get(state, 0) for state in
                                           ("retained", "pending", "submitted", "failed", "completed")),
                "journal_candidates": sum(candidate_counts.values()),
                "selected_pending_qm": candidate_counts.get("pending", 0),
                "submitted_qm": candidate_counts.get("submitted", 0),
                "completed_pending_storage": candidate_counts.get("completed", 0),
                "eligible_candidates": ingested, "ingested_candidates": ingested}
