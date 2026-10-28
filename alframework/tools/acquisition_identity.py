"""Stable frame identity and scalar ranking shared by workers and coordinator."""
from __future__ import annotations

import hashlib
import json


def proposal_key(metadata):
    return json.dumps(metadata.get("proposal_id", ""), sort_keys=True, separators=(",", ":"), default=str)


def candidate_identity(trajectory_id, metadata):
    if "proposal_id" in metadata:
        proposal = ["proposal", str(metadata["proposal_id"]), metadata.get("step")]
    elif "step" in metadata:
        proposal = ["step", metadata["step"]]
    else:
        raise ValueError("Cohort candidates require a stable step or proposal_id.")
    identity = json.dumps([trajectory_id, proposal], sort_keys=True, separators=(",", ":"), default=str)
    return "c" + hashlib.sha256(identity.encode()).hexdigest()[:31]


def scalar_ranking_key(metadata):
    return (-float(metadata["ranking_score"]), int(metadata["acquisition_trajectory_ordinal"]),
            int(metadata["step"]), proposal_key(metadata),
            candidate_identity(metadata["acquisition_trajectory_id"], metadata))
