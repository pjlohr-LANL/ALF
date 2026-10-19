"""Physical-force eligibility and bounded, opt-in rejection evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

SCHEMA = "alf.physical_force_screen.v1"
SOURCE = "ensemble_mean_physical_all_states"


def force_screen_metadata(diagnostics):
    """Small diagnostics retained on candidates and in audit snapshots."""
    keys = ("Fmeanmax", "propagation_Fmax", "lcm_bias_Fmax",
            "lcm_correction_Fmax", "confinement_Fmax")
    result = {key: float(diagnostics[key]) for key in keys if key in diagnostics}
    result.update({key: float(value) for key, value in diagnostics.items()
                   if key.startswith("physical_F") and key.endswith("_max")})
    if "force_screen_schema" in diagnostics:
        result["force_screen_schema"] = diagnostics["force_screen_schema"]
        result["force_screen_source"] = diagnostics["force_screen_source"]
    return result


def physical_force_exceeds(diagnostics, cutoff):
    """Equality passes. Missing and invalid screening values never pass silently."""
    if cutoff is None:
        return False
    cutoff = float(cutoff)
    if not np.isfinite(cutoff) or cutoff < 0:
        raise ValueError("max_force_cutoff must be finite and nonnegative.")
    value = diagnostics.get("Fmeanmax")
    if value is None or np.asarray(value).shape != () or not np.isfinite(value) or value < 0:
        raise ValueError("Physical force screen requires a finite nonnegative Fmeanmax.")
    if diagnostics.get("force_screen_schema") != SCHEMA or diagnostics.get("force_screen_source") != SOURCE:
        raise ValueError("Physical force screen requires physical-force schema/source markers.")
    return float(value) > cutoff


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


class ForceScreenAudit:
    """At most three snapshots per replica; written even for an empty batch result.

    Initial coordinates precede integration and have no evaluated forces. Checked
    snapshots contain the same-frame physical means. LCM bias is relative to the
    selected state's physical force; correction is relative to the pair average.
    """

    def __init__(self, directory, master_directory, molecules, atoms_list, cutoff):
        self.path = None
        if directory:
            root = Path(directory)
            if not root.is_absolute():
                root = Path(master_directory or os.getcwd()) / root
            self.path = root / f"force-screen-{uuid4().hex}.json"
        self.data = {"schema": SCHEMA, "source": SOURCE, "max_force_cutoff": cutoff,
                     "replicas": [{"parent_molecule_id": str(mol.get_moleculeid()),
                                   "initial": self.snapshot(atoms, 0, 0.0),
                                   "last_valid": None, "rejected": None}
                                  for mol, atoms in zip(molecules, atoms_list)]}

    @staticmethod
    def snapshot(atoms, step, time_ps, diagnostics=None, reason=None):
        diagnostics = diagnostics or {}
        return {"atomic_numbers": atoms.numbers.tolist(),
                "positions": atoms.positions.tolist(), "step": int(step),
                "time_ps": float(time_ps), "reason": reason,
                "physical_forces": {key: np.asarray(value).tolist()
                                    for key, value in diagnostics.items()
                                    if key.startswith("F") and key[1:].isdigit()},
                **force_screen_metadata(diagnostics)}

    def record(self, index, kind, atoms, step, time_ps, diagnostics=None, reason=None):
        if self.path is not None:
            self.data["replicas"][index][kind] = self.snapshot(
                atoms, step, time_ps, diagnostics, reason)

    def write(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(_json_value(self.data), indent=2, allow_nan=False) + "\n")
        temporary.replace(self.path)
