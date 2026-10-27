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
    cutoff = validated_force_cutoff(cutoff)
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
                "cell": atoms.cell.array.tolist(), "pbc": atoms.pbc.tolist(),
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


def validated_force_cutoff(cutoff, *, positive=False):
    """Validate a screening cutoff, retaining the configuration's positive rule."""
    value = float(cutoff)
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        if positive:
            raise ValueError("max_force_cutoff must be finite and positive when provided.")
        raise ValueError("max_force_cutoff must be finite and nonnegative.")
    return value


def _force_maximum(forces, batch_ptr=None):
    """Reduce packed atoms or legacy uniform [systems, atoms, 3] forces."""
    import torch

    if batch_ptr is None:
        return torch.linalg.vector_norm(forces, dim=2).amax(dim=1)
    values = torch.linalg.vector_norm(forces.reshape(-1, 3), dim=1)
    counts = batch_ptr[1:] - batch_ptr[:-1]
    indices = torch.repeat_interleave(
        torch.arange(counts.numel(), device=values.device), counts
    )
    result = values.new_zeros(counts.numel())
    return result.scatter_reduce_(0, indices, values, reduce="amax", include_self=True)


def physical_force_diagnostics(state_results, num_graphs, *, batch_ptr=None):
    """Max atom norm of each physical ensemble mean, then max over states."""
    import torch

    maxima = {f"physical_F{state}_max": _force_maximum(
        result["forces"].reshape(num_graphs, -1, 3) if batch_ptr is None else result["forces"], batch_ptr)
        for state, result in state_results.items()}
    return {**maxima, "Fmeanmax": torch.stack(list(maxima.values())).amax(dim=0)}


def lcm_force_diagnostics(lcm_forces, selected_physical_forces, lcm_correction, *, batch_ptr=None):
    """Separate total LCM bias from the correction to pair-average forces."""
    import torch

    return {"lcm_bias_Fmax": _force_maximum(lcm_forces - selected_physical_forces, batch_ptr),
            "lcm_correction_Fmax": _force_maximum(lcm_correction, batch_ptr)}


def propagation_force_diagnostics(propagation_forces, well_forces, *, batch_ptr=None):
    """Propagation includes artificial forces; its maximum is not eligibility."""
    import torch

    return {"confinement_Fmax": (
                torch.zeros(propagation_forces.shape[0] if batch_ptr is None else batch_ptr.numel() - 1,
                            device=propagation_forces.device)
                if well_forces is None else _force_maximum(well_forces, batch_ptr)),
            "propagation_Fmax": _force_maximum(propagation_forces, batch_ptr)}
