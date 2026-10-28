"""Analytic ASE ensemble for exercising the unified sampler without Torch.

The harmonic potential is an integration fixture, not a molecular force field.
"""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, stiffness, offset=0.):
        super().__init__()
        self.stiffness = float(stiffness)
        self.offset = float(offset)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties or ["energy", "forces"], system_changes)
        self.results = {
            "energy": .5 * self.stiffness * float(np.square(atoms.positions).sum()) + self.offset,
            "forces": -self.stiffness * atoms.positions,
        }


def load_harmonic_ensemble(ensemble_directory, device=None, **options):
    """Match ALF's ASE ensemble-loader signature; no checkpoint is required."""
    del ensemble_directory, device
    if options:
        raise ValueError(f"Unexpected harmonic fixture options: {sorted(options)}")
    return [HarmonicCalculator(.9, -.01), HarmonicCalculator(1.1, .01)]


class PhysicalHarmonicEnsemble:
    """A structural physical evaluator with no ALF/Torch model base class."""

    array_backend = "numpy"
    calculator_interface = "native"
    calculator_loader = "examples.unified_sampling.models.load_physical_harmonic_ensemble"
    selected_state_value = 0
    capabilities = {"supports_pbc": True, "supports_partial_pbc": True,
                    "supports_constraints": True, "supports_member_contributions": True}

    def predict(self, atoms_list):
        positions = np.concatenate([atoms.positions for atoms in atoms_list])
        system_energy = np.asarray([.5 * np.square(atoms.positions).sum() for atoms in atoms_list])
        return {
            "energy_contributions": np.stack([.9 * system_energy - .01, 1.1 * system_energy + .01]),
            "force_contributions": np.stack([-.9 * positions, -1.1 * positions]),
        }


def load_physical_harmonic_ensemble(model_reference, requested_properties=None, device=None, **options):
    """Demonstrate the opaque-artifact adapter API without model-family files."""
    del model_reference, device
    request = dict(requested_properties or {})
    if request.get("model_mode", "ground_state") != "ground_state":
        raise ValueError("The analytic physical fixture supports ground-state outputs only.")
    if options:
        raise ValueError(f"Unexpected physical fixture options: {sorted(options)}")
    return PhysicalHarmonicEnsemble()
