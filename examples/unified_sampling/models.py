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
