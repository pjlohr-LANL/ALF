"""Small non-MD proposal method demonstrating the sampler extension contract."""

from __future__ import annotations

import numpy as np

from alframework.samplers.sampling_methods import SamplingObservation, AcquisitionDecision


class ScaledProposalMethod:
    """Evaluate coordinate-scaled proposals; this is not an optimizer."""

    required_capabilities = frozenset({"grouped_evaluation"})

    def create_program(self, context):
        scales = list(context.config["sampling"].get("options", {}).get("scales", [1., 1.01]))
        if not scales or any(not np.isfinite(scale) or scale <= 0 for scale in scales):
            raise ValueError("Proposal scales must be finite and positive.")

        class Program:
            def run(self):
                requests = []
                for parent_index, parent in enumerate(context.molecule_objects):
                    for proposal_index, scale in enumerate(scales):
                        atoms = parent.get_atoms().copy()
                        atoms.positions *= scale
                        requests.append((parent_index, parent, proposal_index, scale, atoms))
                evaluated = context.evaluate_group([row[-1] for row in requests])
                candidates, stopped = [], set()
                for evaluation_index, row in enumerate(requests):
                    parent_index, parent, proposal_index, scale, atoms = row
                    if parent_index in stopped:
                        continue
                    event = context.acquire(SamplingObservation(
                        system_index=parent_index, molecule=parent, atoms=atoms,
                        diagnostics=evaluated["diagnostics"][evaluation_index],
                        step=proposal_index + 1,
                        progress={"proposal_index": proposal_index, "coordinate_scale": float(scale)},
                    ))
                    if event.candidate is not None:
                        candidates.append(event.candidate)
                    if event.stop:
                        stopped.add(parent_index)
                return context.finalize(candidates)

        return Program()


class CoordinateExtentAcquisition:
    """Translation-invariant geometry-only gate; not a production descriptor.

    This toy uses the largest distance to the geometric centroid for isolated
    systems. It requires no uncertainty values or model-family features.
    """

    required_signals = frozenset({"positions"})
    retention = "scalar_top_k"

    def decide(self, observation, context):
        if np.any(observation.atoms.pbc):
            raise ValueError("The toy extent policy supports nonperiodic structures only.")
        settings = context.config["acquisition"]
        threshold = float(settings.get("options", {})["min_extent_A"])
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("min_extent_A must be finite and positive.")
        positions = observation.atoms.positions
        extent = float(np.linalg.norm(positions-positions.mean(axis=0), axis=1).max())
        capture = extent > threshold
        policy = settings.get("trajectory_policy", settings.get("policy", "stop"))
        return AcquisitionDecision(capture=capture, stop=capture and policy == "stop",
            metadata={"ranking_score": extent/threshold, "score_mode": "geometry_extent",
                      "geometry_extent_A": extent, "geometry_threshold_A": threshold})

    def select(self, candidates, context, *, final=False):
        # The coordinator owns cohort-wide selection; no trajectory quota or
        # physical-task-local selection is introduced by this policy.
        return candidates
