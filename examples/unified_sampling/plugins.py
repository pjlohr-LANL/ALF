"""Small non-MD proposal method demonstrating the sampler extension contract."""

from __future__ import annotations

import numpy as np

from alframework.samplers.sampling_methods import SamplingObservation


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
