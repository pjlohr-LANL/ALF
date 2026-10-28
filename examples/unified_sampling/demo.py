"""Run the analytic ASE example without a scheduler, QM, or model checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms

from alframework.samplers.sampling import sampling_task
from alframework.tools.molecules_class import MoleculesObject


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("mlmd", "udd", "proposals"), default="mlmd")
    parser.add_argument("--acquisition-policy", choices=("stop", "continue"), default="stop")
    args = parser.parse_args(argv)
    config = json.loads(Path(__file__).with_name("sampler_ase.json").read_text())
    config["sampling"]["method"] = args.method
    config["acquisition"]["policy"] = args.acquisition_policy
    config["return_top_k"] = 4
    if args.method == "udd":
        config["udd_bias_weight"] = .45
    elif args.method == "proposals":
        config["sampling"] = {
            "method": "examples.unified_sampling.plugins.ScaledProposalMethod",
            "options": {"scales": [1., 1.01]},
        }
        for key in ("dt", "maxt", "Ncheck", "srt_temp", "end_temp", "amp_temp", "per_temp"):
            config.pop(key)
    atoms = Atoms("H2", positions=[[.1, .1, .1], [.8, .1, .1]])
    atoms.set_velocities(np.full((len(atoms), 3), .01))
    result = sampling_task.func(
        molecule_objects=[MoleculesObject(atoms, "unified-demo")], sampler_config=config,
        model_path="unused-model-{:04d}", current_model_id=0, gpus_per_node=0,
        ML_config={}, properties_list={"energy": ["energy", "system", 1.],
                                       "forces": ["forces", "atomic", 1.]},
    )
    if hasattr(result, "reraise"):
        result.reraise()
    print(json.dumps({
        "sampling_method": config["sampling"]["method"],
        "acquisition_policy": args.acquisition_policy,
        "execution_driver": "ase", "inference_interface": "ase",
        "candidates": [{"molecule_id": molecule.get_moleculeid(),
                        "parent_molecule_id": molecule.get_metadata()["parent_molecule_id"],
                        "step": molecule.get_metadata()["step"]} for molecule in result],
    }, indent=2))
    return result


if __name__ == "__main__":
    main()
