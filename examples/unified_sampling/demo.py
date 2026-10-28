"""Run the analytic ASE example without a scheduler, QM, or model checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from ase import Atoms

from alframework.samplers.sampling import sampling_task
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.acquisition_cohorts import AcquisitionCohorts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("mlmd", "udd", "proposals"), default="mlmd")
    parser.add_argument("--acquisition-policy", choices=("stop", "continue"), default="stop")
    parser.add_argument("--physical-evaluator", action="store_true",
                        help="Use the bare NumPy physical-evaluator API instead of ASE calculators")
    parser.add_argument("--acquisition", choices=("uncertainty", "geometry"), default="uncertainty")
    args = parser.parse_args(argv)
    config = json.loads(Path(__file__).with_name("sampler_ase.json").read_text())
    if args.physical_evaluator:
        config["inference"] = {"interface": "native",
            "loader": "examples.unified_sampling.models.load_physical_harmonic_ensemble", "options": {}}
        config["driver"]["device"] = "cpu"
    config["sampling"] = {"propagator": "langevin", "biases": []}
    config["acquisition"]["trajectory_policy"] = args.acquisition_policy
    if args.acquisition == "geometry":
        config["acquisition"].update(strategy="examples.unified_sampling.plugins.CoordinateExtentAcquisition",
                                      options={"min_extent_A": .3})
        config.pop("Escut", None)
        config.pop("Fscut", None)
    pooled = args.acquisition_policy == "continue"
    if pooled:
        config["acquisition"]["selection"] = {"mode": "cohort_top_k", "cohort_size": 1, "k": 4}
    if args.method == "udd":
        config["sampling"]["biases"] = [{"strategy": "udd", "weight": .45, "epsilon": 1e-12}]
    elif args.method == "proposals":
        config["sampling"] = {
            "propagator": "examples.unified_sampling.plugins.ScaledProposalMethod",
            "biases": [],
            "options": {"scales": [1., 1.01]},
        }
        for key in ("dt", "maxt", "Ncheck", "srt_temp", "end_temp", "amp_temp", "per_temp"):
            config.pop(key)
    atoms = Atoms("H2", positions=[[.1, .1, .1], [.8, .1, .1]])
    atoms.set_velocities(np.full((len(atoms), 3), .01))
    with tempfile.TemporaryDirectory(prefix="alf-unified-demo-") as directory:
        parent = MoleculesObject(atoms, "unified-demo")
        pool = AcquisitionCohorts(Path(directory)/"cohort.sqlite", cohort_size=1, k=4) if pooled else None
        try:
            if pool is not None:
                parent = pool.admit(parent, {"generation": 0, "artifact_location": "analytic-fixture",
                    "adapter": config["inference"]["loader"], "interface": config["inference"]["interface"]}, config)
            result = sampling_task.func(
                molecule_objects=[parent], sampler_config=config,
                model_path="unused-model-{:04d}", current_model_id=0, gpus_per_node=0,
                ML_config={}, properties_list={"energy": ["energy", "system", 1.],
                                               "forces": ["forces", "atomic", 1.]},
            )
            if hasattr(result, "reraise"):
                result.reraise()
            if pool is not None:
                pool.accept_results([parent.get_metadata()["acquisition_trajectory_id"]], result)
                result = pool.ready_qm()
        finally:
            if pool is not None:
                pool.close()
    print(json.dumps({
        "sampling_propagator": config["sampling"]["propagator"],
        "sampling_biases": config["sampling"]["biases"],
        "acquisition_policy": args.acquisition_policy,
        "acquisition_strategy": config["acquisition"]["strategy"],
        "execution_driver": "single", "engine": "ase", "inference_interface": config["inference"]["interface"],
        "selection": config["acquisition"]["selection"],
        "candidates": [{"molecule_id": molecule.get_moleculeid(),
                        "parent_molecule_id": molecule.get_metadata()["parent_molecule_id"],
                        "step": molecule.get_metadata()["step"]} for molecule in result],
    }, indent=2))
    return result


if __name__ == "__main__":
    main()
