#!/usr/bin/env python3
"""Run one exact keto GPU4PySCF label and report diagnostic metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import parsl
from ase.io import read

from alframework.qm_interfaces.gpu4pyscf_interface import (
    gpu4pyscf_excited_state_task,
)
from alframework.tools.molecules_class import MoleculesObject
from parsl_configs import config_darwin_debug


EXAMPLE_DIR = Path(__file__).resolve().parent


def _load_json(filename: str):
    with (EXAMPLE_DIR / filename).open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    master = _load_json("master_config_debug.json")
    qm_config = _load_json("QM_config.json")
    if os.environ.get("ALF_GPU4PYSCF_SMOKE_TDA_CONV_TOL"):
        qm_config["tda_conv_tol"] = float(
            os.environ["ALF_GPU4PYSCF_SMOKE_TDA_CONV_TOL"]
        )
    if os.environ.get("ALF_GPU4PYSCF_SMOKE_TDA_MAX_CYCLE"):
        qm_config["tda_max_cycle"] = int(
            os.environ["ALF_GPU4PYSCF_SMOKE_TDA_MAX_CYCLE"]
        )
    if os.environ.get("ALF_GPU4PYSCF_SMOKE_VERBOSITY"):
        qm_config["verbosity"] = int(
            os.environ["ALF_GPU4PYSCF_SMOKE_VERBOSITY"]
        )
    atoms = read(EXAMPLE_DIR / "keto_form_coords.xyz")
    atoms.set_pbc(False)
    molecule = MoleculesObject(atoms, "gpu4pyscf-qm-smoke")

    parsl.load(config_darwin_debug)
    try:
        result = gpu4pyscf_excited_state_task(
            molecule_object=molecule,
            QM_config=qm_config,
            properties_list=master["properties_list"],
            gpus_per_node=int(master["gpus_per_node"]),
        ).result()
    finally:
        parsl.dfk().cleanup()
        parsl.clear()

    metadata = result.get_metadata()
    summary = {
        "converged": result.check_convergence(),
        "metadata": metadata,
        "result_keys": sorted(result.get_results()),
        "result_shapes": {
            key: list(np.asarray(value).shape)
            for key, value in result.get_results().items()
        },
        "finite_results": {
            key: bool(np.isfinite(np.asarray(value)).all())
            for key, value in result.get_results().items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if result.check_convergence() is not True:
        raise RuntimeError(
            "GPU4PySCF QM smoke test failed: "
            f"{metadata.get('qm_error_type')}: "
            f"{metadata.get('qm_error')}"
        )
    expected = {
        key
        for state in range(6)
        for key in (f"sE{state}", f"F{state}")
    }
    if set(result.get_results()) != expected:
        raise RuntimeError(
            "GPU4PySCF QM smoke test returned an incomplete property set."
        )
    if not all(summary["finite_results"].values()):
        raise RuntimeError(
            "GPU4PySCF QM smoke test returned non-finite values."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
