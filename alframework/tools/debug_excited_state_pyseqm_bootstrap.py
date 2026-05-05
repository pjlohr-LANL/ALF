#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from alframework.builders.excited_state_builder import build_excited_state_replay_structures
from alframework.qm_interfaces.pyseqm_interface import label_excited_state_molecule
from alframework.tools.tools import load_config_file


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(np.all(np.isfinite(value))),
            "min": float(np.nanmin(value)) if value.size else None,
            "max": float(np.nanmax(value)) if value.size else None,
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_configs(master_path: Path, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    os.chdir(run_dir)
    master_config = load_config_file(str(master_path))
    builder_config = load_config_file(master_config["builder_config_path"], master_config["master_directory"])
    sampler_config = load_config_file(master_config["sampler_config_path"], master_config["master_directory"])
    qm_config = load_config_file(master_config["QM_config_path"], master_config["master_directory"])
    return master_config, builder_config, sampler_config, qm_config


def _summarize_molecule(index: int, molecule) -> dict[str, Any]:
    atoms = molecule.get_atoms()
    metadata = dict(molecule.get_metadata())
    results = molecule.get_results()
    return {
        "frame_index": int(index),
        "molecule_id": molecule.get_moleculeid(),
        "formula": atoms.get_chemical_formula() if atoms is not None else None,
        "n_atoms": len(atoms) if atoms is not None else None,
        "converged": bool(molecule.check_convergence()),
        "result_keys": sorted(results.keys()),
        "results": _jsonable(results),
        "metadata": _jsonable(metadata),
        "qm_error": metadata.get("qm_error"),
    }


def main(default_run_dir: str | Path | None = None) -> int:
    run_dir_default = Path(default_run_dir or Path.cwd()).resolve()
    parser = argparse.ArgumentParser(
        description="Run excited-state seed bootstrap frames through PySEQM directly and report per-frame failures."
    )
    parser.add_argument("--run-dir", default=str(run_dir_default), help="Excited-state ALF run/example directory.")
    parser.add_argument("--master", default=None, help="Master config to mirror.")
    parser.add_argument("--output", default=None, help="JSON report path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of bootstrap frames to test.")
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=None,
        help="Override gpus_per_node passed to the PySEQM label helper. Defaults to master_config.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    master_path = Path(args.master).expanduser().resolve() if args.master else run_dir / "master_config_3node_darwin_test.json"
    output_path = Path(args.output).expanduser() if args.output else run_dir / "pyseqm_bootstrap_debug_report.json"
    if not output_path.is_absolute():
        output_path = (run_dir / output_path).resolve()

    master_config, builder_config, sampler_config, qm_config = _load_configs(master_path, run_dir)
    bootstrap_set = int(master_config["bootstrap_set"])
    if args.limit is not None:
        bootstrap_set = min(bootstrap_set, int(args.limit))

    moleculeids = [f"mol-boot-{index:010d}" for index in range(bootstrap_set)]
    gpus_per_node = int(args.gpus_per_node if args.gpus_per_node is not None else master_config.get("gpus_per_node", 0))

    report: dict[str, Any] = {
        "master_config": str(master_path),
        "output": str(output_path),
        "bootstrap_set_tested": int(bootstrap_set),
        "gpus_per_node": int(gpus_per_node),
        "qm_config": _jsonable(qm_config),
        "energy_offset_eV": float(qm_config.get("energy_offset_eV", sampler_config.get("energy_offset_eV", 0.0))),
        "frames": [],
    }

    try:
        molecules = build_excited_state_replay_structures(
            moleculeids=moleculeids,
            builder_config=builder_config,
            properties_list=master_config["properties_list"],
            h5_path=master_config["h5_path"],
            current_h5_id=0,
        )
    except Exception as exc:
        report["builder_error"] = repr(exc)
        report["builder_traceback"] = traceback.format_exc()
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Builder failed before PySEQM labeling. Report: {output_path}")
        print(repr(exc))
        return 1

    converged_count = 0
    for index, molecule in enumerate(molecules):
        try:
            labeled = label_excited_state_molecule(
                molecule_object=molecule,
                QM_config=qm_config,
                properties_list=master_config["properties_list"],
                sampler_config=sampler_config,
                gpus_per_node=gpus_per_node,
            )
            frame_report = _summarize_molecule(index, labeled)
        except Exception as exc:
            frame_report = {
                "frame_index": int(index),
                "molecule_id": molecule.get_moleculeid(),
                "converged": False,
                "top_level_error": repr(exc),
                "top_level_traceback": traceback.format_exc(),
            }

        if frame_report.get("converged"):
            converged_count += 1
        report["frames"].append(frame_report)
        status = "converged" if frame_report.get("converged") else "unconverged"
        detail = frame_report.get("qm_error") or frame_report.get("top_level_error") or ""
        print(f"[{index:03d}] {frame_report['molecule_id']}: {status} {detail}")

    report["summary"] = {
        "total": int(len(report["frames"])),
        "converged": int(converged_count),
        "unconverged": int(len(report["frames"]) - converged_count),
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Summary: {report['summary']}")
    print(f"Wrote report: {output_path}")
    return 0 if converged_count == len(report["frames"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
