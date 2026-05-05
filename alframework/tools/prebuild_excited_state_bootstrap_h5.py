#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from alframework.builders.excited_state_builder import build_excited_state_replay_structures
from alframework.tools.tools import load_config_file, store_current_data


def _resolve_config_path(path_value: str, master_directory: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(master_directory) / path
    return path


def _load_configs(master_path: Path, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    os.chdir(run_dir)
    master_config = load_config_file(str(master_path))
    builder_config = load_config_file(master_config["builder_config_path"], master_config["master_directory"])
    qm_config = load_config_file(master_config["QM_config_path"], master_config["master_directory"])
    return master_config, builder_config, qm_config


def _validate_molecule(molecule, properties_list: dict[str, list[Any]]) -> list[str]:
    errors: list[str] = []
    atoms = molecule.get_atoms()
    if atoms is None:
        errors.append("atoms is None")
        return errors
    if molecule.check_convergence() is not True:
        errors.append("converged flag is not True")
    results = molecule.get_results()
    missing = sorted(set(properties_list).difference(results))
    if missing:
        errors.append(f"missing result keys: {missing}")
    return errors


def _write_status(status_path: Path, *, current_h5_id: int) -> None:
    status = {
        "current_training_id": 0,
        "current_model_id": -1,
        "current_h5_id": int(current_h5_id),
        "current_molecule_id": 0,
        "lifetime_failed_builder_tasks": 0,
        "lifetime_failed_sampler_tasks": 0,
        "lifetime_failed_ML_tasks": 0,
        "lifetime_failed_QM_tasks": 0,
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def main(default_run_dir: str | Path | None = None) -> int:
    run_dir_default = Path(default_run_dir or Path.cwd()).resolve()
    parser = argparse.ArgumentParser(
        description="Prebuild ALF bootstrap HDF5 from pre-labeled excited-state seed files."
    )
    parser.add_argument("--run-dir", default=str(run_dir_default), help="Excited-state ALF run/example directory.")
    parser.add_argument("--master", default=None, help="Master config path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of bootstrap frames to prebuild.")
    parser.add_argument("--dry-run", action="store_true", help="Validate molecules without writing HDF5/status.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing target HDF5 shard and status file.")
    parser.add_argument("--no-status", action="store_true", help="Do not write the status file that makes ALF skip bootstrap.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    master_path = Path(args.master).expanduser().resolve() if args.master else run_dir / "master_config_3node_darwin_test.json"
    master_config, builder_config, qm_config = _load_configs(master_path, run_dir)
    if not builder_config.get("prelabeled_seed_file_names"):
        raise RuntimeError("builder_config must define prelabeled_seed_file_names for prebuilt bootstrap HDF5.")
    if not bool(qm_config.get("accept_prelabeled", False)):
        print("Warning: QM_config.accept_prelabeled is not true. HDF5 prebuild can still proceed.")

    bootstrap_set = int(master_config["bootstrap_set"])
    if args.limit is not None:
        bootstrap_set = min(bootstrap_set, int(args.limit))
    if bootstrap_set <= 0:
        raise RuntimeError("bootstrap_set must be positive.")

    h5_path = Path(master_config["h5_path"].format(0))
    status_path = _resolve_config_path(master_config["status_path"], master_config["master_directory"])
    if h5_path.exists() and not args.force and not args.dry_run:
        raise FileExistsError(f"Target HDF5 already exists: {h5_path}. Use --force to overwrite.")
    if status_path.exists() and not args.force and not args.dry_run:
        raise FileExistsError(f"Status file already exists: {status_path}. Use --force to overwrite.")

    moleculeids = [f"mol-boot-{index:010d}" for index in range(bootstrap_set)]
    print(f"Building {bootstrap_set} pre-labeled bootstrap molecules from seed data")
    molecules = build_excited_state_replay_structures(
        moleculeids=moleculeids,
        builder_config=builder_config,
        properties_list=master_config["properties_list"],
        h5_path=master_config["h5_path"],
        current_h5_id=0,
    )

    failed: list[tuple[str, list[str]]] = []
    for molecule in molecules:
        errors = _validate_molecule(molecule, master_config["properties_list"])
        if errors:
            failed.append((molecule.get_moleculeid(), errors))

    if failed:
        print(f"Validation failed for {len(failed)} molecules")
        for moleculeid, errors in failed[:20]:
            print(f"  {moleculeid}: {'; '.join(errors)}")
        return 2

    print(f"Validated {len(molecules)} pre-labeled molecules")
    if args.dry_run:
        print("Dry run complete; no HDF5/status written")
        return 0

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".h5",
        prefix=f".{h5_path.name}.",
        dir=h5_path.parent,
        delete=False,
    ) as handle:
        temp_h5_path = Path(handle.name)

    try:
        if temp_h5_path.exists():
            temp_h5_path.unlink()
        store_current_data(str(temp_h5_path), molecules, master_config["properties_list"])
        if args.force and h5_path.exists():
            h5_path.unlink()
        shutil.move(str(temp_h5_path), str(h5_path))
    finally:
        if temp_h5_path.exists():
            temp_h5_path.unlink()

    print(f"Wrote prebuilt bootstrap HDF5: {h5_path}")
    if not args.no_status:
        if args.force and status_path.exists():
            status_path.unlink()
        _write_status(status_path, current_h5_id=1)
        print(f"Wrote bootstrap-skip status: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
