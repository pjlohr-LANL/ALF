#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_config_path(run_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = run_dir / path
    return path


def _count_h5_systems(path: Path) -> int | None:
    try:
        import h5py
    except Exception:
        return None

    try:
        total = 0
        with h5py.File(path, "r") as handle:
            for group in handle.values():
                if "_id" in group:
                    total += len(group["_id"])
        return total
    except Exception:
        return None


def _print_status(status_path: Path) -> None:
    status = _load_json(status_path)
    print("ALF excited-state run status")
    print("============================")
    if not status:
        print(f"Status: missing ({status_path})")
        return
    for key in [
        "current_training_id",
        "current_model_id",
        "current_h5_id",
        "current_molecule_id",
        "lifetime_failed_builder_tasks",
        "lifetime_failed_sampler_tasks",
        "lifetime_failed_QM_tasks",
        "lifetime_failed_ML_tasks",
    ]:
        print(f"{key}: {status.get(key)}")


def _print_artifacts(run_dir: Path, h5_dir: Path, model_dir: Path) -> None:
    h5_files = sorted(h5_dir.glob("data-*.h5"))
    model_roots = sorted(model_dir.glob("model-*"))
    print("")
    print("Artifacts")
    print("=========")
    print(f"HDF5 shards: {len(h5_files)}")
    for path in h5_files[-5:]:
        saved = _count_h5_systems(path)
        suffix = f" ({saved} saved systems)" if saved is not None else ""
        print(f"  {path.relative_to(run_dir)}{suffix}")
    print(f"Model roots: {len(model_roots)}")
    for path in model_roots[-5:]:
        child_models = sorted(path.glob("model-*"))
        print(f"  {path.relative_to(run_dir)} ({len(child_models)} model dirs)")


def _print_debug_report(run_dir: Path, debug_report: Path) -> None:
    print("")
    print("PySEQM Debug")
    print("============")
    if not debug_report.exists():
        print(f"Report: missing ({debug_report.relative_to(run_dir)})")
        print("Create it with: python -m alframework.tools.debug_excited_state_pyseqm_bootstrap --run-dir <run-dir>")
        return
    report = _load_json(debug_report)
    summary = report.get("summary", {})
    print(f"Report: {debug_report.relative_to(run_dir)}")
    print(
        "Frames: total={total} converged={converged} unconverged={unconverged}".format(
            total=summary.get("total"),
            converged=summary.get("converged"),
            unconverged=summary.get("unconverged"),
        )
    )
    failed = [frame for frame in report.get("frames", []) if not frame.get("converged")]
    for frame in failed[:5]:
        detail = frame.get("qm_error") or frame.get("top_level_error") or "<no error captured>"
        print(f"  {frame.get('molecule_id')}: {detail}")


def _print_parsl(run_dir: Path) -> None:
    print("")
    print("Parsl")
    print("=====")
    run_dirs = sorted((run_dir / "runinfo").glob("[0-9][0-9][0-9]"))
    runinfo_dir = run_dirs[-1] if run_dirs else run_dir / "runinfo" / "000"
    if not runinfo_dir.exists():
        print(f"runinfo missing: {runinfo_dir}")
        return
    print(f"runinfo: {runinfo_dir.relative_to(run_dir)}")

    submit_dir = runinfo_dir / "submit_scripts"
    for ec_path in sorted(submit_dir.glob("*.ec")):
        value = ec_path.read_text(encoding="utf-8").strip()
        print(f"{ec_path.name}: {value or '<empty>'}")

    for executor in ["alf_gpu_executor", "alf_ML_executor", "alf_sampler_executor", "alf_QM_executor"]:
        executor_root = runinfo_dir / executor / "block-0"
        if not executor_root.exists():
            continue
        managers = sorted(executor_root.glob("*/manager.log"))
        workers = sorted(executor_root.glob("*/worker_*.log"))
        status = "ok" if managers else "missing"
        print(f"{executor}: manager_logs={len(managers)} worker_logs={len(workers)} {status}")


def main(default_run_dir: str | Path | None = None) -> int:
    run_dir_default = Path(default_run_dir or Path.cwd()).resolve()
    parser = argparse.ArgumentParser(description="Summarize an excited-state ALF run directory.")
    parser.add_argument("--run-dir", default=str(run_dir_default), help="Excited-state ALF run/example directory.")
    parser.add_argument("--master", default=None, help="Master config path.")
    parser.add_argument("--debug-report", default=None, help="Optional PySEQM bootstrap debug report path.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    master_path = Path(args.master).expanduser().resolve() if args.master else run_dir / "master_config_3node_darwin_test.json"
    config = _load_json(master_path)
    status_path = _resolve_config_path(run_dir, config.get("status_path", "status_small_3node_darwin_test.txt"))
    h5_pattern = _resolve_config_path(run_dir, config.get("h5_path", "h5store_small_3node_darwin_test/data-{:04d}.h5"))
    model_pattern = _resolve_config_path(run_dir, config.get("model_path", "models_small_3node_darwin_test/model-{:04d}"))
    debug_report = (
        Path(args.debug_report).expanduser().resolve()
        if args.debug_report
        else run_dir / "pyseqm_bootstrap_debug_report.json"
    )

    _print_status(status_path)
    _print_artifacts(run_dir, h5_pattern.parent, model_pattern.parent)
    _print_debug_report(run_dir, debug_report)
    _print_parsl(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
