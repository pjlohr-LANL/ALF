#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "metadata_path",
    "molecule_id",
    "backend",
    "batch_size",
    "num_md_steps_completed",
    "num_md_chunks",
    "realtime_simulation",
    "total_wall_s",
    "md_run_wall_s",
    "md_run_cuda_s",
    "model_eval_wall_s",
    "model_eval_cuda_s",
    "host_sync_wall_s",
    "metrics_wall_s",
    "trajectory_io_wall_s",
    "steps_per_total_wall_s",
    "steps_per_md_wall_s",
    "steps_per_md_cuda_s",
    "systems_per_second",
    "atom_steps_per_second",
    "scalar_sync_count",
    "full_sync_count",
    "gpu_name",
    "device",
    "stop_reason",
    "num_qm_candidates",
]


def _load_metadata(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_metadata(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    metadata_path = Path(path).expanduser().resolve()
    root_path = Path(root).expanduser().resolve() if root is not None else metadata_path.parent
    payload = _load_metadata(metadata_path)
    timing = dict(payload.get("timing") or {})
    top_candidates = payload.get("top_candidates") or []
    molecule_id = metadata_path.stem.removeprefix("metadata-")
    try:
        display_path = str(metadata_path.relative_to(root_path))
    except ValueError:
        display_path = str(metadata_path)
    row = {
        "metadata_path": display_path,
        "molecule_id": molecule_id,
        "backend": timing.get("backend", payload.get("dynamics_backend")),
        "batch_size": timing.get("batch_size"),
        "num_md_steps_completed": timing.get("num_md_steps_completed"),
        "num_md_chunks": timing.get("num_md_chunks"),
        "realtime_simulation": payload.get("realtime_simulation"),
        "total_wall_s": timing.get("total_wall_s", payload.get("realtime_simulation")),
        "md_run_wall_s": timing.get("md_run_wall_s"),
        "md_run_cuda_s": timing.get("md_run_cuda_s"),
        "model_eval_wall_s": timing.get("model_eval_wall_s"),
        "model_eval_cuda_s": timing.get("model_eval_cuda_s"),
        "host_sync_wall_s": timing.get("host_sync_wall_s"),
        "metrics_wall_s": timing.get("metrics_wall_s"),
        "trajectory_io_wall_s": timing.get("trajectory_io_wall_s"),
        "steps_per_total_wall_s": timing.get("steps_per_total_wall_s"),
        "steps_per_md_wall_s": timing.get("steps_per_md_wall_s"),
        "steps_per_md_cuda_s": timing.get("steps_per_md_cuda_s"),
        "systems_per_second": timing.get("systems_per_second"),
        "atom_steps_per_second": timing.get("atom_steps_per_second"),
        "scalar_sync_count": timing.get("scalar_sync_count"),
        "full_sync_count": timing.get("full_sync_count"),
        "gpu_name": timing.get("gpu_name"),
        "device": timing.get("device"),
        "stop_reason": payload.get("geometry_reject_reason"),
        "num_qm_candidates": len(top_candidates),
    }
    return row


def summarize_directory(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path).expanduser().resolve()
    paths = sorted(root.glob("metadata-*.json"))
    return [summarize_metadata(item, root=root) for item in paths]


def write_summary(rows: list[dict[str, Any]], output: str | Path, *, fmt: str) -> None:
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        return
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in SUMMARY_FIELDS})


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ALF sampling timing metadata.")
    parser.add_argument("metadata_dir", help="Directory containing metadata-*.json files.")
    parser.add_argument("--output", required=True, help="Output CSV or JSON path.")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Output format.")
    args = parser.parse_args()

    rows = summarize_directory(args.metadata_dir)
    write_summary(rows, args.output, fmt=args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
