from __future__ import annotations

import csv
import json
from pathlib import Path

from alframework.tools.sampling_timing_summary import summarize_directory, write_summary


def test_sampling_timing_summary_handles_timed_and_legacy_metadata(tmp_path: Path):
    timed = {
        "dynamics_backend": "alchemi_baoab",
        "realtime_simulation": 12.0,
        "geometry_reject_reason": None,
        "top_candidates": [{"score": 1.0}, {"score": 0.5}],
        "timing": {
            "backend": "alchemi_baoab",
            "device": "cuda:0",
            "gpu_name": "Fake GPU",
            "batch_size": 1,
            "num_md_steps_completed": 100,
            "num_md_chunks": 10,
            "total_wall_s": 12.0,
            "md_run_wall_s": 8.0,
            "md_run_cuda_s": 7.5,
            "steps_per_total_wall_s": 8.333333333,
            "steps_per_md_wall_s": 12.5,
            "steps_per_md_cuda_s": 13.333333333,
            "systems_per_second": 12.5,
            "atom_steps_per_second": 375.0,
            "scalar_sync_count": 20,
            "full_sync_count": 10,
        },
    }
    legacy = {
        "dynamics_backend": "ase",
        "realtime_simulation": 25.0,
        "geometry_reject_reason": "min_distance",
        "top_candidates": [],
    }
    (tmp_path / "metadata-mol-0001.json").write_text(json.dumps(timed), encoding="utf-8")
    (tmp_path / "metadata-mol-0002.json").write_text(json.dumps(legacy), encoding="utf-8")

    rows = summarize_directory(tmp_path)

    assert rows[0]["backend"] == "alchemi_baoab"
    assert rows[0]["md_run_cuda_s"] == 7.5
    assert rows[0]["num_qm_candidates"] == 2
    assert rows[1]["backend"] == "ase"
    assert rows[1]["total_wall_s"] == 25.0
    assert rows[1]["stop_reason"] == "min_distance"


def test_sampling_timing_summary_writes_csv(tmp_path: Path):
    rows = [
        {
            "metadata_path": "metadata-mol-0001.json",
            "molecule_id": "mol-0001",
            "backend": "ase",
            "total_wall_s": 1.0,
        }
    ]
    output = tmp_path / "summary.csv"

    write_summary(rows, output, fmt="csv")

    with open(output, newline="", encoding="utf-8") as handle:
        loaded = list(csv.DictReader(handle))
    assert loaded[0]["backend"] == "ase"
    assert loaded[0]["total_wall_s"] == "1.0"
