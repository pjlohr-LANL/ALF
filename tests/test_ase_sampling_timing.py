from __future__ import annotations

import json
from pathlib import Path

from alframework.tools.ase_sampling_timing import (
    prepare_ase_sampler_config,
    summarize_ase_timing_results,
)


def test_ase_sampling_timing_prepares_sampler_config(tmp_path: Path):
    config = prepare_ase_sampler_config(
        {
            "dynamics_backend": "alchemi_baoab",
            "alchemi_baoab": {"batch_size": 8},
            "timing": {"enabled": False, "cuda_events": True},
            "udd": {"enabled": True},
            "gap_seeking": {"enabled": True},
        },
        output_dir=tmp_path / "case",
        maxt=0.2,
        ncheck=25,
        fixed_state=0,
    )

    assert config["dynamics_backend"] == "ase"
    assert "alchemi_baoab" not in config
    assert config["timing"]["enabled"] is True
    assert config["udd"]["enabled"] is False
    assert config["gap_seeking"]["enabled"] is False
    assert config["state_selection"] == {"mode": "fixed", "fixed_state": 0}
    assert config["maxt"] == 0.2
    assert config["Ncheck"] == 25
    assert config["meta_dir"].endswith("case/sampling")


def test_ase_sampling_timing_summarizes_fake_metadata(tmp_path: Path):
    sampling_dir = tmp_path / "count_0004" / "repeat_00" / "sampling"
    sampling_dir.mkdir(parents=True)
    for index in range(4):
        payload = {
            "dynamics_backend": "ase",
            "realtime_simulation": 10.0 + index,
            "geometry_reject_reason": None,
            "top_candidates": [{"score": 1.0}],
            "timing": {
                "backend": "ase",
                "batch_size": 1,
                "num_md_steps_completed": 100,
                "num_md_chunks": 10,
                "total_wall_s": 10.0 + index,
                "md_run_wall_s": 5.0,
                "systems_per_second": 20.0,
                "atom_steps_per_second": 300.0,
                "scalar_sync_count": 2,
                "full_sync_count": 0,
            },
        }
        (sampling_dir / f"metadata-sweep-{index}.json").write_text(json.dumps(payload), encoding="utf-8")

    summary = summarize_ase_timing_results(tmp_path)

    assert (tmp_path / "ase_sampling_raw.csv").exists()
    assert (tmp_path / "ase_sampling_summary.json").exists()
    assert summary["num_metadata_rows"] == 4
    assert summary["summary"][0]["trajectory_count"] == 4
    assert summary["summary"][0]["num_metadata_rows"] == 4
    assert summary["summary"][0]["systems_per_second_mean"] == 20.0
