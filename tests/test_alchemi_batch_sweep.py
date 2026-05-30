from __future__ import annotations

import json
from pathlib import Path

from alframework.tools.alchemi_batch_sweep import (
    make_waves,
    parse_int_list,
    prepare_sampler_config,
    summarize_sweep_results,
    sweep_molecule_ids,
)


def test_alchemi_batch_sweep_parses_batch_sizes_and_waves():
    assert parse_int_list("4,8,16") == [4, 8, 16]
    assert sweep_molecule_ids(3) == [
        "sweep-mol-0000000000",
        "sweep-mol-0000000001",
        "sweep-mol-0000000002",
    ]
    assert make_waves([4, 8, 16, 32, 64], [0, 1]) == [
        [(4, 0), (8, 1)],
        [(16, 0), (32, 1)],
        [(64, 0)],
    ]


def test_alchemi_batch_sweep_prepares_sampler_config(tmp_path: Path):
    config = prepare_sampler_config(
        {
            "dynamics_backend": "ase",
            "timing": {"enabled": False},
            "udd": {"enabled": True},
            "gap_seeking": {"enabled": True},
        },
        batch_size=32,
        output_dir=tmp_path / "case",
        maxt=0.2,
        ncheck=25,
        fixed_state=0,
    )

    assert config["dynamics_backend"] == "alchemi_baoab"
    assert config["alchemi_baoab"]["batch_size"] == 32
    assert config["timing"]["enabled"] is True
    assert config["udd"]["enabled"] is False
    assert config["gap_seeking"]["enabled"] is False
    assert config["state_selection"] == {"mode": "fixed", "fixed_state": 0}
    assert config["maxt"] == 0.2
    assert config["Ncheck"] == 25
    assert config["meta_dir"].endswith("case/sampling")


def test_alchemi_batch_sweep_summarizes_fake_metadata(tmp_path: Path):
    sampling_dir = tmp_path / "batch_0004" / "repeat_00" / "sampling"
    sampling_dir.mkdir(parents=True)
    for index in range(4):
        payload = {
            "dynamics_backend": "alchemi_baoab",
            "realtime_simulation": 10.0,
            "geometry_reject_reason": None,
            "top_candidates": [{"score": 1.0}],
            "timing": {
                "backend": "alchemi_baoab",
                "batch_size": 4,
                "num_md_steps_completed": 100,
                "num_md_chunks": 10,
                "total_wall_s": 10.0,
                "md_run_wall_s": 5.0,
                "md_run_cuda_s": 4.0,
                "systems_per_second": 80.0,
                "atom_steps_per_second": 1200.0,
                "scalar_sync_count": 2,
                "full_sync_count": 1,
            },
        }
        (sampling_dir / f"metadata-sweep-{index}.json").write_text(json.dumps(payload), encoding="utf-8")

    summary = summarize_sweep_results(tmp_path)

    assert (tmp_path / "batch_sweep_raw.csv").exists()
    assert (tmp_path / "batch_sweep_summary.json").exists()
    assert summary["num_metadata_rows"] == 4
    assert summary["num_run_rows"] == 1
    assert summary["summary"][0]["batch_size"] == 4
    assert summary["summary"][0]["systems_per_second_mean"] == 80.0
