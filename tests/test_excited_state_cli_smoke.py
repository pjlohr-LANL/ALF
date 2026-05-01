from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_alframework_test_flags_smoke(tmp_path: Path):
    master_path = tmp_path / "master_config.json"
    (tmp_path / "models" / "model-0000").mkdir(parents=True, exist_ok=True)
    (tmp_path / "builder_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "sampler_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "QM_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ML_config.json").write_text('{"n_models": 1}', encoding="utf-8")
    (tmp_path / "status.txt").write_text(
        json.dumps(
            {
                "current_training_id": 1,
                "current_model_id": 0,
                "current_h5_id": 0,
                "current_molecule_id": 0,
                "lifetime_failed_builder_tasks": 0,
                "lifetime_failed_sampler_tasks": 0,
                "lifetime_failed_ML_tasks": 0,
                "lifetime_failed_QM_tasks": 0,
            }
        ),
        encoding="utf-8",
    )
    master_path.write_text(
        json.dumps(
            {
                "master_directory": str(tmp_path),
                "h5_path": "h5store/data-{:04d}.h5",
                "model_path": "models/model-{:04d}",
                "QM_task": "tests.smoke_modules.smoke_qm_task",
                "QM_config_path": "QM_config.json",
                "ML_task": "tests.smoke_modules.smoke_ml_task",
                "ML_config_path": "ML_config.json",
                "builder_task": "tests.smoke_modules.smoke_builder_task",
                "builder_config_path": "builder_config.json",
                "sampler_task": "tests.smoke_modules.smoke_sampler_task",
                "sampler_config_path": "sampler_config.json",
                "properties_list": {
                    "sE0": ["sE0", "system", 1.0],
                    "F0": ["F0", "atomic", 1.0],
                },
                "status_path": "status.txt",
                "target_queued_QM": 1,
                "minimum_QM": 0,
                "save_h5_threshold": 1,
                "parallel_samplers": 1,
                "bootstrap_set": 1,
                "parsl_configuration": "tests.smoke_modules.local_test_config",
                "parsl_debug_configuration": "tests.smoke_modules.local_test_config",
                "gpus_per_node": 0,
                "QM_scratch_dir": "qm_scratch/",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        "-m",
        "alframework",
        "--master",
        str(master_path),
        "--test_builder",
        "--test_sampler",
        "--test_qm",
        "--test_ml",
    ]
    completed = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise AssertionError(
            "alframework smoke run failed.\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    assert "Builder testing returned:" in completed.stdout
    assert "Sampler testing returned:" in completed.stdout
    assert "ML ensemble training status:" in completed.stdout
    assert (tmp_path / "qm_test.h5").exists()
