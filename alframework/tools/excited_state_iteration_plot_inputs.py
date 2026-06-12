#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This script requires numpy. Run it in the same environment used for ALF, "
        "for example: conda activate /vast/home/pjlohr/.conda/envs/atomistic"
    ) from exc


MODEL_RE = re.compile(r"model-(\d{4})$")
CHILD_MODEL_RE = re.compile(r"model-(\d{2})$")
METADATA_RE = re.compile(r"metadata-mol-(\d{4})-\d{10}\.json$")
ENERGY_MAE_RE = re.compile(r"^sE\d+_MAE$")
FORCE_MAE_RE = re.compile(r"^F\d+_MAE$")


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _resolve_path(run_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = run_dir / path
    return path


def _stats(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "n": 0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "n": int(finite.size),
    }


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    try:
        return float(metrics[key])
    except Exception:
        return math.nan


def _macro_mae(metrics: dict[str, Any], regex: re.Pattern[str]) -> float:
    values = [_metric_value(metrics, key) for key in metrics if regex.match(key)]
    values = [value for value in values if np.isfinite(value)]
    if not values:
        return math.nan
    return float(np.mean(values))


def _aggregate_training_summaries(model_root: Path, run_dir: Path) -> dict[str, Any]:
    child_dirs = [
        path
        for path in sorted(model_root.glob("model-*"))
        if path.is_dir() and CHILD_MODEL_RE.match(path.name)
    ]
    summaries: list[tuple[Path, dict[str, Any]]] = []
    for child in child_dirs:
        summary_path = child / "training_summary.json"
        if summary_path.exists():
            summaries.append((summary_path, _load_json(summary_path)))

    raw_metrics: dict[str, dict[str, dict[str, float | int]]] = {"by_split": {}}
    derived: dict[str, dict[str, dict[str, float | int]]] = {"by_split": {}}
    for split in ("train", "valid", "test"):
        split_metrics = [
            dict(((summary.get("metric") or {}).get(split) or {}))
            for _, summary in summaries
        ]
        metric_names = sorted({key for metrics in split_metrics for key in metrics})
        raw_metrics["by_split"][split] = {
            key: _stats([_metric_value(metrics, key) for metrics in split_metrics])
            for key in metric_names
        }
        derived["by_split"][split] = {
            "energy_mae_macro": _stats([_macro_mae(metrics, ENERGY_MAE_RE) for metrics in split_metrics]),
            "force_mae_macro": _stats([_macro_mae(metrics, FORCE_MAE_RE) for metrics in split_metrics]),
        }

    return {
        "n_models": int(len(summaries)),
        "model_dirs": [str(path.parent.relative_to(run_dir)) for path, _ in summaries],
        "source_training_summaries": [str(path.relative_to(run_dir)) for path, _ in summaries],
        "raw_metrics": raw_metrics,
        "derived": derived,
    }


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return ".nan"
        if math.isinf(value):
            return ".inf" if value > 0 else "-.inf"
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    return json.dumps(str(value))


def _write_yaml_value(lines: list[str], value: Any, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                _write_yaml_value(lines, item, indent + 2)
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
    elif isinstance(value, list):
        if not value:
            lines.append(f"{prefix}[]")
            return
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                _write_yaml_value(lines, item, indent + 2)
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
    else:
        lines.append(f"{prefix}{_yaml_scalar(value)}")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    lines: list[str] = []
    _write_yaml_value(lines, data, 0)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _candidate_record(
    *,
    parent_molecule_id: str,
    generation: int,
    rank: int,
    candidate: dict[str, Any],
    uncertainty_scope: str,
    uncertainty_aggregate: str,
) -> dict[str, Any]:
    if uncertainty_scope == "selected_state":
        uE_used = candidate.get("uE_selected", candidate.get("uE_rms", candidate.get("uE_max")))
        uF_used = candidate.get("uF_selected", candidate.get("uF_rms", candidate.get("uF_max")))
    elif uncertainty_aggregate == "rms":
        uE_used = candidate.get("uE_rms", candidate.get("uE_max"))
        uF_used = candidate.get("uF_rms", candidate.get("uF_max"))
    else:
        uE_used = candidate.get("uE_max", candidate.get("uE_rms"))
        uF_used = candidate.get("uF_max", candidate.get("uF_rms"))

    return {
        "iter": int(generation),
        "parent_molecule_id": parent_molecule_id,
        "molecule_id": f"{parent_molecule_id}-cand-{int(rank):02d}",
        "candidate_rank": int(rank),
        "score": candidate.get("score"),
        "uE": candidate.get("uE_max", candidate.get("uE_rms")),
        "uE_used": uE_used,
        "uF": uF_used,
        "uE_selected": candidate.get("uE_selected"),
        "uF_selected": candidate.get("uF_selected"),
        "uncertainty_state": candidate.get("uncertainty_state"),
        "uE_max": candidate.get("uE_max"),
        "uE_rms": candidate.get("uE_rms"),
        "uF_max": candidate.get("uF_max"),
        "uF_rms": candidate.get("uF_rms"),
        "min_gap": candidate.get("min_gap"),
        "min_gap_pair": candidate.get("min_gap_pair"),
        "selected_state": candidate.get("selected_state"),
        "selected_energy_eV": candidate.get("selected_energy_eV"),
        "fmax": candidate.get("fmax"),
        "min_dist": candidate.get("min_dist"),
        "max_nearest_neighbor_distance": candidate.get("max_nearest_neighbor_distance"),
        "step": candidate.get("step"),
        "time_ps": candidate.get("time_ps"),
        "temperature_K": candidate.get("temperature_K"),
        "temperature_target_K": candidate.get("temperature_target_K"),
    }


def _collect_sampler_records(run_dir: Path, sampler_config: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    metadata_dir = _resolve_path(run_dir, str(sampler_config.get("meta_dir", "sampling/")))
    uncertainty_scope = str(
        ((sampler_config.get("score") or {}).get("uncertainty_scope") or "selected_state")
    ).strip().lower()
    uncertainty_aggregate = str(
        ((sampler_config.get("score") or {}).get("uncertainty_aggregate") or "max")
    ).strip().lower()
    records_by_generation: dict[int, list[dict[str, Any]]] = {}
    if not metadata_dir.exists():
        return records_by_generation

    for path in sorted(metadata_dir.glob("metadata-mol-*.json")):
        match = METADATA_RE.match(path.name)
        if not match:
            continue
        generation = int(match.group(1))
        metadata = _load_json(path)
        parent_id = path.stem.replace("metadata-", "", 1)
        candidates = list(metadata.get("top_candidates") or [])
        for rank, candidate in enumerate(candidates):
            records_by_generation.setdefault(generation, []).append(
                _candidate_record(
                    parent_molecule_id=parent_id,
                    generation=generation,
                    rank=rank,
                    candidate=dict(candidate),
                    uncertainty_scope=uncertainty_scope,
                    uncertainty_aggregate=uncertainty_aggregate,
                )
            )
    return records_by_generation


def _discover_generations(model_dir: Path) -> list[tuple[int, Path]]:
    generations: list[tuple[int, Path]] = []
    if not model_dir.exists():
        return generations
    for path in sorted(model_dir.glob("model-*")):
        match = MODEL_RE.match(path.name)
        if path.is_dir() and match:
            generations.append((int(match.group(1)), path))
    return generations


def _ensure_output_file(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def build_iteration_inputs(run_dir: Path, output_dir: Path, overwrite: bool = False) -> None:
    master = _load_json(run_dir / "master_config_3node_darwin_test.json")
    sampler_config = _load_json(_resolve_path(run_dir, master["sampler_config_path"]))
    model_pattern = _resolve_path(run_dir, master["model_path"])
    model_dir = model_pattern.parent
    records_by_generation = _collect_sampler_records(run_dir, sampler_config)

    generations = _discover_generations(model_dir)
    if not generations:
        raise RuntimeError(f"No model generations found in {model_dir}")

    for generation, model_root in generations:
        iter_dir = output_dir / f"iter_{generation:04d}"
        model_out = iter_dir / "models" / "ensemble_summary.yaml"
        sampler_out = iter_dir / "sampling" / "selected_topk.jsonl"
        _ensure_output_file(model_out, overwrite)
        _ensure_output_file(sampler_out, overwrite)

        summary = _aggregate_training_summaries(model_root, run_dir)
        summary["iteration"] = int(generation)
        summary["source_model_root"] = str(model_root.relative_to(run_dir))
        _write_yaml(model_out, summary)

        with open(sampler_out, "w", encoding="utf-8") as handle:
            for record in records_by_generation.get(generation, []):
                handle.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")

        print(
            f"iter_{generation:04d}: wrote {model_out.relative_to(run_dir)} "
            f"and {sampler_out.relative_to(run_dir)} "
            f"({len(records_by_generation.get(generation, []))} selected records)"
        )


def main(default_run_dir: str | Path | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build notebook-compatible ALF iteration plot inputs from flat excited-state ALF outputs."
    )
    parser.add_argument(
        "--run-dir",
        default=str(default_run_dir or Path.cwd()),
        help="Excited-state ALF run/example directory.",
    )
    parser.add_argument("--output-dir", default=None, help="Output iterations directory. Defaults to <run-dir>/iterations.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated files.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir is not None else run_dir / "iterations"
    if not output_dir.is_absolute():
        output_dir = run_dir / output_dir
    output_dir = output_dir.resolve()

    build_iteration_inputs(run_dir, output_dir, overwrite=bool(args.overwrite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
