from __future__ import annotations

import argparse
import copy
import csv
import json
import multiprocessing as mp
import os
import statistics
import sys
from pathlib import Path
from typing import Any

from alframework.tools.sampling_timing_summary import SUMMARY_FIELDS, summarize_directory


DEFAULT_BATCH_SIZES = [4, 8, 16, 32, 64, 128, 256, 512]
AGGREGATE_FIELDS = [
    "total_wall_s",
    "md_run_wall_s",
    "md_run_cuda_s",
    "model_eval_wall_s",
    "model_eval_cuda_s",
    "host_sync_wall_s",
    "metrics_wall_s",
    "steps_per_total_wall_s",
    "steps_per_md_wall_s",
    "steps_per_md_cuda_s",
    "systems_per_second",
    "atom_steps_per_second",
    "scalar_sync_count",
    "full_sync_count",
]


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_int_list(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer.")
    return [int(item) for item in items]


def sweep_molecule_ids(pool_size: int) -> list[str]:
    size = int(pool_size)
    if size <= 0:
        raise ValueError("pool_size must be positive.")
    return [f"sweep-mol-{index:010d}" for index in range(size)]


def make_waves(batch_sizes: list[int], gpus: list[int]) -> list[list[tuple[int, int]]]:
    if not gpus:
        raise ValueError("At least one GPU id is required.")
    waves: list[list[tuple[int, int]]] = []
    for start in range(0, len(batch_sizes), len(gpus)):
        wave_sizes = batch_sizes[start : start + len(gpus)]
        waves.append([(int(batch_size), int(gpus[index])) for index, batch_size in enumerate(wave_sizes)])
    return waves


def resolve_run_path(run_dir: str | Path, value: str) -> str:
    path_text = str(value)
    if os.path.isabs(path_text):
        return path_text
    return str(Path(run_dir).expanduser().resolve() / path_text)


def prepare_sampler_config(
    base_config: dict[str, Any],
    *,
    batch_size: int,
    output_dir: str | Path,
    maxt: float,
    ncheck: int,
    fixed_state: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    output_root = Path(output_dir).expanduser().resolve()
    sampling_dir = output_root / "sampling"
    candidate_dir = sampling_dir / "qm_candidates"
    config["dynamics_backend"] = "alchemi_baoab"
    alchemi = dict(config.get("alchemi_baoab") or {})
    alchemi.update(
        {
            "batch_size": int(batch_size),
            "allow_partial_batches": False,
            "strict_gpu": True,
            "allow_cpu_debug": False,
            "scalar_sync_policy": "control_only",
            "require_same_selected_state": True,
            "batched_gap_switch_policy": "global_lowest_gap_trigger",
        }
    )
    config["alchemi_baoab"] = alchemi
    timing = dict(config.get("timing") or {})
    timing.update({"enabled": True, "cuda_events": True, "record_chunk_timings": False})
    config["timing"] = timing
    config["maxt"] = float(maxt)
    config["Ncheck"] = int(ncheck)
    config["meta_dir"] = str(sampling_dir)
    config["qm_candidate_xyz_dir"] = str(candidate_dir)
    config["trajectory_frequency"] = 0.0
    config["write_traj_binary"] = False
    config["write_traj_xyz"] = False
    config["write_qm_candidate_xyz"] = True
    config["metadata_format"] = "both"
    config["state_selection"] = {"mode": "fixed", "fixed_state": int(fixed_state)}
    udd = dict(config.get("udd") or {})
    udd["enabled"] = False
    config["udd"] = udd
    gap_seeking = dict(config.get("gap_seeking") or {})
    gap_seeking["enabled"] = False
    config["gap_seeking"] = gap_seeking
    return config


def prepare_builder_config(base_config: dict[str, Any], *, fixed_state: int) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["state_selection"] = {"mode": "fixed", "fixed_state": int(fixed_state)}
    return config


def _write_rows_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    extra_fields = ["sweep_batch_size", "repeat", "result_dir"]
    fields = extra_fields + [field for field in SUMMARY_FIELDS if field not in extra_fields]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def summarize_sweep_results(
    result_root: str | Path,
    *,
    ase_summary: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(result_root).expanduser().resolve()
    raw_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for sampling_dir in sorted(root.glob("batch_*/repeat_*/sampling")):
        repeat_dir = sampling_dir.parent
        batch_dir = repeat_dir.parent
        try:
            batch_size = int(batch_dir.name.removeprefix("batch_"))
            repeat = int(repeat_dir.name.removeprefix("repeat_"))
        except ValueError:
            continue
        rows = summarize_directory(sampling_dir)
        for row in rows:
            row = dict(row)
            row["sweep_batch_size"] = batch_size
            row["repeat"] = repeat
            row["result_dir"] = str(repeat_dir.relative_to(root))
            raw_rows.append(row)
        if rows:
            representative = dict(rows[0])
            representative["sweep_batch_size"] = batch_size
            representative["repeat"] = repeat
            representative["result_dir"] = str(repeat_dir.relative_to(root))
            run_rows.append(representative)

    summary_rows: list[dict[str, Any]] = []
    for batch_size in sorted({int(row["sweep_batch_size"]) for row in run_rows}):
        selected = [row for row in run_rows if int(row["sweep_batch_size"]) == batch_size]
        aggregate: dict[str, Any] = {"batch_size": batch_size, "repeats": len(selected)}
        for field in AGGREGATE_FIELDS:
            values = [
                float(row[field])
                for row in selected
                if row.get(field) not in (None, "")
            ]
            if values:
                aggregate[f"{field}_mean"] = float(statistics.mean(values))
                aggregate[f"{field}_stdev"] = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        summary_rows.append(aggregate)

    _write_rows_csv(raw_rows, root / "batch_sweep_raw.csv")
    payload: dict[str, Any] = {
        "summary": summary_rows,
        "num_metadata_rows": len(raw_rows),
        "num_run_rows": len(run_rows),
    }
    if ase_summary is not None:
        ase_path = Path(ase_summary).expanduser().resolve()
        payload["ase_summary_path"] = str(ase_path)
        payload["ase_summary_exists"] = ase_path.exists()
    with open(root / "batch_sweep_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def _run_case_worker(payload: dict[str, Any], queue) -> None:
    try:
        gpu = int(payload["gpu"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["PARSL_WORKER_RANK"] = "0"

        from alframework.builders.excited_state_builder import build_excited_state_replay_structures
        from alframework.samplers.excited_state_sampling import run_excited_state_sampling_batch

        output_dir = Path(payload["output_dir"]).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        sampler_config = dict(payload["sampler_config"])
        builder_config = dict(payload["builder_config"])
        molecule_ids = list(payload["molecule_ids"])
        molecules = build_excited_state_replay_structures(
            moleculeids=molecule_ids,
            builder_config=builder_config,
            properties_list=dict(payload["properties_list"]),
            h5_path=str(payload["h5_path"]),
            current_h5_id=int(payload["current_h5_id"]),
        )
        if len(molecules) != int(payload["batch_size"]):
            raise RuntimeError(f"Expected {payload['batch_size']} molecules, got {len(molecules)}.")
        selected_states = [int(item.get_metadata().get("excited_state")) for item in molecules]
        if len(set(selected_states)) != 1 or selected_states[0] != int(payload["fixed_state"]):
            raise RuntimeError(f"Expected fixed selected state {payload['fixed_state']}, got {selected_states}.")

        run_excited_state_sampling_batch(
            molecule_objects=molecules,
            sampler_config=sampler_config,
            model_path=str(payload["model_path"]),
            current_model_id=int(payload["current_model_id"]),
            gpus_per_node=1,
            properties_list=dict(payload["properties_list"]),
        )
        metadata_paths = sorted((output_dir / "sampling").glob("metadata-*.json"))
        if len(metadata_paths) != int(payload["batch_size"]):
            raise RuntimeError(
                f"Expected {payload['batch_size']} metadata files, found {len(metadata_paths)} in {output_dir / 'sampling'}."
            )
        for metadata_path in metadata_paths:
            data = load_json(metadata_path)
            timing = dict(data.get("timing") or {})
            if int(timing.get("batch_size", -1)) != int(payload["batch_size"]):
                raise RuntimeError(f"{metadata_path} has timing.batch_size={timing.get('batch_size')}.")
        queue.put({"ok": True, "batch_size": int(payload["batch_size"]), "repeat": int(payload["repeat"])})
    except Exception as exc:
        queue.put(
            {
                "ok": False,
                "batch_size": int(payload.get("batch_size", -1)),
                "repeat": int(payload.get("repeat", -1)),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def run_sweep(args: argparse.Namespace) -> dict[str, Any] | None:
    run_dir = Path(args.run_dir).expanduser().resolve()
    master = load_json(run_dir / "master_config_3node_darwin_test.json")
    builder_config = load_json(run_dir / "builder_config.json")
    sampler_config = load_json(run_dir / "sampler_config.json")
    batch_sizes = parse_int_list(args.batch_sizes)
    gpus = parse_int_list(args.gpus)
    if any(size <= 0 for size in batch_sizes):
        raise ValueError("All batch sizes must be positive.")
    max_batch = max(batch_sizes)
    molecule_ids = sweep_molecule_ids(max_batch)
    h5_path = resolve_run_path(run_dir, str(master["h5_path"]))
    model_path = resolve_run_path(run_dir, str(master["model_path"]))
    model_root = Path(model_path.format(int(args.current_model_id)))
    h5_file = Path(h5_path.format(int(args.current_h5_id) - 1))
    if not model_root.exists():
        raise FileNotFoundError(f"Missing model root: {model_root}")
    if not h5_file.exists():
        raise FileNotFoundError(f"Missing HDF5 shard: {h5_file}")

    result_root = (run_dir / args.output_dir).expanduser().resolve()
    waves = make_waves(batch_sizes, gpus)
    if args.dry_run:
        print(f"run_dir: {run_dir}")
        print(f"model: {model_root}")
        print(f"h5: {h5_file}")
        print(f"output: {result_root}")
        for repeat in range(int(args.repeats)):
            for wave_index, wave in enumerate(waves):
                print(f"repeat {repeat} wave {wave_index}: {wave}")
        return None

    ctx = mp.get_context("spawn")
    for repeat in range(int(args.repeats)):
        for wave_index, wave in enumerate(waves):
            print(f"Starting repeat {repeat} wave {wave_index}: {wave}", flush=True)
            queue = ctx.Queue()
            processes = []
            for batch_size, gpu in wave:
                output_dir = result_root / f"batch_{int(batch_size):04d}" / f"repeat_{int(repeat):02d}"
                payload = {
                    "gpu": int(gpu),
                    "batch_size": int(batch_size),
                    "repeat": int(repeat),
                    "fixed_state": int(args.fixed_state),
                    "current_h5_id": int(args.current_h5_id),
                    "current_model_id": int(args.current_model_id),
                    "h5_path": h5_path,
                    "model_path": model_path,
                    "properties_list": dict(master["properties_list"]),
                    "builder_config": prepare_builder_config(builder_config, fixed_state=int(args.fixed_state)),
                    "sampler_config": prepare_sampler_config(
                        sampler_config,
                        batch_size=int(batch_size),
                        output_dir=output_dir,
                        maxt=float(args.maxt),
                        ncheck=int(args.ncheck),
                        fixed_state=int(args.fixed_state),
                    ),
                    "molecule_ids": molecule_ids[: int(batch_size)],
                    "output_dir": str(output_dir),
                }
                process = ctx.Process(target=_run_case_worker, args=(payload, queue))
                process.start()
                processes.append(process)
            for process in processes:
                process.join()
            results = []
            while not queue.empty():
                results.append(queue.get())
            failures = [result for result in results if not result.get("ok")]
            bad_exits = [process.exitcode for process in processes if process.exitcode not in (0, None)]
            if len(results) != len(processes):
                failures.append(
                    {
                        "ok": False,
                        "error": f"Expected {len(processes)} worker results, received {len(results)}.",
                    }
                )
            if failures or bad_exits:
                raise RuntimeError(f"Sweep wave failed. failures={failures}, exitcodes={bad_exits}")
            print(f"Finished repeat {repeat} wave {wave_index}: {results}", flush=True)
    return summarize_sweep_results(result_root, ase_summary=args.ase_summary)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an ALCHEMI BAOAB batch-size sampling timing sweep.")
    parser.add_argument("--run-dir", default=".", help="Example directory containing configs, model-0000, and data-0000.")
    parser.add_argument("--batch-sizes", default=",".join(str(item) for item in DEFAULT_BATCH_SIZES))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--maxt", type=float, default=0.2)
    parser.add_argument("--ncheck", type=int, default=25)
    parser.add_argument("--fixed-state", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--current-h5-id", type=int, default=1)
    parser.add_argument("--current-model-id", type=int, default=0)
    parser.add_argument("--output-dir", default="sweep_results")
    parser.add_argument("--ase-summary", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_sweep(args)
    if summary is not None:
        print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
