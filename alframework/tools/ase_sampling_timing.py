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

from alframework.tools.alchemi_batch_sweep import (
    AGGREGATE_FIELDS,
    load_json,
    parse_int_list,
    prepare_builder_config,
    resolve_run_path,
    sweep_molecule_ids,
)
from alframework.tools.sampling_timing_summary import SUMMARY_FIELDS, summarize_directory


DEFAULT_TRAJECTORY_COUNTS = [4, 8, 16, 32, 64, 128]


def prepare_ase_sampler_config(
    base_config: dict[str, Any],
    *,
    output_dir: str | Path,
    maxt: float,
    ncheck: int,
    fixed_state: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    output_root = Path(output_dir).expanduser().resolve()
    sampling_dir = output_root / "sampling"
    candidate_dir = sampling_dir / "qm_candidates"
    config["dynamics_backend"] = "ase"
    config.pop("alchemi_baoab", None)
    timing = dict(config.get("timing") or {})
    timing.update({"enabled": True, "record_chunk_timings": False})
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


def _write_rows_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    extra_fields = ["trajectory_count", "repeat", "result_dir"]
    fields = extra_fields + [field for field in SUMMARY_FIELDS if field not in extra_fields]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def summarize_ase_timing_results(result_root: str | Path) -> dict[str, Any]:
    root = Path(result_root).expanduser().resolve()
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for sampling_dir in sorted(root.glob("count_*/repeat_*/sampling")):
        repeat_dir = sampling_dir.parent
        count_dir = repeat_dir.parent
        try:
            trajectory_count = int(count_dir.name.removeprefix("count_"))
            repeat = int(repeat_dir.name.removeprefix("repeat_"))
        except ValueError:
            continue
        rows = summarize_directory(sampling_dir)
        for row in rows:
            row = dict(row)
            row["trajectory_count"] = trajectory_count
            row["repeat"] = repeat
            row["result_dir"] = str(repeat_dir.relative_to(root))
            raw_rows.append(row)

    for trajectory_count in sorted({int(row["trajectory_count"]) for row in raw_rows}):
        selected = [row for row in raw_rows if int(row["trajectory_count"]) == trajectory_count]
        repeat_values = sorted({int(row["repeat"]) for row in selected})
        aggregate: dict[str, Any] = {
            "trajectory_count": trajectory_count,
            "repeats": len(repeat_values),
            "num_metadata_rows": len(selected),
        }
        for field in AGGREGATE_FIELDS:
            values = [
                float(row[field])
                for row in selected
                if row.get(field) not in (None, "")
            ]
            if values:
                aggregate[f"{field}_mean"] = float(statistics.mean(values))
                aggregate[f"{field}_stdev"] = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
                aggregate[f"{field}_sum"] = float(sum(values))
        summary_rows.append(aggregate)

    _write_rows_csv(raw_rows, root / "ase_sampling_raw.csv")
    payload = {
        "summary": summary_rows,
        "num_metadata_rows": len(raw_rows),
    }
    with open(root / "ase_sampling_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def _run_sampling_worker(payload: dict[str, Any], queue) -> None:
    try:
        worker_index = int(payload["worker_index"])
        gpus = [int(item) for item in payload.get("gpus", [])]
        if gpus and int(payload.get("gpus_per_node", 0)) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[worker_index % len(gpus)])
            os.environ["PARSL_WORKER_RANK"] = "0"
            gpus_per_node = 1
        else:
            os.environ["PARSL_WORKER_RANK"] = str(worker_index)
            gpus_per_node = int(payload.get("gpus_per_node", 0))

        from alframework.builders.excited_state_builder import build_excited_state_replay_structures
        from alframework.samplers.excited_state_sampling import run_excited_state_sampling

        molecules = build_excited_state_replay_structures(
            moleculeids=[str(payload["molecule_id"])],
            builder_config=dict(payload["builder_config"]),
            properties_list=dict(payload["properties_list"]),
            h5_path=str(payload["h5_path"]),
            current_h5_id=int(payload["current_h5_id"]),
        )
        if len(molecules) != 1:
            raise RuntimeError(f"Expected one molecule, got {len(molecules)}.")
        selected_state = int(molecules[0].get_metadata().get("excited_state"))
        if selected_state != int(payload["fixed_state"]):
            raise RuntimeError(f"Expected fixed selected state {payload['fixed_state']}, got {selected_state}.")

        run_excited_state_sampling(
            molecule_object=molecules[0],
            sampler_config=dict(payload["sampler_config"]),
            model_path=str(payload["model_path"]),
            current_model_id=int(payload["current_model_id"]),
            gpus_per_node=gpus_per_node,
            properties_list=dict(payload["properties_list"]),
        )
        queue.put({"ok": True, "molecule_id": str(payload["molecule_id"])})
    except Exception as exc:
        queue.put(
            {
                "ok": False,
                "molecule_id": str(payload.get("molecule_id", "")),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _run_payloads(payloads: list[dict[str, Any]], *, workers: int) -> None:
    if int(workers) <= 1:
        for payload in payloads:
            queue = mp.Queue()
            _run_sampling_worker(payload, queue)
            result = queue.get()
            if not result.get("ok"):
                raise RuntimeError(f"ASE sampling failed: {result}")
        return

    ctx = mp.get_context("spawn")
    for start in range(0, len(payloads), int(workers)):
        chunk = payloads[start : start + int(workers)]
        queue = ctx.Queue()
        processes = []
        for index, payload in enumerate(chunk):
            payload = dict(payload)
            payload["worker_index"] = index
            process = ctx.Process(target=_run_sampling_worker, args=(payload, queue))
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
            failures.append({"ok": False, "error": f"Expected {len(processes)} results, got {len(results)}."})
        if failures or bad_exits:
            raise RuntimeError(f"ASE sampling workers failed. failures={failures}, exitcodes={bad_exits}")


def run_ase_timing(args: argparse.Namespace) -> dict[str, Any] | None:
    run_dir = Path(args.run_dir).expanduser().resolve()
    master = load_json(run_dir / "master_config_3node_darwin_test.json")
    builder_config = load_json(run_dir / "builder_config.json")
    sampler_config = load_json(run_dir / "sampler_config.json")
    trajectory_counts = parse_int_list(args.trajectory_counts)
    gpus = parse_int_list(args.gpus) if str(args.gpus).strip() else []
    if any(count <= 0 for count in trajectory_counts):
        raise ValueError("All trajectory counts must be positive.")

    molecule_ids = sweep_molecule_ids(max(trajectory_counts))
    h5_path = resolve_run_path(run_dir, str(master["h5_path"]))
    model_path = resolve_run_path(run_dir, str(master["model_path"]))
    model_root = Path(model_path.format(int(args.current_model_id)))
    h5_file = Path(h5_path.format(int(args.current_h5_id) - 1))
    if not model_root.exists():
        raise FileNotFoundError(f"Missing model root: {model_root}")
    if not h5_file.exists():
        raise FileNotFoundError(f"Missing HDF5 shard: {h5_file}")

    result_root = (run_dir / args.output_dir).expanduser().resolve()
    if args.dry_run:
        print(f"run_dir: {run_dir}")
        print(f"model: {model_root}")
        print(f"h5: {h5_file}")
        print(f"output: {result_root}")
        print(f"trajectory_counts: {trajectory_counts}")
        print(f"workers: {int(args.workers)}")
        print(f"gpus_per_node: {int(args.gpus_per_node)}")
        print(f"gpus: {gpus}")
        for repeat in range(int(args.repeats)):
            for count in trajectory_counts:
                print(f"repeat {repeat} count {count}: {molecule_ids[:count]}")
        return None

    for repeat in range(int(args.repeats)):
        for count in trajectory_counts:
            output_dir = result_root / f"count_{int(count):04d}" / f"repeat_{int(repeat):02d}"
            sampler = prepare_ase_sampler_config(
                sampler_config,
                output_dir=output_dir,
                maxt=float(args.maxt),
                ncheck=int(args.ncheck),
                fixed_state=int(args.fixed_state),
            )
            builder = prepare_builder_config(builder_config, fixed_state=int(args.fixed_state))
            payloads = []
            for index, molecule_id in enumerate(molecule_ids[: int(count)]):
                payloads.append(
                    {
                        "worker_index": index,
                        "molecule_id": molecule_id,
                        "repeat": int(repeat),
                        "trajectory_count": int(count),
                        "fixed_state": int(args.fixed_state),
                        "current_h5_id": int(args.current_h5_id),
                        "current_model_id": int(args.current_model_id),
                        "h5_path": h5_path,
                        "model_path": model_path,
                        "properties_list": dict(master["properties_list"]),
                        "builder_config": builder,
                        "sampler_config": sampler,
                        "gpus_per_node": int(args.gpus_per_node),
                        "gpus": gpus,
                    }
                )
            print(f"Starting ASE timing repeat {repeat} count {count} with {len(payloads)} trajectories.", flush=True)
            _run_payloads(payloads, workers=int(args.workers))
            metadata_paths = sorted((output_dir / "sampling").glob("metadata-*.json"))
            if len(metadata_paths) != int(count):
                raise RuntimeError(
                    f"Expected {count} metadata files, found {len(metadata_paths)} in {output_dir / 'sampling'}."
                )
            for metadata_path in metadata_paths:
                data = load_json(metadata_path)
                timing = dict(data.get("timing") or {})
                if timing.get("backend") != "ase":
                    raise RuntimeError(f"{metadata_path} has timing.backend={timing.get('backend')}.")
            print(f"Finished ASE timing repeat {repeat} count {count}.", flush=True)

    return summarize_ase_timing_results(result_root)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ASE excited-state sampling timing without a full ALF loop.")
    parser.add_argument("--run-dir", default=".", help="Example directory containing configs, model-0000, and data-0000.")
    parser.add_argument("--trajectory-counts", default=",".join(str(item) for item in DEFAULT_TRAJECTORY_COUNTS))
    parser.add_argument("--workers", type=int, default=1, help="Concurrent ASE sampler processes. Default is serial.")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids for model evaluation; ignored with --gpus-per-node 0.")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="Use 1 for GPU model eval plus ASE CPU dynamics, 0 for CPU-only.")
    parser.add_argument("--maxt", type=float, default=0.2)
    parser.add_argument("--ncheck", type=int, default=25)
    parser.add_argument("--fixed-state", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--current-h5-id", type=int, default=1)
    parser.add_argument("--current-model-id", type=int, default=0)
    parser.add_argument("--output-dir", default="ase_sampling_results")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_ase_timing(args)
    if summary is not None:
        print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
