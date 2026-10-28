"""Record physical observations, then replay cohort selection and restart.

Observation recording can use a real checkpoint on CUDA/ROCm. Replaying those
identical records isolates scientific selection from physical task width and
completion order. This script does not run MD, QM, or training; use benchmark.py
and thermal_acceptance.py for engine mechanics and throughput.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np
from ase.io import read

from examples.portable_batched_sampling.benchmark import (
    environment, execution_model, fingerprint, load_checkpoint, source_revision,
    synthetic_model, synthetic_structures,
)
from alframework.samplers.sampling import candidate_score, screen_frame, uncertainty_flags
from alframework.tools.acquisition_cohorts import AcquisitionCohorts
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.sampling_validation import configured_score


def _ids(molecules):
    return [row.get_metadata()["acquisition_trajectory_id"] for row in molecules]


def _selection(molecules):
    return [{"candidate_id": row.get_moleculeid(),
             "trajectory_id": row.get_metadata()["acquisition_trajectory_id"],
             "trajectory_ordinal": row.get_metadata()["acquisition_trajectory_ordinal"],
             "step": row.get_metadata()["step"],
             "ranking_score": row.get_metadata()["ranking_score"],
             "selected_state": row.get_metadata().get("selected_state")}
            for row in molecules]


def replay(path, parents, records, config, reference, *, width, reverse, restart, namespace):
    """Use the production coordinator, not a second implementation of top-K."""
    case_config = copy.deepcopy(config)
    case_config["driver"]["strategy"] = "single" if width == 1 else "batched"
    case_config["batching"] = {"batch_size": width, "partial_policy": "timeout", "max_wait_seconds": 5.}
    pool = AcquisitionCohorts(path, cohort_size=len(parents),
                              k=config["acquisition"]["selection"]["k"], namespace=namespace)
    started = time.perf_counter()
    try:
        inputs = [pool.admit(parent, reference, case_config) for parent in parents]
        groups = [list(range(start, min(start+width, len(inputs))))
                  for start in range(0, len(inputs), width)]
        if reverse:
            groups.reverse()
        for group_index, group in enumerate(groups):
            members = [inputs[index] for index in group]
            candidates = []
            for index in group:
                for record in records[index]:
                    molecule = MoleculesObject(record["atoms"].copy(), "physical-task-rank-is-irrelevant")
                    molecule.update_metadata({**inputs[index].get_metadata(), **record["metadata"],
                                              "parent_molecule_id": inputs[index].get_moleculeid()})
                    candidates.append(molecule)
            if reverse:
                candidates.reverse()
            pool.mark_submitted(_ids(members))
            pool.accept_results(_ids(members), candidates)
            before = pool.status()
            pool.accept_results(_ids(members), candidates)
            assert pool.status() == before, "Repeated task delivery changed cohort state"
            if group_index < len(groups)-1:
                assert pool.ready_qm() == [], "An incomplete cohort selected candidates"
            if restart and group_index == (len(groups)-1)//2:
                pool.close()
                pool = AcquisitionCohorts(path, cohort_size=len(parents),
                                          k=config["acquisition"]["selection"]["k"], namespace=namespace)
        selected = pool.ready_qm()
        selection = _selection(selected)
        if selected:
            first_id = selected[0].get_moleculeid()
            pool.mark_qm_submitted(first_id, {"acceptance_fixture": True})
            pool.close()
            pool = AcquisitionCohorts(path, cohort_size=len(parents),
                                      k=config["acquisition"]["selection"]["k"], namespace=namespace)
            assert first_id not in [row.get_moleculeid() for row in pool.ready_qm()]
            assert _selection(pool.ready_qm(include_submitted=True)) == selection
        return {"physical_width": width, "reversed_arrivals": reverse,
                "driver_strategy": case_config["driver"]["strategy"],
                "restart_during_ingestion": restart and len(groups) > 1,
                "restart_after_selection": restart and len(groups) == 1,
                "coordinator_seconds": time.perf_counter()-started,
                "selection": selection, "status": pool.status(),
                "duplicate_delivery_idempotent": True,
                "qm_handoff_restart_preserved": True if selected else None}
    finally:
        pool.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--master")
    parser.add_argument("--ensemble-directory")
    parser.add_argument("--model-generation", default="acceptance-artifact")
    parser.add_argument("--structures")
    parser.add_argument("--selected-state", type=int)
    parser.add_argument("--sampling-method", choices=("mlmd", "udd"), default="mlmd")
    parser.add_argument("--udd-bias-weight", type=float)
    parser.add_argument("--strategy", choices=("uncertainty", "summed"), default="uncertainty")
    parser.add_argument("--trajectory-policy", choices=("stop", "continue"), default="continue")
    parser.add_argument("--cohort-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--physical-widths", nargs="+", type=int, default=[1, 2, 8, 32, 50])
    parser.add_argument("--recording-width", type=int, default=8)
    parser.add_argument("--energy-cutoff", type=float, default=1e-8)
    parser.add_argument("--force-cutoff", type=float, default=1e-8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.cohort_size, args.top_k, args.recording_width, *args.physical_widths) < 1:
        parser.error("Sizes, widths, and K must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in (args.energy_cutoff, args.force_cutoff)):
        parser.error("Cutoffs must be finite and positive")
    if bool(args.master) != bool(args.ensemble_directory) or (args.master and not args.structures):
        parser.error("Checkpoint acceptance requires master, ensemble directory, and structures together")
    if args.sampling_method == "udd" and (args.udd_bias_weight is None or not math.isfinite(args.udd_bias_weight)):
        parser.error("UDD requires a finite explicit --udd-bias-weight")

    import torch
    from alframework.ml_interfaces.batched_inference import PackedBatch
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("The requested GPU is unavailable")
        torch.cuda.set_device(device)
    if torch.version.hip:
        import hippynn
        hippynn.custom_kernels.set_custom_kernels(False)
    structures = read(args.structures, index=":") if args.structures else synthetic_structures()
    raw_model = load_checkpoint(args, device) if args.master else synthetic_model(device, sampling_method=args.sampling_method)
    original = getattr(args, "_configuration", {})
    model, config = execution_model(raw_model, driver="batched", engine="torchsim",
        sampling_method=args.sampling_method, udd_bias_weight=args.udd_bias_weight,
        configuration=original.get("sampler_config"),
        properties_list=original.get("master", {}).get("properties_list"), device=device)
    config["acquisition"] = {"strategy": args.strategy, "trajectory_policy": args.trajectory_policy,
                              "selection": {"mode": "cohort_top_k", "cohort_size": args.cohort_size, "k": args.top_k}}
    config.update(Escut=args.energy_cutoff, Fscut=args.force_cutoff, distcut=.1,
                  max_force_cutoff=1e6, min_time=0.)
    config["score"] = {"mode": "sum" if args.strategy == "summed" else "max"}
    if args.strategy == "summed":
        config["score"].update(w_energy=1., w_force=1., w_gap_uncertainty=0.)
    selected_state = getattr(raw_model, "selected_state_value", args.selected_state)
    if config.get("model_mode", "ground_state") == "excited_state":
        config["state_selection"] = {"mode": "fixed", "state": int(selected_state)}
    score = configured_score(config, original.get("master", {}).get("properties_list"), selected_state)
    parents = [MoleculesObject(structures[index % len(structures)].copy(), f"cohort-acceptance-{index:06d}")
               for index in range(args.cohort_size)]
    records = [[] for _ in parents]
    recording_started = time.perf_counter()
    for step, scale in ((1, 1.), (2, 1.0001)):
        for start in range(0, len(parents), args.recording_width):
            atoms_list = []
            for parent in parents[start:start+args.recording_width]:
                atoms = parent.get_atoms().copy()
                atoms.positions *= scale
                atoms_list.append(atoms)
            model(PackedBatch.from_atoms(atoms_list, device=device))
            for offset, atoms in enumerate(atoms_list):
                index = start + offset
                if args.trajectory_policy == "stop" and records[index]:
                    continue
                diagnostic = copy.deepcopy(model.diagnostics_for_graph(offset))
                flags = uncertainty_flags(diagnostic, Escut=args.energy_cutoff, Fscut=args.force_cutoff)
                _, rejection = screen_frame(atoms, diagnostic, config)
                if rejection is not None or not flags["uncertain"]:
                    continue
                ranking = candidate_score(diagnostic, flags, score,
                                          Escut=args.energy_cutoff, Fscut=args.force_cutoff)
                metadata = {"step": step, "selected_state": selected_state,
                            "Es": float(diagnostic["Es"]), "Fs": float(diagnostic["Fs"]),
                            "Fsmax": float(diagnostic["Fsmax"]), **ranking}
                records[index].append({"atoms": atoms.copy(), "metadata": metadata})
    recording_seconds = time.perf_counter()-recording_started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    journal_root = Path(tempfile.mkdtemp(prefix=args.output.stem+"-journals-", dir=args.output.parent))
    reference = getattr(args, "_model_reference", {"generation": "analytic-acceptance",
        "artifact_location": "analytic-acceptance-fixture"})
    record_fingerprint = hashlib.sha256(json.dumps({"model_reference": reference,
        "configuration": config, "records": [
        [{"metadata": row["metadata"], "numbers": row["atoms"].numbers.tolist(),
          "positions": row["atoms"].positions.tolist(), "cell": row["atoms"].cell.array.tolist(),
          "pbc": row["atoms"].pbc.tolist()} for row in trajectory] for trajectory in records
    ]}, sort_keys=True).encode()).hexdigest()
    namespace = "acceptance-" + record_fingerprint
    cases = []
    for width in args.physical_widths:
        for reverse, restart in ((False, False), (True, True)):
            path = journal_root / f"width-{width}-reverse-{int(reverse)}.sqlite"
            cases.append(replay(path, parents, records, config, reference, width=width,
                                reverse=reverse, restart=restart, namespace=namespace))
    expected = cases[0]["selection"]
    assert all(case["selection"] == expected for case in cases), "Selection changed with physical grouping or restart"
    result = {"schema": "alf.logical_cohort_acceptance.v1", "passed": True,
              "source": source_revision(), "environment": environment(torch, device),
              "hardware_validation": device.type == "cuda", "dynamics_executed": False,
              "validation_scope": ["physical_inference", "sampling_force_pipeline", "eligibility_and_ranking",
                                   "cohort_replay", "restart_and_qm_handoff"],
              "configuration": config, "recording_seconds": recording_seconds,
              "model_reference": reference,
              "recorded_candidates": sum(map(len, records)), "observation_sha256": record_fingerprint,
              "structures": fingerprint(args.structures) if args.structures else None,
              "journal_directory": str(journal_root.resolve()), "cases": cases}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(f"Wrote {args.output}: {len(cases)} invariant cohort replay cases; {len(expected)} selected")
    return result


if __name__ == "__main__":
    main()
