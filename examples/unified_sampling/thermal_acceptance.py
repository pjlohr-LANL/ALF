"""Check thermostat statistics and independent temperature changes analytically.

This is a harmonic integration fixture, not a trained-potential stability run.
ASE execution measures independent singleton trajectories in sequence. Batched
execution advances the same collection through TorchSim or ALCHEMI.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from ase import Atoms, units

from examples.portable_batched_sampling.benchmark import (
    RunnerGroup, environment, source_revision, synthetic_model,
)


def temperature_statistics(values, target, relative_tolerance):
    """Estimate the mean with a lag-one-adjusted block standard error."""
    series = np.asarray(values, dtype=float)
    count = len(series)
    deviation = float(np.std(series, ddof=1))
    correlation = 0.
    if deviation > 0 and np.std(series[:-1]) > 0 and np.std(series[1:]) > 0:
        correlation = float(np.clip(np.corrcoef(series[:-1], series[1:])[0, 1], 0., .99))
    effective = max(1., min(float(count), count * (1-correlation) / (1+correlation)))
    standard_error = deviation / math.sqrt(effective)
    measured = float(np.mean(series))
    allowed = max(relative_tolerance * target, 4 * standard_error)
    return {
        "target_K": float(target), "mean_K": measured,
        "relative_error": abs(measured-target) / target,
        "block_standard_error_K": standard_error,
        "lag_one_correlation": correlation, "effective_blocks": effective,
        "allowed_error_K": allowed, "passed": bool(abs(measured-target) <= allowed),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", choices=("ase", "batched"), default="batched")
    parser.add_argument("--backend", choices=("torchsim", "alchemi"))
    parser.add_argument("--sampling-method", choices=("mlmd", "udd"), default="mlmd")
    parser.add_argument("--udd-bias-weight", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--systems", type=int, default=8)
    parser.add_argument("--atoms-per-system", type=int, default=32)
    parser.add_argument("--temperatures-K", nargs=2, type=float, default=[200., 400.])
    parser.add_argument("--dt-fs", type=float, default=2.)
    parser.add_argument("--friction-per-fs", type=float, default=.02)
    parser.add_argument("--equilibration-steps", type=int, default=500)
    parser.add_argument("--sample-blocks", type=int, default=20)
    parser.add_argument("--steps-per-block", type=int, default=25)
    parser.add_argument("--relative-tolerance", type=float, default=.08)
    parser.add_argument("--seed", type=int, default=91423)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.driver == "batched" and args.backend is None:
        parser.error("Batched execution requires --backend")
    if args.sampling_method == "udd" and args.udd_bias_weight is None:
        parser.error("UDD requires --udd-bias-weight")
    positive = [*args.temperatures_K, args.dt_fs, args.friction_per_fs, args.relative_tolerance]
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        parser.error("Temperatures, timestep, friction, and relative tolerance must be finite and positive")
    if args.temperatures_K[0] == args.temperatures_K[1]:
        parser.error("Choose two distinct targets to test independent temperature changes")
    if args.systems < 2 or args.systems % 2 or args.atoms_per_system < 1:
        parser.error("Use an even system count of at least two and at least one atom per system")
    if args.equilibration_steps < 1 or args.sample_blocks < 8 or args.steps_per_block < 1:
        parser.error("Equilibration/stride must be positive and at least eight sample blocks are required")
    equilibration_span = args.equilibration_steps * args.dt_fs * args.friction_per_fs
    sampling_span = args.sample_blocks * args.steps_per_block * args.dt_fs * args.friction_per_fs
    if min(equilibration_span, sampling_span) < 10:
        parser.error("Equilibration and sampling must each span at least ten friction relaxation times")
    if args.udd_bias_weight is not None and (not math.isfinite(args.udd_bias_weight) or
                                             1 + .1 * args.udd_bias_weight <= 0):
        parser.error("The analytic UDD fixture requires a finite weight with 1 + 0.1 * weight > 0")

    import torch
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("The requested GPU is unavailable; select --device cpu explicitly for a CPU check")
        torch.cuda.set_device(device)
    if args.driver == "batched" and args.backend == "alchemi" and torch.version.hip:
        parser.error("ALCHEMI requires NVIDIA; use TorchSim on ROCm")
    if args.driver == "ase":
        from alframework.samplers.ase_sampling import ASEDynamicsRunner as Runner
    elif args.backend == "alchemi":
        from alframework.samplers.alchemi_sampling import AlchemiDynamicsRunner as Runner
    else:
        from alframework.samplers.torchsim_sampling import TorchSimDynamicsRunner as Runner
    model = synthetic_model(device, sampling_method=args.sampling_method)
    if args.sampling_method == "udd":
        model.configure_sampling_method("udd", {
            "udd_bias_weight": args.udd_bias_weight, "udd_bias_eps": 1e-12,
        })
    structures = []
    for _ in range(args.systems):
        atoms = Atoms("Ar" * args.atoms_per_system,
                      positions=np.zeros((args.atoms_per_system, 3)), cell=np.eye(3) * 10.)
        atoms.set_velocities(np.zeros((len(atoms), 3)))
        structures.append(atoms)
    targets = np.tile(np.asarray(args.temperatures_K), args.systems // 2)
    runner = RunnerGroup(Runner, model, structures, driver=args.driver, dt_fs=args.dt_fs,
                         temperature_K=targets, friction_per_fs=args.friction_per_fs,
                         random_seed=args.seed, device=device)
    initial = structures[0].copy()
    runner.sync_graph_to_atoms(0, initial)
    np.testing.assert_array_equal(initial.get_velocities(), 0.)
    start = time.perf_counter()
    phases = []
    for phase in range(2):
        current_targets = targets if phase == 0 else targets[::-1].copy()
        runner.set_temperature(current_targets)
        runner.run(args.equilibration_steps)
        samples = []
        for _ in range(args.sample_blocks):
            runner.run(args.steps_per_block)
            measured = []
            for index, atoms in enumerate(structures):
                runner.sync_graph_to_atoms(index, atoms)
                measured.append(2 * atoms.get_kinetic_energy() / (3 * len(atoms) * units.kB))
            samples.append(measured)
        samples = np.asarray(samples)
        groups = []
        for target in sorted(set(current_targets)):
            selected = current_targets == target
            statistics = temperature_statistics(samples[:, selected].mean(axis=1),
                                                target, args.relative_tolerance)
            statistics["systems"] = np.where(selected)[0].tolist()
            groups.append(statistics)
        phases.append({"phase": phase, "groups": groups,
                       "per_system_mean_K": samples.mean(axis=0).tolist()})
    result = {
        "schema": "alf.unified_thermal_acceptance.v1",
        "hardware_validation": device.type == "cuda",
        "execution_driver": args.driver,
        "backend": args.backend if args.driver == "batched" else "ase",
        "sampling_method": args.sampling_method,
        "model": "analytic_harmonic_ensemble", "source": source_revision(),
        "environment": environment(torch, device),
        "parameters": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items() if key != "output"},
        "acceptance_rule": "absolute mean-temperature error <= max(relative_tolerance * target, 4 * lag-one-adjusted block SEM)",
        "initial_velocities_preserved": True,
        "elapsed_seconds": time.perf_counter() - start, "phases": phases,
        "passed": all(group["passed"] for phase in phases for group in phase["groups"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}: thermal acceptance {'passed' if result['passed'] else 'failed'}")
    if not result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
