"""Run local runner acceptance and record inference/dynamics throughput.

Execute separately in each backend's worker environment.  Real checkpoints
are selected with --master and --ensemble-directory; the default analytic
model checks engine integration and cannot establish HIPPYNN performance.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from ase import Atoms
from ase.io import read


def environment(torch, device):
    packages = {}
    for package in ("torch", "hippynn", "ase", "numpy", "h5py", "tables",
                    "torch-sim-atomistic", "nvalchemi-toolkit", "nvalchemi-toolkit-ops", "warp-lang"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    gpu = device.type == "cuda"
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if gpu else None,
        "vendor": ("amd" if torch.version.hip else "nvidia") if gpu else "cpu",
        "torch_cuda": torch.version.cuda, "torch_hip": torch.version.hip,
        "packages": packages,
    }


def source_revision():
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "working_tree_dirty": None}
    return {"revision": revision, "working_tree_dirty": dirty}


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(Path(path).resolve()), "sha256": digest.hexdigest()}


def synthetic_model(device):
    import torch
    from alframework.ml_interfaces.batched_inference import BatchedEnsembleCalculator

    class HarmonicEnsemble(BatchedEnsembleCalculator):
        def ensemble_forward(self, batch):
            energies = batch.positions.new_zeros(batch.num_graphs)
            energies.index_add_(0, batch.system_idx, .5 * batch.positions.square().sum(dim=1))
            return {
                "energy_contributions": torch.stack([energies - .01, energies + .01]),
                "force_contributions": torch.stack([-batch.positions, -batch.positions]),
            }

    return HarmonicEnsemble(selected_state_value=0, device=device)


def load_checkpoint(args, device):
    from alframework.ml_interfaces.batched_inference import load_hippynn_batched_model

    master_path = Path(args.master).resolve()
    master = json.loads(master_path.read_text())
    directory = master.get("master_directory")
    root = master_path.parent if directory in (None, "pwd") else Path(directory).expanduser()
    if not root.is_absolute():
        root = master_path.parent / root

    def config(field):
        path = Path(master[field]).expanduser()
        return json.loads((path if path.is_absolute() else root / path).read_text())

    ml_config, sampler_config = config("ML_config_path"), config("sampler_config_path")
    args._configuration = {"master": master, "ML_config": ml_config, "sampler_config": sampler_config}
    mode = sampler_config.get("model_mode", "ground_state")
    selected = args.selected_state
    if mode == "excited_state" and selected is None:
        policy = sampler_config.get("state_selection", {})
        if policy.get("mode") == "fixed":
            selected = int(policy["state"])
        elif "selected_state" in sampler_config:
            selected = int(sampler_config["selected_state"])
        else:
            raise ValueError("Specify --selected-state when benchmarking a cycling excited-state configuration.")
    model = load_hippynn_batched_model(
        str(Path(args.ensemble_directory).resolve()), model_mode=mode,
        selected_state_value=selected, ML_config=ml_config,
        properties_list=master["properties_list"], sampler_config=sampler_config,
        device=device, **dict(sampler_config.get("batched_calculator_options") or {}),
    )
    return model


def synthetic_structures(periodic=False):
    cell = np.asarray([[6., 0., 0.], [.5, 7., 0.], [.2, .3, 8.]])
    structures = [
        Atoms("H2", positions=[[.1, .2, .3], [.8, .2, .3]], cell=cell, pbc=periodic),
        Atoms("H2O", positions=[[.2, .1, .3], [.9, .2, .2], [.3, .8, .2]],
              cell=cell * 1.1, pbc=periodic),
    ]
    for atoms in structures:
        atoms.set_velocities(np.full((len(atoms), 3), .01))
    return structures


def verify_individual_predictions(model, atoms_list, device, *, rtol, atol):
    import torch
    from alframework.ml_interfaces.batched_inference import PackedBatch

    batch = PackedBatch.from_atoms(atoms_list, device=device)
    output = model(batch)
    energies = output["energy"].detach().reshape(-1).clone()
    forces = output["forces"].detach().reshape(-1, 3).clone()
    diagnostics = [copy.deepcopy(model.diagnostics_for_graph(i)) for i in range(len(atoms_list))]
    ptr = batch.batch_ptr.tolist()
    max_energy_error = max_force_error = 0.
    max_diagnostic_errors = dict.fromkeys(("Es", "Fs", "Fsmax", "Fmeanmax"), 0.)
    for index, atoms in enumerate(atoms_list):
        prediction = model(PackedBatch.from_atoms([atoms], device=device))
        single_energy = prediction["energy"].detach().reshape(-1)[0]
        single_forces = prediction["forces"].detach().reshape(-1, 3)
        torch.testing.assert_close(energies[index], single_energy, rtol=rtol, atol=atol)
        torch.testing.assert_close(forces[ptr[index]:ptr[index+1]], single_forces, rtol=rtol, atol=atol)
        single_diagnostics = model.diagnostics_for_graph(0)
        for key in ("Es", "Fs", "Fsmax", "Fmeanmax"):
            np.testing.assert_allclose(diagnostics[index][key], single_diagnostics[key], rtol=rtol, atol=atol)
            max_diagnostic_errors[key] = max(
                max_diagnostic_errors[key],
                float(np.max(np.abs(np.asarray(diagnostics[index][key]) - np.asarray(single_diagnostics[key])))),
            )
        max_energy_error = max(max_energy_error, abs(float(energies[index] - single_energy)))
        max_force_error = max(max_force_error, float((forces[ptr[index]:ptr[index+1]] - single_forces).abs().max()))
    return {"max_energy_error_eV": max_energy_error, "max_force_error_eV_per_A": max_force_error,
            "max_diagnostic_errors": max_diagnostic_errors}


def run_case(model, atoms_list, runner_type, *, device, steps, warmup, iterations, dt_fs,
             temperature_K, parity_rtol, parity_atol):
    import torch
    from alframework.ml_interfaces.batched_inference import PackedBatch

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def reset_peak():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def peak():
        return torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None

    parity = verify_individual_predictions(model, atoms_list, device, rtol=parity_rtol, atol=parity_atol)
    batch = PackedBatch.from_atoms(atoms_list, device=device)
    for _ in range(warmup):
        model(batch)
    synchronize()
    reset_peak()
    start = time.perf_counter()
    for _ in range(iterations):
        model(batch)
    synchronize()
    inference_seconds = time.perf_counter() - start
    inference_peak = peak()
    start = time.perf_counter()
    runner = runner_type(model=model, atoms_list=atoms_list, dt_fs=dt_fs,
                         temperature_K=np.full(len(atoms_list), temperature_K),
                         friction_per_fs=.01, random_seed=42, device=device)
    synchronize()
    initialization_seconds = time.perf_counter() - start
    runner.run(warmup)
    synchronize()
    reset_peak()
    start = time.perf_counter()
    runner.run(steps)
    synchronize()
    dynamics_seconds = time.perf_counter() - start
    dynamics_peak = peak()
    before = atoms_list[0].copy()
    runner.sync_graph_to_atoms(0, before)
    runner.freeze_graph(0)
    runner.set_temperature(np.full(len(atoms_list), temperature_K + 10))
    runner.run(2)
    after = atoms_list[0].copy()
    runner.sync_graph_to_atoms(0, after)
    np.testing.assert_array_equal(after.positions, before.positions)
    np.testing.assert_array_equal(after.cell.array, before.cell.array)
    np.testing.assert_allclose(after.get_velocities(), 0., atol=0.)
    density_checked = False
    if len(atoms_list) > 1 and np.all(atoms_list[-1].pbc):
        initial = atoms_list[-1].copy()
        runner.sync_graph_to_atoms(len(atoms_list)-1, initial)
        runner.set_cell(len(atoms_list)-1, initial.cell.array * 1.01)
        scaled = initial.copy()
        runner.sync_graph_to_atoms(len(atoms_list)-1, scaled)
        np.testing.assert_allclose(scaled.positions, initial.positions * 1.01, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(scaled.get_velocities(), initial.get_velocities(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(scaled.get_kinetic_energy(), initial.get_kinetic_energy(), rtol=1e-5, atol=1e-8)
        density_checked = True
    return {
        "systems": len(atoms_list), "atoms": sum(len(atoms) for atoms in atoms_list),
        "inference_iterations": iterations, "inference_seconds": inference_seconds,
        "inference_systems_per_second": len(atoms_list) * iterations / inference_seconds,
        "runner_initialization_seconds": initialization_seconds,
        "dynamics_steps": steps, "dynamics_seconds": dynamics_seconds,
        "dynamics_system_steps_per_second": len(atoms_list) * steps / dynamics_seconds,
        "inference_peak_allocated_bytes": inference_peak,
        "dynamics_peak_allocated_bytes": dynamics_peak,
        "acceptance": {"prediction_parity": parity, "independent_freeze": True,
                       "density_scaling": density_checked},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("alchemi", "torchsim"), required=True)
    parser.add_argument("--device", default="cuda:0", help="PyTorch device; cuda:0 also names an AMD ROCm GPU")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 8, 32, 50])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--inference-iterations", type=int, default=10)
    parser.add_argument("--dt-fs", type=float, default=.1)
    parser.add_argument("--temperature-K", type=float, default=20.)
    parser.add_argument("--parity-rtol", type=float, default=1e-5,
                        help="Relative tolerance for float32 batch-versus-individual acceptance (default: 1e-5)")
    parser.add_argument("--parity-atol", type=float, default=1e-5,
                        help="Absolute tolerance in each checked quantity's units (default: 1e-5)")
    parser.add_argument("--master", help="Existing ALF master configuration for a real HIPPYNN checkpoint")
    parser.add_argument("--ensemble-directory", help="Directory containing model-* HIPPYNN members")
    parser.add_argument("--selected-state", type=int)
    parser.add_argument("--structures", help="ASE-readable single/multiframe structure file")
    parser.add_argument("--periodic", action="store_true", help="Use periodic structures in analytic mode")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if bool(args.master) != bool(args.ensemble_directory):
        parser.error("--master and --ensemble-directory must be provided together")
    if args.master and not args.structures:
        parser.error("Real checkpoint acceptance requires --structures from its intended model domain")
    if min(args.batch_sizes) < 1 or args.steps < 1 or args.warmup < 0 or args.inference_iterations < 1:
        parser.error("Batch sizes, steps, and inference iterations must be positive; warmup must be nonnegative")
    if any(not math.isfinite(value) or value < 0 for value in (args.parity_rtol, args.parity_atol)):
        parser.error("Parity tolerances must be finite and nonnegative")
    import torch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("The requested GPU is unavailable; use --device cpu for an explicit CPU check")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.backend == "alchemi" and torch.version.hip:
        parser.error("ALCHEMI uses NVIDIA GPU operations; select torchsim for ROCm acceptance")
    if torch.version.hip:
        import hippynn
        hippynn.custom_kernels.set_custom_kernels(False)
    if args.backend == "alchemi":
        from alframework.samplers.alchemi_sampling import AlchemiDynamicsRunner as Runner
    else:
        from alframework.samplers.torchsim_sampling import TorchSimDynamicsRunner as Runner
    structures = read(args.structures, index=":") if args.structures else synthetic_structures(args.periodic)
    model = load_checkpoint(args, device) if args.master else synthetic_model(device)
    records = []
    for size in args.batch_sizes:
        atoms_list = [structures[index % len(structures)].copy() for index in range(size)]
        records.append(run_case(model, atoms_list, Runner, device=device, steps=args.steps,
                                warmup=args.warmup, iterations=args.inference_iterations,
                                dt_fs=args.dt_fs, temperature_K=args.temperature_K,
                                parity_rtol=args.parity_rtol, parity_atol=args.parity_atol))
    result = {
        "schema": "alf.portable_batched_acceptance.v1", "backend": args.backend,
        "model": "hippynn_checkpoint" if args.master else "synthetic_harmonic",
        "ensemble_directory": str(Path(args.ensemble_directory).resolve()) if args.ensemble_directory else None,
        "hardware_validation": device.type == "cuda", "environment": environment(torch, device),
        "source": source_revision(),
        "parameters": {"dt_fs": args.dt_fs, "temperature_K": args.temperature_K,
                       "warmup_steps": args.warmup, "selected_state": args.selected_state,
                       "inference_dtype": str(next(iter(model.parameters()), torch.empty((), dtype=torch.float32)).dtype),
                       "parity_rtol": args.parity_rtol, "parity_atol": args.parity_atol},
        "configuration": getattr(args, "_configuration", None),
        "structures": fingerprint(args.structures) if args.structures else None,
        "checkpoint_files": [fingerprint(path) for path in sorted(Path(args.ensemble_directory).glob("model-*/*.pt"))] if args.ensemble_directory else [],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}: {len(records)} batch sizes, {result['environment']['vendor']} validation")
    return result


if __name__ == "__main__":
    main()
