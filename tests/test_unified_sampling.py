"""Scientific method/acquisition conformance and non-MD extension contracts."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from alframework.ml_interfaces.sampling_inference import NumpyASEEnsembleModel, evaluate_model
from alframework.samplers.sampling import run_sampling
from alframework.samplers.sampling import sampling_device
from alframework.samplers.sampling_methods import SamplingObservation, AcquisitionDecision
from alframework.tools.molecules_class import MoleculesObject


class HarmonicMember(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, strength):
        super().__init__()
        self.strength = strength

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": .5 * self.strength * (atoms.positions**2).sum(),
                        "forces": -self.strength * atoms.positions.copy()}


def ensemble_loader(directory, **kwargs):
    return [HarmonicMember(.8), HarmonicMember(1.2)]


def model():
    return NumpyASEEnsembleModel(ensemble_loader(""))


def molecules():
    structures = [Atoms("H2", positions=[[.2, .1, .1], [1., .1, .2]]),
                  Atoms("H2O", positions=[[.1, .2, .2], [1., .1, .2], [.1, 1., .2]])]
    return [MoleculesObject(atoms, f"replica-{i}") for i, atoms in enumerate(structures)]


def config(driver="ase", method="mlmd", policy="stop"):
    return {
        "sampling": {"method": method}, "acquisition": {"method": "uncertainty", "policy": policy},
        "dynamics": {"driver": driver, "backend": "torchsim", "random_seed": 7},
        "inference": {"interface": "ase", "loader": __name__ + ".ensemble_loader"},
        "batching": {"batch_size": 2}, "dt": .1, "maxt": .0003, "Ncheck": 1,
        "Escut": .001, "Fscut": .001, "distcut": .1, "friction_per_fs": 0.,
        "srt_temp": [0., 0.], "end_temp": [0., 0.], "amp_temp": [0., 0.], "per_temp": [1., 1.],
        "return_top_k": 3, "udd_bias_weight": .45,
    }


class StaticRunner:
    """Present the same frames to every execution path, removing stochastic drift."""
    def __init__(self, *, model, atoms_list, device=None, **kwargs):
        self.model, self.atoms, self.device = model, [a.copy() for a in atoms_list], device
        self.nsteps = 0
        self.evaluate()

    def evaluate(self):
        evaluate_model(self.model, self.atoms, device=self.device)

    def run(self, count):
        self.nsteps += count

    def sync_graph_to_atoms(self, index, atoms):
        atoms.set_positions(self.atoms[index].positions)
        atoms.set_velocities(self.atoms[index].get_velocities())

    def diagnostics_for_graph(self, index):
        return self.model.diagnostics_for_graph(index)

    def freeze_graph(self, index):
        pass

    def set_temperature(self, values):
        pass


@pytest.mark.parametrize("method", ["mlmd", "udd"])
def test_driver_choice_preserves_acquisition_and_unbiased_screening(method):
    cfg = config("batched", method)
    originals = molecules()
    grouped = run_sampling(originals, cfg, model(), runner_factory=StaticRunner, device="cpu")
    individual = []
    cfg["dynamics"]["driver"] = "ase"
    before = deepcopy(cfg)
    for molecule in originals:
        individual.extend(run_sampling([molecule], cfg, model(), runner_factory=StaticRunner, device="cpu"))
    assert cfg == before
    assert len(grouped) == len(individual) == 2
    for batch, single in zip(grouped, individual):
        a, b = batch.get_metadata(), single.get_metadata()
        for key in ("Es", "Fs", "Fsmax", "ranking_score", "Fmeanmax", "uncertainty_reasons"):
            assert a[key] == b[key]
        assert a["sampling_method"] == b["sampling_method"] == method
        assert a["execution_driver"] == "batched" and b["execution_driver"] == "ase"
        if method == "udd":
            assert a["E_en_bias"] == b["E_en_bias"] > 0
            assert a["udd_bias_Fmax"] == b["udd_bias_Fmax"] > 0
    for driver in ("ase", "batched"):
        reject = config(driver, method)
        reject["max_force_cutoff"] = .01
        assert run_sampling(originals[:1] if driver == "ase" else originals,
                            reject, model(), runner_factory=StaticRunner, device="cpu") == []


def test_udd_pipeline_uses_physical_forces_for_eligibility():
    cfg = config(method="udd")
    cfg.update(udd_bias_weight=50., max_force_cutoff=2.)
    result = run_sampling(molecules()[:1], cfg, model(), device="cpu")
    assert len(result) == 1
    record = result[0].get_metadata()
    assert record["Fmeanmax"] < 2. < record["propagation_Fmax"]
    assert record["sampling_method"] == "udd"


def test_ase_accepts_partial_periodic_cells_without_a_nonperiodic_cell_vector():
    molecule = molecules()[0]
    molecule.get_atoms().set_cell([[4., 0., 0.], [0., 4., 0.], [0., 0., 0.]])
    molecule.get_atoms().set_pbc([True, True, False])
    output = run_sampling([molecule], config(), model(), device="cpu")
    assert len(output) == 1
    assert output[0].get_atoms().pbc.tolist() == [True, True, False]
    with pytest.raises(NotImplementedError, match="fully periodic"):
        run_sampling([molecule], config("batched"), model(), device="cpu")


@pytest.mark.parametrize("driver", ["ase", "batched"])
def test_explicit_strict_gpu_is_not_silently_bypassed_by_cpu_selection(driver):
    cfg = config(driver)
    cfg["dynamics"].update(device="cpu", strict_gpu=True)
    with pytest.raises(RuntimeError, match="strict_gpu"):
        sampling_device(cfg, 1)
    cfg["dynamics"]["allow_cpu_debug"] = True
    device, metadata = sampling_device(cfg, 1)
    assert device == metadata["sampler_device"] == "cpu"


class GroupedOptimizationMethod:
    """Example plugin using proposal groups and iterations, with no MD clock."""
    required_capabilities = frozenset({"grouped_evaluation"})

    def create_program(self, context):
        class Program:
            def run(self):
                parent = context.molecule_objects[0]
                proposals = [parent.get_atoms().copy(), parent.get_atoms().copy()]
                proposals[1].positions *= 1.5
                predictions = context.evaluate_group(proposals)
                candidates = []
                for iteration, atoms in enumerate(proposals):
                    event = context.acquire(SamplingObservation(
                        0, parent, atoms, predictions["diagnostics"][iteration],
                        iteration + 1, {"optimization_iteration": iteration + 1}))
                    if event.candidate is not None:
                        candidates.append(event.candidate)
                return context.finalize(candidates)
        return Program()


class UnsupportedOptimizer(GroupedOptimizationMethod):
    required_capabilities = frozenset({"unimplemented_optimizer"})


class StopWithoutCapture:
    def decide(self, observation, context):
        return AcquisitionDecision(capture=False, stop=True)

    def select(self, candidates, context, *, final=False):
        return candidates


class CaptureWithoutThresholds(StopWithoutCapture):
    def decide(self, observation, context):
        return AcquisitionDecision(capture=True, stop=True,
                                   metadata={"ranking_score": 42., "score_mode": "custom"})


@pytest.mark.parametrize("capture", [False, True])
def test_custom_acquisition_has_own_stopping_and_needs_no_uncertainty_thresholds(capture):
    cfg = config()
    cfg["acquisition"] = {"method": __name__ + (".CaptureWithoutThresholds" if capture else ".StopWithoutCapture"),
                          "policy": "custom-progress"}
    del cfg["Escut"], cfg["Fscut"]
    runners = []

    def factory(**kwargs):
        runner = StaticRunner(**kwargs)
        runners.append(runner)
        return runner

    output = run_sampling(molecules()[:1], cfg, model(), runner_factory=factory, device="cpu")
    assert runners[0].nsteps == 1
    assert len(output) == int(capture)
    if capture:
        assert output[0].get_metadata()["ranking_score"] == 42.
        assert output[0].get_metadata()["Escut"] is None


def test_non_md_method_uses_grouped_inference_and_shared_acquisition():
    cfg = config(policy="continue")
    cfg["sampling"]["method"] = __name__ + ".GroupedOptimizationMethod"
    cfg["return_top_k"] = 1
    for key in ("dt", "maxt", "Ncheck", "srt_temp", "end_temp", "amp_temp", "per_temp"):
        del cfg[key]
    result = run_sampling(molecules()[:1], cfg, model(), device="cpu")
    assert len(result) == 1
    record = result[0].get_metadata()
    assert record["optimization_iteration"] == 2
    assert "time_ps" not in record
    assert record["parent_molecule_id"] == "replica-0"
    cfg["max_force_cutoff"] = .001
    assert run_sampling(molecules()[:1], cfg, model(), device="cpu") == []


def test_unavailable_method_capability_errors_without_running():
    cfg = config()
    cfg["sampling"]["method"] = __name__ + ".UnsupportedOptimizer"
    with pytest.raises(ValueError, match="unimplemented_optimizer"):
        run_sampling(molecules()[:1], cfg, model(), device="cpu")


def test_complete_unified_ase_task_works_with_torch_import_blocked():
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','hippynn','torch_sim','nvalchemi','nvalchemiops','warp'}:
   raise ImportError('GPU stack blocked: '+fullname)
sys.meta_path.insert(0,Block())
from tests.test_unified_sampling import config, molecules
from alframework.samplers.sampling import sampling_task
from parsl.app.errors import RemoteExceptionWrapper
result=sampling_task.func(molecules()[:1], config(method='udd'), '/unused/model-{:04d}', 0, 1, {}, {})
if isinstance(result,RemoteExceptionWrapper): result.reraise()
assert len(result)==1
assert result[0].get_metadata()['sampling_method']=='udd'
assert 'torch' not in sys.modules
'''
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
