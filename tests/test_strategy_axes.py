"""Independent public strategy selections and driver-invariant science."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

from alframework.tools.batched_config import normalize_sampling_config, dynamics_settings
from alframework.samplers.sampling import run_sampling
from alframework.samplers.sampling_methods import SamplingContext, SamplingObservation
from alframework.samplers.acquisition import SummedAcquisition
from tests.test_unified_sampling import model, molecules, StaticRunner, ensemble_loader


def canonical_config(strategy="single", engine="ase", *, udd=False, pooled=False, summed=False):
    return {
        "driver": {"strategy": strategy, "engine": engine, "device": "cpu",
                   "random_seed": 7, "allow_cpu_debug": True},
        "sampling": {"propagator": "langevin", "biases": [{"strategy": "udd", "weight": .45}] if udd else []},
        "acquisition": {"strategy": "summed" if summed else "uncertainty", "trajectory_policy": "continue" if pooled else "stop",
                        "selection": {"mode": "cohort_top_k", "cohort_size": 50, "k": 3} if pooled else {"mode": "first_trigger"}},
        "inference": {"interface": "ase", "loader": "tests.test_strategy_axes.ensemble_loader"},
        "batching": {"batch_size": 2}, "dt": .1, "maxt": .0003, "Ncheck": 1,
        "Escut": .001, "Fscut": .001, "distcut": .1, "friction_per_fs": 0.,
        "srt_temp": [0., 0.], "end_temp": [0., 0.], "amp_temp": [0., 0.], "per_temp": [1., 1.],
    }


@pytest.mark.parametrize("engine", ["ase", "alchemi", "torchsim"])
def test_single_execution_does_not_mean_ase_only(engine):
    if engine == "torchsim" and importlib.util.find_spec("torch_sim") is None:
        pytest.skip("TorchSim has its own worker environment.")
    cfg = canonical_config(engine=engine, udd=True)
    result = run_sampling(molecules()[:1], cfg, model(), device="cpu")
    assert len(result) == 1
    metadata = result[0].get_metadata()
    assert metadata["execution_driver"] == "single"
    assert metadata["execution_backend"] == engine
    assert metadata["sampling_propagator"] == "langevin"
    assert metadata["udd_bias_weight"] == .45


@pytest.mark.parametrize("udd", [False, True])
def test_parameter_draws_and_frame_ids_ignore_physical_grouping(udd):
    inputs = molecules()
    for index, molecule in enumerate(inputs):
        molecule.update_metadata({"acquisition_trajectory_id": f"trajectory-{index}",
                                  "acquisition_trajectory_ordinal": index, "trajectory_seed": index + 123})
    batched = canonical_config("batched", "alchemi", udd=udd)
    batched.update(srt_temp=[100., 300.], end_temp=[400., 600.], amp_temp=[10., 30.], per_temp=[1., 5.])
    before = deepcopy(batched)
    original = model()
    together = run_sampling(inputs, batched, original, device="cpu", runner_factory=StaticRunner)
    assert original.udd_bias_weight is None
    assert batched == before
    single = deepcopy(batched)
    single["driver"].update(strategy="single", engine="ase")
    apart = [run_sampling([molecule], single, model(), device="cpu", runner_factory=StaticRunner)[0]
             for molecule in inputs]
    assert {m.get_moleculeid() for m in together} == {m.get_moleculeid() for m in apart}
    expected = {m.get_moleculeid(): m.get_metadata() for m in apart}
    for molecule in together:
        actual, reference = molecule.get_metadata(), expected[molecule.get_moleculeid()]
        for key in ("Tamp", "Tper", "Tsrt", "Tend", "selected_state", "step"):
            assert actual[key] == reference[key]
        assert actual["ranking_score"] == pytest.approx(reference["ranking_score"], rel=1.e-5, abs=1.e-5)


def test_summed_score_cannot_make_a_frame_eligible_without_uncertainty_trigger():
    cfg = canonical_config(summed=True)
    cfg.update(Escut=1., Fscut=1., score={"w_energy": 1., "w_force": 1., "w_gap_uncertainty": 0.})
    cfg = normalize_sampling_config(cfg)
    molecule = molecules()[0]
    context = SamplingContext([molecule], cfg, model(), SummedAcquisition())
    diagnostics = {"Es": .1, "Fs": .1, "Fsmax": .1, "Fsrms": 10.}
    decision = context.acquisition.decide(SamplingObservation(0, molecule, molecule.get_atoms(), diagnostics, 1), context)
    assert decision.metadata["ranking_score"] > 10.
    assert not decision.capture and not decision.stop


@pytest.mark.parametrize("change,match", [
    (lambda c: c["driver"].update(strategy="batched", engine="ase"), "Batched"),
    (lambda c: c["acquisition"].update(trajectory_policy="continue"), "First-trigger"),
    (lambda c: c["acquisition"].update(selection={"mode": "cohort_top_k", "k": 1}), "cohort_size"),
    (lambda c: c["acquisition"].update(selection={"mode": "cohort_top_k", "cohort_size": 3, "k": True}), "positive integer"),
    (lambda c: c.update(score={"mode": "sum"}), "Conflicting"),
])
def test_ambiguous_or_incompatible_strategy_choices_fail_before_work(change, match):
    cfg = canonical_config()
    change(cfg)
    with pytest.raises(ValueError, match=match):
        normalize_sampling_config(cfg)


def test_strategy_normalization_is_idempotent_and_keeps_physical_width_separate():
    cfg = canonical_config("batched", "alchemi", pooled=True)
    resolved = normalize_sampling_config(cfg)
    assert normalize_sampling_config(resolved) == resolved
    assert resolved["batching"]["batch_size"] == 2
    assert resolved["acquisition"]["selection"]["cohort_size"] == 50
    assert dynamics_settings(resolved)["backend"] == "alchemi"


def test_canonical_ase_task_with_udd_stays_torch_free():
    script = '''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','hippynn','torch_sim','nvalchemi','nvalchemiops','warp'}:
   raise ImportError('Optional stack blocked: '+fullname)
sys.meta_path.insert(0,Block())
from tests.test_strategy_axes import canonical_config,molecules
from alframework.samplers.sampling import sampling_task
from parsl.app.errors import RemoteExceptionWrapper
result=sampling_task.func(molecules()[:1],canonical_config(udd=True),'/opaque/model-{:04d}',0,1,{}, {})
if isinstance(result,RemoteExceptionWrapper):result.reraise()
assert len(result)==1 and result[0].get_metadata()['execution_driver']=='single'
assert 'torch' not in sys.modules
'''
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_first_trigger_batches_do_not_mix_pinned_model_generations():
    from alframework.tools.acquisition_driver import first_trigger_input
    from alframework.tools.sampler_batching import SamplerBatchBuffer, sampler_ready_groups
    config = canonical_config("batched", "alchemi")
    config["batching"].update(partial_policy="timeout", max_wait_seconds=5.)
    buffer = SamplerBatchBuffer()
    first, second = molecules()
    for molecule, generation in ((first, 0), (second, 1)):
        reference = {"generation": generation, "artifact_location": f"/opaque/{generation}",
                     "adapter": "example.load", "interface": "native", "options": {}}
        buffer.add(first_trigger_input(molecule, reference, config), config, buffered_at=10.)
    assert sampler_ready_groups(buffer, config, now=11.) == []
    groups = sampler_ready_groups(buffer, config, now=15.)
    assert len(groups) == 2 and all(len(group) == 1 for group in groups)


def test_non_md_local_pruning_preserves_global_ties_and_proposal_order(tmp_path):
    from alframework.samplers.acquisition import UncertaintyAcquisition
    from alframework.tools.acquisition_cohorts import AcquisitionCohorts
    config = normalize_sampling_config(canonical_config("batched", "alchemi", pooled=True))
    config["acquisition"]["selection"].update(cohort_size=2, k=1)
    with AcquisitionCohorts(tmp_path / "ties.sqlite", cohort_size=2, k=1) as pool:
        inputs = [pool.admit(molecule, {"generation": 0}, config) for molecule in molecules()]
        context = SamplingContext(inputs, config, model(), UncertaintyAcquisition())
        candidates = []
        # The first logical trajectory wins equal scores despite its later step.
        # JSON proposal ordering must agree between local pruning and SQLite.
        for index, step, proposal in ((0, 5, "a"), (0, 5, 'a"'), (1, 1, "a")):
            event = context.acquire(SamplingObservation(index, inputs[index], inputs[index].get_atoms(),
                {"Es": .1, "Fs": .1, "Fsmax": .1, "Fsrms": .1}, step, {"proposal_id": proposal}))
            candidates.append(event.candidate)
        selected = context.acquisition.select(candidates, context, final=True)
        output = context.finalize(selected)
        pool.accept_results([item.get_metadata()["acquisition_trajectory_id"] for item in inputs], output)
        winner = pool.ready_qm()[0].get_metadata()
        assert winner["acquisition_trajectory_ordinal"] == 0
        assert winner["step"] == 5
        assert winner["proposal_id"] == "a"
