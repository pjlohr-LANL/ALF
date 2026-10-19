"""Regressions for physical eligibility, independently of propagation."""
import json

import numpy as np
import pytest
import torch
from ase.calculators.calculator import Calculator

from alframework.samplers.alchemi_sampling import (
    ALFAlchemiCalculator, ALFASEAlchemiModel, ALFNativeEnsembleModel,
    run_alchemi_sampling,
)
from alframework.tools.force_screen import SCHEMA, SOURCE, physical_force_exceeds
from tests.test_alchemi_sampling import (
    _torch_batch, _gap_row, _molecule, _config, FakeModel, FakeRunner,
    _topology_molecule, _reference_topology, _enable_topology, TopologyMutationRunner,
)


class States(ALFAlchemiCalculator):
    upper = 5.0
    missing = False

    def ensemble_forward(self, batch):
        states = {state: {"energy_contributions": torch.full((2, 2), state * .02),
                          "force_contributions": torch.full((2, 4, 3), force)}
                  for state, force in [(0, 0.0), (1, self.upper)]}
        # Opposite member deviations cancel; screen the mean, not members.
        states[0]["force_contributions"][0] += 30
        states[0]["force_contributions"][1] -= 30
        return {**states[0], "state_contributions": {} if self.missing else states}


def make_states(**kwargs):
    return States(physical_force_states=[0, 1],
                  gap_seeking_settings={"enabled": True, "rows": [_gap_row()]}, **kwargs)


def test_artificial_force_and_confinement_do_not_screen_or_change_propagation():
    model = make_states()
    model.set_lcm_mode(0, pair=[0, 1], sigma=3.5, alpha_eV=.05)
    output = model(_torch_batch(torch))
    diag = model.diagnostics_for_graph(0)
    assert diag["Fmeanmax"] == pytest.approx(np.sqrt(75))
    assert diag["propagation_Fmax"] > 16
    assert not physical_force_exceeds(diag, 16)
    assert diag["physical_F0_max"] == 0
    assert diag["physical_F1_max"] == pytest.approx(np.sqrt(75))
    expected_e, expected_f = model._lcm_energy_forces(
        torch.tensor(0.), torch.tensor(.02), torch.zeros((2, 3)),
        torch.full((2, 3), 5.), sigma=3.5, alpha_eV=.05)
    torch.testing.assert_close(output["forces"][:2], expected_f)
    assert output["energy"][0].item() == pytest.approx(expected_e.item())
    assert model.diagnostics_for_graph(1)["lcm_bias_Fmax"] == 0
    model.well_params = {"r_start": 0., "force": 100., "mass_weighted": False}
    confined = model(_torch_batch(torch, positions=np.ones((4, 3))))
    diag = model.diagnostics_for_graph(0)
    assert diag["Fmeanmax"] == pytest.approx(np.sqrt(75))
    assert diag["confinement_Fmax"] == pytest.approx(100)
    assert not physical_force_exceeds(diag, 16)
    assert not torch.equal(confined["forces"], output["forces"])


def test_nonselected_state_screens_without_lcm():
    model = make_states()
    model.upper = 20.
    output = model(_torch_batch(torch))
    assert torch.count_nonzero(output["forces"]) == 0
    assert physical_force_exceeds(model.diagnostics_for_graph(0), 16.)


@pytest.mark.parametrize("upper", [float('nan'), float('inf')])
def test_nonfinite_physical_state_fails(upper):
    model = make_states()
    model.upper = upper
    with pytest.raises(ValueError, match="finite"):
        model(_torch_batch(torch))


def test_missing_required_state_fails():
    model = make_states()
    model.missing = True
    with pytest.raises(KeyError, match="states.*1"):
        model(_torch_batch(torch))


@pytest.mark.parametrize("value,raises,reject", [
    (16., False, False), (16.00001, False, True), (15.99999, False, False),
    (None, True, False), (float('nan'), True, False), (float('inf'), True, False),
])
def test_screen_boundaries(value, raises, reject):
    diag = {"Fmeanmax": value, "force_screen_schema": SCHEMA, "force_screen_source": SOURCE}
    if raises:
        with pytest.raises(ValueError):
            physical_force_exceeds(diag, 16)
    else:
        assert physical_force_exceeds(diag, 16) is reject


def checked_diag(force):
    return {"Es": 0., "Fs": 0., "Fsmax": 0., "Fmeanmax": force,
            "force_screen_schema": SCHEMA, "force_screen_source": SOURCE,
            "F0": np.full((2, 3), force / np.sqrt(3))}


def test_audit_has_actual_rejected_frame_and_previous_valid_without_candidates(tmp_path):
    config = _config(batch_size=1)
    config.update(max_force_cutoff=16., force_screen_audit_dir="audit")
    model = FakeModel({0: [checked_diag(1.), checked_diag(20.)]})
    result = run_alchemi_sampling([_molecule('x')], config, model,
                                  runner_factory=FakeRunner, master_directory=str(tmp_path))
    assert result == []
    report = json.loads(next((tmp_path / 'audit').glob('*.json')).read_text())
    row = report['replicas'][0]
    assert set(row) == {'parent_molecule_id', 'initial', 'last_valid', 'rejected'}
    assert row['initial']['positions'][0][0] == 0
    assert row['last_valid']['positions'][0][0] == pytest.approx(.01)
    assert row['rejected']['positions'][0][0] == pytest.approx(.02)
    assert row['rejected']['Fmeanmax'] == 20
    np.testing.assert_allclose(row['rejected']['physical_forces']['F0'], 20 / np.sqrt(3))
    assert row['rejected']['step'] == 2
    assert FakeRunner.last_instance.frozen == {0}


def test_topology_rejection_audit_survives_zero_candidates(tmp_path, monkeypatch):
    import alframework.samplers.alchemi_sampling as mod
    monkeypatch.setattr(mod, 'load_reference_topology', lambda *a, **k: _reference_topology())
    config = _enable_topology(_config(batch_size=1))
    config['force_screen_audit_dir'] = str(tmp_path)
    assert run_alchemi_sampling([_topology_molecule('bad', [[0,0,0],[.5,0,0],[2,0,0]])],
                               config, FakeModel({}), runner_factory=FakeRunner) == []
    row = json.loads(next(tmp_path.glob('*.json')).read_text())['replicas'][0]
    assert row['rejected']['positions'] == row['initial']['positions']
    assert row['rejected']['reason']


@pytest.mark.parametrize('fault', ['missing', 'shape', 'nonfinite'])
def test_ase_requires_all_configured_state_forces(fault):
    class Multi(Calculator):
        def calculate(self, atoms, properties, system_changes):
            self.results = {'sE0': 0., 'F0': np.zeros((2, 3)), 'sE1': .02}
            if fault != 'missing':
                self.results['F1'] = np.zeros((3, 2)) if fault == 'shape' else np.full((2, 3), np.nan)
    model = ALFASEAlchemiModel(Multi(), model_mode='excited_state', selected_state_value=0,
                              physical_force_states=[0,1])
    with pytest.raises((KeyError, ValueError)):
        model(_torch_batch(torch))


def test_transposed_native_physical_forces_fail():
    class Wrong(States):
        def ensemble_forward(self, batch):
            values = super().ensemble_forward(batch)
            values['state_contributions'][1]['force_contributions'] = torch.zeros((2,3,4))
            return values
    with pytest.raises(ValueError, match='shape'):
        Wrong(physical_force_states=[0,1])(_torch_batch(torch))


@pytest.mark.parametrize('fault', [None, 'missing', 'shape', 'nonfinite'])
def test_native_adapter_all_state_screen(fault):
    from nvalchemi.models.base import BaseModelMixin, ModelConfig

    class Member(torch.nn.Module, BaseModelMixin):
        def __init__(self):
            super().__init__()
            self.model_config = ModelConfig(outputs=frozenset({'energy','forces'}),
                autograd_outputs=frozenset(), autograd_inputs=frozenset(),
                active_outputs={'energy','forces'})

        @property
        def embedding_shapes(self):
            return {}

        def compute_embeddings(self, data, **kwargs):
            return data

        def direct_derivative_keys(self):
            return {'forces'}

        def forward(self, batch):
            result = {'energy': torch.zeros((2,1)), 'forces': torch.zeros((4,3))}
            if fault != 'missing':
                force = torch.full((4,3), 20.)
                if fault == 'shape':
                    force = force.T
                if fault == 'nonfinite':
                    force[0,0] = float('nan')
                result['state_contributions'] = {1: {'energy': torch.ones((2,1)), 'forces': force}}
            return result
    model = ALFNativeEnsembleModel([Member()], physical_force_states=[0,1])
    if fault:
        with pytest.raises((KeyError, ValueError)):
            model(_torch_batch(torch))
    else:
        out = model(_torch_batch(torch))
        assert torch.count_nonzero(out['forces']) == 0
        assert physical_force_exceeds(model.diagnostics_for_graph(0), 16)
