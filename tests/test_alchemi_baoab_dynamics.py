import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nvalchemi")

from alframework.samplers.alchemi_baoab_dynamics import ALFExcitedStateAlchemiModel


class FakeNode:
    def __init__(self, name):
        self.name = str(name)
        self.mean = f"{name}.mean"
        self.std = f"{name}.std"


class FakeGraph:
    def node_from_name(self, name):
        return FakeNode(name)


def _batch():
    return types.SimpleNamespace(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        atomic_numbers=torch.tensor([1, 1], dtype=torch.long),
    )


def _predictor_factory(scale_by_node):
    def make_predictor(input_nodes, output_nodes):
        del input_nodes
        energy_node = output_nodes[0]

        class Predictor:
            def __call__(self, **kwargs):
                positions = kwargs["coordinates"]
                scale = float(scale_by_node.get(energy_node, 1.0))
                energy = scale * torch.sum(positions * positions).reshape(1, 1)
                extras = [
                    torch.full((1, 1), float(index), dtype=positions.dtype, device=positions.device)
                    for index, _node in enumerate(output_nodes[1:], start=1)
                ]
                return dict(zip(output_nodes, [energy, *extras]))

        return Predictor()

    return make_predictor


def test_alchemi_model_output_contains_energy_and_forces():
    energy_node = "selected_state_energy"
    model = ALFExcitedStateAlchemiModel(
        ensemble_graph=FakeGraph(),
        energy_node=energy_node,
        extra_properties={"E_mean_S0": "mean0", "E_std_S0": "std0"},
        predictor_factory=_predictor_factory({energy_node: 1.0}),
    )

    output = model(_batch())
    results = model.results_from_batch(_batch())

    assert set(output) == {"energy", "forces"}
    assert output["energy"].shape == (1, 1)
    assert output["forces"].shape == (2, 3)
    np.testing.assert_allclose(results["forces"], [[-0.0, -0.0, -0.0], [-1.0, -0.0, -0.0]])
    assert "E_mean_S0" in results


def test_alchemi_model_energy_mode_can_be_mutated_for_gap_switching():
    initial_node = "state_mean"
    switched_node = "lcm_gap_energy"
    model = ALFExcitedStateAlchemiModel(
        ensemble_graph=FakeGraph(),
        energy_node=initial_node,
        extra_properties=None,
        predictor_factory=_predictor_factory({initial_node: 1.0, switched_node: 3.0}),
    )

    initial_forces = model.results_from_batch(_batch())["forces"]
    model.set_energy_node(switched_node)
    switched_forces = model.results_from_batch(_batch())["forces"]

    np.testing.assert_allclose(initial_forces[1], [-1.0, -0.0, -0.0])
    np.testing.assert_allclose(switched_forces[1], [-3.0, -0.0, -0.0])


def test_alchemi_force_norm_helper_uses_requested_energy_node():
    mean_node = "mean_energy"
    sigma_node = "gap_sigma"
    model = ALFExcitedStateAlchemiModel(
        ensemble_graph=FakeGraph(),
        energy_node=mean_node,
        extra_properties=None,
        predictor_factory=_predictor_factory({mean_node: 1.0, sigma_node: 2.0}),
    )

    mean_norm = model.force_norm_for_energy(_batch(), mean_node)
    sigma_norm = model.force_norm_for_energy(_batch(), sigma_node)

    assert mean_norm == pytest.approx(1.0)
    assert sigma_norm == pytest.approx(2.0)
    assert model.energy_node == mean_node


def test_alchemi_model_uses_force_node_for_detached_ensemble_energy():
    energy_node = "selected_state_energy"
    force_node = "selected_state_force"

    def make_predictor(input_nodes, output_nodes):
        del input_nodes

        class Predictor:
            def __call__(self, **kwargs):
                positions = kwargs["coordinates"]
                values = {}
                for node in output_nodes:
                    if node == energy_node:
                        values[node] = torch.sum(positions * positions).detach().reshape(1, 1)
                    elif node == force_node:
                        values[node] = torch.full_like(positions.reshape(-1, 3), -2.0)
                    else:
                        values[node] = torch.zeros((1, 1), dtype=positions.dtype, device=positions.device)
                return values

        return Predictor()

    model = ALFExcitedStateAlchemiModel(
        ensemble_graph=FakeGraph(),
        energy_node=energy_node,
        force_node=force_node,
        extra_properties=None,
        predictor_factory=make_predictor,
    )

    results = model.results_from_batch(_batch())

    np.testing.assert_allclose(results["forces"], [[-2.0, -2.0, -2.0], [-2.0, -2.0, -2.0]])
