import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from types import SimpleNamespace

from alframework.samplers.ASE_ensemble_constructor import (
    MLMD_calculator,
    Well_Potential,
)
import alframework.samplers.alchemi_sampling as alchemi_module
from alframework.samplers.alchemi_sampling import (
    ALFASEAlchemiModel,
    ALFAlchemiCalculator,
    ALFHippynnAlchemiModel,
    ALFNativeEnsembleModel,
    AlchemiDynamicsRunner,
    DEFAULT_FRICTION_PER_FS,
    alchemi_sampling_task,
    alchemi_calculator_status,
    calculate_uncertainty,
    candidate_score,
    configured_alchemi_gap_rows,
    configured_score,
    load_alchemi_calculator,
    run_alchemi_sampling,
    uncertainty_flags,
)
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.molecular_topology import ReferenceTopology
from alframework.tools.tools import annealing_schedule
from tests.helpers.fakes import FixedCalculator


def _molecule(molecule_id, distance=0.74, state=None):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
    molecule = MoleculesObject(atoms, molecule_id)
    if state is not None:
        molecule.update_metadata({"selected_state": state})
    return molecule


def _topology_molecule(molecule_id, positions=None, state=None):
    if positions is None:
        positions = [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    molecule = MoleculesObject(Atoms("CCC", positions=positions), molecule_id)
    if state is not None:
        molecule.update_metadata({"selected_state": state})
    return molecule


def _reference_topology():
    return ReferenceTopology(
        reference_path="/master/reference.mol",
        reference_sha256="reference-hash",
        reference_format="mol",
        reference_charge=0,
        atomic_numbers=(6, 6, 6),
        bonds=((0, 1), (1, 2)),
        bond_lengths=(1.5, 1.5),
        bond_min_scale=0.70,
        bond_max_scale=1.35,
        connectivity_scale=1.25,
    )


def _enable_topology(config):
    config["topology_check"] = {
        "enabled": True,
        "reference_conformer_path": "reference.mol",
        "reference_format": "auto",
        "reference_charge": 0,
        "bond_min_scale": 0.70,
        "bond_max_scale": 1.35,
        "connectivity_scale": 1.25,
    }
    return config


def _config(policy="stop", batch_size=2, return_top_k=2):
    return {
        "model_mode": "ground_state",
        "uncertainty_policy": policy,
        "return_top_k": return_top_k,
        "dt": 100.0,
        "maxt": 0.2,
        "Ncheck": 1,
        "Escut": 1.0,
        "Fscut": 1.0,
        "min_time": 0.0,
        "distcut": 0.1,
        "srt_temp": [300.0, 300.0],
        "end_temp": [300.0, 300.0],
        "amp_temp": [0.0, 0.0],
        "per_temp": [1.0, 1.0],
        "end_dens": None,
        "amp_dens": None,
        "per_dens": None,
        "alchemi_baoab": {
            "batch_size": batch_size,
            "partial_policy": "full_only",
            "random_seed": 7,
        },
    }


def _gap_row():
    return {
        "lower_state": 0,
        "upper_state": 1,
        "gap_key": "dE01",
        "gap_db_name": "dE01",
    }


def _excited_properties_with_gap():
    return {
        "sE0": ["sE0", "system", 1.0],
        "F0": ["F0", "atomic", 1.0],
        "sE1": ["sE1", "system", 1.0],
        "F1": ["F1", "atomic", 1.0],
        "dE01": ["dE01", "system", 1.0],
    }


def _excited_properties_three_states():
    return {
        "sE0": ["sE0", "system", 1.0],
        "F0": ["F0", "atomic", 1.0],
        "sE1": ["sE1", "system", 1.0],
        "F1": ["F1", "atomic", 1.0],
        "sE2": ["sE2", "system", 1.0],
        "F2": ["F2", "atomic", 1.0],
        "dE01": ["dE01", "system", 1.0],
        "dE12": ["dE12", "system", 1.0],
        "dE02": ["dE02", "system", 1.0],
    }


def _enable_gap_seeking(config, **overrides):
    config["model_mode"] = "excited_state"
    config["gap_seeking"] = {
        "enabled": True,
        "mode": "levine_coe_martinez_switch",
        "switch_policy": "stay_fixed",
        "candidate_pairs": "adjacent",
        "trigger_gap_threshold_eV": 0.05,
        "sigma": 3.5,
        "alpha_eV": 0.05,
        **overrides,
    }
    return config


class FakeModel:
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics


class FakeRunner:
    last_instance = None

    def __init__(
        self,
        *,
        model,
        atoms_list,
        dt_fs,
        temperature_K,
        friction_per_fs,
        random_seed,
        device,
    ):
        del dt_fs, random_seed, device
        type(self).last_instance = self
        self.model = model
        self.initial_temperature_K = np.asarray(temperature_K, dtype=float)
        self.friction_per_fs = float(friction_per_fs)
        self.temperature_calls = []
        self.positions = [atoms.get_positions().copy() for atoms in atoms_list]
        self.velocities = [np.zeros_like(value) for value in self.positions]
        self.frozen = set()
        self.lcm_calls = []
        self.nsteps = 0
        self.evaluation_index = -1

    def run(self, steps):
        for _ in range(int(steps)):
            self.nsteps += 1
            for index, positions in enumerate(self.positions):
                if index not in self.frozen:
                    positions[:, 0] += 0.01

    def evaluate(self):
        self.evaluation_index += 1

    def diagnostics_for_graph(self, graph_index):
        sequence = self.model.diagnostics[graph_index]
        return sequence[min(self.evaluation_index, len(sequence) - 1)]

    def freeze_graph(self, graph_index):
        self.frozen.add(int(graph_index))

    def set_lcm_mode(self, graph_index, *, pair, sigma, alpha_eV):
        self.lcm_calls.append(
            {
                "graph_index": int(graph_index),
                "pair": list(pair),
                "sigma": float(sigma),
                "alpha_eV": float(alpha_eV),
                "step": int(self.nsteps),
            }
        )

    def set_temperature(self, temperature_K):
        self.temperature_K = np.asarray(temperature_K)
        self.temperature_calls.append(self.temperature_K.copy())

    def sync_graph_to_atoms(self, graph_index, atoms):
        atoms.set_positions(self.positions[int(graph_index)])
        atoms.set_velocities(self.velocities[int(graph_index)])


class TopologyMutationRunner(FakeRunner):
    def sync_graph_to_atoms(self, graph_index, atoms):
        updates = getattr(self.model, "position_updates", {})
        positions = updates.get(
            self.evaluation_index + 1,
            {},
        ).get(int(graph_index))
        if positions is not None:
            self.positions[int(graph_index)] = np.asarray(
                positions,
                dtype=float,
            ).copy()
        super().sync_graph_to_atoms(graph_index, atoms)


def _torch_batch(torch, positions=None):
    class Batch:
        pass

    batch = Batch()
    if positions is None:
        positions = np.zeros((4, 3), dtype=float)
    batch.positions = torch.as_tensor(positions, dtype=torch.float32)
    batch.atomic_numbers = torch.ones(4, dtype=torch.long)
    batch.atomic_masses = torch.ones(4)
    batch.num_graphs = 2
    batch.num_nodes_per_graph = torch.as_tensor([2, 2])
    return batch


def test_uncertainty_statistics_match_production_population_std():
    metrics = calculate_uncertainty(
        [1.0, 3.0],
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        ],
    )

    assert metrics["Es"] == 1.0
    assert metrics["Fs"] == 1.0 / 3.0
    assert metrics["Fsmax"] == 1.0
    flags = uncertainty_flags(metrics, Escut=0.5, Fscut=0.5)
    assert flags["uncertain"] is True
    assert flags["uncertainty_score"] == 2.0


def test_shared_torch_reduction_matches_mlmd_calculator():
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"Optional Torch stack is unavailable: {exc}")

    class RawEnsemble(ALFAlchemiCalculator):
        def ensemble_forward(self, batch):
            energy = torch.tensor(
                [[1.0, 1.0], [3.0, 3.0]], device=batch.positions.device
            )
            forces_one = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] * 2,
                device=batch.positions.device,
            )
            forces_two = torch.tensor(
                [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]] * 2,
                device=batch.positions.device,
            )
            return {
                "energy_contributions": energy,
                "force_contributions": torch.stack([forces_one, forces_two]),
            }

    model = RawEnsemble(device=torch.device("cpu"))
    output = model(_torch_batch(torch))
    diagnostics = model.diagnostics_for_graph(0)

    atoms = _molecule("parity").get_atoms()
    ase_model = MLMD_calculator(
        [
            FixedCalculator(1.0, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            FixedCalculator(3.0, [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
        ]
    )
    ase_model.calculate(
        atoms,
        ["energy", "forces", "energy_stdev", "forces_stdev_mean", "forces_stdev_max"],
    )

    np.testing.assert_allclose(output["energy"].detach().numpy(), [[2.0], [2.0]])
    assert diagnostics["Es"] == pytest.approx(ase_model.results["energy_stdev"])
    assert diagnostics["Fs"] == pytest.approx(ase_model.results["forces_stdev_mean"])
    assert diagnostics["Fsmax"] == pytest.approx(ase_model.results["forces_stdev_max"])


def test_shared_calculator_uses_correlated_member_gap_uncertainty():
    import torch

    class RawEnsemble(ALFAlchemiCalculator):
        def ensemble_forward(self, batch):
            zeros = torch.zeros(
                (2, batch.positions.shape[0], 3),
                device=batch.positions.device,
            )
            state_energies = {
                0: torch.tensor(
                    [[0.0, 0.0], [2.0, 2.0]],
                    device=batch.positions.device,
                ),
                1: torch.tensor(
                    [[4.0, 4.0], [2.0, 2.0]],
                    device=batch.positions.device,
                ),
            }
            return {
                "energy_contributions": state_energies[0],
                "force_contributions": zeros,
                "state_energy_contributions": state_energies,
            }

    model = RawEnsemble(
        selected_state_value=0,
        gap_rows=[_gap_row()],
        device=torch.device("cpu"),
    )
    model(_torch_batch(torch))
    diagnostics = model.diagnostics_for_graph(0)

    assert diagnostics["dE01"] == pytest.approx(2.0)
    assert diagnostics["dE01_stdev"] == pytest.approx(2.0)
    assert diagnostics["dE01_model_count"] == 2
    assert diagnostics["dE01_stdev"] != pytest.approx(1.0 - 1.0)


def test_gap_diagnostics_reject_inconsistent_state_ensemble_sizes():
    import torch

    class BadGapEnsemble(ALFAlchemiCalculator):
        def ensemble_forward(self, batch):
            return {
                "energy_contributions": torch.zeros(
                    (2, batch.num_graphs)
                ),
                "force_contributions": torch.zeros(
                    (2, batch.positions.shape[0], 3)
                ),
                "state_energy_contributions": {
                    0: torch.zeros((2, batch.num_graphs)),
                    1: torch.zeros((1, batch.num_graphs)),
                },
            }

    with pytest.raises(ValueError, match="same ordered model members"):
        BadGapEnsemble(
            gap_rows=[_gap_row()],
            device=torch.device("cpu"),
        )(_torch_batch(torch))


def test_lcm_energy_force_formula_and_per_graph_dynamics_are_exact():
    import torch

    settings = {
        "enabled": True,
        "selected_state": 0,
        "rows": [_gap_row()],
        "sigma": 3.5,
        "alpha_eV": 0.05,
    }

    class RawStates(ALFAlchemiCalculator):
        def ensemble_forward(self, batch):
            state_zero_energy = torch.tensor(
                [[1.0, 1.0], [3.0, 3.0]],
                device=batch.positions.device,
            )
            state_one_energy = state_zero_energy + 0.02
            state_zero_forces = torch.stack(
                [
                    torch.ones_like(batch.positions),
                    3.0 * torch.ones_like(batch.positions),
                ]
            )
            state_one_forces = state_zero_forces + 3.0
            states = {
                0: {
                    "energy_contributions": state_zero_energy,
                    "force_contributions": state_zero_forces,
                },
                1: {
                    "energy_contributions": state_one_energy,
                    "force_contributions": state_one_forces,
                },
            }
            return {
                **states[0],
                "state_contributions": states,
            }

    model = RawStates(
        selected_state_value=0,
        gap_rows=[_gap_row()],
        gap_seeking_settings=settings,
        device=torch.device("cpu"),
    )
    model.set_lcm_mode(
        0,
        pair=[0, 1],
        sigma=3.5,
        alpha_eV=0.05,
    )
    output = model(_torch_batch(torch))

    expected_energy, expected_forces = model._lcm_energy_forces(
        torch.tensor(2.0),
        torch.tensor(2.02),
        torch.full((2, 3), 2.0),
        torch.full((2, 3), 5.0),
        sigma=3.5,
        alpha_eV=0.05,
    )
    assert output["energy"][0, 0].item() == pytest.approx(
        expected_energy.item()
    )
    np.testing.assert_allclose(
        output["forces"][:2].detach().numpy(),
        expected_forces.detach().numpy(),
        atol=1.0e-6,
    )
    assert output["energy"][1, 0].item() == pytest.approx(2.0)
    np.testing.assert_allclose(
        output["forces"][2:].detach().numpy(),
        np.full((2, 3), 2.0),
    )
    assert model.diagnostics_for_graph(0)["Es"] == pytest.approx(1.0)
    assert model.diagnostics_for_graph(0)["Fs"] == pytest.approx(1.0)
    assert model.diagnostics_for_graph(0)["dE01"] == pytest.approx(0.02)

    model.well_params = {
        "r_start": 1.0,
        "force": 2.0,
        "origin": [0.0, 0.0, 0.0],
        "mass_weighted": False,
    }
    displaced_batch = _torch_batch(
        torch,
        positions=np.asarray([[2.0, 0.0, 0.0]] * 4),
    )
    without_well_energy = output["energy"].detach().clone()
    without_well_forces = output["forces"].detach().clone()
    with_well = model(displaced_batch)
    np.testing.assert_allclose(
        (with_well["energy"] - without_well_energy).detach().numpy(),
        np.full((2, 1), 4.0),
        atol=1.0e-6,
    )
    expected_force_delta = np.zeros((4, 3))
    expected_force_delta[:, 0] = -2.0
    np.testing.assert_allclose(
        (with_well["forces"] - without_well_forces).detach().numpy(),
        expected_force_delta,
        atol=1.0e-6,
    )
    assert model.diagnostics_for_graph(0)["dE01"] == pytest.approx(0.02)
    assert model.diagnostics_for_graph(0)["Es"] == pytest.approx(1.0)

    coordinate = torch.tensor(0.4, requires_grad=True)
    energy, force = model._lcm_energy_forces(
        coordinate,
        3.0 * coordinate + 0.02,
        torch.tensor([-1.0]),
        torch.tensor([-3.0]),
        sigma=3.5,
        alpha_eV=0.05,
    )
    autograd_force = -torch.autograd.grad(energy, coordinate)[0]
    assert force.item() == pytest.approx(
        autograd_force.item(),
        abs=5.0e-6,
    )


@pytest.mark.parametrize("mass_weighted", [True, False])
def test_alchemi_well_matches_production_without_changing_uncertainty(
    mass_weighted,
):
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"Optional Torch stack is unavailable: {exc}")

    origin = np.asarray([0.5, -0.25, 0.1])
    atoms_list = [
        Atoms(
            "HHe",
            positions=[origin, origin + np.asarray([1.2, 0.0, 0.0])],
        ),
        Atoms(
            "HHe",
            positions=[
                origin + np.asarray([0.0, 1.0, 0.0]),
                origin + np.asarray([0.0, 0.0, -1.5]),
            ],
        ),
    ]
    well_params = {
        "r_start": 0.8,
        "force": 0.3,
        "origin": origin.tolist(),
        "mass_weighted": mass_weighted,
    }

    class Batch:
        positions = torch.as_tensor(
            np.concatenate([atoms.get_positions() for atoms in atoms_list]),
            dtype=torch.float32,
        )
        atomic_numbers = torch.as_tensor(
            np.concatenate([atoms.get_atomic_numbers() for atoms in atoms_list]),
            dtype=torch.long,
        )
        atomic_masses = torch.as_tensor(
            np.concatenate([atoms.get_masses() for atoms in atoms_list]),
            dtype=torch.float32,
        )
        num_graphs = 2
        num_nodes_per_graph = torch.as_tensor([2, 2])

    class RawEnsemble(ALFAlchemiCalculator):
        def ensemble_forward(self, batch):
            return {
                "energy_contributions": torch.tensor(
                    [[1.0, 2.0], [3.0, 6.0]], device=batch.positions.device
                ),
                "force_contributions": torch.stack(
                    [
                        torch.zeros_like(batch.positions),
                        2.0 * torch.ones_like(batch.positions),
                    ]
                ),
            }

    unbiased_model = RawEnsemble(device=torch.device("cpu"))
    biased_model = RawEnsemble(
        well_params=well_params,
        device=torch.device("cpu"),
    )
    unbiased = unbiased_model(Batch())
    biased = biased_model(Batch())

    energy_delta = (biased["energy"] - unbiased["energy"]).detach().cpu().numpy()
    energy_delta = energy_delta.reshape(-1)
    force_delta = (biased["forces"] - unbiased["forces"]).detach().cpu().numpy()
    force_delta = force_delta.reshape(2, 2, 3)
    for graph_index, atoms in enumerate(atoms_list):
        production_well = Well_Potential(**well_params)
        production_well.calculate(atoms, properties=["energy", "forces"])
        assert energy_delta[graph_index] == pytest.approx(
            production_well.results["energy"], abs=1.0e-6
        )
        np.testing.assert_allclose(
            force_delta[graph_index],
            production_well.results["forces"],
            atol=1.0e-6,
        )

    for graph_index in range(2):
        unbiased_diagnostics = unbiased_model.diagnostics_for_graph(graph_index)
        biased_diagnostics = biased_model.diagnostics_for_graph(graph_index)
        for key in ("Es", "Fs", "Fsmax"):
            assert biased_diagnostics[key] == pytest.approx(
                unbiased_diagnostics[key]
            )


def test_ase_fallback_list_matches_shared_ensemble_reduction():
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"Optional Torch stack is unavailable: {exc}")

    model = ALFASEAlchemiModel(
        [
            FixedCalculator(1.0, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            FixedCalculator(3.0, [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
        ],
        model_mode="ground_state",
        selected_state_value=None,
        device=torch.device("cpu"),
        calculator_loader="tests.fake_loader",
    )
    output = model(_torch_batch(torch))
    diagnostics = model.diagnostics_for_graph(1)

    np.testing.assert_allclose(output["energy"].detach().numpy(), [[2.0], [2.0]])
    assert diagnostics["Es"] == pytest.approx(1.0)
    assert diagnostics["Fs"] == pytest.approx(1.0 / 3.0)
    assert diagnostics["Fsmax"] == pytest.approx(1.0)
    assert model.calculator_interface == "ase_fallback"


def test_ase_fallback_uses_standard_uncertainty_properties():
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"Optional Torch stack is unavailable: {exc}")

    class StandardUncertaintyCalculator(Calculator):
        implemented_properties = [
            "energy",
            "forces",
            "energy_stdev",
            "forces_stdev_mean",
            "forces_stdev_max",
        ]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "energy": 4.0,
                "forces": np.zeros((len(atoms), 3)),
                "energy_stdev": 2.5,
                "forces_stdev_mean": 0.4,
                "forces_stdev_max": 1.3,
            }

    model = ALFASEAlchemiModel(
        StandardUncertaintyCalculator(),
        model_mode="ground_state",
        selected_state_value=None,
        device=torch.device("cpu"),
    )
    model(_torch_batch(torch))

    assert model.diagnostics_for_graph(0)["Es"] == pytest.approx(2.5)
    assert model.diagnostics_for_graph(0)["Fs"] == pytest.approx(0.4)
    assert model.diagnostics_for_graph(0)["Fsmax"] == pytest.approx(1.3)


def test_ase_fallback_retains_legacy_neurochem_uncertainty():
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"Optional Torch stack is unavailable: {exc}")

    class NeuroChemLikeCalculator(FixedCalculator):
        Estddev = 0.004

        def get_Fstddev(self):
            return 0.2, 0.7

    model = ALFASEAlchemiModel(
        NeuroChemLikeCalculator(1.0, np.zeros((2, 3))),
        model_mode="ground_state",
        selected_state_value=None,
        device=torch.device("cpu"),
        uncertainty_mode="neurochem",
    )
    model(_torch_batch(torch))

    assert model.diagnostics_for_graph(0)["Es"] == pytest.approx(4.0)
    assert model.diagnostics_for_graph(0)["Fs"] == pytest.approx(0.2)
    assert model.diagnostics_for_graph(0)["Fsmax"] == pytest.approx(0.7)


def test_ase_fallback_rejects_missing_excited_state_properties():
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"Optional Torch stack is unavailable: {exc}")

    model = ALFASEAlchemiModel(
        FixedCalculator(1.0, np.zeros((2, 3))),
        model_mode="excited_state",
        selected_state_value=1,
        device=torch.device("cpu"),
    )
    with pytest.raises(KeyError, match="sE1, F1"):
        model(_torch_batch(torch))


def test_ase_fallback_reads_flattened_excited_state_properties():
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"Optional Torch stack is unavailable: {exc}")

    class ExcitedCalculator(Calculator):
        implemented_properties = ["sE1", "F1"]

        def __init__(self, energy, force):
            super().__init__()
            self.energy = float(energy)
            self.force = float(force)

        def calculate(self, atoms=None, properties=("sE1",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "sE1": self.energy,
                "F1": np.full((len(atoms), 3), self.force),
            }

    model = ALFASEAlchemiModel(
        [ExcitedCalculator(2.0, 1.0), ExcitedCalculator(4.0, 3.0)],
        model_mode="excited_state",
        selected_state_value=1,
        device=torch.device("cpu"),
    )
    output = model(_torch_batch(torch))

    np.testing.assert_allclose(output["energy"].detach().numpy(), [[3.0], [3.0]])
    np.testing.assert_allclose(output["forces"].detach().numpy(), np.full((4, 3), 2.0))
    assert model.diagnostics_for_graph(0)["Es"] == pytest.approx(1.0)


def test_ase_fallback_gap_diagnostics_match_raw_member_reduction():
    import torch

    class MultiStateCalculator(Calculator):
        implemented_properties = ["sE0", "sE1", "F1"]

        def __init__(self, energy0, energy1):
            super().__init__()
            self.energy0 = float(energy0)
            self.energy1 = float(energy1)

        def calculate(
            self,
            atoms=None,
            properties=("sE1",),
            system_changes=all_changes,
        ):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "sE0": self.energy0,
                "sE1": self.energy1,
                "F1": np.zeros((len(atoms), 3)),
            }

    model = ALFASEAlchemiModel(
        [
            MultiStateCalculator(0.0, 4.0),
            MultiStateCalculator(2.0, 2.0),
        ],
        model_mode="excited_state",
        selected_state_value=1,
        gap_rows=[_gap_row()],
        device=torch.device("cpu"),
    )
    model(_torch_batch(torch))
    diagnostics = model.diagnostics_for_graph(0)

    assert diagnostics["dE01"] == pytest.approx(2.0)
    assert diagnostics["dE01_stdev"] == pytest.approx(2.0)
    assert diagnostics["dE01_model_count"] == 2


def test_single_ase_calculator_gap_deviation_is_zero():
    import torch

    class MultiStateCalculator(Calculator):
        implemented_properties = ["sE0", "sE1", "F1"]

        def calculate(
            self,
            atoms=None,
            properties=("sE1",),
            system_changes=all_changes,
        ):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "sE0": 1.0,
                "sE1": 1.75,
                "F1": np.zeros((len(atoms), 3)),
            }

    model = ALFASEAlchemiModel(
        MultiStateCalculator(),
        model_mode="excited_state",
        selected_state_value=1,
        gap_rows=[_gap_row()],
        device=torch.device("cpu"),
    )
    model(_torch_batch(torch))

    assert model.diagnostics_for_graph(0)["dE01"] == pytest.approx(0.75)
    assert model.diagnostics_for_graph(0)["dE01_stdev"] == pytest.approx(0.0)


def test_ase_fallback_gap_diagnostics_require_all_state_energies():
    import torch

    class SelectedOnlyCalculator(Calculator):
        implemented_properties = ["sE1", "F1"]

        def calculate(
            self,
            atoms=None,
            properties=("sE1",),
            system_changes=all_changes,
        ):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "sE1": 2.0,
                "F1": np.zeros((len(atoms), 3)),
            }

    model = ALFASEAlchemiModel(
        SelectedOnlyCalculator(),
        model_mode="excited_state",
        selected_state_value=1,
        gap_rows=[_gap_row()],
        device=torch.device("cpu"),
    )
    with pytest.raises(KeyError, match="sE0"):
        model(_torch_batch(torch))


def test_ase_fallback_lcm_matches_native_state_contributions():
    import torch

    settings = {
        "enabled": True,
        "selected_state": 0,
        "rows": [_gap_row()],
        "sigma": 3.5,
        "alpha_eV": 0.05,
    }

    class MultiStateCalculator(Calculator):
        implemented_properties = ["sE0", "F0", "sE1", "F1"]

        def __init__(self, energy0, force0, energy1, force1):
            super().__init__()
            self.values = (energy0, force0, energy1, force1)

        def calculate(
            self,
            atoms=None,
            properties=("sE0",),
            system_changes=all_changes,
        ):
            super().calculate(atoms, properties, system_changes)
            energy0, force0, energy1, force1 = self.values
            self.results = {
                "sE0": float(energy0),
                "F0": np.full((len(atoms), 3), float(force0)),
                "sE1": float(energy1),
                "F1": np.full((len(atoms), 3), float(force1)),
            }

    ase_model = ALFASEAlchemiModel(
        [
            MultiStateCalculator(1.0, 1.0, 1.02, 4.0),
            MultiStateCalculator(3.0, 3.0, 3.02, 6.0),
        ],
        model_mode="excited_state",
        selected_state_value=0,
        gap_rows=[_gap_row()],
        gap_seeking_settings=settings,
        device=torch.device("cpu"),
    )

    class NativeStates(ALFAlchemiCalculator):
        def ensemble_forward(self, batch):
            state_zero = {
                "energy_contributions": torch.tensor(
                    [[1.0, 1.0], [3.0, 3.0]]
                ),
                "force_contributions": torch.stack(
                    [
                        torch.ones_like(batch.positions),
                        3.0 * torch.ones_like(batch.positions),
                    ]
                ),
            }
            state_one = {
                "energy_contributions": torch.tensor(
                    [[1.02, 1.02], [3.02, 3.02]]
                ),
                "force_contributions": torch.stack(
                    [
                        4.0 * torch.ones_like(batch.positions),
                        6.0 * torch.ones_like(batch.positions),
                    ]
                ),
            }
            return {
                **state_zero,
                "state_contributions": {0: state_zero, 1: state_one},
            }

    native_model = NativeStates(
        selected_state_value=0,
        gap_rows=[_gap_row()],
        gap_seeking_settings=settings,
        device=torch.device("cpu"),
    )
    for model in (ase_model, native_model):
        model.set_lcm_mode(
            0,
            pair=[0, 1],
            sigma=3.5,
            alpha_eV=0.05,
        )
    batch = _torch_batch(torch)
    ase_output = ase_model(batch)
    native_output = native_model(batch)

    np.testing.assert_allclose(
        ase_output["energy"].detach().numpy(),
        native_output["energy"].detach().numpy(),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        ase_output["forces"].detach().numpy(),
        native_output["forces"].detach().numpy(),
        atol=1.0e-6,
    )
    for graph_index in range(2):
        for key in ("Es", "Fs", "Fsmax", "dE01", "dE01_stdev"):
            assert ase_model.diagnostics_for_graph(graph_index)[key] == (
                pytest.approx(
                    native_model.diagnostics_for_graph(graph_index)[key]
                )
            )


def test_ase_fallback_gap_seeking_requires_both_state_forces():
    import torch

    class MissingUpperForce(Calculator):
        implemented_properties = ["sE0", "F0", "sE1"]

        def calculate(
            self,
            atoms=None,
            properties=("sE0",),
            system_changes=all_changes,
        ):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "sE0": 0.0,
                "F0": np.zeros((len(atoms), 3)),
                "sE1": 0.01,
            }

    settings = {
        "enabled": True,
        "selected_state": 0,
        "rows": [_gap_row()],
        "sigma": 3.5,
        "alpha_eV": 0.05,
    }
    model = ALFASEAlchemiModel(
        MissingUpperForce(),
        model_mode="excited_state",
        selected_state_value=0,
        gap_rows=[_gap_row()],
        gap_seeking_settings=settings,
        device=torch.device("cpu"),
    )

    with pytest.raises(KeyError, match="F1"):
        model(_torch_batch(torch))


def test_shared_calculator_rejects_inconsistent_ensemble_sizes():
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"Optional Torch stack is unavailable: {exc}")

    class BadCalculator(ALFAlchemiCalculator):
        def ensemble_forward(self, batch):
            return {
                "energy_contributions": torch.zeros((2, batch.num_graphs)),
                "force_contributions": torch.zeros((1, batch.positions.shape[0], 3)),
            }

    with pytest.raises(ValueError, match="same model count"):
        BadCalculator(device=torch.device("cpu"))(_torch_batch(torch))


def test_stop_mode_freezes_each_replica_at_its_first_uncertainty():
    model = FakeModel(
        {
            0: [{"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0}],
            1: [
                {"Es": 0.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 0.0, "Fs": 2.0, "Fsmax": 0.0},
            ],
        }
    )
    model.calculator_interface = "ase_fallback"
    model.calculator_loader = "custom.ase_loader"
    created = []

    def factory(**kwargs):
        runner = FakeRunner(**kwargs)
        created.append(runner)
        return runner

    outputs = run_alchemi_sampling(
        [_molecule("first"), _molecule("second")],
        _config(policy="stop"),
        model,
        runner_factory=factory,
    )

    assert [item.get_metadata()["parent_molecule_id"] for item in outputs] == [
        "first",
        "second",
    ]
    assert [item.get_metadata()["step"] for item in outputs] == [1, 2]
    assert created[0].frozen == {0, 1}
    assert outputs[0].get_atoms().positions[0, 0] == pytest.approx(0.01)
    assert outputs[1].get_atoms().positions[0, 0] == pytest.approx(0.02)
    assert outputs[0].get_metadata()["calculator_interface"] == "ase_fallback"
    assert outputs[0].get_metadata()["calculator_loader"] == "custom.ase_loader"
    assert created[0].friction_per_fs == pytest.approx(0.02 * units.fs)
    assert DEFAULT_FRICTION_PER_FS == pytest.approx(0.02 * units.fs)


def test_maximum_mean_force_screen_rejects_before_candidate_return():
    model = FakeModel(
        {
            0: [
                {
                    "Es": 2.0,
                    "Fs": 0.0,
                    "Fsmax": 0.0,
                    "Fmeanmax": 20.0,
                }
            ],
            1: [
                {
                    "Es": 2.0,
                    "Fs": 0.0,
                    "Fsmax": 0.0,
                    "Fmeanmax": 1.0,
                }
            ],
        }
    )
    config = _config(policy="stop")
    config["max_force_cutoff"] = 16.0

    outputs = run_alchemi_sampling(
        [_molecule("rejected"), _molecule("accepted")],
        config,
        model,
        runner_factory=FakeRunner,
    )

    assert [item.get_metadata()["parent_molecule_id"] for item in outputs] == [
        "accepted"
    ]
    assert outputs[0].get_metadata()["Fmeanmax"] == pytest.approx(1.0)


def test_topology_gate_freezes_initial_invalid_replica_before_dynamics(
    monkeypatch,
):
    monkeypatch.setattr(
        alchemi_module,
        "load_reference_topology",
        lambda *args, **kwargs: _reference_topology(),
    )
    model = FakeModel(
        {
            1: [{"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0}],
        }
    )
    config = _enable_topology(_config(policy="stop"))
    invalid_positions = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]

    outputs = run_alchemi_sampling(
        [
            _topology_molecule("invalid", invalid_positions),
            _topology_molecule("valid"),
        ],
        config,
        model,
        runner_factory=FakeRunner,
        master_directory="/master",
    )

    runner = FakeRunner.last_instance
    assert runner.nsteps == 1
    assert runner.frozen == {0, 1}
    assert [item.get_metadata()["parent_molecule_id"] for item in outputs] == [
        "valid"
    ]
    metadata = outputs[0].get_metadata()
    assert metadata["topology_valid"] is True
    assert metadata["topology_reference_sha256"] == "reference-hash"
    assert metadata["topology_rejected_replica_count"] == 1
    assert metadata["topology_rejections"][0]["stage"] == "initial"


def test_topology_gate_skips_dynamics_when_every_replica_is_invalid(
    monkeypatch,
):
    monkeypatch.setattr(
        alchemi_module,
        "load_reference_topology",
        lambda *args, **kwargs: _reference_topology(),
    )
    called = False

    def factory(**kwargs):
        nonlocal called
        called = True
        return FakeRunner(**kwargs)

    invalid = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]
    outputs = run_alchemi_sampling(
        [
            _topology_molecule("first", invalid),
            _topology_molecule("second", invalid),
        ],
        _enable_topology(_config(policy="stop")),
        FakeModel({}),
        runner_factory=factory,
    )

    assert outputs == []
    assert called is False


def test_topology_gate_rejects_frame_before_uncertainty_and_lcm(
    monkeypatch,
):
    monkeypatch.setattr(
        alchemi_module,
        "load_reference_topology",
        lambda *args, **kwargs: _reference_topology(),
    )
    triangle_height = np.sqrt(1.5**2 - 0.75**2)
    model = FakeModel(
        {
            1: [
                {
                    "Es": 0.0,
                    "Fs": 0.0,
                    "Fsmax": 0.0,
                    "dE01": 0.2,
                    "dE01_stdev": 0.0,
                    "dE01_model_count": 2,
                }
            ],
        }
    )
    model.position_updates = {
        0: {
            0: [
                [0.0, 0.0, 0.0],
                [0.75, triangle_height, 0.0],
                [1.5, 0.0, 0.0],
            ]
        }
    }
    config = _enable_topology(_config(policy="continue"))
    _enable_gap_seeking(config)

    outputs = run_alchemi_sampling(
        [
            _topology_molecule("rejected", state=0),
            _topology_molecule("survivor", state=0),
        ],
        config,
        model,
        properties_list=_excited_properties_with_gap(),
        runner_factory=TopologyMutationRunner,
    )

    runner = TopologyMutationRunner.last_instance
    assert outputs == []
    assert 0 in runner.frozen
    assert runner.lcm_calls == []


def test_topology_rejection_preserves_earlier_valid_top_k_candidate(
    monkeypatch,
):
    monkeypatch.setattr(
        alchemi_module,
        "load_reference_topology",
        lambda *args, **kwargs: _reference_topology(),
    )
    triangle_height = np.sqrt(1.5**2 - 0.75**2)
    model = FakeModel(
        {
            0: [
                {"Es": 3.0, "Fs": 0.0, "Fsmax": 0.0},
            ],
            1: [
                {"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 4.0, "Fs": 0.0, "Fsmax": 0.0},
            ],
        }
    )
    model.position_updates = {
        1: {
            0: [
                [0.0, 0.0, 0.0],
                [0.75, triangle_height, 0.0],
                [1.5, 0.0, 0.0],
            ]
        }
    }

    outputs = run_alchemi_sampling(
        [_topology_molecule("first"), _topology_molecule("second")],
        _enable_topology(
            _config(policy="continue", return_top_k=2)
        ),
        model,
        runner_factory=TopologyMutationRunner,
    )

    assert [
        item.get_metadata()["parent_molecule_id"] for item in outputs
    ] == ["second", "first"]
    assert [
        item.get_metadata()["uncertainty_score"] for item in outputs
    ] == [4.0, 3.0]
    for item in outputs:
        metadata = item.get_metadata()
        assert metadata["topology_rejected_replica_count"] == 1
        assert metadata["topology_rejections"][0]["batch_index"] == 0
        assert metadata["topology_rejections"][0]["stage"] == "ncheck"


def test_temperature_schedule_matches_legacy_timing_per_replica(monkeypatch):
    temperature_parameters = [
        {"Tamp": 10.0, "Tper": 0.2, "Tsrt": 100.0, "Tend": 160.0},
        {"Tamp": 20.0, "Tper": 0.4, "Tsrt": 200.0, "Tend": 260.0},
    ]
    monkeypatch.setattr(
        alchemi_module,
        "_temperature_parameters",
        lambda sampler_config, count, random_seed: temperature_parameters,
    )
    model = FakeModel(
        {
            0: [
                {"Es": 0.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 0.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0},
            ],
            1: [
                {"Es": 0.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0},
            ],
        }
    )
    config = _config(policy="stop")
    config["maxt"] = 0.3
    created = []

    def factory(**kwargs):
        runner = FakeRunner(**kwargs)
        created.append(runner)
        return runner

    outputs = run_alchemi_sampling(
        [_molecule("first"), _molecule("second")],
        config,
        model,
        runner_factory=factory,
    )

    runner = created[0]
    expected_by_time = [
        np.asarray(
            [
                annealing_schedule(
                    time_ps,
                    config["maxt"],
                    row["Tamp"],
                    row["Tper"],
                    row["Tsrt"],
                    row["Tend"],
                )
                for row in temperature_parameters
            ]
        )
        for time_ps in (0.0, 0.1)
    ]
    np.testing.assert_allclose(runner.initial_temperature_K, expected_by_time[0])
    assert len(runner.temperature_calls) == 2
    for actual, expected in zip(runner.temperature_calls, expected_by_time):
        np.testing.assert_allclose(actual, expected)

    metadata_by_parent = {
        item.get_metadata()["parent_molecule_id"]: item.get_metadata()
        for item in outputs
    }
    assert metadata_by_parent["first"]["temps"] == pytest.approx(
        [expected_by_time[0][0], expected_by_time[1][0]]
    )
    assert metadata_by_parent["second"]["temps"] == pytest.approx(
        [expected_by_time[0][1]]
    )
    for replica_index, parent_id in enumerate(("first", "second")):
        metadata = metadata_by_parent[parent_id]
        for key, value in temperature_parameters[replica_index].items():
            assert metadata[key] == value
        assert metadata["step"] * config["dt"] / 1000.0 == pytest.approx(
            metadata["time_ps"] + config["dt"] / 1000.0
        )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("srt_temp", [300.0], "two-value numeric range"),
        ("end_temp", ["cold", 300.0], "finite numeric values"),
        ("amp_temp", [10.0, -10.0], "minimum must not exceed"),
        ("srt_temp", [300.0, np.inf], "finite numeric values"),
    ],
)
def test_temperature_schedule_rejects_malformed_ranges(key, value, message):
    config = _config()
    config[key] = value

    with pytest.raises(ValueError, match=message):
        run_alchemi_sampling(
            [_molecule("first"), _molecule("second")],
            config,
            FakeModel({}),
            runner_factory=FakeRunner,
        )


def test_temperature_schedule_rejects_sampled_nonpositive_period():
    config = _config()
    config["per_temp"] = [0.0, 0.0]

    with pytest.raises(ValueError, match="nonpositive temperature period"):
        run_alchemi_sampling(
            [_molecule("first"), _molecule("second")],
            config,
            FakeModel({}),
            runner_factory=FakeRunner,
        )


def test_stop_mode_honors_min_time_before_flagging():
    config = _config(policy="stop")
    config["min_time"] = 0.1
    model = FakeModel(
        {
            0: [{"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0}],
            1: [{"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0}],
        }
    )

    outputs = run_alchemi_sampling(
        [_molecule("first"), _molecule("second")],
        config,
        model,
        runner_factory=FakeRunner,
    )

    assert [item.get_metadata()["time_ps"] for item in outputs] == [0.1, 0.1]
    assert [item.get_metadata()["step"] for item in outputs] == [2, 2]


def test_close_contact_is_stopped_and_discarded():
    model = FakeModel(
        {
            0: [{"Es": 3.0, "Fs": 0.0, "Fsmax": 0.0}],
            1: [{"Es": 3.0, "Fs": 0.0, "Fsmax": 0.0}],
        }
    )

    outputs = run_alchemi_sampling(
        [_molecule("first", distance=0.05), _molecule("second", distance=0.05)],
        _config(policy="stop"),
        model,
        runner_factory=FakeRunner,
    )

    assert outputs == []


def test_continue_mode_returns_global_top_k_with_deterministic_order():
    model = FakeModel(
        {
            0: [
                {"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 5.0, "Fs": 0.0, "Fsmax": 0.0},
            ],
            1: [
                {"Es": 4.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 3.0, "Fs": 0.0, "Fsmax": 0.0},
            ],
        }
    )

    outputs = run_alchemi_sampling(
        [_molecule("first"), _molecule("second")],
        _config(policy="continue", return_top_k=2),
        model,
        runner_factory=FakeRunner,
    )

    assert [item.get_metadata()["uncertainty_score"] for item in outputs] == [5.0, 4.0]
    assert [item.get_metadata()["parent_molecule_id"] for item in outputs] == [
        "first",
        "second",
    ]
    assert [item.get_moleculeid() for item in outputs] == [
        "first-cand-0000",
        "second-cand-0001",
    ]


def test_gap_diagnostics_add_provenance_without_changing_ranking():
    config = _config(
        policy="continue",
        batch_size=1,
        return_top_k=1,
    )
    config.update(
        {
            "model_mode": "excited_state",
            "gap_diagnostics": {"enabled": True},
        }
    )
    model = FakeModel(
        {
            0: [
                {
                    "Es": 2.0,
                    "Fs": 0.0,
                    "Fsmax": 0.0,
                    "dE01": 10.0,
                    "dE01_stdev": 0.4,
                    "dE01_model_count": 4,
                },
                {
                    "Es": 1.5,
                    "Fs": 0.0,
                    "Fsmax": 0.0,
                    "dE01": 0.01,
                    "dE01_stdev": 2.0,
                    "dE01_model_count": 4,
                },
            ]
        }
    )
    model.gap_rows = [_gap_row()]

    outputs = run_alchemi_sampling(
        [_molecule("gap-ranked", state=0)],
        config,
        model,
        properties_list=_excited_properties_with_gap(),
        runner_factory=FakeRunner,
    )

    assert len(outputs) == 1
    metadata = outputs[0].get_metadata()
    assert metadata["uncertainty_score"] == pytest.approx(2.0)
    assert metadata["gap_means"] == {"dE01": pytest.approx(10.0)}
    assert metadata["gap_stds"] == {"dE01": pytest.approx(0.4)}
    assert metadata["gap_model_counts"] == {"dE01": 4}
    assert metadata["minimum_abs_gap_key"] == "dE01"


def test_stay_fixed_gap_seeking_switches_replicas_independently_at_ncheck():
    config = _enable_gap_seeking(
        _config(policy="continue", batch_size=2, return_top_k=2)
    )
    quiet = {"Es": 0.0, "Fs": 0.0, "Fsmax": 0.0}
    model = FakeModel(
        {
            0: [
                {
                    **quiet,
                    "dE01": 0.01,
                    "dE01_stdev": 0.0,
                    "dE01_model_count": 2,
                },
                {
                    **quiet,
                    "dE01": 0.20,
                    "dE01_stdev": 0.0,
                    "dE01_model_count": 2,
                },
                {
                    "Es": 3.0,
                    "Fs": 0.0,
                    "Fsmax": 0.0,
                    "dE01": 0.20,
                    "dE01_stdev": 0.0,
                    "dE01_model_count": 2,
                },
            ],
            1: [
                {
                    **quiet,
                    "dE01": 0.10,
                    "dE01_stdev": 0.0,
                    "dE01_model_count": 2,
                },
                {
                    **quiet,
                    "dE01": 0.10,
                    "dE01_stdev": 0.0,
                    "dE01_model_count": 2,
                },
                {
                    "Es": 2.0,
                    "Fs": 0.0,
                    "Fsmax": 0.0,
                    "dE01": 0.01,
                    "dE01_stdev": 0.0,
                    "dE01_model_count": 2,
                },
            ],
        }
    )

    outputs = run_alchemi_sampling(
        [_molecule("first", state=0), _molecule("second", state=0)],
        config,
        model,
        properties_list=_excited_properties_with_gap(),
        runner_factory=FakeRunner,
    )

    assert FakeRunner.last_instance.lcm_calls == [
        {
            "graph_index": 0,
            "pair": [0, 1],
            "sigma": 3.5,
            "alpha_eV": 0.05,
            "step": 1,
        },
        {
            "graph_index": 1,
            "pair": [0, 1],
            "sigma": 3.5,
            "alpha_eV": 0.05,
            "step": 2,
        },
    ]
    by_parent = {
        item.get_metadata()["parent_molecule_id"]: item.get_metadata()
        for item in outputs
    }
    assert by_parent["first"]["gap_seeking_current_mode"] == "lcm"
    assert by_parent["first"]["gap_seeking_switched"] is True
    assert by_parent["first"]["gap_seeking_trigger_step"] == 1
    assert by_parent["first"]["gap_seeking_trigger_time_ps"] == 0.0
    assert by_parent["first"]["gap_seeking_pair"] == [0, 1]
    assert by_parent["first"]["gap_seeking_switch_events"][0]["event"] == (
        "enter_lcm"
    )
    # The second candidate was generated on the direct surface at the same
    # check that subsequently triggered its one-way switch.
    assert by_parent["second"]["gap_seeking_current_mode"] == "direct"
    assert by_parent["second"]["gap_seeking_switched"] is False
    # Two ordinary check evaluations plus one refresh after each distinct
    # per-replica entry event.
    assert FakeRunner.last_instance.evaluation_index == 3


def test_gap_seeking_selects_a_different_adjacent_pair_per_replica():
    config = _enable_gap_seeking(
        _config(policy="continue", batch_size=2, return_top_k=2)
    )
    quiet = {"Es": 0.0, "Fs": 0.0, "Fsmax": 0.0}
    uncertain = {"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0}

    def gaps(d01, d12):
        return {
            "dE01": d01,
            "dE01_stdev": 0.0,
            "dE01_model_count": 2,
            "dE12": d12,
            "dE12_stdev": 0.0,
            "dE12_model_count": 2,
        }

    model = FakeModel(
        {
            0: [
                {**quiet, **gaps(0.01, 0.04)},
                {**quiet, **gaps(0.20, 0.20)},
                {**uncertain, **gaps(0.20, 0.20)},
            ],
            1: [
                {**quiet, **gaps(0.04, 0.01)},
                {**quiet, **gaps(0.20, 0.20)},
                {**uncertain, **gaps(0.20, 0.20)},
            ],
        }
    )

    outputs = run_alchemi_sampling(
        [_molecule("lower", state=1), _molecule("upper", state=1)],
        config,
        model,
        properties_list=_excited_properties_three_states(),
        runner_factory=FakeRunner,
    )

    assert [call["pair"] for call in FakeRunner.last_instance.lcm_calls] == [
        [0, 1],
        [1, 2],
    ]
    metadata = {
        item.get_metadata()["parent_molecule_id"]: item.get_metadata()
        for item in outputs
    }
    assert metadata["lower"]["gap_seeking_current_mode"] == "lcm"
    assert metadata["lower"]["gap_seeking_pair"] == [0, 1]
    assert metadata["upper"]["gap_seeking_current_mode"] == "lcm"
    assert metadata["upper"]["gap_seeking_pair"] == [1, 2]


def test_stop_mode_frozen_replica_does_not_enter_lcm():
    config = _enable_gap_seeking(
        _config(policy="stop", batch_size=1, return_top_k=1)
    )
    model = FakeModel(
        {
            0: [
                {
                    "Es": 2.0,
                    "Fs": 0.0,
                    "Fsmax": 0.0,
                    "dE01": 0.01,
                    "dE01_stdev": 0.0,
                    "dE01_model_count": 2,
                }
            ]
        }
    )

    outputs = run_alchemi_sampling(
        [_molecule("frozen", state=0)],
        config,
        model,
        properties_list=_excited_properties_with_gap(),
        runner_factory=FakeRunner,
    )

    assert len(outputs) == 1
    assert FakeRunner.last_instance.lcm_calls == []
    assert outputs[0].get_metadata()["gap_seeking_switched"] is False


def test_distance_rejected_replica_does_not_enter_lcm():
    config = _enable_gap_seeking(
        _config(policy="continue", batch_size=1, return_top_k=1)
    )
    config["distcut"] = 1.0
    model = FakeModel(
        {
            0: [
                {
                    "Es": 0.0,
                    "Fs": 0.0,
                    "Fsmax": 0.0,
                    "dE01": 0.01,
                    "dE01_stdev": 0.0,
                    "dE01_model_count": 2,
                }
            ]
        }
    )

    outputs = run_alchemi_sampling(
        [_molecule("too-close", distance=0.74, state=0)],
        config,
        model,
        properties_list=_excited_properties_with_gap(),
        runner_factory=FakeRunner,
    )

    assert outputs == []
    assert FakeRunner.last_instance.lcm_calls == []


def test_continue_mode_tie_breaks_by_batch_index():
    tied = {"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0}
    model = FakeModel({0: [tied], 1: [tied]})

    outputs = run_alchemi_sampling(
        [_molecule("first"), _molecule("second")],
        _config(policy="continue", return_top_k=1),
        model,
        runner_factory=FakeRunner,
    )

    assert len(outputs) == 1
    assert outputs[0].get_metadata()["parent_molecule_id"] == "first"


def test_excited_batches_require_a_common_selected_state():
    config = _config(policy="stop")
    config["model_mode"] = "excited_state"
    model = FakeModel({0: [], 1: []})

    with pytest.raises(ValueError, match="same state"):
        run_alchemi_sampling(
            [_molecule("first", state=0), _molecule("second", state=1)],
            config,
            model,
            runner_factory=FakeRunner,
        )


def test_batch_cycle_state_is_preserved_in_candidate_provenance():
    config = _config(policy="stop", batch_size=2)
    config["model_mode"] = "excited_state"
    config["state_selection"] = {
        "mode": "batch_cycle",
        "states": [1, 0],
    }
    uncertain = {"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0}
    model = FakeModel({0: [uncertain], 1: [uncertain]})

    outputs = run_alchemi_sampling(
        [_molecule("mol-0000000000"), _molecule("mol-0000000001")],
        config,
        model,
        runner_factory=FakeRunner,
    )

    assert len(outputs) == 2
    assert all(item.get_metadata()["selected_state"] == 1 for item in outputs)
    assert all(
        item.get_metadata()["selected_state_source"]
        == "state_selection.batch_cycle"
        for item in outputs
    )


def test_resolved_state_must_exist_in_properties():
    config = _config(policy="stop", batch_size=1)
    config["model_mode"] = "excited_state"
    properties = {
        "sE0": ["state_0_energy", "system", 1.0],
        "F0": ["state_0_forces", "atomic", 1.0],
        "sE1": ["state_1_energy", "system", 1.0],
        "F1": ["state_1_forces", "atomic", 1.0],
    }

    with pytest.raises(ValueError, match=r"Selected states \[2\].*available states"):
        alchemi_module._validate_sampling_inputs(
            [_molecule("state-two", state=2)],
            config,
            properties,
        )


def test_gap_diagnostics_configuration_requires_excited_gap_properties():
    config = _config(policy="stop", batch_size=1)
    config["gap_diagnostics"] = {"enabled": True}
    with pytest.raises(ValueError, match="excited_state"):
        alchemi_module.configured_gap_diagnostics(config, {})

    config["model_mode"] = "excited_state"
    with pytest.raises(ValueError, match="at least one dE"):
        alchemi_module.configured_gap_diagnostics(
            config,
            {
                "sE0": ["sE0", "system", 1.0],
                "F0": ["F0", "atomic", 1.0],
            },
        )

    config["gap_diagnostics"]["unexpected"] = True
    with pytest.raises(ValueError, match="Unknown"):
        alchemi_module.configured_gap_diagnostics(
            config,
            _excited_properties_with_gap(),
        )


def test_gap_seeking_configuration_selects_only_adjacent_selected_pairs():
    config = _enable_gap_seeking(
        _config(policy="stop", batch_size=1)
    )

    settings = alchemi_module.configured_gap_seeking(
        config,
        _excited_properties_three_states(),
        selected_state_value=1,
    )

    assert [row["gap_key"] for row in settings["rows"]] == [
        "dE01",
        "dE12",
    ]
    assert settings["selected_state"] == 1
    assert settings["trigger_gap_threshold_eV"] == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda config: config.update({"model_mode": "ground_state"}),
            "excited_state",
        ),
        (
            lambda config: config["gap_seeking"].pop(
                "trigger_gap_threshold_eV"
            ),
            "explicit values",
        ),
        (
            lambda config: config["gap_seeking"].update({"sigma": 0.0}),
            "sigma",
        ),
        (
            lambda config: config["gap_seeking"].update(
                {"switch_policy": "hysteresis"}
            ),
            "stay_fixed",
        ),
        (
            lambda config: config["gap_seeking"].update(
                {"candidate_pairs": "all_adjacent"}
            ),
            "adjacent",
        ),
        (
            lambda config: config["gap_seeking"].update({"mode": "other"}),
            "levine_coe_martinez_switch",
        ),
    ],
)
def test_gap_seeking_configuration_rejects_unsupported_values(
    mutator,
    message,
):
    config = _enable_gap_seeking(
        _config(policy="stop", batch_size=1)
    )
    mutator(config)

    with pytest.raises(ValueError, match=message):
        alchemi_module.configured_gap_seeking(
            config,
            _excited_properties_with_gap(),
            selected_state_value=0,
        )


def test_gap_seeking_requires_an_explicit_adjacent_gap_for_selected_state():
    config = _enable_gap_seeking(
        _config(policy="stop", batch_size=1)
    )

    with pytest.raises(ValueError, match="no explicitly configured adjacent"):
        alchemi_module.configured_gap_seeking(
            config,
            {
                "sE0": ["sE0", "system", 1.0],
                "F0": ["F0", "atomic", 1.0],
                "sE1": ["sE1", "system", 1.0],
                "F1": ["F1", "atomic", 1.0],
            },
            selected_state_value=0,
        )


def test_gap_seeking_requires_forces_for_each_eligible_pair_state():
    config = _enable_gap_seeking(
        _config(policy="stop", batch_size=1)
    )

    with pytest.raises(ValueError, match="missing F1"):
        alchemi_module.configured_gap_seeking(
            config,
            {
                "sE0": ["sE0", "system", 1.0],
                "F0": ["F0", "atomic", 1.0],
                "sE1": ["sE1", "system", 1.0],
                "dE01": ["dE01", "system", 1.0],
            },
            selected_state_value=0,
        )


def test_gap_seeking_trigger_uses_minimum_absolute_gap_and_pair_tie_break():
    settings = {
        "enabled": True,
        "trigger_gap_threshold_eV": 0.05,
        "rows": [
            {
                "lower_state": 1,
                "upper_state": 2,
                "gap_key": "dE12",
            },
            {
                "lower_state": 0,
                "upper_state": 1,
                "gap_key": "dE01",
            },
        ],
    }

    trigger = alchemi_module.select_gap_seeking_trigger(
        {"dE01": -0.02, "dE12": 0.02},
        settings,
    )

    assert trigger["pair"] == [0, 1]
    assert trigger["gap_key"] == "dE01"
    assert (
        alchemi_module.select_gap_seeking_trigger(
            {"dE01": 0.06, "dE12": -0.07},
            settings,
        )
        is None
    )


def test_task_gpu_assignment_is_independent_of_selected_state(monkeypatch):
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 4,
        set_device=lambda device: None,
    )
    fake_torch = SimpleNamespace(
        cuda=fake_cuda,
        device=lambda value: value,
    )
    monkeypatch.setattr(alchemi_module, "torch", fake_torch)
    monkeypatch.setattr(alchemi_module, "ensure_alchemi_available", lambda: None)

    loaded = []

    def fake_load(**kwargs):
        loaded.append(
            (
                int(kwargs["selected_state_value"]),
                str(kwargs["device"]),
            )
        )
        return SimpleNamespace()

    def fake_run(molecule_objects, sampler_config, model, *, device):
        del sampler_config, model, device
        return [molecule_objects[0]]

    monkeypatch.setattr(alchemi_module, "load_alchemi_calculator", fake_load)
    monkeypatch.setattr(alchemi_module, "run_alchemi_sampling", fake_run)
    properties = {
        "sE0": ["state_0_energy", "system", 1.0],
        "F0": ["state_0_forces", "atomic", 1.0],
        "sE1": ["state_1_energy", "system", 1.0],
        "F1": ["state_1_forces", "atomic", 1.0],
    }
    config = _config(policy="stop", batch_size=1)
    config["model_mode"] = "excited_state"

    monkeypatch.setenv("PARSL_WORKER_RANK", "3")
    state_zero = alchemi_sampling_task.func(
        molecule_objects=[_molecule("state-zero", state=0)],
        sampler_config=config,
        model_path="models/model-{:04d}",
        current_model_id=0,
        gpus_per_node=4,
        ML_config={},
        properties_list=properties,
    )[0]
    monkeypatch.setenv("PARSL_WORKER_RANK", "1")
    state_one = alchemi_sampling_task.func(
        molecule_objects=[_molecule("state-one", state=1)],
        sampler_config=config,
        model_path="models/model-{:04d}",
        current_model_id=0,
        gpus_per_node=4,
        ML_config={},
        properties_list=properties,
    )[0]

    assert loaded == [(0, "cuda:3"), (1, "cuda:1")]
    assert state_zero.get_metadata()["sampler_device"] == "cuda:3"
    assert state_one.get_metadata()["sampler_device"] == "cuda:1"
    assert state_zero.get_metadata()["sampler_visible_device"] == 3
    assert state_one.get_metadata()["sampler_visible_device"] == 1
    assert state_zero.get_metadata()["selected_state"] == 0
    assert state_one.get_metadata()["selected_state"] == 1


def test_periodic_and_density_sampling_are_rejected():
    config = _config()
    periodic = _molecule("periodic")
    periodic.get_atoms().set_cell(np.eye(3) * 5.0)
    periodic.get_atoms().set_pbc(True)
    with pytest.raises(NotImplementedError, match="nonperiodic"):
        run_alchemi_sampling(
            [periodic, _molecule("second")],
            config,
            FakeModel({}),
            runner_factory=FakeRunner,
        )

    config = _config()
    config["end_dens"] = [1.0, 2.0]
    with pytest.raises(NotImplementedError, match="density"):
        run_alchemi_sampling(
            [_molecule("first"), _molecule("second")],
            config,
            FakeModel({}),
            runner_factory=FakeRunner,
        )


def test_native_model_ensemble_uses_shared_reduction():
    try:
        import torch
        from nvalchemi.models.base import BaseModelMixin, ModelConfig
    except Exception as exc:
        pytest.skip(f"Optional ALCHEMI stack is unavailable: {exc}")

    class ConstantModel(torch.nn.Module, BaseModelMixin):
        def __init__(self, energy, force):
            torch.nn.Module.__init__(self)
            self.energy_value = float(energy)
            self.force_value = float(force)
            self.model_config = ModelConfig(
                outputs=frozenset({"energy", "forces"}),
                autograd_outputs=frozenset(),
                autograd_inputs=frozenset(),
                active_outputs={"energy", "forces"},
            )

        @property
        def embedding_shapes(self):
            return {}

        def compute_embeddings(self, data, **kwargs):
            return data

        def direct_derivative_keys(self):
            return {"forces"}

        def forward(self, batch):
            return {
                "energy": torch.full(
                    (batch.num_graphs, 1),
                    self.energy_value,
                    device=batch.positions.device,
                ),
                "forces": torch.full_like(batch.positions, self.force_value),
            }

    model = ALFNativeEnsembleModel(
        [ConstantModel(1.0, 1.0), ConstantModel(3.0, 3.0)],
        device=torch.device("cpu"),
    )
    output = model(_torch_batch(torch))
    diagnostics = model.diagnostics_for_graph(0)

    np.testing.assert_allclose(output["energy"].detach().numpy(), [[2.0], [2.0]])
    np.testing.assert_allclose(output["forces"].detach().numpy(), np.full((4, 3), 2.0))
    assert diagnostics["Es"] == pytest.approx(1.0)
    assert diagnostics["Fs"] == pytest.approx(1.0)
    assert diagnostics["Fsmax"] == pytest.approx(1.0)

    seeking_model = ALFNativeEnsembleModel(
        [ConstantModel(1.0, 1.0), ConstantModel(3.0, 3.0)],
        selected_state_value=0,
        gap_rows=[_gap_row()],
        gap_seeking_settings={
            "enabled": True,
            "selected_state": 0,
            "rows": [_gap_row()],
            "sigma": 3.5,
            "alpha_eV": 0.05,
        },
        device=torch.device("cpu"),
    )
    with pytest.raises(KeyError, match="state_contributions"):
        seeking_model(_torch_batch(torch))


def test_single_plain_ase_calculator_has_zero_uncertainty():
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"Optional Torch stack is unavailable: {exc}")

    model = ALFASEAlchemiModel(
        FixedCalculator(1.0, np.zeros((2, 3))),
        model_mode="ground_state",
        selected_state_value=None,
        device=torch.device("cpu"),
    )
    model(_torch_batch(torch))

    assert model.diagnostics_for_graph(0)["Es"] == 0.0
    assert model.diagnostics_for_graph(0)["Fs"] == 0.0
    assert model.diagnostics_for_graph(0)["Fsmax"] == 0.0


def test_calculator_loader_prefers_native_and_forwards_options(monkeypatch):
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"Optional Torch stack is unavailable: {exc}")

    calls = []

    class RawCalculator(ALFAlchemiCalculator):
        def ensemble_forward(self, batch):
            return {
                "energy_contributions": torch.zeros(
                    (1, batch.num_graphs), device=batch.positions.device
                ),
                "force_contributions": torch.zeros(
                    (1, batch.positions.shape[0], 3), device=batch.positions.device
                ),
            }

    def native_loader(directory, **kwargs):
        calls.append((directory, kwargs))
        return RawCalculator(device=kwargs["device"])

    monkeypatch.setattr(alchemi_module, "load_module_from_string", lambda path: native_loader)
    config = _config()
    config.update(
        {
            "alchemi_calculator": "custom.native_loader",
            "alchemi_calculator_options": {"custom_option": 7},
            "ase_calculator": "custom.ase_loader",
        }
    )
    model = load_alchemi_calculator(
        sampler_config=config,
        ensemble_directory="models/0001",
        model_mode="ground_state",
        selected_state_value=None,
        ML_config={},
        properties_list={},
        device=torch.device("cpu"),
    )

    assert calls[0][0] == "models/0001"
    assert calls[0][1]["custom_option"] == 7
    assert model.calculator_interface == "native"
    assert model.calculator_loader == "custom.native_loader"
    assert alchemi_calculator_status(config) == {
        "interface": "native",
        "loader": "custom.native_loader",
    }


def test_calculator_loader_uses_ase_fallback_and_requires_a_loader(monkeypatch):
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"Optional Torch stack is unavailable: {exc}")

    calls = []

    def ase_loader(directory, **kwargs):
        calls.append((directory, kwargs))
        return FixedCalculator(1.0, np.zeros((2, 3)))

    monkeypatch.setattr(alchemi_module, "load_module_from_string", lambda path: ase_loader)
    config = _config()
    config.update(
        {
            "ase_calculator": "custom.ase_loader",
            "ase_calculator_options": {"custom_option": 8},
        }
    )
    model = load_alchemi_calculator(
        sampler_config=config,
        ensemble_directory="models/0002",
        model_mode="ground_state",
        selected_state_value=None,
        ML_config={},
        properties_list={},
        device=torch.device("cpu"),
    )

    assert isinstance(model, ALFASEAlchemiModel)
    assert calls == [
        ("models/0002/", {"custom_option": 8, "device": "cpu"})
    ]
    assert alchemi_calculator_status(config)["interface"] == "ase_fallback"

    with pytest.raises(ValueError, match="either alchemi_calculator"):
        load_alchemi_calculator(
            sampler_config=_config(),
            ensemble_directory="models/0002",
            model_mode="ground_state",
            selected_state_value=None,
            ML_config={},
            properties_list={},
            device=torch.device("cpu"),
        )


def test_installed_alchemi_cpu_dynamics_smoke():
    try:
        import torch
        from nvalchemi.models.base import BaseModelMixin, ModelConfig
    except Exception as exc:
        pytest.skip(f"Optional ALCHEMI stack is unavailable: {exc}")

    class HarmonicModel(torch.nn.Module, BaseModelMixin):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.model_config = ModelConfig(
                outputs=frozenset({"energy", "forces"}),
                autograd_outputs=frozenset(),
                autograd_inputs=frozenset(),
                active_outputs=frozenset({"energy", "forces"}),
            )
            self.last_diagnostics = {}

        @property
        def embedding_shapes(self):
            return {}

        def compute_embeddings(self, data, **kwargs):
            return data

        def direct_derivative_keys(self):
            return {"forces"}

        def set_device(self, device):
            self.device = torch.device(device)

        def forward(self, batch):
            per_atom = 0.5 * torch.sum(batch.positions * batch.positions, dim=1)
            energy = torch.zeros(
                (batch.num_graphs, 1),
                dtype=batch.positions.dtype,
                device=batch.positions.device,
            )
            energy[:, 0].scatter_add_(0, batch.batch_idx.long(), per_atom)
            zeros = torch.zeros(
                batch.num_graphs,
                dtype=batch.positions.dtype,
                device=batch.positions.device,
            )
            self.last_diagnostics = {"Es": zeros, "Fs": zeros, "Fsmax": zeros}
            return {"energy": energy, "forces": -batch.positions}

        def diagnostics_for_graph(self, graph_index):
            return {
                key: float(value[int(graph_index)].detach().cpu().item())
                for key, value in self.last_diagnostics.items()
            }

    runner = AlchemiDynamicsRunner(
        model=HarmonicModel(),
        atoms_list=[_molecule("first").get_atoms(), _molecule("second").get_atoms()],
        dt_fs=0.1,
        temperature_K=np.asarray([10.0, 20.0]),
        friction_per_fs=0.01,
        random_seed=4,
        device=torch.device("cpu"),
    )
    runner.run(1)
    frozen_atoms = _molecule("frozen").get_atoms()
    moving_atoms = _molecule("moving").get_atoms()
    runner.sync_graph_to_atoms(0, frozen_atoms)
    runner.sync_graph_to_atoms(1, moving_atoms)
    frozen_positions = frozen_atoms.get_positions().copy()
    moving_positions = moving_atoms.get_positions().copy()
    runner.freeze_graph(0)
    runner.set_temperature(np.asarray([15.0, 25.0]))
    runner.run(2)
    runner.evaluate()
    atoms = _molecule("result").get_atoms()
    runner.sync_graph_to_atoms(0, atoms)
    runner.sync_graph_to_atoms(1, moving_atoms)

    assert runner.nsteps == 3
    assert np.all(np.isfinite(atoms.get_positions()))
    np.testing.assert_allclose(atoms.get_positions(), frozen_positions)
    assert not np.allclose(moving_atoms.get_positions(), moving_positions)
    assert runner.diagnostics_for_graph(0) == {"Es": 0.0, "Fs": 0.0, "Fsmax": 0.0}


def test_hippynn_adapter_uses_raw_members_and_selects_excited_state(monkeypatch):
    try:
        import hippynn
        import torch
    except Exception as exc:
        pytest.skip(f"Optional HIPPYNN stack is unavailable: {exc}")

    class Node:
        def __init__(self, label):
            self.label = label

    class InputNode:
        def __init__(self, name):
            self.name = name
            self.db_name = name

    class Graph:
        input_nodes = [InputNode("species"), InputNode("coordinates")]

    class Predictor:
        def __init__(self, inputs, outputs, **kwargs):
            del inputs, kwargs
            self.outputs = outputs
            self.model_device = None
            self.return_device = None

        def __call__(self, **kwargs):
            coordinates = kwargs["coordinates"]
            batch_size, atom_count = coordinates.shape[:2]
            values = {}
            for node in self.outputs:
                if node.label.endswith("energy_all"):
                    state = int(node.label[0])
                    center = 2.0 + 2.0 * state
                    values[node] = torch.tensor(
                        [[[center - 1.0], [center + 1.0]]] * batch_size,
                        device=coordinates.device,
                    )
                elif node.label.endswith("force_all"):
                    state = int(node.label[0])
                    center = 1.0 + 2.0 * state
                    low = torch.full(
                        (batch_size, atom_count, 3), center - 1.0,
                        device=coordinates.device,
                    )
                    high = torch.full(
                        (batch_size, atom_count, 3), center + 1.0,
                        device=coordinates.device,
                    )
                    values[node] = torch.stack([low, high], dim=1)
            return values

    monkeypatch.setattr(hippynn.graphs, "Predictor", Predictor)
    state_nodes = []
    for state in (0, 1):
        state_nodes.append(
                {
                    "state": state,
                    "energy_all_node": Node(f"{state}_energy_all"),
                    "force_all_node": Node(f"{state}_force_all"),
                    "energy_model_count": 2,
                    "force_model_count": 2,
                }
        )
    model = ALFHippynnAlchemiModel(
        ensemble_graph=Graph(),
        state_nodes=state_nodes,
        selected_state_value=1,
        gap_rows=[_gap_row()],
        gap_seeking_settings={
            "enabled": True,
            "selected_state": 1,
            "rows": [_gap_row()],
            "sigma": 3.5,
            "alpha_eV": 0.05,
        },
        species_key="species",
        coordinates_key="coordinates",
        device=torch.device("cpu"),
    )

    class Batch:
        positions = torch.zeros((4, 3))
        atomic_numbers = torch.ones(4, dtype=torch.long)
        atomic_masses = torch.ones(4)
        num_graphs = 2
        num_nodes_per_graph = torch.as_tensor([2, 2])

    output = model(Batch())
    diagnostics = model.diagnostics_for_graph(0)

    np.testing.assert_allclose(output["energy"].detach().numpy(), [[4.0], [4.0]])
    np.testing.assert_allclose(output["forces"].detach().numpy(), np.full((4, 3), 3.0))
    assert diagnostics["sE0"] == pytest.approx(2.0)
    assert diagnostics["sE1"] == pytest.approx(4.0)
    assert diagnostics["Es"] == pytest.approx(1.0)
    assert diagnostics["Fs"] == pytest.approx(1.0)
    assert diagnostics["Fsmax"] == pytest.approx(1.0)
    assert diagnostics["dE01"] == pytest.approx(2.0)
    assert diagnostics["dE01_stdev"] == pytest.approx(0.0)
    assert diagnostics["dE01_model_count"] == 2

    model.well_params = {
        "r_start": 1.0,
        "force": 2.0,
        "origin": [0.0, 0.0, 0.0],
        "mass_weighted": False,
    }
    Batch.positions = torch.tensor([[2.0, 0.0, 0.0]] * 4)
    output_with_well = model(Batch())
    np.testing.assert_allclose(
        output_with_well["energy"].detach().numpy(), [[8.0], [8.0]]
    )
    np.testing.assert_allclose(
        output_with_well["forces"].detach().numpy()[:, 0], np.ones(4)
    )
    with_well_diagnostics = model.diagnostics_for_graph(0)
    assert with_well_diagnostics["dE01"] == pytest.approx(2.0)
    assert with_well_diagnostics["dE01_stdev"] == pytest.approx(0.0)
    model.well_params = None
    model.set_lcm_mode(
        0,
        pair=[0, 1],
        sigma=3.5,
        alpha_eV=0.05,
    )
    lcm_output = model(Batch())
    expected_energy, expected_forces = model._lcm_energy_forces(
        torch.tensor(2.0),
        torch.tensor(4.0),
        torch.full((2, 3), 1.0),
        torch.full((2, 3), 3.0),
        sigma=3.5,
        alpha_eV=0.05,
    )
    assert lcm_output["energy"][0, 0].item() == pytest.approx(
        expected_energy.item()
    )
    np.testing.assert_allclose(
        lcm_output["forces"][:2].detach().numpy(),
        expected_forces.detach().numpy(),
        atol=1.0e-6,
    )
    assert lcm_output["energy"][1, 0].item() == pytest.approx(4.0)


def _score_config(**overrides):
    score = {
        "mode": "sum",
        "w_energy": 1.0,
        "w_force": 1.0,
        "w_gap_uncertainty": 5.0,
        "gap_std_cut_eV": 0.01,
    }
    score.update(overrides)
    return score


def _sum_config(score=None, **kwargs):
    config = _config(**kwargs)
    config["model_mode"] = "excited_state"
    config["score"] = _score_config() if score is None else score
    return config


def _sum_diagnostics(*, Es, Fsrms, gap_stds, Fs=0.0, Fsmax=0.0):
    diagnostics = {"Es": Es, "Fs": Fs, "Fsmax": Fsmax, "Fsrms": Fsrms}
    for gap_key, value in gap_stds.items():
        diagnostics[gap_key] = 1.0
        diagnostics[f"{gap_key}_stdev"] = value
    return diagnostics


def test_absent_score_block_preserves_max_ranking():
    score = configured_score(_config(), None, None)

    assert score["mode"] == "max"
    assert score["gap_rows"] == []

    diagnostics = {"Es": 2.0, "Fs": 1.5, "Fsmax": 1.0}
    flags = uncertainty_flags(diagnostics, Escut=1.0, Fscut=1.0)
    result = candidate_score(diagnostics, flags, score, Escut=1.0, Fscut=1.0)

    assert result["score_mode"] == "max"
    assert result["ranking_score"] == flags["uncertainty_score"] == 2.0
    assert result["score_components"] == {}
    assert result["score_gap_pair"] is None


def test_explicit_max_mode_matches_absent_block():
    diagnostics = {"Es": 0.5, "Fs": 3.0, "Fsmax": 1.0}
    flags = uncertainty_flags(diagnostics, Escut=1.0, Fscut=1.0)

    absent = candidate_score(
        diagnostics, flags, configured_score(_config(), None, None),
        Escut=1.0, Fscut=1.0,
    )
    config = _config()
    config["score"] = {"mode": "max"}
    explicit = candidate_score(
        diagnostics, flags, configured_score(config, None, None),
        Escut=1.0, Fscut=1.0,
    )

    assert explicit == absent
    assert explicit["ranking_score"] == 3.0


def test_summed_score_adds_normalized_terms():
    config = _sum_config()
    config["Escut"] = 0.001
    config["Fscut"] = 0.01
    score = configured_score(config, _excited_properties_with_gap(), 0)
    diagnostics = _sum_diagnostics(
        Es=0.002, Fsrms=0.03, gap_stds={"dE01": 0.02}
    )
    flags = uncertainty_flags(
        {**diagnostics, "Fs": 0.02, "Fsmax": 0.05}, Escut=0.001, Fscut=0.01
    )

    result = candidate_score(
        diagnostics, flags, score, Escut=0.001, Fscut=0.01
    )

    # 1.0*(0.002/0.001) + 1.0*(0.03/0.01) + 5.0*(0.02/0.01) = 2 + 3 + 10
    assert result["score_components"] == {
        "energy_uncertainty": pytest.approx(2.0),
        "force_uncertainty": pytest.approx(3.0),
        "gap_uncertainty": pytest.approx(10.0),
    }
    assert result["ranking_score"] == pytest.approx(15.0)
    assert result["score_mode"] == "sum"
    assert result["score_gap_pair"] == [0, 1]
    assert result["score_gap_std"] == pytest.approx(0.02)


def test_summed_score_uses_maximum_gap_deviation_over_pairs():
    config = _sum_config(score=_score_config(gap_pairs="adjacent"))
    score = configured_score(config, _excited_properties_three_states(), 1)
    diagnostics = _sum_diagnostics(
        Es=0.0, Fsrms=0.0, gap_stds={"dE01": 0.004, "dE12": 0.02}
    )
    flags = uncertainty_flags(diagnostics, Escut=1.0, Fscut=1.0)

    result = candidate_score(diagnostics, flags, score, Escut=1.0, Fscut=1.0)

    assert result["score_gap_pair"] == [1, 2]
    assert result["score_gap_std"] == pytest.approx(0.02)
    assert result["ranking_score"] == pytest.approx(5.0 * 0.02 / 0.01)


def test_summed_score_weights_change_candidate_order():
    """The gap term can outrank a frame that wins on force alone."""

    diagnostics_gap = _sum_diagnostics(
        Es=0.0, Fsrms=0.01, gap_stds={"dE01": 0.02}
    )
    diagnostics_force = _sum_diagnostics(
        Es=0.0, Fsrms=0.05, gap_stds={"dE01": 0.001}
    )
    flags = uncertainty_flags(diagnostics_gap, Escut=1.0, Fscut=1.0)
    properties = _excited_properties_with_gap()

    gap_heavy = configured_score(
        _sum_config(score=_score_config(w_gap_uncertainty=5.0)),
        properties,
        0,
    )
    force_only = configured_score(
        _sum_config(score=_score_config(w_gap_uncertainty=0.0)),
        properties,
        0,
    )

    def rank(score, diagnostics):
        return candidate_score(
            diagnostics, flags, score, Escut=1.0, Fscut=0.01
        )["ranking_score"]

    assert rank(gap_heavy, diagnostics_gap) > rank(gap_heavy, diagnostics_force)
    assert rank(force_only, diagnostics_force) > rank(force_only, diagnostics_gap)
    assert force_only["gap_rows"] == []


def test_selected_adjacent_gap_pairs_track_the_selected_state():
    properties = _excited_properties_three_states()
    config = _sum_config()

    assert [
        row["gap_key"] for row in configured_score(config, properties, 0)["gap_rows"]
    ] == ["dE01"]
    assert [
        row["gap_key"] for row in configured_score(config, properties, 1)["gap_rows"]
    ] == ["dE01", "dE12"]
    assert [
        row["gap_key"] for row in configured_score(config, properties, 2)["gap_rows"]
    ] == ["dE12"]


def test_adjacent_and_explicit_gap_pairs_are_normalized():
    properties = _excited_properties_three_states()

    adjacent = configured_score(
        _sum_config(score=_score_config(gap_pairs="adjacent")), properties, 0
    )
    assert [row["gap_key"] for row in adjacent["gap_rows"]] == ["dE01", "dE12"]

    explicit = configured_score(
        _sum_config(score=_score_config(gap_pairs=[[2, 0], [0, 2]])),
        properties,
        0,
    )
    assert [row["gap_key"] for row in explicit["gap_rows"]] == ["dE02"]
    assert explicit["gap_rows"][0]["lower_state"] == 0
    assert explicit["gap_rows"][0]["upper_state"] == 2


def test_score_gap_pairs_reach_the_calculator_without_gap_diagnostics():
    config = _sum_config()
    config["gap_diagnostics"] = {"enabled": False}

    rows, seeking = configured_alchemi_gap_rows(
        config, _excited_properties_three_states(), 1
    )

    assert [row["gap_key"] for row in rows] == ["dE01", "dE12"]
    assert seeking["enabled"] is False


def test_score_configuration_rejects_invalid_options():
    properties = _excited_properties_with_gap()

    with pytest.raises(ValueError, match="Unknown score options: w_gap"):
        configured_score(_sum_config(score=_score_config(w_gap=1.0)), properties, 0)
    with pytest.raises(ValueError, match="score.mode must be"):
        configured_score(_sum_config(score={"mode": "mean"}), properties, 0)
    with pytest.raises(ValueError, match="score.w_force must be"):
        configured_score(
            _sum_config(score=_score_config(w_force=-1.0)), properties, 0
        )
    with pytest.raises(ValueError, match="score.w_energy must be"):
        configured_score(
            _sum_config(score=_score_config(w_energy=float("nan"))), properties, 0
        )
    with pytest.raises(ValueError, match="strictly positive weight"):
        configured_score(
            _sum_config(
                score={
                    "mode": "sum",
                    "w_energy": 0.0,
                    "w_force": 0.0,
                    "w_gap_uncertainty": 0.0,
                }
            ),
            properties,
            0,
        )
    with pytest.raises(TypeError, match="score must be a dictionary"):
        configured_score(_sum_config(score=[1.0]), properties, 0)


def test_gap_term_requires_cutoff_states_and_excited_mode():
    properties = _excited_properties_with_gap()
    missing_cut = _sum_config()
    del missing_cut["score"]["gap_std_cut_eV"]

    with pytest.raises(ValueError, match="requires score.gap_std_cut_eV"):
        configured_score(missing_cut, properties, 0)
    with pytest.raises(ValueError, match="gap_std_cut_eV must be finite"):
        configured_score(
            _sum_config(score=_score_config(gap_std_cut_eV=0.0)), properties, 0
        )

    ground_state = _sum_config()
    ground_state["model_mode"] = "ground_state"
    with pytest.raises(ValueError, match="requires model_mode='excited_state'"):
        configured_score(ground_state, properties, 0)

    with pytest.raises(ValueError, match="requires properties_list"):
        configured_score(_sum_config(), None, 0)
    with pytest.raises(ValueError, match="are not present in properties_list"):
        configured_score(
            _sum_config(score=_score_config(gap_pairs=[[0, 7]])), properties, 0
        )
    with pytest.raises(ValueError, match="cannot reference one state twice"):
        configured_score(
            _sum_config(score=_score_config(gap_pairs=[[1, 1]])), properties, 0
        )
    with pytest.raises(ValueError, match="score.gap_pairs must be"):
        configured_score(
            _sum_config(score=_score_config(gap_pairs="all")), properties, 0
        )
    with pytest.raises(ValueError, match="no adjacent state pair"):
        configured_score(_sum_config(), properties, 5)


def test_summed_score_requires_ensemble_force_deviation():
    score = configured_score(_sum_config(), _excited_properties_with_gap(), 0)
    diagnostics = {"Es": 1.0, "Fs": 1.0, "Fsmax": 1.0, "dE01_stdev": 0.0}
    flags = uncertainty_flags(diagnostics, Escut=1.0, Fscut=1.0)

    with pytest.raises(ValueError, match="requires the Fsrms force deviation"):
        candidate_score(diagnostics, flags, score, Escut=1.0, Fscut=1.0)


def test_summed_score_requires_configured_gap_diagnostic():
    score = configured_score(_sum_config(), _excited_properties_with_gap(), 0)
    diagnostics = _sum_diagnostics(Es=1.0, Fsrms=1.0, gap_stds={})
    flags = uncertainty_flags(diagnostics, Escut=1.0, Fscut=1.0)

    with pytest.raises(ValueError, match="requires calculator diagnostic"):
        candidate_score(diagnostics, flags, score, Escut=1.0, Fscut=1.0)


def test_reduce_contributions_returns_root_mean_square_force_deviation():
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"Optional Torch stack is unavailable: {exc}")

    class RawEnsemble(ALFAlchemiCalculator):
        def ensemble_forward(self, batch):
            energy = torch.tensor(
                [[1.0, 1.0], [3.0, 3.0]], device=batch.positions.device
            )
            member_one = torch.zeros(4, 3, device=batch.positions.device)
            member_two = torch.zeros(4, 3, device=batch.positions.device)
            # Population deviations of [3, 1, 0] eV/Angstrom on graph zero.
            member_two[0, 0] = 6.0
            member_two[0, 1] = 2.0
            return {
                "energy_contributions": energy,
                "force_contributions": torch.stack([member_one, member_two]),
            }

    model = RawEnsemble(device=torch.device("cpu"))
    model(_torch_batch(torch))
    diagnostics = model.diagnostics_for_graph(0)

    deviations = np.array([3.0, 1.0] + [0.0] * 4)
    assert diagnostics["Fsrms"] == pytest.approx(
        np.sqrt(np.mean(deviations ** 2))
    )
    assert diagnostics["Fs"] == pytest.approx(np.mean(np.abs(deviations)))
    assert diagnostics["Fsmax"] == pytest.approx(3.0)
    # RMS sits strictly between the mean-absolute and maximum reductions.
    assert diagnostics["Fs"] < diagnostics["Fsrms"] < diagnostics["Fsmax"]


def test_sum_mode_ranks_gap_uncertainty_above_force_uncertainty():
    """Replica one loses on the max score but wins on the summed score."""

    config = _sum_config(policy="continue", batch_size=2, return_top_k=2)
    config["Escut"] = 0.001
    config["Fscut"] = 0.01
    config["max_candidates_per_replica"] = 1
    model = FakeModel(
        {
            0: [
                _sum_diagnostics(
                    Es=0.0,
                    Fs=0.05,
                    Fsmax=0.05,
                    Fsrms=0.05,
                    gap_stds={"dE01": 0.0},
                )
            ],
            1: [
                _sum_diagnostics(
                    Es=0.0,
                    Fs=0.02,
                    Fsmax=0.02,
                    Fsrms=0.02,
                    gap_stds={"dE01": 0.05},
                )
            ],
        }
    )

    outputs = run_alchemi_sampling(
        [_molecule("first", state=0), _molecule("second", state=0)],
        config,
        model,
        properties_list=_excited_properties_with_gap(),
        runner_factory=FakeRunner,
    )

    metadata = [item.get_metadata() for item in outputs]
    # The max score prefers the higher force deviation ...
    assert metadata[1]["uncertainty_score"] > metadata[0]["uncertainty_score"]
    # ... while the summed score prefers the gap-uncertain frame.
    assert [row["parent_molecule_id"] for row in metadata] == ["second", "first"]
    assert [row["score_mode"] for row in metadata] == ["sum", "sum"]
    assert metadata[0]["ranking_score"] == pytest.approx(2.0 + 25.0)
    assert metadata[1]["ranking_score"] == pytest.approx(5.0)
    assert metadata[0]["score_components"]["gap_uncertainty"] == pytest.approx(25.0)
    assert metadata[0]["score_gap_pair"] == [0, 1]
    assert metadata[0]["Fsrms"] == pytest.approx(0.02)
    # The gap pair is monitored even though gap_diagnostics stays disabled.
    assert metadata[0]["gap_stds"]["dE01"] == pytest.approx(0.05)


def test_replica_candidate_limit_spreads_output_across_trajectories():
    config = _config(policy="continue", batch_size=2, return_top_k=3)
    config["maxt"] = 0.4
    config["max_candidates_per_replica"] = 1
    model = FakeModel(
        {
            0: [
                {"Es": 9.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 8.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 7.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 6.0, "Fs": 0.0, "Fsmax": 0.0},
            ],
            1: [
                {"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0},
            ],
        }
    )

    outputs = run_alchemi_sampling(
        [_molecule("first"), _molecule("second")],
        config,
        model,
        runner_factory=FakeRunner,
    )

    parents = [item.get_metadata()["parent_molecule_id"] for item in outputs]
    assert parents == ["first", "second"]
    assert [item.get_metadata()["ranking_score"] for item in outputs] == [9.0, 2.0]


def test_absent_replica_candidate_limit_allows_one_trajectory_to_dominate():
    config = _config(policy="continue", batch_size=2, return_top_k=3)
    config["maxt"] = 0.4
    model = FakeModel(
        {
            0: [
                {"Es": 9.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 8.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 7.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 6.0, "Fs": 0.0, "Fsmax": 0.0},
            ],
            1: [
                {"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0},
                {"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0},
            ],
        }
    )

    outputs = run_alchemi_sampling(
        [_molecule("first"), _molecule("second")],
        config,
        model,
        runner_factory=FakeRunner,
    )

    parents = [item.get_metadata()["parent_molecule_id"] for item in outputs]
    assert parents == ["first", "first", "first"]


def test_replica_candidate_limit_rejects_invalid_values():
    model = FakeModel({0: [{"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0}]})

    for value in (0, -1, 1.5, True):
        config = _config(policy="continue", batch_size=1, return_top_k=1)
        config["max_candidates_per_replica"] = value
        with pytest.raises(ValueError, match="max_candidates_per_replica"):
            run_alchemi_sampling(
                [_molecule("first")],
                config,
                model,
                runner_factory=FakeRunner,
            )


def test_renamed_return_top_n_key_is_rejected():
    """The old key must error rather than silently defaulting to one candidate."""

    config = _config(policy="continue", batch_size=1, return_top_k=2)
    del config["return_top_k"]
    config["return_top_n"] = 2
    model = FakeModel({0: [{"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0}]})

    with pytest.raises(ValueError, match="return_top_n was renamed to return_top_k"):
        run_alchemi_sampling(
            [_molecule("first")],
            config,
            model,
            runner_factory=FakeRunner,
        )


def test_return_top_k_rejects_invalid_values():
    model = FakeModel({0: [{"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0}]})

    for value in (0, -1, 1.5, True):
        config = _config(policy="continue", batch_size=1)
        config["return_top_k"] = value
        with pytest.raises(ValueError, match="return_top_k"):
            run_alchemi_sampling(
                [_molecule("first")],
                config,
                model,
                runner_factory=FakeRunner,
            )
