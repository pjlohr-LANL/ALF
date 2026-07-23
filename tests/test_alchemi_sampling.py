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
    load_alchemi_calculator,
    run_alchemi_sampling,
    uncertainty_flags,
)
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import annealing_schedule
from tests.helpers.fakes import FixedCalculator


def _molecule(molecule_id, distance=0.74, state=None):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
    molecule = MoleculesObject(atoms, molecule_id)
    if state is not None:
        molecule.update_metadata({"selected_state": state})
    return molecule


def _config(policy="stop", batch_size=2, return_top_n=2):
    return {
        "model_mode": "ground_state",
        "uncertainty_policy": policy,
        "return_top_n": return_top_n,
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


class FakeModel:
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics


class FakeRunner:
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
        self.model = model
        self.initial_temperature_K = np.asarray(temperature_K, dtype=float)
        self.friction_per_fs = float(friction_per_fs)
        self.temperature_calls = []
        self.positions = [atoms.get_positions().copy() for atoms in atoms_list]
        self.velocities = [np.zeros_like(value) for value in self.positions]
        self.frozen = set()
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

    def set_temperature(self, temperature_K):
        self.temperature_K = np.asarray(temperature_K)
        self.temperature_calls.append(self.temperature_K.copy())

    def sync_graph_to_atoms(self, graph_index, atoms):
        atoms.set_positions(self.positions[int(graph_index)])
        atoms.set_velocities(self.velocities[int(graph_index)])


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
        _config(policy="continue", return_top_n=2),
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


def test_continue_mode_tie_breaks_by_batch_index():
    tied = {"Es": 2.0, "Fs": 0.0, "Fsmax": 0.0}
    model = FakeModel({0: [tied], 1: [tied]})

    outputs = run_alchemi_sampling(
        [_molecule("first"), _molecule("second")],
        _config(policy="continue", return_top_n=1),
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
