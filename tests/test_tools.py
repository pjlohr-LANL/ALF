import json

import h5py
import numpy as np
import pytest
from ase import Atoms

from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.tools import (
    annealing_schedule,
    build_input_dict,
    compute_empirical_formula,
    load_config_file,
    pair_atomic_property,
    random_rotation_matrix,
    store_current_data,
    system_checker,
)


def test_annealing_schedule_linear():
    assert annealing_schedule(t=0.0, tmax=10.0, amp=0.0, per=2.0, srt=100.0, end=300.0) == 100.0
    assert annealing_schedule(t=5.0, tmax=10.0, amp=0.0, per=2.0, srt=100.0, end=300.0) == 200.0
    assert annealing_schedule(t=10.0, tmax=10.0, amp=0.0, per=2.0, srt=100.0, end=300.0) == 300.0


def test_annealing_schedule_with_sinusoidal_oscillation():
    result = annealing_schedule(t=1.0, tmax=10.0, amp=50.0, per=2.0, srt=100.0, end=300.0)

    assert np.isclose(result, 170.0)


def test_compute_empirical_formula():
    assert compute_empirical_formula(["H", "O", "H", "C"]) == "C01_H02_O01"


def test_random_rotation_matrix():
    rotation = random_rotation_matrix(randnums=np.array([0.25, 0.5, 0.75]))

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.isclose(abs(np.linalg.det(rotation)), 1.0)


def test_system_checker_accepts_valid_system_and_rejects_bad_data():
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    valid = [{"moleculeid": "h2"}, atoms, {"energy": np.array([-1.0])}]

    assert system_checker(valid)
    assert not system_checker([{"moleculeid": "bad"}, atoms, {"forces": np.array([np.nan])}], kill_on_fail=False)
    with pytest.raises(RuntimeError):
        system_checker([{}, atoms, {}])


def test_load_config_file_paths(tmp_path):
    config_path = tmp_path / "master.json"
    config_path.write_text(
        json.dumps(
            {
                "master_directory": "pwd",
                "scratch_dir": "scratch",
                "model_path": "models/model-{:04d}",
                "absolute_dir": "/already/absolute",
            }
        )
    )

    config = load_config_file(str(config_path))

    assert config["master_directory"].endswith("/")
    assert config["scratch_dir"].endswith("/scratch")
    assert config["model_path"].endswith("/models/model-{:04d}")
    assert config["model_dir"].endswith("/models/")
    assert config["absolute_dir"] == "/already/absolute"


def test_store_current_data_writes_converged_input_order_h5(tmp_path):
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.7, 0.7], [0.0, -0.7, 0.7]],
        cell=np.eye(3) * 8.0,
        pbc=True,
    )
    molecule = MoleculesObject(atoms, "water-0")
    molecule.store_results({"energy": -1.5, "forces": np.arange(9).reshape(3, 3)})
    molecule.set_converged_flag(True)

    unconverged = MoleculesObject(atoms.copy(), "skip-me")
    unconverged.store_results({"energy": 0.0, "forces": np.zeros((3, 3))})
    unconverged.set_converged_flag(False)

    h5_path = tmp_path / "data.h5"
    store_current_data(
        str(h5_path),
        [molecule, unconverged],
        {"energy": ["energy", "system", 2.0], "forces": ["forces", "atomic", 0.5]},
    )

    with h5py.File(h5_path, "r") as h5:
        assert list(h5.keys()) == ["H02_O01"]
        group = h5["H02_O01"]
        # Atoms are stored in the order they arrive, not sorted by atomic
        # number, so the shard is already in reference-topology order.
        assert [item.decode("utf-8") for item in group["species"][()]] == ["O", "H", "H"]
        assert [item.decode("utf-8") for item in group["_id"][()]] == ["water-0"]
        np.testing.assert_allclose(group["energy"][()], [-3.0])
        assert group["forces"].shape == (1, 3, 3)
        assert group["cell"].shape == (1, 3, 3)
        # Per-frame atom mapping is recorded explicitly so replay never has to
        # infer it from a storage-order convention.
        assert group["topology_atom_ids"].shape == (1, 3)
        np.testing.assert_array_equal(group["topology_atom_ids"][()], [[0, 1, 2]])
        np.testing.assert_allclose(
            group["forces"][()],
            np.arange(9).reshape(1, 3, 3) * 0.5,
        )


def _labeled_molecule(molecule_id, *, nacr, nac_pairs, energy=-1.0):
    """One converged three-atom molecule carrying energy, force, and coupling."""

    molecule = MoleculesObject(
        Atoms(
            "OH2",
            positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        ),
        molecule_id,
    )
    molecule.store_results(
        {
            "sE0": energy,
            "F0": np.zeros((3, 3)),
            "nacr": np.asarray(nacr, dtype=np.float64),
        }
    )
    molecule.update_metadata({"nac_pairs": nac_pairs})
    molecule.set_converged_flag(True)
    return molecule


def _nacr_properties():
    return {
        "sE0": ["sE0", "system", 1.0],
        "F0": ["F0", "atomic", 1.0],
        "nacr": ["nacr", "pair_atomic", 1.0],
    }


def test_pair_atomic_property_permutes_the_atom_axis_only():
    # [pairs, atoms, 3] where each atom slice is identifiable by its value.
    values = np.stack(
        [
            np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]),
            np.asarray([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]),
        ]
    )
    reordered = pair_atomic_property(values, np.asarray([2, 0, 1]), "nacr")

    assert reordered.shape == (2, 3, 3)
    # Atoms permuted within each pair; the pair axis order is untouched.
    np.testing.assert_allclose(reordered[0, :, 0], [3.0, 1.0, 2.0])
    np.testing.assert_allclose(reordered[1, :, 0], [6.0, 4.0, 5.0])


def test_pair_atomic_property_rejects_wrong_rank_and_atom_count():
    with pytest.raises(RuntimeError, match="pairs, 3, 3"):
        pair_atomic_property(np.zeros((3, 3)), np.arange(3), "nacr")
    with pytest.raises(RuntimeError, match="pairs, 3, 3"):
        pair_atomic_property(np.zeros((2, 4, 3)), np.arange(3), "nacr")


def test_store_current_data_writes_pair_atomic_and_one_pair_index(tmp_path):
    pairs = [[1, 2], [1, 3], [2, 3]]
    first = _labeled_molecule(
        "mol-0", nacr=np.ones((3, 3, 3)), nac_pairs=pairs, energy=-1.0
    )
    second = _labeled_molecule(
        "mol-1", nacr=np.full((3, 3, 3), 2.0), nac_pairs=pairs, energy=-2.0
    )
    h5path = str(tmp_path / "pair.h5")

    store_current_data(h5path, [first, second], _nacr_properties())

    with h5py.File(h5path, "r") as handle:
        group = handle[list(handle)[0]]
        # Two frames of [pairs, atoms, 3].
        assert group["nacr"].shape == (2, 3, 3, 3)
        np.testing.assert_allclose(group["nacr"][0], np.ones((3, 3, 3)))
        np.testing.assert_allclose(group["nacr"][1], np.full((3, 3, 3), 2.0))
        # The pair index is a per-group constant, not stacked per frame.
        assert group["nac_pairs"].shape == (3, 2)
        np.testing.assert_array_equal(group["nac_pairs"][...], pairs)
        assert group["sE0"].shape == (2,)


def test_store_current_data_rejects_mixed_pair_orderings(tmp_path):
    first = _labeled_molecule(
        "mol-0", nacr=np.ones((1, 3, 3)), nac_pairs=[[1, 2]]
    )
    second = _labeled_molecule(
        "mol-1", nacr=np.ones((1, 3, 3)), nac_pairs=[[2, 3]]
    )

    with pytest.raises(RuntimeError, match="different .*nac_pairs.* orderings"):
        store_current_data(
            str(tmp_path / "mixed.h5"), [first, second], _nacr_properties()
        )


def test_store_current_data_requires_pair_index_metadata(tmp_path):
    molecule = _labeled_molecule(
        "mol-0", nacr=np.ones((1, 3, 3)), nac_pairs=[[1, 2]]
    )
    molecule.get_metadata().pop("nac_pairs")

    with pytest.raises(RuntimeError, match="requires the QM interface to record"):
        store_current_data(
            str(tmp_path / "nometa.h5"), [molecule], _nacr_properties()
        )


def test_store_current_data_still_rejects_unknown_scopes(tmp_path):
    molecule = _labeled_molecule(
        "mol-0", nacr=np.ones((1, 3, 3)), nac_pairs=[[1, 2]]
    )
    properties = _nacr_properties()
    properties["nacr"][1] = "per_pair"

    with pytest.raises(RuntimeError, match="Unknown property format"):
        store_current_data(
            str(tmp_path / "bad.h5"), [molecule], properties
        )
