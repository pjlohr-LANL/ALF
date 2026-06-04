from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from alframework.tools.molecule_payloads import (
    flatten_molecule_output,
    is_sampler_result_ref,
    load_sampler_result_ref,
    molecule_output_to_payload,
    molecule_to_payload,
    payload_to_molecule,
    write_sampler_error_ref,
    write_sampler_result_ref,
)
from alframework.tools.molecules_class import MoleculesObject


def _molecule(molecule_id: str) -> MoleculesObject:
    atoms = Atoms(numbers=[1, 8], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]])
    atoms.set_momenta([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])
    molecule = MoleculesObject(atoms, molecule_id)
    molecule.update_metadata(
        {
            "excited_state": np.int64(1),
            "nearest_neighbor_distances": np.asarray([0.9, 0.9]),
            "score_components": {"energy_uncertainty": np.float64(0.2)},
        }
    )
    molecule.store_results({"sE0": np.float64(-1.0), "F0": np.zeros((2, 3))})
    molecule.set_converged_flag(True)
    return molecule


def test_molecule_payload_roundtrip_reconstructs_clean_molecule():
    payload = molecule_to_payload(_molecule("mol-1"))
    restored = payload_to_molecule(payload)

    assert restored.get_moleculeid() == "mol-1"
    assert restored.check_convergence() is True
    assert restored.get_atoms().calc is None
    assert restored.get_atoms().get_atomic_numbers().tolist() == [1, 8]
    assert restored.get_atoms().get_momenta().shape == (2, 3)
    assert restored.get_metadata()["excited_state"] == 1
    assert restored.get_metadata()["nearest_neighbor_distances"] == [0.9, 0.9]
    assert restored.get_results()["F0"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_flatten_molecule_output_accepts_payloads_and_nested_lists():
    molecule_a = _molecule("a")
    molecule_b = _molecule("b")
    nested = [molecule_to_payload(molecule_a), [molecule_b]]

    flattened = flatten_molecule_output(nested)

    assert [molecule.get_moleculeid() for molecule in flattened] == ["a", "b"]


def test_sampler_payload_list_serializes_with_parsl():
    facade = pytest.importorskip("parsl.serialize.facade")
    payloads = molecule_output_to_payload([_molecule(f"cand-{idx:03d}") for idx in range(100)])

    serialized = facade.serialize(payloads, buffer_threshold=1000000)

    assert isinstance(serialized, bytes)


def test_sampler_result_reference_roundtrip_and_parsl_serialization(tmp_path):
    facade = pytest.importorskip("parsl.serialize.facade")
    result_ref = write_sampler_result_ref(
        [_molecule(f"cand-{idx:03d}") for idx in range(100)],
        {"sampler_result_payload_dir": str(tmp_path)},
    )

    serialized = facade.serialize(result_ref, buffer_threshold=1000000)
    restored = flatten_molecule_output(result_ref)

    assert isinstance(serialized, bytes)
    assert is_sampler_result_ref(result_ref)
    assert len(restored) == 100
    assert restored[0].get_moleculeid() == "cand-000"


def test_sampler_error_reference_is_small_and_loads_as_error(tmp_path):
    facade = pytest.importorskip("parsl.serialize.facade")
    try:
        raise ValueError("example failure")
    except ValueError as exc:
        error_ref = write_sampler_error_ref(exc, {"sampler_result_payload_dir": str(tmp_path)})

    serialized = facade.serialize(error_ref, buffer_threshold=1000000)

    assert isinstance(serialized, bytes)
    assert is_sampler_result_ref(error_ref)
    with pytest.raises(RuntimeError, match="example failure"):
        load_sampler_result_ref(error_ref)
