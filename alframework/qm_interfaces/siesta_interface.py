import os
import re

from parsl import python_app
from ase.calculators.siesta import Siesta

from alframework.tools.molecules_class import MoleculesObject


def _siesta_converged(calc, directory, calc_input, properties):
    """Infer SIESTA convergence from calculator flags and output markers."""
    if hasattr(calc, "converged"):
        try:
            return bool(calc.converged)
        except Exception:
            pass

    label = calc_input.get("label", "siesta")
    output_file = os.path.join(directory, f"{label}.out")
    if os.path.isfile(output_file):
        with open(output_file, "r", errors="ignore") as out_fh:
            output_text = out_fh.read()
        if re.search(r"SCF.*NOT.*CONV", output_text, re.IGNORECASE):
            return False
        if re.search(r"Job\s+completed", output_text, re.IGNORECASE):
            return True

    # Fallback: if all requested properties exist, treat as converged.
    return all(prop in calc.results for prop in properties)


@python_app(executors=["alf_QM_executor"])
def siesta_calculator_task(molecule_object, QM_config, QM_scratch_dir, properties_list):
    """ASE SIESTA calculator task based on parameters from qm_config.json.

    Args:
        molecule_object (MoleculesObject): Molecule and metadata container.
        QM_config (dict): QM configuration dictionary.
        QM_scratch_dir (str): Scratch directory for QM calculations.
        properties_list (dict): Property schema from master config.

    Returns:
        MoleculesObject: Updated molecule object with calculation results.
    """
    assert isinstance(
        molecule_object, MoleculesObject
    ), "molecule_object must be an instance of MoleculesObject"

    directory = QM_scratch_dir + "/" + molecule_object.get_moleculeid()
    if os.path.isdir(directory):
        raise RuntimeError("Scratch directory exists: " + directory)
    os.makedirs(directory)

    command = QM_config["QM_run_command"]
    properties = list(properties_list.keys())
    atoms = molecule_object.get_atoms()

    calc_input = QM_config.get("input", {}).copy()
    if not isinstance(calc_input, dict):
        raise TypeError("QM_config['input'] must be a dictionary for SIESTA.")

    # Allow per-structure k-point overrides from builder metadata.
    if (
        hasattr(molecule_object, "metadata")
        and "kpoints" in molecule_object.metadata
        and "kpts" not in calc_input
    ):
        calc_input["kpts"] = molecule_object.metadata["kpoints"]

    calc = Siesta(directory=directory, command=command, **calc_input)

    calc.calculate(atoms=atoms, properties=properties)

    molecule_object.store_results(calc.results)
    molecule_object.set_converged_flag(
        _siesta_converged(calc, directory, calc_input, properties)
    )

    return molecule_object
