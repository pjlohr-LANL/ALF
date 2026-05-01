from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from parsl import python_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor

from alframework.tools.molecules_class import MoleculesObject


local_test_config = Config(
    executors=[
        ThreadPoolExecutor(label="alf_sampler_executor", max_threads=4),
        ThreadPoolExecutor(label="alf_QM_executor", max_threads=2),
        ThreadPoolExecutor(label="alf_ML_executor", max_threads=2),
    ],
    strategy=None,
)


def _build_water(moleculeid: str) -> MoleculesObject:
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=np.array(
            [
                [0.0000, 0.0000, 0.0000],
                [0.9572, 0.0000, 0.0000],
                [-0.2390, 0.9266, 0.0000],
            ],
            dtype=float,
        ),
    )
    return MoleculesObject(atoms, str(moleculeid))


@python_app(executors=["alf_sampler_executor"])
def smoke_builder_task(moleculeid=None, moleculeids=None, builder_config=None):
    del builder_config
    if moleculeids is None:
        return _build_water(str(moleculeid))
    return [_build_water(str(mid)) for mid in moleculeids]


@python_app(executors=["alf_sampler_executor"])
def smoke_sampler_task(
    molecule_object,
    sampler_config,
    model_path=None,
    current_model_id=None,
    gpus_per_node=None,
    properties_list=None,
):
    del sampler_config, model_path, current_model_id, gpus_per_node, properties_list
    molecule_object.update_metadata({"smoke_sampler": True})
    return molecule_object


@python_app(executors=["alf_QM_executor"])
def smoke_qm_task(molecule_object, QM_config, properties_list):
    del QM_config
    natoms = len(molecule_object.get_atoms())
    results = {}
    for prop_key, schema in dict(properties_list or {}).items():
        storage_mode = str(schema[1]).lower()
        if storage_mode == "atomic":
            results[prop_key] = np.zeros((natoms, 3), dtype=float)
        else:
            results[prop_key] = float(len(prop_key))
    molecule_object.store_results(results)
    molecule_object.set_converged_flag(True)
    return molecule_object


@python_app(executors=["alf_ML_executor"])
def smoke_ml_task(
    ML_config,
    h5_dir,
    model_path,
    current_training_id,
    gpus_per_node=None,
    properties_list=None,
    remove_existing=False,
    h5_test_dir=None,
):
    del ML_config, h5_dir, gpus_per_node, properties_list, remove_existing, h5_test_dir
    model_root = Path(model_path.format(int(current_training_id)))
    model_dir = model_root / "model-00"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "training_log.txt").write_text("Training complete\n", encoding="utf-8")
    return [True], int(current_training_id)
