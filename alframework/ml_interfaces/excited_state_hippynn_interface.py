from __future__ import annotations

import csv
import glob
import json
import math
import multiprocessing
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from parsl import python_app

from alframework.tools.excited_state_tools import derive_gap_property_table, derive_state_property_table


def _flatten_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value).reshape(-1)


def _write_predicted_vs_target_csv(path: str | Path, predicted: Any, target: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = _flatten_to_numpy(predicted)
    true = _flatten_to_numpy(target)
    if pred.shape != true.shape:
        raise ValueError(f"Predicted and target shapes do not match for CSV export: {pred.shape} vs {true.shape}")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Predicted", "Actual"])
        writer.writerows(zip(pred.tolist(), true.tolist()))


def _configure_cuda_visible_devices(device_string: str, from_multiprocessing_nGPU: int | None) -> tuple[str, str | None]:
    requested = str(device_string).strip().lower()
    if requested == "cpu":
        return "<cpu>", None

    if requested == "from_multiprocessing":
        process = multiprocessing.current_process()
        gpu_index = 0
        if from_multiprocessing_nGPU is not None and int(from_multiprocessing_nGPU) > 0:
            gpu_index = (process._identity[-1] - 1) % int(from_multiprocessing_nGPU)
        cuda_visible = str(gpu_index)
    else:
        cuda_visible = str(device_string)

    # This must happen before torch is imported in the child process.
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
    return cuda_visible, cuda_visible


def _resolve_device(device_string: str, cuda_visible: str | None):
    import torch

    requested = str(device_string).strip().lower()
    if requested == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu"), "<cpu>"

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return device, str(cuda_visible)


def _build_network(species_node, positions_node, cell_node, network_choice: int, network_params: dict[str, Any]):
    from hippynn.graphs import networks

    params = dict(network_params or {})
    if cell_node is None:
        parents = (species_node, positions_node)
        periodic = False
    else:
        parents = (species_node, positions_node, cell_node)
        periodic = True

    if int(network_choice) == 0:
        return networks.Hipnn("hipnn_model", parents, module_kwargs=params, periodic=periodic)
    return networks.HipHopnn("hiphopnn_model", parents, module_kwargs=params, periodic=periodic)


def _infer_species_and_n_atoms(h5_train_dir: str, species_key: str, coordinates_key: str) -> tuple[list[int], int]:
    from ase.data import atomic_numbers

    from alframework.tools.pyanitools import anidataloader

    all_species: set[int] = set()
    n_atoms = None
    for shard_path in sorted(glob.glob(os.path.join(str(h5_train_dir), "*.h5"))):
        loader = anidataloader(shard_path)
        try:
            for group in loader:
                species = group.get(species_key)
                coords = group.get(coordinates_key)
                if species is None or coords is None:
                    continue
                if n_atoms is None:
                    n_atoms = int(np.asarray(coords).shape[1])
                for symbol in np.asarray(species).reshape(-1).tolist():
                    text = symbol.decode("utf-8") if isinstance(symbol, bytes) else str(symbol)
                    all_species.add(int(atomic_numbers[text]))
                if n_atoms is not None and all_species:
                    break
        finally:
            loader.cleanup()
        if n_atoms is not None and all_species:
            break
    if n_atoms is None or not all_species:
        raise RuntimeError("Could not infer species or atom count from ALF HDF5 training shards.")
    return [0] + sorted(all_species), int(n_atoms)


class DataDumper:
    def __init__(self, node, saved: str) -> None:
        from hippynn import plotting
        from hippynn.graphs.indextypes.reduce_funcs import elementwise_compare_reduce

        if hasattr(node, "main_output"):
            node = node.main_output
        reduced_true, reduced_pred = elementwise_compare_reduce(node.true, node.pred)
        self._plotter = plotting.plotters.Plotter((reduced_pred, reduced_true), plt_fn=None, saved=saved, shown=False)

    def make_plot(self, data_args):
        pred, true = data_args
        fig = plt.figure()

        def custom_saver(fname: str, **kwargs: Any) -> None:
            del kwargs
            csv_path = Path(fname)
            if csv_path.suffix.lower() != ".csv":
                csv_path = csv_path.with_suffix(".csv")
            _write_predicted_vs_target_csv(csv_path, pred, true)

        fig.savefig = custom_saver
        return fig

    def __getattr__(self, item):
        return getattr(self._plotter, item)


def train_single_excited_state_model(
    *,
    model_id: int,
    model_dir: str,
    h5_train_dir: str,
    properties_list: dict[str, list[Any]],
    ML_config: dict[str, Any],
    from_multiprocessing_nGPU: int | None = None,
) -> dict[str, Any]:
    selected_cuda_visible, _ = _configure_cuda_visible_devices(
        device_string=str(ML_config.get("device_string", "from_multiprocessing")),
        from_multiprocessing_nGPU=from_multiprocessing_nGPU,
    )

    import torch
    import hippynn
    from hippynn import plotting
    from hippynn.databases.h5_pyanitools import PyAniDirectoryDB
    from hippynn.experiment import setup_training, train_model
    from hippynn.experiment.controllers import PatienceController, RaiseBatchSizeOnPlateau
    from hippynn.graphs import inputs, loss, physics, targets

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_default_dtype(torch.float32)
    hippynn.settings.WARN_LOW_DISTANCES = False
    hippynn.settings.TRANSPARENT_PLOT = True

    device, cuda_visible = _resolve_device(
        device_string=str(ML_config.get("device_string", "from_multiprocessing")),
        cuda_visible=selected_cuda_visible,
    )

    seed = int(ML_config.get("random_seed", 114514)) + int(model_id)
    torch.manual_seed(seed)
    np.random.seed(seed)

    coordinates_key = str(ML_config.get("coordinates_key", "coordinates"))
    species_key = str(ML_config.get("species_key", "species"))
    cell_key = ML_config.get("cell_key")
    cell_key = None if cell_key in {None, "None"} else str(cell_key)

    train_forces = bool(ML_config.get("train_forces", True))
    export_force_gradients = bool(ML_config.get("export_force_gradients", train_forces))
    state_table = derive_state_property_table(
        properties_list,
        require_forces=train_forces or export_force_gradients,
    )
    gap_config = dict(ML_config.get("gap_targets") or {})
    gap_targets_enabled = bool(gap_config.get("enabled", False))
    gap_target_mode = str(gap_config.get("mode", "derived")).strip().lower()
    if gap_targets_enabled and gap_target_mode not in {"derived", "head"}:
        raise ValueError(f"Unsupported gap_targets.mode: {gap_target_mode!r}. Use 'derived' or 'head'.")
    gap_table = derive_gap_property_table(
        properties_list,
        gap_config=gap_config if gap_config else None,
        require_properties=gap_targets_enabled,
    )
    network_params = dict(ML_config.get("network_params") or {})
    if "possible_species" not in network_params:
        network_params["possible_species"], inferred_n_atoms = _infer_species_and_n_atoms(
            h5_train_dir,
            species_key,
            coordinates_key,
        )
    else:
        inferred_n_atoms = None

    n_atoms = int(ML_config.get("n_atoms") or inferred_n_atoms or 0)
    if n_atoms <= 0:
        raise ValueError("n_atoms must be provided or inferable for excited-state training.")

    model_root = Path(model_dir).expanduser().resolve()
    model_root.mkdir(parents=True, exist_ok=True)

    with hippynn.tools.active_directory(str(model_root)):
        with hippynn.tools.log_terminal("training_log.txt", "wt"):
            print(f"Model ID: {int(model_id)}")
            print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
            print(f"Training device: {device}")
            species_node = inputs.SpeciesNode(db_name=species_key)
            positions_node = inputs.PositionsNode(db_name=coordinates_key)
            positions_node.requires_grad = True
            cell_node = inputs.CellNode(db_name=cell_key) if cell_key is not None else None

            network = _build_network(
                species_node=species_node,
                positions_node=positions_node,
                cell_node=cell_node,
                network_choice=int(ML_config.get("network_choice", 1)),
                network_params=network_params,
            )

            energy_outputs: list[tuple[dict[str, Any], Any]] = []
            energy_by_state: dict[int, Any] = {}
            gap_outputs: list[tuple[dict[str, Any], Any]] = []
            force_outputs: list[tuple[dict[str, Any], Any]] = []
            validation_losses: dict[str, Any] = {}
            plotters: list[Any] = []
            total_loss = None

            exports = dict(ML_config.get("exports") or {})
            export_pdf = bool(exports.get("pdf", True))
            export_png = bool(exports.get("png", False))
            export_csv = bool(exports.get("csv", False))
            export_subdir = Path(str(exports.get("output_subdir", "plots")))
            export_subdir_abs = model_root / export_subdir
            export_subdir_abs.mkdir(parents=True, exist_ok=True)

            for row in state_table:
                energy_node = targets.HEnergyNode(f"dE{int(row['state'])}", network, module_kwargs=None)
                mol_energy = energy_node.mol_energy
                mol_energy.db_name = str(row["energy_db_name"])
                energy_outputs.append((row, mol_energy))
                energy_by_state[int(row["state"])] = mol_energy

                if export_force_gradients and row["force_key"] is not None:
                    force_node = physics.GradientNode(
                        str(row["force_key"]),
                        (mol_energy, positions_node),
                        sign=int(ML_config.get("sign", -1)),
                        db_name=str(row["force_db_name"]),
                    )
                    force_outputs.append((row, force_node))

            if gap_targets_enabled:
                for row in gap_table:
                    lower_state = int(row["lower_state"])
                    upper_state = int(row["upper_state"])
                    if gap_target_mode == "head":
                        gap_node = targets.HEnergyNode(f"gap_{row['gap_key']}", network, module_kwargs=None)
                        gap_output = gap_node.mol_energy
                    else:
                        if lower_state not in energy_by_state or upper_state not in energy_by_state:
                            raise ValueError(
                                f"Gap target {row['gap_key']} requires states {lower_state} and {upper_state}, "
                                "but one of those state energy outputs is missing."
                            )
                        gap_output = energy_by_state[upper_state] - energy_by_state[lower_state]
                    gap_output.name = str(row["gap_key"])
                    gap_output.db_name = str(row["gap_db_name"])
                    gap_outputs.append((row, gap_output))

            for row, energy_output in energy_outputs:
                rmse = loss.MSELoss.of_node(energy_output) ** 0.5
                mae = loss.MAELoss.of_node(energy_output)
                combined = rmse + mae
                validation_losses[f"{row['energy_key']}_RMSE"] = rmse
                validation_losses[f"{row['energy_key']}_MAE"] = mae
                validation_losses[f"{row['energy_key']}_Loss"] = combined
                weighted = float(ML_config.get("energy_weight", 1.0)) * combined
                total_loss = weighted if total_loss is None else total_loss + weighted
                if export_pdf:
                    plotters.append(
                        plotting.Hist2D.compare(
                            energy_output,
                            saved=str(export_subdir_abs / f"{row['energy_key']}.pdf"),
                            shown=False,
                        )
                    )
                if export_png:
                    plotters.append(
                        plotting.Hist2D.compare(
                            energy_output,
                            saved=str(export_subdir_abs / f"{row['energy_key']}.png"),
                            shown=False,
                        )
                    )
                if export_csv:
                    plotters.append(
                        DataDumper(energy_output, saved=str(export_subdir_abs / f"{row['energy_key']}.csv"))
                    )

            for row, gap_output in gap_outputs:
                gap_rmse = loss.MSELoss.of_node(gap_output) ** 0.5
                gap_mae = loss.MAELoss.of_node(gap_output)
                combined = gap_rmse + gap_mae
                validation_losses[f"{row['gap_key']}_RMSE"] = gap_rmse
                validation_losses[f"{row['gap_key']}_MAE"] = gap_mae
                validation_losses[f"{row['gap_key']}_Loss"] = combined
                total_loss = total_loss + float(gap_config.get("weight", 1.0)) * combined
                if export_pdf:
                    plotters.append(
                        plotting.Hist2D.compare(
                            gap_output,
                            saved=str(export_subdir_abs / f"{row['gap_key']}.pdf"),
                            shown=False,
                        )
                    )
                if export_png:
                    plotters.append(
                        plotting.Hist2D.compare(
                            gap_output,
                            saved=str(export_subdir_abs / f"{row['gap_key']}.png"),
                            shown=False,
                        )
                    )
                if export_csv:
                    plotters.append(DataDumper(gap_output, saved=str(export_subdir_abs / f"{row['gap_key']}.csv")))

            if force_outputs:
                force_norm = math.sqrt(3.0 * float(n_atoms))
                for row, force_output in force_outputs:
                    if train_forces:
                        force_rmse = loss.MSELoss.of_node(force_output) ** 0.5
                        force_mae = loss.MAELoss.of_node(force_output)
                        combined = (force_rmse + force_mae) / force_norm
                        validation_losses[f"{row['force_key']}_RMSE"] = force_rmse
                        validation_losses[f"{row['force_key']}_MAE"] = force_mae
                        validation_losses[f"{row['force_key']}_Loss"] = combined
                        total_loss = total_loss + float(ML_config.get("force_weight", 1.0)) * combined
                        if export_pdf:
                            plotters.append(
                                plotting.Hist2D.compare(
                                    force_output,
                                    saved=str(export_subdir_abs / f"{row['force_key']}.pdf"),
                                    shown=False,
                                )
                            )
                        if export_png:
                            plotters.append(
                                plotting.Hist2D.compare(
                                    force_output,
                                    saved=str(export_subdir_abs / f"{row['force_key']}.png"),
                                    shown=False,
                                )
                            )
                        if export_csv:
                            plotters.append(
                                DataDumper(force_output, saved=str(export_subdir_abs / f"{row['force_key']}.csv"))
                            )
                    else:
                        validation_losses[f"{row['force_key']}_GradientRMS"] = (
                            loss.MeanSq.of_node(force_output) ** 0.5
                        )

            l2_reg = loss.l2reg(network)
            validation_losses["L2"] = l2_reg
            validation_losses["Loss_wo_L2"] = total_loss
            validation_losses["Loss"] = total_loss + float(ML_config.get("l2_weight", 2e-5)) * l2_reg

            plot_maker = plotting.PlotMaker(*plotters, plot_every=int(ML_config.get("plot_frequency", 50))) if plotters else None
            training_modules, db_info = hippynn.experiment.assemble_for_training(
                validation_losses["Loss"],
                validation_losses,
                plot_maker=plot_maker,
            )

            database = PyAniDirectoryDB(
                directory=h5_train_dir,
                allow_unfound=True,
                quiet=False,
                seed=np.random.randint(1_000_000_000),
                inputs=None,
                targets=None,
            )
            arrays = database.arr_dict
            arrays[species_key] = arrays[species_key].to(torch.int64)
            for key, value in list(arrays.items()):
                if hasattr(value, "dtype") and value.dtype == torch.float64:
                    arrays[key] = value.to(torch.float32)

            database.inputs = db_info["inputs"]
            database.targets = db_info["targets"]
            for key in list(arrays):
                if key not in db_info["inputs"] and key not in db_info["targets"] and key != "indices":
                    del arrays[key]

            database.make_random_split("valid", float(ML_config.get("valid_size", 0.1)))
            database.make_random_split("test", float(ML_config.get("test_size", 0.1)))
            database.split_the_rest("train")

            optimizer = torch.optim.AdamW(
                training_modules.model.parameters(),
                lr=float(ML_config.get("learning_rate", network_params.get("learning_rate", 5e-4))),
            )
            scheduler = RaiseBatchSizeOnPlateau(
                optimizer=optimizer,
                **dict(ML_config.get("scheduler_options") or {"max_batch_size": 128, "patience": 15, "factor": 0.5}),
            )
            controller = PatienceController(
                optimizer=optimizer,
                scheduler=scheduler,
                stopping_key="Loss",
                batch_size=int(ML_config.get("batch_size", 64)),
                eval_batch_size=int(ML_config.get("eval_batch_size", 128)),
                max_epochs=int(ML_config.get("max_epochs", 1000)),
                fraction_train_eval=0.1,
                termination_patience=int(ML_config.get("termination_patience", 55)),
            )
            setup_params = hippynn.experiment.SetupParams(controller=controller)

            training_modules, controller, metric_tracker = setup_training(
                training_modules=training_modules,
                setup_params=setup_params,
            )
            metric_tracker = train_model(
                training_modules,
                database,
                controller,
                metric_tracker,
                callbacks=None,
                batch_callbacks=None,
            )

            summary = {
                "model_id": int(model_id),
                "device": str(device),
                "cuda_visible_devices": str(cuda_visible),
                "state_table": state_table,
                "gap_table": gap_table,
                "train_forces": train_forces,
                "export_force_gradients": export_force_gradients,
                "gap_target_mode": gap_target_mode if gap_targets_enabled else "disabled",
                "metric": metric_tracker.best_metric_values,
                "avg_epoch_time": float(np.average(metric_tracker.epoch_times)),
                "loss": float(metric_tracker.best_metric_values["valid"]["Loss"]),
            }
            with open("training_summary.json", "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)

    return {"model_id": int(model_id), "model_dir": str(model_root)}


def _excited_state_training_worker(arg_dict: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = train_single_excited_state_model(**arg_dict)
        return {"ok": True, "payload": payload}
    except Exception as exc:
        return {
            "ok": False,
            "payload": {
                "model_id": int(arg_dict["model_id"]),
                "model_dir": str(arg_dict["model_dir"]),
                "error": repr(exc),
            },
        }


def _training_complete(model_dir: str) -> bool:
    log_path = Path(model_dir) / "training_log.txt"
    if not log_path.exists():
        return False
    return "Training complete" in log_path.read_text(encoding="utf-8")


def train_excited_state_ensemble(
    *,
    ML_config: dict[str, Any],
    h5_dir: str,
    model_path: str,
    current_training_id: int,
    gpus_per_node: int,
    properties_list: dict[str, list[Any]],
) -> tuple[list[bool], int]:
    n_models = int(ML_config["n_models"])
    workers = max(1, min(int(gpus_per_node) if int(gpus_per_node) > 0 else 1, n_models))
    general_configuration = dict(ML_config)
    params_list: list[dict[str, Any]] = []
    for model_id in range(n_models):
        params_list.append(
            {
                "model_id": int(model_id),
                "model_dir": model_path.format(current_training_id) + f"/model-{int(model_id):02d}",
                "h5_train_dir": str(h5_dir),
                "properties_list": dict(properties_list),
                "ML_config": general_configuration,
                "from_multiprocessing_nGPU": int(gpus_per_node) if int(gpus_per_node) > 0 else None,
            }
        )

    pool = multiprocessing.Pool(processes=workers)
    try:
        raw_results = pool.map(_excited_state_training_worker, params_list)
    finally:
        pool.close()
        pool.join()

    completed: list[bool] = []
    for result in raw_results:
        if result["ok"]:
            completed.append(_training_complete(result["payload"]["model_dir"]))
        else:
            completed.append(False)
            error_path = Path(result["payload"]["model_dir"]).expanduser().resolve() / "training_error.json"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            with open(error_path, "w", encoding="utf-8") as handle:
                json.dump(result["payload"], handle, indent=2)

    return completed, int(current_training_id)


def load_excited_state_ensemble(
    ensemble_directory: str,
    properties_list: dict[str, list[Any]],
    device: str = "cuda:0",
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    import hippynn

    state_table = derive_state_property_table(properties_list, require_forces=False)
    gap_table = derive_gap_property_table(properties_list)
    ensemble_root = Path(ensemble_directory).expanduser().resolve()
    ensemble_graph, _ = hippynn.graphs.make_ensemble(str(ensemble_root / "model-*"))
    if hasattr(ensemble_graph, "to"):
        ensemble_graph.to(device)
    resolved_rows: list[dict[str, Any]] = []
    for row in state_table:
        resolved = dict(row)
        resolved["energy_node_base"] = f"ensemble_{row['energy_db_name']}"
        if row["force_db_name"] is not None:
            resolved["force_node_base"] = f"ensemble_{row['force_db_name']}"
        else:
            resolved["force_node_base"] = None
        ensemble_graph.node_from_name(resolved["energy_node_base"])
        if resolved["force_node_base"] is not None:
            ensemble_graph.node_from_name(resolved["force_node_base"])
        resolved_rows.append(resolved)

    resolved_gap_rows: list[dict[str, Any]] = []
    for row in gap_table:
        resolved = dict(row)
        resolved["gap_node_base"] = f"ensemble_{row['gap_db_name']}"
        try:
            ensemble_graph.node_from_name(resolved["gap_node_base"])
        except Exception:
            # Older excited-state models do not expose derived gap nodes. The
            # sampler can still fall back to state-energy differences.
            continue
        resolved_gap_rows.append(resolved)
    return ensemble_graph, resolved_rows, resolved_gap_rows


@python_app(executors=["alf_ML_executor"])
def train_excited_state_HIPPYNN_ensemble_task(
    ML_config,
    h5_dir,
    model_path,
    current_training_id,
    gpus_per_node,
    properties_list,
    remove_existing=False,
    h5_test_dir=None,
):
    del remove_existing, h5_test_dir
    return train_excited_state_ensemble(
        ML_config=dict(ML_config or {}),
        h5_dir=str(h5_dir),
        model_path=str(model_path),
        current_training_id=int(current_training_id),
        gpus_per_node=int(gpus_per_node),
        properties_list=dict(properties_list or {}),
    )
