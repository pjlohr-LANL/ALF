"""TorchSim's low-level BAOAB integrator behind ALF's shared runner interface.

TorchSim is imported only when a runner is constructed.  Models operate on
ALF's packed representation and provide their own neighbor implementation;
the adapter never dispatches to TorchSim's NVIDIA-specific neighbor helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import units


KB_EV_PER_K = 8.617333262145e-5


def ensure_torchsim_available():
    """Load the optional engine with an actionable environment error."""
    try:
        import torch
        from torch_sim.integrators.md import MDState
        from torch_sim.integrators.nvt import nvt_langevin_init, nvt_langevin_step
        from torch_sim.models.interface import ModelInterface
    except Exception as exc:
        raise ImportError(
            "TorchSim sampling requires Python 3.12+ and ALF's "
            "torchsim_dynamics extra (torch-sim-atomistic==0.6.2). "
            "Install it in a separate worker environment using the appropriate "
            "CPU, CUDA, or ROCm PyTorch build."
        ) from exc
    return torch, MDState, nvt_langevin_init, nvt_langevin_step, ModelInterface


class TorchSimDynamicsRunner:
    """Fixed-membership, ragged batches with independent ALF replica controls.

    Public time/friction arguments use fs and inverse fs.  TorchSim's low-level
    functions use the natural eV/angstrom/amu units also used by ASE velocities;
    its high-level runner's picosecond conversion must not be applied here.
    """

    def __init__(
        self,
        *,
        model: Any,
        atoms_list: list[Any],
        dt_fs: float,
        temperature_K: Any,
        friction_per_fs: float,
        random_seed: int,
        device: Any,
    ) -> None:
        torch, MDState, initialize, self._step, ModelInterface = ensure_torchsim_available()
        from alframework.ml_interfaces.batched_inference import PackedBatch

        if not atoms_list:
            raise ValueError("TorchSim requires at least one system.")
        if any(atoms.constraints for atoms in atoms_list):
            raise NotImplementedError("The ALF TorchSim runner does not yet support ASE constraints.")
        self._torch = torch
        self.model = model
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.model.set_device(self.device)
        self.batch = PackedBatch.from_atoms(atoms_list, device=self.device, dtype=self.dtype)
        self._batch_ptr = self.batch.batch_ptr
        self._num_graphs = len(atoms_list)
        counts = [len(atoms) for atoms in atoms_list]
        offsets = np.cumsum([0, *counts]).tolist()
        self._slices = [slice(start, end) for start, end in zip(offsets[:-1], offsets[1:])]
        self._frozen = torch.zeros(self._num_graphs, dtype=torch.bool, device=self.device)
        self._has_frozen = False
        self._frozen_positions = self.batch.positions.clone()
        if not np.isfinite(dt_fs) or float(dt_fs) <= 0:
            raise ValueError("dt_fs must be finite and positive.")
        if not np.isfinite(friction_per_fs) or float(friction_per_fs) < 0:
            raise ValueError("friction_per_fs must be finite and nonnegative.")
        self._dt = float(dt_fs) * units.fs
        self._gamma = float(friction_per_fs) / units.fs
        self._temperature = self._temperatures(temperature_K)
        runner = self

        class ModelAdapter(ModelInterface):
            def __init__(self):
                super().__init__()
                self._device = runner.device
                self._dtype = runner.dtype
                self._compute_stress = False
                self._compute_forces = True

            def forward(self, state, **kwargs):
                del kwargs
                if runner._has_frozen:
                    # Restore before inference, not just after the engine's
                    # final half-kick. A stopped system must never expose a
                    # thermostat-displaced geometry to physical validation.
                    frozen_atoms = runner._frozen[state.system_idx]
                    state.positions[frozen_atoms] = runner._frozen_positions[frozen_atoms]
                    state.momenta[frozen_atoms] = 0.0
                runner.batch = runner._packed_view(state)
                result = runner.model(runner.batch)
                return {
                    "energy": result["energy"].reshape(runner._num_graphs),
                    "forces": result["forces"].reshape(-1, 3),
                }

        self._adapter = ModelAdapter()
        # Explicit momenta, including all-zero momenta, prevent TorchSim from
        # substituting a Maxwell-Boltzmann initialization for the builder input.
        initial_state = MDState(
            positions=self.batch.positions.clone(),
            masses=self.batch.atomic_masses.clone(),
            atomic_numbers=self.batch.atomic_numbers.clone(),
            cell=self.batch.cell.transpose(-1, -2).contiguous(),
            pbc=self.batch.pbc.clone(),
            system_idx=self.batch.system_idx,
            momenta=self.batch.velocities * self.batch.atomic_masses.unsqueeze(-1),
            energy=torch.zeros(self._num_graphs, dtype=self.dtype, device=self.device),
            forces=torch.zeros_like(self.batch.positions),
            _rng=int(random_seed),
        )
        self.state = initialize(initial_state, self._adapter, kT=self._temperature * KB_EV_PER_K)
        self.batch = self._packed_view(self.state)
        self.nsteps = 0

    def _packed_view(self, state):
        # Membership, species, masses, and offsets were validated once on
        # construction. Refresh tensor references without revalidating static
        # GPU metadata or synchronizing it to the host on every force call.
        self.batch.positions = state.positions
        self.batch.atomic_numbers = state.atomic_numbers
        self.batch.atomic_masses = state.masses
        self.batch.velocities = state.momenta / state.masses.unsqueeze(-1)
        self.batch.cell = state.cell.transpose(-1, -2)
        self.batch.pbc = self._torch.as_tensor(state.pbc, device=self.device, dtype=self._torch.bool)
        return self.batch

    def _temperatures(self, values):
        torch = self._torch
        temperatures = torch.as_tensor(values, dtype=self.dtype, device=self.device).reshape(-1)
        if temperatures.numel() == 1:
            temperatures = temperatures.expand(self._num_graphs).clone()
        if temperatures.numel() != self._num_graphs:
            raise ValueError("temperature_K must contain one value per system or a scalar.")
        if not bool(torch.isfinite(temperatures).all()) or bool((temperatures < 0).any()):
            raise ValueError("temperature_K must contain finite nonnegative temperatures.")
        return temperatures

    def _slice(self, graph_index: int):
        index = int(graph_index)
        if index < 0 or index >= self._num_graphs:
            raise IndexError("TorchSim graph index is outside the batch.")
        return index, self._slices[index]

    def run(self, steps: int) -> None:
        count = int(steps)
        if count < 0:
            raise ValueError("steps must be nonnegative.")
        frozen_atoms = self._frozen[self.state.system_idx]
        has_frozen = bool(self._frozen.any())
        if has_frozen:
            fixed_positions = self.state.positions[frozen_atoms].clone()
            fixed_momenta = self.state.momenta[frozen_atoms].clone()
        for _ in range(count):
            self.state = self._step(
                self.state,
                self._adapter,
                dt=self._dt,
                kT=self._temperature * KB_EV_PER_K,
                gamma=self._gamma,
            )
            if has_frozen:
                self.state.positions[frozen_atoms] = fixed_positions
                self.state.momenta[frozen_atoms] = fixed_momenta
            self.nsteps += 1
        if has_frozen and count:
            # Reconcile diagnostics and force caches with restored frozen
            # coordinates; each unfrozen system evolves independently.
            self.evaluate()
        else:
            self.batch = self._packed_view(self.state)

    def evaluate(self) -> None:
        result = self._adapter(self.state)
        self.state.energy = result["energy"]
        self.state.forces = result["forces"]
        self.batch = self._packed_view(self.state)

    def diagnostics_for_graph(self, graph_index: int) -> dict[str, Any]:
        self._slice(graph_index)
        return self.model.diagnostics_for_graph(graph_index)

    def set_lcm_mode(self, graph_index: int, *, pair, sigma: float, alpha_eV: float) -> None:
        self._slice(graph_index)
        self.model.set_lcm_mode(graph_index, pair=pair, sigma=sigma, alpha_eV=alpha_eV)

    def freeze_graph(self, graph_index: int) -> None:
        index, atoms = self._slice(graph_index)
        self._frozen[index] = True
        self._has_frozen = True
        self._frozen_positions[atoms] = self.state.positions[atoms]
        self.state.momenta[atoms] = 0.0
        self.batch = self._packed_view(self.state)

    def set_temperature(self, temperature_K: Any) -> None:
        self._temperature = self._temperatures(temperature_K)

    def set_cell(self, graph_index: int, cell, *, scale_atoms: bool = True, evaluate: bool = True) -> None:
        index, atoms = self._slice(graph_index)
        if bool(self._frozen[index]):
            return
        torch = self._torch
        new_cell = torch.as_tensor(cell, dtype=self.dtype, device=self.device)
        if new_cell.shape != (3, 3) or not bool(torch.isfinite(new_cell).all()):
            raise ValueError("Cell must be a finite 3 by 3 matrix.")
        if float(torch.linalg.det(new_cell)) <= 0:
            raise ValueError("Cell must have positive volume.")
        if scale_atoms:
            old_cell = self.state.cell[index].T
            if float(torch.linalg.det(old_cell)) <= 0:
                raise ValueError("Cannot scale positions from a cell without positive volume.")
            fractional = torch.linalg.solve(old_cell.T, self.state.positions[atoms].T).T
            self.state.positions[atoms] = fractional @ new_cell
        self.state.cell[index] = new_cell.T
        if evaluate:
            self.evaluate()

    def sync_graph_to_atoms(self, graph_index: int, atoms) -> None:
        index, selection = self._slice(graph_index)
        atoms.set_positions(self.state.positions[selection].detach().cpu().numpy())
        atoms.set_velocities(
            (self.state.momenta[selection] / self.state.masses[selection, None]).detach().cpu().numpy()
        )
        atoms.set_cell(self.state.cell[index].T.detach().cpu().numpy(), scale_atoms=False)
        pbc = self._torch.as_tensor(self.state.pbc)
        atoms.set_pbc(pbc[index].detach().cpu().numpy() if pbc.ndim == 2 else pbc.cpu().numpy())
