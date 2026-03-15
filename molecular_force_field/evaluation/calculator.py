"""ASE Calculator wrapper for E3NN models.

Supported tensor product modes (all modes that can be loaded via
``build_e3trans_from_checkpoint``):
  spherical, spherical-save, spherical-save-cue,
  partial-cartesian, partial-cartesian-loose,
  pure-cartesian, pure-cartesian-sparse, pure-cartesian-sparse-save,
  pure-cartesian-ictd, pure-cartesian-ictd-save

The ``MyE3NNCalculator`` also supports ICTD models with:
  * **External field** — pass ``external_tensor`` (shape ``(rank_dim,)`` or
    ``(1, rank_dim)`` on the device) to ``__init__``.  This tensor is passed
    through to ``model.forward(external_tensor=...)`` every ``calculate()``
    call.  Set ``external_tensor=None`` (default) for models that were trained
    without external fields.
  * **Physical tensor outputs** — pass ``return_physical_tensors=True`` to
    ``__init__`` for ICTD models trained with ``--physical-tensors``.  The
    results are stored in ``calculator.results["physical_tensors"]`` as a
    ``dict[str, dict[int, np.ndarray]]`` (name → l → array).
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from molecular_force_field.utils.graph_utils import radius_graph_pbc_gpu
from molecular_force_field.utils.tensor_utils import map_tensor_values

logger = logging.getLogger(__name__)


class MyE3NNCalculator(Calculator):
    """ASE Calculator wrapper for all FSCETP tensor product modes.

    Args:
        model: Loaded e3trans model (any supported tensor product mode).
        atomic_energies_dict: Mapping from atomic number (int) to E0 (float).
        device: Torch device.
        max_radius: Neighbour-search cutoff (Å).
        external_tensor: Optional external field tensor for ICTD models trained
            with ``--external-tensor-rank``.  Shape ``(rank_dim,)`` or
            ``(1, rank_dim)`` on *device*.  ``None`` for non-field models.
        return_physical_tensors: If ``True``, call
            ``model.forward(return_physical_tensors=True)`` and store outputs
            in ``results["physical_tensors"]``.  Requires an ICTD model with
            physical tensor heads.
        **kwargs: Forwarded to the ASE ``Calculator`` base class.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model: torch.nn.Module,
        atomic_energies_dict: Dict[int, float],
        device: torch.device,
        max_radius: float,
        external_tensor: Optional[torch.Tensor] = None,
        return_physical_tensors: bool = False,
        **kwargs: Any,
    ):
        Calculator.__init__(self, **kwargs)
        self.model = model
        self.device = device
        self.max_radius = max_radius
        self.external_tensor = external_tensor
        self.return_physical_tensors = return_physical_tensors

        self.keys = torch.tensor(
            list(atomic_energies_dict.keys()), device=device
        )
        self.values = torch.tensor(
            list(atomic_energies_dict.values()), device=device
        )

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        if return_physical_tensors:
            self.implemented_properties = ["energy", "forces"]  # extended below

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        pos = torch.tensor(
            self.atoms.get_positions(), dtype=torch.float64, device=self.device
        )
        A = torch.tensor(
            self.atoms.get_atomic_numbers(),
            dtype=torch.float64,
            device=self.device,
        )

        if any(self.atoms.pbc):
            cell = torch.tensor(
                self.atoms.get_cell().array,
                dtype=torch.float64,
                device=self.device,
            ).unsqueeze(0)
            pbc = tuple(bool(x) for x in self.atoms.pbc)
        else:
            cell = (
                torch.eye(3, dtype=torch.float64, device=self.device).unsqueeze(0)
                * 100.0
            )
            pbc = (False, False, False)

        edge_src, edge_dst, edge_shifts = radius_graph_pbc_gpu(
            pos, self.max_radius, cell, pbc=pbc
        )

        pos.requires_grad_(True)
        batch_idx = torch.zeros(len(pos), dtype=torch.long, device=self.device)

        mapped_A = map_tensor_values(A, self.keys, self.values)
        E_offset = mapped_A.sum()

        # Build forward kwargs — only pass what the model accepts
        fwd_kwargs: Dict[str, Any] = {}
        if self.external_tensor is not None:
            fwd_kwargs["external_tensor"] = self.external_tensor
        if self.return_physical_tensors:
            fwd_kwargs["return_physical_tensors"] = True

        out = self.model(pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell, **fwd_kwargs)

        if self.return_physical_tensors and isinstance(out, tuple):
            atom_energies, physical_out = out
        else:
            atom_energies = out
            physical_out = None

        E_total = atom_energies.sum() + E_offset
        grads = torch.autograd.grad(E_total, pos)[0]

        self.results["energy"] = E_total.item()
        self.results["forces"] = -grads.detach().cpu().numpy()

        if physical_out is not None:
            self.results["physical_tensors"] = {
                name: {
                    l: blk.detach().cpu().numpy()
                    for l, blk in l_dict.items()
                }
                for name, l_dict in physical_out.items()
            }


class DDPCalculator(Calculator):
    """
    ASE Calculator 的 DDP 版本：仅在 rank 0 与 ASE 交互，每次 calculate() 时通过
    run_one_ddp_inference_from_ase_atoms 与其它 rank 协同完成推理（多卡分摊大结构）。
    需用 torchrun 启动，且非 rank 0 进程需在别处运行 worker 循环。
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, model, atomic_energies_dict, device, max_radius, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.model = model
        self.device = device
        self.max_radius = max_radius
        self.atomic_energies_dict = atomic_energies_dict or {}
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        from molecular_force_field.cli.inference_ddp import run_one_ddp_inference_from_ase_atoms
        dtype = next(self.model.parameters()).dtype
        energy, forces = run_one_ddp_inference_from_ase_atoms(
            self.atoms,
            self.model,
            self.max_radius,
            self.device,
            dtype,
            return_forces=True,
            atomic_energies_dict=self.atomic_energies_dict,
        )
        if energy is not None and forces is not None:
            self.results["energy"] = energy
            self.results["forces"] = forces.cpu().numpy()
