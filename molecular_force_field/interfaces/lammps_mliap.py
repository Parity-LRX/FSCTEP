"""LAMMPS ML-IAP unified / ML-IAP-Kokkos interface for the molecular force field.

Provides LAMMPS_MLIAP_MFF, a subclass of MLIAPUnified that computes per-atom forces
via autograd on a dummy position tensor (dE/d(pos)), avoiding the O(N*M) edge-force
gradient storage of the traditional per-pair approach.

Per-atom forces are written directly into the LAMMPS force buffer (data.f), and
global virial is handled automatically by LAMMPS's virial_fdotr_compute().

支持两种运行模式：
- 标准 ML-IAP unified（CPU）：activate_mliappy + 直接写入 data.f
- ML-IAP-Kokkos（GPU）：activate_mliappy_kokkos + GPU tensor 直接写入

仅以下五种模型支持：e3nn_layers、e3nn_layers_channelwise、cue_layers_channelwise、
pure_cartesian_ictd_layers、pure_cartesian_ictd_layers_full（因其支持 precomputed_edge_vec）。

Usage:
    # Export:
    python -m molecular_force_field.cli.export_mliap checkpoint.pth --elements H O

    # LAMMPS input:
    pair_style mliap unified model-mliap.pt 0
    pair_coeff * * H O
"""

from __future__ import annotations

import io
import inspect
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.dlpack as torch_dlpack

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except (ImportError, OSError):
    class MLIAPUnified:
        """Stub when lammps is not installed or shared lib is missing."""
        def __init__(self, interface=None, element_types=None,
                     ndescriptors=None, nparams=None, rcutfac=None):
            self.interface = interface
            self.element_types = element_types
            self.ndescriptors = ndescriptors
            self.nparams = nparams
            self.rcutfac = rcutfac

        def pickle(self, fname):
            import pickle
            with open(fname, "wb") as fp:
                pickle.dump(self, fp)

from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.models.e3nn_layers_channelwise import (
    E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise,
)
from molecular_force_field.models.pure_cartesian_sparse_layers import (
    PureCartesianSparseTransformerLayer,
)
from molecular_force_field.models.pure_cartesian_sparse_layers_save import (
    PureCartesianSparseTransformerLayerSave,
)
from molecular_force_field.models.pure_cartesian_ictd_layers import (
    PhysicalTensorICTDRecovery,
    PureCartesianICTDTransformerLayer as PureCartesianICTDTransformerLayerSave,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_full import PureCartesianICTDTransformerLayer
from molecular_force_field.models.pure_cartesian_ictd_layers_o3 import PureCartesianICTDO3TransformerLayer
from molecular_force_field.utils.config import ModelConfig
from molecular_force_field.utils.checkpoint_metadata import (
    derive_long_range_far_max_radius_multiplier,
    get_checkpoint_e3_state_dict,
    infer_external_tensor_rank_from_state_dict,
    infer_physical_tensor_outputs_from_state_dict,
)
from molecular_force_field.utils.external_tensor_specs import normalize_external_tensor_specs
from molecular_force_field.models.zbl import maybe_wrap_model_with_zbl
from molecular_force_field.utils.tensor_utils import map_tensor_values


_GLOBAL_PHYS_LAYOUT = (
    ("charge", 1),
    ("dipole", 3),
    ("polarizability", 9),
    ("quadrupole", 9),
)
_ATOM_PHYS_LAYOUT = (
    ("charge_per_atom", 1),
    ("dipole_per_atom", 3),
    ("polarizability_per_atom", 9),
    ("quadrupole_per_atom", 9),
    ("born_effective_charge_per_atom", 9),
)


def _layout_total_dim(layout: tuple[tuple[str, int], ...]) -> int:
    return sum(dim for _, dim in layout)


def _layout_offsets(layout: tuple[tuple[str, int], ...]) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    start = 0
    for name, dim in layout:
        out[name] = (start, start + dim)
        start += dim
    return out


_GLOBAL_PHYS_TOTAL_DIM = _layout_total_dim(_GLOBAL_PHYS_LAYOUT)
_ATOM_PHYS_TOTAL_DIM = _layout_total_dim(_ATOM_PHYS_LAYOUT)
_GLOBAL_PHYS_OFFSETS = _layout_offsets(_GLOBAL_PHYS_LAYOUT)
_ATOM_PHYS_OFFSETS = _layout_offsets(_ATOM_PHYS_LAYOUT)
_PHYS_MASK_WIDTH = max(len(_GLOBAL_PHYS_LAYOUT), len(_ATOM_PHYS_LAYOUT))


def _resolve_model_external_tensor_rank(model: nn.Module) -> int | None:
    ext_rank = getattr(model, "external_tensor_rank", None)
    if ext_rank is not None:
        return int(ext_rank)
    conv = getattr(model, "e3_conv_emb", None)
    ext_rank = getattr(conv, "external_tensor_rank", None)
    if ext_rank is not None:
        return int(ext_rank)
    nested = getattr(model, "model", None)
    if nested is not None and nested is not model:
        return _resolve_model_external_tensor_rank(nested)
    return None


def _resolve_model_external_tensor_specs(model: nn.Module) -> list[dict] | None:
    specs = getattr(model, "external_tensor_specs", None)
    if specs is not None:
        return list(specs)
    conv = getattr(model, "e3_conv_emb", None)
    specs = getattr(conv, "external_tensor_specs", None)
    if specs is not None:
        return list(specs)
    nested = getattr(model, "model", None)
    if nested is not None and nested is not model:
        return _resolve_model_external_tensor_specs(nested)
    return None


def _resolve_model_external_tensor_total_numel(model: nn.Module) -> int:
    total = getattr(model, "external_tensor_total_numel", None)
    if total is not None:
        return int(total)
    conv = getattr(model, "e3_conv_emb", None)
    total = getattr(conv, "external_tensor_total_numel", None)
    if total is not None:
        return int(total)
    specs = _resolve_model_external_tensor_specs(model)
    if specs:
        return int(sum(int(spec.get("numel", 0)) for spec in specs))
    ext_rank = _resolve_model_external_tensor_rank(model)
    if ext_rank is None:
        return 0
    return 3 ** int(ext_rank) if int(ext_rank) > 0 else 1


def _canonicalize_cartesian_rank2(x: torch.Tensor) -> torch.Tensor:
    # Keep the user-facing LAMMPS output simple: always expose a full 3x3 Cartesian tensor.
    return x.reshape(*x.shape[:-2], 9)


def _physical_block_is_cartesian(block: torch.Tensor, rank: int) -> bool:
    if block is None:
        return False
    if rank == 0:
        return block.dim() >= 2 and block.shape[-1] == 1
    if rank == 1:
        return block.dim() >= 3 and block.shape[-1] == 3
    if rank == 2:
        return block.dim() >= 4 and block.shape[-2:] == (3, 3)
    return False


def _recover_rank1_cartesian(blocks: dict[int, torch.Tensor]) -> torch.Tensor:
    blk = blocks.get(1)
    if blk is None:
        raise ValueError("Missing l=1 block for rank-1 Cartesian physical tensor")
    if not _physical_block_is_cartesian(blk, 1):
        raise ValueError(f"rank-1 Cartesian block must have shape (..., C, 3), got {tuple(blk.shape)}")
    if blk.shape[-2] != 1:
        raise ValueError("rank-1 Cartesian physical tensor export currently requires channels_out=1")
    return blk[..., 0, :]


def _recover_rank2_cartesian(blocks: dict[int, torch.Tensor], *, include_trace: bool) -> torch.Tensor:
    mat = None
    if 2 in blocks:
        blk = blocks[2]
        if not _physical_block_is_cartesian(blk, 2):
            raise ValueError(f"rank-2 Cartesian block must have shape (..., C, 3, 3), got {tuple(blk.shape)}")
        if blk.shape[-3] != 1:
            raise ValueError("rank-2 Cartesian physical tensor export currently requires channels_out=1")
        mat = blk[..., 0, :, :]
    if mat is None:
        if include_trace:
            raise ValueError("Missing l=2 block for rank-2 Cartesian physical tensor")
        return torch.zeros(0)
    mat = 0.5 * (mat + mat.transpose(-1, -2))
    if include_trace and 0 in blocks:
        trace_blk = blocks[0]
        if not _physical_block_is_cartesian(trace_blk, 0):
            raise ValueError(f"rank-0 Cartesian block must have shape (..., C, 1), got {tuple(trace_blk.shape)}")
        if trace_blk.shape[-2] != 1:
            raise ValueError("rank-0 Cartesian physical tensor export currently requires channels_out=1")
        trace = trace_blk[..., 0, 0]
        eye = torch.eye(3, device=mat.device, dtype=mat.dtype)
        mat = mat + trace.unsqueeze(-1).unsqueeze(-1) * eye
    return mat


def _recover_cartesian_physical_tensors(
    physical_out: dict[str, dict[int, torch.Tensor]] | None,
    *,
    num_graphs: int,
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    global_phys = torch.zeros(num_graphs, _GLOBAL_PHYS_TOTAL_DIM, device=device, dtype=dtype)
    atom_phys = torch.zeros(num_nodes, _ATOM_PHYS_TOTAL_DIM, device=device, dtype=dtype)
    global_mask = torch.zeros(_PHYS_MASK_WIDTH, device=device, dtype=dtype)
    atom_mask = torch.zeros(_PHYS_MASK_WIDTH, device=device, dtype=dtype)
    if not physical_out:
        return global_phys, atom_phys, global_mask, atom_mask

    recover_rank1 = PhysicalTensorICTDRecovery(rank=1, channels_in=1, lmax_in=1, include_trace_chain=False).to(device)
    recover_rank2_full = PhysicalTensorICTDRecovery(
        rank=2,
        channels_in=1,
        lmax_in=2,
        include_trace_chain=True,
        rank2_mode="symmetric",
    ).to(device)
    recover_rank2_l2 = PhysicalTensorICTDRecovery(rank=2, channels_in=1, lmax_in=2, include_trace_chain=False).to(device)
    recover_rank2_bec = PhysicalTensorICTDRecovery(
        rank=2,
        channels_in=1,
        lmax_in=2,
        include_trace_chain=True,
        rank2_mode="full",
    ).to(device)

    global_mask_index = {name: idx for idx, (name, _) in enumerate(_GLOBAL_PHYS_LAYOUT)}
    atom_mask_index = {name: idx for idx, (name, _) in enumerate(_ATOM_PHYS_LAYOUT)}

    for name, blocks in physical_out.items():
        if not isinstance(blocks, dict):
            continue
        if name in _GLOBAL_PHYS_OFFSETS:
            start, end = _GLOBAL_PHYS_OFFSETS[name]
            if name == "charge":
                blk = blocks.get(0)
                if blk is None:
                    continue
                if _physical_block_is_cartesian(blk, 0):
                    if blk.shape[-2] != 1:
                        raise ValueError(f"{name} channels_out>1 is not supported for LibTorch export")
                    global_phys[:, start:end] = blk[..., 0, 0].reshape(blk.shape[0], 1)
                else:
                    if blk.shape[-2] != 1:
                        raise ValueError(f"{name} channels_out>1 is not supported for LibTorch export")
                    global_phys[:, start:end] = blk[..., 0, :].reshape(blk.shape[0], 1)
            elif name == "dipole":
                rec = _recover_rank1_cartesian(blocks) if _physical_block_is_cartesian(blocks.get(1), 1) else recover_rank1(blocks)
                global_phys[:, start:end] = rec.reshape(rec.shape[0], 3)
            elif name == "polarizability":
                rec = (
                    _recover_rank2_cartesian(blocks, include_trace=True)
                    if _physical_block_is_cartesian(blocks.get(2), 2)
                    else recover_rank2_full(blocks)
                )
                global_phys[:, start:end] = _canonicalize_cartesian_rank2(rec)
            elif name == "quadrupole":
                l2_only = {2: blocks[2]} if 2 in blocks else {}
                if not l2_only:
                    continue
                rec = (
                    _recover_rank2_cartesian(l2_only, include_trace=False)
                    if _physical_block_is_cartesian(l2_only.get(2), 2)
                    else recover_rank2_l2(l2_only)
                )
                global_phys[:, start:end] = _canonicalize_cartesian_rank2(rec)
            global_mask[global_mask_index[name]] = 1.0
        elif name in _ATOM_PHYS_OFFSETS:
            start, end = _ATOM_PHYS_OFFSETS[name]
            if name == "charge_per_atom":
                blk = blocks.get(0)
                if blk is None:
                    continue
                if _physical_block_is_cartesian(blk, 0):
                    if blk.shape[-2] != 1:
                        raise ValueError(f"{name} channels_out>1 is not supported for LibTorch export")
                    atom_phys[:, start:end] = blk[..., 0, 0].reshape(blk.shape[0], 1)
                else:
                    if blk.shape[-2] != 1:
                        raise ValueError(f"{name} channels_out>1 is not supported for LibTorch export")
                    atom_phys[:, start:end] = blk[..., 0, :].reshape(blk.shape[0], 1)
            elif name == "dipole_per_atom":
                rec = _recover_rank1_cartesian(blocks) if _physical_block_is_cartesian(blocks.get(1), 1) else recover_rank1(blocks)
                atom_phys[:, start:end] = rec.reshape(rec.shape[0], 3)
            elif name == "polarizability_per_atom":
                rec = (
                    _recover_rank2_cartesian(blocks, include_trace=True)
                    if _physical_block_is_cartesian(blocks.get(2), 2)
                    else recover_rank2_full(blocks)
                )
                atom_phys[:, start:end] = _canonicalize_cartesian_rank2(rec)
            elif name == "quadrupole_per_atom":
                l2_only = {2: blocks[2]} if 2 in blocks else {}
                if not l2_only:
                    continue
                rec = (
                    _recover_rank2_cartesian(l2_only, include_trace=False)
                    if _physical_block_is_cartesian(l2_only.get(2), 2)
                    else recover_rank2_l2(l2_only)
                )
                atom_phys[:, start:end] = _canonicalize_cartesian_rank2(rec)
            elif name == "born_effective_charge_per_atom":
                if _physical_block_is_cartesian(blocks.get(2), 2):
                    rec = _recover_rank2_cartesian(blocks, include_trace=True)
                    if 1 in blocks and _physical_block_is_cartesian(blocks.get(1), 1):
                        anti = blocks[1][..., 0, :]
                        rec = rec.clone()
                        rec[:, 1, 2] += anti[:, 0]
                        rec[:, 2, 1] -= anti[:, 0]
                        rec[:, 2, 0] += anti[:, 1]
                        rec[:, 0, 2] -= anti[:, 1]
                        rec[:, 0, 1] += anti[:, 2]
                        rec[:, 1, 0] -= anti[:, 2]
                else:
                    rec = recover_rank2_bec(blocks)
                atom_phys[:, start:end] = _canonicalize_cartesian_rank2(rec)
            atom_mask[atom_mask_index[name]] = 1.0

    return global_phys, atom_phys, global_mask, atom_mask


# ---------------------------------------------------------------------------
# Multi-GPU message passing via LAMMPS Kokkos forward/reverse exchange
# ---------------------------------------------------------------------------

class LAMMPS_MP(torch.autograd.Function):
    """Autograd-compatible wrapper for LAMMPS Kokkos ghost communication.

    forward_exchange: copies local features → ghost atoms (across GPUs / PBC).
    reverse_exchange: accumulates ghost gradients back to local atoms.
    """

    @staticmethod
    def forward(ctx, feats: torch.Tensor, data) -> torch.Tensor:
        ctx.vec_len = feats.shape[-1]
        ctx.data = data
        out = torch.empty_like(feats)
        data.forward_exchange(feats, out, ctx.vec_len)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs
        gout = torch.empty_like(grad)
        ctx.data.reverse_exchange(grad, gout, ctx.vec_len)
        return gout, None


class _TorchScriptEdgeVecCore(nn.Module):
    """Core wrapper to make precomputed_edge_vec traceable (positional arg).

    LAMMPS LibTorch 接口始终返回 per-atom energy；若模型配置了 physical tensor
    heads，则额外导出固定 schema 的体系级/逐原子笛卡尔物理张量，供 C++ 侧缓存。
    """

    def __init__(self, model: nn.Module, *, export_reciprocal_source: bool = False):
        super().__init__()
        self.model = model
        self.external_tensor_rank = _resolve_model_external_tensor_rank(model)
        self.external_tensor_specs = _resolve_model_external_tensor_specs(model)
        self.external_tensor_total_numel = _resolve_model_external_tensor_total_numel(model)
        self.has_physical_tensor_heads = (
            hasattr(model, "physical_tensor_heads") and getattr(model, "physical_tensor_heads", None) is not None
        )
        self.export_reciprocal_source = bool(export_reciprocal_source)

    def forward(
        self,
        pos: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        edge_vec: torch.Tensor,
        external_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            kwargs = {
                "precomputed_edge_vec": edge_vec,
                "return_physical_tensors": self.has_physical_tensor_heads,
                "return_reciprocal_source": self.export_reciprocal_source,
            }
            if self.external_tensor_total_numel > 0:
                kwargs["external_tensor"] = external_tensor
            out = self.model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, **kwargs)
        except TypeError:
            kwargs = {
                "precomputed_edge_vec": edge_vec,
                "return_physical_tensors": self.has_physical_tensor_heads,
                "return_reciprocal_source": self.export_reciprocal_source,
            }
            if self.external_tensor_total_numel > 0:
                kwargs["external_tensor"] = external_tensor
            out = self.model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, **kwargs)

        reciprocal_source = torch.empty((pos.shape[0], 0), device=pos.device, dtype=pos.dtype)
        if isinstance(out, tuple):
            atom_energy = out[0]
            if self.has_physical_tensor_heads:
                physical_out = out[1] if len(out) > 1 else None
                if self.export_reciprocal_source and len(out) > 2:
                    reciprocal_source = out[2]
            else:
                physical_out = None
                if self.export_reciprocal_source and len(out) > 1:
                    reciprocal_source = out[1]
        else:
            atom_energy = out
            physical_out = None
        global_phys, atom_phys, global_mask, atom_mask = _recover_cartesian_physical_tensors(
            physical_out,
            num_graphs=cell.size(0),
            num_nodes=pos.shape[0],
            device=pos.device,
            dtype=atom_energy.dtype,
        )
        return atom_energy, global_phys, atom_phys, global_mask, atom_mask, reciprocal_source


class _TorchScriptEdgeVecCoreWithFidelity(nn.Module):
    """Core wrapper with a runtime graph-level fidelity id positional input."""

    def __init__(self, model: nn.Module, *, export_reciprocal_source: bool = False):
        super().__init__()
        self.model = model
        self.external_tensor_rank = _resolve_model_external_tensor_rank(model)
        self.external_tensor_specs = _resolve_model_external_tensor_specs(model)
        self.external_tensor_total_numel = _resolve_model_external_tensor_total_numel(model)
        self.has_physical_tensor_heads = (
            hasattr(model, "physical_tensor_heads") and getattr(model, "physical_tensor_heads", None) is not None
        )
        self.export_reciprocal_source = bool(export_reciprocal_source)
        self.num_fidelity_levels = int(getattr(model, "num_fidelity_levels", 0) or 0)

    def forward(
        self,
        pos: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        edge_vec: torch.Tensor,
        external_tensor: torch.Tensor,
        fidelity_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        kwargs = {
            "precomputed_edge_vec": edge_vec,
            "return_physical_tensors": self.has_physical_tensor_heads,
            "return_reciprocal_source": self.export_reciprocal_source,
            "fidelity_ids": fidelity_ids.to(device=pos.device, dtype=torch.long).view(-1),
        }
        if self.external_tensor_total_numel > 0:
            kwargs["external_tensor"] = external_tensor
        out = self.model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, **kwargs)

        reciprocal_source = torch.empty((pos.shape[0], 0), device=pos.device, dtype=pos.dtype)
        if isinstance(out, tuple):
            atom_energy = out[0]
            if self.has_physical_tensor_heads:
                physical_out = out[1] if len(out) > 1 else None
                if self.export_reciprocal_source and len(out) > 2:
                    reciprocal_source = out[2]
            else:
                physical_out = None
                if self.export_reciprocal_source and len(out) > 1:
                    reciprocal_source = out[1]
        else:
            atom_energy = out
            physical_out = None
        global_phys, atom_phys, global_mask, atom_mask = _recover_cartesian_physical_tensors(
            physical_out,
            num_graphs=cell.size(0),
            num_nodes=pos.shape[0],
            device=pos.device,
            dtype=atom_energy.dtype,
        )
        return atom_energy, global_phys, atom_phys, global_mask, atom_mask, reciprocal_source


class _TorchScriptEdgeVecAdapter(nn.Module):
    """Adapter that preserves the original forward signature used by AtomForcesWrapper."""

    def __init__(self, core: torch.jit.ScriptModule):
        super().__init__()
        self.core = core

    def __getstate__(self):
        # Make this module picklable via torch.save by serializing the ScriptModule to bytes.
        buf = io.BytesIO()
        torch.jit.save(self.core, buf)
        return {"core_bytes": buf.getvalue()}

    def __setstate__(self, state):
        # Restore ScriptModule from bytes.
        # IMPORTANT: TorchScript graphs may contain constant tensors that do NOT move with `.to()`.
        # So we should load constants onto the right device up-front.
        nn.Module.__init__(self)
        pref = os.environ.get("MLIAP_TORCHSCRIPT_MAP_LOCATION", "").strip().lower()
        if pref in ("cpu", "cuda"):
            map_loc = pref
        else:
            map_loc = "cuda" if torch.cuda.is_available() else "cpu"
        core = torch.jit.load(io.BytesIO(state["core_bytes"]), map_location=map_loc)
        self.core = core

    def forward(
        self,
        pos: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        *,
        precomputed_edge_vec: Optional[torch.Tensor] = None,
        external_tensor: Optional[torch.Tensor] = None,
        sync_after_scatter=None,
    ) -> torch.Tensor:
        if precomputed_edge_vec is None:
            raise ValueError("TorchScript model requires precomputed_edge_vec")
        # sync_after_scatter is ignored in TorchScript mode.
        if external_tensor is None:
            external_tensor = torch.empty(0, dtype=pos.dtype, device=pos.device)
        out = self.core(
            pos, A, batch, edge_src, edge_dst, edge_shifts, cell, precomputed_edge_vec, external_tensor
        )
        return out[0] if isinstance(out, tuple) else out


class _TorchScriptEdgeVecAdapterWithFidelity(nn.Module):
    """Adapter for TorchScript cores that require runtime fidelity_ids."""

    def __init__(self, core: torch.jit.ScriptModule):
        super().__init__()
        self.core = core

    def __getstate__(self):
        buf = io.BytesIO()
        torch.jit.save(self.core, buf)
        return {"core_bytes": buf.getvalue()}

    def __setstate__(self, state):
        nn.Module.__init__(self)
        pref = os.environ.get("MLIAP_TORCHSCRIPT_MAP_LOCATION", "").strip().lower()
        if pref in ("cpu", "cuda"):
            map_loc = pref
        else:
            map_loc = "cuda" if torch.cuda.is_available() else "cpu"
        self.core = torch.jit.load(io.BytesIO(state["core_bytes"]), map_location=map_loc)

    def forward(
        self,
        pos: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        *,
        precomputed_edge_vec: Optional[torch.Tensor] = None,
        external_tensor: Optional[torch.Tensor] = None,
        fidelity_ids: Optional[torch.Tensor] = None,
        sync_after_scatter=None,
    ) -> torch.Tensor:
        if precomputed_edge_vec is None:
            raise ValueError("TorchScript model requires precomputed_edge_vec")
        if fidelity_ids is None:
            raise ValueError("TorchScript model requires fidelity_ids")
        if external_tensor is None:
            external_tensor = torch.empty(0, dtype=pos.dtype, device=pos.device)
        out = self.core(
            pos,
            A,
            batch,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
            precomputed_edge_vec,
            external_tensor,
            fidelity_ids.to(device=pos.device, dtype=torch.long).view(-1),
        )
        return out[0] if isinstance(out, tuple) else out


def _maybe_torchscript_trace_model(
    model: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
    enable: bool,
    export_reciprocal_source: bool = False,
) -> nn.Module:
    """Optionally trace a model to TorchScript for faster Python dispatch."""
    if not enable:
        return model

    # Only trace in eval mode; gradients w.r.t. inputs are still supported.
    model.eval()

    # Trace a core wrapper that takes edge_vec as positional arg.
    num_fidelity_levels = int(getattr(model, "num_fidelity_levels", 0) or 0)
    fixed_fidelity_id = getattr(model, "fixed_fidelity_id", None)
    runtime_fidelity = num_fidelity_levels > 0 and fixed_fidelity_id is None
    core_cls = _TorchScriptEdgeVecCoreWithFidelity if runtime_fidelity else _TorchScriptEdgeVecCore
    core = core_cls(model, export_reciprocal_source=export_reciprocal_source).to(device=device)

    ext_total_numel = _resolve_model_external_tensor_total_numel(model)

    # Example inputs (dynamic shapes should still work for most ops).
    N = 32
    E = 256
    pos = torch.zeros(N, 3, device=device, dtype=dtype)
    A = torch.ones(N, device=device, dtype=torch.long)
    batch = torch.zeros(N, device=device, dtype=torch.long)
    edge_src = torch.randint(0, N, (E,), device=device, dtype=torch.long)
    edge_dst = torch.randint(0, N, (E,), device=device, dtype=torch.long)
    edge_shifts = torch.zeros(E, 3, device=device, dtype=dtype)
    cell = (torch.eye(3, device=device, dtype=dtype).unsqueeze(0) * 100.0)
    edge_vec = torch.randn(E, 3, device=device, dtype=dtype)
    if ext_total_numel <= 0:
        external_tensor = torch.empty(0, device=device, dtype=dtype)
    else:
        external_tensor = torch.zeros(ext_total_numel, device=device, dtype=dtype)
    fidelity_ids = torch.zeros(1, device=device, dtype=torch.long)

    try:
        # Prewarm one-time caches before tracing to keep Python-side setup out of the trace.
        try:
            with torch.no_grad():
                for m in core.modules():
                    prewarm = getattr(m, "prewarm_caches", None)
                    if callable(prewarm):
                        prewarm(device=device, dtype=dtype)
                # One eager run to lock in branches and fill caches.
                if runtime_fidelity:
                    _ = core(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec, external_tensor, fidelity_ids)
                else:
                    _ = core(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec, external_tensor)
        except Exception:
            pass

        trace_inputs = (
            (pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec, external_tensor, fidelity_ids)
            if runtime_fidelity
            else (pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec, external_tensor)
        )
        core_ts = torch.jit.trace(core, trace_inputs, check_trace=False, strict=False)
        try:
            core_ts = torch.jit.freeze(core_ts.eval())
        except Exception:
            core_ts = core_ts.eval()
        return _TorchScriptEdgeVecAdapterWithFidelity(core_ts) if runtime_fidelity else _TorchScriptEdgeVecAdapter(core_ts)
    except Exception as e:
        raise RuntimeError(f"TorchScript trace failed: {e}")


class AtomForcesWrapper(nn.Module):
    """Wrapper that computes per-atom energies and per-atom forces via autograd on pos.

    Instead of differentiating through edge_vec (O(npairs) leaf gradient), this
    wrapper uses a dummy ``pos`` tensor as the autograd leaf and constructs
    ``edge_vec = pos[dst] - pos[src] + rij`` so that the gradient accumulates
    into per-atom forces (O(natoms) leaf gradient).
    """

    def __init__(self, model: nn.Module, atomic_energy_keys: torch.Tensor,
                 atomic_energy_values: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_energy_keys", atomic_energy_keys)
        self.register_buffer("atomic_energy_values", atomic_energy_values)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _build_forward_kwargs(
        self,
        *,
        edge_vec: torch.Tensor,
        external_tensor: Optional[torch.Tensor],
        fidelity_ids: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        params = inspect.signature(self.model.forward).parameters
        kwargs: dict[str, torch.Tensor] = {}
        if "precomputed_edge_vec" in params:
            kwargs["precomputed_edge_vec"] = edge_vec
        if external_tensor is not None and "external_tensor" in params:
            kwargs["external_tensor"] = external_tensor
        if fidelity_ids is not None and "fidelity_ids" in params:
            kwargs["fidelity_ids"] = fidelity_ids
        return kwargs

    def forward(
        self,
        rij: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        nlocal: int,
        external_tensor: Optional[torch.Tensor] = None,
        fidelity_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total energy, per-atom energies and per-atom forces.

        Args:
            rij: (E, 3) pair distance vectors from LAMMPS (detached)
            A: (N,) atomic numbers
            batch: (N,) batch indices (all zeros for single structure)
            edge_src/edge_dst: (E,) edge indices
            edge_shifts: (E, 3) PBC shift integers
            cell: (1, 3, 3) cell matrix
            nlocal: number of local (owned) atoms

        Returns:
            (total_energy, atom_energies[:nlocal], atom_forces) where
            atom_forces = -dE/d(pos), shape (N, 3)
        """
        ntotal = A.size(0)
        pos = torch.zeros(ntotal, 3, dtype=rij.dtype, device=rij.device,
                          requires_grad=True)

        edge_vec = pos[edge_dst] - pos[edge_src] + rij.detach()

        kwargs = self._build_forward_kwargs(
            edge_vec=edge_vec,
            external_tensor=external_tensor,
            fidelity_ids=fidelity_ids,
        )
        atom_energies = self.model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, **kwargs)

        mapped_A = map_tensor_values(
            A.to(dtype=self.atomic_energy_values.dtype),
            self.atomic_energy_keys,
            self.atomic_energy_values,
        )
        E_offset = mapped_A[:nlocal].sum()
        E_total = atom_energies[:nlocal].sum() + E_offset

        neg_forces = torch.autograd.grad(E_total, pos, create_graph=False)[0]
        atom_forces = -neg_forces

        return E_total, atom_energies[:nlocal].detach(), atom_forces.detach()


class EdgeForcesWrapper(nn.Module):
    """Legacy wrapper: per-pair forces via autograd on edge_vec (O(npairs) gradient).

    Kept for backward compatibility.  Prefer :class:`AtomForcesWrapper`.
    """

    def __init__(self, model: nn.Module, atomic_energy_keys: torch.Tensor,
                 atomic_energy_values: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_energy_keys", atomic_energy_keys)
        self.register_buffer("atomic_energy_values", atomic_energy_values)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _build_forward_kwargs(
        self,
        *,
        edge_vec: torch.Tensor,
        external_tensor: Optional[torch.Tensor],
        fidelity_ids: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        params = inspect.signature(self.model.forward).parameters
        kwargs: dict[str, torch.Tensor] = {}
        if "precomputed_edge_vec" in params:
            kwargs["precomputed_edge_vec"] = edge_vec
        if external_tensor is not None and "external_tensor" in params:
            kwargs["external_tensor"] = external_tensor
        if fidelity_ids is not None and "fidelity_ids" in params:
            kwargs["fidelity_ids"] = fidelity_ids
        return kwargs

    def forward(
        self,
        edge_vec: torch.Tensor,
        A: torch.Tensor,
        batch: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_shifts: torch.Tensor,
        cell: torch.Tensor,
        nlocal: int,
        external_tensor: Optional[torch.Tensor] = None,
        fidelity_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute total energy, per-atom energies and per-pair forces."""
        pos = torch.zeros(A.size(0), 3, dtype=edge_vec.dtype, device=edge_vec.device)

        kwargs = self._build_forward_kwargs(
            edge_vec=edge_vec,
            external_tensor=external_tensor,
            fidelity_ids=fidelity_ids,
        )
        atom_energies = self.model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, **kwargs)

        mapped_A = map_tensor_values(
            A.to(dtype=self.atomic_energy_values.dtype),
            self.atomic_energy_keys,
            self.atomic_energy_values,
        )
        E_offset = mapped_A[:nlocal].sum()
        E_total = atom_energies[:nlocal].sum() + E_offset

        pair_forces = torch.autograd.grad(E_total, edge_vec, create_graph=False)[0]

        return E_total, atom_energies[:nlocal].detach(), pair_forces.detach()


class LAMMPS_MLIAP_MFF(MLIAPUnified):
    """ML-IAP unified interface for the molecular force field.

    Computes per-atom forces via autograd on a dummy position tensor and writes
    them directly into the LAMMPS force buffer.  Global virial is handled by
    LAMMPS's ``virial_fdotr_compute()`` automatically.

    Implements the three required methods of MLIAPUnified:
    - compute_forces(data)
    - compute_descriptors(data)
    - compute_gradients(data)

    Attributes set for LAMMPS:
    - element_types: list of element symbols (e.g. ["H", "O"])
    - rcutfac: cutoff radius
    - ndescriptors / nparams: set to 1 (not used directly)
    """

    def __init__(
        self,
        model: nn.Module,
        element_types: List[str],
        max_radius: float,
        atomic_energy_keys: torch.Tensor,
        atomic_energy_values: torch.Tensor,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__(
            interface=None,
            element_types=element_types,
            ndescriptors=1,
            nparams=1,
            rcutfac=max_radius,
        )
        self.device = device
        self.dtype = dtype
        self.wrapper = AtomForcesWrapper(model, atomic_energy_keys, atomic_energy_values)
        self.wrapper = self.wrapper.to(dtype=dtype).to(device)
        self.initialized = False

        # Buffer cache for compute_forces (reuse when ntotal/npairs unchanged)
        self._cache_ntotal: Optional[int] = None
        self._cache_npairs: Optional[int] = None
        self._cache_batch: Optional[torch.Tensor] = None
        self._cache_edge_shifts: Optional[torch.Tensor] = None
        self._cache_cell: Optional[torch.Tensor] = None
        # DLPack/cupy indices often come as int32; converting to int64 is a copy.
        # Cache converted indices by underlying (cupy) pointer to avoid per-step copies.
        self._cache_elems_ptr: Optional[int] = None
        self._cache_elems_i64: Optional[torch.Tensor] = None
        self._cache_pair_i_ptr: Optional[int] = None
        self._cache_pair_j_ptr: Optional[int] = None
        self._cache_pair_i_i64: Optional[torch.Tensor] = None
        self._cache_pair_j_i64: Optional[torch.Tensor] = None

        # Build elem index → atomic number Z lookup table.
        # LAMMPS data.elems is 0-based: index 0 = element_types[0], etc.
        from ase.data import atomic_numbers as ase_Z
        self._elem_to_Z = torch.tensor(
            [ase_Z.get(s, 0) for s in element_types], dtype=torch.long,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        element_types: List[str],
        max_radius: Optional[float] = None,
        atomic_energy_keys: Optional[List[int]] = None,
        atomic_energy_values: Optional[List[float]] = None,
        device: str = "cpu",
        embed_size: Optional[List[int]] = None,
        output_size: Optional[int] = None,
        tensor_product_mode: Optional[str] = None,
        num_interaction: Optional[int] = None,
        ictd_tp_path_policy: Optional[str] = None,
        ictd_tp_max_rank_other: Optional[int] = None,
        torchscript: bool = False,
        force_naive: bool = False,
    ) -> "LAMMPS_MLIAP_MFF":
        """Create LAMMPS_MLIAP_MFF from a checkpoint file.

        Supports tensor_product_mode from checkpoint or argument:
        - "spherical": E3_TransformerLayer_multi (e3nn_layers)
        - "spherical-save": E3_TransformerLayer_multi_channelwise (e3nn_layers_channelwise)
        - "spherical-save-cue": E3_TransformerLayer_multi (cue_layers_channelwise, cuEquivariance GPU)
        - "pure-cartesian-sparse": PureCartesianSparseTransformerLayer
        - "pure-cartesian-sparse-save": PureCartesianSparseTransformerLayerSave
        - "pure-cartesian-ictd": PureCartesianICTDTransformerLayer (pure_cartesian_ictd_layers_full)
        - "pure-cartesian-ictd-save": PureCartesianICTDTransformerLayer (pure_cartesian_ictd_layers)
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        selected_state_dict, state_source = get_checkpoint_e3_state_dict(ckpt)
        if state_source == "ema":
            print("[LAMMPS_MLIAP_MFF] Using EMA weights from checkpoint")

        arch_meta = ckpt.get("model_hyperparameters", {})

        dtype_raw = ckpt.get("dtype", arch_meta.get("dtype", torch.float64))
        if isinstance(dtype_raw, str):
            dtype = torch.float64 if dtype_raw in ("float64", "double") else torch.float32
        else:
            dtype = dtype_raw

        mode = tensor_product_mode or ckpt.get("tensor_product_mode", "spherical")
        # Use max_radius from checkpoint if saved; otherwise use CLI override; finally fall back to 5.0.
        radius = float(ckpt.get("max_radius", max_radius if max_radius is not None else 5.0))
        if "max_radius" in ckpt:
            print(f"[LAMMPS_MLIAP_MFF] 使用 checkpoint 中的 max_radius: {radius:.2f} Å")
        config = ModelConfig(
            dtype=dtype,
            channel_in=int(arch_meta.get("channel_in", 64)),
            channel_in2=int(arch_meta.get("channel_in2", 32)),
            channel_in3=int(arch_meta.get("channel_in3", 32)),
            channel_in4=int(arch_meta.get("channel_in4", 32)),
            channel_in5=int(arch_meta.get("channel_in5", 32)),
            max_atomvalue=int(arch_meta.get("max_atomvalue", 10)),
            embedding_dim=int(arch_meta.get("embedding_dim", 16)),
            main_hidden_sizes3=list(arch_meta.get("main_hidden_sizes3", [64, 32])),
            embed_size=embed_size if embed_size is not None else list(arch_meta.get("embed_size", [128, 128, 128])),
            output_size=int(output_size if output_size is not None else arch_meta.get("output_size", 8)),
            irreps_output_conv_channels=arch_meta.get("irreps_output_conv_channels"),
            lmax=int(arch_meta.get("lmax", 2)),
            function_type=str(arch_meta.get("function_type", "gaussian")),
            num_layers=int(arch_meta.get("num_layers", 1)),
            number_of_basis=int(arch_meta.get("number_of_basis", 8)),
            number_of_basis_main=int(arch_meta.get("number_of_basis_main", 8)),
            emb_number_main_2=list(arch_meta.get("emb_number_main_2", [64, 64, 64])),
            max_radius=radius,
            max_radius_main=float(arch_meta.get("max_radius_main", radius)),
        )
        if num_interaction is None:
            num_interaction = int(arch_meta.get("num_interaction", 2))
        invariant_channels = int(arch_meta.get("invariant_channels", 32))

        if atomic_energy_keys is not None and atomic_energy_values is not None:
            aek = torch.tensor(atomic_energy_keys, dtype=torch.long)
            aev = torch.tensor(atomic_energy_values, dtype=dtype)
        elif ckpt.get("atomic_energy_keys") is not None and ckpt.get("atomic_energy_values") is not None:
            aek_raw = ckpt.get("atomic_energy_keys")
            aev_raw = ckpt.get("atomic_energy_values")
            aek = aek_raw.detach().cpu().to(dtype=torch.long) if isinstance(aek_raw, torch.Tensor) else torch.tensor(aek_raw, dtype=torch.long)
            aev = aev_raw.detach().cpu().to(dtype=dtype) if isinstance(aev_raw, torch.Tensor) else torch.tensor(aev_raw, dtype=dtype)
            print("[LAMMPS_MLIAP_MFF] 使用 checkpoint 中的 atomic_energy_keys/atomic_energy_values")
        else:
            config.load_atomic_energies_from_file("fitted_E0.csv")
            aek = config.atomic_energy_keys
            aev = config.atomic_energy_values

        external_tensor_rank = ckpt.get("external_tensor_rank", arch_meta.get("external_tensor_rank"))
        external_tensor_irrep = ckpt.get("external_tensor_irrep", arch_meta.get("external_tensor_irrep"))
        external_tensor_specs = ckpt.get("external_tensor_specs", arch_meta.get("external_tensor_specs"))
        num_fidelity_levels = int(arch_meta.get("num_fidelity_levels", 0) or 0)
        multi_fidelity_mode = str(arch_meta.get("multi_fidelity_mode", "conditioning") or "conditioning")
        o3_irrep_preset = str(arch_meta.get("o3_irrep_preset", "auto"))
        o3_active_irreps = arch_meta.get("o3_active_irreps")
        if external_tensor_rank is None:
            external_tensor_rank = infer_external_tensor_rank_from_state_dict(selected_state_dict)
        external_tensor_specs = normalize_external_tensor_specs(
            external_tensor_specs,
            external_tensor_rank=external_tensor_rank,
            external_tensor_irrep=external_tensor_irrep,
            external_tensor_parity=None,
        )
        physical_tensor_outputs = ckpt.get("physical_tensor_outputs", arch_meta.get("physical_tensor_outputs"))
        if physical_tensor_outputs is None:
            physical_tensor_outputs = infer_physical_tensor_outputs_from_state_dict(selected_state_dict)
        long_range_mode = str(arch_meta.get("long_range_mode", "none"))
        long_range_hidden_dim = int(arch_meta.get("long_range_hidden_dim", 64))
        long_range_boundary = str(arch_meta.get("long_range_boundary", "nonperiodic"))
        long_range_neutralize = bool(arch_meta.get("long_range_neutralize", True))
        long_range_filter_hidden_dim = int(arch_meta.get("long_range_filter_hidden_dim", 64))
        long_range_kmax = int(arch_meta.get("long_range_kmax", 2))
        long_range_mesh_size = int(arch_meta.get("long_range_mesh_size", 16))
        long_range_slab_padding_factor = int(arch_meta.get("long_range_slab_padding_factor", 2))
        long_range_include_k0 = bool(arch_meta.get("long_range_include_k0", False))
        long_range_source_channels = int(arch_meta.get("long_range_source_channels", 1))
        long_range_backend = str(arch_meta.get("long_range_backend", "dense_pairwise"))
        long_range_reciprocal_backend = str(arch_meta.get("long_range_reciprocal_backend", "direct_kspace"))
        long_range_energy_partition = str(arch_meta.get("long_range_energy_partition", "potential"))
        long_range_green_mode = str(arch_meta.get("long_range_green_mode", "poisson"))
        long_range_assignment = str(arch_meta.get("long_range_assignment", "cic"))
        long_range_theta = float(arch_meta.get("long_range_theta", 0.5))
        long_range_leaf_size = int(arch_meta.get("long_range_leaf_size", 32))
        long_range_multipole_order = int(arch_meta.get("long_range_multipole_order", 0))
        long_range_far_source_dim = int(arch_meta.get("long_range_far_source_dim", 16))
        long_range_far_num_shells = int(arch_meta.get("long_range_far_num_shells", 3))
        long_range_far_shell_growth = float(arch_meta.get("long_range_far_shell_growth", 2.0))
        long_range_far_tail = bool(arch_meta.get("long_range_far_tail", True))
        long_range_far_tail_bins = int(arch_meta.get("long_range_far_tail_bins", 2))
        long_range_far_stats = str(arch_meta.get("long_range_far_stats", "mean,count,mean_r,rms_r"))
        if arch_meta.get("long_range_far_max_radius_multiplier") is None:
            long_range_far_max_radius_multiplier = derive_long_range_far_max_radius_multiplier(
                long_range_far_num_shells,
                long_range_far_shell_growth,
            )
        else:
            long_range_far_max_radius_multiplier = float(arch_meta.get("long_range_far_max_radius_multiplier"))
        long_range_far_source_norm = bool(arch_meta.get("long_range_far_source_norm", True))
        long_range_far_gate_init = float(arch_meta.get("long_range_far_gate_init", 0.0))
        feature_spectral_mode = str(arch_meta.get("feature_spectral_mode", "none"))
        feature_spectral_bottleneck_dim = int(arch_meta.get("feature_spectral_bottleneck_dim", 8))
        feature_spectral_mesh_size = int(arch_meta.get("feature_spectral_mesh_size", 16))
        feature_spectral_filter_hidden_dim = int(arch_meta.get("feature_spectral_filter_hidden_dim", 64))
        feature_spectral_boundary = str(arch_meta.get("feature_spectral_boundary", "periodic"))
        feature_spectral_slab_padding_factor = int(arch_meta.get("feature_spectral_slab_padding_factor", 2))
        feature_spectral_neutralize = bool(arch_meta.get("feature_spectral_neutralize", True))
        feature_spectral_include_k0 = bool(arch_meta.get("feature_spectral_include_k0", False))
        feature_spectral_assignment = str(arch_meta.get("feature_spectral_assignment", "cic"))
        feature_spectral_gate_init = float(arch_meta.get("feature_spectral_gate_init", 0.0))

        if mode == "pure-cartesian-ictd":
            model = PureCartesianICTDTransformerLayer(
                max_embed_radius=config.max_radius,
                main_max_radius=config.max_radius_main,
                main_number_of_basis=config.number_of_basis_main,
                hidden_dim_conv=config.channel_in,
                hidden_dim_sh=config.get_hidden_dim_sh(),
                hidden_dim=config.emb_number_main_2,
                channel_in2=config.channel_in2,
                embedding_dim=config.embedding_dim,
                max_atomvalue=config.max_atomvalue,
                output_size=config.output_size,
                embed_size=config.embed_size,
                main_hidden_sizes3=config.main_hidden_sizes3,
                num_layers=config.num_layers,
                num_interaction=num_interaction,
                invariant_channels=invariant_channels,
                function_type_main=config.function_type,
                lmax=config.lmax,
                physical_tensor_outputs=physical_tensor_outputs,
                external_tensor_rank=external_tensor_rank,
                external_tensor_irrep=external_tensor_irrep,
                external_tensor_specs=external_tensor_specs,
                num_fidelity_levels=num_fidelity_levels,
                multi_fidelity_mode=multi_fidelity_mode,
                internal_compute_dtype=dtype,
                device=torch.device(device),
                long_range_mode=long_range_mode,
                long_range_hidden_dim=long_range_hidden_dim,
                long_range_boundary=long_range_boundary,
                long_range_neutralize=long_range_neutralize,
                long_range_filter_hidden_dim=long_range_filter_hidden_dim,
                long_range_kmax=long_range_kmax,
                long_range_mesh_size=long_range_mesh_size,
                long_range_slab_padding_factor=long_range_slab_padding_factor,
                long_range_include_k0=long_range_include_k0,
                long_range_source_channels=long_range_source_channels,
                long_range_backend=long_range_backend,
                long_range_reciprocal_backend=long_range_reciprocal_backend,
                long_range_energy_partition=long_range_energy_partition,
                long_range_green_mode=long_range_green_mode,
                long_range_assignment=long_range_assignment,
                long_range_theta=long_range_theta,
                long_range_leaf_size=long_range_leaf_size,
                long_range_multipole_order=long_range_multipole_order,
                long_range_far_source_dim=long_range_far_source_dim,
                long_range_far_num_shells=long_range_far_num_shells,
                long_range_far_shell_growth=long_range_far_shell_growth,
                long_range_far_tail=long_range_far_tail,
                long_range_far_tail_bins=long_range_far_tail_bins,
                long_range_far_stats=long_range_far_stats,
                long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
                long_range_far_source_norm=long_range_far_source_norm,
                long_range_far_gate_init=long_range_far_gate_init,
                feature_spectral_mode=feature_spectral_mode,
                feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
                feature_spectral_mesh_size=feature_spectral_mesh_size,
                feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
                feature_spectral_boundary=feature_spectral_boundary,
                feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
                feature_spectral_neutralize=feature_spectral_neutralize,
                feature_spectral_include_k0=feature_spectral_include_k0,
                feature_spectral_assignment=feature_spectral_assignment,
                feature_spectral_gate_init=feature_spectral_gate_init,
            ).to(device)
        elif mode == "pure-cartesian-ictd-o3":
            model = PureCartesianICTDO3TransformerLayer(
                max_embed_radius=config.max_radius,
                main_max_radius=config.max_radius_main,
                main_number_of_basis=config.number_of_basis_main,
                hidden_dim_conv=config.channel_in,
                hidden_dim_sh=config.get_hidden_dim_sh(),
                hidden_dim=config.emb_number_main_2,
                channel_in2=config.channel_in2,
                embedding_dim=config.embedding_dim,
                max_atomvalue=config.max_atomvalue,
                output_size=config.output_size,
                embed_size=config.embed_size,
                main_hidden_sizes3=config.main_hidden_sizes3,
                num_layers=config.num_layers,
                num_interaction=num_interaction,
                invariant_channels=invariant_channels,
                function_type_main=config.function_type,
                lmax=config.lmax,
                physical_tensor_outputs=physical_tensor_outputs,
                external_tensor_rank=external_tensor_rank,
                external_tensor_irrep=external_tensor_irrep,
                external_tensor_specs=external_tensor_specs,
                o3_irrep_preset=o3_irrep_preset,
                o3_active_irreps=o3_active_irreps,
                num_fidelity_levels=num_fidelity_levels,
                multi_fidelity_mode=multi_fidelity_mode,
                internal_compute_dtype=dtype,
                device=torch.device(device),
                long_range_mode=long_range_mode,
                long_range_hidden_dim=long_range_hidden_dim,
                long_range_boundary=long_range_boundary,
                long_range_neutralize=long_range_neutralize,
                long_range_filter_hidden_dim=long_range_filter_hidden_dim,
                long_range_kmax=long_range_kmax,
                long_range_mesh_size=long_range_mesh_size,
                long_range_slab_padding_factor=long_range_slab_padding_factor,
                long_range_include_k0=long_range_include_k0,
                long_range_source_channels=long_range_source_channels,
                long_range_backend=long_range_backend,
                long_range_reciprocal_backend=long_range_reciprocal_backend,
                long_range_energy_partition=long_range_energy_partition,
                long_range_green_mode=long_range_green_mode,
                long_range_assignment=long_range_assignment,
                long_range_theta=long_range_theta,
                long_range_leaf_size=long_range_leaf_size,
                long_range_multipole_order=long_range_multipole_order,
                long_range_far_source_dim=long_range_far_source_dim,
                long_range_far_num_shells=long_range_far_num_shells,
                long_range_far_shell_growth=long_range_far_shell_growth,
                long_range_far_tail=long_range_far_tail,
                long_range_far_tail_bins=long_range_far_tail_bins,
                long_range_far_stats=long_range_far_stats,
                long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
                long_range_far_source_norm=long_range_far_source_norm,
                long_range_far_gate_init=long_range_far_gate_init,
                feature_spectral_mode=feature_spectral_mode,
                feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
                feature_spectral_mesh_size=feature_spectral_mesh_size,
                feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
                feature_spectral_boundary=feature_spectral_boundary,
                feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
                feature_spectral_neutralize=feature_spectral_neutralize,
                feature_spectral_include_k0=feature_spectral_include_k0,
                feature_spectral_assignment=feature_spectral_assignment,
                feature_spectral_gate_init=feature_spectral_gate_init,
            ).to(device)
        elif mode == "pure-cartesian-ictd-save":
            ictd_tp_path_policy = ictd_tp_path_policy or ckpt.get("ictd_tp_path_policy") or arch_meta.get("ictd_tp_path_policy", "full")
            ictd_tp_max_rank_other = (
                ictd_tp_max_rank_other
                if ictd_tp_max_rank_other is not None
                else ckpt.get("ictd_tp_max_rank_other", arch_meta.get("ictd_tp_max_rank_other"))
            )
            model = PureCartesianICTDTransformerLayerSave(
                max_embed_radius=config.max_radius,
                main_max_radius=config.max_radius_main,
                main_number_of_basis=config.number_of_basis_main,
                hidden_dim_conv=config.channel_in,
                hidden_dim_sh=config.get_hidden_dim_sh(),
                hidden_dim=config.emb_number_main_2,
                channel_in2=config.channel_in2,
                embedding_dim=config.embedding_dim,
                max_atomvalue=config.max_atomvalue,
                output_size=config.output_size,
                embed_size=config.embed_size,
                main_hidden_sizes3=config.main_hidden_sizes3,
                num_layers=config.num_layers,
                num_interaction=num_interaction,
                invariant_channels=invariant_channels,
                function_type_main=config.function_type,
                lmax=config.lmax,
                ictd_tp_path_policy=ictd_tp_path_policy,
                ictd_tp_max_rank_other=ictd_tp_max_rank_other,
                internal_compute_dtype=dtype,
                device=torch.device(device),
                long_range_mode=long_range_mode,
                long_range_hidden_dim=long_range_hidden_dim,
                long_range_boundary=long_range_boundary,
                long_range_neutralize=long_range_neutralize,
                long_range_filter_hidden_dim=long_range_filter_hidden_dim,
                long_range_kmax=long_range_kmax,
                long_range_mesh_size=long_range_mesh_size,
                long_range_slab_padding_factor=long_range_slab_padding_factor,
                long_range_include_k0=long_range_include_k0,
                long_range_source_channels=long_range_source_channels,
                long_range_backend=long_range_backend,
                long_range_reciprocal_backend=long_range_reciprocal_backend,
                long_range_energy_partition=long_range_energy_partition,
                long_range_green_mode=long_range_green_mode,
                long_range_assignment=long_range_assignment,
                long_range_theta=long_range_theta,
                long_range_leaf_size=long_range_leaf_size,
                long_range_multipole_order=long_range_multipole_order,
                long_range_far_source_dim=long_range_far_source_dim,
                long_range_far_num_shells=long_range_far_num_shells,
                long_range_far_shell_growth=long_range_far_shell_growth,
                long_range_far_tail=long_range_far_tail,
                long_range_far_tail_bins=long_range_far_tail_bins,
                long_range_far_stats=long_range_far_stats,
                long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
                long_range_far_source_norm=long_range_far_source_norm,
                long_range_far_gate_init=long_range_far_gate_init,
                feature_spectral_mode=feature_spectral_mode,
                feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
                feature_spectral_mesh_size=feature_spectral_mesh_size,
                feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
                feature_spectral_boundary=feature_spectral_boundary,
                feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
                feature_spectral_neutralize=feature_spectral_neutralize,
                feature_spectral_include_k0=feature_spectral_include_k0,
                feature_spectral_assignment=feature_spectral_assignment,
                feature_spectral_gate_init=feature_spectral_gate_init,
            ).to(device)
        elif mode in ("pure-cartesian-sparse", "pure-cartesian-sparse-save"):
            max_rank_other = int(ckpt.get("max_rank_other", arch_meta.get("max_rank_other", 1)))
            k_policy = str(ckpt.get("k_policy", arch_meta.get("k_policy", "k0")))
            sparse_cls = (
                PureCartesianSparseTransformerLayerSave
                if mode == "pure-cartesian-sparse-save"
                else PureCartesianSparseTransformerLayer
            )
            model = sparse_cls(
                max_embed_radius=config.max_radius,
                main_max_radius=config.max_radius_main,
                main_number_of_basis=config.number_of_basis_main,
                hidden_dim_conv=config.channel_in,
                hidden_dim_sh=config.get_hidden_dim_sh(),
                hidden_dim=config.emb_number_main_2,
                channel_in2=config.channel_in2,
                embedding_dim=config.embedding_dim,
                max_atomvalue=config.max_atomvalue,
                output_size=config.output_size,
                embed_size=config.embed_size,
                main_hidden_sizes3=config.main_hidden_sizes3,
                num_layers=config.num_layers,
                num_interaction=num_interaction,
                invariant_channels=invariant_channels,
                function_type_main=config.function_type,
                lmax=config.lmax,
                max_rank_other=max_rank_other,
                k_policy=k_policy,
                physical_tensor_outputs=physical_tensor_outputs,
                external_tensor_rank=external_tensor_rank,
                external_tensor_specs=external_tensor_specs,
                num_fidelity_levels=num_fidelity_levels,
                multi_fidelity_mode=multi_fidelity_mode,
                device=torch.device(device),
                long_range_mode=long_range_mode,
                long_range_hidden_dim=long_range_hidden_dim,
                long_range_boundary=long_range_boundary,
                long_range_neutralize=long_range_neutralize,
                long_range_filter_hidden_dim=long_range_filter_hidden_dim,
                long_range_kmax=long_range_kmax,
                long_range_mesh_size=long_range_mesh_size,
                long_range_slab_padding_factor=long_range_slab_padding_factor,
                long_range_include_k0=long_range_include_k0,
                long_range_source_channels=long_range_source_channels,
                long_range_backend=long_range_backend,
                long_range_reciprocal_backend=long_range_reciprocal_backend,
                long_range_energy_partition=long_range_energy_partition,
                long_range_green_mode=long_range_green_mode,
                long_range_assignment=long_range_assignment,
                long_range_theta=long_range_theta,
                long_range_leaf_size=long_range_leaf_size,
                long_range_multipole_order=long_range_multipole_order,
                long_range_far_source_dim=long_range_far_source_dim,
                long_range_far_num_shells=long_range_far_num_shells,
                long_range_far_shell_growth=long_range_far_shell_growth,
                long_range_far_tail=long_range_far_tail,
                long_range_far_tail_bins=long_range_far_tail_bins,
                long_range_far_stats=long_range_far_stats,
                long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
                long_range_far_source_norm=long_range_far_source_norm,
                long_range_far_gate_init=long_range_far_gate_init,
                feature_spectral_mode=feature_spectral_mode,
                feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
                feature_spectral_mesh_size=feature_spectral_mesh_size,
                feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
                feature_spectral_boundary=feature_spectral_boundary,
                feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
                feature_spectral_neutralize=feature_spectral_neutralize,
                feature_spectral_include_k0=feature_spectral_include_k0,
                feature_spectral_assignment=feature_spectral_assignment,
                feature_spectral_gate_init=feature_spectral_gate_init,
            ).to(device)
        elif mode == "spherical-save-cue":
            try:
                import cuequivariance_torch  # noqa: F401
            except Exception as e:
                raise ImportError(
                    "tensor_product_mode='spherical-save-cue' requires cuEquivariance.\n"
                    "Install: pip install cuequivariance-torch cuequivariance-ops-torch-cu12\n"
                    f"Original error: {e}"
                ) from e
            from molecular_force_field.models.cue_layers_channelwise import (
                E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise_cue,
            )
            model = E3_TransformerLayer_multi_channelwise_cue(
                max_embed_radius=config.max_radius,
                main_max_radius=config.max_radius_main,
                main_number_of_basis=config.number_of_basis_main,
                irreps_input=config.get_irreps_output_conv(),
                irreps_query=config.get_irreps_query_main(),
                irreps_key=config.get_irreps_key_main(),
                irreps_value=config.get_irreps_value_main(),
                irreps_output=config.get_irreps_output_conv_2(),
                irreps_sh=config.get_irreps_sh_transformer(),
                hidden_dim_sh=config.get_hidden_dim_sh(),
                hidden_dim=config.emb_number_main_2,
                channel_in2=config.channel_in2,
                embedding_dim=config.embedding_dim,
                max_atomvalue=config.max_atomvalue,
                output_size=config.output_size,
                embed_size=config.embed_size,
                main_hidden_sizes3=config.main_hidden_sizes3,
                num_layers=config.num_layers,
                num_interaction=num_interaction,
                function_type_main=config.function_type,
                device=torch.device(device),
                force_naive=force_naive,
                long_range_mode=long_range_mode,
                long_range_hidden_dim=long_range_hidden_dim,
                long_range_boundary=long_range_boundary,
                long_range_neutralize=long_range_neutralize,
                long_range_filter_hidden_dim=long_range_filter_hidden_dim,
                long_range_kmax=long_range_kmax,
                long_range_mesh_size=long_range_mesh_size,
                long_range_slab_padding_factor=long_range_slab_padding_factor,
                long_range_include_k0=long_range_include_k0,
                long_range_source_channels=long_range_source_channels,
                long_range_backend=long_range_backend,
                long_range_reciprocal_backend=long_range_reciprocal_backend,
                long_range_energy_partition=long_range_energy_partition,
                long_range_green_mode=long_range_green_mode,
                long_range_assignment=long_range_assignment,
                long_range_theta=long_range_theta,
                long_range_leaf_size=long_range_leaf_size,
                long_range_multipole_order=long_range_multipole_order,
                long_range_far_source_dim=long_range_far_source_dim,
                long_range_far_num_shells=long_range_far_num_shells,
                long_range_far_shell_growth=long_range_far_shell_growth,
                long_range_far_tail=long_range_far_tail,
                long_range_far_tail_bins=long_range_far_tail_bins,
                long_range_far_stats=long_range_far_stats,
                long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
                long_range_far_source_norm=long_range_far_source_norm,
                long_range_far_gate_init=long_range_far_gate_init,
                feature_spectral_mode=feature_spectral_mode,
                feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
                feature_spectral_mesh_size=feature_spectral_mesh_size,
                feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
                feature_spectral_boundary=feature_spectral_boundary,
                feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
                feature_spectral_neutralize=feature_spectral_neutralize,
                feature_spectral_include_k0=feature_spectral_include_k0,
                feature_spectral_assignment=feature_spectral_assignment,
                feature_spectral_gate_init=feature_spectral_gate_init,
            ).to(device)
        elif mode == "spherical-save":
            model = E3_TransformerLayer_multi_channelwise(
                max_embed_radius=config.max_radius,
                main_max_radius=config.max_radius_main,
                main_number_of_basis=config.number_of_basis_main,
                irreps_input=config.get_irreps_output_conv(),
                irreps_query=config.get_irreps_query_main(),
                irreps_key=config.get_irreps_key_main(),
                irreps_value=config.get_irreps_value_main(),
                irreps_output=config.get_irreps_output_conv_2(),
                irreps_sh=config.get_irreps_sh_transformer(),
                hidden_dim_sh=config.get_hidden_dim_sh(),
                hidden_dim=config.emb_number_main_2,
                channel_in2=config.channel_in2,
                embedding_dim=config.embedding_dim,
                max_atomvalue=config.max_atomvalue,
                output_size=config.output_size,
                embed_size=config.embed_size,
                main_hidden_sizes3=config.main_hidden_sizes3,
                num_layers=config.num_layers,
                num_interaction=num_interaction,
                invariant_channels=invariant_channels,
                function_type_main=config.function_type,
                device=torch.device(device),
                long_range_mode=long_range_mode,
                long_range_hidden_dim=long_range_hidden_dim,
                long_range_boundary=long_range_boundary,
                long_range_neutralize=long_range_neutralize,
                long_range_filter_hidden_dim=long_range_filter_hidden_dim,
                long_range_kmax=long_range_kmax,
                long_range_mesh_size=long_range_mesh_size,
                long_range_slab_padding_factor=long_range_slab_padding_factor,
                long_range_include_k0=long_range_include_k0,
                long_range_source_channels=long_range_source_channels,
                long_range_backend=long_range_backend,
                long_range_reciprocal_backend=long_range_reciprocal_backend,
                long_range_energy_partition=long_range_energy_partition,
                long_range_green_mode=long_range_green_mode,
                long_range_assignment=long_range_assignment,
                long_range_theta=long_range_theta,
                long_range_leaf_size=long_range_leaf_size,
                long_range_multipole_order=long_range_multipole_order,
                long_range_far_source_dim=long_range_far_source_dim,
                long_range_far_num_shells=long_range_far_num_shells,
                long_range_far_shell_growth=long_range_far_shell_growth,
                long_range_far_tail=long_range_far_tail,
                long_range_far_tail_bins=long_range_far_tail_bins,
                long_range_far_stats=long_range_far_stats,
                long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
                long_range_far_source_norm=long_range_far_source_norm,
                long_range_far_gate_init=long_range_far_gate_init,
                feature_spectral_mode=feature_spectral_mode,
                feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
                feature_spectral_mesh_size=feature_spectral_mesh_size,
                feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
                feature_spectral_boundary=feature_spectral_boundary,
                feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
                feature_spectral_neutralize=feature_spectral_neutralize,
                feature_spectral_include_k0=feature_spectral_include_k0,
                feature_spectral_assignment=feature_spectral_assignment,
                feature_spectral_gate_init=feature_spectral_gate_init,
            ).to(device)
        else:
            model = E3_TransformerLayer_multi(
                max_embed_radius=config.max_radius,
                main_max_radius=config.max_radius_main,
                main_number_of_basis=config.number_of_basis_main,
                irreps_input=config.get_irreps_output_conv(),
                irreps_query=config.get_irreps_query_main(),
                irreps_key=config.get_irreps_key_main(),
                irreps_value=config.get_irreps_value_main(),
                irreps_output=config.get_irreps_output_conv_2(),
                irreps_sh=config.get_irreps_sh_transformer(),
                hidden_dim_sh=config.get_hidden_dim_sh(),
                hidden_dim=config.emb_number_main_2,
                channel_in2=config.channel_in2,
                embedding_dim=config.embedding_dim,
                max_atomvalue=config.max_atomvalue,
                output_size=config.output_size,
                embed_size=config.embed_size,
                main_hidden_sizes3=config.main_hidden_sizes3,
                num_layers=config.num_layers,
                num_interaction=num_interaction,
                invariant_channels=invariant_channels,
                function_type_main=config.function_type,
                device=torch.device(device),
                long_range_mode=long_range_mode,
                long_range_hidden_dim=long_range_hidden_dim,
                long_range_boundary=long_range_boundary,
                long_range_neutralize=long_range_neutralize,
                long_range_filter_hidden_dim=long_range_filter_hidden_dim,
                long_range_kmax=long_range_kmax,
                long_range_mesh_size=long_range_mesh_size,
                long_range_slab_padding_factor=long_range_slab_padding_factor,
                long_range_include_k0=long_range_include_k0,
                long_range_source_channels=long_range_source_channels,
                long_range_backend=long_range_backend,
                long_range_reciprocal_backend=long_range_reciprocal_backend,
                long_range_energy_partition=long_range_energy_partition,
                long_range_green_mode=long_range_green_mode,
                long_range_assignment=long_range_assignment,
                long_range_theta=long_range_theta,
                long_range_leaf_size=long_range_leaf_size,
                long_range_multipole_order=long_range_multipole_order,
                long_range_far_source_dim=long_range_far_source_dim,
                long_range_far_num_shells=long_range_far_num_shells,
                long_range_far_shell_growth=long_range_far_shell_growth,
                long_range_far_tail=long_range_far_tail,
                long_range_far_tail_bins=long_range_far_tail_bins,
                long_range_far_stats=long_range_far_stats,
                long_range_far_max_radius_multiplier=long_range_far_max_radius_multiplier,
                long_range_far_source_norm=long_range_far_source_norm,
                long_range_far_gate_init=long_range_far_gate_init,
                feature_spectral_mode=feature_spectral_mode,
                feature_spectral_bottleneck_dim=feature_spectral_bottleneck_dim,
                feature_spectral_mesh_size=feature_spectral_mesh_size,
                feature_spectral_filter_hidden_dim=feature_spectral_filter_hidden_dim,
                feature_spectral_boundary=feature_spectral_boundary,
                feature_spectral_slab_padding_factor=feature_spectral_slab_padding_factor,
                feature_spectral_neutralize=feature_spectral_neutralize,
                feature_spectral_include_k0=feature_spectral_include_k0,
                feature_spectral_assignment=feature_spectral_assignment,
                feature_spectral_gate_init=feature_spectral_gate_init,
            ).to(device)
        if mode == "spherical-save-cue":
            load_result = model.load_state_dict(selected_state_dict, strict=False)
            if load_result.unexpected_keys or load_result.missing_keys:
                import warnings
                if load_result.unexpected_keys:
                    warnings.warn(
                        f"spherical-save-cue: {len(load_result.unexpected_keys)} unexpected keys in checkpoint "
                        "(cuEquivariance 版本差异?), 已忽略"
                    )
                if load_result.missing_keys:
                    # cuEquivariance auto-generated CG coefficient buffers (e.g. .graphs.0.graph.cN)
                    # are not stored in older checkpoints but get recomputed at init. Safe to skip.
                    cue_auto_buffers = [k for k in load_result.missing_keys if ".graphs." in k and ".graph.c" in k]
                    real_missing = [k for k in load_result.missing_keys if k not in cue_auto_buffers]
                    if cue_auto_buffers:
                        warnings.warn(
                            f"spherical-save-cue: {len(cue_auto_buffers)} auto-generated CG buffers 未在 checkpoint 中"
                            "（cuEquivariance 版本差异），已由模型自动初始化"
                        )
                    if real_missing:
                        raise RuntimeError(
                            f"spherical-save-cue: {len(real_missing)} missing learned keys: "
                            f"{real_missing[:10]}... checkpoint 与模型结构不匹配."
                        )
        else:
            model.load_state_dict(selected_state_dict, strict=True)

        model = maybe_wrap_model_with_zbl(model, arch_meta)

        # Optional TorchScript tracing
        use_ts = bool(torchscript) or (os.environ.get("MLIAP_USE_TORCHSCRIPT", "").lower() in ("1", "true", "yes"))
        if use_ts:
            _ts_supported = ("pure-cartesian-ictd", "pure-cartesian-ictd-o3", "pure-cartesian-ictd-save", "spherical-save-cue")
            if mode not in _ts_supported:
                raise ValueError(f"TorchScript export is only supported for {_ts_supported}, got {mode!r}")
            model = _maybe_torchscript_trace_model(
                model,
                device=torch.device(device),
                dtype=dtype,
                enable=True,
            )

        return cls(
            model=model,
            element_types=element_types,
            max_radius=radius,
            atomic_energy_keys=aek,
            atomic_energy_values=aev,
            device=device,
            dtype=dtype,
        )

    def _init_device(self, data):
        """Detect device from data tensors (GPU if Kokkos, else CPU)."""
        try:
            using_kokkos = "kokkos" in data.__class__.__module__.lower()
        except Exception:
            using_kokkos = False

        self._using_kokkos = using_kokkos
        self._has_gpu_api = hasattr(data, "update_pair_forces_gpu")
        self._has_exchange = hasattr(data, "forward_exchange")

        if using_kokkos:
            device = torch.as_tensor(data.elems).device
        else:
            device = torch.device("cpu")
        self.device = device
        self.wrapper = self.wrapper.to(device)
        self._elem_to_Z = self._elem_to_Z.to(device)

        # Optional: torch.compile for 2-5x speedup (PyTorch 2.0+)
        if os.environ.get("MLIAP_USE_COMPILE", "").lower() in ("1", "true", "yes"):
            try:
                # Prewarm one-time caches (keeps Python-side setup out of Dynamo tracing)
                if os.environ.get("MLIAP_PREWARM", "1").lower() not in ("0", "false", "no"):
                    try:
                        with torch.no_grad():
                            for m in self.wrapper.model.modules():
                                prewarm = getattr(m, "prewarm_caches", None)
                                if callable(prewarm):
                                    prewarm(device=self.device, dtype=self.dtype)
                        print("[MLIAP] prewarmed model caches for compile", flush=True)
                    except Exception as e:
                        print(f"[MLIAP] cache prewarm skipped: {e}", flush=True)

                # In Kokkos+CUDA environments, CUDA Graph capture can cause warnings
                # and sometimes interfere with Kokkos finalize. Disable CUDA graphs by default.
                if using_kokkos and os.environ.get("MLIAP_DISABLE_CUDAGRAPHS", "").lower() not in ("0", "false", "no"):
                    try:
                        import torch._inductor.config as _icfg  # type: ignore
                        _icfg.triton.cudagraphs = False
                        print("[MLIAP] disabled inductor CUDA graphs (Kokkos safety)", flush=True)
                    except Exception:
                        pass

                mode = os.environ.get("MLIAP_COMPILE_MODE", "reduce-overhead")
                self.wrapper = torch.compile(self.wrapper, mode=mode, dynamic=True)
                print(f"[MLIAP] torch.compile enabled (mode={mode})", flush=True)
            except Exception as e:
                print(f"[MLIAP] torch.compile failed: {e}, using eager", flush=True)

        self.initialized = True

    def _to_torch(self, x, *, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convert LAMMPS-provided arrays to torch.Tensor with minimal copies.

        Supports:
        - torch.Tensor (no copy unless dtype/device mismatch)
        - numpy.ndarray (CPU)
        - cupy.ndarray / objects implementing DLPack (GPU, zero-copy when dtype matches)
        """
        if torch.is_tensor(x):
            t = x
        elif isinstance(x, np.ndarray):
            t = torch.as_tensor(x)
        else:
            # Prefer DLPack for GPU arrays (e.g., cupy.ndarray)
            if hasattr(x, "__dlpack__"):
                t = torch_dlpack.from_dlpack(x)
            elif hasattr(x, "toDlpack"):
                t = torch_dlpack.from_dlpack(x.toDlpack())
            else:
                # Fallback: try torch.as_tensor (may copy)
                t = torch.as_tensor(x)

        if device is not None and t.device != device:
            t = t.to(device)
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return t

    def compute_forces(self, data):
        """Compute per-atom forces and write directly into the LAMMPS force buffer.

        Uses ``dE/d(pos)`` (per-atom, O(N)) instead of ``dE/d(edge_vec)``
        (per-pair, O(N*M)), reducing autograd leaf-gradient storage.

        Global virial is handled by LAMMPS C++ side via ``virial_fdotr_compute()``.

        Supports two modes:
        - Standard ML-IAP (CPU): writes to ``data.f`` numpy view
        - ML-IAP-Kokkos (GPU): writes to ``data.f`` GPU tensor
        """
        natoms = data.nlocal
        ntotal = data.ntotal
        npairs = data.npairs

        if not self.initialized:
            self._init_device(data)
            if os.environ.get("MLIAP_DEBUG", "").lower() in ("1", "true", "yes"):
                def _fmt_arr(x):
                    try:
                        if torch.is_tensor(x):
                            return (f"torch.Tensor(shape={tuple(x.shape)}, dtype={x.dtype}, "
                                    f"device={x.device}, contiguous={x.is_contiguous()})")
                        if isinstance(x, np.ndarray):
                            return (f"np.ndarray(shape={x.shape}, dtype={x.dtype}, "
                                    f"c_contig={bool(x.flags['C_CONTIGUOUS'])})")
                        # cupy.ndarray (or other array-likes) common in ML-IAP-Kokkos
                        mod = type(x).__module__
                        name = type(x).__name__
                        if mod.startswith("cupy"):
                            try:
                                dev = getattr(getattr(x, "device", None), "id", None)
                            except Exception:
                                dev = None
                            return f"{mod}.{name}(shape={getattr(x,'shape',None)}, dtype={getattr(x,'dtype',None)}, device={dev})"
                        return f"{type(x)}"
                    except Exception as _e:  # pragma: no cover
                        return f"{type(x)} (fmt_error={_e})"

                model_cls = type(self.wrapper.model).__name__
                model_mod = type(self.wrapper.model).__module__
                nghost = ntotal - natoms
                inflate = ntotal / natoms if natoms > 0 else 0
                pairs_per_local = npairs / natoms if natoms > 0 else 0
                print(
                    f"[MLIAP] model={model_mod}.{model_cls}, kokkos={self._using_kokkos}, "
                    f"gpu_api={self._has_gpu_api}, device={self.device}",
                    flush=True,
                )
                print(
                    f"[MLIAP] nlocal={natoms}, ntotal={ntotal}, nghost={nghost}, npairs={npairs} | "
                    f"ntotal/nlocal={inflate:.2f}x, npairs/nlocal={pairs_per_local:.1f}",
                    flush=True,
                )
                print(f"[MLIAP] data.rij: {_fmt_arr(getattr(data, 'rij', None))}", flush=True)
                print(f"[MLIAP] data.elems: {_fmt_arr(getattr(data, 'elems', None))}", flush=True)
                print(f"[MLIAP] data.pair_i: {_fmt_arr(getattr(data, 'pair_i', None))}", flush=True)
                print(f"[MLIAP] data.pair_j: {_fmt_arr(getattr(data, 'pair_j', None))}", flush=True)
                print(f"[MLIAP] data.f: {_fmt_arr(getattr(data, 'f', None))}", flush=True)
                print(f"[MLIAP] data.eatoms: {_fmt_arr(getattr(data, 'eatoms', None))}", flush=True)

                # DLPack / zero-copy verification (Cupy -> Torch)
                def _ptr_cupy(x):
                    try:
                        # cupy.ndarray
                        return int(x.data.ptr)
                    except Exception:
                        return None

                def _verify_dlpack(name: str, x, *, dtype=None):
                    try:
                        t = self._to_torch(x, dtype=dtype, device=self.device)
                        cupy_ptr = _ptr_cupy(x) if type(x).__module__.startswith("cupy") else None
                        torch_ptr = int(t.data_ptr())
                        extra = ""
                        if cupy_ptr is not None:
                            extra = f", cupy_ptr=0x{cupy_ptr:x}, torch_ptr=0x{torch_ptr:x}, same_ptr={cupy_ptr == torch_ptr}"
                        print(
                            f"[MLIAP] to_torch({name}): shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}{extra}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[MLIAP] to_torch({name}) failed: {e}", flush=True)

                _verify_dlpack("rij", getattr(data, "rij", None), dtype=self.dtype)
                # For int32 indices (cupy): dtype=None shows true zero-copy;
                # dtype=torch.long shows the required cast (copy) for PyTorch indexing.
                _verify_dlpack("elems(raw)", getattr(data, "elems", None), dtype=None)
                _verify_dlpack("elems(i64)", getattr(data, "elems", None), dtype=torch.long)
                _verify_dlpack("pair_i(raw)", getattr(data, "pair_i", None), dtype=None)
                _verify_dlpack("pair_i(i64)", getattr(data, "pair_i", None), dtype=torch.long)
                _verify_dlpack("pair_j(raw)", getattr(data, "pair_j", None), dtype=None)
                _verify_dlpack("pair_j(i64)", getattr(data, "pair_j", None), dtype=torch.long)
                _verify_dlpack("f", getattr(data, "f", None), dtype=None)
                _verify_dlpack("eatoms", getattr(data, "eatoms", None), dtype=None)

        if natoms == 0 or npairs <= 1:
            return

        # --- Build tensors from LAMMPS data ---
        rij = self._to_torch(data.rij, dtype=self.dtype, device=self.device)

        # elems/pair_i/pair_j are often cupy int32; casting to int64 is required for indexing.
        # Cache casts by cupy pointer to avoid per-step copies when neighbor list is unchanged.
        try:
            elems_ptr = int(data.elems.data.ptr)  # cupy
        except Exception:
            elems_ptr = None
        if (
            elems_ptr is not None
            and self._cache_elems_ptr == elems_ptr
            and self._cache_elems_i64 is not None
            and int(self._cache_elems_i64.numel()) == int(ntotal)
        ):
            elem_idx = self._cache_elems_i64
        else:
            elem_raw = self._to_torch(data.elems, dtype=None, device=self.device)
            elem_idx = elem_raw.to(torch.long) if elem_raw.dtype != torch.long else elem_raw
            if elems_ptr is not None:
                self._cache_elems_ptr = elems_ptr
                self._cache_elems_i64 = elem_idx

        lut = self._elem_to_Z.to(device=self.device)
        species = lut[elem_idx]

        try:
            pair_i_ptr = int(data.pair_i.data.ptr)  # cupy
        except Exception:
            pair_i_ptr = None
        try:
            pair_j_ptr = int(data.pair_j.data.ptr)  # cupy
        except Exception:
            pair_j_ptr = None

        if (
            pair_i_ptr is not None
            and self._cache_pair_i_ptr == pair_i_ptr
            and self._cache_pair_i_i64 is not None
            and int(self._cache_pair_i_i64.numel()) == int(npairs)
        ):
            edge_src = self._cache_pair_i_i64
        else:
            pi_raw = self._to_torch(data.pair_i, dtype=None, device=self.device)
            edge_src = pi_raw.to(torch.long) if pi_raw.dtype != torch.long else pi_raw
            if pair_i_ptr is not None:
                self._cache_pair_i_ptr = pair_i_ptr
                self._cache_pair_i_i64 = edge_src

        if (
            pair_j_ptr is not None
            and self._cache_pair_j_ptr == pair_j_ptr
            and self._cache_pair_j_i64 is not None
            and int(self._cache_pair_j_i64.numel()) == int(npairs)
        ):
            edge_dst = self._cache_pair_j_i64
        else:
            pj_raw = self._to_torch(data.pair_j, dtype=None, device=self.device)
            edge_dst = pj_raw.to(torch.long) if pj_raw.dtype != torch.long else pj_raw
            if pair_j_ptr is not None:
                self._cache_pair_j_ptr = pair_j_ptr
                self._cache_pair_j_i64 = edge_dst

        # Robustness: Kokkos buffers can be reused; enforce consistent edge length.
        # All edge-index arrays must match rij length.
        E_rij = int(rij.shape[0])
        E_i = int(edge_src.numel())
        E_j = int(edge_dst.numel())
        Emin = min(E_rij, E_i, E_j)
        if Emin <= 0:
            return
        if Emin != E_rij or Emin != E_i or Emin != E_j:
            if os.environ.get("MLIAP_DEBUG", "").lower() in ("1", "true", "yes"):
                print(
                    f"[MLIAP] WARN: edge length mismatch (rij={E_rij}, pair_i={E_i}, pair_j={E_j}), "
                    f"using Emin={Emin}",
                    flush=True,
                )
            rij = rij[:Emin]
            edge_src = edge_src[:Emin]
            edge_dst = edge_dst[:Emin]
            # keep npairs consistent for downstream buffers
            npairs = Emin

        # Reuse buffers when sizes unchanged (typical in MD)
        if self._cache_ntotal == ntotal and self._cache_batch is not None:
            batch = self._cache_batch.zero_()
        else:
            batch = torch.zeros(ntotal, dtype=torch.long, device=self.device)
            self._cache_batch = batch
            self._cache_ntotal = ntotal
        if self._cache_npairs == npairs and self._cache_edge_shifts is not None:
            edge_shifts = self._cache_edge_shifts.zero_()
        else:
            edge_shifts = torch.zeros(npairs, 3, dtype=self.dtype, device=self.device)
            self._cache_edge_shifts = edge_shifts
            self._cache_npairs = npairs
        if self._cache_cell is None:
            self._cache_cell = torch.eye(3, dtype=self.dtype, device=self.device).unsqueeze(0) * 100.0
        cell = self._cache_cell

        # --- Forward (atom forces via dE/d(pos)) ---
        E_total, atom_energies, atom_forces = self.wrapper(
            rij, species, batch, edge_src, edge_dst,
            edge_shifts, cell, nlocal=natoms,
        )

        # --- Write back to LAMMPS ---
        # NOTE: .item() forces GPU→CPU sync; unavoidable for LAMMPS energy
        data.energy = E_total.item()

        atom_e = atom_energies.squeeze(-1).detach()
        forces = atom_forces.detach()
        if forces.dtype != torch.float64:
            forces = forces.to(torch.float64)

        if self._using_kokkos and self._has_gpu_api:
            self._writeback_kokkos(data, atom_e, forces, natoms, ntotal)
        else:
            self._writeback_cpu(data, atom_e, forces, natoms, ntotal)

    # ------------------------------------------------------------------
    # Write-back helpers
    # ------------------------------------------------------------------

    def _writeback_cpu(self, data, atom_e, forces, natoms, ntotal):
        """CPU path: write per-atom energies and forces via numpy views."""
        ae_np = atom_e.cpu().numpy().astype(np.float64)
        data.eatoms = ae_np

        f_view = data.f
        forces_np = forces.cpu().numpy()
        flat_f = np.asarray(f_view).ravel()
        flat_f[:ntotal * 3] += forces_np[:ntotal].ravel()

    def _writeback_kokkos(self, data, atom_e, forces, natoms, ntotal):
        """Kokkos GPU path: write per-atom energies and forces via GPU tensors."""
        eatoms_t = self._to_torch(data.eatoms, device=self.device)
        eatoms_t.copy_(atom_e[:natoms].to(eatoms_t.dtype))

        f_t = self._to_torch(data.f, device=self.device)
        f_flat = f_t.view(-1)
        f_flat[:ntotal * 3] += forces[:ntotal].to(f_t.dtype).view(-1)

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass
