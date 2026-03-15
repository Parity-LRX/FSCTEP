from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn


_ZBL_COEFF = (0.1818, 0.5099, 0.2802, 0.02817)
_ZBL_EXP = (3.2, 0.9423, 0.4029, 0.2016)
_COULOMB_EV_ANGSTROM = 14.3996454784255
_BOHR_RADIUS_ANGSTROM = 0.529177210903
_ZBL_SCREENING_PREFAC = 0.8854 * _BOHR_RADIUS_ANGSTROM


@dataclass
class ZBLConfig:
    enabled: bool = False
    inner_cutoff: float = 0.8
    outer_cutoff: float = 1.2
    exponent: float = 0.23
    energy_scale: float = 1.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "ZBLConfig":
        if mapping is None:
            return cls()
        return cls(
            enabled=bool(mapping.get("zbl_enabled", False)),
            inner_cutoff=float(mapping.get("zbl_inner_cutoff", 0.8)),
            outer_cutoff=float(mapping.get("zbl_outer_cutoff", 1.2)),
            exponent=float(mapping.get("zbl_exponent", 0.23)),
            energy_scale=float(mapping.get("zbl_energy_scale", 1.0)),
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        if self.inner_cutoff <= 0.0:
            raise ValueError("zbl_inner_cutoff must be > 0")
        if self.outer_cutoff <= self.inner_cutoff:
            raise ValueError("zbl_outer_cutoff must be > zbl_inner_cutoff")
        if self.exponent <= 0.0:
            raise ValueError("zbl_exponent must be > 0")
        if self.energy_scale < 0.0:
            raise ValueError("zbl_energy_scale must be >= 0")


def _smooth_switch(distance: torch.Tensor, inner: float, outer: float) -> torch.Tensor:
    if outer <= inner:
        raise ValueError("outer cutoff must be greater than inner cutoff")
    t = (distance - inner) / (outer - inner)
    t = torch.clamp(t, 0.0, 1.0)
    poly = ((-6.0 * t + 15.0) * t - 10.0) * t * t * t + 1.0
    return torch.where(distance <= inner, torch.ones_like(distance), torch.where(distance >= outer, torch.zeros_like(distance), poly))


def compute_zbl_pair_energy(
    edge_vec: torch.Tensor,
    atomic_numbers: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    *,
    inner_cutoff: float,
    outer_cutoff: float,
    exponent: float,
    energy_scale: float,
) -> torch.Tensor:
    if edge_vec.ndim != 2 or edge_vec.size(-1) != 3:
        raise ValueError("edge_vec must have shape (E, 3)")
    distance = torch.linalg.norm(edge_vec, dim=-1)
    safe_distance = torch.clamp(distance, min=1.0e-12)

    zi = atomic_numbers.index_select(0, edge_src).to(dtype=edge_vec.dtype)
    zj = atomic_numbers.index_select(0, edge_dst).to(dtype=edge_vec.dtype)
    screening = _ZBL_SCREENING_PREFAC / (
        torch.pow(torch.clamp(zi, min=1.0), exponent) + torch.pow(torch.clamp(zj, min=1.0), exponent)
    )
    x = safe_distance / screening
    phi = torch.zeros_like(x)
    for coeff, exponent_i in zip(_ZBL_COEFF, _ZBL_EXP):
        phi = phi + coeff * torch.exp(-exponent_i * x)
    energy = energy_scale * _COULOMB_EV_ANGSTROM * zi * zj * phi / safe_distance
    return energy * _smooth_switch(safe_distance, inner_cutoff, outer_cutoff)


def _edge_vec_from_inputs(
    pos: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_shifts: torch.Tensor,
    cell: torch.Tensor,
    precomputed_edge_vec: Optional[torch.Tensor],
) -> torch.Tensor:
    if precomputed_edge_vec is not None:
        return precomputed_edge_vec
    edge_vec = pos.index_select(0, edge_dst) - pos.index_select(0, edge_src)
    if edge_shifts.numel() == 0:
        return edge_vec
    if cell.dim() == 2:
        cell_matrix = cell
    elif cell.dim() == 3:
        if cell.size(0) != 1:
            raise ValueError("ZBL wrapper currently expects a single cell matrix for one graph")
        cell_matrix = cell[0]
    else:
        raise ValueError("cell must have shape (3, 3) or (1, 3, 3)")
    return edge_vec + edge_shifts.to(dtype=pos.dtype) @ cell_matrix.to(dtype=pos.dtype)


class ZBLRepulsionWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, config: ZBLConfig):
        super().__init__()
        config.validate()
        self.base_model = base_model
        self.zbl_enabled = bool(config.enabled)
        self.zbl_inner_cutoff = float(config.inner_cutoff)
        self.zbl_outer_cutoff = float(config.outer_cutoff)
        self.zbl_exponent = float(config.exponent)
        self.zbl_energy_scale = float(config.energy_scale)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def state_dict(self, *args, **kwargs):
        return self.base_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self.base_model.load_state_dict(state_dict, strict=strict, assign=assign)

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
        return_physical_tensors: bool = False,
        return_reciprocal_source: bool = False,
        sync_after_scatter=None,
    ):
        kwargs = {
            "precomputed_edge_vec": precomputed_edge_vec,
            "sync_after_scatter": sync_after_scatter,
        }
        if external_tensor is not None:
            kwargs["external_tensor"] = external_tensor
        if return_physical_tensors:
            kwargs["return_physical_tensors"] = True
        if return_reciprocal_source:
            kwargs["return_reciprocal_source"] = True

        out = self.base_model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, **kwargs)
        if not self.zbl_enabled:
            return out

        atom_energy = out[0] if isinstance(out, tuple) else out
        edge_vec = _edge_vec_from_inputs(pos, edge_src, edge_dst, edge_shifts, cell, precomputed_edge_vec)
        pair_energy = compute_zbl_pair_energy(
            edge_vec,
            A,
            edge_src,
            edge_dst,
            inner_cutoff=self.zbl_inner_cutoff,
            outer_cutoff=self.zbl_outer_cutoff,
            exponent=self.zbl_exponent,
            energy_scale=self.zbl_energy_scale,
        )
        zbl_atom = torch.zeros(atom_energy.shape[0], dtype=atom_energy.dtype, device=atom_energy.device)
        half_pair = 0.5 * pair_energy.to(dtype=atom_energy.dtype)
        zbl_atom.index_add_(0, edge_src, half_pair)
        zbl_atom.index_add_(0, edge_dst, half_pair)
        if atom_energy.dim() == 2:
            zbl_atom = zbl_atom.unsqueeze(-1)
        atom_energy = atom_energy + zbl_atom
        if isinstance(out, tuple):
            return (atom_energy, *out[1:])
        return atom_energy


def maybe_wrap_model_with_zbl(
    model: nn.Module,
    source: Mapping[str, Any] | ZBLConfig | None,
) -> nn.Module:
    config = source if isinstance(source, ZBLConfig) else ZBLConfig.from_mapping(source)
    config.validate()
    if not config.enabled:
        return model
    if isinstance(model, ZBLRepulsionWrapper):
        return model
    return ZBLRepulsionWrapper(model, config)
