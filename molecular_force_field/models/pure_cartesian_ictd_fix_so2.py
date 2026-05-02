from __future__ import annotations

import math

import torch
import torch.nn as nn

from molecular_force_field.models.ictd_irreps import EdgeWeightedPathPreservingTensorProduct
from molecular_force_field.models.ictd_irreps_2d import HarmonicElementwiseProductSO2
from molecular_force_field.models.mlp import MainNet
from molecular_force_field.models.node_local_so2_helpers import (
    _get_harmonic_full_maps,
    apply_channel_linear,
    build_node_local_frames,
    concat_node_local_so2_states,
    get_local_q,
    harmonic_row_rotation,
    node_local_so2_change_frame,
    node_local_so2_invariants,
    so3_local_so2_basis_cpu,
)
from molecular_force_field.models.pure_cartesian_ictd_fix import (
    MACEStyleScalarReadoutSO3,
    PathPreservingLinearSO3,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_node_so2 import (
    LowRankLinear,
    _adapt_local_state_for_product5,
)
from molecular_force_field.models.pure_cartesian_ictd_layers import resolve_save_multiple_mix_channels
from molecular_force_field.utils.scatter import scatter


def _resolve_internal_compute_dtype(internal_compute_dtype: torch.dtype | None) -> torch.dtype:
    return torch.get_default_dtype() if internal_compute_dtype is None else internal_compute_dtype


def _init_so2_path_tp_weight_to_one_(module: nn.Module | None) -> None:
    if module is None:
        return
    weight = getattr(module, "weight", None)
    if isinstance(weight, nn.Parameter):
        with torch.no_grad():
            weight.fill_(1.0)


def _init_node_local_mixer_identity_(module: nn.Module | None) -> None:
    if module is None:
        return
    adapters = getattr(module, "adapters", None)
    if adapters is None:
        return
    for adapter in adapters.values():
        if isinstance(adapter, nn.Linear) and adapter.weight.shape[0] == adapter.weight.shape[1]:
            with torch.no_grad():
                adapter.weight.zero_()
                adapter.weight.diagonal().fill_(1.0)
                if adapter.bias is not None:
                    adapter.bias.zero_()


def _init_element_conditioned_local_identity_(module: nn.Module | None) -> None:
    if module is None:
        return
    weights = getattr(module, "weights", None)
    if weights is None:
        return
    with torch.no_grad():
        for weight in weights.values():
            weight.zero_()
            diag = torch.arange(weight.shape[-1], device=weight.device)
            weight[:, diag, diag] = 1.0
        bias = getattr(module, "bias", None)
        if bias is not None:
            for value in bias.values():
                value.zero_()


def _build_node_local_moments_escn(
    *,
    edge_vec: torch.Tensor,
    edge_length: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_mask: torch.Tensor,
    num_nodes: int,
    lmax: int,
    number_of_basis: int,
    max_radius: float,
    function_type: str,
    G_node: torch.Tensor | None,
    compute_dtype: torch.dtype,
):
    """eSCN-style: global ICTD SH → per-node SO3 scatter → per-node rotation.

    Σ rad_i · Y_l(G_j · n_i)  =  D_l(G_j) · Σ rad_i · Y_l(n_i)

    Spherical harmonics evaluated once in the global frame, scattered to
    per-node SO3 blocks, then ``so3_blocks_to_node_local_so2`` rotates and
    decomposes into (l,m).  O(E·l²) harmonic eval + O(N·l²) rotation replaces
    the original O(E) per-edge vector rotation + O(E·l²) harmonic eval."""
    from e3nn.math import soft_one_hot_linspace as _soft_one_hot
    from molecular_force_field.models.ictd_irreps import direction_harmonics_all as _sh

    dtype = edge_vec.dtype
    radial = (
        _soft_one_hot(
            edge_length,
            0.0,
            float(max_radius),
            int(number_of_basis),
            basis=str(function_type),
            cutoff=True,
        )
        .mul(int(number_of_basis) ** 0.5)
        .to(dtype=dtype)
    )
    neighbor_count = (
        scatter(edge_mask.to(dtype=dtype).view(-1), edge_dst, dim=0, dim_size=int(num_nodes), reduce="sum")
        .clamp(min=1.0)
    )

    n_global = edge_vec / edge_length.clamp(min=1e-8).unsqueeze(-1).to(dtype=dtype)
    Y_all = _sh(n_global, int(lmax))

    so3_blocks: dict[int, torch.Tensor] = {}
    for l in range(int(lmax) + 1):
        so3_blocks[l] = Y_all[l].unsqueeze(1).to(dtype=dtype)

    if G_node is None and int(lmax) > 0:
        raise ValueError("G_node is required when lmax > 0")
    if int(lmax) == 0:
        return radial, {(0, 0): so3_blocks[0]}, neighbor_count
    moments = _so3_blocks_to_node_local_so2_fast(
        so3_blocks,
        G_node[edge_dst],
        int(lmax),
        compute_dtype=compute_dtype,
    )
    return radial, moments, neighbor_count


def _build_edge_local_so3_moments_escn(
    *,
    edge_vec: torch.Tensor,
    edge_length: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_mask: torch.Tensor,
    num_nodes: int,
    lmax: int,
    number_of_basis: int,
    max_radius: float,
    function_type: str,
    G_node: torch.Tensor | None,
    compute_dtype: torch.dtype,
):
    from e3nn.math import soft_one_hot_linspace as _soft_one_hot
    from molecular_force_field.models.ictd_irreps import direction_harmonics_all as _sh

    dtype = edge_vec.dtype
    radial = (
        _soft_one_hot(
            edge_length,
            0.0,
            float(max_radius),
            int(number_of_basis),
            basis=str(function_type),
            cutoff=True,
        )
        .mul(int(number_of_basis) ** 0.5)
        .to(dtype=dtype)
    )
    neighbor_count = (
        scatter(edge_mask.to(dtype=dtype).view(-1), edge_dst, dim=0, dim_size=int(num_nodes), reduce="sum")
        .clamp(min=1.0)
    )
    if G_node is None and int(lmax) > 0:
        raise ValueError("G_node is required when lmax > 0")
    if int(lmax) == 0:
        n_local = edge_vec / edge_length.clamp(min=1e-8).unsqueeze(-1).to(dtype=dtype)
    else:
        edge_local_vec = torch.einsum(
            "nij,nj->ni",
            G_node[edge_dst].to(dtype=dtype),
            edge_vec,
        )
        n_local = edge_local_vec / edge_length.clamp(min=1e-8).unsqueeze(-1).to(dtype=dtype)
    Y_all = _sh(n_local, int(lmax))
    blocks = {l: Y_all[l].unsqueeze(1).to(dtype=dtype) for l in range(int(lmax) + 1)}
    if int(lmax) == 0:
        return radial, {0: blocks[0]}, neighbor_count
    return radial, blocks, neighbor_count


def _rotate_full_harmonic_fast(full: torch.Tensor, R: torch.Tensor, l: int) -> torch.Tensor:
    l = int(l)
    if l == 1:
        return torch.einsum("nai,nci->nca", R, full)
    if l == 2:
        tens = full.reshape(full.shape[0], full.shape[1], 3, 3)
        return torch.einsum("nai,ncij,nbj->ncab", R, tens, R).reshape(full.shape[0], full.shape[1], 9)
    if l == 3:
        tens = full.reshape(full.shape[0], full.shape[1], 3, 3, 3)
        return torch.einsum("nxi,nyj,nzk,nqijk->nqxyz", R, R, R, tens).reshape(
            full.shape[0],
            full.shape[1],
            27,
        )
    raise ValueError(f"fast full harmonic rotation only supports l=1,2,3, got {l}")


def _so3_blocks_to_node_local_so3_fast(
    blocks: dict[int, torch.Tensor],
    G: torch.Tensor,
    lmax: int,
    *,
    compute_dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    local: dict[int, torch.Tensor] = {}
    Gc = G.to(dtype=compute_dtype)
    for l in range(int(lmax) + 1):
        blk = blocks[l].to(dtype=compute_dtype)
        if l == 0:
            local[l] = blk
        elif l in (1, 2, 3):
            H, C = _get_harmonic_full_maps(l, G.device, compute_dtype)
            full = torch.matmul(blk, H)
            local[l] = _rotate_full_harmonic_fast(full, Gc, l)
            local[l] = torch.matmul(local[l], C)
        else:
            D = harmonic_row_rotation(Gc, l)
            local[l] = torch.matmul(blk, D)
    return local


def _so3_blocks_to_node_local_so2_fast(
    blocks: dict[int, torch.Tensor],
    G: torch.Tensor,
    lmax: int,
    *,
    compute_dtype: torch.dtype,
) -> dict[tuple[int, int], torch.Tensor]:
    local: dict[tuple[int, int], torch.Tensor] = {}
    device = G.device
    Gc = G.to(dtype=compute_dtype)
    for l in range(int(lmax) + 1):
        blk = blocks[l].to(dtype=compute_dtype)
        slices = so3_local_so2_basis_cpu(l)[1]
        if l == 0:
            blk_freq = blk
        else:
            Q = get_local_q(l, device, compute_dtype)
            if l in (1, 2, 3):
                H, C = _get_harmonic_full_maps(l, device, compute_dtype)
                full = torch.matmul(blk, H)
                if l == 1:
                    full_local = torch.einsum("nai,nci->nca", Gc, full)
                elif l == 2:
                    tens = full.reshape(full.shape[0], full.shape[1], 3, 3)
                    full_local = torch.einsum("nai,ncij,nbj->ncab", Gc, tens, Gc).reshape(
                        full.shape[0], full.shape[1], 9
                    )
                else:
                    tens = full.reshape(full.shape[0], full.shape[1], 3, 3, 3)
                    full_local = torch.einsum(
                        "nxi,nyj,nzk,nqijk->nqxyz",
                        Gc,
                        Gc,
                        Gc,
                        tens,
                    ).reshape(full.shape[0], full.shape[1], 27)
                blk_local = torch.matmul(full_local, C)
            else:
                D = harmonic_row_rotation(Gc, l)
                blk_local = torch.matmul(blk, D)
            blk_freq = torch.matmul(blk_local, Q)
        for m, (s, e) in enumerate(slices):
            local[(l, m)] = blk_freq[..., s:e]
    return local


def _local_so2_to_so3_local_blocks(
    local: dict[tuple[int, int], torch.Tensor],
    lmax: int,
    *,
    compute_dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    blocks: dict[int, torch.Tensor] = {}
    sample = next(iter(local.values()))
    device = sample.device
    for l in range(int(lmax) + 1):
        freq = torch.cat([local[(l, m)] for m in range(l + 1)], dim=-1).to(dtype=compute_dtype)
        if l == 0:
            blocks[l] = freq
        else:
            q = get_local_q(l, device, compute_dtype)
            blocks[l] = torch.matmul(freq, q.transpose(-1, -2))
    return blocks


def _local_so2_to_target_frame_so3_blocks(
    local: dict[tuple[int, int], torch.Tensor],
    R: torch.Tensor,
    lmax: int,
    *,
    compute_dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    blocks: dict[int, torch.Tensor] = {}
    device = R.device
    Rc = R.to(dtype=compute_dtype)
    for l in range(int(lmax) + 1):
        freq = torch.cat([local[(l, m)] for m in range(l + 1)], dim=-1).to(dtype=compute_dtype)
        if l == 0:
            blocks[l] = freq
            continue

        q = get_local_q(l, device, compute_dtype)
        blk_local = torch.matmul(freq, q.transpose(-1, -2))
        if l in (1, 2, 3):
            H, C = _get_harmonic_full_maps(l, device, compute_dtype)
            full = torch.matmul(blk_local, H)
            blocks[l] = _rotate_full_harmonic_fast(full, Rc, l)
            blocks[l] = torch.matmul(blocks[l], C)
        else:
            D = harmonic_row_rotation(Rc, l)
            blocks[l] = torch.matmul(blk_local, D)
    return blocks


def _so3_local_blocks_to_target_frame_so3_blocks(
    blocks_in: dict[int, torch.Tensor],
    R: torch.Tensor,
    lmax: int,
    *,
    compute_dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    blocks: dict[int, torch.Tensor] = {}
    device = R.device
    Rc = R.to(dtype=compute_dtype)
    for l in range(int(lmax) + 1):
        blk_local = blocks_in[l].to(dtype=compute_dtype)
        if l == 0:
            blocks[l] = blk_local
        elif l in (1, 2, 3):
            H, C = _get_harmonic_full_maps(l, device, compute_dtype)
            full = torch.matmul(blk_local, H)
            blocks[l] = _rotate_full_harmonic_fast(full, Rc, l)
            blocks[l] = torch.matmul(blocks[l], C)
        else:
            D = harmonic_row_rotation(Rc, l)
            blocks[l] = torch.matmul(blk_local, D)
    return blocks


def _so3_local_blocks_to_local_so2(
    blocks: dict[int, torch.Tensor],
    lmax: int,
    *,
    out_dtype: torch.dtype,
    compute_dtype: torch.dtype,
) -> dict[tuple[int, int], torch.Tensor]:
    local: dict[tuple[int, int], torch.Tensor] = {}
    sample = next(iter(blocks.values()))
    device = sample.device
    for l in range(int(lmax) + 1):
        block = blocks[l].to(dtype=compute_dtype)
        if l == 0:
            freq = block
        else:
            q = get_local_q(l, device, compute_dtype)
            freq = torch.matmul(block, q)
        slices = so3_local_so2_basis_cpu(l)[1]
        for m, (s, e) in enumerate(slices):
            local[(l, m)] = freq[..., s:e].to(dtype=out_dtype)
    return local


def _split_so3_flat_blocks(flat: torch.Tensor, channels: int, lmax: int) -> dict[int, torch.Tensor]:
    blocks: dict[int, torch.Tensor] = {}
    offset = 0
    for l in range(int(lmax) + 1):
        width = int(channels) * (2 * l + 1)
        blocks[l] = flat[..., offset : offset + width].reshape(flat.shape[0], int(channels), 2 * l + 1)
        offset += width
    return blocks


def _path_preserving_linear_so3_blocks(
    linear: PathPreservingLinearSO3,
    blocks: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for l in range(linear.lmax + 1):
        x_l = blocks[l]
        if x_l.shape[-2] == 0:
            out[l] = torch.zeros(
                *x_l.shape[:-2],
                linear.out_channels,
                2 * l + 1,
                dtype=x_l.dtype,
                device=x_l.device,
            )
        else:
            weight = linear.weights[str(l)].to(dtype=x_l.dtype, device=x_l.device)
            out[l] = torch.einsum("oc,ncm->nom", weight, x_l)
    return out


class _RecoupleAlias:
    def __init__(self, module: nn.Module):
        self._module = module

    @property
    def weight(self) -> torch.Tensor:
        return self._module.weight

    @property
    def internal_compute_dtype(self) -> torch.dtype | None:
        return self._module.internal_compute_dtype


class _CatEdgeWeightedPathPreservingTensorProduct(EdgeWeightedPathPreservingTensorProduct):
    def forward(
        self,
        x1: dict[int, torch.Tensor],
        x2: dict[int, torch.Tensor],
        gates: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor]:
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        compute_dtype = self.internal_compute_dtype

        if gates is not None:
            if gates.device != device or gates.dtype != dtype:
                gates = gates.to(device=device, dtype=dtype)
            if gates.shape[-1] == self.num_paths * self.channels:
                gates = gates.view(*gates.shape[:-1], self.num_paths, self.channels)
            elif gates.shape[-2:] != (self.num_paths, self.channels):
                raise ValueError(
                    f"Expected gates shape (..., {self.num_paths * self.channels}) or "
                    f"(..., {self.num_paths}, {self.channels}), got {tuple(gates.shape)}"
                )

        chunks: dict[int, list[torch.Tensor | None]] = {
            l: [None] * int(self.path_counts_by_l.get(l, 0))
            for l in range(self.lmax + 1)
        }
        proj_list = self._get_proj_group_list(device=device, dtype=dtype)
        w = self.weight.to(device=device, dtype=compute_dtype)

        for g_idx, group in enumerate(self._groups):
            l1 = int(group["l1"])
            l2 = int(group["l2"])
            a = x1.get(l1)
            b = x2.get(l2)
            if a is None or b is None:
                continue

            a_comp = a.to(dtype=compute_dtype) if a.dtype != compute_dtype else a
            b_comp = b.to(dtype=compute_dtype) if b.dtype != compute_dtype else b
            m1 = 2 * l1 + 1
            m2 = 2 * l2 + 1
            pair = (a_comp.unsqueeze(-1) * b_comp.unsqueeze(-2)).reshape(
                *batch_shape,
                self.channels,
                m1 * m2,
            )
            y = torch.matmul(pair, proj_list[g_idx])

            for p_idx, l3, s, e in group["segments"]:
                p_idx = int(p_idx)
                l3 = int(l3)
                seg = y[..., int(s) : int(e)]
                seg = seg * w[p_idx].view(*([1] * len(batch_shape)), self.channels, 1)
                if gates is not None:
                    seg = seg * gates[..., p_idx, :].unsqueeze(-1)
                if seg.dtype != dtype:
                    seg = seg.to(dtype=dtype)
                chunks[l3][int(self.path_offset[p_idx])] = seg

        out: dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            pieces = chunks[l]
            if pieces:
                zero = None
                filled: list[torch.Tensor] = []
                for piece in pieces:
                    if piece is None:
                        if zero is None:
                            zero = torch.zeros(
                                *batch_shape,
                                self.channels,
                                2 * l + 1,
                                device=device,
                                dtype=dtype,
                            )
                        filled.append(zero)
                    else:
                        filled.append(piece)
                out[l] = torch.cat(filled, dim=-2)
            else:
                out[l] = torch.zeros(
                    *batch_shape,
                    0,
                    2 * l + 1,
                    device=device,
                    dtype=dtype,
                )
        return out


class NodeLocalSO2ChannelMixer(nn.Module):
    """Per-l channel mixing, matching SO3's per-l EquivariantChannelLinearSO3Rect pattern.

    All m components within the same l share the same channel linear, exactly as
    SO3 uses one Linear per l for all (2l+1) spatial components."""

    def __init__(self, in_channels: int, out_channels: int, lmax: int, bias: bool = False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.lmax = int(lmax)
        self.adapters = nn.ModuleDict(
            {str(l): nn.Linear(self.in_channels, self.out_channels, bias=bias) for l in range(self.lmax + 1)}
        )
        self._l_slices = [(l * l, (l + 1) * (l + 1)) for l in range(self.lmax + 1)]

    def forward_flat(self, flat: torch.Tensor) -> torch.Tensor:
        out_parts = []
        for l, (start, end) in enumerate(self._l_slices):
            block = flat[..., start:end]
            out_parts.append(apply_channel_linear(block, self.adapters[str(l)]))
        return torch.cat(out_parts, dim=-1)

    def forward(self, local: dict[tuple[int, int], torch.Tensor]) -> dict[tuple[int, int], torch.Tensor]:
        return _flat_to_local_so2(
            self.forward_flat(_local_to_flat_so2(local, self.lmax)),
            self.out_channels,
            self.lmax,
        )


class LocalPerLScaleSO2(nn.Module):
    def __init__(self, channels: int, lmax: int, init_scales: list[float] | tuple[float, ...]):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        if len(init_scales) != self.lmax + 1:
            raise ValueError(f"Expected {self.lmax + 1} init scales, got {len(init_scales)}")
        scales = torch.as_tensor(init_scales, dtype=torch.get_default_dtype()).clamp_min(1e-6)
        self.log_scale = nn.Parameter(scales.log())

    def forward(self, local: dict[tuple[int, int], torch.Tensor]) -> dict[tuple[int, int], torch.Tensor]:
        scales = self.log_scale.exp().to(dtype=next(iter(local.values())).dtype, device=next(iter(local.values())).device)
        return {(l, m): block * scales[l] for (l, m), block in local.items()}

    def forward_flat(self, flat: torch.Tensor) -> torch.Tensor:
        scales = self.log_scale.exp().to(dtype=flat.dtype, device=flat.device)
        pieces = [
            flat[..., l * l : (l + 1) * (l + 1)] * scales[l]
            for l in range(self.lmax + 1)
        ]
        return torch.cat(pieces, dim=-1)


class ElementConditionedLocalLinearSO2(nn.Module):
    def __init__(self, num_elements: int, channels: int, lmax: int, bias: bool = False):
        super().__init__()
        self.num_elements = int(num_elements)
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.weights = nn.ParameterDict(
            {
                str(l): nn.Parameter(torch.randn(self.num_elements, self.channels, self.channels) * 0.02)
                for l in range(self.lmax + 1)
            }
        )
        if bias:
            self.bias = nn.ParameterDict(
                {
                    str(l): nn.Parameter(torch.zeros(self.num_elements, self.channels))
                    for l in range(self.lmax + 1)
                }
            )
        else:
            self.bias = None

    def forward_flat(self, flat: torch.Tensor, atom_types: torch.Tensor) -> torch.Tensor:
        idx = atom_types.long().clamp(min=0, max=self.num_elements - 1)
        out_parts = []
        for l in range(self.lmax + 1):
            start = l * l
            end = (l + 1) * (l + 1)
            block = flat[..., start:end]
            weight = self.weights[str(l)][idx].to(dtype=block.dtype)
            out_block = torch.einsum("noi,nid->nod", weight, block)
            if self.bias is not None:
                out_block = out_block + self.bias[str(l)][idx].to(dtype=block.dtype).unsqueeze(-1)
            out_parts.append(out_block)
        return torch.cat(out_parts, dim=-1)

    def forward(
        self,
        local: dict[tuple[int, int], torch.Tensor],
        atom_types: torch.Tensor,
    ) -> dict[tuple[int, int], torch.Tensor]:
        return _flat_to_local_so2(
            self.forward_flat(_local_to_flat_so2(local, self.lmax), atom_types),
            self.channels,
            self.lmax,
        )


def _pad_so2_frequency_dim2(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 2:
        return x
    if x.shape[-1] != 1:
        raise ValueError(f"Expected SO2 frequency dim 1 or 2, got {x.shape[-1]}")
    return torch.nn.functional.pad(x, (0, 1))


def _build_scalar_source_feature_moment(
    scalar_feats: torch.Tensor,
    *,
    edge_radial: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_count: torch.Tensor,
) -> torch.Tensor:
    dtype = scalar_feats.dtype
    num_nodes = scalar_feats.shape[0]
    mask = edge_mask.to(dtype=dtype).view(-1, 1, 1, 1)
    weighted = (
        scalar_feats[edge_src][:, :, None, None]
        * edge_radial[:, None, :, None].to(dtype=dtype)
    )
    return scatter(
        weighted * mask,
        edge_dst,
        dim=0,
        dim_size=num_nodes,
        reduce="sum",
    ) / neighbor_count.to(dtype=dtype).sqrt().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def _build_source_feature_moments_flat_so2(
    local: dict[tuple[int, int], torch.Tensor],
    *,
    edge_radial: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_count: torch.Tensor,
    lmax: int,
    G_node: torch.Tensor | None = None,
    compute_dtype: torch.dtype | None = None,
) -> dict[tuple[int, int], torch.Tensor]:
    from molecular_force_field.models.ictd_irreps_2d import _so2_flat_index_cpu

    sample = next(iter(local.values()))
    num_nodes = sample.shape[0]
    dtype = sample.dtype
    if G_node is not None and int(lmax) > 0:
        edge_local = {key: value[edge_src] for key, value in local.items()}
        R_src_to_dst = torch.matmul(
            G_node[edge_dst],
            G_node[edge_src].transpose(-1, -2),
        )
        edge_local = node_local_so2_change_frame(
            edge_local,
            R_src_to_dst,
            int(lmax),
            compute_dtype=dtype if compute_dtype is None else compute_dtype,
        )
        edge_flat = _local_to_flat_so2(edge_local, int(lmax)).to(dtype=dtype)
    else:
        edge_flat = _local_to_flat_so2(local, int(lmax))[edge_src]
    weighted = edge_flat.unsqueeze(-2) * edge_radial[:, None, :, None].to(dtype=dtype)
    weighted = weighted * edge_mask.to(dtype=dtype).view(-1, 1, 1, 1)
    source_flat = scatter(
        weighted,
        edge_dst,
        dim=0,
        dim_size=num_nodes,
        reduce="sum",
    ) / neighbor_count.to(dtype=dtype).sqrt().view(-1, 1, 1, 1)

    starts, dims, _ = _so2_flat_index_cpu(int(lmax))
    return {
        (l, m): source_flat[..., start : start + dims[(l, m)]]
        for (l, m), start in starts.items()
    }


class _FastSO2PathWeightedTP(nn.Module):
    """Path-weighted SO2 TP with one batched einsum per output frequency.

    This matches ``HarmonicPathWeightedTensorProductSO2``'s path ordering and
    weights, but removes the hot per-path einsum loop from the interaction
    recoupling path.
    """

    def __init__(
        self,
        channels: int,
        mmax: int,
        *,
        allowed_paths: list[tuple[int, int, int]] | None = None,
        path_policy: str = "full",
        max_rank_other: int | None = None,
        normalization: str = "component",
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        from molecular_force_field.models.ictd_irreps_2d import (
            _normalize_cg_tensor,
            _so2_valid_outputs,
            build_cg_tensor_so2,
            so2_irrep_dim,
        )

        self.channels = int(channels)
        self.mmax = int(mmax)
        self._normalization = normalization
        self.internal_compute_dtype = internal_compute_dtype

        all_paths: list[tuple[int, int, int]] = []
        for m1 in range(self.mmax + 1):
            for m2 in range(self.mmax + 1):
                for m3 in _so2_valid_outputs(m1, m2, mmax=self.mmax):
                    all_paths.append((m1, m2, m3))
        if allowed_paths is not None:
            allowed_set = {tuple(map(int, p)) for p in allowed_paths}
            self.paths = [p for p in all_paths if p in allowed_set]
        elif path_policy == "full":
            self.paths = all_paths
        elif path_policy == "max_rank_other":
            if max_rank_other is None:
                raise ValueError("path_policy='max_rank_other' requires max_rank_other")
            self.paths = [p for p in all_paths if min(p[0], p[1]) <= int(max_rank_other)]
        else:
            raise ValueError(f"Unknown path_policy={path_policy!r}")

        self.num_paths = len(self.paths)
        self.weight = nn.Parameter(torch.randn(self.num_paths, self.channels) * 0.02)
        self._groups: dict[int, list[tuple[int, int, int]]] = {}
        for p_idx, (m1, m2, m3) in enumerate(self.paths):
            self._groups.setdefault(m3, []).append((p_idx, m1, m2))
        self._scalar_left_diagonal = (
            len(self.paths) == self.mmax + 1
            and all((m1 == 0 and m2 == m3) for m1, m2, m3 in self.paths)
        )
        for m3 in range(self.mmax + 1):
            cgs = []
            group = self._groups.get(m3, [])
            if group:
                self.register_buffer(
                    f"_path_group_{m3}",
                    torch.tensor([p_idx for p_idx, _, _ in group], dtype=torch.long),
                    persistent=False,
                )
                self.register_buffer(
                    f"_m1_group_{m3}",
                    torch.tensor([m1 for _, m1, _ in group], dtype=torch.long),
                    persistent=False,
                )
                self.register_buffer(
                    f"_m2_group_{m3}",
                    torch.tensor([m2 for _, _, m2 in group], dtype=torch.long),
                    persistent=False,
                )
            for _, m1, m2 in self._groups.get(m3, []):
                C = build_cg_tensor_so2(m1, m2, m3)
                C = _normalize_cg_tensor(C, so2_irrep_dim(m3), normalization)
                padded = torch.zeros(2, 2, so2_irrep_dim(m3), dtype=torch.get_default_dtype())
                padded[: C.shape[0], : C.shape[1], : C.shape[2]] = C.to(dtype=torch.get_default_dtype())
                cgs.append(padded)
            if cgs:
                self.register_buffer(f"_cg_group_{m3}", torch.stack(cgs, dim=0), persistent=False)

    def forward(self, x1: dict[int, torch.Tensor], x2: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        compute_dtype = dtype if self.internal_compute_dtype is None else self.internal_compute_dtype

        if self._scalar_left_diagonal:
            scalar = x1[0].to(dtype=compute_dtype)
            out: dict[int, torch.Tensor] = {}
            for m3 in range(self.mmax + 1):
                group = self._groups.get(m3, [])
                d3 = 1 if m3 == 0 else 2
                if not group:
                    out[m3] = torch.zeros(*batch_shape, self.channels, d3, device=device, dtype=dtype)
                    continue
                path_idx, _, m2 = group[0]
                b = _pad_so2_frequency_dim2(x2[m2].to(dtype=compute_dtype))
                cg = getattr(self, f"_cg_group_{m3}")[0, 0].to(device=device, dtype=compute_dtype)
                w = self.weight[path_idx].to(device=device, dtype=compute_dtype)
                y = torch.matmul(b, cg) * scalar * w.view(*([1] * len(batch_shape)), self.channels, 1)
                out[m3] = y.to(dtype=dtype)
            return out

        def _packed_block(src: dict[int, torch.Tensor], m: int) -> torch.Tensor:
            if m in src:
                return _pad_so2_frequency_dim2(src[m].to(dtype=compute_dtype))
            return torch.zeros(*batch_shape, self.channels, 2, device=device, dtype=compute_dtype)

        x1_packed = torch.stack(
            [_packed_block(x1, m) for m in range(self.mmax + 1)],
            dim=0,
        )
        x2_packed = torch.stack(
            [_packed_block(x2, m) for m in range(self.mmax + 1)],
            dim=0,
        )
        out: dict[int, torch.Tensor] = {}
        for m3 in range(self.mmax + 1):
            group = self._groups.get(m3, [])
            d3 = 1 if m3 == 0 else 2
            if not group:
                out[m3] = torch.zeros(*batch_shape, self.channels, d3, device=device, dtype=dtype)
                continue
            m1_idx = getattr(self, f"_m1_group_{m3}").to(device=device)
            m2_idx = getattr(self, f"_m2_group_{m3}").to(device=device)
            path_idx = getattr(self, f"_path_group_{m3}").to(device=device)
            a = x1_packed.index_select(0, m1_idx)
            b = x2_packed.index_select(0, m2_idx)
            cg = getattr(self, f"_cg_group_{m3}").to(device=device, dtype=compute_dtype)
            w = self.weight.index_select(0, path_idx).to(device=device, dtype=compute_dtype)
            y = torch.einsum("p...ca,p...cb,pabk,pc->...ck", a, b, cg, w)
            out[m3] = y.to(dtype=dtype)
        return out


class NodeLocalL0ReadoutSO2(nn.Module):
    """Reads only (l=0,m=0) scalar, matching SO3's EquivariantScalarReadoutSO3."""

    def __init__(self, channels: int, output_init_std: float = 0.003):
        super().__init__()
        self.channels = int(channels)
        self.readout = nn.Linear(self.channels, 1, bias=True)
        nn.init.normal_(self.readout.weight, mean=0.0, std=float(output_init_std))
        nn.init.zeros_(self.readout.bias)

    def forward(self, local: dict[tuple[int, int], torch.Tensor]) -> torch.Tensor:
        return self.readout(local[(0, 0)].squeeze(-1))


class _FixedSO2FusionProduct5Adapter(nn.Module):
    def __init__(self, segments: int, channels: int):
        super().__init__()
        self.segments = int(segments)
        self.channels = int(channels)
        self.scale = 1.0 / math.sqrt(float(max(self.segments, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.segments == 1:
            return x
        return x.reshape(*x.shape[:-1], self.segments, self.channels).sum(dim=-2) * self.scale


def _make_fusion_product5_adapters(channels: int, lmax: int) -> nn.ModuleDict:
    channels = int(channels)
    lmax = int(lmax)
    shared_adapt = nn.ModuleDict()
    for m in range(lmax + 1):
        in_ch = (lmax - m + 1) * channels
        segments = lmax - m + 1
        if in_ch == channels:
            shared_adapt[str(m)] = nn.Identity()
        else:
            shared_adapt[str(m)] = _FixedSO2FusionProduct5Adapter(segments, channels)
    return shared_adapt


class FusionProduct5ReadoutSO2(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        lmax: int,
        feature_channels: list[int] | tuple[int, ...],
        hidden_sizes: list[int],
        internal_compute_dtype: torch.dtype | None = None,
        output_init_std: float = 0.003,
        shared_adapt: nn.ModuleDict | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.feature_channels = [int(c) for c in feature_channels]
        self.num_feature_blocks = len(self.feature_channels)
        self.product5_muls_by_m = {m: self.channels for m in range(self.lmax + 1)}
        self._p5_base_mul = self.product5_muls_by_m[0]
        self._p5_adapt = (
            _make_fusion_product5_adapters(self.channels, self.lmax)
            if shared_adapt is None
            else shared_adapt
        )
        self.num_feature_blocks = len(self.feature_channels)

        self.product_5 = HarmonicElementwiseProductSO2(
            mmax=self.lmax,
            mul=self.num_feature_blocks * self._p5_base_mul,
            irreps_out="0",
            internal_compute_dtype=internal_compute_dtype,
        )
        sum_mul = sum(self.product5_muls_by_m[m] for m in range(self.lmax + 1))
        self.proj_total = MainNet(self.num_feature_blocks * sum_mul, hidden_sizes, 1, output_init_std=output_init_std)

    def product5_features(self, local_states: list[dict[tuple[int, int], torch.Tensor]]) -> torch.Tensor:
        T_blocks: dict[int, torch.Tensor] = {}
        adapted_features = [
            _adapt_local_state_for_product5(local, self._p5_adapt, self.lmax)
            for local in local_states
        ]
        for m in range(self.lmax + 1):
            T_blocks[m] = torch.cat([adapted[m] for adapted in adapted_features], dim=-2)
        proj_dtype = next(self.proj_total.parameters()).dtype
        return self.product_5(T_blocks, T_blocks).to(dtype=proj_dtype)

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj_total(features)

    def forward(self, local_states: list[dict[tuple[int, int], torch.Tensor]]) -> torch.Tensor:
        return self.project_features(self.product5_features(local_states))


def _project_fusion_features_multihead_so2(
    readouts: nn.ModuleList,
    features: torch.Tensor,
) -> torch.Tensor:
    nets = [readout.proj_total for readout in readouts]
    if not nets:
        raise ValueError("Expected at least one fusion readout")
    if any(len(net.layers) != len(nets[0].layers) for net in nets):
        return torch.stack([readout.project_features(features) for readout in readouts], dim=0)

    x_base = features
    input_norm = nets[0].input_norm
    mean = x_base.mean(dim=-1, keepdim=True)
    var = x_base.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x_base - mean) * torch.rsqrt(var + input_norm.eps)
    input_weight = torch.stack([net.input_norm.weight for net in nets], dim=0).to(dtype=x_base.dtype)
    input_bias = torch.stack([net.input_norm.bias for net in nets], dim=0).to(dtype=x_base.dtype)
    x = x_norm.unsqueeze(0) * input_weight[:, None, :] + input_bias[:, None, :]

    num_layers = len(nets[0].layers)
    for layer_idx in range(num_layers):
        weight = torch.stack([net.layers[layer_idx].weight for net in nets], dim=0).to(dtype=x.dtype)
        bias = torch.stack([net.layers[layer_idx].bias for net in nets], dim=0).to(dtype=x.dtype)
        x = torch.einsum("hni,hoi->hno", x, weight) + bias[:, None, :]

        ln0 = nets[0].layer_norms[layer_idx]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + ln0.eps)
        ln_weight = torch.stack([net.layer_norms[layer_idx].weight for net in nets], dim=0).to(dtype=x.dtype)
        ln_bias = torch.stack([net.layer_norms[layer_idx].bias for net in nets], dim=0).to(dtype=x.dtype)
        x = x * ln_weight[:, None, :] + ln_bias[:, None, :]
        x = torch.nn.functional.silu(x)

    weight = torch.stack([net.output.weight for net in nets], dim=0).to(dtype=x.dtype)
    bias = torch.stack([net.output.bias for net in nets], dim=0).to(dtype=x.dtype)
    return torch.einsum("hni,hoi->hno", x, weight) + bias[:, None, :]


class SO2InitialInteraction(nn.Module):
    """Scalar → SO2 initial interaction, semantically interaction[0] in the pipeline.

    Takes scalar features (N, C) and edge geometry, produces a full Dict[(l,m)]
    of SO2 local features.  Equivalent to SO3's first interaction that maps the
    scalar node embedding to the full lmax ICTD space."""

    def __init__(
        self,
        *,
        channels: int,
        lmax: int,
        number_of_basis: int,
        num_elements: int,
        interaction_compute_dtype: torch.dtype | None = None,
        message_scale_init: list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.number_of_basis = int(number_of_basis)
        self.num_elements = int(num_elements)
        self.message_selector = ElementConditionedLocalLinearSO2(
            self.num_elements,
            self.channels,
            self.lmax,
            bias=False,
        )
        _init_element_conditioned_local_identity_(self.message_selector)
        self.message_output_scale = (
            LocalPerLScaleSO2(self.channels, self.lmax, message_scale_init)
            if message_scale_init is not None
            else nn.Identity()
        )
        self.linear_up_exact = nn.Linear(self.channels, self.channels, bias=False)
        with torch.no_grad():
            self.linear_up_exact.weight.zero_()
            self.linear_up_exact.weight.diagonal().fill_(1.0)
        self.tp_exact = _CatEdgeWeightedPathPreservingTensorProduct(
            channels=self.channels,
            lmax=self.lmax,
            allowed_paths=[(0, l, l) for l in range(self.lmax + 1)],
            internal_compute_dtype=interaction_compute_dtype,
        )
        _init_so2_path_tp_weight_to_one_(self.tp_exact)
        self.recouple = _RecoupleAlias(self.tp_exact)
        self.fc_exact = nn.Sequential(
            nn.Linear(self.number_of_basis, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp_exact.num_paths * self.channels),
        )
        self.message_linear_exact = PathPreservingLinearSO3(
            {
                l: self.channels * int(self.tp_exact.path_counts_by_l.get(l, 0))
                for l in range(self.lmax + 1)
            },
            self.channels,
            self.lmax,
        )

    def forward(
        self,
        *,
        scalar_feats: torch.Tensor,
        atom_types: torch.Tensor,
        edge_radial: torch.Tensor,
        node_moments: dict[tuple[int, int], torch.Tensor],
        node_moments_so3: dict[int, torch.Tensor] | None = None,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_mask: torch.Tensor,
        neighbor_count: torch.Tensor,
        avg_num_neighbors: float = 1.0,
    ) -> dict[tuple[int, int], torch.Tensor]:
        dtype = scalar_feats.dtype
        del neighbor_count
        source = self.linear_up_exact(scalar_feats)[edge_src].unsqueeze(-1)
        geom_blocks = (
            node_moments_so3
            if node_moments_so3 is not None
            else _local_so2_to_so3_local_blocks(
                node_moments,
                self.lmax,
                compute_dtype=dtype,
            )
        )
        gates = self.fc_exact(edge_radial.to(dtype=dtype))
        edge_blocks = self.tp_exact({0: source}, geom_blocks, gates)
        edge_message_blocks = _path_preserving_linear_so3_blocks(
            self.message_linear_exact,
            edge_blocks,
        )
        mask = edge_mask.to(dtype=dtype).view(-1, 1, 1)
        inv_avg = 1.0 / max(float(avg_num_neighbors), 1e-8)
        message_so3_blocks = {
            l: scatter(
                edge_message_blocks[l] * mask,
                edge_dst,
                dim=0,
                dim_size=scalar_feats.shape[0],
                reduce="sum",
            )
            * inv_avg
            for l in range(self.lmax + 1)
        }
        message = _so3_local_blocks_to_local_so2(
            message_so3_blocks,
            self.lmax,
            out_dtype=dtype,
            compute_dtype=dtype,
        )
        message_flat = self.message_selector.forward_flat(
            _local_to_flat_so2(message, self.lmax),
            atom_types,
        )
        if not isinstance(self.message_output_scale, nn.Identity):
            message_flat = self.message_output_scale.forward_flat(message_flat)
        return _flat_to_local_so2(message_flat, self.channels, self.lmax)


class SO2LocalInteractionBlock(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        lmax: int,
        number_of_basis: int,
        max_radius: float,
        function_type: str,
        num_elements: int,
        use_self_connection: bool = True,
        interaction_compute_dtype: torch.dtype | None = None,
        message_scale_init: list[float] | tuple[float, ...] | None = None,
        sc_scale_init: list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        del max_radius, function_type
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.number_of_basis = int(number_of_basis)
        self.use_self_connection = bool(use_self_connection)
        self.message_output_scale = (
            LocalPerLScaleSO2(self.channels, self.lmax, message_scale_init)
            if message_scale_init is not None
            else nn.Identity()
        )
        self.self_connection = (
            ElementConditionedLocalLinearSO2(num_elements, self.channels, self.lmax, bias=False)
            if self.use_self_connection
            else None
        )
        _init_element_conditioned_local_identity_(self.self_connection)
        self.sc_output_scale = (
            LocalPerLScaleSO2(self.channels, self.lmax, sc_scale_init)
            if sc_scale_init is not None
            else nn.Identity()
        )
        self.linear_up_exact = NodeLocalSO2ChannelMixer(self.channels, self.channels, self.lmax)
        _init_node_local_mixer_identity_(self.linear_up_exact)
        self.tp_exact = _CatEdgeWeightedPathPreservingTensorProduct(
            channels=self.channels,
            lmax=self.lmax,
            internal_compute_dtype=interaction_compute_dtype,
        )
        _init_so2_path_tp_weight_to_one_(self.tp_exact)
        self.recouple = _RecoupleAlias(self.tp_exact)
        self.fc_exact = nn.Sequential(
            nn.Linear(self.number_of_basis, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp_exact.num_paths * self.channels),
        )
        self.message_linear_exact = PathPreservingLinearSO3(
            {
                l: self.channels * int(self.tp_exact.path_counts_by_l.get(l, 0))
                for l in range(self.lmax + 1)
            },
            self.channels,
            self.lmax,
        )

    def forward(
        self,
        *,
        atom_types: torch.Tensor,
        local: dict[tuple[int, int], torch.Tensor],
        edge_radial: torch.Tensor,
        node_moments: dict[tuple[int, int], torch.Tensor],
        node_moments_so3: dict[int, torch.Tensor] | None = None,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_mask: torch.Tensor,
        neighbor_count: torch.Tensor,
        G_node: torch.Tensor | None = None,
        avg_num_neighbors: float = 1.0,
    ) -> tuple[dict[tuple[int, int], torch.Tensor], dict[tuple[int, int], torch.Tensor] | None]:
        dtype = next(iter(local.values())).dtype
        del neighbor_count
        local_for_sc = local
        local_up = self.linear_up_exact(local)
        source_node_blocks = _local_so2_to_so3_local_blocks(
            local_up,
            self.lmax,
            compute_dtype=dtype,
        )
        source_blocks = {
            l: source_node_blocks[l][edge_src]
            for l in range(self.lmax + 1)
        }
        if G_node is not None and self.lmax > 0:
            R_src_to_dst = torch.matmul(
                G_node[edge_dst],
                G_node[edge_src].transpose(-1, -2),
            )
            source_blocks = _so3_local_blocks_to_target_frame_so3_blocks(
                source_blocks,
                R_src_to_dst,
                self.lmax,
                compute_dtype=dtype if self.recouple.internal_compute_dtype is None else self.recouple.internal_compute_dtype,
            )
        geom_blocks = (
            node_moments_so3
            if node_moments_so3 is not None
            else _local_so2_to_so3_local_blocks(
                node_moments,
                self.lmax,
                compute_dtype=dtype,
            )
        )
        gates = self.fc_exact(edge_radial.to(dtype=dtype))
        edge_blocks = self.tp_exact(source_blocks, geom_blocks, gates)
        edge_message_blocks = _path_preserving_linear_so3_blocks(
            self.message_linear_exact,
            edge_blocks,
        )
        mask = edge_mask.to(dtype=dtype).view(-1, 1, 1)
        num_nodes = next(iter(local.values())).shape[0]
        inv_avg = 1.0 / max(float(avg_num_neighbors), 1e-8)
        message_so3_blocks = {
            l: scatter(edge_message_blocks[l] * mask, edge_dst, dim=0, dim_size=num_nodes, reduce="sum")
            * inv_avg
            for l in range(self.lmax + 1)
        }
        message = _so3_local_blocks_to_local_so2(
            message_so3_blocks,
            self.lmax,
            out_dtype=dtype,
            compute_dtype=dtype,
        )
        message_flat = _local_to_flat_so2(message, self.lmax)
        if not isinstance(self.message_output_scale, nn.Identity):
            message_flat = self.message_output_scale.forward_flat(message_flat)
        message = _flat_to_local_so2(message_flat, self.channels, self.lmax)
        sc = None
        if self.self_connection is not None:
            sc_flat = self.self_connection.forward_flat(
                _local_to_flat_so2(local_for_sc, self.lmax),
                atom_types,
            )
            if not isinstance(self.sc_output_scale, nn.Identity):
                sc_flat = self.sc_output_scale.forward_flat(sc_flat)
            sc = _flat_to_local_so2(sc_flat, self.channels, self.lmax)
        return message, sc


_CONTRACTION_BATCH_EXAMPLE = 10
_CONTRACTION_ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]


def _local_to_flat_so2(
    local: dict[tuple[int, int], torch.Tensor],
    lmax: int,
) -> torch.Tensor:
    """Flatten Dict[(l,m)] → (N, C, D) where D = (lmax+1)²."""
    from molecular_force_field.models.ictd_irreps_2d import _so2_flat_index_cpu

    _, dims, D = _so2_flat_index_cpu(lmax)
    sample = next(iter(local.values()))
    batch_shape = sample.shape[:-2]
    C = sample.shape[-2]
    flat = torch.zeros(*batch_shape, C, D, device=sample.device, dtype=sample.dtype)
    starts, _, _ = _so2_flat_index_cpu(lmax)
    for (l, m), blk in local.items():
        start = starts[(l, m)]
        d = dims[(l, m)]
        flat[:, :, start : start + d] = blk.reshape(*batch_shape, C, d)
    return flat


def _flat_to_local_so2(
    flat: torch.Tensor,
    channels: int,
    lmax: int,
) -> dict[tuple[int, int], torch.Tensor]:
    """Convert (N, C, D) back to Dict[(l,m)]."""
    from molecular_force_field.models.ictd_irreps_2d import _so2_flat_index_cpu

    starts, dims, D = _so2_flat_index_cpu(lmax)
    out: dict[tuple[int, int], torch.Tensor] = {}
    for (l, m), start in starts.items():
        d = dims[(l, m)]
        out[(l, m)] = flat[:, :, start : start + d].reshape(
            *flat.shape[:-2], channels, d
        )
    return out


def _local_so2_full_change_of_basis(lmax: int, dtype: torch.dtype) -> torch.Tensor:
    """Block-diagonal SO3-harmonic -> local SO2-frequency basis matrix."""
    return torch.block_diag(
        *[so3_local_so2_basis_cpu(l)[0].to(dtype=dtype) for l in range(int(lmax) + 1)]
    ).contiguous()


def _transform_u_tensor_axis_to_local(u: torch.Tensor, q: torch.Tensor, axis: int) -> torch.Tensor:
    """Apply x_so3 = x_local @ q.T to one tensor axis."""
    axis = int(axis)
    out = torch.tensordot(u, q, dims=([axis], [0]))
    last = out.ndim - 1
    perm = list(range(out.ndim))
    perm = perm[:axis] + [last] + perm[axis:last]
    return out.permute(perm).contiguous()


def _strict_so3_u_tensor_in_local_so2_basis(
    *,
    lmax: int,
    output_l: int,
    correlation: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """SO3 ictd-pure-u tensor expressed in the node-local SO2 basis.

    The node-local SO2 state is a change of basis of each SO3 l-block:
    x_local = x_so3 @ Q_l.  Strict ictd-pure-u alignment means transforming the
    SO3 U matrix by the same block-diagonal Q on every input leg and by Q_l on
    the output leg, keeping the SO3 CG coefficients exactly.
    """
    from molecular_force_field.models.ictd_irreps import ictd_u_matrix_so3

    lmax = int(lmax)
    output_l = int(output_l)
    correlation = int(correlation)
    u = ictd_u_matrix_so3(
        lmax=lmax,
        output_l=output_l,
        correlation=correlation,
        irrep_normalization="component",
        dtype=dtype,
    ).contiguous()

    q_all = _local_so2_full_change_of_basis(lmax, dtype=dtype)
    if output_l == 0:
        out = u
        input_axis0 = 0
    else:
        q_out = so3_local_so2_basis_cpu(output_l)[0].to(dtype=dtype)
        out = torch.tensordot(q_out.transpose(0, 1), u, dims=([1], [0])).contiguous()
        input_axis0 = 1

    for axis in range(input_axis0, input_axis0 + correlation):
        out = _transform_u_tensor_axis_to_local(out, q_all, axis)
    return out.contiguous()


class SO2PureUContraction(nn.Module):
    """SO2 equivalent of _ICTDPureUContraction — opt_einsum_fx-optimized einsum graphs."""

    def __init__(
        self,
        *,
        u_tensors: dict[int, torch.Tensor],
        output_l: int,
        num_elements: int,
        num_features: int,
        lmax: int,
    ):
        import opt_einsum_fx

        super().__init__()
        self.output_l = int(output_l)
        self.correlation = int(max(u_tensors))
        self.num_elements = int(num_elements)
        self.num_features = int(num_features)
        for nu in range(1, self.correlation + 1):
            self.register_buffer(f"U_matrix_{nu}", u_tensors[nu])

        self.contractions_weighting = nn.ModuleList()
        self.contractions_features = nn.ModuleList()
        self.weights = nn.ParameterList([])

        D = (int(lmax) + 1) ** 2

        for i in range(self.correlation, 0, -1):
            num_params = self.U_tensors(i).size()[-1]
            num_equiv = 2 * self.output_l + 1
            num_ell = self.U_tensors(i).size()[-2]

            if i == self.correlation:
                num_abc = i + min(self.output_l, 1) - 1
                left_chars = [_CONTRACTION_ALPHABET[j] for j in range(max(num_abc, 0))]
                subscript = "".join(
                    left_chars + ["ik,ekc,bci,be -> bc"] + left_chars
                )
                graph_mod = torch.fx.symbolic_trace(
                    lambda x, y, w, z: torch.einsum(subscript, x, y, w, z)
                )
                self.graph_opt_main = opt_einsum_fx.optimize_einsums_full(
                    model=graph_mod,
                    example_inputs=(
                        torch.randn([num_equiv] + [num_ell] * i + [num_params]).squeeze(0),
                        torch.randn((self.num_elements, num_params, self.num_features)),
                        torch.randn((_CONTRACTION_BATCH_EXAMPLE, self.num_features, num_ell)),
                        torch.randn((_CONTRACTION_BATCH_EXAMPLE, self.num_elements)),
                    ),
                )
                self.weights_max = nn.Parameter(
                    torch.randn((self.num_elements, num_params, self.num_features))
                    / max(num_params, 1)
                )
            else:
                num_w = i + min(self.output_l, 1)
                num_f = i - 1 + min(self.output_l, 1)
                left_w = [_CONTRACTION_ALPHABET[j] for j in range(max(num_w, 0))]
                left_f = [_CONTRACTION_ALPHABET[j] for j in range(max(num_f, 0))]
                sub_w = "".join(left_w + ["k,ekc,be->bc"] + left_w)
                sub_f = "".join(["bc"] + left_f + ["i,bci->bc"] + left_f)

                gw = torch.fx.symbolic_trace(lambda x, y, z: torch.einsum(sub_w, x, y, z))
                gf = torch.fx.symbolic_trace(lambda x, y: torch.einsum(sub_f, x, y))
                self.contractions_weighting.append(
                    opt_einsum_fx.optimize_einsums_full(
                        model=gw,
                        example_inputs=(
                            torch.randn([num_equiv] + [num_ell] * i + [num_params]).squeeze(0),
                            torch.randn((self.num_elements, num_params, self.num_features)),
                            torch.randn((_CONTRACTION_BATCH_EXAMPLE, self.num_elements)),
                        ),
                    )
                )
                self.contractions_features.append(
                    opt_einsum_fx.optimize_einsums_full(
                        model=gf,
                        example_inputs=(
                            torch.randn(
                                [_CONTRACTION_BATCH_EXAMPLE, self.num_features, num_equiv]
                                + [num_ell] * i
                            ).squeeze(2),
                            torch.randn((_CONTRACTION_BATCH_EXAMPLE, self.num_features, num_ell)),
                        ),
                    )
                )
                self.weights.append(
                    nn.Parameter(
                        torch.randn((self.num_elements, num_params, self.num_features))
                        / max(num_params, 1)
                    )
                )

    def U_tensors(self, nu: int) -> torch.Tensor:
        return dict(self.named_buffers())[f"U_matrix_{int(nu)}"]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.graph_opt_main(
            self.U_tensors(self.correlation), self.weights_max, x, y
        )
        for i, (weight, cw, cf) in enumerate(
            zip(self.weights, self.contractions_weighting, self.contractions_features)
        ):
            c_tensor = cw(
                self.U_tensors(self.correlation - i - 1), weight, y
            )
            c_tensor = c_tensor + out
            out = cf(c_tensor, x)
        return out.view(out.shape[0], -1)


class SO2PureUSymmetricContraction(nn.Module):
    """SO2 equivalent of ICTDPureUSymmetricContractionSO3 — per-l contractions.

    If ``shared_u`` is provided, reuses pre-computed U tensors (avoids per-block
    buffer duplication).  ``shared_u`` is a dict mapping ``(output_l, nu)`` to
    the corresponding buffer tensor."""

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_lmax: int,
        correlation: int = 3,
        shared_u: dict[tuple[int, int], torch.Tensor] | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.target_lmax = int(target_lmax)
        self.contractions = nn.ModuleList()
        dtype = torch.get_default_dtype()
        for l in range(self.target_lmax + 1):
            u_tensors = {}
            for nu in range(1, int(correlation) + 1):
                if shared_u is not None and (l, nu) in shared_u:
                    u_tensors[nu] = shared_u[(l, nu)]
                else:
                    u_tensors[nu] = _strict_so3_u_tensor_in_local_so2_basis(
                        lmax=self.lmax,
                        output_l=l,
                        correlation=nu,
                        dtype=dtype,
                    )
            self.contractions.append(
                SO2PureUContraction(
                    u_tensors=u_tensors,
                    output_l=l,
                    num_elements=int(num_elements),
                    num_features=self.channels,
                    lmax=self.lmax,
                )
            )

    def forward_flat_from_flat(self, x: torch.Tensor, node_attrs: torch.Tensor) -> torch.Tensor:
        out_parts = [
            c(x, node_attrs).view(x.shape[0], self.channels, 2 * l + 1)
            for l, c in enumerate(self.contractions)
        ]
        return torch.cat(out_parts, dim=-1).view(
            x.shape[0],
            self.channels,
            (self.target_lmax + 1) ** 2,
        )

    def forward_flat(
        self, node_feats: dict[tuple[int, int], torch.Tensor], node_attrs: torch.Tensor
    ) -> torch.Tensor:
        return self.forward_flat_from_flat(_local_to_flat_so2(node_feats, self.lmax), node_attrs)

    def forward(
        self, node_feats: dict[tuple[int, int], torch.Tensor], node_attrs: torch.Tensor
    ) -> dict[tuple[int, int], torch.Tensor]:
        out_flat = self.forward_flat(node_feats, node_attrs)
        return _flat_to_local_so2(
            out_flat,
            self.channels,
            self.target_lmax,
        )


class SO2PureUProductBlock(nn.Module):
    """ictd-pure-u product block — replaces SO2ProductBasisBlock."""

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        correlation: int = 3,
        shared_u: dict | None = None,
        output_scale_init: list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.num_elements = int(num_elements)
        self.symmetric_contractions = SO2PureUSymmetricContraction(
            num_elements=num_elements,
            channels=channels,
            lmax=lmax,
            target_lmax=lmax,
            correlation=correlation,
            shared_u=shared_u,
        )
        self.linear = NodeLocalSO2ChannelMixer(self.channels, self.channels, self.lmax)
        _init_node_local_mixer_identity_(self.linear)
        self.output_scale = (
            LocalPerLScaleSO2(self.channels, self.lmax, output_scale_init)
            if output_scale_init is not None
            else nn.Identity()
        )

    def forward(
        self,
        *,
        atom_types: torch.Tensor,
        message: dict[tuple[int, int], torch.Tensor],
        sc: dict[tuple[int, int], torch.Tensor] | None,
    ) -> dict[tuple[int, int], torch.Tensor]:
        y = torch.nn.functional.one_hot(
            atom_types.long().clamp(min=0, max=self.num_elements - 1),
            num_classes=self.num_elements,
        ).to(dtype=next(iter(message.values())).dtype)
        out_flat = self.symmetric_contractions.forward_flat(message, y)
        out_flat = self.linear.forward_flat(out_flat)
        if not isinstance(self.output_scale, nn.Identity):
            out_flat = self.output_scale.forward_flat(out_flat)
        if sc is not None:
            out_flat = out_flat + _local_to_flat_so2(sc, self.lmax)
        out = _flat_to_local_so2(out_flat, self.channels, self.lmax)
        return out


class SO2PureUScalarProductBlock(nn.Module):
    """ictd-pure-u scalar product block — replaces SO2ScalarProductBasisBlock."""

    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        correlation: int = 3,
        shared_u: dict | None = None,
        sc_scale_init: float | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.num_elements = int(num_elements)
        self.symmetric_contractions = SO2PureUSymmetricContraction(
            num_elements=num_elements,
            channels=channels,
            lmax=lmax,
            target_lmax=0,
            correlation=correlation,
            shared_u=shared_u,
        )
        self.linear = nn.Linear(self.channels, self.channels, bias=False)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight.diagonal().fill_(1.0)
        if sc_scale_init is not None:
            self.register_buffer(
                "sc_scale",
                torch.tensor(float(sc_scale_init), dtype=torch.get_default_dtype()),
            )
        else:
            self.sc_scale = None

    def forward(
        self,
        *,
        atom_types: torch.Tensor,
        message: dict[tuple[int, int], torch.Tensor],
        sc: dict[tuple[int, int], torch.Tensor] | None,
    ) -> torch.Tensor:
        y = torch.nn.functional.one_hot(
            atom_types.long().clamp(min=0, max=self.num_elements - 1),
            num_classes=self.num_elements,
        ).to(dtype=next(iter(message.values())).dtype)
        scalar = self.symmetric_contractions.forward_flat(message, y).squeeze(-1)
        out = self.linear(scalar)
        if sc is not None:
            sc_scalar = sc[(0, 0)].squeeze(-1)
            if self.sc_scale is not None:
                sc_scalar = self.sc_scale.to(dtype=out.dtype, device=out.device) * sc_scalar
            out = out + sc_scalar
        return out


class SO2PureUFusionMixBlock(nn.Module):
    """ictd-pure-u fusion mix block — replaces SO2FusionMixBlock."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        lmax: int,
        num_elements: int,
        correlation: int = 3,
        shared_u: dict | None = None,
    ):
        super().__init__()
        self.num_elements = int(num_elements)
        self.reduce = NodeLocalSO2ChannelMixer(
            int(in_channels), int(hidden_channels), int(lmax)
        )
        self.symmetric_contractions = SO2PureUSymmetricContraction(
            num_elements=int(num_elements),
            channels=int(hidden_channels),
            lmax=int(lmax),
            target_lmax=int(lmax),
            correlation=int(correlation),
            shared_u=shared_u,
        )
        self.project = NodeLocalSO2ChannelMixer(
            int(hidden_channels), int(out_channels), int(lmax)
        )
        _init_node_local_mixer_identity_(self.project)

    def forward(
        self,
        x: dict[tuple[int, int], torch.Tensor],
        atom_types: torch.Tensor,
    ) -> dict[tuple[int, int], torch.Tensor]:
        y = torch.nn.functional.one_hot(
            atom_types.long().clamp(min=0, max=self.num_elements - 1),
            num_classes=self.num_elements,
        ).to(dtype=next(iter(x.values())).dtype)
        reduced_flat = self.reduce.forward_flat(_local_to_flat_so2(x, self.symmetric_contractions.lmax))
        contracted_flat = self.symmetric_contractions.forward_flat_from_flat(reduced_flat, y)
        projected_flat = self.project.forward_flat(contracted_flat)
        return _flat_to_local_so2(
            projected_flat,
            self.project.out_channels,
            self.symmetric_contractions.target_lmax,
        )


def _build_shared_u_tensors(
    lmax: int, correlation: int, dtype: torch.dtype
) -> dict[tuple[int, int], torch.Tensor]:
    """Pre-compute all U tensors once so they can be shared across product blocks."""
    shared: dict[tuple[int, int], torch.Tensor] = {}
    for l in range(lmax + 1):
        for nu in range(1, correlation + 1):
            shared[(l, nu)] = _strict_so3_u_tensor_in_local_so2_basis(
                lmax=lmax,
                output_l=l,
                correlation=nu,
                dtype=dtype,
            )
    return shared


class PureCartesianICTDFixSO2(nn.Module):
    """
    SO2-aligned counterpart of PureCartesianICTDFix with a single global->local
    projection stage. Hidden states remain in node-local SO2 basis throughout
    interaction / product / fusion, and only scalar outputs are read out.
    """

    def __init__(
        self,
        max_embed_radius: float,
        main_max_radius: float,
        main_number_of_basis: int,
        hidden_dim_conv: int,
        hidden_dim_sh: int,
        hidden_dim: int,
        channel_in2: int = 32,
        embedding_dim: int = 16,
        max_atomvalue: int = 10,
        atomic_numbers: list[int] | tuple[int, ...] | None = None,
        output_size: int = 8,
        embed_size=None,
        main_hidden_sizes3=None,
        num_layers: int = 1,
        num_interaction: int = 2,
        device=None,
        function_type_main: str = "gaussian",
        lmax: int = 2,
        ictd_Lmax: int = 6,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        max_rank_other: int = 1,
        k_policy: str = "k0",
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
        product5_muls_by_l: dict[int, int] | None = None,
        invariant_channels: int = 32,
        long_range_mode: str = "none",
        long_range_hidden_dim: int = 64,
        long_range_boundary: str = "nonperiodic",
        long_range_neutralize: bool = True,
        long_range_filter_hidden_dim: int = 64,
        long_range_kmax: int = 2,
        long_range_mesh_size: int = 16,
        long_range_slab_padding_factor: int = 2,
        long_range_include_k0: bool = False,
        long_range_source_channels: int = 1,
        long_range_backend: str = "dense_pairwise",
        long_range_reciprocal_backend: str = "direct_kspace",
        long_range_energy_partition: str = "potential",
        long_range_green_mode: str = "poisson",
        long_range_assignment: str = "cic",
        long_range_mesh_fft_full_ewald: bool = False,
        long_range_theta: float = 0.5,
        long_range_leaf_size: int = 32,
        long_range_multipole_order: int = 0,
        long_range_far_source_dim: int = 16,
        long_range_far_num_shells: int = 3,
        long_range_far_shell_growth: float = 2.0,
        long_range_far_tail: bool = True,
        long_range_far_tail_bins: int = 2,
        long_range_far_stats: str = "mean,count,mean_r,rms_r",
        long_range_far_max_radius_multiplier: float | None = None,
        long_range_far_source_norm: bool = True,
        long_range_far_gate_init: float = 0.0,
        feature_spectral_mode: str = "none",
        feature_spectral_bottleneck_dim: int = 8,
        feature_spectral_mesh_size: int = 16,
        feature_spectral_filter_hidden_dim: int = 64,
        feature_spectral_boundary: str = "periodic",
        feature_spectral_slab_padding_factor: int = 2,
        feature_spectral_neutralize: bool = True,
        feature_spectral_include_k0: bool = False,
        feature_spectral_assignment: str = "cic",
        feature_spectral_gate_init: float = 0.0,
        equivariant_post_linear: bool = False,
        ictd_save_tp_mode: str = "fully-connected",
        ictd_fix_route: str = "baseline",
        ictd_fix_contraction_combine: str = "softmax",
        ictd_fix_product_backend: str = "ictd-pure-u",
        ictd_fix_interaction_scale: str = "none",
        ictd_fix_fusion_scale_init: float = 0.1,
        ictd_fix_fusion_heads: int = 1,
        ictd_fix_fusion_head_weight_mode: str = "softmax",
        ictd_fix_readout_head_scale_init: float = 1.0,
        ictd_fix_readout_head_scale_trainable: bool = False,
        save_contraction_order: int = 3,
        save_multiple_mix_channels: int | None = None,
        avg_num_neighbors: float | None = None,
        interaction_compute_dtype: torch.dtype | None = None,
        contraction_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        del hidden_dim_sh, hidden_dim, channel_in2, output_size, embed_size, num_layers
        del ictd_Lmax, ictd_tp_path_policy, ictd_tp_max_rank_other, max_rank_other, k_policy
        del ictd_tp_backend, product5_muls_by_l, invariant_channels
        del long_range_hidden_dim, long_range_boundary, long_range_neutralize
        del long_range_filter_hidden_dim, long_range_kmax, long_range_mesh_size
        del long_range_slab_padding_factor, long_range_include_k0, long_range_source_channels
        del long_range_backend, long_range_reciprocal_backend, long_range_energy_partition
        del long_range_green_mode, long_range_assignment, long_range_mesh_fft_full_ewald
        del long_range_theta, long_range_leaf_size, long_range_multipole_order
        del long_range_far_source_dim, long_range_far_num_shells, long_range_far_shell_growth
        del long_range_far_tail, long_range_far_tail_bins, long_range_far_stats
        del long_range_far_max_radius_multiplier, long_range_far_source_norm, long_range_far_gate_init
        del feature_spectral_bottleneck_dim, feature_spectral_mesh_size
        del feature_spectral_filter_hidden_dim, feature_spectral_boundary
        del feature_spectral_slab_padding_factor, feature_spectral_neutralize
        del feature_spectral_include_k0, feature_spectral_assignment, feature_spectral_gate_init
        del ictd_save_tp_mode
        self.ictd_fix_contraction_combine = str(ictd_fix_contraction_combine)
        if self.ictd_fix_contraction_combine not in {"softmax", "free", "path-free"}:
            raise ValueError(
                f"ictd_fix_contraction_combine must be 'softmax', 'free', or 'path-free', got {ictd_fix_contraction_combine!r}"
            )
        requested_product_backend = str(ictd_fix_product_backend)
        if requested_product_backend == "ictd-mace-u":
            requested_product_backend = "ictd-bridge-u"
        if requested_product_backend not in {"ictd", "native-mace", "ictd-bridge-u", "ictd-pure-u"}:
            raise ValueError(
                "ictd_fix_product_backend must be 'ictd', 'native-mace', 'ictd-bridge-u', "
                f"'ictd-mace-u' alias, or 'ictd-pure-u', got {ictd_fix_product_backend!r}"
            )
        self.ictd_fix_requested_product_backend = requested_product_backend
        self.ictd_fix_product_backend = "ictd-pure-u"
        self.ictd_fix_product_backend_fallback = self.ictd_fix_product_backend != self.ictd_fix_requested_product_backend
        self.ictd_fix_interaction_scale = str(ictd_fix_interaction_scale)
        if self.ictd_fix_interaction_scale not in {"none", "mace-rms"}:
            raise ValueError(
                f"ictd_fix_interaction_scale must be 'none' or 'mace-rms', got {ictd_fix_interaction_scale!r}"
            )
        if main_hidden_sizes3 is None:
            main_hidden_sizes3 = [64]
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if int(num_interaction) < 2:
            raise ValueError(f"num_interaction must be >= 2, got {num_interaction}")
        if ictd_fix_route not in {"baseline", "fusion"}:
            raise ValueError(f"ictd_fix_route must be 'baseline' or 'fusion', got {ictd_fix_route!r}")
        if ictd_fix_fusion_head_weight_mode not in {"softmax", "free"}:
            raise ValueError(
                "ictd_fix_fusion_head_weight_mode must be 'softmax' or 'free', "
                f"got {ictd_fix_fusion_head_weight_mode!r}"
            )
        if long_range_mode != "none" or feature_spectral_mode != "none":
            raise NotImplementedError(
                "pure-cartesian-ictd-fix-so2 currently supports only long_range_mode=none and feature_spectral_mode=none"
            )

        self.channels = int(hidden_dim_conv)
        self.lmax = int(lmax)
        self.num_interaction = int(num_interaction)
        self.max_radius = float(main_max_radius)
        self.number_of_basis = int(main_number_of_basis)
        self.function_type = str(function_type_main)
        self.avg_num_neighbors = None if avg_num_neighbors is None else float(avg_num_neighbors)
        self.ictd_fix_route = str(ictd_fix_route)
        self.ictd_fix_fusion_scale_init = float(ictd_fix_fusion_scale_init)
        self.ictd_fix_fusion_heads = int(ictd_fix_fusion_heads)
        self.ictd_fix_fusion_head_weight_mode = str(ictd_fix_fusion_head_weight_mode)
        self.ictd_fix_readout_head_scale_init = float(ictd_fix_readout_head_scale_init)
        self.ictd_fix_readout_head_scale_trainable = bool(ictd_fix_readout_head_scale_trainable)
        if self.ictd_fix_fusion_heads < 1:
            raise ValueError(f"ictd_fix_fusion_heads must be >= 1, got {self.ictd_fix_fusion_heads}")
        self.internal_compute_dtype = _resolve_internal_compute_dtype(internal_compute_dtype)
        self.interaction_compute_dtype = interaction_compute_dtype
        self.contraction_compute_dtype = contraction_compute_dtype
        self.max_atomvalue = int(max_atomvalue)

        if atomic_numbers is None:
            atomic_numbers = tuple(range(self.max_atomvalue))
        else:
            atomic_numbers = tuple(sorted({int(z) for z in atomic_numbers}))
            if len(atomic_numbers) == 0:
                raise ValueError("atomic_numbers must not be empty")
        self.atomic_numbers = atomic_numbers
        self.num_elements = len(self.atomic_numbers)
        map_size = max(self.max_atomvalue, max(self.atomic_numbers) + 1)
        atomic_number_to_index = torch.full((map_size,), -1, dtype=torch.long)
        for idx, z in enumerate(self.atomic_numbers):
            if z < 0:
                raise ValueError(f"atomic_numbers must be non-negative, got {z}")
            atomic_number_to_index[z] = idx
        self.register_buffer("atomic_number_to_index", atomic_number_to_index, persistent=False)

        self.node_embedding = nn.Embedding(self.num_elements, self.channels)

        self.interactions = nn.ModuleList()
        self.ictd_fix_effective_product_backends: list[str] = []
        message_scale_init = None
        if self.ictd_fix_interaction_scale == "mace-rms":
            message_scale_init = [0.625, 0.561, 0.540, 0.403][: self.lmax + 1]
        self.interactions.append(
            SO2InitialInteraction(
                channels=self.channels,
                lmax=self.lmax,
                number_of_basis=self.number_of_basis,
                num_elements=self.num_elements,
                interaction_compute_dtype=self.interaction_compute_dtype,
                message_scale_init=message_scale_init,
            )
        )
        for layer_idx in range(1, self.num_interaction):
            message_scale_init = None
            sc_scale_init = None
            if self.ictd_fix_interaction_scale == "mace-rms":
                message_presets = {
                    1: [0.489, 0.745, 0.741, 0.620],
                }
                message_scale_init = message_presets.get(layer_idx, [0.5] * (self.lmax + 1))[: self.lmax + 1]
                sc_scale_init = [0.342] + [0.5] * self.lmax
            self.interactions.append(
                SO2LocalInteractionBlock(
                    channels=self.channels,
                    lmax=self.lmax,
                    number_of_basis=self.number_of_basis,
                    max_radius=self.max_radius,
                    function_type=self.function_type,
                    num_elements=self.num_elements,
                    use_self_connection=True,
                    interaction_compute_dtype=self.interaction_compute_dtype,
                    message_scale_init=message_scale_init,
                    sc_scale_init=sc_scale_init,
                )
            )

        self._u_cache = _build_shared_u_tensors(
            self.lmax, save_contraction_order, torch.get_default_dtype()
        )

        product_target_lmax = [self.lmax if layer_idx < self.num_interaction - 1 else 0 for layer_idx in range(self.num_interaction)]
        self.products = nn.ModuleList()
        for layer_idx, target_lmax in enumerate(product_target_lmax):
            self.ictd_fix_effective_product_backends.append(self.ictd_fix_product_backend)
            is_last = layer_idx == self.num_interaction - 1
            if not is_last and target_lmax == self.lmax:
                self.products.append(
                    SO2PureUProductBlock(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.lmax,
                        correlation=save_contraction_order,
                        shared_u=self._u_cache,
                    )
                )
            else:
                self.products.append(
                    SO2PureUScalarProductBlock(
                        num_elements=self.num_elements,
                        channels=self.channels,
                        lmax=self.lmax,
                        correlation=save_contraction_order,
                        shared_u=self._u_cache,
                        sc_scale_init=3.0,
                    )
                )

        self.layer_energy_readouts = nn.ModuleList(
            [NodeLocalL0ReadoutSO2(self.channels, output_init_std=0.003) for _ in range(self.num_interaction - 1)]
        )
        self.last_layer_energy_readout = MACEStyleScalarReadoutSO3(self.channels, hidden_channels=16, output_init_std=0.003)
        if self.ictd_fix_readout_head_scale_trainable:
            self.readout_head_scales = nn.Parameter(
                torch.full((2,), self.ictd_fix_readout_head_scale_init, dtype=torch.get_default_dtype())
            )
        else:
            self.readout_head_scales = None

        if self.ictd_fix_route == "fusion":
            hidden_mix = resolve_save_multiple_mix_channels(self.channels, self.num_interaction, save_multiple_mix_channels)
            self.save_multiple_mix_channels = int(hidden_mix)
            self.ictd_fix_fusion_mix_backend = "ictd-pure-u"
            self.multiple_contraction_mix = SO2PureUFusionMixBlock(
                in_channels=self.channels * self.num_interaction,
                hidden_channels=hidden_mix,
                out_channels=self.channels,
                lmax=self.lmax,
                num_elements=self.num_elements,
                correlation=save_contraction_order,
                shared_u=self._u_cache,
            )
            self.multiple_contract_fuse = None
            shared_fusion_p5_adapt = _make_fusion_product5_adapters(self.channels, self.lmax)
            self.fusion_readouts = nn.ModuleList(
                [
                    FusionProduct5ReadoutSO2(
                        channels=self.channels,
                        lmax=self.lmax,
                        feature_channels=[self.channels] * (self.num_interaction + 1),
                        hidden_sizes=main_hidden_sizes3,
                        internal_compute_dtype=internal_compute_dtype,
                        output_init_std=0.003,
                        shared_adapt=shared_fusion_p5_adapt,
                    )
                    for _ in range(self.ictd_fix_fusion_heads)
                ]
            )
            self.fusion_readout = self.fusion_readouts[0]
            if self.ictd_fix_fusion_heads == 1:
                self.fusion_head_logits = None
                self.fusion_head_weights = None
                self.fusion_energy_scale = nn.Parameter(
                    torch.tensor(self.ictd_fix_fusion_scale_init, dtype=torch.get_default_dtype())
                )
            elif self.ictd_fix_fusion_head_weight_mode == "softmax":
                self.fusion_head_logits = nn.Parameter(
                    torch.zeros(self.ictd_fix_fusion_heads, dtype=torch.get_default_dtype())
                )
                self.fusion_head_weights = None
                self.fusion_energy_scale = nn.Parameter(
                    torch.tensor(self.ictd_fix_fusion_scale_init, dtype=torch.get_default_dtype())
                )
            else:
                self.fusion_head_logits = None
                self.fusion_head_weights = nn.Parameter(
                    torch.full(
                        (self.ictd_fix_fusion_heads,),
                        self.ictd_fix_fusion_scale_init / self.ictd_fix_fusion_heads,
                        dtype=torch.get_default_dtype(),
                    )
                )
                self.fusion_energy_scale = None
        else:
            self.save_multiple_mix_channels = None
            self.multiple_contraction_mix = None
            self.multiple_contract_fuse = None
            self.ictd_fix_fusion_mix_backend = None
            self.fusion_readouts = None
            self.fusion_readout = None
            self.fusion_head_logits = None
            self.fusion_head_weights = None
            self.fusion_energy_scale = None

    def _combined_local_summary(
        self,
        layer_states: list[dict[tuple[int, int], torch.Tensor]],
        *,
        scalar_last: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [node_local_so2_invariants(local, self.lmax) for local in layer_states]
        if scalar_last is not None:
            parts.append(scalar_last)
        return torch.cat(parts, dim=-1)

    def _readout_head_scale(self, index: int, ref: torch.Tensor) -> torch.Tensor:
        if self.readout_head_scales is None:
            return ref.new_tensor(self.ictd_fix_readout_head_scale_init)
        return self.readout_head_scales[index].to(dtype=ref.dtype, device=ref.device)

    def forward(
        self,
        pos,
        A,
        batch,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        *,
        precomputed_edge_vec=None,
        return_combined_features: bool = False,
        sync_after_scatter: callable | None = None,
        return_physical_tensors: bool = False,
        return_reciprocal_source: bool = False,
    ):
        del sync_after_scatter
        if return_physical_tensors:
            raise ValueError("pure-cartesian-ictd-fix-so2 does not currently support return_physical_tensors=True")

        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        sort_idx = torch.argsort(edge_dst)
        edge_src = edge_src[sort_idx]
        edge_dst = edge_dst[sort_idx]
        edge_shifts = edge_shifts[sort_idx]

        if precomputed_edge_vec is not None:
            edge_vec = precomputed_edge_vec[sort_idx]
        else:
            edge_batch_idx = batch[edge_src]
            edge_cells = cell[edge_batch_idx]
            shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
            edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)
        edge_mask = (edge_length <= self.max_radius).to(dtype=pos.dtype).unsqueeze(-1)
        num_nodes = pos.size(0)

        G_node = (
            None
            if self.lmax == 0
            else build_node_local_frames(edge_vec, edge_dst, num_nodes, compute_dtype=self.internal_compute_dtype)
        )
        edge_radial, node_moments_so3, neighbor_count = _build_edge_local_so3_moments_escn(
            edge_vec=edge_vec,
            edge_length=edge_length,
            edge_dst=edge_dst,
            edge_mask=edge_mask,
            num_nodes=num_nodes,
            lmax=self.lmax,
            number_of_basis=self.number_of_basis,
            max_radius=self.max_radius,
            function_type=self.function_type,
            G_node=G_node,
            compute_dtype=self.internal_compute_dtype,
        )
        node_moments: dict[tuple[int, int], torch.Tensor] = {}
        avg_num_neighbors = (
            float(edge_mask.detach().sum().item()) / float(max(num_nodes, 1))
            if self.avg_num_neighbors is None
            else self.avg_num_neighbors
        )

        A_long = A.long()
        if int(A_long.max().item()) >= self.atomic_number_to_index.numel():
            raise ValueError(
                f"Encountered atomic number {int(A_long.max().item())}, but compact mapping supports only up to "
                f"{self.atomic_number_to_index.numel() - 1}. atomic_numbers={self.atomic_numbers}"
            )
        compact_idx = self.atomic_number_to_index[A_long]
        if torch.any(compact_idx < 0):
            bad = torch.unique(A_long[compact_idx < 0]).tolist()
            raise ValueError(
                f"Encountered atomic numbers without compact mapping: {bad}. "
                f"Configured atomic_numbers={self.atomic_numbers}"
            )

        scalar_feats = self.node_embedding(compact_idx)
        local = self.interactions[0](
            scalar_feats=scalar_feats,
            atom_types=compact_idx,
            edge_radial=edge_radial,
            node_moments=node_moments,
            node_moments_so3=node_moments_so3,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_mask=edge_mask,
            neighbor_count=neighbor_count,
            avg_num_neighbors=avg_num_neighbors,
        )
        local = self.products[0](atom_types=compact_idx, message=local, sc=None)
        layer_states = [local]
        if self.num_interaction > 1:
            e0 = self.layer_energy_readouts[0](local)
            total_energy = self._readout_head_scale(0, e0) * e0
        else:
            total_energy = None

        scalar_last = None
        for layer_idx, interaction in enumerate(self.interactions[1:], start=1):
            message, sc = interaction(
                atom_types=compact_idx,
                local=layer_states[-1],
                edge_radial=edge_radial,
                node_moments=node_moments,
                node_moments_so3=node_moments_so3,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_mask=edge_mask,
                neighbor_count=neighbor_count,
                G_node=G_node,
                avg_num_neighbors=avg_num_neighbors,
            )
            if layer_idx == self.num_interaction - 1:
                if self.ictd_fix_route == "fusion":
                    last_preproduct_state = {
                        k: message[k] + (sc[k] if sc is not None else 0.0)
                        for k in message
                    }
                    layer_states.append(last_preproduct_state)
                    scalar_last = self.products[layer_idx](atom_types=compact_idx, message=message, sc=sc)
                else:
                    scalar_last = self.products[layer_idx](atom_types=compact_idx, message=message, sc=sc)
            else:
                local = self.products[layer_idx](atom_types=compact_idx, message=message, sc=sc)
                layer_states.append(local)
                e_layer = self.layer_energy_readouts[layer_idx](local)
                e_layer = self._readout_head_scale(0, e_layer) * e_layer
                total_energy = e_layer if total_energy is None else (total_energy + e_layer)

        if scalar_last is None:
            raise RuntimeError("Expected scalar_last from the final SO2 scalar product block")
        last_energy = self.last_layer_energy_readout(scalar_last)
        last_energy = self._readout_head_scale(1, last_energy) * last_energy
        total_energy = last_energy if total_energy is None else (total_energy + last_energy)

        if self.ictd_fix_route == "fusion":
            mix_input_local = concat_node_local_so2_states(layer_states, self.lmax)
            g_mix = self.multiple_contraction_mix(mix_input_local, compact_idx)
            fusion_inputs = layer_states + [g_mix]
            if self.ictd_fix_fusion_heads == 1:
                fusion_energy = self.fusion_readout(fusion_inputs)
                fusion_energy = (
                    self.fusion_energy_scale.to(dtype=fusion_energy.dtype, device=fusion_energy.device)
                    * fusion_energy
                )
            else:
                shared_fusion_features = self.fusion_readouts[0].product5_features(fusion_inputs)
                head_energies = _project_fusion_features_multihead_so2(
                    self.fusion_readouts,
                    shared_fusion_features,
                )
                if self.ictd_fix_fusion_head_weight_mode == "softmax":
                    head_weights = torch.softmax(
                        self.fusion_head_logits.to(
                            dtype=head_energies.dtype,
                            device=head_energies.device,
                        ),
                        dim=0,
                    )
                    head_weights = self.fusion_energy_scale.to(
                        dtype=head_energies.dtype,
                        device=head_energies.device,
                    ) * head_weights
                else:
                    head_weights = self.fusion_head_weights.to(
                        dtype=head_energies.dtype,
                        device=head_energies.device,
                    )
                head_weights = head_weights.view(self.ictd_fix_fusion_heads, 1, 1)
                fusion_energy = (head_energies * head_weights).sum(dim=0)
            total_energy = total_energy + fusion_energy
            combined_feature_states = layer_states + [g_mix]
        else:
            combined_feature_states = layer_states

        out = total_energy.sum(dim=-1, keepdim=True)
        if return_combined_features:
            combined_features = self._combined_local_summary(
                combined_feature_states,
                scalar_last=scalar_last,
            )
            if return_reciprocal_source:
                return out, combined_features, out.new_empty((out.size(0), 0))
            return out, combined_features
        if return_reciprocal_source:
            return out, out.new_empty((out.size(0), 0))
        return out
