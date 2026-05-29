from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from molecular_force_field.models.radial_basis import soft_one_hot_linspace_mace_cutoff as soft_one_hot_linspace

from molecular_force_field.models.ictd_irreps import direction_harmonics_all
from molecular_force_field.models.mlp import MainNet
from molecular_force_field.models.node_local_so2_helpers import (
    apply_channel_linear,
    build_node_local_frames,
    concat_node_local_so2_states,
    node_local_segments_total,
    node_local_so2_invariants,
    node_local_so2_to_so3_blocks,
    pack_node_local_so2,
    so3_local_so2_basis_cpu,
    so3_blocks_to_node_local_so2,
    unpack_node_local_so2,
)
from molecular_force_field.models.ictd_irreps_2d import (
    HarmonicElementwiseProductSO2,
    HarmonicPathWeightedTensorProductSO2,
    build_cg_tensor_so2,
    so2_irrep_dim,
)
from molecular_force_field.models.pure_cartesian_ictd_layers import (
    _merge_irreps,
    _split_irreps,
)
from molecular_force_field.utils.scatter import scatter


class NodeLocalSO2MixerRect(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lmax: int,
        *,
        gated: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.lmax = int(lmax)
        self.gated = bool(gated)
        self.total_segments = node_local_segments_total(self.lmax)
        self.inv_dim = self.in_channels * self.total_segments

        self.mixers = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        for m in range(self.lmax + 1):
            seg = self.lmax - m + 1
            self.mixers[str(m)] = nn.Linear(self.in_channels * seg, self.out_channels * seg, bias=False)
            if self.gated:
                self.gates[str(m)] = nn.Linear(self.in_channels * seg, self.out_channels * seg, bias=True)

    def forward(self, local: Dict[Tuple[int, int], torch.Tensor]) -> Dict[Tuple[int, int], torch.Tensor]:
        packed = pack_node_local_so2(local, self.lmax)
        out_packed: Dict[int, torch.Tensor] = {}
        for m in range(self.lmax + 1):
            seg = self.lmax - m + 1
            dim = packed[m].shape[-1]
            mixer_dtype = self.mixers[str(m)].weight.dtype
            flat = packed[m].reshape(packed[m].shape[0], seg * self.in_channels, dim).to(dtype=mixer_dtype)
            mixed = apply_channel_linear(flat, self.mixers[str(m)])
            mixed = mixed.reshape(packed[m].shape[0], seg, self.out_channels, dim)
            if self.gated:
                local_summary = (packed[m].square().sum(dim=-1) / math.sqrt(float(dim))).reshape(
                    packed[m].shape[0], seg * self.in_channels
                )
                gate = torch.sigmoid(self.gates[str(m)](local_summary.to(dtype=self.gates[str(m)].weight.dtype))).reshape(
                    packed[m].shape[0], seg, self.out_channels, 1
                )
                mixed = mixed * gate
            out_packed[m] = mixed.to(dtype=packed[m].dtype)
        return unpack_node_local_so2(out_packed, self.lmax)


class NodeLocalSO2Mixer(NodeLocalSO2MixerRect):
    def __init__(self, channels: int, lmax: int, *, gated: bool = True):
        super().__init__(channels, channels, lmax, gated=gated)


class NodeLocalSO2Identity(nn.Module):
    def forward(self, local: Dict[Tuple[int, int], torch.Tensor]) -> Dict[Tuple[int, int], torch.Tensor]:
        return local


class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, min(int(rank), self.in_features, self.out_features))
        self.left = nn.Linear(self.in_features, self.rank, bias=False)
        self.right = nn.Linear(self.rank, self.out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.right(self.left(x))


def build_node_local_moments(
    *,
    edge_local_vec: torch.Tensor,
    edge_length: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_mask: torch.Tensor,
    num_nodes: int,
    lmax: int,
    number_of_basis: int,
    max_radius: float,
    function_type: str,
) -> Tuple[torch.Tensor, Dict[Tuple[int, int], torch.Tensor], torch.Tensor]:
    dtype = edge_local_vec.dtype
    radial = soft_one_hot_linspace(
        edge_length,
        0.0,
        float(max_radius),
        int(number_of_basis),
        basis=str(function_type),
        cutoff=True,
    ).mul(int(number_of_basis) ** 0.5).to(dtype=dtype)
    n_local = edge_local_vec / edge_length.clamp(min=1e-8).unsqueeze(-1).to(dtype=dtype)
    Y_list = direction_harmonics_all(n_local, int(lmax))
    mask = edge_mask.to(dtype=dtype).view(-1, 1, 1)
    neighbor_count = scatter(
        edge_mask.to(dtype=dtype).view(-1),
        edge_dst,
        dim=0,
        dim_size=int(num_nodes),
        reduce="sum",
    ).clamp(min=1.0)
    moments: Dict[Tuple[int, int], torch.Tensor] = {}
    for l in range(int(lmax) + 1):
        y_l = Y_list[l].to(dtype=dtype)
        for m, (s, e) in enumerate(so3_local_so2_basis_cpu(l)[1]):
            weighted = radial.unsqueeze(-1) * y_l[:, s:e].unsqueeze(1)
            moments[(l, m)] = scatter(
                weighted * mask,
                edge_dst,
                dim=0,
                dim_size=int(num_nodes),
                reduce="sum",
            ) / neighbor_count.to(dtype=dtype).sqrt().unsqueeze(-1).unsqueeze(-1)
    return radial, moments, neighbor_count


def build_source_feature_moments(
    local: Dict[Tuple[int, int], torch.Tensor],
    *,
    edge_radial: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_count: torch.Tensor,
    lmax: int,
) -> Dict[Tuple[int, int], torch.Tensor]:
    sample = next(iter(local.values()))
    num_nodes = sample.shape[0]
    dtype = sample.dtype
    mask = edge_mask.to(dtype=dtype).view(-1, 1, 1, 1)
    out: Dict[Tuple[int, int], torch.Tensor] = {}
    for l in range(int(lmax) + 1):
        for m in range(l + 1):
            src_blk = local[(l, m)][edge_src]
            weighted = src_blk.unsqueeze(-2) * edge_radial[:, None, :, None].to(dtype=dtype)
            out[(l, m)] = scatter(
                weighted * mask,
                edge_dst,
                dim=0,
                dim_size=num_nodes,
                reduce="sum",
            ) / neighbor_count.to(dtype=dtype).sqrt().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return out


class NodeLocalSO2EmbeddingConv(nn.Module):
    def __init__(
        self,
        *,
        max_radius: float,
        number_of_basis: int,
        channels: int,
        lmax: int,
        embedding_dim: int,
        max_atomvalue: int,
        function_type: str,
        interaction_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        del embedding_dim
        self.max_radius = float(max_radius)
        self.number_of_basis = int(number_of_basis)
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.adapter_rank = max(4, self.channels // 4)
        self.atom_embedding = nn.Embedding(int(max_atomvalue), self.channels)
        self.source_adapters = nn.ModuleDict(
            {
                str(m): LowRankLinear(
                    (self.lmax - m + 1) * self.channels * self.number_of_basis,
                    self.channels,
                    self.adapter_rank,
                    bias=False,
                )
                for m in range(self.lmax + 1)
            }
        )
        self.geom_adapters = nn.ModuleDict(
            {
                str(m): LowRankLinear(
                    (self.lmax - m + 1) * self.number_of_basis,
                    self.channels,
                    self.adapter_rank,
                    bias=False,
                )
                for m in range(self.lmax + 1)
            }
        )
        self.recouple = HarmonicPathWeightedTensorProductSO2(
            channels=self.channels,
            mmax=self.lmax,
            internal_compute_dtype=interaction_compute_dtype,
        )
        self.out_expand = nn.ModuleDict(
            {
                str(m): LowRankLinear(
                    self.channels,
                    self.channels * (self.lmax - m + 1),
                    self.adapter_rank,
                    bias=False,
                )
                for m in range(self.lmax + 1)
            }
        )

    def _adapt_source(
        self,
        source_moments: Dict[Tuple[int, int], torch.Tensor],
        dtype: torch.dtype,
    ) -> Dict[int, torch.Tensor]:
        out: Dict[int, torch.Tensor] = {}
        for m in range(self.lmax + 1):
            parts = [
                source_moments[(l, m)].reshape(
                    source_moments[(l, m)].shape[0],
                    self.channels * self.number_of_basis,
                    source_moments[(l, m)].shape[-1],
                )
                for l in range(m, self.lmax + 1)
            ]
            flat = torch.cat(parts, dim=1)
            out[m] = apply_channel_linear(flat.to(dtype=dtype), self.source_adapters[str(m)])
        return out

    def _adapt_geom(
        self,
        node_moments: Dict[Tuple[int, int], torch.Tensor],
        dtype: torch.dtype,
    ) -> Dict[int, torch.Tensor]:
        out: Dict[int, torch.Tensor] = {}
        for m in range(self.lmax + 1):
            parts = [
                node_moments[(l, m)].reshape(node_moments[(l, m)].shape[0], self.number_of_basis, node_moments[(l, m)].shape[-1])
                for l in range(m, self.lmax + 1)
            ]
            flat = torch.cat(parts, dim=1)
            out[m] = apply_channel_linear(flat.to(dtype=dtype), self.geom_adapters[str(m)])
        return out

    def _expand_out(self, coupled: Dict[int, torch.Tensor]) -> Dict[Tuple[int, int], torch.Tensor]:
        out: Dict[Tuple[int, int], torch.Tensor] = {}
        for m in range(self.lmax + 1):
            seg = self.lmax - m + 1
            blk = apply_channel_linear(coupled[m], self.out_expand[str(m)])
            blk = blk.reshape(blk.shape[0], seg, self.channels, blk.shape[-1])
            for seg_idx, l in enumerate(range(m, self.lmax + 1)):
                out[(l, m)] = blk[:, seg_idx]
        return out

    def forward(
        self,
        A: torch.Tensor,
        *,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_radial: torch.Tensor,
        node_moments: Dict[Tuple[int, int], torch.Tensor],
        neighbor_count: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        atom_feat = self.atom_embedding(A.long().clamp(min=0, max=self.atom_embedding.num_embeddings - 1))
        scalar_local = {
            (l, m): torch.zeros(
                atom_feat.shape[0],
                self.channels,
                1 if m == 0 else 2,
                device=atom_feat.device,
                dtype=atom_feat.dtype,
            )
            for l in range(self.lmax + 1)
            for m in range(l + 1)
        }
        for l in range(self.lmax + 1):
            scalar_local[(l, 0)] = atom_feat.unsqueeze(-1) if l == 0 else torch.zeros_like(scalar_local[(l, 0)])
        source_moments = build_source_feature_moments(
            scalar_local,
            edge_radial=edge_radial,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_mask=edge_mask,
            neighbor_count=neighbor_count,
            lmax=self.lmax,
        )
        dtype = atom_feat.dtype
        src_dtype = next(self.source_adapters["0"].parameters()).dtype
        geom_dtype = next(self.geom_adapters["0"].parameters()).dtype
        src_dict = self._adapt_source(source_moments, dtype=src_dtype)
        geom_dict = self._adapt_geom(node_moments, dtype=geom_dtype)
        coupled = self.recouple(src_dict, geom_dict)
        return self._expand_out({m: coupled[m].to(dtype=dtype) for m in coupled})


class NodeLocalSO2NodeMomentInteraction(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        lmax: int,
        number_of_basis: int,
        max_radius: float,
        function_type: str,
        internal_compute_dtype: torch.dtype,
        interaction_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        del max_radius, function_type, internal_compute_dtype
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.number_of_basis = int(number_of_basis)
        self.adapter_rank = max(4, self.channels // 4)
        self.source_adapters = nn.ModuleDict(
            {
                str(m): LowRankLinear(
                    (self.lmax - m + 1) * self.channels * self.number_of_basis,
                    self.channels,
                    self.adapter_rank,
                    bias=False,
                )
                for m in range(self.lmax + 1)
            }
        )
        self.geom_adapters = nn.ModuleDict(
            {
                str(m): LowRankLinear(
                    (self.lmax - m + 1) * self.number_of_basis,
                    self.channels,
                    self.adapter_rank,
                    bias=False,
                )
                for m in range(self.lmax + 1)
            }
        )
        self.recouple = HarmonicPathWeightedTensorProductSO2(
            channels=self.channels,
            mmax=self.lmax,
            internal_compute_dtype=interaction_compute_dtype,
        )
        self.out_expand = nn.ModuleDict(
            {
                str(m): LowRankLinear(
                    self.channels,
                    self.channels * (self.lmax - m + 1),
                    self.adapter_rank,
                    bias=False,
                )
                for m in range(self.lmax + 1)
            }
        )
        self.local_mixer = NodeLocalSO2Mixer(self.channels, self.lmax, gated=True)

    def _adapt_source(
        self,
        source_moments: Dict[Tuple[int, int], torch.Tensor],
        dtype: torch.dtype,
    ) -> Dict[int, torch.Tensor]:
        out: Dict[int, torch.Tensor] = {}
        for m in range(self.lmax + 1):
            parts = [
                source_moments[(l, m)].reshape(
                    source_moments[(l, m)].shape[0],
                    self.channels * self.number_of_basis,
                    source_moments[(l, m)].shape[-1],
                )
                for l in range(m, self.lmax + 1)
            ]
            flat = torch.cat(parts, dim=1)
            out[m] = apply_channel_linear(flat.to(dtype=dtype), self.source_adapters[str(m)])
        return out

    def _adapt_geom(
        self,
        node_moments: Dict[Tuple[int, int], torch.Tensor],
        dtype: torch.dtype,
    ) -> Dict[int, torch.Tensor]:
        out: Dict[int, torch.Tensor] = {}
        for m in range(self.lmax + 1):
            parts = [
                node_moments[(l, m)].reshape(node_moments[(l, m)].shape[0], self.number_of_basis, node_moments[(l, m)].shape[-1])
                for l in range(m, self.lmax + 1)
            ]
            flat = torch.cat(parts, dim=1)
            out[m] = apply_channel_linear(flat.to(dtype=dtype), self.geom_adapters[str(m)])
        return out

    def _expand_out(self, coupled: Dict[int, torch.Tensor], dtype: torch.dtype) -> Dict[Tuple[int, int], torch.Tensor]:
        out: Dict[Tuple[int, int], torch.Tensor] = {}
        for m in range(self.lmax + 1):
            seg = self.lmax - m + 1
            blk = apply_channel_linear(coupled[m].to(dtype=dtype), self.out_expand[str(m)])
            blk = blk.reshape(blk.shape[0], seg, self.channels, blk.shape[-1])
            for seg_idx, l in enumerate(range(m, self.lmax + 1)):
                out[(l, m)] = blk[:, seg_idx]
        return out

    def forward(
        self,
        local: Dict[Tuple[int, int], torch.Tensor],
        *,
        edge_radial: torch.Tensor,
        node_moments: Dict[Tuple[int, int], torch.Tensor],
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_mask: torch.Tensor,
        neighbor_count: torch.Tensor,
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        dtype = next(iter(local.values())).dtype
        source_moments = build_source_feature_moments(
            local,
            edge_radial=edge_radial,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_mask=edge_mask,
            neighbor_count=neighbor_count,
            lmax=self.lmax,
        )
        src_dtype = next(self.source_adapters["0"].parameters()).dtype
        geom_dtype = next(self.geom_adapters["0"].parameters()).dtype
        src_dict = self._adapt_source(source_moments, dtype=src_dtype)
        geom_dict = self._adapt_geom(node_moments, dtype=geom_dtype)
        coupled = self.recouple(src_dict, geom_dict)
        agg_local = self._expand_out(coupled, dtype=dtype)
        mixed_local = self.local_mixer(agg_local)
        return {key: mixed_local[key] + local[key] for key in mixed_local}


def _local_key_name(l: int, m: int) -> str:
    return f"{int(l)}_{int(m)}"


def _m_key_name(m: int) -> str:
    return f"m{int(m)}"


class PackedHarmonicPathWeightedTensorProductSO2(nn.Module):
    def __init__(
        self,
        *,
        hidden_channels: int,
        lmax: int,
        normalization: str = "component",
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.lmax = int(lmax)
        self.normalization = str(normalization)
        self.internal_compute_dtype = internal_compute_dtype

        self.paths: List[Tuple[int, int, int]] = []
        self.weight = None
        self._cg_cpu_f64: List[torch.Tensor] = []
        for m1 in range(self.lmax + 1):
            for m2 in range(self.lmax + 1):
                outs = [abs(m1 - m2), m1 + m2]
                for m3 in outs:
                    if m3 > self.lmax:
                        continue
                    path = (m1, m2, m3)
                    if path in self.paths:
                        continue
                    self.paths.append(path)
                    C = build_cg_tensor_so2(m1, m2, m3)
                    if self.normalization == "component":
                        C = C * math.sqrt(float(so2_irrep_dim(m3)))
                    elif self.normalization == "none":
                        pass
                    elif self.normalization != "norm":
                        raise ValueError(f"Unsupported normalization={self.normalization!r}")
                    self._cg_cpu_f64.append(C.to(torch.float64))
        self.num_paths = len(self.paths)
        self.weight = nn.Parameter(torch.randn(self.num_paths, self.hidden_channels) * 0.02)
        self._cg_cache_by_dev_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}

    def _get_cg_list(self, device: torch.device, compute_dtype: torch.dtype) -> List[torch.Tensor]:
        key = (str(device), str(compute_dtype))
        cached = self._cg_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        cg_list = [C.to(device=device, dtype=compute_dtype) for C in self._cg_cpu_f64]
        self._cg_cache_by_dev_dtype[key] = cg_list
        return cg_list

    def forward(self, x1: Dict[int, torch.Tensor], x2: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        sample = next(iter(x1.values()))
        device = sample.device
        dtype = sample.dtype
        compute_dtype = dtype if self.internal_compute_dtype is None else self.internal_compute_dtype
        out: Dict[int, torch.Tensor] = {
            m: torch.zeros_like(x1[m]) for m in range(self.lmax + 1)
        }
        cg_list = self._get_cg_list(device, compute_dtype)
        weight = self.weight.to(device=device, dtype=compute_dtype)

        for p_idx, (m1, m2, m3) in enumerate(self.paths):
            seg = min(x1[m1].shape[1], x2[m2].shape[1], out[m3].shape[1])
            if seg <= 0:
                continue
            a = x1[m1][:, :seg].to(dtype=compute_dtype)
            b = x2[m2][:, :seg].to(dtype=compute_dtype)
            y = torch.einsum("bshm,bshn,mnk->bshk", a, b, cg_list[p_idx])
            y = y * weight[p_idx].view(1, 1, self.hidden_channels, 1)
            out[m3][:, :seg] = out[m3][:, :seg] + y.to(dtype=dtype)
        return out


def _adapt_local_state_for_product5(
    local: Dict[Tuple[int, int], torch.Tensor],
    adapters: nn.ModuleDict,
    lmax: int,
) -> Dict[int, torch.Tensor]:
    packed = pack_node_local_so2(local, lmax)
    out: Dict[int, torch.Tensor] = {}
    for m in range(int(lmax) + 1):
        seg = int(lmax) - m + 1
        dim = packed[m].shape[-1]
        flat = packed[m].reshape(packed[m].shape[0], seg * packed[m].shape[-2], dim)
        out[m] = apply_channel_linear(flat, adapters[str(m)])
    return out


class NodeLocalSO2SymmetricContraction(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        lmax: int,
        correlation: int = 3,
        max_atomvalue: int = 10,
        contraction_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.lmax = int(lmax)
        self.correlation = int(correlation)
        if self.correlation < 1:
            raise ValueError(f"correlation must be >= 1, got {self.correlation}")

        self.reduce = NodeLocalSO2MixerRect(self.in_channels, self.hidden_channels, self.lmax, gated=False)
        self.tp_layers = nn.ModuleList(
            [
                PackedHarmonicPathWeightedTensorProductSO2(
                    hidden_channels=self.hidden_channels,
                    lmax=self.lmax,
                    internal_compute_dtype=contraction_compute_dtype,
                )
                for _ in range(max(self.correlation - 1, 0))
            ]
        )
        self.scalar_dim = self.hidden_channels * node_local_segments_total(self.lmax)
        self.element_embedding = nn.Embedding(int(max_atomvalue), self.scalar_dim)
        self.order_gate = nn.Linear(self.scalar_dim, self.correlation, bias=True)
        self.basis_weight = nn.ParameterDict()
        self.element_weight = nn.ModuleDict()
        self.out_mix = nn.ModuleDict()
        for m in range(self.lmax + 1):
            key = _m_key_name(m)
            self.basis_weight[key] = nn.Parameter(
                torch.randn(self.correlation, self.hidden_channels) / math.sqrt(self.correlation)
            )
            self.element_weight[key] = nn.Embedding(int(max_atomvalue), self.correlation * self.hidden_channels)
            self.out_mix[key] = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)

    def forward(
        self,
        local: Dict[Tuple[int, int], torch.Tensor],
        atom_types: torch.Tensor | None = None,
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        base = self.reduce(local)
        scalars = node_local_so2_invariants(base, self.lmax)
        if atom_types is not None:
            idx = atom_types.long().clamp(min=0, max=self.element_embedding.num_embeddings - 1)
            scalars = scalars + self.element_embedding(idx).to(dtype=scalars.dtype)
        else:
            idx = None
        order_gate = torch.sigmoid(self.order_gate(scalars.to(dtype=self.order_gate.weight.dtype))).to(dtype=base[(0, 0)].dtype)

        base_packed = pack_node_local_so2(base, self.lmax)
        order_blocks: List[Dict[int, torch.Tensor]] = [base_packed]
        current = base_packed
        for tp in self.tp_layers:
            current = tp(current, base_packed)
            order_blocks.append(current)

        out: Dict[Tuple[int, int], torch.Tensor] = {}
        for m in range(self.lmax + 1):
            key = _m_key_name(m)
            stacked = torch.stack([blk[m] for blk in order_blocks], dim=-1)
            coeff = self.basis_weight[key].to(dtype=stacked.dtype).unsqueeze(0)
            if idx is not None:
                elem = self.element_weight[key](idx).reshape(stacked.shape[0], self.correlation, self.hidden_channels)
                coeff = coeff + elem.to(dtype=stacked.dtype)
            coeff = coeff * order_gate.unsqueeze(-1)
            mixed = torch.einsum("bshdr,brh->bshd", stacked, coeff) / math.sqrt(self.correlation)
            seg = mixed.shape[1]
            dim = mixed.shape[-1]
            mixed = mixed.reshape(mixed.shape[0] * seg, self.hidden_channels, dim)
            mixed = apply_channel_linear(mixed, self.out_mix[key])
            mixed = mixed.reshape(-1, seg, self.hidden_channels, dim)
            for seg_idx, l in enumerate(range(m, self.lmax + 1)):
                out[(l, m)] = mixed[:, seg_idx]
        return out


class NodeLocalSO2Interaction(NodeLocalSO2NodeMomentInteraction):
    pass


class PureCartesianICTDNodeSO2TransformerLayer(nn.Module):
    def __init__(
        self,
        max_embed_radius: float,
        main_max_radius: float,
        main_number_of_basis: int,
        hidden_dim_conv: int,
        hidden_dim_sh: int | None = None,
        hidden_dim: int | None = None,
        *,
        output_size: int = 8,
        lmax: int = 2,
        embedding_dim: int = 16,
        max_atomvalue: int = 10,
        num_interaction: int = 3,
        embed_size: List[int] | None = None,
        function_type_main: str = "gaussian",
        device: torch.device | None = None,
        save_readout_mode: str = "elementwise-scalar",
        save_contraction_order: int = 3,
        save_multiple_mix_channels: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        interaction_compute_dtype: torch.dtype | None = None,
        contraction_compute_dtype: torch.dtype | None = None,
        node_axis_mode: str = "geometry",
        node_so2_mixing: str = "packed",
        node_contraction_mode: str = "mace-like",
        equivariant_post_linear: bool = False,
        **_: object,
    ):
        super().__init__()
        del hidden_dim_sh, hidden_dim
        if embed_size is None:
            embed_size = [128, 128, 128]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.channels = int(hidden_dim_conv)
        self.lmax = int(lmax)
        self.num_interaction = int(num_interaction)
        if self.num_interaction < 2:
            raise ValueError(f"num_interaction must be >= 2, got {self.num_interaction}")
        self.max_radius = float(main_max_radius)
        self.number_of_basis = int(main_number_of_basis)
        self.function_type = str(function_type_main)
        self.save_readout_mode = str(save_readout_mode)
        if self.save_readout_mode == "mace-contraction":
            self.save_readout_mode = "multiple-contraction"
        if self.save_readout_mode not in {"elementwise-scalar", "multiple-contraction"}:
            raise ValueError(f"Unsupported save_readout_mode={self.save_readout_mode!r}")
        self.save_contraction_order = int(save_contraction_order)
        self.save_multiple_mix_channels = self.channels if save_multiple_mix_channels is None else int(save_multiple_mix_channels)
        self.node_axis_mode = str(node_axis_mode)
        self.node_so2_mixing = str(node_so2_mixing)
        self.node_contraction_mode = str(node_contraction_mode)
        if self.node_axis_mode != "geometry":
            raise ValueError(f"Unsupported node_axis_mode={self.node_axis_mode!r}")
        if self.node_so2_mixing != "packed":
            raise ValueError(f"Unsupported node_so2_mixing={self.node_so2_mixing!r}")
        if self.node_contraction_mode != "mace-like":
            raise ValueError(f"Unsupported node_contraction_mode={self.node_contraction_mode!r}")
        self.internal_compute_dtype = torch.get_default_dtype() if internal_compute_dtype is None else internal_compute_dtype
        self.interaction_compute_dtype = interaction_compute_dtype
        self.contraction_compute_dtype = contraction_compute_dtype

        self.local_conv_emb = NodeLocalSO2EmbeddingConv(
            max_radius=max_embed_radius,
            number_of_basis=main_number_of_basis,
            channels=self.channels,
            embedding_dim=embedding_dim,
            max_atomvalue=max_atomvalue,
            lmax=self.lmax,
            function_type=function_type_main,
            interaction_compute_dtype=self.interaction_compute_dtype,
        )
        self.post_local_linears = nn.ModuleList()
        for _ in range(self.num_interaction):
            if equivariant_post_linear:
                self.post_local_linears.append(NodeLocalSO2MixerRect(self.channels, self.channels, self.lmax, gated=False))
            else:
                self.post_local_linears.append(NodeLocalSO2Identity())

        self.interaction_layers = nn.ModuleList(
            [
                NodeLocalSO2Interaction(
                    channels=self.channels,
                    lmax=self.lmax,
                    number_of_basis=self.number_of_basis,
                    max_radius=self.max_radius,
                    function_type=self.function_type,
                    internal_compute_dtype=self.internal_compute_dtype,
                    interaction_compute_dtype=self.interaction_compute_dtype,
                )
                for _ in range(self.num_interaction - 1)
            ]
        )

        self.inv_dim = self.channels * node_local_segments_total(self.lmax)
        self.layer_energy_readouts = nn.ModuleList(
            [nn.Linear(self.inv_dim, 1, bias=True) for _ in range(self.num_interaction)]
        )
        self.layer_energy_gates = nn.Parameter(0.1 * torch.ones(self.num_interaction, dtype=torch.get_default_dtype()))
        self.product5_muls_by_m = {m: self.channels for m in range(self.lmax + 1)}
        self._p5_base_mul = self.product5_muls_by_m[0]
        if any(self.product5_muls_by_m[m] != self._p5_base_mul for m in range(self.lmax + 1)):
            raise ValueError(
                "PureCartesianICTDNodeSO2TransformerLayer currently requires uniform product5_muls_by_m across m"
            )

        if self.save_readout_mode == "multiple-contraction":
            self.multiple_contraction_last = NodeLocalSO2SymmetricContraction(
                in_channels=self.channels,
                hidden_channels=self.channels,
                lmax=self.lmax,
                correlation=self.save_contraction_order,
                max_atomvalue=max_atomvalue,
                contraction_compute_dtype=self.contraction_compute_dtype,
            )
            self.multiple_contraction_mix = NodeLocalSO2SymmetricContraction(
                in_channels=self.channels * self.num_interaction,
                hidden_channels=self.save_multiple_mix_channels,
                lmax=self.lmax,
                correlation=self.save_contraction_order,
                max_atomvalue=max_atomvalue,
                contraction_compute_dtype=self.contraction_compute_dtype,
            )
            self.multiple_contract_fuse = NodeLocalSO2MixerRect(
                self.channels + self.save_multiple_mix_channels,
                self.channels,
                self.lmax,
                gated=False,
            )
            self.product5_feature_blocks = self.num_interaction + 1
        else:
            self.multiple_contraction_last = None
            self.multiple_contraction_mix = None
            self.multiple_contract_fuse = None
            self.product5_feature_blocks = self.num_interaction

        self._p5_adapt = nn.ModuleList()
        for _ in range(self.num_interaction):
            layer_adapt = nn.ModuleDict()
            for m in range(self.lmax + 1):
                in_ch = (self.lmax - m + 1) * self.channels
                out_ch = self.product5_muls_by_m[m]
                if in_ch == out_ch:
                    layer_adapt[str(m)] = nn.Identity()
                else:
                    layer_adapt[str(m)] = nn.Linear(in_ch, out_ch, bias=False)
            self._p5_adapt.append(layer_adapt)
        self._p5_contract_adapt = nn.ModuleDict()
        if self.save_readout_mode == "multiple-contraction":
            for m in range(self.lmax + 1):
                in_ch = (self.lmax - m + 1) * self.channels
                out_ch = self.product5_muls_by_m[m]
                if in_ch == out_ch:
                    self._p5_contract_adapt[str(m)] = nn.Identity()
                else:
                    self._p5_contract_adapt[str(m)] = nn.Linear(in_ch, out_ch, bias=False)

        self.product_5 = HarmonicElementwiseProductSO2(
            mmax=self.lmax,
            mul=self.product5_feature_blocks * self._p5_base_mul,
            irreps_out="0",
            internal_compute_dtype=internal_compute_dtype,
        )
        sum_mul = sum(self.product5_muls_by_m[m] for m in range(self.lmax + 1))
        proj_in_dim = self.product5_feature_blocks * sum_mul

        self.proj_total = MainNet(proj_in_dim, embed_size, 1, output_init_std=0.003)

    def _build_node_frames(
        self,
        edge_vec: torch.Tensor,
        edge_dst: torch.Tensor,
        num_nodes: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del dtype
        return build_node_local_frames(
            edge_vec,
            edge_dst,
            num_nodes,
            compute_dtype=self.internal_compute_dtype,
        )

    def _global_to_local(
        self,
        x: torch.Tensor,
        G_node: torch.Tensor,
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        blocks = _split_irreps(x, self.channels, self.lmax)
        return so3_blocks_to_node_local_so2(blocks, G_node, self.lmax, compute_dtype=self.internal_compute_dtype)

    def _local_to_global(
        self,
        local: Dict[Tuple[int, int], torch.Tensor],
        G_node: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        blocks = node_local_so2_to_so3_blocks(
            local,
            G_node,
            self.lmax,
            out_dtype=out_dtype,
            compute_dtype=self.internal_compute_dtype,
        )
        return _merge_irreps(blocks, self.channels, self.lmax)

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
        if return_physical_tensors:
            raise ValueError("node-so2 backbone does not support return_physical_tensors=True")

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

        G_node = self._build_node_frames(edge_vec, edge_dst, num_nodes, dtype=dtype)

        edge_local_vec = torch.einsum(
            "eij,ej->ei",
            G_node[edge_dst].to(dtype=dtype),
            edge_vec.to(dtype=dtype),
        )
        edge_radial, node_moments, neighbor_count = build_node_local_moments(
            edge_local_vec=edge_local_vec,
            edge_length=edge_length,
            edge_dst=edge_dst,
            edge_mask=edge_mask,
            num_nodes=num_nodes,
            lmax=self.lmax,
            number_of_basis=self.number_of_basis,
            max_radius=self.max_radius,
            function_type=self.function_type,
        )

        f1_local = self.local_conv_emb(
            A,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_radial=edge_radial,
            node_moments=node_moments,
            neighbor_count=neighbor_count,
            edge_mask=edge_mask,
        )
        f1_local = self.post_local_linears[0](f1_local)
        features_local = [f1_local]

        for layer_idx, layer in enumerate(self.interaction_layers, start=1):
            local_next = layer(
                features_local[-1],
                edge_radial=edge_radial,
                node_moments=node_moments,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_mask=edge_mask,
                neighbor_count=neighbor_count,
            )
            local_next = self.post_local_linears[layer_idx](local_next)
            features_local.append(local_next)

        if self.save_readout_mode == "multiple-contraction":
            g_last_local = self.multiple_contraction_last(features_local[-1], A)
            mix_input_local = concat_node_local_so2_states(features_local[:-1] + [g_last_local], self.lmax)
            g_mix_local = self.multiple_contraction_mix(mix_input_local, A)
            fused_local = self.multiple_contract_fuse(
                concat_node_local_so2_states([g_last_local, g_mix_local], self.lmax)
            )
        else:
            fused_local = None

        T_blocks: Dict[int, torch.Tensor] = {}
        adapted_features = [
            _adapt_local_state_for_product5(local, self._p5_adapt[i], self.lmax)
            for i, local in enumerate(features_local)
        ]
        adapted_fused = (
            _adapt_local_state_for_product5(fused_local, self._p5_contract_adapt, self.lmax)
            if self.save_readout_mode == "multiple-contraction"
            else None
        )
        for m in range(self.lmax + 1):
            parts = [adapted[m] for adapted in adapted_features]
            if adapted_fused is not None:
                parts.append(adapted_fused[m])
            T_blocks[m] = torch.cat(parts, dim=-2)

        proj_dtype = next(self.proj_total.parameters()).dtype
        f_prod5 = self.product_5(T_blocks, T_blocks).to(dtype=proj_dtype)
        e_out = self.proj_total(f_prod5)

        aux_energy = None
        for idx, local in enumerate(features_local):
            layer_inv = node_local_so2_invariants(local, self.lmax).to(
                dtype=self.layer_energy_readouts[idx].weight.dtype
            )
            e_layer = self.layer_energy_gates[idx] * self.layer_energy_readouts[idx](
                layer_inv
            )
            aux_energy = e_layer if aux_energy is None else aux_energy + e_layer
        if aux_energy is not None:
            e_out = e_out + aux_energy

        out = e_out.sum(dim=-1, keepdim=True)
        if return_combined_features:
            features_global = [
                self._local_to_global(local, G_node, dtype)
                for local in features_local
            ]
            if fused_local is not None:
                fused_global = self._local_to_global(fused_local, G_node, dtype)
                combined_features = torch.cat(features_global + [fused_global], dim=-1)
            else:
                combined_features = torch.cat(features_global, dim=-1)
            if return_reciprocal_source:
                return out, combined_features, out.new_empty((out.size(0), 0))
            return out, combined_features
        if return_reciprocal_source:
            return out, out.new_empty((out.size(0), 0))
        return out
