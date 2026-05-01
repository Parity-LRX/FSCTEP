from __future__ import annotations

import math

import torch
import torch.nn as nn

from molecular_force_field.models.ictd_irreps_2d import HarmonicElementwiseProductSO2
from molecular_force_field.models.mlp import MainNet
from molecular_force_field.models.node_local_so2_helpers import (
    apply_channel_linear,
    build_node_local_frames,
    concat_node_local_so2_states,
    node_local_so2_invariants,
    so3_blocks_to_node_local_so2,
)
from molecular_force_field.models.pure_cartesian_ictd_fix import MACEStyleScalarReadoutSO3
from molecular_force_field.models.pure_cartesian_ictd_layers_node_so2 import (
    LowRankLinear,
    _adapt_local_state_for_product5,
    build_source_feature_moments,
)
from molecular_force_field.models.pure_cartesian_ictd_layers import resolve_save_multiple_mix_channels
from molecular_force_field.utils.scatter import scatter


def _resolve_internal_compute_dtype(internal_compute_dtype: torch.dtype | None) -> torch.dtype:
    return torch.get_default_dtype() if internal_compute_dtype is None else internal_compute_dtype


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
    G_node: torch.Tensor,
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
    n_global = edge_vec / edge_length.clamp(min=1e-8).unsqueeze(-1).to(dtype=dtype)
    Y_all = _sh(n_global, int(lmax))

    mask = edge_mask.to(dtype=dtype).view(-1, 1, 1)
    neighbor_count = (
        scatter(edge_mask.to(dtype=dtype).view(-1), edge_dst, dim=0, dim_size=int(num_nodes), reduce="sum")
        .clamp(min=1.0)
    )

    so3_blocks: dict[int, torch.Tensor] = {}
    for l in range(int(lmax) + 1):
        weighted = radial.unsqueeze(-1) * Y_all[l].unsqueeze(1)
        so3_blocks[l] = (
            scatter(weighted * mask, edge_dst, dim=0, dim_size=int(num_nodes), reduce="sum")
            / neighbor_count.to(dtype=dtype).sqrt().unsqueeze(-1).unsqueeze(-1)
        )

    moments = so3_blocks_to_node_local_so2(so3_blocks, G_node, int(lmax), compute_dtype=compute_dtype)
    return radial, moments, neighbor_count


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

    def forward(self, local: dict[tuple[int, int], torch.Tensor]) -> dict[tuple[int, int], torch.Tensor]:
        return {
            (l, m): apply_channel_linear(block, self.adapters[str(l)])
            for (l, m), block in local.items()
        }


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


class ElementConditionedLocalLinearSO2(nn.Module):
    def __init__(self, num_elements: int, channels: int, lmax: int, bias: bool = False):
        super().__init__()
        self.num_elements = int(num_elements)
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.weights = nn.ParameterDict(
            {
                f"{l}_{m}": nn.Parameter(torch.randn(self.num_elements, self.channels, self.channels) * 0.02)
                for l in range(self.lmax + 1)
                for m in range(l + 1)
            }
        )
        if bias:
            self.bias = nn.ParameterDict(
                {
                    f"{l}_{m}": nn.Parameter(torch.zeros(self.num_elements, self.channels))
                    for l in range(self.lmax + 1)
                    for m in range(l + 1)
                }
            )
        else:
            self.bias = None

    def forward(
        self,
        local: dict[tuple[int, int], torch.Tensor],
        atom_types: torch.Tensor,
    ) -> dict[tuple[int, int], torch.Tensor]:
        idx = atom_types.long().clamp(min=0, max=self.num_elements - 1)
        out: dict[tuple[int, int], torch.Tensor] = {}
        for (l, m), block in local.items():
            key = f"{l}_{m}"
            weight = self.weights[key][idx].to(dtype=block.dtype)
            out_block = torch.einsum("noi,nid->nod", weight, block)
            if self.bias is not None:
                out_block = out_block + self.bias[key][idx].to(dtype=block.dtype).unsqueeze(-1)
            out[(l, m)] = out_block
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
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.feature_channels = [int(c) for c in feature_channels]
        self.num_feature_blocks = len(self.feature_channels)
        self.product5_muls_by_m = {m: self.channels for m in range(self.lmax + 1)}
        self._p5_base_mul = self.product5_muls_by_m[0]
        shared_adapt = nn.ModuleDict()
        for m in range(self.lmax + 1):
            in_ch = (self.lmax - m + 1) * self.channels
            out_ch = self.product5_muls_by_m[m]
            if in_ch == out_ch:
                shared_adapt[str(m)] = nn.Identity()
            else:
                shared_adapt[str(m)] = nn.Linear(in_ch, out_ch, bias=False)
        self._p5_adapt = shared_adapt
        self.num_feature_blocks = len(self.feature_channels)

        self.product_5 = HarmonicElementwiseProductSO2(
            mmax=self.lmax,
            mul=self.num_feature_blocks * self._p5_base_mul,
            irreps_out="0",
            internal_compute_dtype=internal_compute_dtype,
        )
        sum_mul = sum(self.product5_muls_by_m[m] for m in range(self.lmax + 1))
        self.proj_total = MainNet(self.num_feature_blocks * sum_mul, hidden_sizes, 1, output_init_std=output_init_std)

    def forward(self, local_states: list[dict[tuple[int, int], torch.Tensor]]) -> torch.Tensor:
        T_blocks: dict[int, torch.Tensor] = {}
        adapted_features = [
            _adapt_local_state_for_product5(local, self._p5_adapt, self.lmax)
            for local in local_states
        ]
        for m in range(self.lmax + 1):
            T_blocks[m] = torch.cat([adapted[m] for adapted in adapted_features], dim=-2)
        proj_dtype = next(self.proj_total.parameters()).dtype
        f_prod5 = self.product_5(T_blocks, T_blocks).to(dtype=proj_dtype)
        return self.proj_total(f_prod5)


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
        interaction_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
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
        from molecular_force_field.models.ictd_irreps_2d import HarmonicPathWeightedTensorProductSO2

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

    def _adapt_source(self, source_moments, dtype):
        out = {}
        for m in range(self.lmax + 1):
            parts = [
                source_moments[(l, m)].reshape(
                    source_moments[(l, m)].shape[0],
                    self.channels * self.number_of_basis,
                    source_moments[(l, m)].shape[-1],
                )
                for l in range(m, self.lmax + 1)
            ]
            out[m] = apply_channel_linear(torch.cat(parts, dim=1).to(dtype=dtype), self.source_adapters[str(m)])
        return out

    def _adapt_geom(self, node_moments, dtype):
        out = {}
        for m in range(self.lmax + 1):
            parts = [
                node_moments[(l, m)].reshape(node_moments[(l, m)].shape[0], self.number_of_basis, node_moments[(l, m)].shape[-1])
                for l in range(m, self.lmax + 1)
            ]
            out[m] = apply_channel_linear(torch.cat(parts, dim=1).to(dtype=dtype), self.geom_adapters[str(m)])
        return out

    def _expand_out(self, coupled):
        out = {}
        for m in range(self.lmax + 1):
            seg = self.lmax - m + 1
            blk = apply_channel_linear(coupled[m], self.out_expand[str(m)])
            blk = blk.reshape(blk.shape[0], seg, self.channels, blk.shape[-1])
            for seg_idx, l in enumerate(range(m, self.lmax + 1)):
                out[(l, m)] = blk[:, seg_idx]
        return out

    def forward(
        self,
        *,
        scalar_feats: torch.Tensor,
        edge_radial: torch.Tensor,
        node_moments: dict[tuple[int, int], torch.Tensor],
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_mask: torch.Tensor,
        neighbor_count: torch.Tensor,
    ) -> dict[tuple[int, int], torch.Tensor]:
        dtype = scalar_feats.dtype
        scalar_local = {
            (l, m): torch.zeros(
                scalar_feats.shape[0], self.channels,
                1 if m == 0 else 2, device=scalar_feats.device, dtype=dtype,
            )
            for l in range(self.lmax + 1) for m in range(l + 1)
        }
        scalar_local[(0, 0)] = scalar_feats.unsqueeze(-1)

        source_moments = build_source_feature_moments(
            scalar_local,
            edge_radial=edge_radial, edge_src=edge_src, edge_dst=edge_dst,
            edge_mask=edge_mask, neighbor_count=neighbor_count, lmax=self.lmax,
        )
        src_dtype = next(self.source_adapters["0"].parameters()).dtype
        geom_dtype = next(self.geom_adapters["0"].parameters()).dtype
        src_dict = self._adapt_source(source_moments, dtype=src_dtype)
        geom_dict = self._adapt_geom(node_moments, dtype=geom_dtype)
        coupled = self.recouple(src_dict, geom_dict)
        return self._expand_out({m: coupled[m].to(dtype=dtype) for m in coupled})


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
        self.adapter_rank = max(4, self.channels // 4)
        self.use_self_connection = bool(use_self_connection)

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
        from molecular_force_field.models.ictd_irreps_2d import HarmonicPathWeightedTensorProductSO2

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
        self.local_mixer = NodeLocalSO2ChannelMixer(self.channels, self.channels, self.lmax)
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
        self.sc_output_scale = (
            LocalPerLScaleSO2(self.channels, self.lmax, sc_scale_init)
            if sc_scale_init is not None
            else nn.Identity()
        )

    def _adapt_source(
        self,
        source_moments: dict[tuple[int, int], torch.Tensor],
        dtype: torch.dtype,
    ) -> dict[int, torch.Tensor]:
        out: dict[int, torch.Tensor] = {}
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
        node_moments: dict[tuple[int, int], torch.Tensor],
        dtype: torch.dtype,
    ) -> dict[int, torch.Tensor]:
        out: dict[int, torch.Tensor] = {}
        for m in range(self.lmax + 1):
            parts = [
                node_moments[(l, m)].reshape(node_moments[(l, m)].shape[0], self.number_of_basis, node_moments[(l, m)].shape[-1])
                for l in range(m, self.lmax + 1)
            ]
            flat = torch.cat(parts, dim=1)
            out[m] = apply_channel_linear(flat.to(dtype=dtype), self.geom_adapters[str(m)])
        return out

    def _expand_out(self, coupled: dict[int, torch.Tensor], dtype: torch.dtype) -> dict[tuple[int, int], torch.Tensor]:
        out: dict[tuple[int, int], torch.Tensor] = {}
        for m in range(self.lmax + 1):
            seg = self.lmax - m + 1
            blk = apply_channel_linear(coupled[m].to(dtype=dtype), self.out_expand[str(m)])
            blk = blk.reshape(blk.shape[0], seg, self.channels, blk.shape[-1])
            for seg_idx, l in enumerate(range(m, self.lmax + 1)):
                out[(l, m)] = blk[:, seg_idx]
        return out

    def forward(
        self,
        *,
        atom_types: torch.Tensor,
        local: dict[tuple[int, int], torch.Tensor],
        edge_radial: torch.Tensor,
        node_moments: dict[tuple[int, int], torch.Tensor],
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_mask: torch.Tensor,
        neighbor_count: torch.Tensor,
    ) -> tuple[dict[tuple[int, int], torch.Tensor], dict[tuple[int, int], torch.Tensor] | None]:
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
        message = self.local_mixer(self._expand_out(coupled, dtype=dtype))
        message = self.message_output_scale(message) if not isinstance(self.message_output_scale, nn.Identity) else message
        sc = None
        if self.self_connection is not None:
            sc = self.self_connection(local, atom_types)
            sc = self.sc_output_scale(sc) if not isinstance(self.sc_output_scale, nn.Identity) else sc
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
                    from molecular_force_field.models.ictd_irreps_2d import (
                        ictd_u_tensor_so2,
                    )

                    u_tensors[nu] = ictd_u_tensor_so2(
                        lmax=self.lmax,
                        output_l=l,
                        correlation=nu,
                        normalization="component",
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

    def forward(
        self, node_feats: dict[tuple[int, int], torch.Tensor], node_attrs: torch.Tensor
    ) -> dict[tuple[int, int], torch.Tensor]:
        x = _local_to_flat_so2(node_feats, self.lmax)
        out_parts = [c(x, node_attrs) for c in self.contractions]
        out_flat = torch.cat(out_parts, dim=-1)
        D_target = (self.target_lmax + 1) ** 2
        return _flat_to_local_so2(
            out_flat.view(x.shape[0], self.channels, D_target),
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
        out = self.linear(self.symmetric_contractions(message, y))
        if sc is not None:
            out = {key: out[key] + sc[key] for key in out}
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
        contracted = self.symmetric_contractions(message, y)
        scalar = contracted[(0, 0)].squeeze(-1)
        out = self.linear(scalar)
        if sc is not None:
            out = out + sc[(0, 0)].squeeze(-1)
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

    def forward(
        self,
        x: dict[tuple[int, int], torch.Tensor],
        atom_types: torch.Tensor,
    ) -> dict[tuple[int, int], torch.Tensor]:
        y = torch.nn.functional.one_hot(
            atom_types.long().clamp(min=0, max=self.num_elements - 1),
            num_classes=self.num_elements,
        ).to(dtype=next(iter(x.values())).dtype)
        return self.project(self.symmetric_contractions(self.reduce(x), y))


def _build_shared_u_tensors(
    lmax: int, correlation: int, dtype: torch.dtype
) -> dict[tuple[int, int], torch.Tensor]:
    """Pre-compute all U tensors once so they can be shared across product blocks."""
    from molecular_force_field.models.ictd_irreps_2d import ictd_u_tensor_so2

    shared: dict[tuple[int, int], torch.Tensor] = {}
    for l in range(lmax + 1):
        for nu in range(1, correlation + 1):
            shared[(l, nu)] = ictd_u_tensor_so2(
                lmax=lmax, output_l=l, correlation=nu,
                normalization="component", dtype=dtype,
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
        save_contraction_order: int = 3,
        save_multiple_mix_channels: int | None = None,
        avg_num_neighbors: float | None = None,
        interaction_compute_dtype: torch.dtype | None = None,
        contraction_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        del hidden_dim_sh, hidden_dim, channel_in2, output_size, embed_size, num_layers
        del ictd_Lmax, ictd_tp_path_policy, ictd_tp_max_rank_other, max_rank_other, k_policy
        del ictd_tp_backend, product5_muls_by_l, invariant_channels, avg_num_neighbors
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
        self.ictd_fix_product_backend = str(ictd_fix_product_backend)
        self.ictd_fix_interaction_scale = str(ictd_fix_interaction_scale)
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
        self.ictd_fix_route = str(ictd_fix_route)
        self.ictd_fix_fusion_scale_init = float(ictd_fix_fusion_scale_init)
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
        self.interactions.append(
            SO2InitialInteraction(
                channels=self.channels,
                lmax=self.lmax,
                number_of_basis=self.number_of_basis,
                interaction_compute_dtype=self.interaction_compute_dtype,
            )
        )
        for _ in range(self.num_interaction - 1):
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
                )
            )

        self._u_cache = _build_shared_u_tensors(
            self.lmax, save_contraction_order, torch.get_default_dtype()
        )

        product_target_lmax = [self.lmax if layer_idx < self.num_interaction - 1 else 0 for layer_idx in range(self.num_interaction)]
        self.products = nn.ModuleList()
        for layer_idx, target_lmax in enumerate(product_target_lmax):
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
                    )
                )

        self.layer_energy_readouts = nn.ModuleList(
            [NodeLocalL0ReadoutSO2(self.channels, output_init_std=0.003) for _ in range(self.num_interaction - 1)]
        )
        self.last_layer_energy_readout = MACEStyleScalarReadoutSO3(self.channels, hidden_channels=16, output_init_std=0.003)

        if self.ictd_fix_route == "fusion":
            hidden_mix = resolve_save_multiple_mix_channels(self.channels, self.num_interaction, save_multiple_mix_channels)
            self.save_multiple_mix_channels = int(hidden_mix)
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
            self.fusion_readout = FusionProduct5ReadoutSO2(
                channels=self.channels,
                lmax=self.lmax,
                feature_channels=[self.channels] * (self.num_interaction + 1),
                hidden_sizes=main_hidden_sizes3,
                internal_compute_dtype=internal_compute_dtype,
                output_init_std=0.003,
            )
            self.fusion_energy_scale = nn.Parameter(
                torch.tensor(self.ictd_fix_fusion_scale_init, dtype=torch.get_default_dtype())
            )
        else:
            self.save_multiple_mix_channels = None
            self.multiple_contraction_mix = None
            self.multiple_contract_fuse = None
            self.fusion_readout = None
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

        G_node = build_node_local_frames(edge_vec, edge_dst, num_nodes, compute_dtype=self.internal_compute_dtype)
        edge_radial, node_moments, neighbor_count = _build_node_local_moments_escn(
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
            edge_radial=edge_radial,
            node_moments=node_moments,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_mask=edge_mask,
            neighbor_count=neighbor_count,
        )
        local = self.products[0](atom_types=compact_idx, message=local, sc=None)
        layer_states = [local]
        total_energy = self.layer_energy_readouts[0](local) if self.num_interaction > 1 else None

        scalar_last = None
        for layer_idx, interaction in enumerate(self.interactions[1:], start=1):
            message, sc = interaction(
                atom_types=compact_idx,
                local=layer_states[-1],
                edge_radial=edge_radial,
                node_moments=node_moments,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_mask=edge_mask,
                neighbor_count=neighbor_count,
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
                total_energy = e_layer if total_energy is None else (total_energy + e_layer)

        if scalar_last is None:
            raise RuntimeError("Expected scalar_last from the final SO2 scalar product block")
        last_energy = self.last_layer_energy_readout(scalar_last)
        total_energy = last_energy if total_energy is None else (total_energy + last_energy)

        if self.ictd_fix_route == "fusion":
            mix_input_local = concat_node_local_so2_states(layer_states, self.lmax)
            g_mix = self.multiple_contraction_mix(mix_input_local, compact_idx)
            fusion_energy = self.fusion_readout(layer_states + [g_mix])
            fusion_energy = self.fusion_energy_scale.to(dtype=fusion_energy.dtype, device=fusion_energy.device) * fusion_energy
            total_energy = total_energy + fusion_energy
            combined_features = self._combined_local_summary(layer_states + [g_mix], scalar_last=scalar_last)
        else:
            combined_features = self._combined_local_summary(layer_states, scalar_last=scalar_last)

        out = total_energy.sum(dim=-1, keepdim=True)
        if return_combined_features:
            if return_reciprocal_source:
                return out, combined_features, out.new_empty((out.size(0), 0))
            return out, combined_features
        if return_reciprocal_source:
            return out, out.new_empty((out.size(0), 0))
        return out
