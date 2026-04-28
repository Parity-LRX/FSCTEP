from __future__ import annotations

import math

import torch
import torch.nn as nn
from e3nn.math import soft_one_hot_linspace

from molecular_force_field.models.ictd_irreps_2d import HarmonicElementwiseProductSO2
from molecular_force_field.models.mlp import MainNet
from molecular_force_field.models.node_local_so2_helpers import (
    apply_channel_linear,
    build_node_local_frames,
    concat_node_local_so2_states,
    node_local_segments_total,
    node_local_so2_invariants,
)
from molecular_force_field.models.pure_cartesian_ictd_fix import MACEStyleScalarReadoutSO3
from molecular_force_field.models.pure_cartesian_ictd_layers_node_so2 import (
    LowRankLinear,
    NodeLocalSO2EmbeddingConv,
    NodeLocalSO2Identity,
    NodeLocalSO2MixerRect,
    NodeLocalSO2SymmetricContraction,
    _adapt_local_state_for_product5,
    build_node_local_moments,
    build_source_feature_moments,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_so2 import (
    HarmonicLocalO2ContractionTensorProduct,
)
from molecular_force_field.models.pure_cartesian_ictd_layers import resolve_save_multiple_mix_channels


def _resolve_internal_compute_dtype(internal_compute_dtype: torch.dtype | None) -> torch.dtype:
    return torch.get_default_dtype() if internal_compute_dtype is None else internal_compute_dtype


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


class NodeLocalScalarReadoutSO2(nn.Module):
    def __init__(self, channels: int, lmax: int, output_init_std: float = 0.003):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        inv_dim = self.channels * ((self.lmax + 1) * (self.lmax + 2) // 2)
        self.readout = nn.Linear(inv_dim, 1, bias=True)
        nn.init.normal_(self.readout.weight, mean=0.0, std=float(output_init_std))
        nn.init.zeros_(self.readout.bias)

    def forward(self, local: dict[tuple[int, int], torch.Tensor]) -> torch.Tensor:
        inv = node_local_so2_invariants(local, self.lmax).to(dtype=self.readout.weight.dtype)
        return self.readout(inv)


class NodeLocalMACEStyleScalarReadoutSO2(nn.Module):
    def __init__(self, channels: int, lmax: int, hidden_channels: int = 16, output_init_std: float = 0.003):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        inv_dim = self.channels * ((self.lmax + 1) * (self.lmax + 2) // 2)
        self.net = nn.Sequential(
            nn.Linear(inv_dim, int(hidden_channels), bias=True),
            nn.SiLU(),
            nn.Linear(int(hidden_channels), 1, bias=True),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=float(output_init_std))
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, local: dict[tuple[int, int], torch.Tensor]) -> torch.Tensor:
        inv = node_local_so2_invariants(local, self.lmax).to(dtype=self.net[0].weight.dtype)
        return self.net(inv)


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
        self._p5_adapt = nn.ModuleList()
        for block_channels in self.feature_channels:
            layer_adapt = nn.ModuleDict()
            for m in range(self.lmax + 1):
                in_ch = (self.lmax - m + 1) * block_channels
                out_ch = self.product5_muls_by_m[m]
                if in_ch == out_ch:
                    layer_adapt[str(m)] = nn.Identity()
                else:
                    layer_adapt[str(m)] = nn.Linear(in_ch, out_ch, bias=False)
            self._p5_adapt.append(layer_adapt)

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
            _adapt_local_state_for_product5(local, self._p5_adapt[i], self.lmax)
            for i, local in enumerate(local_states)
        ]
        for m in range(self.lmax + 1):
            T_blocks[m] = torch.cat([adapted[m] for adapted in adapted_features], dim=-2)
        proj_dtype = next(self.proj_total.parameters()).dtype
        f_prod5 = self.product_5(T_blocks, T_blocks).to(dtype=proj_dtype)
        return self.proj_total(f_prod5)


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
        self.local_mixer = NodeLocalSO2MixerRect(self.channels, self.channels, self.lmax, gated=True)
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


def _o2_key_name(key: tuple[str, int]) -> str:
    kind, value = key
    if kind == "scalar":
        return f"scalar_{'p' if value > 0 else 'm'}1"
    return f"freq_{int(value)}"


class NodeLocalO2SymmetricContraction(nn.Module):
    """
    O(2)-complete contraction stack for node-local SO(2) hidden states.

    The external hidden state stays in canonical local SO(2) blocks `(l, m)`,
    but higher-order products carry an internal latent `0o` branch so scalar
    parity is complete like the SO(3) multiple-contraction path.
    """

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
                HarmonicLocalO2ContractionTensorProduct(
                    channels=self.hidden_channels,
                    lmax=self.lmax,
                    internal_compute_dtype=contraction_compute_dtype,
                )
                for _ in range(max(self.correlation - 1, 0))
            ]
        )
        self.scalar_dim = self.hidden_channels * node_local_segments_total(self.lmax)
        self.element_embedding = nn.Embedding(int(max_atomvalue), self.scalar_dim)
        self.order_gate = nn.Linear(self.scalar_dim, self.correlation, bias=True)

        self._contributors_by_m = {m: [l for l in range(m, self.lmax + 1)] for m in range(self.lmax + 1)}
        self.hidden_active_irreps: list[tuple[str, int]] = [("scalar", 1), ("scalar", -1)] + [
            ("freq", m) for m in range(1, self.lmax + 1)
        ]
        self._segments_by_key = {("scalar", 1): len(self._contributors_by_m[0]), ("scalar", -1): len(self._contributors_by_m[0])}
        for m in range(1, self.lmax + 1):
            self._segments_by_key[("freq", m)] = len(self._contributors_by_m[m])

        self.basis_weight = nn.ParameterDict()
        self.element_weight = nn.ModuleDict()
        self.out_mix = nn.ModuleDict()
        for key in self.hidden_active_irreps:
            name = _o2_key_name(key)
            self.basis_weight[name] = nn.Parameter(
                torch.randn(self.correlation, self.hidden_channels) / math.sqrt(self.correlation)
            )
            self.element_weight[name] = nn.Embedding(int(max_atomvalue), self.correlation * self.hidden_channels)
            self.out_mix[name] = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)

    def _local_to_hidden_o2(self, local: dict[tuple[int, int], torch.Tensor]) -> dict[tuple[str, int], torch.Tensor]:
        sample = next(iter(local.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        scalar_even = torch.cat([local[(l, 0)] for l in self._contributors_by_m[0]], dim=-2)
        scalar_odd = torch.zeros(*batch_shape, scalar_even.shape[-2], 1, device=device, dtype=dtype)
        out: dict[tuple[str, int], torch.Tensor] = {
            ("scalar", 1): scalar_even,
            ("scalar", -1): scalar_odd,
        }
        for m in range(1, self.lmax + 1):
            out[("freq", m)] = torch.cat([local[(l, m)] for l in self._contributors_by_m[m]], dim=-2)
        return out

    def _hidden_o2_to_local(
        self,
        hidden: dict[tuple[str, int], torch.Tensor],
    ) -> dict[tuple[int, int], torch.Tensor]:
        out: dict[tuple[int, int], torch.Tensor] = {}
        scalar_even = hidden[("scalar", 1)]
        step0 = self.hidden_channels
        for seg_idx, l in enumerate(self._contributors_by_m[0]):
            s = seg_idx * step0
            e = s + step0
            out[(l, 0)] = scalar_even[..., s:e, :]
        for m in range(1, self.lmax + 1):
            blk = hidden[("freq", m)]
            step = self.hidden_channels
            for seg_idx, l in enumerate(self._contributors_by_m[m]):
                s = seg_idx * step
                e = s + step
                out[(l, m)] = blk[..., s:e, :]
        return out

    def forward(
        self,
        local: dict[tuple[int, int], torch.Tensor],
        atom_types: torch.Tensor | None = None,
    ) -> dict[tuple[int, int], torch.Tensor]:
        base = self.reduce(local)
        scalars = node_local_so2_invariants(base, self.lmax)
        if atom_types is not None:
            idx = atom_types.long().clamp(min=0, max=self.element_embedding.num_embeddings - 1)
            scalars = scalars + self.element_embedding(idx).to(dtype=scalars.dtype)
        else:
            idx = None
        order_gate = torch.sigmoid(self.order_gate(scalars.to(dtype=self.order_gate.weight.dtype))).to(dtype=base[(0, 0)].dtype)

        base_hidden = self._local_to_hidden_o2(base)
        order_blocks: list[dict[tuple[str, int], torch.Tensor]] = [base_hidden]
        current = base_hidden
        for tp in self.tp_layers:
            current = tp(current, base_hidden)
            order_blocks.append(current)

        hidden_out: dict[tuple[str, int], torch.Tensor] = {}
        for key in self.hidden_active_irreps:
            name = _o2_key_name(key)
            seg = self._segments_by_key[key]
            packed_orders = []
            for blk in order_blocks:
                tensor = blk[key]
                dim = tensor.shape[-1]
                packed_orders.append(tensor.reshape(*tensor.shape[:-2], seg, self.hidden_channels, dim))
            stacked = torch.stack(packed_orders, dim=-1)
            coeff = self.basis_weight[name].to(dtype=stacked.dtype).unsqueeze(0)
            if idx is not None:
                elem = self.element_weight[name](idx).reshape(stacked.shape[0], self.correlation, self.hidden_channels)
                coeff = coeff + elem.to(dtype=stacked.dtype)
            coeff = coeff * order_gate.unsqueeze(-1)
            mixed = torch.einsum("bshdr,brh->bshd", stacked, coeff) / math.sqrt(self.correlation)
            prefix = mixed.shape[:-3]
            seg = mixed.shape[-3]
            dim = mixed.shape[-1]
            mixed = mixed.reshape(-1, self.hidden_channels, dim)
            mixed = apply_channel_linear(mixed, self.out_mix[name])
            mixed = mixed.reshape(*prefix, seg, self.hidden_channels, dim)
            hidden_out[key] = mixed.reshape(*prefix, seg * self.hidden_channels, dim)
        return self._hidden_o2_to_local(hidden_out)


class SO2ProductBasisBlock(nn.Module):
    def __init__(self, *, channels: int, lmax: int, correlation: int = 3, max_atomvalue: int = 10, contraction_compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.symmetric_contractions = NodeLocalO2SymmetricContraction(
            in_channels=self.channels,
            hidden_channels=self.channels,
            lmax=self.lmax,
            correlation=int(correlation),
            max_atomvalue=int(max_atomvalue),
            contraction_compute_dtype=contraction_compute_dtype,
        )
        self.linear = NodeLocalSO2MixerRect(self.channels, self.channels, self.lmax, gated=False)

    def forward(
        self,
        *,
        atom_types: torch.Tensor,
        message: dict[tuple[int, int], torch.Tensor],
        sc: dict[tuple[int, int], torch.Tensor] | None,
    ) -> dict[tuple[int, int], torch.Tensor]:
        out = self.linear(self.symmetric_contractions(message, atom_types))
        if sc is not None:
            out = {key: out[key] + sc[key] for key in out}
        return out


class SO2ScalarProductBasisBlock(nn.Module):
    def __init__(self, *, channels: int, lmax: int, correlation: int = 3, max_atomvalue: int = 10, contraction_compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.symmetric_contractions = NodeLocalO2SymmetricContraction(
            in_channels=self.channels,
            hidden_channels=self.channels,
            lmax=self.lmax,
            correlation=int(correlation),
            max_atomvalue=int(max_atomvalue),
            contraction_compute_dtype=contraction_compute_dtype,
        )
        self.linear = nn.Linear(self.channels, self.channels, bias=False)

    def forward(
        self,
        *,
        atom_types: torch.Tensor,
        message: dict[tuple[int, int], torch.Tensor],
        sc: dict[tuple[int, int], torch.Tensor] | None,
    ) -> torch.Tensor:
        del atom_types
        contracted = self.symmetric_contractions(message)
        scalar = contracted[(0, 0)].squeeze(-1)
        out = self.linear(scalar)
        if sc is not None:
            out = out + sc[(0, 0)].squeeze(-1)
        return out


class SO2FusionMixBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        lmax: int,
        correlation: int = 3,
        max_atomvalue: int = 10,
        contraction_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.reduce = NodeLocalSO2MixerRect(int(in_channels), int(hidden_channels), int(lmax), gated=False)
        self.symmetric_contractions = NodeLocalO2SymmetricContraction(
            in_channels=int(hidden_channels),
            hidden_channels=int(hidden_channels),
            lmax=int(lmax),
            correlation=int(correlation),
            max_atomvalue=int(max_atomvalue),
            contraction_compute_dtype=contraction_compute_dtype,
        )
        self.project = NodeLocalSO2MixerRect(int(hidden_channels), int(out_channels), int(lmax), gated=False)

    def forward(self, x: dict[tuple[int, int], torch.Tensor], atom_types: torch.Tensor) -> dict[tuple[int, int], torch.Tensor]:
        return self.project(self.symmetric_contractions(self.reduce(x), atom_types))


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
        del ictd_save_tp_mode, ictd_fix_contraction_combine, ictd_fix_product_backend, ictd_fix_interaction_scale
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

        self.local_conv_emb = NodeLocalSO2EmbeddingConv(
            max_radius=max_embed_radius,
            number_of_basis=main_number_of_basis,
            channels=self.channels,
            embedding_dim=embedding_dim,
            max_atomvalue=self.num_elements,
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

        self.interactions = nn.ModuleList(
            [
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
                for _ in range(self.num_interaction - 1)
            ]
        )

        product_target_lmax = [self.lmax if layer_idx < self.num_interaction - 1 else 0 for layer_idx in range(self.num_interaction)]
        self.products = nn.ModuleList()
        for target_lmax in product_target_lmax:
            if target_lmax == self.lmax:
                self.products.append(
                    SO2ProductBasisBlock(
                        channels=self.channels,
                        lmax=self.lmax,
                        correlation=save_contraction_order,
                        max_atomvalue=self.num_elements,
                        contraction_compute_dtype=self.contraction_compute_dtype,
                    )
                )
            else:
                self.products.append(
                    SO2ScalarProductBasisBlock(
                        channels=self.channels,
                        lmax=self.lmax,
                        correlation=save_contraction_order,
                        max_atomvalue=self.num_elements,
                        contraction_compute_dtype=self.contraction_compute_dtype,
                    )
                )

        self.layer_energy_readouts = nn.ModuleList(
            [NodeLocalScalarReadoutSO2(self.channels, self.lmax, output_init_std=0.003) for _ in range(self.num_interaction - 1)]
        )
        self.last_layer_energy_readout = MACEStyleScalarReadoutSO3(self.channels, hidden_channels=16, output_init_std=0.003)

        if self.ictd_fix_route == "fusion":
            hidden_mix = resolve_save_multiple_mix_channels(self.channels, self.num_interaction, save_multiple_mix_channels)
            self.save_multiple_mix_channels = self.channels
            self.multiple_contraction_mix = SO2FusionMixBlock(
                in_channels=self.channels * self.num_interaction,
                hidden_channels=hidden_mix,
                out_channels=self.channels,
                lmax=self.lmax,
                correlation=save_contraction_order,
                max_atomvalue=self.num_elements,
                contraction_compute_dtype=self.contraction_compute_dtype,
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

        local = self.local_conv_emb(
            compact_idx,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_radial=edge_radial,
            node_moments=node_moments,
            neighbor_count=neighbor_count,
            edge_mask=edge_mask,
        )
        local = self.post_local_linears[0](local)
        local = self.products[0](atom_types=compact_idx, message=local, sc=None)
        layer_states = [local]
        total_energy = self.layer_energy_readouts[0](local) if self.num_interaction > 1 else None

        scalar_last = None
        for layer_idx, interaction in enumerate(self.interactions, start=1):
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
                    last_preproduct_state = {k: self.post_local_linears[layer_idx](message[k] + (sc[k] if sc is not None else 0.0)) for k in message}
                    layer_states.append(last_preproduct_state)
                    scalar_last = self.products[layer_idx](atom_types=compact_idx, message=message, sc=sc)
                else:
                    scalar_last = self.products[layer_idx](atom_types=compact_idx, message=message, sc=sc)
            else:
                local = self.products[layer_idx](atom_types=compact_idx, message=message, sc=sc)
                local = self.post_local_linears[layer_idx](local)
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
