from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import nn as e3nn_nn
from e3nn import o3
from e3nn.math import soft_one_hot_linspace

from molecular_force_field.models._mace_symmetric_contraction import (
    MaceSymmetricContraction,
)
from molecular_force_field.utils.scatter import scatter


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, *, dim_size: int) -> torch.Tensor:
    return scatter(src, index, dim=0, dim_size=dim_size, reduce="sum")


def _hidden_irreps(channels: int, lmax: int) -> o3.Irreps:
    return o3.Irreps(" + ".join(f"{int(channels)}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(int(lmax) + 1)))


def _coupling_dim(lmax: int) -> int:
    return sum(2 * l + 1 for l in range(int(lmax) + 1))


def _flat_to_mace_features(x: torch.Tensor, channels: int, lmax: int) -> torch.Tensor:
    parts = []
    offset = 0
    for l in range(int(lmax) + 1):
        width = int(channels) * (2 * l + 1)
        parts.append(x[:, offset : offset + width].reshape(x.size(0), int(channels), 2 * l + 1))
        offset += width
    return torch.cat(parts, dim=-1)


def _scalar_from_hidden(x: torch.Tensor, channels: int) -> torch.Tensor:
    return x[:, : int(channels)]


def _build_initial_state(scalars: torch.Tensor, channels: int, lmax: int) -> torch.Tensor:
    parts = [scalars]
    for l in range(1, int(lmax) + 1):
        parts.append(scalars.new_zeros((scalars.size(0), int(channels) * (2 * l + 1))))
    return torch.cat(parts, dim=-1)


def _node_type_indices(node_attrs: torch.Tensor) -> torch.Tensor:
    if node_attrs.dim() == 1:
        return node_attrs.long()
    return node_attrs.argmax(dim=-1).long()


def _tp_out_irreps_with_instructions(
    irreps1: o3.Irreps,
    irreps2: o3.Irreps,
    target_irreps: o3.Irreps,
) -> Tuple[o3.Irreps, List[Tuple[int, int, int, str, bool]]]:
    trainable = True
    irreps_out_list: List[Tuple[int, o3.Irrep]] = []
    instructions: List[Tuple[int, int, int, str, bool]] = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:
                if ir_out in target_irreps:
                    k = len(irreps_out_list)
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))
    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()
    instructions = [(i1, i2, permut[i_out], mode, train) for i1, i2, i_out, mode, train in instructions]
    instructions = sorted(instructions, key=lambda x: x[2])
    return irreps_out, instructions


class SphericalResidualInteractionBlock(nn.Module):
    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        input_irreps: o3.Irreps,
        edge_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        sc_irreps: o3.Irreps,
        number_of_basis: int,
        function_type: str,
        avg_num_neighbors: float | None,
        use_self_connection: bool,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.number_of_basis = int(number_of_basis)
        self.function_type = str(function_type)
        self.avg_num_neighbors = None if avg_num_neighbors is None else float(avg_num_neighbors)
        self.use_self_connection = bool(use_self_connection)
        self.hidden_irreps = _hidden_irreps(self.channels, self.lmax)
        self.input_irreps = o3.Irreps(input_irreps)
        self.edge_irreps = o3.Irreps(edge_irreps)
        self.target_irreps = o3.Irreps(target_irreps)
        self.sc_irreps = o3.Irreps(sc_irreps)
        self.edge_attrs_irreps = o3.Irreps.spherical_harmonics(self.lmax)
        self.node_attrs_irreps = o3.Irreps(f"{int(num_elements)}x0e")

        self.linear_up = o3.Linear(self.input_irreps, self.edge_irreps)
        irreps_mid, instructions = _tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.conv_tp_weights = e3nn_nn.FullyConnectedNet(
            [self.number_of_basis, 64, 64, 64, self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )
        self.linear = o3.Linear(irreps_mid, self.target_irreps)
        if self.use_self_connection:
            # MACE RealAgnosticResidualInteractionBlock:
            # sc = FullyConnectedTensorProduct(node_feats, node_attrs)
            self.skip_tp = o3.FullyConnectedTensorProduct(
                self.input_irreps,
                self.node_attrs_irreps,
                self.sc_irreps,
            )
        else:
            # MACE first interaction block:
            # message = FullyConnectedTensorProduct(message, node_attrs)
            self.skip_tp = o3.FullyConnectedTensorProduct(
                self.target_irreps,
                self.node_attrs_irreps,
                self.target_irreps,
            )

    def forward(
        self,
        *,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        sync_after_scatter=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        num_nodes = node_feats.size(0)

        sc = self.skip_tp(node_feats, node_attrs) if self.use_self_connection else None
        up = self.linear_up(node_feats)
        weights = self.conv_tp_weights(edge_feats)
        edge_message = self.conv_tp(up[edge_src], edge_attrs, weights)
        if edge_mask is not None:
            edge_message = edge_message * edge_mask.to(dtype=edge_message.dtype)
        message = _scatter_sum(edge_message, edge_dst, dim_size=num_nodes)
        if self.avg_num_neighbors is None:
            if edge_mask is not None:
                avg_num_neighbors = float(edge_mask.detach().sum().item()) / float(max(num_nodes, 1))
            else:
                avg_num_neighbors = float(edge_src.numel()) / float(max(num_nodes, 1))
        else:
            avg_num_neighbors = self.avg_num_neighbors
        message = self.linear(message) / max(avg_num_neighbors, 1e-8)
        if sync_after_scatter is not None:
            message = sync_after_scatter(message)
        if not self.use_self_connection:
            message = self.skip_tp(message, node_attrs)
        return message, sc


class NativeMACEProductBasisBlock(nn.Module):
    def __init__(
        self,
        *,
        num_elements: int,
        channels: int,
        lmax: int,
        target_irreps: o3.Irreps,
        correlation: int,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.hidden_irreps = _hidden_irreps(self.channels, self.lmax)
        self.target_irreps = o3.Irreps(target_irreps)
        self.symmetric_contractions = MaceSymmetricContraction(
            irreps_in=self.hidden_irreps,
            irreps_out=self.target_irreps,
            correlation=int(correlation),
            num_elements=int(num_elements),
            use_reduced_cg=False,
        )
        self.linear = o3.Linear(self.target_irreps, self.target_irreps)

    def forward(self, node_feats: torch.Tensor, sc: torch.Tensor | None, node_attrs: torch.Tensor) -> torch.Tensor:
        x = _flat_to_mace_features(node_feats, self.channels, self.lmax)
        out = self.linear(self.symmetric_contractions(x, node_attrs))
        if sc is not None:
            if sc.shape[-1] == out.shape[-1]:
                out = out + sc
            else:
                out = out + _scalar_from_hidden(sc, self.channels)
        return out


class SphericalScalarMLPReadout(nn.Module):
    def __init__(self, channels: int, output_init_std: float = 0.003):
        super().__init__()
        self.linear_1 = o3.Linear(f"{int(channels)}x0e", "16x0e")
        self.activation = nn.SiLU()
        self.linear_2 = o3.Linear("16x0e", "1x0e")
        with torch.no_grad():
            if hasattr(self.linear_2, "weight") and self.linear_2.weight is not None:
                self.linear_2.weight.normal_(mean=0.0, std=float(output_init_std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.activation(self.linear_1(x)))


class SphericalFix(nn.Module):
    """
    e3nn/native-MACE counterpart of pure-cartesian-ictd-fix baseline.
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
        ictd_fix_route: str = "baseline",
        save_contraction_order: int = 3,
        avg_num_neighbors: float | None = None,
        **_unused,
    ):
        super().__init__()
        del main_max_radius, hidden_dim_sh, hidden_dim, channel_in2, embedding_dim, output_size, embed_size, main_hidden_sizes3, num_layers
        long_range_mode = _unused.get("long_range_mode", "none")
        feature_spectral_mode = _unused.get("feature_spectral_mode", "none")
        if long_range_mode != "none":
            raise NotImplementedError("spherical-fix v1 does not support long_range_mode != 'none'")
        if feature_spectral_mode != "none":
            raise NotImplementedError("spherical-fix v1 does not support feature_spectral_mode != 'none'")
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if int(num_interaction) < 2:
            raise ValueError(f"num_interaction must be >= 2, got {num_interaction}")
        if ictd_fix_route != "baseline":
            raise NotImplementedError("spherical-fix v1 supports only ictd_fix_route='baseline'")

        self.channels = int(hidden_dim_conv)
        self.lmax = int(lmax)
        self.num_interaction = int(num_interaction)
        self.max_radius = float(max_embed_radius)
        self.number_of_basis = int(main_number_of_basis)
        self.function_type = str(function_type_main)
        self.avg_num_neighbors = None if avg_num_neighbors is None else float(avg_num_neighbors)
        self.hidden_irreps = _hidden_irreps(self.channels, self.lmax)
        self.scalar_irreps = o3.Irreps(f"{self.channels}x0e")
        self.edge_irreps = o3.Irreps.spherical_harmonics(self.lmax)
        self.coupling_dim = _coupling_dim(self.lmax)

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

        self.node_embedding = nn.Linear(self.num_elements, self.channels, bias=False)
        product_target_irreps = [
            self.hidden_irreps if layer_idx < self.num_interaction - 1 else self.scalar_irreps
            for layer_idx in range(self.num_interaction)
        ]
        self.interactions = nn.ModuleList()
        self.products = nn.ModuleList()
        for layer_idx, product_target in enumerate(product_target_irreps):
            input_irreps = self.scalar_irreps if layer_idx == 0 else self.hidden_irreps
            self.interactions.append(
                SphericalResidualInteractionBlock(
                    num_elements=self.num_elements,
                    channels=self.channels,
                    lmax=self.lmax,
                    input_irreps=input_irreps,
                    edge_irreps=input_irreps,
                    target_irreps=self.hidden_irreps,
                    sc_irreps=product_target,
                    number_of_basis=self.number_of_basis,
                    function_type=self.function_type,
                    avg_num_neighbors=self.avg_num_neighbors,
                    use_self_connection=(layer_idx > 0),
                )
            )
            self.products.append(
                NativeMACEProductBasisBlock(
                    num_elements=self.num_elements,
                    channels=self.channels,
                    lmax=self.lmax,
                    target_irreps=product_target,
                    correlation=save_contraction_order,
                )
            )
        self.layer_energy_readouts = nn.ModuleList(
            [o3.Linear(self.hidden_irreps, "1x0e") for _ in range(self.num_interaction - 1)]
        )
        self.last_layer_energy_readout = SphericalScalarMLPReadout(self.channels)

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
        sync_after_scatter=None,
        return_physical_tensors: bool = False,
        return_reciprocal_source: bool = False,
    ):
        if return_physical_tensors:
            raise ValueError("spherical-fix does not currently support return_physical_tensors=True")
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        sort_idx = torch.argsort(edge_dst)
        edge_src = edge_src[sort_idx]
        edge_dst = edge_dst[sort_idx]
        edge_shifts = edge_shifts[sort_idx]
        edge_index = torch.stack([edge_src, edge_dst], dim=0)

        if precomputed_edge_vec is not None:
            edge_vec = precomputed_edge_vec[sort_idx].to(dtype=dtype)
        else:
            edge_batch_idx = batch[edge_src]
            edge_cells = cell[edge_batch_idx]
            shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
            edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs

        edge_length = edge_vec.norm(dim=1)
        edge_mask = (edge_length <= self.max_radius).to(dtype=dtype).unsqueeze(-1)
        edge_attrs = o3.spherical_harmonics(
            self.edge_irreps,
            edge_vec,
            normalize=True,
            normalization="component",
        ).to(dtype=dtype)
        edge_feats = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis=self.function_type,
            cutoff=True,
        ).mul(self.number_of_basis ** 0.5).to(dtype=dtype)

        A_long = A.long()
        if int(A_long.max().item()) >= self.atomic_number_to_index.numel():
            raise ValueError(
                f"Encountered atomic number {int(A_long.max().item())}, but compact mapping supports only up to "
                f"{self.atomic_number_to_index.numel() - 1}. atomic_numbers={self.atomic_numbers}"
            )
        compact_idx = self.atomic_number_to_index[A_long]
        if torch.any(compact_idx < 0):
            bad = torch.unique(A_long[compact_idx < 0]).tolist()
            raise ValueError(f"Encountered atomic numbers without compact mapping: {bad}.")
        node_attrs = F.one_hot(compact_idx, num_classes=self.num_elements).to(dtype=dtype)
        h = self.node_embedding(node_attrs)

        layer_states: List[torch.Tensor] = []
        total_energy = None
        for layer_idx, (interaction, product) in enumerate(zip(self.interactions, self.products)):
            message, sc = interaction(
                node_attrs=node_attrs,
                node_feats=h,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                edge_mask=edge_mask,
                sync_after_scatter=sync_after_scatter,
            )
            h = product(message, sc, node_attrs)
            layer_states.append(h)
            if layer_idx < self.num_interaction - 1:
                e_layer = self.layer_energy_readouts[layer_idx](h)
            else:
                e_layer = self.last_layer_energy_readout(h)
            total_energy = e_layer if total_energy is None else total_energy + e_layer

        out = total_energy.sum(dim=-1, keepdim=True)
        if return_combined_features:
            combined = torch.cat(layer_states, dim=-1)
            if return_reciprocal_source:
                return out, combined, out.new_empty((out.size(0), 0))
            return out, combined
        if return_reciprocal_source:
            return out, out.new_empty((out.size(0), 0))
        return out
