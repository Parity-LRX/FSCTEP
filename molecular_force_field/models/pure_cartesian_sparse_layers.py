"""
Pure Cartesian Sparse layers (δ/ε, 3^L) built on top of pure_cartesian.py.

This is NOT the old CG/irreps 'sparse' variant. This is a sparse variant within the
pure Cartesian tensor algebra itself, implemented by restricting rank-rank interaction paths.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from e3nn.math import soft_one_hot_linspace
from molecular_force_field.utils.scatter import scatter

from molecular_force_field.models.mlp import RobustScalarWeightedSum
from molecular_force_field.models.pure_cartesian import (
    PureCartesianTensorProductO3Sparse,
    PureCartesianElementwiseTensorProductO3,
    edge_rank_powers,
    split_by_rank_o3,
    merge_by_rank_o3,
    total_dim_o3,
)


class PureCartesianSparseE3Conv(nn.Module):
    """
    First convolution for pure-cartesian-sparse (neighbor in TP).

      - Build f_in = source scalar ⊗ edge rank powers n^{⊗L}
      - Second operand x2 = neighbor (edge_dst) scalar features (rank0 only), C2=output_size
      - TP(f_in, x2; weights(r)) with C1=output_size, C2=output_size
      - scatter to nodes, normalize by neighbor_count (same as pure_cartesian_layers).
    """

    def __init__(
        self,
        max_radius: float,
        number_of_basis: int,
        channels_out: int,
        embedding_dim: int = 16,
        max_atomvalue: int = 10,
        output_size: int = 8,
        Lmax: int = 2,
        function_type: str = "gaussian",
        max_rank_other: int = 1,
        k_policy: str = "k0",
    ):
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.channels_out = channels_out
        self.output_size = output_size
        self.Lmax = Lmax
        self.function_type = function_type

        self.atom_embedding = nn.Embedding(max_atomvalue, embedding_dim)
        self.atom_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, output_size),
        )

        # Second operand is neighbor scalar (C2=output_size), same as PureCartesianE3Conv.
        self.tp2 = PureCartesianTensorProductO3Sparse(
            C1=output_size,
            C2=output_size,
            Cout=channels_out,
            Lmax=Lmax,
            max_rank_other=max_rank_other,
            allow_epsilon=False,
            k_policy=k_policy,
            share_parity_weights=True,
            assume_pseudo_zero=True,
            internal_weights=False,
        )

        self.fc = nn.Sequential(
            nn.Linear(number_of_basis, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp2.weight_numel),
        )

        self.output_dim = total_dim_o3(channels_out, Lmax)

    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        edge_batch_idx = batch[edge_src]
        edge_cells = cell[edge_batch_idx]
        shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
        edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_vec = edge_vec.to(dtype=dtype)
        edge_length = edge_vec.norm(dim=1)

        Ai = self.atom_mlp(self.atom_embedding(A.long()))
        edge_vec = edge_vec.to(dtype=Ai.dtype)

        e = edge_rank_powers(edge_vec, self.Lmax, normalize=True)

        batch_shape = (edge_src.shape[0],)
        f1_blocks = {(0, 0): Ai[edge_src], (1, 0): torch.zeros_like(Ai[edge_src])}
        for L in range(1, self.Lmax + 1):
            base = Ai[edge_src].view(*batch_shape, self.output_size, *([1] * L)) * e[L].view(*batch_shape, 1, *([3] * L))
            f1_blocks[(0, L)] = base
            f1_blocks[(1, L)] = torch.zeros_like(base)
        f_in = merge_by_rank_o3(f1_blocks, self.output_size, self.Lmax)

        # Second operand: neighbor (edge_dst) scalar, rank0 only, C2=output_size
        x2_blocks = {(0, 0): Ai[edge_dst], (1, 0): torch.zeros_like(Ai[edge_dst])}
        for L in range(1, self.Lmax + 1):
            z = torch.zeros(Ai.size(0), self.output_size, *([3] * L), device=Ai.device, dtype=Ai.dtype)[edge_dst]
            x2_blocks[(0, L)] = z
            x2_blocks[(1, L)] = torch.zeros_like(z)
        x2 = merge_by_rank_o3(x2_blocks, self.output_size, self.Lmax)

        emb = soft_one_hot_linspace(
            edge_length, 0.0, self.max_radius, self.number_of_basis, basis=self.function_type, cutoff=True
        ).mul(self.number_of_basis ** 0.5)
        emb = emb.to(dtype=Ai.dtype)
        weights = self.fc(emb)

        edge_features = self.tp2(f_in, x2, weights)

        num_nodes = pos.size(0)
        neighbor_count = scatter(torch.ones_like(edge_dst), edge_dst, dim=0, dim_size=num_nodes).clamp(min=1).to(edge_features.dtype)
        out = scatter(edge_features, edge_dst, dim=0, dim_size=num_nodes).div(neighbor_count.unsqueeze(-1))
        return out


class PureCartesianSparseE3Conv2(nn.Module):
    """
    Second convolution for pure-cartesian-sparse:
      node rank features x edge rank powers -> node rank features
    """

    def __init__(
        self,
        max_radius: float,
        number_of_basis: int,
        channels_in: int,
        channels_out: int,
        Lmax: int = 2,
        function_type: str = "gaussian",
        max_rank_other: int = 1,
        k_policy: str = "k0",
    ):
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.Lmax = Lmax
        self.function_type = function_type

        # Edge operand has C2=1 and is rank-only geometry; so sparse restriction is natural.
        self.tp = PureCartesianTensorProductO3Sparse(
            C1=channels_in,
            C2=1,
            Cout=channels_out,
            Lmax=Lmax,
            max_rank_other=max_rank_other,
            allow_epsilon=False,
            k_policy=k_policy,
            share_parity_weights=True,
            assume_pseudo_zero=True,
            internal_weights=False,
        )

        self.fc = nn.Sequential(
            nn.Linear(number_of_basis, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp.weight_numel),
        )

        self.output_dim = total_dim_o3(channels_out, Lmax)

    def forward(self, f_in, pos, batch, edge_src, edge_dst, edge_shifts, cell):
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        edge_batch_idx = batch[edge_src]
        edge_cells = cell[edge_batch_idx]
        shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
        edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_vec = edge_vec.to(dtype=f_in.dtype)
        edge_length = edge_vec.norm(dim=1)

        e = edge_rank_powers(edge_vec, self.Lmax, normalize=True)
        e_blocks = {(0, 0): e[0].view(-1, 1), (1, 0): torch.zeros_like(e[0].view(-1, 1))}
        for L in range(1, self.Lmax + 1):
            base = e[L].view(-1, 1, *([3] * L))
            e_blocks[(0, L)] = base
            e_blocks[(1, L)] = torch.zeros_like(base)
        e_flat = merge_by_rank_o3(e_blocks, 1, self.Lmax)

        emb = soft_one_hot_linspace(
            edge_length, 0.0, self.max_radius, self.number_of_basis, basis=self.function_type, cutoff=True
        ).mul(self.number_of_basis ** 0.5)
        emb = emb.to(dtype=f_in.dtype)
        weights = self.fc(emb)

        num_nodes = pos.size(0)
        edge_features = self.tp(f_in[edge_src], e_flat, weights)
        out = scatter(edge_features, edge_dst, dim=0, dim_size=num_nodes, reduce="sum")
        avg_num_neighbors = float(edge_src.numel()) / float(max(num_nodes, 1))
        out = out / max(avg_num_neighbors, 1e-8)
        return out


class PureCartesianSparseInvariantBilinear(nn.Module):
    """
    Same invariant readout as pure_cartesian_layers, but kept local to avoid cross-import churn.
    """

    def __init__(self, channels: int, out_channels: int, Lmax: int):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.Lmax = Lmax

        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(out_channels, channels, channels) * 0.02)
            for _ in range(Lmax + 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = split_by_rank_o3(x, self.channels, self.Lmax)
        outs = []
        for L in range(self.Lmax + 1):
            t0 = blocks[(0, L)]
            t1 = blocks[(1, L)]
            t0_flat = t0.reshape(t0.shape[0], self.channels, -1)
            t1_flat = t1.reshape(t1.shape[0], self.channels, -1)
            # e3nn-style component normalization: divide by sqrt(3^L)
            # Use math.sqrt for better numerical precision than 3**(L/2)
            scale = 1.0 / math.sqrt(3 ** L) if L > 0 else 1.0
            gram0 = torch.einsum("nci,ndi->ncd", t0_flat, t0_flat) * scale
            gram1 = torch.einsum("nci,ndi->ncd", t1_flat, t1_flat) * scale
            gram = gram0 + gram1
            outs.append(torch.einsum("ocd,ncd->no", self.W[L], gram))
        return sum(outs)


class PureCartesianSparseTransformerLayer(nn.Module):
    """
    Full pure-cartesian-sparse transformer layer.

    This mirrors PureCartesianTransformerLayer, but swaps the dense TP with the sparse TP
    by restricting allowed rank-rank interactions (typically to scalar/vector couplings).
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
        output_size: int = 8,
        embed_size=None,
        main_hidden_sizes3=None,
        num_layers: int = 1,
        num_interaction: int = 2,
        device=None,
        function_type_main: str = "gaussian",
        lmax: int = 2,
        max_rank_other: int = 1,
        k_policy: str = "k0",
    ):
        super().__init__()
        if embed_size is None:
            embed_size = [128, 128, 128]
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.Lmax = lmax
        self.channels = hidden_dim_conv
        self.feature_dim = total_dim_o3(self.channels, self.Lmax)

        self.num_interaction = int(num_interaction)
        if self.num_interaction < 2:
            raise ValueError(f"num_interaction must be >= 2, got {self.num_interaction}")

        self.e3_conv_layers = nn.ModuleList()
        self.e3_conv_layers.append(
            PureCartesianSparseE3Conv(
                max_radius=max_embed_radius,
                number_of_basis=main_number_of_basis,
                channels_out=self.channels,
                embedding_dim=embedding_dim,
                max_atomvalue=max_atomvalue,
                output_size=output_size,
                Lmax=self.Lmax,
                function_type=function_type_main,
                max_rank_other=max_rank_other,
                k_policy=k_policy,
            )
        )
        for _ in range(1, self.num_interaction):
            self.e3_conv_layers.append(
                PureCartesianSparseE3Conv2(
                    max_radius=max_embed_radius,
                    number_of_basis=main_number_of_basis,
                    channels_in=self.channels,
                    channels_out=self.channels,
                    Lmax=self.Lmax,
                    function_type=function_type_main,
                    max_rank_other=max_rank_other,
                    k_policy=k_policy,
                )
            )

        self.combined_channels = self.channels * self.num_interaction
        self.combined_dim = total_dim_o3(self.combined_channels, self.Lmax)

        scalar_channels = (self.num_interaction - 1) * 32
        self.product_3 = PureCartesianSparseInvariantBilinear(
            channels=self.combined_channels,
            out_channels=scalar_channels,
            Lmax=self.Lmax,
        )

        # Match e3nn-style product_5: invariants from T=cat([f1..fn,f_prod3]) via elementwise 0e product
        self.product_5_o3 = PureCartesianElementwiseTensorProductO3(channels=self.channels, Lmax=self.Lmax)
        from molecular_force_field.models.mlp import MainNet  # local import to avoid unused at module top
        self.proj_total = MainNet(
            self.num_interaction * self.product_5_o3.dim_out + scalar_channels,
            embed_size,
            17,
        )
        self.weighted_sum = RobustScalarWeightedSum(17, init_weights="zero")

    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        sort_idx = torch.argsort(edge_dst)
        edge_src = edge_src[sort_idx]
        edge_dst = edge_dst[sort_idx]
        edge_shifts = edge_shifts[sort_idx]

        features = []
        f_prev = self.e3_conv_layers[0](pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        features.append(f_prev)
        for conv in self.e3_conv_layers[1:]:
            f_prev = conv(f_prev, pos, batch, edge_src, edge_dst, edge_shifts, cell)
            features.append(f_prev)

        # Combine features correctly: split by rank, merge true/pseudo separately
        # f1..fn are each (N, channels * sum 3^L) with format [true, pseudo]
        # We need (N, n*channels * sum 3^L) with format [combined_true, combined_pseudo]
        blocks_list = [split_by_rank_o3(f, self.channels, self.Lmax) for f in features]
        combined_blocks = {}
        for s in (0, 1):
            for L in range(self.Lmax + 1):
                # Concatenate per-layer blocks along channel dimension
                # For L=0: shape is (N, channels), cat on dim=1 -> (N, n*channels)
                # For L>0: shape is (N, channels, 3,...), cat on dim=1 -> (N, n*channels, 3,...)
                combined_blocks[(s, L)] = torch.cat(
                    [blocks[(s, L)] for blocks in blocks_list],
                    dim=1,
                )
        f_combine = merge_by_rank_o3(combined_blocks, self.combined_channels, self.Lmax)  # (N, combined_dim)

        f_prod3 = self.product_3(f_combine)

        invs = [self.product_5_o3(f, f) for f in features]
        inv3 = f_prod3 * f_prod3
        f_prod5 = torch.cat(invs + [inv3], dim=-1)

        product_proj = self.proj_total(f_prod5)
        e_out = self.weighted_sum(product_proj)
        return e_out.sum(dim=-1, keepdim=True)


__all__ = ["PureCartesianSparseTransformerLayer"]

