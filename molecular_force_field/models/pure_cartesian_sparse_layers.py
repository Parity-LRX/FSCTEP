"""
Pure Cartesian Sparse layers (δ/ε, 3^L) built on top of pure_cartesian.py.

This is NOT the old CG/irreps 'sparse' variant. This is a sparse variant within the
pure Cartesian tensor algebra itself, implemented by restricting rank-rank interaction paths.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
from e3nn.math import soft_one_hot_linspace
from molecular_force_field.utils.scatter import scatter

from molecular_force_field.models.mlp import RobustScalarWeightedSum
from molecular_force_field.models.long_range import apply_long_range_modules, configure_long_range_modules
from molecular_force_field.models.pure_cartesian import (
    PureCartesianTensorProductO3Sparse,
    PureCartesianElementwiseTensorProductO3,
    edge_rank_powers,
    split_by_rank_o3,
    merge_by_rank_o3,
    total_dim_o3,
)


def _apply_channel_adapter_cartesian(x_l: torch.Tensor, adapter: nn.Module) -> torch.Tensor:
    """Apply a channel-only adapter to a Cartesian rank block."""
    if isinstance(adapter, nn.Identity):
        return x_l
    if x_l.dim() == 2:
        return adapter(x_l)
    batch_size = x_l.shape[0]
    spatial_shape = x_l.shape[2:]
    y = adapter(x_l.reshape(batch_size, x_l.shape[1], -1).transpose(1, 2)).transpose(1, 2)
    return y.reshape(batch_size, y.shape[1], *spatial_shape)


def _canonicalize_rank2_external_tensor(external_tensor: torch.Tensor) -> torch.Tensor:
    if external_tensor.shape[-1:] == (6,):
        xx, yy, zz, xy, xz, yz = external_tensor.unbind(dim=-1)
        row0 = torch.stack((xx, xy, xz), dim=-1)
        row1 = torch.stack((xy, yy, yz), dim=-1)
        row2 = torch.stack((xz, yz, zz), dim=-1)
        external_tensor = torch.stack((row0, row1, row2), dim=-2)
    elif external_tensor.shape[-1:] == (9,):
        external_tensor = external_tensor.reshape(*external_tensor.shape[:-1], 3, 3)
    elif external_tensor.shape[-2:] != (3, 3):
        raise ValueError(
            "rank-2 external_tensor must have shape (..., 3, 3), (..., 9), or (..., 6)"
        )
    return 0.5 * (external_tensor + external_tensor.transpose(-1, -2))


def build_cartesian_physical_tensor_blocks(
    tensor: torch.Tensor,
    *,
    rank: int,
    include_trace_chain: bool,
) -> Dict[int, torch.Tensor]:
    """Convert Cartesian tensors to rank-keyed full-Cartesian blocks."""
    if rank == 0:
        if tensor.shape[-1:] == (1,):
            tensor = tensor[..., 0]
        return {0: tensor.unsqueeze(-1).unsqueeze(-1)}
    if rank == 1:
        if tensor.shape[-1] != 3:
            raise ValueError(f"rank-1 Cartesian tensor must have trailing dim 3, got {tuple(tensor.shape)}")
        return {1: tensor.unsqueeze(-2)}
    if rank == 2:
        mat = _canonicalize_rank2_external_tensor(tensor)
        blocks: Dict[int, torch.Tensor] = {}
        if include_trace_chain:
            trace = mat.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
            blocks[0] = trace.unsqueeze(-1).unsqueeze(-1)
            eye = torch.eye(3, device=mat.device, dtype=mat.dtype)
            mat = mat - trace.unsqueeze(-1).unsqueeze(-1) * eye
        blocks[2] = mat.unsqueeze(-3)
        return blocks
    raise ValueError(f"Unsupported Cartesian physical tensor rank={rank}")


def normalize_sparse_external_tensor_blocks(
    external_tensor: torch.Tensor,
    *,
    external_tensor_rank: int,
    num_nodes: int,
    batch: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[int, torch.Tensor]:
    blocks = build_cartesian_physical_tensor_blocks(
        external_tensor.to(device=device, dtype=dtype),
        rank=external_tensor_rank,
        include_trace_chain=(external_tensor_rank >= 2),
    )
    num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    out: Dict[int, torch.Tensor] = {}
    for l, t in blocks.items():
        if t.dim() == l + 1:
            t = t.unsqueeze(0)
        if t.dim() != l + 2:
            raise ValueError(
                f"external_tensor block l={l} must have shape (..., 1, 3,...); got {tuple(t.shape)}"
            )
        if t.shape[-(l + 1)] != 1:
            raise ValueError(f"external_tensor block l={l} must have a singleton channel axis")
        if num_graphs > 0 and t.shape[0] == num_graphs:
            t = t[batch]
        elif t.shape[0] == 1 and num_nodes > 1:
            t = t.expand(num_nodes, *t.shape[1:])
        elif t.shape[0] != num_nodes:
            raise ValueError(
                f"external_tensor block l={l} first dim must be N={num_nodes} (or 1 / num_graphs), got {t.shape[0]}"
            )
        out[l] = t.contiguous()
    return out


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
        external_tensor_rank: int | None = None,
        external_tensor_scale_init: float = 0.0,
    ):
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.channels_out = channels_out
        self.output_size = output_size
        self.Lmax = Lmax
        self.function_type = function_type
        self.external_tensor_rank = int(external_tensor_rank) if external_tensor_rank is not None else None

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
        if self.external_tensor_rank is not None:
            self.external_tensor_scale_by_l = nn.Parameter(
                torch.full((self.Lmax + 1,), float(external_tensor_scale_init))
            )
        else:
            self.external_tensor_scale_by_l = None

    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, *, external_tensor: torch.Tensor | None = None):
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

        ext_blocks = None
        if external_tensor is not None:
            if self.external_tensor_rank is None:
                raise ValueError(
                    "external_tensor was provided but pure-cartesian-sparse was not configured with external_tensor_rank"
                )
            ext_blocks = normalize_sparse_external_tensor_blocks(
                external_tensor,
                external_tensor_rank=self.external_tensor_rank,
                num_nodes=Ai.shape[0],
                batch=batch,
                device=Ai.device,
                dtype=Ai.dtype,
            )

        e = edge_rank_powers(edge_vec, self.Lmax, normalize=True)

        batch_shape = (edge_src.shape[0],)
        f1_blocks = {(0, 0): Ai[edge_src], (1, 0): torch.zeros_like(Ai[edge_src])}
        for L in range(1, self.Lmax + 1):
            base = Ai[edge_src].view(*batch_shape, self.output_size, *([1] * L)) * e[L].view(*batch_shape, 1, *([3] * L))
            f1_blocks[(0, L)] = base
            f1_blocks[(1, L)] = torch.zeros_like(base)
        if ext_blocks is not None:
            for L, t in ext_blocks.items():
                if L > self.Lmax:
                    continue
                t_e = t[edge_src]
                t_e = t_e * Ai[edge_src].view(*batch_shape, self.output_size, *([1] * L))
                scale = self.external_tensor_scale_by_l[L] if self.external_tensor_scale_by_l is not None else 1.0
                f1_blocks[(0, L)] = f1_blocks[(0, L)] + t_e * scale
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
        num_edges_t = edge_features.new_ones(edge_src.shape[0]).sum()
        num_nodes_t = out.new_ones(out.shape[0]).sum().clamp_min(1.0)
        avg_num_neighbors = (num_edges_t / num_nodes_t).clamp_min(1e-8)
        out = out / avg_num_neighbors
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
        invariant_channels: int = 32,
        embed_size=None,
        main_hidden_sizes3=None,
        num_layers: int = 1,
        num_interaction: int = 2,
        device=None,
        function_type_main: str = "gaussian",
        lmax: int = 2,
        max_rank_other: int = 1,
        k_policy: str = "k0",
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
        feature_spectral_gate_init: float = 0.0,
        physical_tensor_outputs: dict[str, dict] | None = None,
        external_tensor_rank: int | None = None,
        external_tensor_scale_init: float = 0.0,
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
        self.invariant_channels = int(invariant_channels)
        self.max_rank_other = int(max_rank_other)
        self.k_policy = str(k_policy)
        self.lmax = int(lmax)
        self.external_tensor_rank = int(external_tensor_rank) if external_tensor_rank is not None else None
        self.physical_tensor_representation = "cartesian"

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
                max_rank_other=self.max_rank_other,
                k_policy=self.k_policy,
                external_tensor_rank=self.external_tensor_rank,
                external_tensor_scale_init=external_tensor_scale_init,
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
                    max_rank_other=self.max_rank_other,
                    k_policy=self.k_policy,
                )
            )

        self.combined_channels = self.channels * self.num_interaction
        self.combined_dim = total_dim_o3(self.combined_channels, self.Lmax)

        scalar_channels = (self.num_interaction - 1) * self.invariant_channels
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
        configure_long_range_modules(
            self,
            feature_dim=self.num_interaction * self.product_5_o3.dim_out + scalar_channels,
            cutoff_radius=max_embed_radius,
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
            feature_spectral_gate_init=feature_spectral_gate_init,
        )
        self.e3_conv_emb = self.e3_conv_layers[0]

        self._physical_tensor_specs: dict[str, dict] | None = None
        self.physical_tensor_heads: nn.ModuleDict | None = None
        if physical_tensor_outputs is not None:
            self._physical_tensor_specs = {}
            heads = nn.ModuleDict()
            for name, spec_in in physical_tensor_outputs.items():
                if not isinstance(spec_in, dict):
                    raise ValueError(f"physical_tensor_outputs[{name!r}] must be a dict, got {type(spec_in)}")
                spec = dict(spec_in)
                ls = spec.get("ls")
                if ls is None:
                    raise ValueError(f"physical_tensor_outputs[{name!r}] missing 'ls'")
                ls = [int(l) for l in ls]
                for l in ls:
                    if not (0 <= l <= self.Lmax):
                        raise ValueError(f"physical_tensor_outputs[{name!r}] l={l} out of range 0..lmax={self.Lmax}")
                reduce = str(spec.get("reduce", "sum")).strip().lower()
                if reduce not in ("sum", "mean", "none"):
                    raise ValueError(f"physical_tensor_outputs[{name!r}] reduce must be 'sum'|'mean'|'none', got {reduce!r}")
                channels_out = spec.get("channels_out", 1)
                if isinstance(channels_out, int):
                    ch_out_by_l = {l: int(channels_out) for l in ls}
                elif isinstance(channels_out, dict):
                    ch_out_by_l = {int(k): int(v) for k, v in channels_out.items()}
                    for l in ls:
                        if l not in ch_out_by_l:
                            raise ValueError(f"physical_tensor_outputs[{name!r}] channels_out missing l={l}")
                else:
                    raise ValueError(f"physical_tensor_outputs[{name!r}] channels_out must be int or dict, got {type(channels_out)}")
                per_l = nn.ModuleDict()
                for l in ls:
                    if ch_out_by_l[l] <= 0:
                        raise ValueError(f"physical_tensor_outputs[{name!r}] channels_out[{l}] must be positive")
                    if ch_out_by_l[l] == self.combined_channels:
                        per_l[str(l)] = nn.Identity()
                    else:
                        per_l[str(l)] = nn.Linear(self.combined_channels, ch_out_by_l[l], bias=False)
                heads[name] = per_l
                self._physical_tensor_specs[name] = {"ls": ls, "reduce": reduce, "channels_out": ch_out_by_l}
            self.physical_tensor_heads = heads

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
        external_tensor: torch.Tensor | None = None,
        return_physical_tensors: bool = False,
        return_reciprocal_source: bool = False,
        **_unused,
    ):
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        sort_idx = torch.argsort(edge_dst)
        edge_src = edge_src[sort_idx]
        edge_dst = edge_dst[sort_idx]
        edge_shifts = edge_shifts[sort_idx]

        features = []
        f_prev = self.e3_conv_layers[0](
            pos,
            A,
            batch,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
            external_tensor=external_tensor,
        )
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

        physical_out: dict[str, dict[int, torch.Tensor]] | None = None
        if return_physical_tensors:
            if self.physical_tensor_heads is None or self._physical_tensor_specs is None:
                raise ValueError("return_physical_tensors=True but physical_tensor_outputs was not set in __init__")
            physical_out = {}
            true_blocks = {(0, L): combined_blocks[(0, L)] for L in range(self.Lmax + 1)}
            num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
            counts = None
            for name, per_l in self.physical_tensor_heads.items():
                spec = self._physical_tensor_specs[name]
                out_blocks: dict[int, torch.Tensor] = {}
                for l in spec["ls"]:
                    y_l = _apply_channel_adapter_cartesian(true_blocks[(0, l)], per_l[str(l)])
                    if l == 0:
                        y_l = y_l.unsqueeze(-1)
                    if spec["reduce"] == "none":
                        out_blocks[l] = y_l
                    else:
                        yg = scatter(y_l, batch, dim=0, dim_size=num_graphs, reduce="sum")
                        if spec["reduce"] == "mean":
                            if counts is None:
                                counts = scatter(
                                    torch.ones_like(batch, dtype=yg.dtype),
                                    batch,
                                    dim=0,
                                    dim_size=num_graphs,
                                    reduce="sum",
                                ).clamp(min=1.0)
                            view_shape = (counts.shape[0],) + (1,) * (yg.dim() - 1)
                            yg = yg / counts.view(view_shape)
                        out_blocks[l] = yg
                physical_out[name] = out_blocks

        f_prod3 = self.product_3(f_combine)

        invs = [self.product_5_o3(f, f) for f in features]
        inv3 = f_prod3 * f_prod3
        f_prod5 = torch.cat(invs + [inv3], dim=-1)
        f_prod5, long_range_energy, reciprocal_source, defer_long_range_to_runtime = apply_long_range_modules(
            self,
            f_prod5,
            pos,
            batch,
            cell,
            edge_src=edge_src,
            edge_dst=edge_dst,
            return_reciprocal_source=return_reciprocal_source,
        )

        product_proj = self.proj_total(f_prod5)
        e_out = self.weighted_sum(product_proj)
        atom_energies = e_out.sum(dim=-1, keepdim=True)
        if long_range_energy is not None and not defer_long_range_to_runtime:
            atom_energies = atom_energies + long_range_energy
        if reciprocal_source is None and return_reciprocal_source:
            reciprocal_source = atom_energies.new_empty((atom_energies.size(0), 0))
        if return_physical_tensors and return_reciprocal_source:
            assert physical_out is not None
            return atom_energies, physical_out, reciprocal_source
        if return_physical_tensors:
            assert physical_out is not None
            return atom_energies, physical_out
        if return_reciprocal_source:
            return atom_energies, reciprocal_source
        return atom_energies


__all__ = [
    "PureCartesianSparseTransformerLayer",
    "build_cartesian_physical_tensor_blocks",
    "normalize_sparse_external_tensor_blocks",
]
