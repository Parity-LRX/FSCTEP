"""
Pure-Cartesian ICTD-IRREPS layers (internal representation is irreps, NOT 3^L).

Goal (speed):
  - Replace internal 3^L rank-tensor features with an irreps representation of size
    sum_{l=0..lmax}(2l+1) = (lmax+1)^2 per channel.
  - Compute edge direction features WITHOUT spherical harmonics:
      use harmonic polynomials / ICTD basis (see `ictd_irreps.direction_harmonics`).
  - Perform tensor products in the SAME harmonic basis using coupling tensors computed
    by polynomial multiplication + trace-chain projection (see `ictd_irreps`).

This provides an SO(3)-equivariant irreps message passing stack without ever materializing
3^L tensors in the forward pass.

torch.compile: Supported for inference when sync_after_scatter=None (e.g. single-GPU
or validation). Use torch.compile(module, mode="reduce-overhead", dynamic=False) for
best results. torch_scatter may introduce graph breaks; training (double backward) is
not compiled.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn.math import soft_one_hot_linspace
import math

from molecular_force_field.models.pure_cartesian_ictd_layers import PhysicalTensorICTDEmbedding
from molecular_force_field.models.long_range import build_feature_spectral_module, build_long_range_module
from molecular_force_field.models.ictd_irreps import (
    HarmonicElementwiseProduct,
    HarmonicFullyConnectedTensorProduct,
    direction_harmonics,
    direction_harmonics_all,
)
from molecular_force_field.models.mlp import RobustScalarWeightedSum
from molecular_force_field.models.mlp import MainNet


def _irreps_total_dim(channels: int, lmax: int) -> int:
    return channels * (lmax + 1) ** 2


def _split_irreps(x: torch.Tensor, channels: int, lmax: int) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    idx = 0
    for l in range(lmax + 1):
        d = channels * (2 * l + 1)
        blk = x[..., idx: idx + d]
        idx += d
        out[l] = blk.view(*x.shape[:-1], channels, 2 * l + 1)
    return out


def _merge_irreps(blocks: dict[int, torch.Tensor], channels: int, lmax: int) -> torch.Tensor:
    parts = []
    for l in range(lmax + 1):
        parts.append(blocks[l].reshape(*blocks[l].shape[:-2], channels * (2 * l + 1)))
    return torch.cat(parts, dim=-1)

def _irreps_elementwise_tensor_product_0e(x1: torch.Tensor, x2: torch.Tensor, channels: int, lmax: int) -> torch.Tensor:
    """
    Irreps analogue of e3nn ElementwiseTensorProduct filtered to 0e:
    for each l-block, contract over m (angular index) to produce one scalar per channel.

    x: (..., channels*(lmax+1)^2) arranged as concat over l=0..lmax of (channels, 2l+1).
    Returns: (..., channels*(lmax+1)) arranged as concat over l=0..lmax of (channels).
    """
    b1 = _split_irreps(x1, channels, lmax)
    b2 = _split_irreps(x2, channels, lmax)
    outs = []
    for l in range(lmax + 1):
        # (..., channels, 2l+1) -> (..., channels)
        # e3nn-style component normalization: divide by sqrt(2l+1)
        outs.append((b1[l] * b2[l]).sum(dim=-1) / ((2 * l + 1) ** 0.5))
    return torch.cat(outs, dim=-1)


def _apply_channel_adapter_per_l(x_l: torch.Tensor, adapter: nn.Module) -> torch.Tensor:
    """
    Apply a channel adapter Linear/Identity to an irreps block.

    x_l: (..., Cin, 2l+1)
    adapter: maps Cin -> Cout (or Identity)
    Returns: (..., Cout, 2l+1)
    """
    if isinstance(adapter, nn.Identity):
        return x_l
    # move channel to last dim for nn.Linear
    # (..., Cin, m) -> (..., m, Cin) -> Linear -> (..., m, Cout) -> (..., Cout, m)
    y = adapter(x_l.movedim(-2, -1))
    return y.movedim(-1, -2)


def _elementwise_tensor_product_0e_blocks(
    b1: dict[int, torch.Tensor],
    b2: dict[int, torch.Tensor],
    muls_by_l: dict[int, int],
    lmax: int,
) -> torch.Tensor:
    """
    Generalized elementwise 0e invariant builder aligned with e3nn's
    ElementwiseTensorProduct(..., ["0e"], normalization="component") semantics:

      out_l[c] = sum_m x_l[c,m] * y_l[c,m] / sqrt(2l+1)

    b1[l], b2[l]: (..., mul_l, 2l+1)
    Returns concatenated (..., sum_l mul_l) in l=0..lmax order.
    """
    outs = []
    for l in range(lmax + 1):
        x = b1[l]
        y = b2[l]
        # component normalization like e3nn
        outs.append((x * y).sum(dim=-1) / math.sqrt(2 * l + 1))
    return torch.cat(outs, dim=-1)


class ICTDIrrepsE3Conv(nn.Module):
    """
    First convolution in ICTD-irreps space:
      scalar(Ai) ⊗ Y_l(n) -> output_size copies of each l
      then (irreps) ⊗ scalar(Aj) with per-edge radial weights -> channels_out irreps
    """

    def __init__(
        self,
        max_radius: float,
        number_of_basis: int,
        channels_out: int,
        embedding_dim: int = 16,
        max_atomvalue: int = 10,
        output_size: int = 8,
        lmax: int = 2,
        function_type: str = "gaussian",
        # ICTD tensor-product path control (e3nn-instructions-like)
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        # Internal computation dtype for ICTD operations (default: float64 for stability)
        internal_compute_dtype: torch.dtype = torch.float64,
        # Optional: external physical tensor input (e.g. uniform electric field).
        # If set, you may pass `external_tensor` to forward() and it will be embedded into ICTD irreps blocks.
        external_tensor_rank: int | None = None,
        external_tensor_input_repr: str = "cartesian",
        external_tensor_channels_in: int = 1,
        external_tensor_include_trace_chain: bool = True,
        # Learnable per-l scale for external injection; default 0 keeps behavior unchanged at init.
        external_tensor_scale_init: float = 0.0,
    ):
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.channels_out = channels_out
        self.output_size = output_size
        self.lmax = lmax
        self.function_type = function_type

        self.atom_embedding = nn.Embedding(max_atomvalue, embedding_dim)
        self.atom_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, output_size),
        )

        self.tp2 = HarmonicFullyConnectedTensorProduct(
            mul_in1=output_size,
            mul_in2=output_size,
            mul_out=channels_out,
            lmax=lmax,
            internal_weights=True,
            path_policy=ictd_tp_path_policy,
            max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=internal_compute_dtype,
        )
        self.fc = nn.Sequential(
            nn.Linear(number_of_basis, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp2.num_paths),
        )

        # External tensor embedding (optional)
        self.external_tensor_rank = int(external_tensor_rank) if external_tensor_rank is not None else None
        if self.external_tensor_rank is not None:
            self.external_tensor_embed = PhysicalTensorICTDEmbedding(
                rank=self.external_tensor_rank,
                lmax_out=self.lmax,
                channels_in=int(external_tensor_channels_in),
                channels_out=1,
                input_repr=str(external_tensor_input_repr),
                include_trace_chain=bool(external_tensor_include_trace_chain),
                internal_compute_dtype=internal_compute_dtype,
            )
            self.external_tensor_scale_by_l = nn.Parameter(
                torch.full((self.lmax + 1,), float(external_tensor_scale_init))
            )
        else:
            self.external_tensor_embed = None
            self.external_tensor_scale_by_l = None

        self.output_dim = _irreps_total_dim(channels_out, lmax)

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
        precomputed_n=None,
        precomputed_edge_length=None,
        precomputed_Y_list=None,
        # Optional external tensor:
        external_tensor: torch.Tensor | None = None,
        external_tensor_blocks: dict[int, torch.Tensor] | None = None,
    ):
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        if precomputed_n is None or precomputed_edge_length is None:
            edge_batch_idx = batch[edge_src]
            edge_cells = cell[edge_batch_idx]
            shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
            edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
            edge_length = edge_vec.norm(dim=1)
            n = edge_vec / edge_length.clamp(min=1e-8).unsqueeze(-1)
        else:
            n = precomputed_n
            edge_length = precomputed_edge_length

        Ai = self.atom_mlp(self.atom_embedding(A.long()))  # (N, output_size)
        n = n.to(dtype=Ai.dtype)
        edge_length = edge_length.to(dtype=Ai.dtype)

        # External tensor blocks (optional). Shape expectation per l:
        #   (N or 1, mul, 2l+1)
        ext_blocks: dict[int, torch.Tensor] | None = None
        if external_tensor_blocks is not None:
            ext_blocks = external_tensor_blocks
        elif external_tensor is not None:
            if self.external_tensor_embed is None:
                raise ValueError(
                    "external_tensor was provided but ICTDIrrepsE3Conv was not configured with external_tensor_rank"
                )
            ext_blocks = self.external_tensor_embed(external_tensor, return_blocks=True)

        if ext_blocks is not None:
            num_nodes = Ai.shape[0]
            num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
            norm_blocks: dict[int, torch.Tensor] = {}
            for l in range(self.lmax + 1):
                t = ext_blocks.get(l)
                if t is None:
                    continue
                if t.shape[-1] != 2 * l + 1:
                    raise ValueError(f"external_tensor_blocks[{l}] last dim must be 2l+1={2*l+1}, got {t.shape[-1]}")
                # Accept:
                #   - (N, mul, 2l+1) per-node
                #   - (1, mul, 2l+1) broadcast
                #   - (mul, 2l+1) global (broadcast)
                #   - (B, mul, 2l+1) per-graph (mapped to nodes via batch)
                if t.dim() == 2:
                    t = t.unsqueeze(0).expand(num_nodes, *t.shape)
                elif t.dim() == 3:
                    if num_graphs > 0 and t.shape[0] == num_graphs:
                        t = t[batch]
                    elif t.shape[0] == 1 and num_nodes > 1:
                        t = t.expand(num_nodes, *t.shape[1:])
                    elif t.shape[0] != num_nodes:
                        raise ValueError(
                            f"external_tensor_blocks[{l}] first dim must be N={num_nodes} (or 1), got {t.shape[0]}"
                        )
                else:
                    raise ValueError(
                        f"external_tensor_blocks[{l}] must have shape (mul,2l+1) or (N,mul,2l+1); got {tuple(t.shape)}"
                    )
                norm_blocks[l] = t.to(device=Ai.device, dtype=Ai.dtype)
            ext_blocks = norm_blocks

        if precomputed_Y_list is None:
            Y_list = direction_harmonics_all(n, self.lmax)
        else:
            Y_list = precomputed_Y_list
        Y = {l: Y_list[l] for l in range(self.lmax + 1)}  # (E, 2l+1)

        f_in = {l: Ai[edge_src].unsqueeze(-1) * Y[l].unsqueeze(-2) for l in range(self.lmax + 1)}  # (E, output_size, 2l+1)
        if ext_blocks is not None:
            for l in range(self.lmax + 1):
                t = ext_blocks.get(l)
                if t is None:
                    continue
                t_e = t[edge_src]  # (E, mul, 2l+1)
                if t_e.shape[-2] == 1:
                    # scalar(Ai) ⊗ external_irrep_l
                    t_e = t_e * Ai[edge_src].unsqueeze(-1)
                elif t_e.shape[-2] == self.output_size:
                    # already channel-matched
                    pass
                else:
                    raise ValueError(
                        f"external_tensor_blocks[{l}] mul must be 1 or output_size={self.output_size}, got {t_e.shape[-2]}"
                    )
                scale = self.external_tensor_scale_by_l[l] if self.external_tensor_scale_by_l is not None else 1.0
                f_in[l] = f_in[l] + t_e * scale

        x2 = {l: torch.zeros(edge_src.shape[0], self.output_size, 2 * l + 1, device=Ai.device, dtype=Ai.dtype) for l in range(self.lmax + 1)}
        x2[0] = Ai[edge_dst].unsqueeze(-1)  # scalar only

        emb = soft_one_hot_linspace(edge_length, 0.0, self.max_radius, self.number_of_basis, basis=self.function_type, cutoff=True)
        emb = emb.mul(self.number_of_basis ** 0.5).to(dtype=Ai.dtype)
        gates = self.fc(emb)
        out_blocks = self.tp2(f_in, x2, gates)
        edge_features = _merge_irreps(out_blocks, self.channels_out, self.lmax)

        num_nodes = pos.size(0)
        neighbor_count = scatter(torch.ones_like(edge_dst), edge_dst, dim=0, dim_size=num_nodes).clamp(min=1).to(edge_features.dtype)
        out = scatter(edge_features, edge_dst, dim=0, dim_size=num_nodes).div(neighbor_count.unsqueeze(-1))
        return out


class PureCartesianICTDTransformerLayer(nn.Module):
    """
    Pure-Cartesian Transformer layer with ICTD (trace-chain) invariants for readout.

    ICTD-irreps internal model:
      - node features are stored as irreps blocks l=0..lmax (2l+1 dims each)
      - edge direction irreps Y_l(n) are computed from Cartesian n without spherical harmonics
      - tensor products use harmonic-basis CG tensors computed by polynomial multiplication + trace-chain projection
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
        ictd_Lmax: int = 6,
        # ICTD tensor-product path control (e3nn-instructions-like)
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        # Keep these for backward compatibility; currently unused in ICTD mode.
        max_rank_other: int = 1,
        k_policy: str = "k0",
        # Internal computation dtype for ICTD operations (default: float64 for stability)
        internal_compute_dtype: torch.dtype = torch.float64,
        # Optional: allow per-l multiplicities for the "product_5-like" scalar invariant vector.
        # If None: keep current behavior (mul_l = channels for all l).
        # If provided: dict l->mul_l for l=0..lmax; used only for the readout invariants.
        product5_muls_by_l: dict[int, int] | None = None,
        invariant_channels: int = 32,
        # Optional: physical tensor outputs (equivariant heads).
        # Example:
        #   physical_tensor_outputs={
        #     "dipole": {"ls":[1], "channels_out": 1, "reduce":"sum"},
        #     "polarizability": {"ls":[0,2], "channels_out": 1, "reduce":"sum"},
        #   }
        physical_tensor_outputs: dict[str, dict] | None = None,
        # Optional: external tensor (e.g. electric field) embedding configuration for conv1.
        external_tensor_rank: int | None = None,
        external_tensor_input_repr: str = "cartesian",
        external_tensor_channels_in: int = 1,
        external_tensor_include_trace_chain: bool = True,
        external_tensor_scale_init: float = 0.0,
        # Optional prototype long-range module driven by readout invariants.
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
    ):
        super().__init__()
        if embed_size is None:
            embed_size = [128, 128, 128]
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.lmax = int(lmax)
        self.channels = int(hidden_dim_conv)
        self.irreps_dim = _irreps_total_dim(self.channels, self.lmax)

        self.num_interaction = int(num_interaction)
        if self.num_interaction < 2:
            raise ValueError(f"num_interaction must be >= 2, got {self.num_interaction}")
        self.invariant_channels = int(invariant_channels)

        self.max_radius = float(max_embed_radius)
        self.number_of_basis = int(main_number_of_basis)
        self.function_type = str(function_type_main)
        self.external_tensor_rank = int(external_tensor_rank) if external_tensor_rank is not None else None
        self.long_range_mode = str(long_range_mode)
        self.long_range_hidden_dim = int(long_range_hidden_dim)
        self.long_range_boundary = str(long_range_boundary)
        self.long_range_neutralize = bool(long_range_neutralize)
        self.long_range_filter_hidden_dim = int(long_range_filter_hidden_dim)
        self.long_range_kmax = int(long_range_kmax)
        self.long_range_mesh_size = int(long_range_mesh_size)
        self.long_range_slab_padding_factor = int(long_range_slab_padding_factor)
        self.long_range_include_k0 = bool(long_range_include_k0)
        self.long_range_source_channels = int(long_range_source_channels)
        self.long_range_backend = str(long_range_backend)
        self.long_range_reciprocal_backend = str(long_range_reciprocal_backend)
        self.long_range_energy_partition = str(long_range_energy_partition)
        self.long_range_green_mode = str(long_range_green_mode)
        self.long_range_assignment = str(long_range_assignment)
        self.long_range_theta = float(long_range_theta)
        self.long_range_leaf_size = int(long_range_leaf_size)
        self.long_range_multipole_order = int(long_range_multipole_order)
        self.long_range_far_source_dim = int(long_range_far_source_dim)
        self.long_range_far_num_shells = int(long_range_far_num_shells)
        self.long_range_far_shell_growth = float(long_range_far_shell_growth)
        self.long_range_far_tail = bool(long_range_far_tail)
        self.long_range_far_tail_bins = int(long_range_far_tail_bins)
        self.long_range_far_stats = str(long_range_far_stats)
        self.long_range_far_max_radius_multiplier = (
            None if long_range_far_max_radius_multiplier is None else float(long_range_far_max_radius_multiplier)
        )
        self.long_range_far_source_norm = bool(long_range_far_source_norm)
        self.long_range_far_gate_init = float(long_range_far_gate_init)
        self.feature_spectral_mode = str(feature_spectral_mode)
        self.feature_spectral_bottleneck_dim = int(feature_spectral_bottleneck_dim)
        self.feature_spectral_mesh_size = int(feature_spectral_mesh_size)
        self.feature_spectral_filter_hidden_dim = int(feature_spectral_filter_hidden_dim)
        self.feature_spectral_boundary = str(feature_spectral_boundary)
        self.feature_spectral_slab_padding_factor = int(feature_spectral_slab_padding_factor)
        self.feature_spectral_neutralize = bool(feature_spectral_neutralize)
        self.feature_spectral_include_k0 = bool(feature_spectral_include_k0)
        self.feature_spectral_gate_init = float(feature_spectral_gate_init)

        # conv1
        self.e3_conv_emb = ICTDIrrepsE3Conv(
            max_radius=max_embed_radius,
            number_of_basis=main_number_of_basis,
            channels_out=self.channels,
            embedding_dim=embedding_dim,
            max_atomvalue=max_atomvalue,
            output_size=output_size,
            lmax=self.lmax,
            function_type=function_type_main,
            ictd_tp_path_policy=ictd_tp_path_policy,
            ictd_tp_max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=internal_compute_dtype,
            external_tensor_rank=external_tensor_rank,
            external_tensor_input_repr=external_tensor_input_repr,
            external_tensor_channels_in=external_tensor_channels_in,
            external_tensor_include_trace_chain=external_tensor_include_trace_chain,
            external_tensor_scale_init=external_tensor_scale_init,
        )

        # conv2..convN: node irreps (mul=C) x edge Y_l (mul=1) -> node irreps (mul=C), per-edge weights
        self.tp2_layers = nn.ModuleList()
        self.fc2_layers = nn.ModuleList()
        for _ in range(self.num_interaction - 1):
            tp2 = HarmonicFullyConnectedTensorProduct(
                mul_in1=self.channels,
                mul_in2=1,
                mul_out=self.channels,
                lmax=self.lmax,
                internal_weights=True,
                path_policy=ictd_tp_path_policy,
                max_rank_other=ictd_tp_max_rank_other,
                internal_compute_dtype=internal_compute_dtype,
            )
            fc2 = nn.Sequential(
                nn.Linear(main_number_of_basis, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, tp2.num_paths),
            )
            self.tp2_layers.append(tp2)
            self.fc2_layers.append(fc2)

        # Readout invariants:
        #  - scalars: per-l channel Gram -> 32
        #  - norms: per-l per-channel L2 over m
        combined_channels = self.channels * self.num_interaction
        scalar_channels = (self.num_interaction - 1) * self.invariant_channels
        self.W_read = nn.ParameterList([
            nn.Parameter(torch.randn(scalar_channels, combined_channels, combined_channels) * 0.02)
            for _ in range(self.lmax + 1)
        ])
        self.readout_linear = nn.Sequential(
            nn.Linear(scalar_channels + (self.lmax + 1) * combined_channels, embed_size[0]),
            nn.SiLU(),
            nn.Linear(embed_size[0], 17),
        )
        self.weighted_sum = RobustScalarWeightedSum(17, init_weights="zero")
        # Match e3nn-style product_5:
        # T = cat([f1..fn, scalars]); ElementwiseTensorProduct(T,T)->0e
        if product5_muls_by_l is None:
            self.product5_muls_by_l = {l: self.channels for l in range(self.lmax + 1)}
        else:
            self.product5_muls_by_l = {int(k): int(v) for k, v in product5_muls_by_l.items()}
            for l in range(self.lmax + 1):
                if l not in self.product5_muls_by_l:
                    raise ValueError(f"product5_muls_by_l missing l={l} (must cover 0..lmax)")
                if self.product5_muls_by_l[l] <= 0:
                    raise ValueError(f"product5_muls_by_l[{l}] must be positive, got {self.product5_muls_by_l[l]}")

        # Optional per-l channel adapters for f1..fn to match desired muls_by_l.
        self._p5_adapt = nn.ModuleList()
        for _ in range(self.num_interaction):
            layer_adapt = nn.ModuleDict()
            for l in range(self.lmax + 1):
                out_ch = self.product5_muls_by_l[l]
                if out_ch == self.channels:
                    layer_adapt[str(l)] = nn.Identity()
                else:
                    layer_adapt[str(l)] = nn.Linear(self.channels, out_ch, bias=False)
            self._p5_adapt.append(layer_adapt)

        # HarmonicElementwiseProduct replaces manual _irreps_elementwise_tensor_product_0e.
        # For the 0e-only path, mul is metadata only (actual mul inferred from input shape).
        self.product_5 = HarmonicElementwiseProduct(
            lmax=self.lmax,
            mul=combined_channels,
            irreps_out="0e",
            internal_compute_dtype=internal_compute_dtype,
        )

        sum_mul = sum(self.product5_muls_by_l[l] for l in range(self.lmax + 1))
        self.proj_total = MainNet(self.num_interaction * sum_mul + scalar_channels, embed_size, 17)
        self.long_range_module = build_long_range_module(
            mode=self.long_range_mode,
            feature_dim=self.proj_total.input_size if hasattr(self.proj_total, "input_size") else self.num_interaction * sum_mul + scalar_channels,
            hidden_dim=self.long_range_hidden_dim,
            boundary=self.long_range_boundary,
            neutralize=self.long_range_neutralize,
            filter_hidden_dim=self.long_range_filter_hidden_dim,
            kmax=self.long_range_kmax,
            mesh_size=self.long_range_mesh_size,
            slab_padding_factor=self.long_range_slab_padding_factor,
            include_k0=self.long_range_include_k0,
            source_channels=self.long_range_source_channels,
            backend=self.long_range_backend,
            reciprocal_backend=self.long_range_reciprocal_backend,
            energy_partition=self.long_range_energy_partition,
            green_mode=self.long_range_green_mode,
            assignment=self.long_range_assignment,
            theta=self.long_range_theta,
            leaf_size=self.long_range_leaf_size,
            multipole_order=self.long_range_multipole_order,
            far_source_dim=self.long_range_far_source_dim,
            far_num_shells=self.long_range_far_num_shells,
            far_shell_growth=self.long_range_far_shell_growth,
            far_tail=self.long_range_far_tail,
            far_tail_bins=self.long_range_far_tail_bins,
            far_stats=self.long_range_far_stats,
            far_max_radius_multiplier=self.long_range_far_max_radius_multiplier,
            far_source_norm=self.long_range_far_source_norm,
            far_gate_init=self.long_range_far_gate_init,
            cutoff_radius=self.max_radius,
        )
        self.long_range_num_k = getattr(self.long_range_module, "num_k", None) if self.long_range_module is not None else None
        self.feature_spectral_module = build_feature_spectral_module(
            mode=self.feature_spectral_mode,
            feature_dim=self.proj_total.input_size if hasattr(self.proj_total, "input_size") else self.num_interaction * sum_mul + scalar_channels,
            bottleneck_dim=self.feature_spectral_bottleneck_dim,
            mesh_size=self.feature_spectral_mesh_size,
            filter_hidden_dim=self.feature_spectral_filter_hidden_dim,
            boundary=self.feature_spectral_boundary,
            slab_padding_factor=self.feature_spectral_slab_padding_factor,
            neutralize=self.feature_spectral_neutralize,
            include_k0=self.feature_spectral_include_k0,
            gate_init=self.feature_spectral_gate_init,
        )
        self.long_range_runtime_backend = "none"
        self.long_range_runtime_source_kind = "none"
        self.long_range_runtime_source_channels = 0
        self.long_range_runtime_source_layout = "none"
        self.long_range_runtime_source_boundary = self.long_range_boundary
        self.long_range_runtime_source_slab_padding_factor = self.long_range_slab_padding_factor
        if self.long_range_module is not None and bool(getattr(self.long_range_module, "exports_reciprocal_source", False)):
            self.long_range_runtime_backend = str(getattr(self.long_range_module, "runtime_backend", "none"))
            self.long_range_runtime_source_kind = str(getattr(self.long_range_module, "source_kind", "latent_charge"))
            self.long_range_runtime_source_channels = int(getattr(self.long_range_module, "source_channels", self.long_range_source_channels))
            self.long_range_runtime_source_layout = str(getattr(self.long_range_module, "source_layout", "channels_last"))
            self.long_range_runtime_source_boundary = self.long_range_boundary
            self.long_range_runtime_source_slab_padding_factor = self.long_range_slab_padding_factor
            self.reciprocal_source_channels = self.long_range_runtime_source_channels
            self.reciprocal_source_boundary = self.long_range_runtime_source_boundary
            self.reciprocal_source_slab_padding_factor = self.long_range_runtime_source_slab_padding_factor
        elif self.feature_spectral_module is not None:
            self.long_range_runtime_backend = "mesh_fft"
            self.long_range_runtime_source_kind = "feature_bottleneck"
            self.long_range_runtime_source_channels = self.feature_spectral_bottleneck_dim
            self.long_range_runtime_source_layout = "channels_last"
            self.long_range_runtime_source_boundary = self.feature_spectral_boundary
            self.long_range_runtime_source_slab_padding_factor = self.feature_spectral_slab_padding_factor
            self.reciprocal_source_channels = self.feature_spectral_bottleneck_dim
            self.reciprocal_source_boundary = self.feature_spectral_boundary
            self.reciprocal_source_slab_padding_factor = self.feature_spectral_slab_padding_factor
        else:
            self.reciprocal_source_channels = 0
            self.reciprocal_source_boundary = "periodic"
            self.reciprocal_source_slab_padding_factor = 2

        # Optional physical tensor heads (equivariant linear maps per l).
        self._physical_tensor_specs: dict[str, dict] | None = None
        self.physical_tensor_heads: nn.ModuleDict | None = None
        if physical_tensor_outputs is not None:
            self._physical_tensor_specs = {}
            heads = nn.ModuleDict()
            combined_channels = self.channels * self.num_interaction
            for name, spec_in in physical_tensor_outputs.items():
                if not isinstance(spec_in, dict):
                    raise ValueError(f"physical_tensor_outputs[{name!r}] must be a dict, got {type(spec_in)}")
                spec = dict(spec_in)
                ls = spec.get("ls", None)
                if ls is None:
                    raise ValueError(f"physical_tensor_outputs[{name!r}] missing 'ls' (list of l to output)")
                ls = [int(l) for l in ls]
                for l in ls:
                    if not (0 <= l <= self.lmax):
                        raise ValueError(f"physical_tensor_outputs[{name!r}] l={l} out of range 0..lmax={self.lmax}")
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
                    if ch_out_by_l[l] == combined_channels:
                        per_l[str(l)] = nn.Identity()
                    else:
                        per_l[str(l)] = nn.Linear(combined_channels, ch_out_by_l[l], bias=False)
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
        sync_after_scatter: callable | None = None,
        # Optional external tensor injected at conv1:
        external_tensor: torch.Tensor | None = None,
        external_tensor_blocks: dict[int, torch.Tensor] | None = None,
        # Optional physical tensor outputs (default False：推理/MD 仅需能量和力时不计算物理张量头).
        return_physical_tensors: bool = False,
        return_reciprocal_source: bool = False,
    ):
        """
        sync_after_scatter: 若提供，在 conv1 输出后、每个 tp2 scatter 后对节点特征做同步（DDP 下同步 ghost 节点）。
        默认 None，单卡行为不变。
        """
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
        n = edge_vec / edge_length.clamp(min=1e-8).unsqueeze(-1)

        # conv1
        # compute Y_l once and reuse
        Y_list = direction_harmonics_all(n.to(dtype=next(self.parameters()).dtype), self.lmax)
        f1 = self.e3_conv_emb(
            pos, A, batch, edge_src, edge_dst, edge_shifts, cell,
            precomputed_n=n,
            precomputed_edge_length=edge_length,
            precomputed_Y_list=Y_list,
            external_tensor=external_tensor,
            external_tensor_blocks=external_tensor_blocks,
        )  # (N, C*(lmax+1)^2)
        if sync_after_scatter is not None:
            f1 = sync_after_scatter(f1)
        features = [f1]

        # conv2..convN: node irreps x edge Y_l -> node irreps
        num_nodes = pos.size(0)
        neighbor_count = scatter(torch.ones_like(edge_dst), edge_dst, dim=0, dim_size=num_nodes).clamp(min=1)
        emb_base = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis=self.function_type,
            cutoff=True,
        ).mul(self.number_of_basis ** 0.5)
        for tp2, fc2 in zip(self.tp2_layers, self.fc2_layers):
            f_prev = features[-1]
            n = n.to(dtype=f_prev.dtype)
            edge_length = edge_length.to(dtype=f_prev.dtype)
            Y = {l: Y_list[l].to(dtype=f_prev.dtype).unsqueeze(-2) for l in range(self.lmax + 1)}  # (E,1,2l+1)
            emb = emb_base.to(dtype=f_prev.dtype)
            gates = fc2(emb)

            x1 = _split_irreps(f_prev, self.channels, self.lmax)
            x1e = {l: x1[l][edge_src] for l in range(self.lmax + 1)}
            edge_blocks = tp2(x1e, Y, gates)  # dict l -> (E, C, 2l+1)
            edge_flat = _merge_irreps(edge_blocks, self.channels, self.lmax)
            neighbor_count = neighbor_count.to(edge_flat.dtype)
            f_next = scatter(edge_flat, edge_dst, dim=0, dim_size=num_nodes).div(neighbor_count.unsqueeze(-1))
            if sync_after_scatter is not None:
                f_next = sync_after_scatter(f_next)
            features.append(f_next)

        f_combine = torch.cat(features, dim=-1)  # (N, nC*(lmax+1)^2)

        xb = _split_irreps(f_combine, self.channels * self.num_interaction, self.lmax)
        physical_out: dict[str, dict[int, torch.Tensor]] | None = None
        if return_physical_tensors:
            if self.physical_tensor_heads is None or self._physical_tensor_specs is None:
                raise ValueError("return_physical_tensors=True but physical_tensor_outputs was not set in __init__")
            physical_out = {}
            num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
            counts = None
            for name, per_l in self.physical_tensor_heads.items():
                spec = self._physical_tensor_specs[name]
                ls = spec["ls"]
                reduce = spec["reduce"]
                out_blocks: dict[int, torch.Tensor] = {}
                for l in ls:
                    y_l = _apply_channel_adapter_per_l(xb[l], per_l[str(l)])  # (N, Cout, 2l+1)
                    if reduce == "none":
                        out_blocks[l] = y_l
                    else:
                        yg = scatter(y_l, batch, dim=0, dim_size=num_graphs, reduce="sum")
                        if reduce == "mean":
                            if counts is None:
                                counts = scatter(
                                    torch.ones_like(batch, dtype=yg.dtype),
                                    batch,
                                    dim=0,
                                    dim_size=num_graphs,
                                    reduce="sum",
                                ).clamp(min=1.0)
                            yg = yg / counts.view(-1, 1, 1)
                        out_blocks[l] = yg
                physical_out[name] = out_blocks

        scalars = torch.zeros(
            f_combine.shape[0],
            (self.num_interaction - 1) * self.invariant_channels,
            device=f_combine.device,
            dtype=f_combine.dtype,
        )
        for l in range(self.lmax + 1):
            t = xb[l]  # (N,nC,2l+1)
            # e3nn-style component normalization: divide by sqrt(2l+1)
            gram = torch.einsum("ncm,ndm->ncd", t, t) / ((2 * l + 1) ** 0.5)  # (N,2C,2C)
            scalars = scalars + torch.einsum("ocd,ncd->no", self.W_read[l], gram)

        # Build T = cat(features, scalars) per l, then product_5(T, T) → 0e.
        # Scalars are appended to the l=0 block so they also go through normalized EWP.
        all_identity = all(
            isinstance(self._p5_adapt[i][str(l)], nn.Identity)
            for i in range(self.num_interaction)
            for l in range(self.lmax + 1)
        )
        splits = [_split_irreps(f, self.channels, self.lmax) for f in features]
        T_blocks: dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            parts = []
            for i in range(len(features)):
                b_l = splits[i][l]
                if not all_identity:
                    b_l = _apply_channel_adapter_per_l(b_l, self._p5_adapt[i][str(l)])
                parts.append(b_l)
            T_blocks[l] = torch.cat(parts, dim=-2)
        T_blocks[0] = torch.cat([T_blocks[0], scalars.unsqueeze(-1)], dim=-2)
        f_prod5 = self.product_5(T_blocks, T_blocks)
        feature_reciprocal_source = None
        if self.feature_spectral_module is not None:
            f_prod5, feature_reciprocal_source = self.feature_spectral_module(f_prod5, pos, batch, cell)

        long_range_energy = None
        reciprocal_source = None
        defer_long_range_to_runtime = False
        if self.long_range_module is not None:
            if return_reciprocal_source and bool(getattr(self.long_range_module, "exports_reciprocal_source", False)):
                long_range_energy, reciprocal_source = self.long_range_module(
                    f_prod5,
                    pos,
                    batch,
                    cell,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    return_source=True,
                )
                defer_long_range_to_runtime = reciprocal_source.numel() > 0
            else:
                long_range_energy = self.long_range_module(
                    f_prod5,
                    pos,
                    batch,
                    cell,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                )
        if reciprocal_source is None:
            reciprocal_source = feature_reciprocal_source

        product_proj = self.proj_total(f_prod5)
        e_out = self.weighted_sum(product_proj)
        e_scalar = e_out.sum(dim=-1, keepdim=True)
        if long_range_energy is not None and not defer_long_range_to_runtime:
            e_scalar = e_scalar + long_range_energy
        if reciprocal_source is None and return_reciprocal_source:
            reciprocal_source = e_scalar.new_empty((e_scalar.size(0), 0))
        if return_physical_tensors and return_reciprocal_source:
            assert physical_out is not None
            assert reciprocal_source is not None
            return e_scalar, physical_out, reciprocal_source
        if return_physical_tensors:
            assert physical_out is not None
            return e_scalar, physical_out
        if return_reciprocal_source:
            assert reciprocal_source is not None
            return e_scalar, reciprocal_source
        return e_scalar
