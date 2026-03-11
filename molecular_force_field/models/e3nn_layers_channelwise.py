"""
Channelwise (scatter-sum) convolution blocks implemented with pure e3nn + torch_scatter.

Convolution form:
  - linear_up(node_feats) ->
  - channelwise TensorProduct(node_feats[senders], edge_attrs, w(edge_feats)) ->
  - scatter_sum to receivers ->
  - linear ->
  - divide by avg_num_neighbors (global scalar, not per-node degree)

This module is a drop-in alternative to `e3nn_layers.py` when you want
channelwise edge convolution (no neighbor feature in the TP, sum aggregation,
global normalization).

Notes vs the original `e3nn_layers.py`:
  - No neighbor feature enters the TensorProduct (node_feats ⊗ edge_attrs only).
  - Aggregation is scatter-sum and normalization uses avg_num_neighbors (global), not per-node mean.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
from e3nn import o3
from e3nn import nn as e3nn_nn
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate
from molecular_force_field.utils.scatter import scatter

from molecular_force_field.models.mlp import MainNet2, MainNet, RobustScalarWeightedSum


def _scatter_sum_maybe_compiled(
    src: torch.Tensor,
    index: torch.Tensor,
    *,
    dim_size: int,
) -> torch.Tensor:
    """
    Scatter-sum helper that is torch.compile-friendly.

    - Eager: uses the project's `scatter` compat helper (prefers torch_scatter when available).
    - Under torch.compile (Dynamo tracing): uses PyTorch native index_add, which
      avoids custom-op graph breaks and tends to compile more reliably.
    """
    is_compiling = False
    try:
        # torch._dynamo.is_compiling() is available in PyTorch 2.x
        import torch._dynamo  # type: ignore

        is_compiling = bool(torch._dynamo.is_compiling())
    except Exception:
        is_compiling = False

    if not is_compiling:
        return scatter(src, index, dim=0, dim_size=dim_size, reduce="sum")

    # Native sum aggregation: out[i] += src[j] where index[j] == i
    out = src.new_zeros((dim_size, src.size(-1)))
    return out.index_add(0, index, src)


def tp_out_irreps_with_instructions(
    irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
) -> Tuple[o3.Irreps, List[Tuple[int, int, int, str, bool]]]:
    """
    Build *channelwise* tensor product instructions.

      - output multiplicity is taken from irreps1 (preserve channels)
      - only paths whose irrep is in target_irreps are kept
      - instructions use mode "uvu" (e3nn convention)
      - weights are per-edge (internal_weights=False in the TP module)
    """
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


class _ChannelwiseEdgeConv(nn.Module):
    """
    Core channelwise edge convolution:
      node_feats -> linear_up -> TP(node_feats[senders], edge_attrs, weights(edge_feats)) -> scatter_sum -> linear -> /avg_num_neighbors
    """

    def __init__(
        self,
        irreps_node_input: o3.Irreps,
        irreps_node_output: o3.Irreps,
        edge_attrs_lmax: int,
        r_max: float,
        number_of_basis: int,
        radial_hidden: List[int],
        function_type: str = "gaussian",
        avg_num_neighbors: float | None = None,
    ):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_edge_attrs = o3.Irreps.spherical_harmonics(lmax=int(edge_attrs_lmax))
        self.r_max = float(r_max)
        self.number_of_basis = int(number_of_basis)
        self.function_type = str(function_type)
        self.avg_num_neighbors = None if avg_num_neighbors is None else float(avg_num_neighbors)

        self.linear_up = o3.Linear(self.irreps_node_input, self.irreps_node_output)

        # Channelwise TensorProduct: (node_feats) ⊗ (edge_attrs) -> irreps_mid
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.irreps_node_output, self.irreps_edge_attrs, self.irreps_node_output
        )
        self.irreps_mid = irreps_mid
        self.conv_tp = o3.TensorProduct(
            self.irreps_node_output,
            self.irreps_edge_attrs,
            self.irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Per-edge TP weights from radial basis embedding
        self.conv_tp_weights = e3nn_nn.FullyConnectedNet(
            [self.number_of_basis] + list(radial_hidden) + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        self.linear = o3.Linear(self.irreps_mid, self.irreps_node_output)

    def forward(
        self,
        node_feats: torch.Tensor,  # (N, irreps_node_input.dim)
        edge_vec: torch.Tensor,  # (E, 3)
        edge_length: torch.Tensor,  # (E,)
        senders: torch.Tensor,  # (E,)
        receivers: torch.Tensor,  # (E,)
        num_nodes: int,
    ) -> torch.Tensor:
        # Edge attrs (spherical harmonics)
        edge_attrs = o3.spherical_harmonics(
            self.irreps_edge_attrs, edge_vec, normalize=True, normalization="component"
        )

        # Radial embedding -> per-edge TP weights
        emb = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.r_max,
            self.number_of_basis,
            basis=self.function_type,
            cutoff=True,
        ).mul(self.number_of_basis ** 0.5)
        w = self.conv_tp_weights(emb)

        # Linear-up then channelwise TP on edges
        x = self.linear_up(node_feats)  # (N, dim_out)
        mji = self.conv_tp(x[senders], edge_attrs, w)  # (E, irreps_mid.dim)

        # Scatter-sum to receivers
        msg = _scatter_sum_maybe_compiled(mji, receivers, dim_size=num_nodes)
        msg = self.linear(msg)

        # Normalize by avg_num_neighbors (global)
        if self.avg_num_neighbors is None:
            avg = float(senders.numel()) / float(max(num_nodes, 1))
        else:
            avg = self.avg_num_neighbors
        msg = msg / max(avg, 1e-8)
        return msg


class E3Conv(nn.Module):
    """
    First convolution (node scalars -> irreps_output) with channelwise edge TP.

    Signature matches the original `e3nn_layers.E3Conv.forward` so you can swap imports.
    """

    def __init__(
        self,
        max_radius: float,
        number_of_basis: int,
        irreps_output: str,
        embedding_dim: int = 16,
        max_atomvalue: int = 10,
        output_size: int = 8,
        main_hidden_sizes3=None,
        emb_number=None,
        function_type: str = "gaussian",
        edge_attrs_lmax: int = 2,
        avg_num_neighbors: float | None = None,
        **_unused,
    ):
        super().__init__()
        self.max_radius = float(max_radius)
        self.number_of_basis = int(number_of_basis)
        self.function_type = str(function_type)

        if main_hidden_sizes3 is None:
            main_hidden_sizes3 = [64, 32]
        if emb_number is None:
            emb_number = [64, 64, 64]

        default_dtype = torch.get_default_dtype()
        self.atom_embedding = nn.Embedding(
            num_embeddings=max_atomvalue, embedding_dim=embedding_dim, dtype=default_dtype
        )
        self.fitnet1 = MainNet2(
            input_size=embedding_dim, hidden_sizes=main_hidden_sizes3, output_size=output_size
        )

        self.node_irreps_in = o3.Irreps(f"{output_size}x0e")
        self.node_irreps_out = o3.Irreps(irreps_output)

        self.conv = _ChannelwiseEdgeConv(
            irreps_node_input=self.node_irreps_in,
            irreps_node_output=self.node_irreps_out,
            edge_attrs_lmax=edge_attrs_lmax,
            r_max=self.max_radius,
            number_of_basis=self.number_of_basis,
            radial_hidden=list(emb_number),
            function_type=self.function_type,
            avg_num_neighbors=avg_num_neighbors,
        )

    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, *, precomputed_edge_vec=None):
        if precomputed_edge_vec is not None:
            edge_vec = precomputed_edge_vec
        else:
            edge_batch_idx = batch[edge_src]
            edge_cells = cell[edge_batch_idx]
            shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
            edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)

        num_nodes = pos.size(0)
        atom_embeddings = self.atom_embedding(A.long())
        node_scalars = self.fitnet1(atom_embeddings)  # (N, output_size)

        return self.conv(
            node_feats=node_scalars,
            edge_vec=edge_vec,
            edge_length=edge_length,
            senders=edge_src,
            receivers=edge_dst,
            num_nodes=num_nodes,
        )


class E3Conv2(nn.Module):
    """
    Subsequent convolution (node irreps -> node irreps) with channelwise edge TP.

    Signature matches the original `e3nn_layers.E3Conv2.forward`.
    """

    def __init__(
        self,
        max_radius: float,
        number_of_basis: int,
        irreps_input_conv: str,
        irreps_output: str,
        function_type: str = "gaussian",
        edge_attrs_lmax: int = 2,
        emb_number=None,
        avg_num_neighbors: float | None = None,
        **_unused,
    ):
        super().__init__()
        self.max_radius = float(max_radius)
        self.number_of_basis = int(number_of_basis)
        self.function_type = str(function_type)
        if emb_number is None:
            emb_number = [64, 64, 64]

        self.node_irreps_in = o3.Irreps(irreps_input_conv)
        self.node_irreps_out = o3.Irreps(irreps_output)

        self.conv = _ChannelwiseEdgeConv(
            irreps_node_input=self.node_irreps_in,
            irreps_node_output=self.node_irreps_out,
            edge_attrs_lmax=edge_attrs_lmax,
            r_max=self.max_radius,
            number_of_basis=self.number_of_basis,
            radial_hidden=list(emb_number),
            function_type=self.function_type,
            avg_num_neighbors=avg_num_neighbors,
        )

    def forward(self, f_in, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, *, precomputed_edge_vec=None):
        if precomputed_edge_vec is not None:
            edge_vec = precomputed_edge_vec
        else:
            edge_batch_idx = batch[edge_src]
            edge_cells = cell[edge_batch_idx]
            shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
            edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)

        num_nodes = pos.size(0)
        return self.conv(
            node_feats=f_in,
            edge_vec=edge_vec,
            edge_length=edge_length,
            senders=edge_src,
            receivers=edge_dst,
            num_nodes=num_nodes,
        )


class E3_TransformerLayer_multi(nn.Module):
    """
    Same outer model as `e3nn_layers.E3_TransformerLayer_multi`, but its internal
    convolution blocks use channelwise edge convolution (channelwise TP + scatter_sum).

    Cross-layer fusion and readout are unchanged; only the conv form differs.
    """

    def __init__(
        self,
        max_embed_radius,
        main_max_radius,
        main_number_of_basis,
        irreps_input,
        irreps_query,
        irreps_key,
        irreps_value,
        irreps_output,
        irreps_sh,
        hidden_dim_sh,
        hidden_dim,
        channel_in2=32,
        embedding_dim=16,
        max_atomvalue=10,
        output_size=8,
        embed_size=None,
        main_hidden_sizes3=None,
        num_layers=1,
        device=None,
        function_type_main="gaussian",
        num_interaction=2,
    ):
        super().__init__()

        if embed_size is None:
            embed_size = [128, 128, 128]
        if main_hidden_sizes3 is None:
            main_hidden_sizes3 = [64, 32]
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # In original code, hidden_dim is overloaded to mean the radial MLP widths.
        emb_number = [64, 64, 64] if not isinstance(hidden_dim, list) else hidden_dim

        # Gate irreps (kept identical to original)
        irreps_scalars = o3.Irreps(f"{32}x0e + {32}x0e")
        irreps_gates = o3.Irreps(f"{channel_in2}x0e + {channel_in2}x0e")
        irreps_gated = o3.Irreps(f"{channel_in2}x1o + {channel_in2}x2e")

        self.gate_layer = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[torch.tanh, torch.tanh],
            irreps_gates=irreps_gates,
            act_gates=[torch.tanh, torch.tanh],
            irreps_gated=irreps_gated,
        )
        self.irreps_output_conv = self.gate_layer.irreps_in
        self.irreps_input = self.gate_layer.irreps_out
        self.irreps_sh = irreps_sh
        self.max_radius = main_max_radius
        self.number_of_basis = main_number_of_basis

        self.num_interaction = int(num_interaction)
        if self.num_interaction < 2:
            raise ValueError(f"num_interaction must be >= 2, got {self.num_interaction}")

        # --- Convolution stack (channelwise edge conv) ---
        self.e3_conv_layers = nn.ModuleList()
        self.e3_conv_layers.append(
            E3Conv(
                max_radius=max_embed_radius,
                number_of_basis=main_number_of_basis,
                irreps_output=irreps_input,
                embedding_dim=embedding_dim,
                max_atomvalue=max_atomvalue,
                output_size=output_size,
                main_hidden_sizes3=main_hidden_sizes3,
                embed_size=embed_size,  # ignored by this file's E3Conv
                emb_number=emb_number,
                function_type=function_type_main,
                atomic_energy_keys=torch.tensor([1, 6, 7, 8], device=self.device),  # ignored
            )
        )
        for _ in range(1, self.num_interaction):
            self.e3_conv_layers.append(
                E3Conv2(
                    max_radius=max_embed_radius,
                    number_of_basis=main_number_of_basis,
                    irreps_input_conv=irreps_input,
                    irreps_output=irreps_input,
                    embedding_dim=embedding_dim,  # ignored by E3Conv2 here
                    max_atomvalue=max_atomvalue,  # ignored
                    output_size=output_size,  # ignored
                    main_hidden_sizes3=main_hidden_sizes3,  # ignored
                    embed_size=embed_size,  # ignored
                    emb_number=emb_number,
                    function_type=function_type_main,
                    atomic_energy_keys=torch.tensor([1, 6, 7, 8], device=self.device),  # ignored
                )
            )

        self.f2_proj = o3.Linear(self.irreps_output_conv, self.irreps_output_conv)

        # --- Everything below is copied from original e3nn_layers.py (unchanged) ---
        self.product_1 = o3.FullyConnectedTensorProduct(
            self.irreps_output_conv,
            self.irreps_output_conv,
            "16x0e",
            shared_weights=True,
            internal_weights=True,
            normalization="component",
        )

        irreps_input_multi = o3.Irreps(irreps_input) * self.num_interaction
        scalar_channels = (self.num_interaction - 1) * 32
        self.product_3 = o3.FullyConnectedTensorProduct(
            irreps_input_multi,
            irreps_input_multi,
            f"{scalar_channels}x0e",
            shared_weights=True,
            internal_weights=True,
            normalization="component",
        )

        self.product_2 = o3.FullyConnectedTensorProduct(
            self.product_3.irreps_out,
            self.product_3.irreps_out,
            "64x0e",
            shared_weights=True,
            internal_weights=True,
            normalization="component",
        )

        irreps_product_5 = irreps_input_multi + self.product_3.irreps_out
        self.product_5 = o3.ElementwiseTensorProduct(
            irreps_product_5,
            irreps_product_5,
            ["0e"],
            normalization="component",
        )

        self.proj_total = MainNet(self.product_5.irreps_out.dim, embed_size, 17)
        self.weight_mlp = MainNet2(main_number_of_basis * 2, [64, 32, 16], 1)
        self.batch_norm = e3nn_nn.BatchNorm(irreps=self.irreps_input)

        default_dtype = torch.get_default_dtype()
        self.atom_embedding = nn.Embedding(
            num_embeddings=max_atomvalue, embedding_dim=embedding_dim, dtype=default_dtype
        )

        self.linear_layer1 = o3.Linear(self.irreps_output_conv, "1x0e")
        self.linear_layer2 = o3.Linear(self.irreps_output_conv, "1x0e")
        self.linear_layer3 = o3.Linear(self.irreps_input, "1x0e")

        self.linear_layers = nn.ModuleList(
            [o3.Linear(self.irreps_input, self.irreps_input) for _ in range(num_layers)]
        )
        self.linear_layers4 = nn.ModuleList(
            [o3.Linear(self.irreps_input, "1x0e") for _ in range(num_layers)]
        )

        if isinstance(hidden_dim_sh, int):
            hidden_dim_sh_irreps = f"{hidden_dim_sh}x0e"
        else:
            hidden_dim_sh_irreps = hidden_dim_sh
        self.linear_layer_2 = o3.Linear(self.irreps_input, hidden_dim_sh_irreps)
        self.non_linearity = nn.SiLU()
        self.linear_layer_3 = o3.Linear(hidden_dim_sh_irreps, "1x0e")

        self.tp_featrue = o3.FullyConnectedTensorProduct(
            irreps_in1="16x0e",
            irreps_in2="1x0e + 1x1o + 1x2e",
            irreps_out=self.irreps_sh,
            shared_weights=True,
            internal_weights=True,
            normalization="component",
        )

        self.tensor_product = o3.FullyConnectedTensorProduct(
            irreps_in1=self.irreps_input,
            irreps_in2=self.irreps_input,
            irreps_out=self.irreps_input,
            shared_weights=True,
            internal_weights=True,
            normalization="norm",
        )

        self.num_features = 17
        self.readout = o3.TensorSquare(
            irreps_in="6x0e + 3x1o + 2x2e",
            irreps_out=f"{self.num_features}x0e",
        )
        self.weighted_sum = RobustScalarWeightedSum(self.num_features, init_weights="zero")

    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, *, precomputed_edge_vec=None):
        sort_idx = torch.argsort(edge_dst)
        edge_src = edge_src[sort_idx]
        edge_dst = edge_dst[sort_idx]
        edge_shifts = edge_shifts[sort_idx]

        if precomputed_edge_vec is not None:
            edge_vec = precomputed_edge_vec[sort_idx]
        else:
            edge_vec = None

        features = []
        f_prev = self.e3_conv_layers[0](pos, A, batch, edge_src, edge_dst, edge_shifts, cell, precomputed_edge_vec=edge_vec)
        features.append(f_prev)
        for conv in self.e3_conv_layers[1:]:
            f_prev = conv(f_prev, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, precomputed_edge_vec=edge_vec)
            features.append(f_prev)

        f_combine = torch.cat(features, dim=-1)
        f_combine_product = self.product_3(f_combine, f_combine)

        T = torch.cat(features + [f_combine_product], dim=-1)
        f2_product_5 = self.product_5(T, T)

        product_proj = self.proj_total(f2_product_5)
        e_out = torch.cat([product_proj], dim=-1)
        e_out = (1) * self.weighted_sum(e_out)
        atom_energies = e_out.sum(dim=-1, keepdim=True)
        return atom_energies
