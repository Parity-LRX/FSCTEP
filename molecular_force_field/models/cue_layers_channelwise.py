"""
Channelwise (scatter-sum) convolution blocks implemented with NVIDIA cuEquivariance (Cuequivariance).

This file is an alternative backend to `e3nn_layers_channelwise.py`.

Design goals:
  - Keep the same *public* class names / forward signatures as the e3nn version:
      - `E3Conv`, `E3Conv2`, `E3_TransformerLayer_multi`
  - Avoid importing cuEquivariance at module import time, so users without the
    dependency can still import the package (the error is raised only if this
    mode is actually instantiated).

Notes:
  - We run the equivariant feature tensors in `cue.ir_mul` layout (cuEquivariance-preferred)
    throughout this module to match its fastest kernels.
  - The radial embedding here is a lightweight implementation to generate
    per-edge weights. It supports the same `function_type` strings as the CLI,
    but it is not bitwise identical to e3nn's `soft_one_hot_linspace`.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn

from molecular_force_field.models.long_range import build_feature_spectral_module, build_long_range_module
from molecular_force_field.models.mlp import MainNet2, MainNet, RobustScalarWeightedSum


def _require_cue():
    """
    Lazily import cuEquivariance and its PyTorch bindings.
    """
    warnings.filterwarnings(
        "ignore",
        message=r".*EquivariantTensorProduct is deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*TensorProductUniform1d is deprecated.*",
        category=UserWarning,
    )
    try:
        import cuequivariance as cue  # type: ignore
        import cuequivariance_torch as cuet  # type: ignore
        from cuequivariance.group_theory.experimental.e3nn import (  # type: ignore
            O3_e3nn,
        )
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "cuEquivariance is required for tensor_product_mode='spherical-save-cue'.\n"
            "Install (on CUDA Linux):\n"
            "  pip install cuequivariance-torch cuequivariance-ops-torch-cu12\n"
            "See docs: https://docs.nvidia.com/cuda/cuequivariance/\n"
            f"Original import error: {e}"
        ) from e
    return cue, cuet, O3_e3nn


@dataclass(frozen=True)
class _IrrepSeg:
    mul: int
    l: int
    parity: str  # "e" or "o"

    @property
    def dim(self) -> int:
        return self.mul * (2 * self.l + 1)

    @property
    def irrep_str(self) -> str:
        return f"{self.l}{self.parity}"


def _parse_irreps_e3nn(irreps: str) -> List[_IrrepSeg]:
    """
    Parse an e3nn-style irreps string like "64x0e + 64x1o + 64x2e".
    """
    s = str(irreps).replace(" ", "")
    if not s:
        return []
    parts = s.split("+")
    out: List[_IrrepSeg] = []
    for p in parts:
        if not p:
            continue
        if "x" in p:
            mul_s, ir_s = p.split("x", 1)
            mul = int(mul_s)
        else:
            mul = 1
            ir_s = p
        if len(ir_s) < 2:
            raise ValueError(f"Bad irrep token: {p!r} in {irreps!r}")
        parity = ir_s[-1]
        l = int(ir_s[:-1])
        if parity not in ("e", "o"):
            raise ValueError(f"Bad parity in token: {p!r} (expected 'e' or 'o')")
        out.append(_IrrepSeg(mul=mul, l=l, parity=parity))
    return out


def _irreps_dim(segs: Sequence[_IrrepSeg]) -> int:
    return int(sum(s.dim for s in segs))


def _scale_irreps(segs: Sequence[_IrrepSeg], factor: int) -> List[_IrrepSeg]:
    return [_IrrepSeg(mul=int(s.mul * factor), l=s.l, parity=s.parity) for s in segs]


def _irreps_to_string(segs: Sequence[_IrrepSeg]) -> str:
    return " + ".join([f"{s.mul}x{s.l}{s.parity}" for s in segs])


def _sh_irreps_string(lmax: int) -> str:
    segs = [_IrrepSeg(mul=1, l=l, parity=("e" if (l % 2 == 0) else "o")) for l in range(int(lmax) + 1)]
    return _irreps_to_string(segs)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        last = int(in_dim)
        for h in hidden:
            layers.append(nn.Linear(last, int(h)))
            layers.append(nn.SiLU())
            last = int(h)
        layers.append(nn.Linear(last, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _radial_embedding(
    r: torch.Tensor,
    *,
    r_max: float,
    number_of_basis: int,
    function_type: str,
) -> torch.Tensor:
    """
    Produce a radial basis embedding (E, number_of_basis).
    This is a lightweight alternative to e3nn's `soft_one_hot_linspace`.
    """
    nb = int(number_of_basis)
    if nb <= 0:
        raise ValueError(f"number_of_basis must be > 0, got {number_of_basis}")
    r_max = float(r_max)
    if r_max <= 0:
        raise ValueError(f"r_max must be > 0, got {r_max}")

    x = (r / r_max).clamp(min=0.0, max=1.0)
    idx = torch.arange(nb, device=r.device, dtype=r.dtype)

    ft = str(function_type)
    if ft == "gaussian":
        centers = idx / max(nb - 1, 1)
        sigma = 1.0 / max(nb - 1, 1)
        emb = torch.exp(-0.5 * ((x[:, None] - centers[None, :]) / max(sigma, 1e-6)) ** 2)
    elif ft in ("fourier", "cosine"):
        k = idx + 1.0
        emb = torch.cos(math.pi * k[None, :] * x[:, None])
    elif ft == "bessel":
        k = idx + 1.0
        t = math.pi * k[None, :] * x[:, None]
        emb = torch.sin(t) / torch.clamp(t, min=1e-6)
    elif ft == "smooth_finite":
        centers = idx / max(nb - 1, 1)
        d = (x[:, None] - centers[None, :]).abs()
        emb = (1.0 - d).clamp(min=0.0) ** 2 * (1.0 + 2.0 * d)
    else:
        centers = idx / max(nb - 1, 1)
        sigma = 1.0 / max(nb - 1, 1)
        emb = torch.exp(-0.5 * ((x[:, None] - centers[None, :]) / max(sigma, 1e-6)) ** 2)

    emb = emb * (nb**0.5)
    emb = emb * (r <= r_max).to(dtype=r.dtype)[:, None]
    return emb.reshape(-1, nb)


def _elementwise_tp_invariants(x1: torch.Tensor, x2: torch.Tensor, segs: Sequence[_IrrepSeg]) -> torch.Tensor:
    """
    Compute the scalar invariants corresponding to e3nn ElementwiseTensorProduct(..., ["0e"])
    for x1 and x2 having the same irreps structure (mul_ir layout).

    For each segment (mul x l{e/o}), we return `mul` scalars:
      inv[c] = sum_m x1[c,m] * x2[c,m]
    """
    if x1.shape != x2.shape:
        raise ValueError(f"x1 and x2 must have same shape, got {x1.shape} vs {x2.shape}")
    N = x1.size(0)
    out_chunks: List[torch.Tensor] = []
    offset = 0
    for seg in segs:
        d = 2 * seg.l + 1
        width = seg.mul * d
        a = x1[:, offset : offset + width].reshape(N, seg.mul, d)
        b = x2[:, offset : offset + width].reshape(N, seg.mul, d)
        out_chunks.append((a * b).sum(dim=-1))  # (N, mul)
        offset += width
    if offset != x1.size(1):
        raise ValueError(f"Irreps dim mismatch: parsed={offset}, tensor_dim={x1.size(1)}")
    return torch.cat(out_chunks, dim=-1) if out_chunks else x1.new_zeros((N, 0))


def _elementwise_tp_invariants_ir_mul(x1: torch.Tensor, x2: torch.Tensor, segs: Sequence[_IrrepSeg]) -> torch.Tensor:
    """
    Same invariants as `_elementwise_tp_invariants`, but for `cue.ir_mul` layout.

    In `ir_mul`, for each segment (mul x l{e/o}), values are arranged as:
      [m0: mul channels][m1: mul channels]...[m(2l): mul channels]
    so the slice reshapes to (N, d, mul) where d=2l+1.
    """
    if x1.shape != x2.shape:
        raise ValueError(f"x1 and x2 must have same shape, got {x1.shape} vs {x2.shape}")
    N = x1.size(0)
    out_chunks: List[torch.Tensor] = []
    offset = 0
    for seg in segs:
        d = 2 * seg.l + 1
        width = seg.mul * d
        a = x1[:, offset : offset + width].reshape(N, d, seg.mul)
        b = x2[:, offset : offset + width].reshape(N, d, seg.mul)
        out_chunks.append((a * b).sum(dim=1))  # (N, mul)
        offset += width
    if offset != x1.size(1):
        raise ValueError(f"Irreps dim mismatch: parsed={offset}, tensor_dim={x1.size(1)}")
    return torch.cat(out_chunks, dim=-1) if out_chunks else x1.new_zeros((N, 0))


class _PureTorchElementwiseTP(nn.Module):
    """Pure-PyTorch replacement for cuet.EquivariantTensorProduct(..., filter=["0e"]).

    Computes per-irrep scalar invariants in ir_mul layout via CG-normalized
    dot products: for each segment (mul, l), output[u] = sum_m(x1[m,u]*x2[m,u]) / sqrt(2l+1).
    The 1/sqrt(2l+1) factor matches the Clebsch-Gordan coefficient for (l,l)->0.
    """

    def __init__(self, segs: List[_IrrepSeg]):
        super().__init__()
        self._dims: List[Tuple[int, int]] = [(seg.mul, 2 * seg.l + 1) for seg in segs]

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        N = x1.size(0)
        out_chunks: List[torch.Tensor] = []
        offset = 0
        for mul, d in self._dims:
            width = mul * d
            a = x1[:, offset : offset + width].reshape(N, d, mul)
            b = x2[:, offset : offset + width].reshape(N, d, mul)
            out_chunks.append((a * b).sum(dim=1) * (d ** -0.5))
            offset += width
        return torch.cat(out_chunks, dim=-1) if out_chunks else x1.new_zeros((N, 0))


class _CueChannelwiseEdgeConv(nn.Module):
    """
    Core channelwise edge convolution in cuEquivariance:
      node_feats -> Linear -> ChannelWiseTensorProduct(node_feats[senders], edge_sh, w(edge_radial)) -> scatter_sum -> Linear -> /avg_num_neighbors
    """

    def __init__(
        self,
        irreps_node_input: str,
        irreps_node_output: str,
        *,
        edge_attrs_lmax: int,
        r_max: float,
        number_of_basis: int,
        radial_hidden: Sequence[int],
        function_type: str = "gaussian",
        avg_num_neighbors: float | None = None,
        device=None,
        dtype=None,
        force_naive: bool = False,
    ):
        super().__init__()
        cue, cuet, O3_e3nn = _require_cue()
        dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_fast = (dev.type == "cuda") and not force_naive

        self.r_max = float(r_max)
        self.number_of_basis = int(number_of_basis)
        self.function_type = str(function_type)
        self.avg_num_neighbors = None if avg_num_neighbors is None else float(avg_num_neighbors)
        self.edge_attrs_lmax = int(edge_attrs_lmax)

        self.irreps_node_input_str = str(irreps_node_input)
        self.irreps_node_output_str = str(irreps_node_output)
        self.irreps_edge_attrs_str = _sh_irreps_string(self.edge_attrs_lmax)

        self.irreps_node_input = cue.Irreps(O3_e3nn, self.irreps_node_input_str)
        self.irreps_node_output = cue.Irreps(O3_e3nn, self.irreps_node_output_str)
        self.irreps_edge_attrs = cue.Irreps(O3_e3nn, self.irreps_edge_attrs_str)

        layout = cue.ir_mul

        self.sh = cuet.SphericalHarmonics(
            ls=list(range(self.edge_attrs_lmax + 1)),
            normalize=True,
            device=device,
            math_dtype=dtype,
            method="uniform_1d" if use_fast else "naive",
        )

        self.linear_up = cuet.Linear(
            self.irreps_node_input,
            self.irreps_node_output,
            layout=layout,
            internal_weights=True,
            device=device,
            dtype=dtype,
            method="fused_tp" if use_fast else "naive",
        )

        filter_irreps_out = [ir for (_mul, ir) in self.irreps_node_output]  # type: ignore[assignment]
        self.conv_tp = cuet.ChannelWiseTensorProduct(
            self.irreps_node_output,
            self.irreps_edge_attrs,
            filter_irreps_out=filter_irreps_out,
            shared_weights=False,
            internal_weights=False,
            layout=layout,
            device=device,
            dtype=dtype,
            method="uniform_1d" if use_fast else "naive",
        )
        self.irreps_mid = self.conv_tp.irreps_out

        self.conv_tp_weights = _MLP(
            in_dim=self.number_of_basis,
            hidden=list(radial_hidden),
            out_dim=int(self.conv_tp.weight_numel),
        )

        self.linear = cuet.Linear(
            self.irreps_mid,
            self.irreps_node_output,
            layout=layout,
            internal_weights=True,
            device=device,
            dtype=dtype,
            method="fused_tp" if use_fast else "naive",
        )

        self._force_naive = force_naive

    def forward(
        self,
        node_feats: torch.Tensor,  # (N, irreps_node_input.dim)
        edge_vec: torch.Tensor,  # (E, 3)
        edge_length: torch.Tensor,  # (E,)
        senders: torch.Tensor,  # (E,)
        receivers: torch.Tensor,  # (E,)
        num_nodes: int,
    ) -> torch.Tensor:
        edge_attrs = self.sh(edge_vec)

        emb = _radial_embedding(
            edge_length,
            r_max=self.r_max,
            number_of_basis=self.number_of_basis,
            function_type=self.function_type,
        )
        w = self.conv_tp_weights(emb)

        x = self.linear_up(node_feats)

        if self._force_naive:
            # Decompose gather-TP-scatter so that torch.jit.trace keeps
            # the output size dynamic (node_feats.size(0) is symbolic).
            x_src = x[senders]
            tp_edge = self.conv_tp(x_src, edge_attrs, w)
            msg = node_feats.new_zeros(node_feats.size(0), tp_edge.size(1))
            msg = msg.scatter_add(0, receivers.unsqueeze(1).expand_as(tp_edge), tp_edge)
        else:
            msg = self.conv_tp(
                x, edge_attrs, w,
                indices_1=senders,
                indices_out=receivers,
                size_out=int(num_nodes),
            )
        msg = self.linear(msg)

        if self._force_naive:
            avg = senders.size(0) / max(node_feats.size(0), 1)
        elif self.avg_num_neighbors is None:
            avg = float(senders.numel()) / float(max(int(num_nodes), 1))
        else:
            avg = self.avg_num_neighbors
        msg = msg / max(avg, 1e-8)
        return msg


class E3Conv(nn.Module):
    """
    First convolution (node scalars -> irreps_output) with channelwise edge TP (cuEquivariance backend).
    Signature matches the original `e3nn_layers.E3Conv.forward`.
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
        device=None,
        dtype=None,
        force_naive: bool = False,
        **_unused,
    ):
        super().__init__()
        if main_hidden_sizes3 is None:
            main_hidden_sizes3 = [64, 32]
        if emb_number is None:
            emb_number = [64, 64, 64]

        default_dtype = torch.get_default_dtype() if dtype is None else dtype
        self.atom_embedding = nn.Embedding(
            num_embeddings=max_atomvalue, embedding_dim=embedding_dim, dtype=default_dtype
        )
        self.fitnet1 = MainNet2(
            input_size=embedding_dim, hidden_sizes=main_hidden_sizes3, output_size=output_size
        )

        self.node_irreps_in_str = f"{int(output_size)}x0e"
        self.node_irreps_out_str = str(irreps_output)

        self.conv = _CueChannelwiseEdgeConv(
            irreps_node_input=self.node_irreps_in_str,
            irreps_node_output=self.node_irreps_out_str,
            edge_attrs_lmax=edge_attrs_lmax,
            r_max=float(max_radius),
            number_of_basis=int(number_of_basis),
            radial_hidden=list(emb_number),
            function_type=str(function_type),
            avg_num_neighbors=avg_num_neighbors,
            device=device,
            dtype=default_dtype,
            force_naive=force_naive,
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
        node_scalars = self.fitnet1(atom_embeddings)

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
    Subsequent convolution (node irreps -> node irreps) with channelwise edge TP (cuEquivariance backend).
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
        device=None,
        dtype=None,
        force_naive: bool = False,
        **_unused,
    ):
        super().__init__()
        if emb_number is None:
            emb_number = [64, 64, 64]
        default_dtype = torch.get_default_dtype() if dtype is None else dtype

        self.conv = _CueChannelwiseEdgeConv(
            irreps_node_input=str(irreps_input_conv),
            irreps_node_output=str(irreps_output),
            edge_attrs_lmax=edge_attrs_lmax,
            r_max=float(max_radius),
            number_of_basis=int(number_of_basis),
            radial_hidden=list(emb_number),
            function_type=str(function_type),
            avg_num_neighbors=avg_num_neighbors,
            device=device,
            dtype=default_dtype,
            force_naive=force_naive,
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
    Channelwise edge convolution Transformer block with cuEquivariance backend.

    This mirrors the *used* forward path of `e3nn_layers_channelwise.E3_TransformerLayer_multi`:
      - run a stack of channelwise convolutions (num_interaction)
      - concatenate features across interactions
      - compute scalar mixing via a fully-connected TP (product_3)
      - compute per-irrep invariants (ElementwiseTensorProduct(..., ["0e"]) equivalent)
      - MLP readout -> weighted sum -> per-structure energy
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
        force_naive: bool = False,
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
        cue, cuet, O3_e3nn = _require_cue()

        if embed_size is None:
            embed_size = [128, 128, 128]
        if main_hidden_sizes3 is None:
            main_hidden_sizes3 = [64, 32]
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.num_interaction = int(num_interaction)
        if self.num_interaction < 2:
            raise ValueError(f"num_interaction must be >= 2, got {self.num_interaction}")
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

        emb_number = [64, 64, 64] if not isinstance(hidden_dim, list) else hidden_dim

        self.irreps_input_str = str(irreps_input)
        self._irreps_input_segs = _parse_irreps_e3nn(self.irreps_input_str)

        # --- Convolution stack (channelwise edge conv) ---
        self.e3_conv_layers = nn.ModuleList()
        self.e3_conv_layers.append(
            E3Conv(
                max_radius=max_embed_radius,
                number_of_basis=main_number_of_basis,
                irreps_output=self.irreps_input_str,
                embedding_dim=embedding_dim,
                max_atomvalue=max_atomvalue,
                output_size=output_size,
                main_hidden_sizes3=main_hidden_sizes3,
                emb_number=emb_number,
                function_type=function_type_main,
                edge_attrs_lmax=2,
                device=self.device,
                force_naive=force_naive,
            )
        )
        for _ in range(1, self.num_interaction):
            self.e3_conv_layers.append(
                E3Conv2(
                    max_radius=max_embed_radius,
                    number_of_basis=main_number_of_basis,
                    irreps_input_conv=self.irreps_input_str,
                    irreps_output=self.irreps_input_str,
                    emb_number=emb_number,
                    function_type=function_type_main,
                    edge_attrs_lmax=2,
                    device=self.device,
                    force_naive=force_naive,
                )
            )

        # --- product_3: scalar mixing TP (cuEquivariance) ---
        irreps_input_multi_segs = _scale_irreps(self._irreps_input_segs, self.num_interaction)
        self.irreps_input_multi_str = _irreps_to_string(irreps_input_multi_segs)

        layout = cue.ir_mul
        self._cue_irreps_in_multi = cue.Irreps(O3_e3nn, self.irreps_input_multi_str)
        zeroe_filter = [ir for (_mul, ir) in cue.Irreps(O3_e3nn, "0e")]
        product_3_desc = cue.descriptors.elementwise_tensor_product(
            self._cue_irreps_in_multi,
            self._cue_irreps_in_multi,
            irreps3_filter=zeroe_filter,
        )
        self.product_3 = cuet.EquivariantTensorProduct(
            product_3_desc,
            layout=layout,
            device=self.device,
            math_dtype=torch.get_default_dtype(),
        )
        self._cue_irreps_scalar_out = product_3_desc.outputs[0].irreps
        self.scalar_channels = int(product_3_desc.outputs[0].dim)
        self.product_3_out_str = str(self._cue_irreps_scalar_out)

        # --- product_5 equivalent: ElementwiseTensorProduct(..., ["0e"]) with cuet native TP ---
        # In e3nn code: irreps_product_5 = irreps_input_multi + product_3.irreps_out
        product_5_in_str = f"{self.irreps_input_multi_str} + {self.product_3_out_str}"
        self._cue_product_5_in = cue.Irreps(O3_e3nn, product_5_in_str)
        product_5_desc = cue.descriptors.elementwise_tensor_product(
            self._cue_product_5_in,
            self._cue_product_5_in,
            irreps3_filter=zeroe_filter,
        )
        self.product_5 = cuet.EquivariantTensorProduct(
            product_5_desc,
            layout=layout,
            device=self.device,
            math_dtype=torch.get_default_dtype(),
        )
        product_5_out_dim = int(product_5_desc.outputs[0].dim)

        self.proj_total = MainNet(product_5_out_dim, embed_size, 17)
        self.weighted_sum = RobustScalarWeightedSum(17, init_weights="zero")
        self.long_range_module = build_long_range_module(
            mode=self.long_range_mode,
            feature_dim=product_5_out_dim,
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
            cutoff_radius=max_embed_radius,
        )
        self.long_range_num_k = getattr(self.long_range_module, "num_k", None) if self.long_range_module is not None else None
        self.feature_spectral_module = build_feature_spectral_module(
            mode=self.feature_spectral_mode,
            feature_dim=product_5_out_dim,
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

    def make_torchscript_portable(self) -> "E3_TransformerLayer_multi":
        """Replace cuet.EquivariantTensorProduct modules (product_3 / product_5)
        with pure-PyTorch equivalents so the traced graph contains no
        cuEquivariance custom ops.  Call *before* ``torch.jit.trace``.
        """
        p3_segs = _scale_irreps(self._irreps_input_segs, self.num_interaction)
        self.product_3 = _PureTorchElementwiseTP(p3_segs).to(self.device)

        p3_out_segs = [_IrrepSeg(mul=s.mul, l=0, parity="e") for s in p3_segs]
        p5_segs = p3_segs + p3_out_segs
        self.product_5 = _PureTorchElementwiseTP(p5_segs).to(self.device)
        return self

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
        sync_after_scatter=None,
        return_physical_tensors: bool = False,
        return_reciprocal_source: bool = False,
    ):
        if return_physical_tensors:
            raise ValueError("spherical-save-cue does not currently support return_physical_tensors=True")
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        sort_idx = torch.argsort(edge_dst)
        edge_src = edge_src[sort_idx]
        edge_dst = edge_dst[sort_idx]
        edge_shifts = edge_shifts[sort_idx]

        if precomputed_edge_vec is not None:
            edge_vec = precomputed_edge_vec.to(dtype=dtype)[sort_idx]
        else:
            edge_vec = None

        features: List[torch.Tensor] = []
        f_prev = self.e3_conv_layers[0](pos, A, batch, edge_src, edge_dst, edge_shifts, cell, precomputed_edge_vec=edge_vec)
        features.append(f_prev)
        for conv in self.e3_conv_layers[1:]:
            f_prev = conv(f_prev, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, precomputed_edge_vec=edge_vec)
            features.append(f_prev)

        f_combine = torch.cat(features, dim=-1)
        f_combine_product = self.product_3(f_combine, f_combine)  # (N, scalar_channels)

        T = torch.cat(features + [f_combine_product], dim=-1)
        f2_product_5 = self.product_5(T, T)  # (N, product_5_out_dim)
        feature_reciprocal_source = None
        if self.feature_spectral_module is not None:
            f2_product_5, feature_reciprocal_source = self.feature_spectral_module(f2_product_5, pos, batch, cell)

        long_range_energy = None
        reciprocal_source = None
        defer_long_range_to_runtime = False
        if self.long_range_module is not None:
            if return_reciprocal_source and bool(getattr(self.long_range_module, "exports_reciprocal_source", False)):
                long_range_energy, reciprocal_source = self.long_range_module(
                    f2_product_5,
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
                    f2_product_5,
                    pos,
                    batch,
                    cell,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                )
        if reciprocal_source is None:
            reciprocal_source = feature_reciprocal_source

        product_proj = self.proj_total(f2_product_5)
        e_out = torch.cat([product_proj], dim=-1)
        e_out = (1) * self.weighted_sum(e_out)
        atom_energies = e_out.sum(dim=-1, keepdim=True)
        if long_range_energy is not None and not defer_long_range_to_runtime:
            atom_energies = atom_energies + long_range_energy
        if reciprocal_source is None and return_reciprocal_source:
            reciprocal_source = atom_energies.new_empty((atom_energies.size(0), 0))
        if return_reciprocal_source:
            assert reciprocal_source is not None
            return atom_energies, reciprocal_source
        return atom_energies

