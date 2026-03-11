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
"""

from __future__ import annotations

import itertools
import torch
import torch.nn as nn
from molecular_force_field.utils.scatter import scatter
from e3nn.math import soft_one_hot_linspace
import math
from functools import lru_cache

from molecular_force_field.models.ictd_irreps import (
    HarmonicElementwiseProduct,
    HarmonicFullyConnectedTensorProduct,
    direction_harmonics,
    direction_harmonics_all,
    build_harmonic_projectors,
    build_harmonic_reconstructors,
    sym_dim,
)
from molecular_force_field.models.ictd_fast import _counts_list
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


@lru_cache(maxsize=None)
def _sym_rank_linear_indices_and_coefs(L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute:
      - linear indices into a flattened (..., 3,3,...,3) rank-L tensor (last axis flattened)
        for a canonical representative of each (a,b,c) monomial in Sym^L.
      - multinomial coefficients multinomial(L; a,b,c) that map the symmetric tensor entry
        to the polynomial coefficient t_{abc}.

    Ordering of (a,b,c) follows `molecular_force_field.models.ictd_fast._counts_list(L)`,
    which is the canonical order expected by ICTD projectors in `ictd_irreps`.

    Returns:
      lin_idx: (Dsym(L),) int64
      coefs:   (Dsym(L),) float64
    """
    L = int(L)
    if L < 0:
        raise ValueError(f"L must be >= 0, got {L}")
    counts = _counts_list(L)
    D = len(counts)
    if L == 0:
        return torch.zeros(1, dtype=torch.int64), torch.ones(1, dtype=torch.float64)

    # Flattening convention for a rank-L tensor shaped (3,)*L:
    # linear index = sum_{k=0..L-1} i_k * 3^{L-1-k}
    pow3 = [3 ** (L - 1 - k) for k in range(L)]
    lin = torch.empty(D, dtype=torch.int64)
    coefs = torch.empty(D, dtype=torch.float64)
    for d, (a, b, c) in enumerate(counts):
        idx_seq = ([0] * a) + ([1] * b) + ([2] * c)  # canonical representative
        if len(idx_seq) != L:
            raise RuntimeError("Internal error: bad exponent tuple length")
        li = 0
        for ik, pk in zip(idx_seq, pow3):
            li += int(ik) * int(pk)
        lin[d] = li
        coefs[d] = float(math.factorial(L) / (math.factorial(a) * math.factorial(b) * math.factorial(c)))
    return lin.contiguous(), coefs.contiguous()


@lru_cache(maxsize=None)
def _sym_rank_linear_permutation_indices(L: int) -> tuple[list[torch.Tensor], torch.Tensor]:
    """
    For each canonical (a,b,c) monomial in Sym^L, return the flattened linear
    indices of all unique cartesian permutations and the matching multinomial
    coefficient.  This is the inverse lookup for `_sym_rank_linear_indices_and_coefs`.
    """
    counts = _counts_list(L)
    if L == 0:
        return [torch.zeros(1, dtype=torch.int64)], torch.ones(1, dtype=torch.float64)

    pow3 = [3 ** (L - 1 - k) for k in range(L)]
    perm_indices: list[torch.Tensor] = []
    coefs = []
    for (a, b, c) in counts:
        idx_seq = ([0] * a) + ([1] * b) + ([2] * c)
        uniq = sorted(set(itertools.permutations(idx_seq, L)))
        lin = []
        for tup in uniq:
            li = 0
            for ik, pk in zip(tup, pow3):
                li += int(ik) * int(pk)
            lin.append(li)
        perm_indices.append(torch.tensor(lin, dtype=torch.int64))
        coefs.append(float(math.factorial(L) / (math.factorial(a) * math.factorial(b) * math.factorial(c))))
    return perm_indices, torch.tensor(coefs, dtype=torch.float64)


class PhysicalTensorICTDEmbedding(nn.Module):
    """
    Embed a (symmetric) Cartesian physical tensor of rank L into ICTD-irreps blocks.

    Supported inputs (per-sample, with optional channel/multiplicity dimension):
      - cartesian symmetric tensor: (..., Cin, 3, 3, ..., 3) with L trailing 3's
        (assumes symmetry across index permutations; a single canonical entry is used)
      - monomial coefficients (Sym^L): (..., Cin, Dsym(L)) where Dsym(L) = (L+2 choose 2)

    Output:
      - dict l -> (..., Cout, 2l+1) for all l=0..lmax_out (missing l are zero)
      - or a flattened (..., Cout*(lmax_out+1)^2) vector compatible with this model's
        internal irreps layout (concatenated l-blocks).

    Internals:
      Uses the trace-chain projectors from `ictd_irreps.build_harmonic_projectors` to decompose
      Sym^L into harmonic irreps blocks l=L, L-2, ..., (0 or 1). This matches how ICTD builds
      Cartesian tensor products: polynomial multiplication in Sym-space followed by trace-chain
      projection into irreps.
    """

    def __init__(
        self,
        *,
        rank: int,
        lmax_out: int,
        channels_in: int = 1,
        channels_out: int | None = None,
        input_repr: str = "cartesian",
        include_trace_chain: bool = True,
        internal_compute_dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.rank = int(rank)
        self.lmax_out = int(lmax_out)
        self.channels_in = int(channels_in)
        self.channels_out = int(channels_out) if channels_out is not None else int(channels_in)
        self.input_repr = str(input_repr).strip().lower()
        self.include_trace_chain = bool(include_trace_chain)
        self.internal_compute_dtype = internal_compute_dtype

        if self.rank < 0:
            raise ValueError(f"rank must be >= 0, got {self.rank}")
        if self.lmax_out < 0:
            raise ValueError(f"lmax_out must be >= 0, got {self.lmax_out}")
        if self.channels_in <= 0:
            raise ValueError(f"channels_in must be positive, got {self.channels_in}")
        if self.channels_out <= 0:
            raise ValueError(f"channels_out must be positive, got {self.channels_out}")
        if self.input_repr not in ("cartesian", "monomial"):
            raise ValueError(f"input_repr must be 'cartesian' or 'monomial', got {self.input_repr!r}")

        # Precompute Sym^L lookup for cartesian -> monomial coefficients
        lin_idx, coefs = _sym_rank_linear_indices_and_coefs(self.rank)
        self.register_buffer("_sym_lin_idx_cpu", lin_idx, persistent=False)
        self.register_buffer("_sym_multinom_cpu", coefs, persistent=False)

        # Per-l channel mixing (Cin -> Cout). Only needed for l present in trace chain and <= lmax_out.
        self._adapters = nn.ModuleDict()
        for l in range(self.lmax_out + 1):
            if self.channels_in == self.channels_out:
                self._adapters[str(l)] = nn.Identity()
            else:
                self._adapters[str(l)] = nn.Linear(self.channels_in, self.channels_out, bias=False)

        # Cache projectors per (device, dtype) to avoid repeated .to() allocations.
        # key: (device_str, dtype_str) -> dict l -> P (2l+1, Dsym(rank)) in internal_compute_dtype
        self._proj_cache: dict[tuple[str, str], dict[int, torch.Tensor]] = {}

        self.output_dim = _irreps_total_dim(self.channels_out, self.lmax_out)

    def _get_projectors(self, device: torch.device, dtype: torch.dtype) -> dict[int, torch.Tensor]:
        # Use internal_compute_dtype for stability; output casting happens later.
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._proj_cache.get(key)
        if cached is not None:
            return cached

        proj = build_harmonic_projectors(Lmax=self.rank)  # cached CPU float64
        D = sym_dim(self.rank)

        P_by_l: dict[int, torch.Tensor] = {}
        if self.include_trace_chain:
            ls = list(range(self.rank, -1, -2))
        else:
            ls = [self.rank]
        for l in ls:
            if l > self.lmax_out:
                continue
            P = proj.P[(self.rank, l)]  # (2l+1, D) CPU float64
            if P.shape[-1] != D:
                raise RuntimeError("ICTD projector has unexpected Sym dimension")
            P_by_l[l] = P.to(device=device, dtype=compute_dtype).contiguous()

        self._proj_cache[key] = P_by_l
        return P_by_l

    def _ensure_channel_dim_monomial(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (..., D) or (..., Cin, D) where channel dim is -2.
        if x.dim() >= 2 and x.shape[-2] == self.channels_in:
            return x
        return x.unsqueeze(-2)

    def _ensure_channel_dim_cartesian(self, T: torch.Tensor) -> torch.Tensor:
        # Accept (..., Cin, 3,3,...,3) where channel dim is -(rank+1),
        # or (..., 3,3,...,3) when channels_in==1 (we insert the Cin dim).
        if self.rank == 0:
            # Scalar: accept (..., Cin) or (...) when Cin==1.
            if T.dim() >= 1 and T.shape[-1] == self.channels_in:
                return T
            if self.channels_in == 1:
                return T.unsqueeze(-1)
            raise ValueError(f"rank=0 cartesian tensor must have trailing channels_in={self.channels_in}; got shape={tuple(T.shape)}")

        if T.dim() >= (self.rank + 1) and T.shape[-(self.rank + 1)] == self.channels_in:
            return T
        if self.channels_in == 1:
            return T.unsqueeze(-(self.rank + 1))
        raise ValueError(f"Expected channels_in={self.channels_in} at dim -{self.rank + 1}, got shape={tuple(T.shape)}")

    def _to_monomial_coeffs(self, T: torch.Tensor) -> torch.Tensor:
        """
        Convert a symmetric cartesian tensor (..., Cin, 3,3,...,3) to monomial coefficients
        (..., Cin, Dsym(rank)) consistent with ICTD ordering.
        """
        if self.rank == 0:
            Tch = self._ensure_channel_dim_cartesian(T)  # (..., Cin)
            return Tch.unsqueeze(-1)  # (..., Cin, 1)

        # Expect L trailing 3's (channel dim may be absent when channels_in==1)
        if T.dim() < self.rank:
            raise ValueError(f"cartesian input must have at least rank={self.rank} dims; got shape={tuple(T.shape)}")
        if tuple(T.shape[-self.rank:]) != (3,) * self.rank:
            raise ValueError(f"cartesian input last {self.rank} dims must be all 3; got shape={tuple(T.shape)}")

        T = self._ensure_channel_dim_cartesian(T)

        # Flatten the 3^L cartesian indices and gather one canonical representative per (a,b,c).
        lin_idx = self._sym_lin_idx_cpu.to(device=T.device)
        coefs = self._sym_multinom_cpu.to(device=T.device, dtype=T.dtype)
        flat = T.reshape(*T.shape[:-self.rank], 3 ** self.rank)  # (..., Cin, 3^L)
        vals = flat.index_select(-1, lin_idx)  # (..., Cin, Dsym)
        return vals * coefs

    def forward(self, tensor: torch.Tensor, *, return_blocks: bool = False) -> torch.Tensor | dict[int, torch.Tensor]:
        """
        Args:
          tensor:
            - if input_repr="cartesian": (..., Cin, 3,3,...,3) (rank trailing dims)
            - if input_repr="monomial": (..., Cin, Dsym(rank))
          return_blocks: if True, return dict l->(..., Cout, 2l+1); else return flattened (..., Cout*(lmax_out+1)^2)
        """
        dtype = tensor.dtype
        device = tensor.device

        if self.input_repr == "cartesian":
            t = self._to_monomial_coeffs(tensor)  # (..., Cin, Dsym)
        else:
            t = self._ensure_channel_dim_monomial(tensor)
            D = sym_dim(self.rank)
            if t.shape[-1] != D:
                raise ValueError(f"monomial input last dim must be Dsym(rank)={D}, got {t.shape[-1]}")
            if t.shape[-2] != self.channels_in:
                raise ValueError(f"monomial input channel dim must be channels_in={self.channels_in}, got {t.shape[-2]}")

        # Project Sym^L -> harmonic blocks using trace-chain projectors
        P_by_l = self._get_projectors(device=device, dtype=dtype)
        compute_dtype = self.internal_compute_dtype
        t_comp = t.to(dtype=compute_dtype) if t.dtype != compute_dtype else t

        out_blocks: dict[int, torch.Tensor] = {}
        batch_shape = t.shape[:-2]
        for l in range(self.lmax_out + 1):
            if l in P_by_l:
                P = P_by_l[l]  # (2l+1, Dsym) compute_dtype
                # (..., Cin, D) x (2l+1, D) -> (..., Cin, 2l+1)
                c = torch.einsum("...cd,md->...cm", t_comp, P)
                c = c.to(dtype=dtype) if c.dtype != dtype else c
                c = _apply_channel_adapter_per_l(c, self._adapters[str(l)])  # (..., Cout, 2l+1)
                out_blocks[l] = c
            else:
                out_blocks[l] = torch.zeros(*batch_shape, self.channels_out, 2 * l + 1, device=device, dtype=dtype)

        if return_blocks:
            return out_blocks
        return _merge_irreps(out_blocks, self.channels_out, self.lmax_out)


class PhysicalTensorICTDRecovery(nn.Module):
    """
    Recover a symmetric Cartesian tensor from ICTD irreps blocks.

    This is the inverse companion of :class:`PhysicalTensorICTDEmbedding` at the
    representation level:
      irreps blocks -> Sym^L monomial coefficients -> symmetric Cartesian tensor.

    Notes
    -----
    - Exact recovery requires that the provided blocks contain the complete trace
      chain for the target rank (e.g. l=0 and l=2 for a general symmetric rank-2 tensor).
    - If only the highest-l block is given (e.g. l=2), the recovered Cartesian
      tensor is the corresponding traceless symmetric component.
    """

    def __init__(
        self,
        *,
        rank: int,
        channels_in: int = 1,
        lmax_in: int | None = None,
        include_trace_chain: bool = True,
        internal_compute_dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.rank = int(rank)
        self.channels_in = int(channels_in)
        self.lmax_in = int(lmax_in) if lmax_in is not None else int(rank)
        self.include_trace_chain = bool(include_trace_chain)
        self.internal_compute_dtype = internal_compute_dtype

        if self.rank < 0:
            raise ValueError(f"rank must be >= 0, got {self.rank}")
        if self.channels_in <= 0:
            raise ValueError(f"channels_in must be positive, got {self.channels_in}")
        if self.lmax_in < 0:
            raise ValueError(f"lmax_in must be >= 0, got {self.lmax_in}")

        self._recon_cache: dict[tuple[str, str], dict[int, torch.Tensor]] = {}

    def _get_reconstructors(self, device: torch.device, dtype: torch.dtype) -> dict[int, torch.Tensor]:
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._recon_cache.get(key)
        if cached is not None:
            return cached

        recon = build_harmonic_reconstructors(Lmax=self.rank)
        if self.include_trace_chain:
            ls = list(range(self.rank, -1, -2))
        else:
            ls = [self.rank]

        out: dict[int, torch.Tensor] = {}
        for l in ls:
            if l > self.lmax_in:
                continue
            V = recon.V[(self.rank, l)]
            out[l] = V.to(device=device, dtype=compute_dtype).contiguous()
        self._recon_cache[key] = out
        return out

    def _ensure_channel_dim_blocks(self, blk: torch.Tensor, l: int) -> torch.Tensor:
        if blk.shape[-1] != 2 * l + 1:
            raise ValueError(f"irreps block for l={l} must end with 2l+1={2*l+1}, got shape={tuple(blk.shape)}")
        if blk.dim() >= 2 and blk.shape[-2] == self.channels_in:
            return blk
        if self.channels_in == 1:
            return blk.unsqueeze(-2)
        raise ValueError(
            f"Expected channels_in={self.channels_in} at dim -2 for l={l} block, got shape={tuple(blk.shape)}"
        )

    def _to_monomial_coeffs(self, blocks: dict[int, torch.Tensor]) -> torch.Tensor:
        device = next(iter(blocks.values())).device
        dtype = next(iter(blocks.values())).dtype
        compute_dtype = self.internal_compute_dtype
        recons = self._get_reconstructors(device=device, dtype=dtype)
        D = sym_dim(self.rank)

        sample_block = next(iter(blocks.values()))
        batch_shape = self._ensure_channel_dim_blocks(sample_block, next(iter(blocks.keys()))).shape[:-2]
        t = torch.zeros(*batch_shape, self.channels_in, D, device=device, dtype=compute_dtype)

        for l, V in recons.items():
            blk = blocks.get(l)
            if blk is None:
                continue
            blk = self._ensure_channel_dim_blocks(blk, l)
            blk_comp = blk.to(dtype=compute_dtype) if blk.dtype != compute_dtype else blk
            t = t + torch.einsum("...cm,dm->...cd", blk_comp, V)
        return t.to(dtype=dtype) if t.dtype != dtype else t

    def _monomial_to_cartesian(self, t: torch.Tensor, *, squeeze_channel: bool) -> torch.Tensor:
        if self.rank == 0:
            out = t[..., 0]
            return out.squeeze(-1) if squeeze_channel and self.channels_in == 1 else out

        perm_indices, coefs = _sym_rank_linear_permutation_indices(self.rank)
        coefs = coefs.to(device=t.device, dtype=t.dtype)
        flat = torch.zeros(*t.shape[:-1], 3 ** self.rank, device=t.device, dtype=t.dtype)
        for d, idxs_cpu in enumerate(perm_indices):
            idxs = idxs_cpu.to(device=t.device)
            vals = (t[..., d] / coefs[d]).unsqueeze(-1).expand(*t.shape[:-1], idxs.numel())
            idxs = idxs.view(*([1] * (flat.dim() - 1)), idxs.numel()).expand(*t.shape[:-1], idxs.numel())
            flat.scatter_(-1, idxs, vals)
        out = flat.reshape(*t.shape[:-1], *([3] * self.rank))
        if squeeze_channel and self.channels_in == 1:
            out = out.squeeze(-self.rank - 1)
        return out

    def forward(
        self,
        blocks: dict[int, torch.Tensor] | torch.Tensor,
        *,
        input_repr: str = "blocks",
        squeeze_channel: bool | None = None,
        return_monomial: bool = False,
    ) -> torch.Tensor:
        """
        Args:
          blocks:
            - if input_repr="blocks": dict l->(..., Cin, 2l+1) or dict l->(..., 2l+1) for Cin=1
            - if input_repr="flat": (..., Cin*(lmax_in+1)^2)
          squeeze_channel:
            If True and channels_in==1, drop the channel dimension in the returned
            Cartesian tensor to match common `(B,3)` / `(B,3,3)` label shapes.
        """
        if squeeze_channel is None:
            squeeze_channel = (self.channels_in == 1)

        if input_repr == "flat":
            if not torch.is_tensor(blocks):
                raise TypeError("input_repr='flat' expects a tensor input")
            blocks_dict = _split_irreps(blocks, self.channels_in, self.lmax_in)
        elif input_repr == "blocks":
            if torch.is_tensor(blocks):
                raise TypeError("input_repr='blocks' expects a dict[int, Tensor] input")
            blocks_dict = blocks
        else:
            raise ValueError(f"input_repr must be 'blocks' or 'flat', got {input_repr!r}")

        t = self._to_monomial_coeffs(blocks_dict)
        if return_monomial:
            return t.squeeze(-2) if squeeze_channel and self.channels_in == 1 else t
        return self._monomial_to_cartesian(t, squeeze_channel=squeeze_channel)

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
    First convolution in ICTD-irreps space (channelwise form: no neighbor in TP).

      scalar(Ai) ⊗ Y_l(n) -> f_in (output_size copies per l)
      then TP(f_in, edge_Y; weights(r)) -> channels_out irreps  [second operand is edge geometry only, mul=1]
      scatter_sum to receivers, normalize by avg_num_neighbors (global).
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
        # Normalize messages by this (default None = use num_edges/num_nodes at runtime)
        avg_num_neighbors: float | None = None,
        # Internal computation dtype for ICTD operations (default: float64 for stability)
        internal_compute_dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.channels_out = channels_out
        self.output_size = output_size
        self.lmax = lmax
        self.function_type = function_type
        self.avg_num_neighbors = avg_num_neighbors

        self.atom_embedding = nn.Embedding(max_atomvalue, embedding_dim)
        self.atom_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, output_size),
        )

        # Channelwise: (node ⊗ edge_Y) only; second input has mul=1 (edge geometry)
        self.tp2 = HarmonicFullyConnectedTensorProduct(
            mul_in1=output_size,
            mul_in2=1,
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

        self.output_dim = _irreps_total_dim(channels_out, lmax)

    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, *, precomputed_n=None, precomputed_edge_length=None, precomputed_Y_list=None):
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

        if precomputed_Y_list is None:
            Y_list = direction_harmonics_all(n, self.lmax)
        else:
            Y_list = precomputed_Y_list
        Y = {l: Y_list[l] for l in range(self.lmax + 1)}  # (E, 2l+1)

        f_in = {l: Ai[edge_src].unsqueeze(-1) * Y[l].unsqueeze(-2) for l in range(self.lmax + 1)}  # (E, output_size, 2l+1)
        # Second operand: edge geometry only (mul=1), no neighbor
        x2 = {l: Y_list[l].unsqueeze(-2) for l in range(self.lmax + 1)}  # (E, 1, 2l+1)

        emb = soft_one_hot_linspace(edge_length, 0.0, self.max_radius, self.number_of_basis, basis=self.function_type, cutoff=True)
        emb = emb.mul(self.number_of_basis ** 0.5).to(dtype=Ai.dtype)
        gates = self.fc(emb)
        out_blocks = self.tp2(f_in, x2, gates)
        edge_features = _merge_irreps(out_blocks, self.channels_out, self.lmax)

        num_nodes = pos.size(0)
        out = scatter(edge_features, edge_dst, dim=0, dim_size=num_nodes, reduce="sum")
        avg = self.avg_num_neighbors if self.avg_num_neighbors is not None else (float(edge_src.numel()) / float(max(num_nodes, 1)))
        out = out / max(avg, 1e-8)
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

        self.max_radius = float(max_embed_radius)
        self.number_of_basis = int(main_number_of_basis)
        self.function_type = str(function_type_main)

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
        scalar_channels = (self.num_interaction - 1) * 32
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
            # Validate keys and values
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
        self.product_5 = HarmonicElementwiseProduct(
            lmax=self.lmax,
            mul=combined_channels,
            irreps_out="0e",
            internal_compute_dtype=internal_compute_dtype,
        )

        sum_mul = sum(self.product5_muls_by_l[l] for l in range(self.lmax + 1))
        self.proj_total = MainNet(self.num_interaction * sum_mul + scalar_channels, embed_size, 17)

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
    ):
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
        )  # (N, C*(lmax+1)^2)
        if sync_after_scatter is not None:
            f1 = sync_after_scatter(f1)
        features = [f1]

        # conv2..convN: node irreps x edge Y_l -> node irreps (channelwise); scatter_sum then / avg_num_neighbors
        num_nodes = pos.size(0)
        avg_num_neighbors = float(edge_src.numel()) / float(max(num_nodes, 1))
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
            f_next = scatter(edge_flat, edge_dst, dim=0, dim_size=num_nodes, reduce="sum") / max(avg_num_neighbors, 1e-8)
            if sync_after_scatter is not None:
                f_next = sync_after_scatter(f_next)
            features.append(f_next)

        f_combine = torch.cat(features, dim=-1)  # (N, nC*(lmax+1)^2)

        xb = _split_irreps(f_combine, self.channels * self.num_interaction, self.lmax)
        scalars = torch.zeros(f_combine.shape[0], (self.num_interaction - 1) * 32, device=f_combine.device, dtype=f_combine.dtype)
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

        product_proj = self.proj_total(f_prod5)
        e_out = self.weighted_sum(product_proj)
        out = e_out.sum(dim=-1, keepdim=True)
        if return_combined_features:
            return out, f_combine
        return out

