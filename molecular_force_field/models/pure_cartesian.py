"""
Pure Cartesian (index-based) tensor representations and tensor products.

Key idea:
  - Represent rank-L features as full Cartesian tensors with 3^L components
    (stored as (..., channels, 3, 3, ..., 3) with L copies of 3).
  - Build O(3)-equivariant bilinear maps using only:
      - outer products
      - Kronecker delta contractions (δ_ij)
      - Levi-Civita contractions (ε_ijk)
    No spherical harmonics, no CG / wigner_3j, no explicit parity bookkeeping.

This file provides:
  - utilities to pack/unpack rank tensors to/from flat vectors
  - rotation action on rank tensors (natural tensor representation)
  - PureCartesianTensorProduct: a learnable equivariant tensor product over ranks
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def rank_dim(L: int) -> int:
    return 3 ** L


def total_dim(channels: int, Lmax: int) -> int:
    return channels * sum(rank_dim(L) for L in range(Lmax + 1))

def total_dim_o3(channels: int, Lmax: int) -> int:
    """
    O(3) strict representation needs a Z2 grading: true tensor (even) vs pseudotensor (odd).
    We represent both, so dimension doubles.
    """
    return 2 * total_dim(channels, Lmax)


def _rank_shape(L: int) -> Tuple[int, ...]:
    return (3,) * L


def split_by_rank(x: torch.Tensor, channels: int, Lmax: int) -> Dict[int, torch.Tensor]:
    """
    Split a flat feature vector (..., channels * sum_{L=0..Lmax} 3^L)
    into rank tensors:
      rank 0: (..., channels)
      rank L>=1: (..., channels, 3, 3, ..., 3) (L times)
    """
    batch_shape = x.shape[:-1]
    out: Dict[int, torch.Tensor] = {}
    idx = 0
    for L in range(Lmax + 1):
        d = channels * rank_dim(L)
        block = x[..., idx:idx + d]
        idx += d
        if L == 0:
            out[L] = block.view(*batch_shape, channels)
        else:
            out[L] = block.view(*batch_shape, channels, *_rank_shape(L))
    return out


def split_by_rank_o3(x: torch.Tensor, channels: int, Lmax: int) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Split flat O(3)-graded features into blocks keyed by (s, L) where:
      - s=0: true tensor
      - s=1: pseudotensor (transforms with an extra det(R) factor)
    Each block has shape:
      - L=0: (..., channels)
      - L>0: (..., channels, 3,3,...,3)
    """
    batch_shape = x.shape[:-1]
    out: Dict[Tuple[int, int], torch.Tensor] = {}
    idx = 0
    for s in (0, 1):
        for L in range(Lmax + 1):
            d = channels * rank_dim(L)
            block = x[..., idx:idx + d]
            idx += d
            if L == 0:
                out[(s, L)] = block.view(*batch_shape, channels)
            else:
                out[(s, L)] = block.view(*batch_shape, channels, *_rank_shape(L))
    return out


def merge_by_rank(blocks: Dict[int, torch.Tensor], channels: int, Lmax: int) -> torch.Tensor:
    """
    Inverse of split_by_rank. Returns (..., channels * sum 3^L).
    """
    # Determine batch shape from a tensor block:
    # - rank 0 block shape: (*batch, channels)
    # - rank L block shape: (*batch, channels, 3,3,...)
    sample = blocks[0] if 0 in blocks else next(iter(blocks.values()))
    batch_shape = sample.shape[:-1]  # drop the channel axis only
    parts: List[torch.Tensor] = []
    for L in range(Lmax + 1):
        t = blocks[L]
        parts.append(t.reshape(*batch_shape, channels * rank_dim(L)))
    return torch.cat(parts, dim=-1)


def merge_by_rank_o3(blocks: Dict[Tuple[int, int], torch.Tensor], channels: int, Lmax: int) -> torch.Tensor:
    """
    Inverse of split_by_rank_o3. Concatenate (s=0 then s=1), and within each, L=0..Lmax.
    """
    sample = blocks[(0, 0)]
    batch_shape = sample.shape[:-1]
    parts: List[torch.Tensor] = []
    for s in (0, 1):
        for L in range(Lmax + 1):
            t = blocks[(s, L)]
            parts.append(t.reshape(*batch_shape, channels * rank_dim(L)))
    return torch.cat(parts, dim=-1)

def _o3_elementwise_tensor_product_0e_from_blocks(
    blocks1: Dict[Tuple[int, int], torch.Tensor],
    blocks2: Dict[Tuple[int, int], torch.Tensor],
    channels: int,
    Lmax: int,
) -> torch.Tensor:
    """
    Pure-Cartesian O(3) elementwise tensor product, filtered to 0e scalars only.

    Inputs are O(3)-graded blocks keyed by (s, L):
      - s=0: true tensor
      - s=1: pseudotensor

    For each rank L and each channel c, we compute the O(3)-even scalar:
        out[L,c] = <x1[(0,L)][c], x2[(0,L)][c]> + <x1[(1,L)][c], x2[(1,L)][c]>

    where the inner product is the full δ-contraction over all tensor indices.

    This mirrors e3nn's `o3.ElementwiseTensorProduct(..., ["0e"])` behavior:
      - it does NOT output pseudoscalars (0o), so we do not include cross terms (0·1 or 1·0).

    Returns:
      (..., channels * (Lmax+1)) with blocks concatenated in L=0..Lmax order.
    """
    batch_shape = blocks1[(0, 0)].shape[:-1]
    outs: List[torch.Tensor] = []
    for L in range(Lmax + 1):
        t10 = blocks1[(0, L)]
        t11 = blocks1[(1, L)]
        t20 = blocks2[(0, L)]
        t21 = blocks2[(1, L)]
        # Flatten tensor indices -> (..., channels, 3^L)
        if L == 0:
            t10f = t10.reshape(*batch_shape, channels, 1)
            t11f = t11.reshape(*batch_shape, channels, 1)
            t20f = t20.reshape(*batch_shape, channels, 1)
            t21f = t21.reshape(*batch_shape, channels, 1)
        else:
            t10f = t10.reshape(*batch_shape, channels, -1)
            t11f = t11.reshape(*batch_shape, channels, -1)
            t20f = t20.reshape(*batch_shape, channels, -1)
            t21f = t21.reshape(*batch_shape, channels, -1)
        # e3nn-style component normalization:
        # if each Cartesian component has unit variance, Σ_i x_i y_i has variance = dim.
        # Divide by sqrt(dim) to keep output components ~ unit scale.
        scale = 1.0 / math.sqrt(rank_dim(L))
        inv = ((t10f * t20f).sum(dim=-1) + (t11f * t21f).sum(dim=-1)) * scale  # (..., channels)
        outs.append(inv)
    return torch.cat(outs, dim=-1)


class PureCartesianElementwiseTensorProductO3(nn.Module):
    """
    O(3)-graded pure-cartesian analogue of `e3nn.o3.ElementwiseTensorProduct(..., ["0e"])`.

    It maps two O(3)-graded feature vectors of shape (..., total_dim_o3(channels, Lmax))
    to a concatenation of rank-wise 0e scalar invariants of shape (..., channels*(Lmax+1)).
    """

    def __init__(self, channels: int, Lmax: int):
        super().__init__()
        self.channels = int(channels)
        self.Lmax = int(Lmax)

    @property
    def dim_in(self) -> int:
        return total_dim_o3(self.channels, self.Lmax)

    @property
    def dim_out(self) -> int:
        return self.channels * (self.Lmax + 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        blocks1 = split_by_rank_o3(x1, self.channels, self.Lmax)
        blocks2 = split_by_rank_o3(x2, self.channels, self.Lmax)
        return _o3_elementwise_tensor_product_0e_from_blocks(blocks1, blocks2, self.channels, self.Lmax)


def epsilon_tensor(device=None, dtype=None) -> torch.Tensor:
    """
    Levi-Civita symbol ε_{ijk} with indices in {0,1,2}.
    """
    eps = torch.zeros(3, 3, 3, device=device, dtype=dtype)
    eps[0, 1, 2] = 1
    eps[1, 2, 0] = 1
    eps[2, 0, 1] = 1
    eps[2, 1, 0] = -1
    eps[1, 0, 2] = -1
    eps[0, 2, 1] = -1
    return eps


def rotate_rank_tensor(t: torch.Tensor, R: torch.Tensor, L: int, pseudo: int = 0) -> torch.Tensor:
    """
    Apply natural O(3) action to a rank-L tensor.

    t: (..., channels, 3,3,...,3) (L times)
    R: (..., 3, 3) or (3,3)
    """
    # For pseudotensors, include det(R) factor under O(3) reflections.
    # Under SO(3), det(R)=+1 so this is neutral.
    det = None
    if pseudo:
        if R.dim() == 2:
            det = torch.det(R).to(dtype=t.dtype)
        else:
            det = torch.det(R).to(dtype=t.dtype).view(*R.shape[:-2], *([1] * (t.dim() - R.dim())))
    if L == 0:
        return t if not pseudo else (t * det)
    # Build einsum: t_{... c i1 i2 ... iL} -> t'_{... c j1 j2 ... jL}
    # with R_{j k} acting on each index: j = R * i
    # Equation: ...c i1 i2 ... iL, j1 i1, j2 i2, ... -> ...c j1 j2 ...
    # We'll use letters for indices; Lmax in this project is small (<=4 typical).
    letters = "abcdefghijklmnopqrstuvwxyz"
    # Reserve distinct labels:
    #  - channel axis: use 'z' (not used elsewhere)
    #  - tensor indices: use from start of letters excluding 'z'
    assert 2 * L + 1 <= len(letters) - 1, "Rank too large for einsum index builder."
    c = "z"
    pool = [ch for ch in letters if ch != c]
    i_idx = pool[:L]              # i0..i_{L-1}
    j_idx = pool[L:2 * L]         # j0..j_{L-1}

    # t subscripts
    t_sub = "..." + c + "".join(i_idx)

    # R subscripts: each is j_k i_k
    r_subs = []
    for k in range(L):
        r_subs.append("..." + j_idx[k] + i_idx[k])

    out_sub = "..." + c + "".join(j_idx)
    eq = ",".join([t_sub] + r_subs) + "->" + out_sub

    # Broadcast R if needed
    if R.dim() == 2:
        R_use = R
        Rs = [R_use] * L
    else:
        Rs = [R] * L
    out = torch.einsum(eq, t, *Rs)
    if pseudo:
        out = out * det
    return out


def edge_rank_powers(edge_vec: torch.Tensor, Lmax: int, normalize: bool = True) -> Dict[int, torch.Tensor]:
    """
    Build rank-L edge tensors from a Cartesian edge vector.

    Returns dict:
      0: (..., 1) scalar 1
      1: (..., 3) unit direction n
      L: (..., 3,...,3) n^{⊗L} (full outer power)
    """
    device, dtype = edge_vec.device, edge_vec.dtype
    batch_shape = edge_vec.shape[:-1]
    if normalize:
        n = edge_vec / edge_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    else:
        n = edge_vec

    out: Dict[int, torch.Tensor] = {0: torch.ones(*batch_shape, 1, device=device, dtype=dtype)}
    if Lmax >= 1:
        out[1] = n  # (..., 3)
    for L in range(2, Lmax + 1):
        # n^{⊗L} via iterative outer product
        t = out[L - 1]  # (..., 3,...,3) with L-1 indices
        # append new index with broadcasting
        n_exp = n.view(*batch_shape, *([1] * (L - 1)), 3)
        out[L] = t.unsqueeze(-1) * n_exp
    return out


@dataclass(frozen=True)
class _Path:
    L1: int
    L2: int
    Lout: int
    k_delta: int
    use_epsilon: bool


def _enumerate_paths(Lmax: int) -> List[_Path]:
    """
    Enumerate canonical contraction paths producing ranks up to Lmax.
    For each (L1,L2) we allow:
      - delta contractions: Lout = L1+L2-2k
      - epsilon + delta:   Lout = L1+L2-2k-1
    """
    paths: List[_Path] = []
    for L1 in range(Lmax + 1):
        for L2 in range(Lmax + 1):
            max_k = min(L1, L2)
            for k in range(max_k + 1):
                Lout = L1 + L2 - 2 * k
                if 0 <= Lout <= Lmax:
                    paths.append(_Path(L1=L1, L2=L2, Lout=Lout, k_delta=k, use_epsilon=False))
            # epsilon consumes one index from each input, so need L1>=1 and L2>=1
            if L1 >= 1 and L2 >= 1:
                for k in range(min(L1 - 1, L2 - 1) + 1):
                    Lout = L1 + L2 - 2 * k - 1
                    if 0 <= Lout <= Lmax:
                        paths.append(_Path(L1=L1, L2=L2, Lout=Lout, k_delta=k, use_epsilon=True))
    return paths


def _enumerate_paths_sparse(
    Lmax: int,
    max_rank_other: int = 1,
    allow_epsilon: bool = True,
    k_policy: str = "k0",
) -> List[_Path]:
    """
    Enumerate a *sparser* set of contraction paths.

    Motivation:
      The dense pure-cartesian TP considers all (L1, L2) pairs up to Lmax, which is very expensive.
      In message passing, one operand is often a low-rank geometric feature (scalar/vector).
      This sparse enumerator keeps only interactions where at least one side has rank <= max_rank_other.

    This still guarantees strict O(3) equivariance because every remaining path is built
    purely from δ and ε contractions.
    """
    paths: List[_Path] = []

    # k policy:
    #  - if min(L1,L2)=0 (scalar-tensor): only k=0 (pure tensor product) makes sense
    #  - if min(L1,L2)=1 (vector-tensor): choose k according to k_policy:
    #       - "both": keep k in {0,1}  (gives Lout=L+1 and L-1)
    #       - "k0":   keep only k=0   (only L+1, promotes higher rank)
    #       - "k1":   keep only k=1   (only L-1, contracts back to lower rank)
    #    and optionally epsilon with k=0 only (gives Lout=L), to propagate pseudos when desired.
    if k_policy not in {"both", "k0", "k1"}:
        raise ValueError(f"Invalid k_policy={k_policy!r}. Must be one of: 'both', 'k0', 'k1'.")
    for L1 in range(Lmax + 1):
        for L2 in range(Lmax + 1):
            m = min(L1, L2)
            if m > max_rank_other:
                continue

            if m == 0:
                # delta-only, k=0
                Lout = L1 + L2
                if 0 <= Lout <= Lmax:
                    paths.append(_Path(L1=L1, L2=L2, Lout=Lout, k_delta=0, use_epsilon=False))
                continue

            if m == 1:
                # delta paths: k=0 and k=1 (if possible)
                if k_policy == "both":
                    k_list = (0, 1)
                elif k_policy == "k0":
                    k_list = (0,)
                else:
                    k_list = (1,)

                for k in k_list:
                    if k > min(L1, L2):
                        continue
                    Lout = L1 + L2 - 2 * k
                    if 0 <= Lout <= Lmax:
                        paths.append(_Path(L1=L1, L2=L2, Lout=Lout, k_delta=k, use_epsilon=False))

                # epsilon path: keep only k=0 for speed (Lout = L1+L2-1)
                if allow_epsilon and L1 >= 1 and L2 >= 1:
                    Lout = L1 + L2 - 1
                    if 0 <= Lout <= Lmax:
                        paths.append(_Path(L1=L1, L2=L2, Lout=Lout, k_delta=0, use_epsilon=True))

            # m>1 are excluded by max_rank_other

    return paths


def _einsum_for_path(L1: int, L2: int, k_delta: int, use_epsilon: bool) -> Tuple[str, List[str], List[str]]:
    """
    Build einsum equation for a canonical path.

    Inputs are:
      A: (..., a, i1..iL1)
      B: (..., b, j1..jL2)
    Output is:
      (..., a, b, out_indices...)  (channels preserved; channel mixing happens outside)

    Canonical pattern:
      - delta contractions: contract last k_delta indices pairwise: i_{L1-k+r} with j_{L2-k+r}
      - if use_epsilon: also consume the last remaining index of A and B (after delta) via ε_{p i j}
        producing a new output index p appended at the end.
    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    # Reserve: channel indices a,b and output epsilon index p if needed
    # tensor indices use letters
    # We'll keep channel labels 'a','b' explicitly.
    # Use tensor letters starting from 'c' to avoid collision with 'a','b'.
    tensor_letters = [ch for ch in letters if ch not in {"a", "b"}]

    # Allocate indices
    i_idx = tensor_letters[:L1]
    j_idx = tensor_letters[L1:L1 + L2]
    pos = L1 + L2

    # delta contractions on tails
    if k_delta > 0:
        for r in range(k_delta):
            i_tail = i_idx[L1 - k_delta + r]
            j_tail = j_idx[L2 - k_delta + r]
            # make them the same symbol (contract)
            # choose i_tail as the shared symbol
            j_idx[L2 - k_delta + r] = i_tail

    eps_terms: List[str] = []
    out_idx: List[str] = []

    # remaining free indices after delta
    i_free = i_idx[: L1 - k_delta]
    j_free = j_idx[: L2 - k_delta]

    if use_epsilon:
        # Need at least one free index on each side to feed epsilon.
        assert len(i_free) >= 1 and len(j_free) >= 1
        i_eps = i_free[-1]
        j_eps = j_free[-1]
        i_free = i_free[:-1]
        j_free = j_free[:-1]
        p = tensor_letters[pos]
        eps_terms = [f"{p}{i_eps}{j_eps}"]
        out_idx = i_free + j_free + [p]
    else:
        out_idx = i_free + j_free

    A_sub = "...a" + "".join(i_idx)
    B_sub = "...b" + "".join(j_idx)
    in_terms = [A_sub, B_sub]
    if use_epsilon:
        in_terms += eps_terms
    out_sub = "...ab" + "".join(out_idx)
    eq = ",".join(in_terms) + "->" + out_sub
    return eq, i_idx, j_idx


class PureCartesianTensorProduct(nn.Module):
    """
    Learnable O(3)-equivariant tensor product on full Cartesian rank tensors (3^L components).

    Representation:
      Input1: flat (..., C1 * sum_{L<=Lmax} 3^L)
      Input2: flat (..., C2 * sum_{L<=Lmax} 3^L)
      Output: flat (..., Cout * sum_{L<=Lmax} 3^L)

    For each output rank Lout and each contraction path producing that rank from (L1,L2),
    we compute a canonical δ/ε contraction (equivariant by construction), then mix channels
    with weights W[path, Cout, C1, C2].
    """

    def __init__(self, C1: int, C2: int, Cout: int, Lmax: int, internal_weights: bool = True):
        super().__init__()
        self.C1, self.C2, self.Cout, self.Lmax = C1, C2, Cout, Lmax
        self.internal_weights = internal_weights

        self.paths: List[_Path] = _enumerate_paths(Lmax)
        # Group by output rank for efficient accumulation
        self.paths_by_Lout: Dict[int, List[int]] = {L: [] for L in range(Lmax + 1)}
        for idx, p in enumerate(self.paths):
            self.paths_by_Lout[p.Lout].append(idx)

        # One weight tensor per path: (Cout, C1, C2)
        self.weight_numel = len(self.paths) * Cout * C1 * C2
        if internal_weights:
            self.weight = nn.Parameter(torch.randn(self.weight_numel) * 0.02)
        else:
            self.register_parameter("weight", None)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_shape = x1.shape[:-1]
        if self.internal_weights:
            w = self.weight
            per_sample = False
        else:
            assert weights is not None
            w = weights
            per_sample = w.dim() > 1 and w.shape[:-1] == batch_shape

        A = split_by_rank(x1, self.C1, self.Lmax)
        B = split_by_rank(x2, self.C2, self.Lmax)

        eps = epsilon_tensor(device=x1.device, dtype=x1.dtype)

        # Allocate output blocks
        out_blocks: Dict[int, torch.Tensor] = {}
        for L in range(self.Lmax + 1):
            if L == 0:
                out_blocks[L] = torch.zeros(*batch_shape, self.Cout, device=x1.device, dtype=x1.dtype)
            else:
                out_blocks[L] = torch.zeros(*batch_shape, self.Cout, *_rank_shape(L), device=x1.device, dtype=x1.dtype)

        # Iterate paths and accumulate
        w_idx = 0
        for p in self.paths:
            # e3nn-style component normalization for index contractions:
            # - each contracted index sums over 3 components
            # - delta: k_delta contracted indices
            # - epsilon: consumes 2 indices (one from each input) and sums them (3*3)
            n_contract = p.k_delta + (2 if p.use_epsilon else 0)
            contract_dim = (3 ** n_contract)
            norm = 1.0 / math.sqrt(contract_dim) if contract_dim > 1 else 1.0

            # slice weights for this path
            if per_sample:
                Wp = w[..., w_idx:w_idx + self.Cout * self.C1 * self.C2].view(*batch_shape, self.Cout, self.C1, self.C2)
            else:
                Wp = w[w_idx:w_idx + self.Cout * self.C1 * self.C2].view(self.Cout, self.C1, self.C2)
            w_idx += self.Cout * self.C1 * self.C2

            t1 = A[p.L1]
            t2 = B[p.L2]

            # Build contraction output with preserved input channels (C1,C2)
            if p.L1 == 0 and p.L2 == 0:
                # scalar * scalar -> scalar
                # out_ab = t1_a * t2_b
                out_ab = t1.unsqueeze(-1) * t2.unsqueeze(-2)  # (..., C1, C2)
            elif p.L1 == 0 and p.L2 > 0 and not p.use_epsilon and p.k_delta == 0:
                # scalar * tensor -> tensor, keep both channel axes: (..., C1, C2, idx...)
                idx_letters = "ijklmnop"[: p.L2]
                eq = f"...a,...b{idx_letters}->...ab{idx_letters}"
                out_ab = torch.einsum(eq, t1, t2)
            elif p.L2 == 0 and p.L1 > 0 and not p.use_epsilon and p.k_delta == 0:
                idx_letters = "ijklmnop"[: p.L1]
                eq = f"...a{idx_letters},...b->...ab{idx_letters}"
                out_ab = torch.einsum(eq, t1, t2)
            else:
                # General case via generated einsum
                t1_use = t1
                t2_use = t2
                eq, _, _ = _einsum_for_path(p.L1, p.L2, p.k_delta, p.use_epsilon)
                if p.use_epsilon:
                    out_ab = torch.einsum(eq, t1_use, t2_use, eps) * norm
                else:
                    out_ab = torch.einsum(eq, t1_use, t2_use) * norm

            # Mix channels: (..., C1, C2, tensor_indices...) with Wp -> (..., Cout, tensor_indices...)
            if p.Lout == 0:
                if per_sample:
                    out_c = torch.einsum("...cab,...ab->...c", Wp, out_ab)
                else:
                    out_c = torch.einsum("cab,...ab->...c", Wp, out_ab)
                out_blocks[0] = out_blocks[0] + out_c
            else:
                # bring out_ab to (..., C1, C2, idx...)
                # then contract channel indices
                tensor_nd = p.Lout
                # Build einsum dynamically: W(c,a,b) and out(a,b,idx...) -> out(c,idx...)
                idx_letters = "ijklmnop"[:tensor_nd]
                if per_sample:
                    eq2 = f"...cab,...ab{idx_letters}->...c{idx_letters}"
                else:
                    eq2 = f"cab,...ab{idx_letters}->...c{idx_letters}"
                out_c = torch.einsum(eq2, Wp, out_ab)
                out_blocks[p.Lout] = out_blocks[p.Lout] + out_c

        return merge_by_rank(out_blocks, self.Cout, self.Lmax)


class PureCartesianTensorProductO3(nn.Module):
    """
    O(3)-strict version of PureCartesianTensorProduct.

    We keep an internal Z2 grading:
      s=0: true tensor
      s=1: pseudotensor

    Parity bookkeeping is not done via explicit irreps labels, but via:
      s_out = s1 xor s2 xor use_epsilon
    because each Levi-Civita contraction introduces one factor of det(R).
    """

    def __init__(self, C1: int, C2: int, Cout: int, Lmax: int, internal_weights: bool = True):
        super().__init__()
        self.C1, self.C2, self.Cout, self.Lmax = C1, C2, Cout, Lmax
        self.internal_weights = internal_weights

        self.paths: List[_Path] = _enumerate_paths(Lmax)
        # 4 parity-combos (s1,s2) per path
        self.weight_numel = len(self.paths) * 4 * Cout * C1 * C2
        if internal_weights:
            self.weight = nn.Parameter(torch.randn(self.weight_numel) * 0.02)
        else:
            self.register_parameter("weight", None)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_shape = x1.shape[:-1]
        if self.internal_weights:
            w = self.weight
            per_sample = False
        else:
            assert weights is not None
            w = weights
            per_sample = w.dim() > 1 and w.shape[:-1] == batch_shape

        A = split_by_rank_o3(x1, self.C1, self.Lmax)
        B = split_by_rank_o3(x2, self.C2, self.Lmax)

        eps = epsilon_tensor(device=x1.device, dtype=x1.dtype)

        out_blocks: Dict[Tuple[int, int], torch.Tensor] = {}
        for s in (0, 1):
            for L in range(self.Lmax + 1):
                if L == 0:
                    out_blocks[(s, L)] = torch.zeros(*batch_shape, self.Cout, device=x1.device, dtype=x1.dtype)
                else:
                    out_blocks[(s, L)] = torch.zeros(*batch_shape, self.Cout, *_rank_shape(L), device=x1.device, dtype=x1.dtype)

        w_idx = 0
        for p in self.paths:
            # e3nn-style component normalization for index contractions
            n_contract = p.k_delta + (2 if p.use_epsilon else 0)
            contract_dim = (3 ** n_contract)
            norm = 1.0 / math.sqrt(contract_dim) if contract_dim > 1 else 1.0
            # For each (s1,s2) we have a distinct W tensor (Cout,C1,C2)
            for s1 in (0, 1):
                for s2 in (0, 1):
                    if per_sample:
                        Wp = w[..., w_idx:w_idx + self.Cout * self.C1 * self.C2].view(*batch_shape, self.Cout, self.C1, self.C2)
                    else:
                        Wp = w[w_idx:w_idx + self.Cout * self.C1 * self.C2].view(self.Cout, self.C1, self.C2)
                    w_idx += self.Cout * self.C1 * self.C2

                    t1 = A[(s1, p.L1)]
                    t2 = B[(s2, p.L2)]

                    # Contraction for tensor indices
                    if p.L1 == 0 and p.L2 == 0:
                        out_ab = t1.unsqueeze(-1) * t2.unsqueeze(-2)
                    elif p.L1 == 0 and p.L2 > 0 and (not p.use_epsilon) and p.k_delta == 0:
                        idx_letters = "ijklmnop"[: p.L2]
                        eq = f"...a,...b{idx_letters}->...ab{idx_letters}"
                        out_ab = torch.einsum(eq, t1, t2)
                    elif p.L2 == 0 and p.L1 > 0 and (not p.use_epsilon) and p.k_delta == 0:
                        idx_letters = "ijklmnop"[: p.L1]
                        eq = f"...a{idx_letters},...b->...ab{idx_letters}"
                        out_ab = torch.einsum(eq, t1, t2)
                    else:
                        eq, _, _ = _einsum_for_path(p.L1, p.L2, p.k_delta, p.use_epsilon)
                        if p.use_epsilon:
                            out_ab = torch.einsum(eq, t1, t2, eps) * norm
                        else:
                            out_ab = torch.einsum(eq, t1, t2) * norm

                    s_out = s1 ^ s2 ^ (1 if p.use_epsilon else 0)

                    if p.Lout == 0:
                        if per_sample:
                            out_c = torch.einsum("...cab,...ab->...c", Wp, out_ab)
                        else:
                            out_c = torch.einsum("cab,...ab->...c", Wp, out_ab)
                        out_blocks[(s_out, 0)] = out_blocks[(s_out, 0)] + out_c
                    else:
                        idx_letters = "ijklmnop"[: p.Lout]
                        if per_sample:
                            eq2 = f"...cab,...ab{idx_letters}->...c{idx_letters}"
                        else:
                            eq2 = f"cab,...ab{idx_letters}->...c{idx_letters}"
                        out_c = torch.einsum(eq2, Wp, out_ab)
                        out_blocks[(s_out, p.Lout)] = out_blocks[(s_out, p.Lout)] + out_c

        return merge_by_rank_o3(out_blocks, self.Cout, self.Lmax)


class PureCartesianTensorProductO3Sparse(nn.Module):
    """
    Sparse O(3)-strict pure-cartesian tensor product.

    This is a computationally cheaper variant of PureCartesianTensorProductO3 that
    *restricts* the set of rank-rank interactions to those where at least one operand
    has rank <= max_rank_other (default: 1, i.e. scalar/vector).

    It preserves strict O(3) equivariance (including reflections) because it is still
    constructed solely from δ and ε contractions, with the same Z2 grading rule:
        s_out = s1 xor s2 xor use_epsilon
    """

    def __init__(
        self,
        C1: int,
        C2: int,
        Cout: int,
        Lmax: int,
        max_rank_other: int = 1,
        allow_epsilon: bool = True,
        k_policy: str = "k0",
        share_parity_weights: bool = True,
        assume_pseudo_zero: bool = False,
        internal_weights: bool = True,
    ):
        super().__init__()
        self.C1, self.C2, self.Cout, self.Lmax = C1, C2, Cout, Lmax
        self.max_rank_other = max_rank_other
        self.allow_epsilon = allow_epsilon
        self.k_policy = k_policy
        self.share_parity_weights = share_parity_weights
        self.assume_pseudo_zero = assume_pseudo_zero
        self.internal_weights = internal_weights

        self.paths: List[_Path] = _enumerate_paths_sparse(
            Lmax,
            max_rank_other=max_rank_other,
            allow_epsilon=allow_epsilon,
            k_policy=k_policy,
        )
        # If epsilon is disabled and callers guarantee pseudo inputs are identically zero,
        # then only s1=s2=0 contributes and s_out is always 0. We can drop all other parity weights.
        if assume_pseudo_zero and (not allow_epsilon):
            parity_factor = 1
        else:
            parity_factor = 2 if share_parity_weights else 4
        self.weight_numel = len(self.paths) * parity_factor * Cout * C1 * C2
        if internal_weights:
            self.weight = nn.Parameter(torch.randn(self.weight_numel) * 0.02)
        else:
            self.register_parameter("weight", None)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_shape = x1.shape[:-1]
        if self.internal_weights:
            w = self.weight
            per_sample = False
        else:
            assert weights is not None
            w = weights
            per_sample = w.dim() > 1 and w.shape[:-1] == batch_shape

        A = split_by_rank_o3(x1, self.C1, self.Lmax)
        B = split_by_rank_o3(x2, self.C2, self.Lmax)
        eps = epsilon_tensor(device=x1.device, dtype=x1.dtype)

        out_blocks: Dict[Tuple[int, int], torch.Tensor] = {}
        for s in (0, 1):
            for L in range(self.Lmax + 1):
                if L == 0:
                    out_blocks[(s, L)] = torch.zeros(*batch_shape, self.Cout, device=x1.device, dtype=x1.dtype)
                else:
                    out_blocks[(s, L)] = torch.zeros(*batch_shape, self.Cout, *_rank_shape(L), device=x1.device, dtype=x1.dtype)

        w_idx = 0
        for p in self.paths:
            # e3nn-style component normalization for index contractions
            n_contract = p.k_delta + (2 if p.use_epsilon else 0)
            contract_dim = (3 ** n_contract)
            norm = 1.0 / math.sqrt(contract_dim) if contract_dim > 1 else 1.0
            # Fastest path: pseudo inputs are guaranteed 0 and epsilon disabled.
            # Only s1=s2=0 contributes, and s_out=0 always.
            if self.assume_pseudo_zero and (not self.allow_epsilon):
                if per_sample:
                    Wp = w[..., w_idx:w_idx + self.Cout * self.C1 * self.C2].view(*batch_shape, self.Cout, self.C1, self.C2)
                else:
                    Wp = w[w_idx:w_idx + self.Cout * self.C1 * self.C2].view(self.Cout, self.C1, self.C2)
                w_idx += self.Cout * self.C1 * self.C2

                t1 = A[(0, p.L1)]
                t2 = B[(0, p.L2)]

                if p.L1 == 0 and p.L2 == 0:
                    out_ab = t1.unsqueeze(-1) * t2.unsqueeze(-2)
                elif p.L1 == 0 and p.L2 > 0 and (not p.use_epsilon) and p.k_delta == 0:
                    idx_letters = "ijklmnop"[: p.L2]
                    eq = f"...a,...b{idx_letters}->...ab{idx_letters}"
                    out_ab = torch.einsum(eq, t1, t2)
                elif p.L2 == 0 and p.L1 > 0 and (not p.use_epsilon) and p.k_delta == 0:
                    idx_letters = "ijklmnop"[: p.L1]
                    eq = f"...a{idx_letters},...b->...ab{idx_letters}"
                    out_ab = torch.einsum(eq, t1, t2)
                else:
                    eq, _, _ = _einsum_for_path(p.L1, p.L2, p.k_delta, p.use_epsilon)
                    if p.use_epsilon:
                        out_ab = torch.einsum(eq, t1, t2, eps) * norm
                    else:
                        out_ab = torch.einsum(eq, t1, t2) * norm

                s_out = 0

                if p.Lout == 0:
                    if per_sample:
                        out_c = torch.einsum("...cab,...ab->...c", Wp, out_ab)
                    else:
                        out_c = torch.einsum("cab,...ab->...c", Wp, out_ab)
                    out_blocks[(s_out, 0)] = out_blocks[(s_out, 0)] + out_c
                else:
                    idx_letters = "ijklmnop"[: p.Lout]
                    if per_sample:
                        eq2 = f"...cab,...ab{idx_letters}->...c{idx_letters}"
                    else:
                        eq2 = f"cab,...ab{idx_letters}->...c{idx_letters}"
                    out_c = torch.einsum(eq2, Wp, out_ab)
                    out_blocks[(s_out, p.Lout)] = out_blocks[(s_out, p.Lout)] + out_c
                continue

            # General (keeps pseudo channels and/or epsilon paths)
            if self.share_parity_weights:
                # Two weights per path: indexed by s_out in {0,1}.
                if per_sample:
                    W0 = w[..., w_idx:w_idx + self.Cout * self.C1 * self.C2].view(*batch_shape, self.Cout, self.C1, self.C2)
                    w_idx += self.Cout * self.C1 * self.C2
                    W1 = w[..., w_idx:w_idx + self.Cout * self.C1 * self.C2].view(*batch_shape, self.Cout, self.C1, self.C2)
                    w_idx += self.Cout * self.C1 * self.C2
                else:
                    W0 = w[w_idx:w_idx + self.Cout * self.C1 * self.C2].view(self.Cout, self.C1, self.C2)
                    w_idx += self.Cout * self.C1 * self.C2
                    W1 = w[w_idx:w_idx + self.Cout * self.C1 * self.C2].view(self.Cout, self.C1, self.C2)
                    w_idx += self.Cout * self.C1 * self.C2
            else:
                W0 = W1 = None

            for s1 in (0, 1):
                for s2 in (0, 1):
                    if not self.share_parity_weights:
                        if per_sample:
                            Wp = w[..., w_idx:w_idx + self.Cout * self.C1 * self.C2].view(*batch_shape, self.Cout, self.C1, self.C2)
                        else:
                            Wp = w[w_idx:w_idx + self.Cout * self.C1 * self.C2].view(self.Cout, self.C1, self.C2)
                        w_idx += self.Cout * self.C1 * self.C2

                    t1 = A[(s1, p.L1)]
                    t2 = B[(s2, p.L2)]

                    if p.L1 == 0 and p.L2 == 0:
                        out_ab = t1.unsqueeze(-1) * t2.unsqueeze(-2)
                    elif p.L1 == 0 and p.L2 > 0 and (not p.use_epsilon) and p.k_delta == 0:
                        idx_letters = "ijklmnop"[: p.L2]
                        eq = f"...a,...b{idx_letters}->...ab{idx_letters}"
                        out_ab = torch.einsum(eq, t1, t2)
                    elif p.L2 == 0 and p.L1 > 0 and (not p.use_epsilon) and p.k_delta == 0:
                        idx_letters = "ijklmnop"[: p.L1]
                        eq = f"...a{idx_letters},...b->...ab{idx_letters}"
                        out_ab = torch.einsum(eq, t1, t2)
                    else:
                        eq, _, _ = _einsum_for_path(p.L1, p.L2, p.k_delta, p.use_epsilon)
                        if p.use_epsilon:
                            out_ab = torch.einsum(eq, t1, t2, eps) * norm
                        else:
                            out_ab = torch.einsum(eq, t1, t2) * norm

                    s_out = s1 ^ s2 ^ (1 if p.use_epsilon else 0)
                    if self.share_parity_weights:
                        Wp = W0 if s_out == 0 else W1

                    if p.Lout == 0:
                        if per_sample:
                            out_c = torch.einsum("...cab,...ab->...c", Wp, out_ab)
                        else:
                            out_c = torch.einsum("cab,...ab->...c", Wp, out_ab)
                        out_blocks[(s_out, 0)] = out_blocks[(s_out, 0)] + out_c
                    else:
                        idx_letters = "ijklmnop"[: p.Lout]
                        if per_sample:
                            eq2 = f"...cab,...ab{idx_letters}->...c{idx_letters}"
                        else:
                            eq2 = f"cab,...ab{idx_letters}->...c{idx_letters}"
                        out_c = torch.einsum(eq2, Wp, out_ab)
                        out_blocks[(s_out, p.Lout)] = out_blocks[(s_out, p.Lout)] + out_c

        return merge_by_rank_o3(out_blocks, self.Cout, self.Lmax)

