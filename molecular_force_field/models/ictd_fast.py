"""
Fast (precomputed) irreducible Cartesian tensor decomposition utilities.

This module provides a practical "ICTD-like" decomposition path that is fast for
small Lmax (<=6), without spherical harmonics and without CG/Wigner-3j.

Key idea for L >= 3:
  - Restrict to the *symmetric* subspace Sym^L(R^3), which is the relevant case
    for many geometric tensors (e.g. n^{⊗L}) and for STF/trace chains.
  - Use the polynomial / harmonic correspondence:
      Symmetric rank-L tensors <-> homogeneous polynomials of degree L
      STF tensors (l = L) <-> harmonic polynomials (Δ f = 0)
  - Build a G-orthonormal basis of the harmonic subspace via the nullspace of
    the Laplacian operator in the monomial basis, where the inner product is the
    tensor inner product (diagonal weights).
  - Precompute a projection matrix P_L : R^{3^L} -> R^{2L+1} such that
      y = P_L vec(T)
    yields the STF (l=L) coordinates of the symmetrized part of a rank-L tensor.

For L=2 (generic rank-2), we also provide the classical full ICTD decomposition:
  T = STF(l=2) + antisymmetric(l=1 pseudo) + trace(l=0).

Notes / Scope:
  - The precomputed P_L projects to the STF component of the *symmetric* part.
    For a generic rank-L tensor with non-symmetric Young components, this is not
    the full SO(3) irreducible decomposition.
  - This is intentionally engineered for speed and for Lmax <= 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def _counts_list(L: int) -> List[Tuple[int, int, int]]:
    """All (a,b,c) with a+b+c=L in a deterministic order."""
    out = []
    for a in range(L + 1):
        for b in range(L - a + 1):
            c = L - a - b
            out.append((a, b, c))
    return out


def _multinomial_count(L: int, a: int, b: int, c: int) -> int:
    """Number of distinct permutations for index counts (a,b,c): L!/(a!b!c!)."""
    return math.factorial(L) // (math.factorial(a) * math.factorial(b) * math.factorial(c))


def _build_sum_matrix_full_to_counts(L: int, device=None, dtype=None) -> torch.Tensor:
    """
    Build Ssum[L] with shape (Dsym, 3^L) such that:
      (Ssum vec(T))[abc] = sum_{i1..iL with counts(0,1,2)=(a,b,c)} T_{i1..iL}

    This corresponds to summing over all tensor components with the same count triple.
    """
    counts = _counts_list(L)
    idx_of: Dict[Tuple[int, int, int], int] = {t: i for i, t in enumerate(counts)}
    Dsym = len(counts)
    Dfull = 3**L
    S = torch.zeros(Dsym, Dfull, device=device, dtype=dtype)

    # Enumerate all base-3 index tuples by their flat id.
    for flat in range(Dfull):
        x = flat
        a = b = c = 0
        for _ in range(L):
            d = x % 3
            x //= 3
            if d == 0:
                a += 1
            elif d == 1:
                b += 1
            else:
                c += 1
        row = idx_of[(a, b, c)]
        S[row, flat] = 1.0
    return S


def _build_laplacian_matrix(L: int, device=None, dtype=None) -> torch.Tensor:
    """
    Laplacian operator Δ = ∂_x^2 + ∂_y^2 + ∂_z^2 on homogeneous polynomials:
      Sym^L -> Sym^{L-2}
    in the monomial coefficient basis indexed by (a,b,c), a+b+c=L.

    This is the standard way to characterize harmonic polynomials (kernel of Δ),
    which correspond to STF tensors of rank L.
    """
    assert L >= 2
    src = _counts_list(L)
    dst = _counts_list(L - 2)
    j_of: Dict[Tuple[int, int, int], int] = {t: i for i, t in enumerate(dst)}
    Dsrc = len(src)
    Ddst = len(dst)
    D = torch.zeros(Ddst, Dsrc, device=device, dtype=dtype)

    for col, (a, b, c) in enumerate(src):
        if a >= 2:
            row = j_of[(a - 2, b, c)]
            D[row, col] += float(a * (a - 1))
        if b >= 2:
            row = j_of[(a, b - 2, c)]
            D[row, col] += float(b * (b - 1))
        if c >= 2:
            row = j_of[(a, b, c - 2)]
            D[row, col] += float(c * (c - 1))
    return D


def _harmonic_basis_coeffs(L: int, device=None, dtype=None) -> torch.Tensor:
    """
    Return B_L with shape (Dsym, 2L+1), whose columns form a G-orthonormal basis
    of ker(Δ) in degree L, where G is the tensor inner-product gram matrix in the
    (a,b,c) symmetric-count basis:
      G = diag( 1 / w_{abc} ),  w_{abc} = L!/(a!b!c!)

    IMPORTANT:
      We represent symmetric tensors by count-sums:
        s_{abc} = sum_{i1..iL with counts=(a,b,c)} T_{i1..iL}
      For a symmetric tensor, the per-orbit component is t_{abc} = s_{abc}/w_{abc},
      and the tensor inner product becomes:
        <T,U> = sum_{abc} w_{abc} t_{abc} u_{abc} = sum_{abc} (1/w_{abc}) s^T_{abc} s^U_{abc}.
      Hence the correct Gram weight in the s_{abc} coordinates is 1/w_{abc}.
    """
    if L == 0:
        return torch.ones(1, 1, device=device, dtype=dtype)
    if L == 1:
        # Sym^1 is already harmonic (Δ maps to negative degree).
        # In count basis: (1,0,0),(0,1,0),(0,0,1)
        w = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
        S = torch.diag(torch.sqrt(w))
        # Take identity then normalize in G (already).
        return torch.eye(3, device=device, dtype=dtype)

    D = _build_laplacian_matrix(L, device=device, dtype=dtype)  # (Ddst, Dsrc)
    Dsrc = D.shape[1]
    null_dim = 2 * L + 1

    # Nullspace via SVD; we know the expected nullity for harmonic polynomials in 3D.
    # Sym^L dim = (L+2 choose 2), rank(Δ) = dim(Sym^{L-2}), so nullity = 2L+1.
    _, _, Vh = torch.linalg.svd(D, full_matrices=True)
    N = Vh[-null_dim:].T.contiguous()  # (Dsrc, null_dim)

    # Weighted orthonormalization under tensor inner product:
    counts = _counts_list(L)
    w = torch.tensor([float(_multinomial_count(L, a, b, c)) for (a, b, c) in counts], device=device, dtype=dtype)
    sqrt_g = 1.0 / torch.sqrt(w)  # sqrt(1/w)
    # Map to standard inner product: u -> sqrt(G) u  where G=diag(1/w)
    Nw = sqrt_g[:, None] * N
    Q, _ = torch.linalg.qr(Nw, mode="reduced")  # (Dsrc, null_dim), orthonormal in standard dot
    B = Q / sqrt_g[:, None]  # back to coefficient basis, columns orthonormal w.r.t G
    return B


@dataclass(frozen=True)
class STFProjectors:
    """
    Precomputed projectors up to Lmax for STF (l=L) component.

    For each L, stores P_L with shape (2L+1, 3^L) such that:
      y = P_L vec(T)
    gives STF coordinates of the symmetrized part of T (rank L).
    """

    Lmax: int
    P: Dict[int, torch.Tensor]


@lru_cache(maxsize=None)
def build_stf_projectors(Lmax: int) -> STFProjectors:
    """
    Build STF projectors on CPU/float64 for stability; they can be moved to device/dtype later.
    """
    P: Dict[int, torch.Tensor] = {}
    for L in range(Lmax + 1):
        if L == 0:
            P[0] = torch.ones(1, 1, dtype=torch.float64)
            continue
        if L == 1:
            # Identity from R^3 -> R^3
            P[1] = torch.eye(3, dtype=torch.float64)
            continue
        # Build in symmetric-count basis
        Ssum = _build_sum_matrix_full_to_counts(L, device=None, dtype=torch.float64)  # (Dsym, 3^L)
        B = _harmonic_basis_coeffs(L, device=None, dtype=torch.float64)               # (Dsym, 2L+1)
        # Coordinate map for G-orthonormal basis (B^T G c).
        counts = _counts_list(L)
        w = torch.tensor([float(_multinomial_count(L, a, b, c)) for (a, b, c) in counts], dtype=torch.float64)
        g = (1.0 / w).view(-1, 1)  # (Dsym,1)
        # STF coords = B^T G s = B^T (g ⊙ s)  where s = Ssum vec(T)
        P[L] = (B.T @ (g * Ssum)).contiguous()  # (2L+1, 3^L)
    return STFProjectors(Lmax=Lmax, P=P)


class FastSymmetricSTF(nn.Module):
    """
    Fast STF (l=L) projection for rank tensors up to Lmax.

    This provides project_stf(T, L) -> (..., 2L+1) for L>=0.
    For L>=2, it projects the STF component of the *symmetric* part of T.
    """

    def __init__(self, Lmax: int = 6):
        super().__init__()
        self.Lmax = int(Lmax)
        proj = build_stf_projectors(self.Lmax)
        # Register as buffers so they follow .to(device/dtype)
        for L, P in proj.P.items():
            self.register_buffer(f"P_stf_{L}", P)

    def _P(self, L: int) -> torch.Tensor:
        P = getattr(self, f"P_stf_{L}")
        return P

    def project_stf(self, T: torch.Tensor, L: int) -> torch.Tensor:
        """
        Args:
            T: (..., 3,3,...,3) with L copies of 3, or (..., 3^L) flat.
            L: rank
        Returns:
            (..., 2L+1) STF coordinates in a fixed orthonormal harmonic basis.
        """
        if L == 0:
            return T[..., :1] if T.shape[-1] != 1 else T
        if T.shape[-1] != 3**L:
            Tflat = T.reshape(*T.shape[:-L], 3**L)
        else:
            Tflat = T
        P = self._P(L).to(dtype=Tflat.dtype, device=Tflat.device)
        return Tflat @ P.T


def _build_r2k_lift(L_src: int, k: int, device=None, dtype=None) -> torch.Tensor:
    """
    Lift map induced by multiplication with r^{2k} = (x^2 + y^2 + z^2)^k:
      Sym^{L_src} -> Sym^{L_dst},  L_dst = L_src + 2k
    in the (a,b,c) monomial-count basis.
    """
    assert k >= 0
    if k == 0:
        src = _counts_list(L_src)
        return torch.eye(len(src), device=device, dtype=dtype)
    L_dst = L_src + 2 * k
    src = _counts_list(L_src)
    dst = _counts_list(L_dst)
    j_of: Dict[Tuple[int, int, int], int] = {t: i for i, t in enumerate(dst)}
    M = torch.zeros(len(dst), len(src), device=device, dtype=dtype)
    # (x^2+y^2+z^2)^k = sum_{u+v+w=k} k!/(u!v!w!) x^{2u} y^{2v} z^{2w}
    coeffs: List[Tuple[int, int, int, float]] = []
    for u in range(k + 1):
        for v in range(k - u + 1):
            w = k - u - v
            coef = math.factorial(k) / (math.factorial(u) * math.factorial(v) * math.factorial(w))
            coeffs.append((u, v, w, float(coef)))
    for col, (a, b, c) in enumerate(src):
        for u, v, w, coef in coeffs:
            row = j_of[(a + 2 * u, b + 2 * v, c + 2 * w)]
            M[row, col] += coef
    return M


@dataclass(frozen=True)
class TraceChainProjectors:
    """
    For each L, store projectors P_{L->l} : R^{3^L} -> R^{2l+1} for l=L,L-2,...
    corresponding to the symmetric trace chain decomposition.
    """
    Lmax: int
    P: Dict[Tuple[int, int], torch.Tensor]  # (L,l) -> (2l+1, 3^L)


@lru_cache(maxsize=None)
def build_trace_chain_projectors(Lmax: int) -> TraceChainProjectors:
    """
    Build projectors for the symmetric trace chain:
      Sym^L ~= ⊕_{k=0..floor(L/2)} r^{2k} Harm^{L-2k}
    where Harm^l is the harmonic (STF) subspace of degree l (dim 2l+1).

    Coordinates are with respect to a fixed G-orthonormal basis under
    G_L = diag(1/w_{abc}) in the count-sum coordinates.
    """
    P: Dict[Tuple[int, int], torch.Tensor] = {}
    for L in range(Lmax + 1):
        if L == 0:
            P[(0, 0)] = torch.ones(1, 1, dtype=torch.float64)
            continue
        if L == 1:
            P[(1, 1)] = torch.eye(3, dtype=torch.float64)
            continue

        # Build s = Ssum vec(T) in Sym^L count-sum basis
        Ssum = _build_sum_matrix_full_to_counts(L, device=None, dtype=torch.float64)  # (Dsym_L, 3^L)
        counts_L = _counts_list(L)
        wL = torch.tensor([float(_multinomial_count(L, a, b, c)) for (a, b, c) in counts_L], dtype=torch.float64)
        gL = (1.0 / wL)  # diagonal entries of G_L
        sqrt_gL = torch.sqrt(gL)  # (Dsym_L,)

        # Build basis blocks for each l=L-2k lifted by r^{2k}
        blocks: List[Tuple[int, torch.Tensor]] = []
        kmax = L // 2
        for k in range(kmax + 1):
            l = L - 2 * k
            B_l = _harmonic_basis_coeffs(l, device=None, dtype=torch.float64)  # (Dsym_l, 2l+1), G_l-orthonormal
            M = _build_r2k_lift(l, k, device=None, dtype=torch.float64)        # (Dsym_L, Dsym_l)
            V = (M @ B_l).contiguous()                                         # (Dsym_L, 2l+1)
            # Orthonormalize in G_L using weighted QR: apply sqrt(G_L), QR, then undo.
            Vw = sqrt_gL[:, None] * V
            Qw, _ = torch.linalg.qr(Vw, mode="reduced")
            Q = Qw / sqrt_gL[:, None]
            blocks.append((l, Q))

        # NOTE: The r^{2k}H^{L-2k} subspaces are theoretically orthogonal for the
        # standard L2(S^2) inner product; under this discrete tensor inner product
        # in count-sum coordinates, they are also well-conditioned for L<=6.
        # We therefore only orthonormalize within each block for speed/simplicity.

        # Construct projectors: y_l = Q_l^T G_L s = Q_l^T (gL ⊙ s)
        GS = (gL.view(-1, 1) * Ssum)  # (Dsym_L, 3^L)
        for l, Q in blocks:
            P[(L, l)] = (Q.T @ GS).contiguous()  # (2l+1, 3^L)
    return TraceChainProjectors(Lmax=Lmax, P=P)


class FastSymmetricTraceChain(nn.Module):
    """
    Fast symmetric trace-chain projectors up to Lmax.

    Provides project_chain(T, L) -> dict {l: (..., 2l+1)} for l=L,L-2,...
    where each output is the coefficient vector in a fixed G-orthonormal basis
    for the subspace r^{2k} Harm^l inside Sym^L.
    """

    def __init__(self, Lmax: int = 6):
        super().__init__()
        self.Lmax = int(Lmax)
        proj = build_trace_chain_projectors(self.Lmax)
        for (L, l), P in proj.P.items():
            self.register_buffer(f"P_chain_{L}_{l}", P)

    def _P(self, L: int, l: int) -> torch.Tensor:
        return getattr(self, f"P_chain_{L}_{l}")

    def project_chain(self, T: torch.Tensor, L: int) -> Dict[int, torch.Tensor]:
        """
        Args:
            T: (..., 3,3,...,3) with L copies of 3, or (..., 3^L) flat.
            L: rank
        Returns:
            dict mapping l in {L,L-2,...} to (..., 2l+1)
        """
        if L == 0:
            t0 = T[..., :1] if T.shape[-1] != 1 else T
            return {0: t0}
        if T.shape[-1] != 3**L:
            Tflat = T.reshape(*T.shape[:-L], 3**L)
        else:
            Tflat = T
        out: Dict[int, torch.Tensor] = {}
        for k in range(L // 2 + 1):
            l = L - 2 * k
            P = self._P(L, l).to(dtype=Tflat.dtype, device=Tflat.device)
            out[l] = Tflat @ P.T
        return out


def decompose_rank2_generic(T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generic rank-2 ICTD decomposition (matches the textbook formula):
      T_ij = STF_ij (l=2, 5d) + A_ij (l=1, pseudo 3d) + (1/3) δ_ij Tr(T) (l=0, 1d)

    Returns:
      l0: (..., 1) scalar  (normalized by 1/sqrt(3) is *not* applied here)
      l1: (..., 3) pseudovector v_k = (1/2) ε_{kij} (T_ij - T_ji)
      l2_stf: (..., 3, 3) STF tensor (symmetric traceless)
    """
    # Ensure matrix shape
    Tm = T.view(*T.shape[:-1], 3, 3) if T.shape[-1] == 9 else T
    # Trace / scalar
    tr = (Tm[..., 0, 0] + Tm[..., 1, 1] + Tm[..., 2, 2]).unsqueeze(-1)  # (...,1)
    # Symmetric traceless
    sym = 0.5 * (Tm + Tm.transpose(-2, -1))
    tr_part = (tr[..., 0] / 3.0).unsqueeze(-1).unsqueeze(-1) * torch.eye(3, device=Tm.device, dtype=Tm.dtype)
    stf = sym - tr_part
    # Antisymmetric -> pseudovector via epsilon contraction
    A = 0.5 * (Tm - Tm.transpose(-2, -1))
    # v_k = (1/2) ε_{kij} A_ij = (1/2) ε_{kij} * (1/2)(T_ij - T_ji) = (1/2) ε_{kij} T_ij (since ε antisym)
    eps = torch.zeros(3, 3, 3, device=Tm.device, dtype=Tm.dtype)
    eps[0, 1, 2] = 1
    eps[1, 2, 0] = 1
    eps[2, 0, 1] = 1
    eps[2, 1, 0] = -1
    eps[1, 0, 2] = -1
    eps[0, 2, 1] = -1
    v = torch.einsum("kij,...ij->...k", eps, A)
    return tr, v, stf

