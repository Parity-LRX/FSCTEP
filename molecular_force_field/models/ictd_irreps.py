"""
ICTD/trace-chain based irreps (2l+1) utilities and tensor products WITHOUT spherical harmonics.

Goal:
  - Provide an SO(3)-irreps representation built from harmonic polynomials
    (a.k.a. STF tensors / Laplacian kernel), derived purely from Cartesian algebra.
  - Provide Clebsch-Gordan-like coupling tensors computed in THIS basis using only:
      - polynomial multiplication (in monomial coefficient space)
      - trace-chain / harmonic projection (via Laplacian kernel)
    No e3nn spherical_harmonics and no e3nn wigner_3j are used here.

Important:
  - The basis for each l is fixed by our construction (harmonic nullspace + weighted orthonormalization).
    The CG tensors are computed consistently in the same basis, so equivariance is exact by construction.
  - Arbitrary lmax is supported. Small lmax (<=6) is the fastest due to Triton kernel coverage;
    higher lmax works correctly via automatic PyTorch fallback.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from molecular_force_field.models.ictd_fast import (
    _counts_list,
    _build_laplacian_matrix,
    _build_r2k_lift,
)

# ---------------------------------------------------------------------------
# torch.compile (Dynamo) integration helpers
# ---------------------------------------------------------------------------
try:
    import torch._dynamo as _dynamo  # type: ignore

    def _dynamo_disable(fn):  # pragma: no cover
        return _dynamo.disable(fn)
except Exception:  # pragma: no cover
    def _dynamo_disable(fn):
        return fn

# Optional: FlashTP-style fused outer-product + projection (Triton); set ICTD_USE_TRITON_TP=0 to disable
try:
    _triton_tp = __import__(
        "molecular_force_field.models.ictd_irreps_triton",
        fromlist=["_tp_fused_outer_proj", "_tp_fused_outer_proj_sparse", "_tp_fused_outer_proj_channel_mix"],
    )
    _tp_fused_outer_proj = getattr(_triton_tp, "_tp_fused_outer_proj", None)
    _tp_fused_outer_proj_sparse = getattr(_triton_tp, "_tp_fused_outer_proj_sparse", None)
    _tp_fused_outer_proj_channel_mix = getattr(_triton_tp, "_tp_fused_outer_proj_channel_mix", None)
except Exception:
    _tp_fused_outer_proj = None
    _tp_fused_outer_proj_sparse = None
    _tp_fused_outer_proj_channel_mix = None

# Sparse CG: use sparse projection when zero fraction >= this (0.4 = 40% zeros)
_SPARSE_MIN_ZERO_FRAC = 0.4
_SPARSE_ZERO_THRESHOLD = 1e-12
# Set ICTD_USE_SPARSE_TP=0 to disable sparse path (use dense Triton or PyTorch only)
_USE_SPARSE_TP = os.environ.get("ICTD_USE_SPARSE_TP", "1") == "1"


def sym_dim(L: int) -> int:
    """dim Sym^L(R^3) = (L+2 choose 2)."""
    return (L + 2) * (L + 1) // 2


def _double_factorial(n: int) -> int:
    if n <= 0:
        return 1
    out = 1
    for k in range(n, 0, -2):
        out *= k
    return out


def _gaussian_moment(n: int) -> float:
    # E[x^n] for x~N(0,1)
    if n % 2 == 1:
        return 0.0
    return float(_double_factorial(n - 1))


@lru_cache(maxsize=None)
def _gram_gaussian(L: int) -> torch.Tensor:
    """
    O(3)-invariant Gram matrix on Sym^L (monomial coefficient basis).

    For monomials x^a y^b z^c (with a+b+c=L),
      <m_{abc}, m_{a'b'c'}> = E[x^{a+a'}] E[y^{b+b'}] E[z^{c+c'}]
    under isotropic Gaussian measure, which is rotation-invariant.
    """
    counts = _counts_list(L)
    D = len(counts)
    G = torch.zeros(D, D, dtype=torch.float64)
    for i, (a, b, c) in enumerate(counts):
        for j, (a2, b2, c2) in enumerate(counts):
            G[i, j] = _gaussian_moment(a + a2) * _gaussian_moment(b + b2) * _gaussian_moment(c + c2)
    return G


@lru_cache(maxsize=None)
def _harmonic_basis_cpu_f64(L: int) -> torch.Tensor:
    """
    Harmonic basis in monomial (t_{abc}) coordinates, CPU float64.
    Computed once per L and cached; use _harmonic_basis_t(L, device, dtype) for device/dtype.
    Returns shape (Dsym(L), 2L+1).
    """
    if L == 0:
        return torch.ones(1, 1, dtype=torch.float64)
    if L == 1:
        return torch.eye(3, dtype=torch.float64)

    # Build harmonic subspace as nullspace of Laplacian on Sym^L -> Sym^{L-2}
    Delta = _build_laplacian_matrix(L, dtype=torch.float64)  # (D_{L-2}, D_L)
    _, s, vh = torch.linalg.svd(Delta, full_matrices=True)
    rank = int((s > 1e-12).sum().item())
    B = vh[rank:].T.contiguous()  # (D_L, 2L+1)

    G = _gram_gaussian(L)
    M = B.T @ G @ B
    evals, evecs = torch.linalg.eigh(M)
    evals = torch.clamp(evals, min=1e-14)
    W = evecs @ torch.diag(evals.rsqrt()) @ evecs.T
    return (B @ W).contiguous()


def _harmonic_basis_t(L: int, device=None, dtype=None) -> torch.Tensor:
    """
    Harmonic basis in monomial coefficient (t_{abc}) coordinates.

    Returns B_t with shape (Dsym(L), 2L+1). Base matrix is computed once per L (cached on CPU)
    and copied to the requested device/dtype.
    """
    B = _harmonic_basis_cpu_f64(L)
    return B.to(device=device, dtype=dtype)


@dataclass(frozen=True)
class HarmonicProjectors:
    """
    Projection matrices for the symmetric trace chain on Sym^L:
      Sym^L ~= ⊕_{k=0..floor(L/2)} r^{2k} Harm^{l},  l=L-2k

    We return projectors that map monomial coefficient vectors t_L (dim Sym^L)
    to harmonic coordinates c_l in the canonical basis B_t(l) (dim 2l+1).

      c_l = P_{L->l} t_L
    """

    Lmax: int
    P: Dict[Tuple[int, int], torch.Tensor]  # (L,l) -> (2l+1, Dsym(L))


@dataclass(frozen=True)
class HarmonicReconstructors:
    """
    Reconstruction matrices for the symmetric trace chain on Sym^L:
      t_L = sum_l V_{L<-l} c_l

    where ``c_l`` are harmonic coordinates in the canonical ICTD basis and
    ``t_L`` are monomial coefficients in Sym^L.
    """

    Lmax: int
    V: Dict[Tuple[int, int], torch.Tensor]  # (L,l) -> (Dsym(L), 2l+1)


@lru_cache(maxsize=None)
def build_harmonic_projectors(Lmax: int) -> HarmonicProjectors:
    """
    Build all P_{L->l} on CPU/float64 for stability; move to device/dtype at runtime.
    """
    P: Dict[Tuple[int, int], torch.Tensor] = {}
    for L in range(Lmax + 1):
        D_L = sym_dim(L)
        GL = _gram_gaussian(L)               # (D_L,D_L)

        for k in range(L // 2 + 1):
            l = L - 2 * k
            # Harmonic basis at degree l in t-coords
            B_l = _harmonic_basis_t(l, dtype=torch.float64)  # (D_l, 2l+1)
            # Lift to degree L via r^{2k}
            M = _build_r2k_lift(l, k, dtype=torch.float64)   # (D_L, D_l)
            V = (M @ B_l).contiguous()                       # (D_L, 2l+1)

            # Weighted least squares projection onto span(V) under <.,.>_L with diag(wL):
            # c = (V^T W V)^{-1} V^T W t
            G = V.T @ GL @ V  # (2l+1,2l+1)
            # Stabilize: symmetric positive definite for small L; use solve.
            Pinv = torch.linalg.solve(G, V.T @ GL)  # (2l+1, D_L)
            P[(L, l)] = Pinv.contiguous()

    return HarmonicProjectors(Lmax=Lmax, P=P)


@lru_cache(maxsize=None)
def build_harmonic_reconstructors(Lmax: int) -> HarmonicReconstructors:
    """
    Build all V_{L<-l} on CPU/float64 for stability; move to device/dtype at runtime.
    These matrices reconstruct monomial coefficients from ICTD harmonic coordinates.
    """
    V: Dict[Tuple[int, int], torch.Tensor] = {}
    for L in range(Lmax + 1):
        for k in range(L // 2 + 1):
            l = L - 2 * k
            B_l = _harmonic_basis_t(l, dtype=torch.float64)  # (D_l, 2l+1)
            M = _build_r2k_lift(l, k, dtype=torch.float64)   # (D_L, D_l)
            V[(L, l)] = (M @ B_l).contiguous()               # (D_L, 2l+1)
    return HarmonicReconstructors(Lmax=Lmax, V=V)


def direction_harmonics(n: torch.Tensor, l: int) -> torch.Tensor:
    """
    Compute harmonic (irrep) coordinates of the symmetric tensor n^{⊗l} in our basis.

    n: (..., 3) unit vector (or any vector; scaling changes non-homogeneously for trace-chain, so use unit).
    Returns: (..., 2l+1)

    Derivation:
      Polynomial p(x,y,z) = (n_x x + n_y y + n_z z)^l
      has monomial coefficients t_{abc} = multinomial(l;a,b,c) n_x^a n_y^b n_z^c.
      Project to harmonic coordinates via:
        c = B_l^T W_l t    (since B_l orthonormal under W_l)
    """
    if l == 0:
        return torch.ones(*n.shape[:-1], 1, device=n.device, dtype=n.dtype)
    counts = _counts_list(l)
    # t_{abc}
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    t_list = []
    for (a, b, c) in counts:
        coef = math.factorial(l) / (math.factorial(a) * math.factorial(b) * math.factorial(c))
        t_list.append((nx**a) * (ny**b) * (nz**c) * float(coef))
    t = torch.stack(t_list, dim=-1)  # (..., Dsym(l))
    B = _harmonic_basis_t(l, device=n.device, dtype=n.dtype)  # (Dsym, 2l+1)
    # coords under Gram: c = B^T G t
    G = _gram_gaussian(l).to(device=n.device, dtype=n.dtype)  # (Dsym, Dsym)
    c = torch.einsum("...d,md,mc->...c", t, G, B)
    return c


@lru_cache(maxsize=None)
def _dir_monomial_exps_coefs(l: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute monomial exponents and multinomial coefficients for degree l.

    Returns:
      exps: (Dsym(l), 3) int64, rows are (a,b,c)
      coefs: (Dsym(l),) float64, multinomial(l; a,b,c)
    """
    counts = _counts_list(l)
    exps = torch.tensor(counts, dtype=torch.int64)  # (D,3)
    if l == 0:
        coefs = torch.ones(1, dtype=torch.float64)
    else:
        coefs_list = []
        for (a, b, c) in counts:
            coefs_list.append(math.factorial(l) / (math.factorial(a) * math.factorial(b) * math.factorial(c)))
        coefs = torch.tensor(coefs_list, dtype=torch.float64)
    return exps, coefs


@lru_cache(maxsize=None)
def _dir_proj_cpu_f64(l: int) -> torch.Tensor:
    """
    Precompute P_l = G_l @ B_l on CPU float64:
      t (..., Dsym) -> c (..., 2l+1) via c = t @ P_l
    """
    if l == 0:
        return torch.ones(1, 1, dtype=torch.float64)
    B = _harmonic_basis_t(l, dtype=torch.float64)  # (Dsym, 2l+1)
    G = _gram_gaussian(l)  # (Dsym, Dsym) float64
    return (G @ B).contiguous()  # (Dsym, 2l+1)


_dir_proj_cache_by_dev_dtype: Dict[Tuple[str, str, int], torch.Tensor] = {}

# Optional CUDA path (Triton fused kernel). PyTorch (N,D)@(D,K) is often faster due to cuBLAS;
# set ICTD_USE_TRITON=1 to try Triton anyway (e.g. to reduce peak memory by not materializing t).
def _direction_harmonics_triton_optional(
    n: torch.Tensor, l: int, exps: torch.Tensor, coefs: torch.Tensor, P: torch.Tensor
) -> torch.Tensor | None:
    import os
    if os.environ.get("ICTD_USE_TRITON", "0") != "1":
        return None
    try:
        from molecular_force_field.models.ictd_irreps_triton import direction_harmonics_triton
        return direction_harmonics_triton(n, l, exps, coefs, P)
    except Exception:
        return None


def direction_harmonics_fast(n: torch.Tensor, l: int) -> torch.Tensor:
    """
    Faster version of direction_harmonics with:
      - vectorized monomial evaluation
      - cached projection matrix (G@B) per (device,dtype,l)
    """
    if l == 0:
        return torch.ones(*n.shape[:-1], 1, device=n.device, dtype=n.dtype)

    key = (str(n.device), str(n.dtype), int(l))
    P = _dir_proj_cache_by_dev_dtype.get(key)
    if P is None:
        P = _dir_proj_cpu_f64(l).to(device=n.device, dtype=n.dtype)
        _dir_proj_cache_by_dev_dtype[key] = P

    exps, coefs = _dir_monomial_exps_coefs(l)
    exps = exps.to(device=n.device)
    coefs = coefs.to(device=n.device, dtype=n.dtype)

    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    a = exps[:, 0]
    b = exps[:, 1]
    c = exps[:, 2]
    # (..., Dsym) — GPU: one fused broadcast pow kernel; very fast
    t = (nx.unsqueeze(-1) ** a) * (ny.unsqueeze(-1) ** b) * (nz.unsqueeze(-1) ** c)
    t = t * coefs
    # (..., 2l+1)
    return t @ P


def parse_irreps_string(irreps: str) -> List[Tuple[int, int]]:
    """
    Parse e3nn-style irreps string into (mul, l) list.
    Examples: "0e + 1o + 2e" -> [(1,0), (1,1), (1,2)]; "5x0e + 2x2e" -> [(5,0), (2,2)].
    """
    out: List[Tuple[int, int]] = []
    for part in irreps.replace(",", " ").split("+"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d*)x?(\d+)(e|o)$", part.strip(), re.IGNORECASE)
        if not m:
            raise ValueError(f"Invalid irreps token: {part!r}")
        mul_s, l_s, _ = m.groups()
        mul = int(mul_s) if mul_s else 1
        l_val = int(l_s)
        out.append((mul, l_val))
    return out


def parse_irreps_to_l3_list(irreps: str, allowed_l3: Optional[List[int]] = None) -> List[int]:
    """
    Parse e3nn-style irreps string to an ordered list of l values (e.g. for output filtering).
    Examples: "0e + 2e" -> [0, 2]; "2e + 0e" -> [2, 0].
    If allowed_l3 is provided, only l that are in allowed_l3 are included (e.g. l⊗l gives only even l3).
    """
    parts = parse_irreps_string(irreps)
    out: List[int] = []
    seen: Set[int] = set()
    for _, l_val in parts:
        if allowed_l3 is not None and l_val not in allowed_l3:
            continue
        if l_val not in seen:
            seen.add(l_val)
            out.append(l_val)
    return out


def direction_harmonics_irreps(n: torch.Tensor, irreps: str) -> torch.Tensor:
    """
    Like e3nn spherical_harmonics(irreps_out, x): compute direction harmonics in ICTD basis
    for the given irreps and return a single tensor (..., dim).

    irreps: e3nn-style string, e.g. "0e + 1o + 2e" or "5x0e + 3x1o + 10x2e".
    Returns: (..., sum over (mul * (2l+1))) in order of irreps.
    """
    parts = parse_irreps_string(irreps)
    chunks: List[torch.Tensor] = []
    for mul, l_val in parts:
        h = direction_harmonics_fast(n, l_val)  # (..., 2l+1)
        chunks.append(h.unsqueeze(-2).expand(*h.shape[:-1], mul, 2 * l_val + 1).reshape(*h.shape[:-1], mul * (2 * l_val + 1)))
    return torch.cat(chunks, dim=-1)


def direction_harmonics_all(n: torch.Tensor, lmax: int) -> List[torch.Tensor]:
    """
    Compute direction harmonics for all l=0..lmax.
    Returns a list Y where Y[l] has shape (..., 2l+1).
    """
    return [direction_harmonics_fast(n, l) for l in range(int(lmax) + 1)]


def ictd_l2_to_rank2(c: torch.Tensor) -> torch.Tensor:
    """
    Convert ICTD l=2 (5D) harmonic coordinates to 3x3 symmetric traceless tensor.

    The ICTD basis is built from monomials (x^a y^b z^c) with a+b+c=2.
    Monomial order: z^2, yz, y^2, xz, xy, x^2 (from _counts_list).
    Output T satisfies T(R·n) = R @ T(n) @ R.T under rotation R.

    Args:
        c: (..., 5) ICTD l=2 coordinates
    Returns:
        T: (..., 3, 3) symmetric traceless matrix
    """
    B = _harmonic_basis_t(2, device=c.device, dtype=c.dtype)  # (6, 5)
    t = torch.einsum("dm,...m->...d", B, c)  # (..., 6) monomial coeffs
    # t order: [zz, yz, yy, xz, xy, xx]. Multinomial: xy,xz,yz have factor 2.
    # T_ij from polynomial: T[0,1]=coef_xy/2, etc.
    T = torch.zeros(*c.shape[:-1], 3, 3, device=c.device, dtype=c.dtype)
    T[..., 0, 0] = t[..., 5]
    T[..., 0, 1] = T[..., 1, 0] = t[..., 4] * 0.5
    T[..., 0, 2] = T[..., 2, 0] = t[..., 3] * 0.5
    T[..., 1, 1] = t[..., 2]
    T[..., 1, 2] = T[..., 2, 1] = t[..., 1] * 0.5
    T[..., 2, 2] = t[..., 0]
    return T


@dataclass(frozen=True)
class CGKey:
    l1: int
    l2: int
    l3: int


@lru_cache(maxsize=None)
def _build_poly_mult_matrix(l1: int, l2: int, L: int) -> torch.Tensor:
    """
    Precompute M_poly: (DL, D1*D2) sparse-ish matrix for polynomial multiplication.
    tL = M_poly @ (t1.outer(t2).flatten()) maps monomial product to Sym^L.
    """
    counts1 = _counts_list(l1)
    counts2 = _counts_list(l2)
    countsL = _counts_list(L)
    idxL = {t: i for i, t in enumerate(countsL)}
    D1, D2, DL = len(counts1), len(counts2), len(countsL)
    M = torch.zeros(DL, D1 * D2, dtype=torch.float64)
    for i, c1 in enumerate(counts1):
        for j, c2 in enumerate(counts2):
            k = idxL[(c1[0] + c2[0], c1[1] + c2[1], c1[2] + c2[2])]
            M[k, i * D2 + j] = 1.0
    return M.contiguous()


@lru_cache(maxsize=None)
def build_cg_tensor(l1: int, l2: int, l3: int) -> torch.Tensor:
    """
    Build the coupling tensor C_{m1,m2,m3} in OUR harmonic basis.

    Semantics:
      Given harmonic coefficient vectors a in R^{2l1+1}, b in R^{2l2+1},
      define polynomial product at degree L=l1+l2, then project to the trace-chain block l3.
      The result is a harmonic coefficient vector c in R^{2l3+1}:
        c[m3] = sum_{m1,m2} a[m1] b[m2] C[m1,m2,m3]

    This is an SO(3)-equivariant intertwiner by construction.
    Uses vectorized matrix ops instead of Python loops for speed.
    """
    L = l1 + l2
    if not (abs(l1 - l2) <= l3 <= l1 + l2) or ((l1 + l2 + l3) % 2 == 1):
        return torch.zeros(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1, dtype=torch.float64)

    proj = build_harmonic_projectors(Lmax=L)
    P_L_l3 = proj.P[(L, l3)]  # (2l3+1, DL)
    B1 = _harmonic_basis_t(l1, dtype=torch.float64)  # (D1, 2l1+1)
    B2 = _harmonic_basis_t(l2, dtype=torch.float64)  # (D2, 2l2+1)
    M_poly = _build_poly_mult_matrix(l1, l2, L)  # (DL, D1*D2)

    m1_dim, m2_dim = 2 * l1 + 1, 2 * l2 + 1

    outer = torch.einsum("im,jn->ijmn", B1, B2)  # (D1, D2, m1_dim, m2_dim)
    outer_flat = outer.reshape(B1.shape[0] * B2.shape[0], m1_dim * m2_dim)
    tL = M_poly @ outer_flat  # (DL, m1*m2)
    c3 = P_L_l3 @ tL  # (2l3+1, m1*m2)
    C = c3.T.reshape(m1_dim, m2_dim, 2 * l3 + 1)
    return C.contiguous()


def cg_tensor_sparsity(C: torch.Tensor, threshold: float = 1e-10) -> Tuple[int, int, float]:
    """
    Return (numel, num_nonzero, zero_fraction) for an ICTD CG tensor.
    Many (l1,l2,l3) triples yield 60--85%% zeros (exact or |x|<=threshold); useful for sparse kernels.
    """
    n = C.numel()
    nz = (C.abs() > threshold).sum().item()
    return n, nz, 1.0 - (nz / n)


class HarmonicElementwiseProduct(nn.Module):
    """
    Element-wise tensor product in ICTD basis, analogous to e3nn ElementwiseTensorProduct.

    Pairs same-l blocks: for each l, x1[l] and x2[l] have shape (..., mul, 2l+1);
    computes l⊗l -> l3 with CG and (optionally) filters to output irreps.

    - irreps_out="0e": only scalar invariants per (l, channel): (x1*x2).sum(m)/sqrt(2l+1).
      Output shape (..., mul * (lmax+1)).
    - irreps_out="0e + 2e", "2e + 0e", etc.: output only the requested l3 (order preserved).
      l⊗l yields only even l3 (0e, 2e, 4e, ...); odd l in the string are ignored.
      Returns a single tensor (..., sum over mul_l3*(2l3+1)) in irreps order.
    - irreps_out=None or "full": output all l3 from l⊗l for l=0..lmax (only even l3 by parity).
      Output dict l3 -> (..., mul_l3, 2l3+1) where mul_l3 = mul * (number of l that contribute to l3).

    normalization (str):
      "component" (default, same as e3nn): CG tensors are scaled so that each output m3-component
          has unit variance when inputs have i.i.d. unit-variance components.
          Factor per (l, l3) path: alpha = sqrt(2*l3+1) / ||C_raw||_F.
      "norm": CG tensors are scaled so that the output L2-norm has unit expected squared norm.
          Factor: alpha = 1 / ||C_raw||_F.
      "none": use raw CG tensors from build_cg_tensor (no rescaling).
    """


    def __init__(
        self,
        lmax: int,
        mul: int,
        irreps_out: str | None = "0e",
        normalization: str = "component",
        internal_compute_dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.lmax = int(lmax)
        self.mul = int(mul)
        self.internal_compute_dtype = internal_compute_dtype
        self._normalization = normalization
        self._irreps_out = irreps_out.strip().lower() if (irreps_out and isinstance(irreps_out, str)) else "full"
        self._output_0e_only = self._irreps_out == "0e"

        # Precompute which (l, l3) paths exist: l⊗l -> l3, (2l+l3) even
        self._paths: List[Tuple[int, int]] = []
        for l in range(self.lmax + 1):
            for l3 in range(0, 2 * l + 1):
                if (2 * l + l3) % 2 == 0:
                    self._paths.append((l, l3))
        allowed_l3 = sorted(set(l3 for (_, l3) in self._paths))
        if self._irreps_out not in ("0e", "full"):
            self._filter_l3: Optional[List[int]] = parse_irreps_to_l3_list(self._irreps_out, allowed_l3)
        else:
            self._filter_l3 = None

        # Build CG tensors eagerly and apply normalization.
        #   component: each output m3-component has unit variance when inputs have
        #              i.i.d. unit-variance components → alpha = sqrt(2l3+1) / ||C||_F
        #   norm:      output L2-norm has unit expected squared norm
        #              → alpha = 1 / ||C||_F
        #   none:      use raw CG tensors from build_cg_tensor
        self._cg_cache: List[torch.Tensor] = []
        for (l, l3) in self._paths:
            C = build_cg_tensor(l, l, l3)
            C_fn = C.norm().item()
            if normalization == "component" and C_fn > 1e-30:
                C = C * (math.sqrt(2 * l3 + 1) / C_fn)
            elif normalization == "norm" and C_fn > 1e-30:
                C = C * (1.0 / C_fn)
            self._cg_cache.append(C)

        # For the 0e fast path: precompute per-l scalar factor from the (diagonal)
        # normalized CG of l⊗l→0 so that out = factor * (a · b).
        self._0e_factors: List[float] = []
        for l in range(self.lmax + 1):
            path_idx = next(i for i, (ll, l3) in enumerate(self._paths) if ll == l and l3 == 0)
            self._0e_factors.append(self._cg_cache[path_idx][0, 0, 0].item())

        self._cg_cache_device_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}

    def _get_cg_list(self, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        key = (str(device), str(dtype))
        if key in self._cg_cache_device_dtype:
            return self._cg_cache_device_dtype[key]
        compute_dtype = self.internal_compute_dtype
        cg_list = [C.to(device=device, dtype=compute_dtype) for C in self._cg_cache]
        self._cg_cache_device_dtype[key] = cg_list
        return cg_list

    def forward(
        self,
        x1: Dict[int, torch.Tensor],
        x2: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor] | torch.Tensor:
        """
        x1, x2: dict l -> (..., mul, 2l+1). Same keys (l=0..lmax).
        If irreps_out=="0e": returns (..., mul*(lmax+1)).
        If irreps_out is a filter string (e.g. "0e + 2e"): returns (..., sum of mul_l3*(2l3+1)).
        Else (full): returns dict l3 -> (..., mul_l3, 2l3+1).
        """
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype

        if self._output_0e_only:
            out_list = []
            for l in range(self.lmax + 1):
                a = x1[l]
                b = x2[l]
                out_list.append((a * b).sum(dim=-1) * self._0e_factors[l])
            return torch.cat(out_list, dim=-1)

        compute_dtype = self.internal_compute_dtype
        cg_list = self._get_cg_list(device, dtype)
        out: Dict[int, List[torch.Tensor]] = {}
        for idx, (l, l3) in enumerate(self._paths):
            C = cg_list[idx]
            a = x1[l].to(dtype=compute_dtype)
            b = x2[l].to(dtype=compute_dtype)
            a_flat = a.reshape(-1, self.mul, 2 * l + 1)
            b_flat = b.reshape(-1, self.mul, 2 * l + 1)
            out_l3 = torch.einsum("bcm,bcn,mno->bco", a_flat, b_flat, C)
            out_l3 = out_l3.reshape(*batch_shape, self.mul, 2 * l3 + 1).to(dtype=dtype)
            out.setdefault(l3, []).append(out_l3)
        result: Dict[int, torch.Tensor] = {}
        for l3 in out:
            result[l3] = torch.cat(out[l3], dim=-2)
        if self._filter_l3 is not None:
            return torch.cat(
                [result[l3].reshape(*batch_shape, -1) for l3 in self._filter_l3 if l3 in result],
                dim=-1,
            )
        return result


class HarmonicFullyConnectedTensorProduct(nn.Module):
    """
    Fully-connected tensor product in harmonic/ICTD basis (SO(3) irreps, no spherical harmonics).

    Representation:
      input features are a dict l -> (..., mul_l, 2l+1) (mul_l is multiplicity/channels for that l).
      output is similarly l -> (..., mul_out_l, 2l+1).

    We follow the same "W[mul_out, mul1, mul2]" weight structure per (l1,l2->l3) path.
    """

    def __init__(
        self,
        mul_in1: int,
        mul_in2: int,
        mul_out: int,
        lmax: int,
        internal_weights: bool = True,
        *,
        # e3nn-instructions-like control: explicitly choose which (l1,l2,l3) paths exist.
        # If provided, this is the most precise "pruning" mechanism.
        allowed_paths: List[Tuple[int, int, int]] | None = None,
        # Convenience policy to generate allowed_paths.
        # - "full": keep all CG-allowed paths
        # - "max_rank_other": keep paths with min(l1,l2) <= max_rank_other (like sparse heuristic)
        path_policy: str = "full",
        max_rank_other: int | None = None,
        # CG normalization (same convention as e3nn TP):
        #   "component" (default): alpha = sqrt(2*l3+1) / ||C||_F per path
        #   "norm": alpha = 1 / ||C||_F per path
        #   "none": raw CG tensors
        normalization: str = "component",
        # Internal computation dtype for CG tensors and projections (default: float64 for stability)
        internal_compute_dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.mul_in1 = mul_in1
        self.mul_in2 = mul_in2
        self.mul_out = mul_out
        self.lmax = lmax
        self.internal_weights = internal_weights
        self._normalization = normalization
        self.internal_compute_dtype = internal_compute_dtype

        # Enumerate all valid (l1,l2,l3) with parity selection (even step)
        all_paths: List[Tuple[int, int, int]] = []
        for l1 in range(lmax + 1):
            for l2 in range(lmax + 1):
                for l3 in range(abs(l1 - l2), min(l1 + l2, lmax) + 1):
                    if (l1 + l2 + l3) % 2 == 1:
                        continue
                    all_paths.append((l1, l2, l3))

        if allowed_paths is not None:
            allowed_set = set(allowed_paths)
            self.paths = [p for p in all_paths if p in allowed_set]
        else:
            if path_policy == "full":
                self.paths = all_paths
            elif path_policy == "max_rank_other":
                if max_rank_other is None:
                    raise ValueError("path_policy='max_rank_other' requires max_rank_other")
                self.paths = [p for p in all_paths if min(p[0], p[1]) <= int(max_rank_other)]
            else:
                raise ValueError(f"Unknown path_policy={path_policy!r}")

        self.num_paths = len(self.paths)
        self.weight_numel = self.num_paths * mul_out * mul_in1 * mul_in2

        if internal_weights:
            # (P, mul_out, mul1, mul2)
            self.weight = nn.Parameter(torch.randn(self.num_paths, mul_out, mul_in1, mul_in2) * 0.02)
        else:
            self.register_parameter("weight", None)

        # Cache CG tensors to avoid per-forward .to(device,dtype) allocations.
        #
        # build_cg_tensor(l1,l2,l3) returns a CPU float64 tensor (and is itself lru_cached),
        # but calling .to(device,dtype) for every path on every forward is costly (especially
        # for higher lmax with many paths). We keep:
        # - a CPU float64 list (built lazily) for the paths
        # - per-(device,dtype) converted lists for fast reuse
        self._cg_cpu_f64: List[torch.Tensor] | None = None
        self._cg_cache_by_dev_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}

        # Group paths by (l1,l2). This enables an e3nn-like factorization:
        #   1) build the (l1,l2) tensor-product basis once (NO mul_out)
        #   2) apply per-path (mul_out,mul1,mul2) weights as a separate contraction
        # This avoids repeating the expensive m-contractions for every output channel and path.
        #
        # Each group stores:
        #   - l1, l2
        #   - p_indices: indices into self.paths (and self.weight / gates vector)
        #   - l3_list: l3 per path in group (aligned with p_indices)
        #   - segments: list of (p_idx, l3, start, end) into concatenated K_total
        self._groups: List[Dict[str, object]] = []
        groups_tmp: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for p_idx, (l1, l2, l3) in enumerate(self.paths):
            groups_tmp.setdefault((l1, l2), []).append((p_idx, l3))
        for (l1, l2), items in sorted(groups_tmp.items()):
            p_indices = [p for (p, _) in items]
            l3_list = [l3 for (_, l3) in items]
            segments = []
            start = 0
            for p_idx, l3 in items:
                kdim = 2 * l3 + 1
                segments.append((p_idx, l3, start, start + kdim))
                start += kdim
            self._groups.append(
                {
                    "l1": l1,
                    "l2": l2,
                    "p_indices": p_indices,
                    "l3_list": l3_list,
                    "segments": segments,
                    "k_total": start,
                }
            )

        # Cache per-group projection matrices per (device,dtype) matching _groups:
        #   U_g: (m1*m2, K_total), where K_total = sum_{paths in group} (2*l3+1)
        self._proj_group_cache_by_dev_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}
        # Sparse U per group when zero_frac >= _SPARSE_MIN_ZERO_FRAC: list of None or (d_idx, k_idx, vals)
        self._proj_sparse_cache_by_dev_dtype: Dict[Tuple[str, str], List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]] = {}

    @_dynamo_disable
    def _get_cg_list(self, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        # Use internal_compute_dtype for CG tensors (for numerical stability)
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._cg_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached

        if self._cg_cpu_f64 is None:
            self._cg_cpu_f64 = []
            for (l1, l2, l3) in self.paths:
                C = build_cg_tensor(l1, l2, l3)
                C_fn = C.norm().item()
                if self._normalization == "component" and C_fn > 1e-30:
                    C = C * (math.sqrt(2 * l3 + 1) / C_fn)
                elif self._normalization == "norm" and C_fn > 1e-30:
                    C = C * (1.0 / C_fn)
                self._cg_cpu_f64.append(C)

        cg_list = [C.to(device=device, dtype=compute_dtype) for C in self._cg_cpu_f64]
        self._cg_cache_by_dev_dtype[key] = cg_list
        return cg_list

    @_dynamo_disable
    def _get_proj_group_list(self, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        """
        Returns a list of projection matrices U_g, one per (l1,l2) group:
          U_g: (m1*m2, K_total)
        such that for tensor-product coefficients t_{m1,m2} (flattened to m1*m2),
        the concatenated outputs for all paths in the group are:
          y_concat = t_flat @ U_g
        """
        # Use internal_compute_dtype for projection matrices (for numerical stability)
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._proj_group_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached

        cg_list = self._get_cg_list(device=device, dtype=dtype)
        proj_list: List[torch.Tensor] = []
        sparse_list: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
        for g in self._groups:
            l1 = int(g["l1"])  # type: ignore[arg-type]
            l2 = int(g["l2"])  # type: ignore[arg-type]
            segments = g["segments"]  # type: ignore[assignment]
            k_total = int(g["k_total"])  # type: ignore[arg-type]

            m1 = 2 * l1 + 1
            m2 = 2 * l2 + 1
            U = torch.zeros(m1 * m2, k_total, device=device, dtype=compute_dtype)
            for p_idx, _l3, s, e in segments:  # type: ignore[misc]
                C = cg_list[int(p_idx)]  # (m1,m2,kdim)
                U[:, int(s): int(e)] = C.reshape(m1 * m2, int(e) - int(s))
            proj_list.append(U)

            # Build sparse (d_idx, k_idx, vals) only on CUDA (Triton is CUDA-only; on CPU it adds overhead)
            if device.type == "cuda":
                n = U.numel()
                nz = (U.abs() > _SPARSE_ZERO_THRESHOLD).sum().item()
                zero_frac = 1.0 - (nz / n) if n else 0.0
                if zero_frac >= _SPARSE_MIN_ZERO_FRAC:
                    mask = U.abs() > _SPARSE_ZERO_THRESHOLD
                    nz_flat = mask.nonzero(as_tuple=False)  # (nnz, 2)
                    d_idx = nz_flat[:, 0].contiguous()
                    k_idx = nz_flat[:, 1].contiguous()
                    vals = U[mask].contiguous()
                    sparse_list.append((d_idx, k_idx, vals))
                else:
                    sparse_list.append(None)
            else:
                sparse_list.append(None)

        self._proj_group_cache_by_dev_dtype[key] = proj_list
        self._proj_sparse_cache_by_dev_dtype[key] = sparse_list
        return proj_list

    @_dynamo_disable
    def _get_proj_sparse_list(self, device: torch.device, dtype: torch.dtype) -> List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Return sparse (d_idx, k_idx, vals) per group; None where dense is used. Call _get_proj_group_list first."""
        self._get_proj_group_list(device=device, dtype=dtype)
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        return self._proj_sparse_cache_by_dev_dtype[key]

    @_dynamo_disable
    def prewarm_caches(self, device: torch.device, dtype: torch.dtype) -> None:
        """Pre-build internal caches on (device, dtype).

        This keeps one-time Python-side work (and `.item()` calls) out of torch.compile tracing.
        Safe to call multiple times.
        """
        _ = self._get_cg_list(device=device, dtype=dtype)
        _ = self._get_proj_group_list(device=device, dtype=dtype)
        _ = self._get_proj_sparse_list(device=device, dtype=dtype)

    def forward(
        self,
        x1: Dict[int, torch.Tensor],
        x2: Dict[int, torch.Tensor],
        weights: torch.Tensor | None = None,
    ) -> Dict[int, torch.Tensor]:
        # Determine batch shape from any present block
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        compute_dtype = self.internal_compute_dtype

        if self.internal_weights:
            # Assume module has already been moved to the right device/dtype by caller.
            w_param = self.weight  # (P, o, i, j)
        else:
            assert weights is not None
            w = weights
            if w.shape[-1] not in (self.weight_numel, self.num_paths):
                raise ValueError(f"weights last-dim must be weight_numel={self.weight_numel} or num_paths={self.num_paths}, got {w.shape[-1]}")

        # Make weights/gates device+dtype consistent once (avoid per-path .to()).
        # Only convert when needed; calling .to() unconditionally can add overhead.
        if weights is not None and (weights.device != device or weights.dtype != dtype):
            weights = weights.to(device=device, dtype=dtype)

        # init output
        out: Dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            out[l] = torch.zeros(*batch_shape, self.mul_out, 2 * l + 1, device=device, dtype=dtype)

        # Fast path: internal_weights + per-path scalar gates (this is what our models use).
        if self.internal_weights and (weights is None or weights.shape[-1] == self.num_paths):
            proj_list = self._get_proj_group_list(device=device, dtype=dtype)
            # Only fetch sparse list on CUDA (Triton is CUDA-only; avoids extra work on CPU)
            sparse_list = self._get_proj_sparse_list(device=device, dtype=dtype) if device.type == "cuda" else None
            for g_idx, g in enumerate(self._groups):
                l1 = int(g["l1"])  # type: ignore[arg-type]
                l2 = int(g["l2"])  # type: ignore[arg-type]
                segments = g["segments"]  # type: ignore[assignment]
                k_total = int(g["k_total"])  # type: ignore[arg-type]

                a = x1.get(l1)
                b = x2.get(l2)
                if a is None or b is None:
                    continue

                # e3nn-like factorization:
                # 1) project to concatenated k space once: (..., i, j, K_total)
                # 2) batch channel mixing for all paths in group, then segment and accumulate
                m1 = 2 * l1 + 1
                m2 = 2 * l2 + 1
                U = proj_list[g_idx]  # (m1*m2, K_total) in compute_dtype
                # Convert inputs to compute_dtype for numerical stability
                a_comp = a.to(dtype=compute_dtype) if a.dtype != compute_dtype else a
                b_comp = b.to(dtype=compute_dtype) if b.dtype != compute_dtype else b
                # FlashTP-style: fused projection+channel-mix (one kernel), or sparse/dense TP then per-path mix
                y = None
                used_fused_mix = False
                if a_comp.is_cuda and a_comp.dim() >= 2:
                    B_flat = 1
                    for s in batch_shape:
                        B_flat *= int(s)
                    a_flat = a_comp.reshape(B_flat, self.mul_in1, m1)
                    b_flat = b_comp.reshape(B_flat, self.mul_in2, m2)
                    num_paths_in_group = len(segments)
                    # Try fused outer-product + projection + channel mixing (one kernel, no y write-back)
                    if (
                        _tp_fused_outer_proj_channel_mix is not None
                        and num_paths_in_group <= 16
                    ):
                        W_stack = torch.stack(
                            [w_param[int(p_idx)].to(dtype=compute_dtype) for p_idx, _, _, _ in segments],
                            dim=0,
                        )
                        W_stack = W_stack.to(device=a_comp.device).contiguous().view(
                            num_paths_in_group, self.mul_out, self.mul_in1 * self.mul_in2
                        )
                        out_buf = _tp_fused_outer_proj_channel_mix(
                            a_flat, b_flat, U, W_stack, segments, k_total, self.mul_out, m1, m2
                        )
                        if out_buf is not None:
                            out_buf = out_buf.to(dtype=dtype) if out_buf.dtype != dtype else out_buf
                            for seg_idx, (p_idx, l3, s, e) in enumerate(segments):  # type: ignore[misc]
                                seg_out = out_buf[:, seg_idx, :, int(s) : int(e)]
                                if weights is not None:
                                    seg_out = seg_out * weights[..., int(p_idx), None, None]
                                out[int(l3)] = out[int(l3)] + seg_out
                            used_fused_mix = True
                    if not used_fused_mix:
                        sparse_repr = (sparse_list[g_idx] if _USE_SPARSE_TP else None) if sparse_list is not None else None
                        if sparse_repr is not None and _tp_fused_outer_proj_sparse is not None:
                            d_idx, k_idx, vals = sparse_repr
                            y_flat = _tp_fused_outer_proj_sparse(a_flat, b_flat, d_idx, k_idx, vals, m1, m2, k_total)
                            if y_flat is not None:
                                y = y_flat.reshape(*batch_shape, self.mul_in1, self.mul_in2, k_total)
                        if y is None and _tp_fused_outer_proj is not None:
                            y_flat = _tp_fused_outer_proj(a_flat, b_flat, U, m1, m2)
                            if y_flat is not None:
                                y = y_flat.reshape(*batch_shape, self.mul_in1, self.mul_in2, k_total)
                if not used_fused_mix:
                    if y is None:
                        # PyTorch path: outer product then matmul projection
                        t_mn = (a_comp.unsqueeze(-2).unsqueeze(-1) * b_comp.unsqueeze(-3).unsqueeze(-2))  # (..., i, j, m1, m2)
                        t_flat = t_mn.reshape(*batch_shape, self.mul_in1, self.mul_in2, m1 * m2)
                        if not t_flat.is_contiguous():
                            t_flat = t_flat.contiguous()
                        y = torch.matmul(t_flat, U)  # (..., i, j, K_total) in compute_dtype

                    # Per-path channel mixing
                    i, j = self.mul_in1, self.mul_in2
                    ij = i * j
                    for p_idx, l3, s, e in segments:  # type: ignore[misc]
                        Wp = w_param[int(p_idx)]  # (o,i,j)
                        Wp_comp = Wp.to(dtype=compute_dtype) if Wp.dtype != compute_dtype else Wp
                        y_seg = y[..., :, :, int(s): int(e)]  # (..., i, j, kdim)
                        kdim = int(e) - int(s)
                        y2 = y_seg.movedim(-1, -3).contiguous().view(*y_seg.shape[:-3], kdim, ij)  # (..., k, ij)
                        W2 = Wp_comp.contiguous().view(Wp_comp.shape[0], ij)  # (o, ij)
                        out_seg = torch.matmul(y2, W2.transpose(0, 1)).movedim(-1, -2)  # (..., o, k)
                        out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                        if weights is not None:
                            gate = weights[..., int(p_idx)]
                            out_seg = out_seg * gate[..., None, None]
                        out[int(l3)] = out[int(l3)] + out_seg
        # Fast path: external full per-example weights (..., weight_numel).
        # Still uses the e3nn-like factorization (projection first, then channel mixing).
        elif weights is not None and weights.shape[-1] == self.weight_numel:
            proj_list = self._get_proj_group_list(device=device, dtype=dtype)
            sparse_list = self._get_proj_sparse_list(device=device, dtype=dtype) if device.type == "cuda" else None
            # Reshape once:
            #   weights_full: (..., P, o, i, j)
            weights_full = weights.view(*batch_shape, self.num_paths, self.mul_out, self.mul_in1, self.mul_in2)
            for g_idx, g in enumerate(self._groups):
                l1 = int(g["l1"])  # type: ignore[arg-type]
                l2 = int(g["l2"])  # type: ignore[arg-type]
                segments = g["segments"]  # type: ignore[assignment]
                k_total = int(g["k_total"])  # type: ignore[arg-type]

                a = x1.get(l1)
                b = x2.get(l2)
                if a is None or b is None:
                    continue

                m1 = 2 * l1 + 1
                m2 = 2 * l2 + 1
                U = proj_list[g_idx]  # (m1*m2, K_total) in compute_dtype
                # Convert inputs to compute_dtype for numerical stability
                a_comp = a.to(dtype=compute_dtype) if a.dtype != compute_dtype else a
                b_comp = b.to(dtype=compute_dtype) if b.dtype != compute_dtype else b
                # FlashTP-style: sparse or dense fused outer-product + projection
                y = None
                if a_comp.is_cuda and a_comp.dim() >= 2:
                    B_flat = 1
                    for s in batch_shape:
                        B_flat *= int(s)
                    a_flat = a_comp.reshape(B_flat, self.mul_in1, m1)
                    b_flat = b_comp.reshape(B_flat, self.mul_in2, m2)
                    sparse_repr = (sparse_list[g_idx] if _USE_SPARSE_TP else None) if sparse_list is not None else None
                    if sparse_repr is not None and _tp_fused_outer_proj_sparse is not None:
                        d_idx, k_idx, vals = sparse_repr
                        y_flat = _tp_fused_outer_proj_sparse(a_flat, b_flat, d_idx, k_idx, vals, m1, m2, k_total)
                        if y_flat is not None:
                            y = y_flat.reshape(*batch_shape, self.mul_in1, self.mul_in2, k_total)
                    if y is None and _tp_fused_outer_proj is not None:
                        y_flat = _tp_fused_outer_proj(a_flat, b_flat, U, m1, m2)
                        if y_flat is not None:
                            y = y_flat.reshape(*batch_shape, self.mul_in1, self.mul_in2, k_total)
                if y is None:
                    t_mn = (a_comp.unsqueeze(-2).unsqueeze(-1) * b_comp.unsqueeze(-3).unsqueeze(-2))
                    t_flat = t_mn.reshape(*batch_shape, self.mul_in1, self.mul_in2, m1 * m2)
                    if not t_flat.is_contiguous():
                        t_flat = t_flat.contiguous()
                    y = torch.matmul(t_flat, U)
                if y.shape[-1] != k_total:
                    raise RuntimeError("ICTD TP projection produced wrong K_total")

                # Batch channel mixing for all paths in this group
                num_paths_in_group = len(segments)
                # Extract weights for this group: (..., P_g, o, i, j)
                p_indices = [int(p_idx) for p_idx, _, _, _ in segments]
                W_stack = weights_full[..., p_indices, :, :, :]  # (..., P_g, o, i, j)
                W_stack_comp = W_stack.to(dtype=compute_dtype) if W_stack.dtype != compute_dtype else W_stack
                
                # Reshape y for batched matmul: (..., i*j, K_total)
                i, j = self.mul_in1, self.mul_in2
                y_reshaped = y.permute(*range(len(batch_shape)), -3, -2, -1).reshape(*batch_shape, i * j, k_total)  # (..., ij, K)
                
                # Reshape W_stack: (..., P_g, o, i*j)
                W_reshaped = W_stack_comp.reshape(*batch_shape, num_paths_in_group, self.mul_out, i * j)  # (..., P_g, o, ij)
                
                # Batched matmul: (..., ij, K) @ (..., P_g, ij, o) -> (..., P_g, o, K)
                # We need: (..., 1, ij, K) @ (..., P_g, ij, o) -> (..., P_g, 1, o, K) -> (..., P_g, o, K)
                y_expanded = y_reshaped.unsqueeze(-3)  # (..., 1, ij, K)
                W_transposed = W_reshaped.transpose(-2, -1)  # (..., P_g, ij, o)
                out_group = torch.matmul(y_expanded, W_transposed)  # (..., P_g, 1, o, K) in compute_dtype
                out_group = out_group.squeeze(-2).permute(*range(len(batch_shape)), 0, 1, 2)  # (..., P_g, o, K_total)
                # Convert back to output dtype
                out_group = out_group.to(dtype=dtype) if out_group.dtype != dtype else out_group
                
                # Segment and accumulate to output
                for seg_idx, (p_idx, l3, s, e) in enumerate(segments):  # type: ignore[misc]
                    kdim = int(e) - int(s)
                    out_seg = out_group[..., seg_idx, :, int(s): int(e)]  # (..., o, kdim)
                    out[int(l3)] = out[int(l3)] + out_seg
        else:
            # Fallback: original per-path loop (supports external per-example weights).
            cg_list = self._get_cg_list(device=device, dtype=dtype)
            idx = 0
            for p_idx, (l1, l2, l3) in enumerate(self.paths):
                if self.internal_weights:
                    gate = 1.0
                    if weights is not None and weights.shape[-1] == self.num_paths:
                        gate = weights[..., p_idx]
                    Wp = w_param[p_idx]  # (o,i,j)
                else:
                    assert weights is not None
                    block = self.mul_out * self.mul_in1 * self.mul_in2
                    Wp = weights[..., idx: idx + block].view(*batch_shape, self.mul_out, self.mul_in1, self.mul_in2)
                    idx += block
                    gate = 1.0

                a = x1.get(l1)
                b = x2.get(l2)
                if a is None or b is None:
                    continue

                # Convert to compute_dtype for numerical stability
                a_comp = a.to(dtype=compute_dtype) if a.dtype != compute_dtype else a
                b_comp = b.to(dtype=compute_dtype) if b.dtype != compute_dtype else b
                Wp_comp = Wp.to(dtype=compute_dtype) if Wp.dtype != compute_dtype else Wp
                C = cg_list[p_idx]  # (m1,m2,m3) in compute_dtype
                out_l3 = torch.einsum("...im,...jn,mnk,oij->...ok", a_comp, b_comp, C, Wp_comp)
                # Convert back to output dtype
                out_l3 = out_l3.to(dtype=dtype) if out_l3.dtype != dtype else out_l3
                if not isinstance(gate, float):
                    out_l3 = out_l3 * gate[..., None, None]
                out[l3] = out[l3] + out_l3

        return out

