"""
ICTD-style SO(2) / O(2) irreps and tensor products in a real Cartesian basis.

This module mirrors the 3D `ictd_irreps.py` design at a smaller scope:

  - harmonic bases are built directly in the 2D monomial basis
  - symmetric trace-chain projectors / reconstructors are available for Sym^L(R^2)
  - direction harmonics, CG-like couplings, elementwise products, fully-connected
    tensor products, and lightweight path-weighted tensor products are provided

The public basis convention is:

  - SO(2): integer frequency `m >= 0`
      dim(m) = 1 for m=0
      dim(m) = 2 for m>0, ordered as [cos(m theta), sin(m theta)]
  - O(2): keys are either scalar parity (`"0e"`, `"0o"`) or frequency strings
      (`"1"`, `"2"`, ...)

This is a standalone 2D operator family. It does not handle 3D local frames.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _resolve_internal_compute_dtype(internal_compute_dtype: torch.dtype | None) -> torch.dtype:
    return torch.get_default_dtype() if internal_compute_dtype is None else internal_compute_dtype


def so2_irrep_dim(m: int) -> int:
    m = int(m)
    if m < 0:
        raise ValueError(f"frequency m must be >= 0, got {m}")
    return 1 if m == 0 else 2


def sym_dim_2d(L: int) -> int:
    """dim Sym^L(R^2) = L + 1."""
    L = int(L)
    if L < 0:
        raise ValueError(f"L must be >= 0, got {L}")
    return L + 1


def _counts_list_2d(L: int) -> List[Tuple[int, int]]:
    """All (a, b) with a + b = L in deterministic order."""
    L = int(L)
    if L < 0:
        raise ValueError(f"L must be >= 0, got {L}")
    return [(a, L - a) for a in range(L + 1)]


def _double_factorial(n: int) -> int:
    if n <= 0:
        return 1
    out = 1
    for k in range(n, 0, -2):
        out *= k
    return out


def _gaussian_moment(n: int) -> float:
    if n % 2 == 1:
        return 0.0
    return float(_double_factorial(n - 1))


@lru_cache(maxsize=None)
def _gram_gaussian_2d(L: int) -> torch.Tensor:
    """
    O(2)-invariant Gram matrix on Sym^L in the monomial basis x^a y^b.
    """
    counts = _counts_list_2d(L)
    D = len(counts)
    G = torch.zeros(D, D, dtype=torch.float64)
    for i, (a, b) in enumerate(counts):
        for j, (a2, b2) in enumerate(counts):
            G[i, j] = _gaussian_moment(a + a2) * _gaussian_moment(b + b2)
    return G


@lru_cache(maxsize=None)
def _build_laplacian_matrix_2d(L: int) -> torch.Tensor:
    """
    2D Laplacian Δ = ∂_x^2 + ∂_y^2 on homogeneous polynomials Sym^L -> Sym^{L-2}.
    """
    L = int(L)
    if L < 2:
        raise ValueError(f"L must be >= 2, got {L}")
    src = _counts_list_2d(L)
    dst = _counts_list_2d(L - 2)
    j_of = {t: i for i, t in enumerate(dst)}
    D = torch.zeros(len(dst), len(src), dtype=torch.float64)
    for col, (a, b) in enumerate(src):
        if a >= 2:
            D[j_of[(a - 2, b)], col] += float(a * (a - 1))
        if b >= 2:
            D[j_of[(a, b - 2)], col] += float(b * (b - 1))
    return D


@lru_cache(maxsize=None)
def _build_r2k_lift_2d(L_src: int, k: int) -> torch.Tensor:
    """
    Multiply by (x^2 + y^2)^k in the 2D monomial basis.
    """
    L_src = int(L_src)
    k = int(k)
    if L_src < 0 or k < 0:
        raise ValueError(f"L_src and k must be >= 0, got {L_src}, {k}")
    L_dst = L_src + 2 * k
    src = _counts_list_2d(L_src)
    dst = _counts_list_2d(L_dst)
    idx_dst = {t: i for i, t in enumerate(dst)}
    M = torch.zeros(len(dst), len(src), dtype=torch.float64)
    for col, (a, b) in enumerate(src):
        for q in range(k + 1):
            a_dst = a + 2 * q
            b_dst = b + 2 * (k - q)
            M[idx_dst[(a_dst, b_dst)], col] += float(math.comb(k, q))
    return M.contiguous()


def _fix_vector_sign(v: torch.Tensor) -> torch.Tensor:
    out = v.clone()
    for idx in range(out.numel()):
        val = float(out[idx].item())
        if abs(val) > 1e-12:
            if val < 0.0:
                out = -out
            break
    return out


@lru_cache(maxsize=None)
def _harmonic_basis_cpu_f64_2d(m: int) -> torch.Tensor:
    """
    Harmonic basis in monomial coefficients for the 2D real basis:
      m = 0: [1]
      m > 0: [Re((x+i y)^m), Im((x+i y)^m)] normalized under the Gaussian Gram.
    """
    m = int(m)
    if m < 0:
        raise ValueError(f"m must be >= 0, got {m}")
    if m == 0:
        return torch.ones(1, 1, dtype=torch.float64)

    counts = _counts_list_2d(m)
    cos_vec = torch.zeros(len(counts), dtype=torch.float64)
    sin_vec = torch.zeros(len(counts), dtype=torch.float64)
    for idx, (a, b) in enumerate(counts):
        coef = float(math.comb(m, a))
        phase = b % 4
        if phase == 0:
            cos_val = 1.0
            sin_val = 0.0
        elif phase == 1:
            cos_val = 0.0
            sin_val = 1.0
        elif phase == 2:
            cos_val = -1.0
            sin_val = 0.0
        else:
            cos_val = 0.0
            sin_val = -1.0
        cos_vec[idx] = coef * cos_val
        sin_vec[idx] = coef * sin_val

    G = _gram_gaussian_2d(m)

    cos_norm = torch.sqrt(torch.clamp(cos_vec @ (G @ cos_vec), min=1e-30))
    cos_vec = cos_vec / cos_norm

    proj = cos_vec @ (G @ sin_vec)
    sin_vec = sin_vec - proj * cos_vec
    sin_norm = torch.sqrt(torch.clamp(sin_vec @ (G @ sin_vec), min=1e-30))
    sin_vec = sin_vec / sin_norm

    cos_vec = _fix_vector_sign(cos_vec)
    sin_vec = _fix_vector_sign(sin_vec)
    if (m % 2) == 0:
        cos_vec = -cos_vec
    return torch.stack([cos_vec, sin_vec], dim=-1).contiguous()


def _harmonic_basis_t_2d(m: int, device=None, dtype=None) -> torch.Tensor:
    return _harmonic_basis_cpu_f64_2d(int(m)).to(device=device, dtype=dtype)


@dataclass(frozen=True)
class HarmonicProjectors2D:
    Lmax: int
    P: Dict[Tuple[int, int], torch.Tensor]  # (L, m) -> (dim(m), Dsym(L))


@dataclass(frozen=True)
class HarmonicReconstructors2D:
    Lmax: int
    V: Dict[Tuple[int, int], torch.Tensor]  # (L, m) -> (Dsym(L), dim(m))


@lru_cache(maxsize=None)
def build_harmonic_projectors_2d(Lmax: int) -> HarmonicProjectors2D:
    P: Dict[Tuple[int, int], torch.Tensor] = {}
    Lmax = int(Lmax)
    for L in range(Lmax + 1):
        G_L = _gram_gaussian_2d(L)
        for k in range(L // 2 + 1):
            m = L - 2 * k
            B_m = _harmonic_basis_t_2d(m, dtype=torch.float64)
            M = _build_r2k_lift_2d(m, k)
            V = (M @ B_m).contiguous()
            G = V.T @ G_L @ V
            P[(L, m)] = torch.linalg.solve(G, V.T @ G_L).contiguous()
    return HarmonicProjectors2D(Lmax=Lmax, P=P)


@lru_cache(maxsize=None)
def build_harmonic_reconstructors_2d(Lmax: int) -> HarmonicReconstructors2D:
    V: Dict[Tuple[int, int], torch.Tensor] = {}
    Lmax = int(Lmax)
    for L in range(Lmax + 1):
        for k in range(L // 2 + 1):
            m = L - 2 * k
            B_m = _harmonic_basis_t_2d(m, dtype=torch.float64)
            M = _build_r2k_lift_2d(m, k)
            V[(L, m)] = (M @ B_m).contiguous()
    return HarmonicReconstructors2D(Lmax=Lmax, V=V)


@lru_cache(maxsize=None)
def _dir_monomial_exps_coefs_2d(m: int) -> Tuple[torch.Tensor, torch.Tensor]:
    counts = _counts_list_2d(m)
    exps = torch.tensor(counts, dtype=torch.int64)
    coefs = torch.tensor([float(math.comb(m, a)) for a, _ in counts], dtype=torch.float64)
    return exps, coefs


@lru_cache(maxsize=None)
def _dir_proj_cpu_f64_2d(m: int) -> torch.Tensor:
    if int(m) == 0:
        return torch.ones(1, 1, dtype=torch.float64)
    B = _harmonic_basis_t_2d(m, dtype=torch.float64)
    G = _gram_gaussian_2d(m)
    return (G @ B).contiguous()


_dir_proj_cache_by_dev_dtype_2d: Dict[Tuple[str, str, int], torch.Tensor] = {}
_dir_exps_cache_by_dev_2d: Dict[Tuple[str, int], torch.Tensor] = {}
_dir_coefs_cache_by_dev_dtype_2d: Dict[Tuple[str, str, int], torch.Tensor] = {}


def _integer_power_table(x: torch.Tensor, max_power: int) -> torch.Tensor:
    if max_power < 0:
        raise ValueError(f"max_power must be >= 0, got {max_power}")
    powers = [torch.ones_like(x)]
    cur = torch.ones_like(x)
    for _ in range(int(max_power)):
        cur = cur * x
        powers.append(cur)
    return torch.stack(powers, dim=-1)


def direction_harmonics_so2(n: torch.Tensor, m: int) -> torch.Tensor:
    """
    Harmonic coordinates of degree `m` for 2D directions `n[..., 2]`.
    """
    m = int(m)
    if n.shape[-1] != 2:
        raise ValueError(f"direction_harmonics_so2 expects last dim 2, got {n.shape[-1]}")
    if m < 0:
        raise ValueError(f"m must be >= 0, got {m}")
    if m == 0:
        return torch.ones(*n.shape[:-1], 1, device=n.device, dtype=n.dtype)

    key = (str(n.device), str(n.dtype), m)
    P = _dir_proj_cache_by_dev_dtype_2d.get(key)
    if P is None:
        P = _dir_proj_cpu_f64_2d(m).to(device=n.device, dtype=n.dtype)
        _dir_proj_cache_by_dev_dtype_2d[key] = P

    exps_key = (str(n.device), m)
    exps = _dir_exps_cache_by_dev_2d.get(exps_key)
    if exps is None:
        exps = _dir_monomial_exps_coefs_2d(m)[0].to(device=n.device)
        _dir_exps_cache_by_dev_2d[exps_key] = exps

    coefs_key = (str(n.device), str(n.dtype), m)
    coefs = _dir_coefs_cache_by_dev_dtype_2d.get(coefs_key)
    if coefs is None:
        coefs = _dir_monomial_exps_coefs_2d(m)[1].to(device=n.device, dtype=n.dtype)
        _dir_coefs_cache_by_dev_dtype_2d[coefs_key] = coefs

    nx, ny = n[..., 0], n[..., 1]
    a = exps[:, 0]
    b = exps[:, 1]
    x_pows = _integer_power_table(nx, m)
    y_pows = _integer_power_table(ny, m)
    t = x_pows[..., a] * y_pows[..., b]
    t = t * coefs
    return t @ P


def direction_harmonics_all_so2(n: torch.Tensor, mmax: int) -> List[torch.Tensor]:
    mmax = int(mmax)
    if mmax < 0:
        raise ValueError(f"mmax must be >= 0, got {mmax}")
    return [direction_harmonics_so2(n, m) for m in range(mmax + 1)]


def _parse_so2_irreps_string(irreps: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for part in re.split(r"[+,]", irreps):
        tok = part.strip().lower()
        if not tok:
            continue
        m = re.match(r"^(\d*)x?(\d+)$", tok)
        if not m:
            raise ValueError(f"Invalid SO(2) irrep token: {part!r}")
        m_val = int(m.group(2))
        if m_val not in seen:
            out.append(m_val)
            seen.add(m_val)
    return out


def direction_harmonics_irreps_o2(n: torch.Tensor, irreps: str | Sequence[str]) -> torch.Tensor:
    """
    Canonical O(2) direction harmonics for 2D directions.

    Supported outputs:
      - "0e" -> scalar ones
      - "m" for m>=1 -> SO(2) frequency block

    "0o" is not constructible from a single plain 2D direction and raises.
    """
    if isinstance(irreps, str):
        tokens = [tok.strip() for tok in re.split(r"[+,]", irreps) if tok.strip()]
    else:
        tokens = [str(tok).strip() for tok in irreps if str(tok).strip()]

    chunks: List[torch.Tensor] = []
    for tok in tokens:
        key = _normalize_o2_irrep_key(tok)
        if key == ("scalar", -1):
            raise ValueError(
                "direction_harmonics_irreps_o2 does not support '0o': a single plain 2D direction "
                "only generates canonical outputs '0e' and frequency blocks."
            )
        if key == ("scalar", 1):
            chunks.append(torch.ones(*n.shape[:-1], 1, device=n.device, dtype=n.dtype))
        else:
            chunks.append(direction_harmonics_so2(n, key[1]))
    if not chunks:
        raise ValueError("No irreps requested")
    return torch.cat(chunks, dim=-1)


def split_flat_irreps_so2(x: torch.Tensor, channels: int, mmax: int) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    idx = 0
    for m in range(int(mmax) + 1):
        d = int(channels) * so2_irrep_dim(m)
        blk = x[..., idx : idx + d]
        idx += d
        out[m] = blk.view(*x.shape[:-1], int(channels), so2_irrep_dim(m))
    return out


def merge_flat_irreps_so2(blocks: Dict[int, torch.Tensor], channels: int, mmax: int) -> torch.Tensor:
    parts = []
    for m in range(int(mmax) + 1):
        parts.append(blocks[m].reshape(*blocks[m].shape[:-2], int(channels) * so2_irrep_dim(m)))
    return torch.cat(parts, dim=-1)


def _apply_channel_adapter(block: torch.Tensor, adapter: nn.Module) -> torch.Tensor:
    if isinstance(adapter, nn.Identity):
        return block
    param = next(adapter.parameters(), None)
    linear_dtype = block.dtype if param is None else param.dtype
    y = adapter(block.to(dtype=linear_dtype).movedim(-2, -1))
    return y.movedim(-1, -2).to(dtype=block.dtype)


class EquivariantChannelLinearSO2(nn.Module):
    def __init__(self, channels: int, mmax: int, bias: bool = False):
        super().__init__()
        self.channels = int(channels)
        self.mmax = int(mmax)
        self.adapters = nn.ModuleDict(
            {str(m): nn.Linear(self.channels, self.channels, bias=bias) for m in range(self.mmax + 1)}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = split_flat_irreps_so2(x, self.channels, self.mmax)
        for m in range(self.mmax + 1):
            blocks[m] = _apply_channel_adapter(blocks[m], self.adapters[str(m)])
        return merge_flat_irreps_so2(blocks, self.channels, self.mmax)


class EquivariantChannelLinearSO2Rect(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mmax: int, bias: bool = False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.mmax = int(mmax)
        self.adapters = nn.ModuleDict(
            {str(m): nn.Linear(self.in_channels, self.out_channels, bias=bias) for m in range(self.mmax + 1)}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = split_flat_irreps_so2(x, self.in_channels, self.mmax)
        for m in range(self.mmax + 1):
            blocks[m] = _apply_channel_adapter(blocks[m], self.adapters[str(m)])
        return merge_flat_irreps_so2(blocks, self.out_channels, self.mmax)


O2IrrepKey = Tuple[str, int]


def _normalize_o2_irrep_key(key: str | O2IrrepKey) -> O2IrrepKey:
    if isinstance(key, tuple):
        tag, value = key
        tag = str(tag).strip().lower()
        value_i = int(value)
        if tag == "scalar" and value_i in (-1, 1):
            return ("scalar", value_i)
        if tag == "freq" and value_i >= 1:
            return ("freq", value_i)
        raise ValueError(f"Invalid O(2) key {key!r}")

    tok = str(key).strip().lower()
    if tok == "0e":
        return ("scalar", 1)
    if tok == "0o":
        return ("scalar", -1)
    if tok.isdigit() and int(tok) >= 1:
        return ("freq", int(tok))
    raise ValueError(f"Invalid O(2) irrep token {key!r}")


def _o2_key_to_token(key: O2IrrepKey) -> str:
    key = _normalize_o2_irrep_key(key)
    if key[0] == "scalar":
        return "0e" if key[1] == 1 else "0o"
    return str(key[1])


def _o2_irrep_dim(key: O2IrrepKey) -> int:
    key = _normalize_o2_irrep_key(key)
    return 1 if key[0] == "scalar" else 2


def parse_o2_active_irreps(active_irreps: Sequence[str | O2IrrepKey]) -> List[O2IrrepKey]:
    out = [_normalize_o2_irrep_key(key) for key in active_irreps]
    if len(set(out)) != len(out):
        raise ValueError(f"Duplicate O(2) irreps are not supported: {active_irreps!r}")
    return out


def split_flat_irreps_o2(
    x: torch.Tensor,
    channels: int,
    active_irreps: Sequence[str | O2IrrepKey],
) -> Dict[O2IrrepKey, torch.Tensor]:
    keys = parse_o2_active_irreps(active_irreps)
    out: Dict[O2IrrepKey, torch.Tensor] = {}
    idx = 0
    for key in keys:
        d = int(channels) * _o2_irrep_dim(key)
        blk = x[..., idx : idx + d]
        idx += d
        out[key] = blk.view(*x.shape[:-1], int(channels), _o2_irrep_dim(key))
    return out


def merge_flat_irreps_o2(
    blocks: Dict[O2IrrepKey, torch.Tensor],
    channels: int,
    active_irreps: Sequence[str | O2IrrepKey],
) -> torch.Tensor:
    keys = parse_o2_active_irreps(active_irreps)
    parts = [blocks[key].reshape(*blocks[key].shape[:-2], int(channels) * _o2_irrep_dim(key)) for key in keys]
    return torch.cat(parts, dim=-1)


class EquivariantChannelLinearO2(nn.Module):
    def __init__(self, channels: int, active_irreps: Sequence[str | O2IrrepKey], bias: bool = False):
        super().__init__()
        self.channels = int(channels)
        self.active_irreps = parse_o2_active_irreps(active_irreps)
        self.adapters = nn.ModuleDict(
            {_o2_key_to_token(key): nn.Linear(self.channels, self.channels, bias=bias) for key in self.active_irreps}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = split_flat_irreps_o2(x, self.channels, self.active_irreps)
        for key in self.active_irreps:
            blocks[key] = _apply_channel_adapter(blocks[key], self.adapters[_o2_key_to_token(key)])
        return merge_flat_irreps_o2(blocks, self.channels, self.active_irreps)


class EquivariantChannelLinearO2Rect(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        active_irreps: Sequence[str | O2IrrepKey],
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.active_irreps = parse_o2_active_irreps(active_irreps)
        self.adapters = nn.ModuleDict(
            {
                _o2_key_to_token(key): nn.Linear(self.in_channels, self.out_channels, bias=bias)
                for key in self.active_irreps
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = split_flat_irreps_o2(x, self.in_channels, self.active_irreps)
        for key in self.active_irreps:
            blocks[key] = _apply_channel_adapter(blocks[key], self.adapters[_o2_key_to_token(key)])
        return merge_flat_irreps_o2(blocks, self.out_channels, self.active_irreps)


@lru_cache(maxsize=None)
def _build_poly_mult_matrix_2d(m1: int, m2: int, M: int) -> torch.Tensor:
    counts1 = _counts_list_2d(m1)
    counts2 = _counts_list_2d(m2)
    countsM = _counts_list_2d(M)
    idxM = {t: i for i, t in enumerate(countsM)}
    D1, D2, DM = len(counts1), len(counts2), len(countsM)
    out = torch.zeros(DM, D1 * D2, dtype=torch.float64)
    for i, c1 in enumerate(counts1):
        for j, c2 in enumerate(counts2):
            k = idxM[(c1[0] + c2[0], c1[1] + c2[1])]
            out[k, i * D2 + j] = 1.0
    return out.contiguous()


@lru_cache(maxsize=None)
def build_cg_tensor_so2(m1: int, m2: int, m3: int) -> torch.Tensor:
    """
    Symmetric-product SO(2) coupling in the real [cos, sin] basis.
    """
    m1 = int(m1)
    m2 = int(m2)
    m3 = int(m3)
    if min(m1, m2, m3) < 0:
        raise ValueError(f"m1, m2, m3 must be >= 0, got {(m1, m2, m3)}")
    if m3 > m1 + m2 or ((m1 + m2 + m3) % 2 == 1):
        return torch.zeros(so2_irrep_dim(m1), so2_irrep_dim(m2), so2_irrep_dim(m3), dtype=torch.float64)

    M = m1 + m2
    proj = build_harmonic_projectors_2d(M)
    P = proj.P[(M, m3)]
    B1 = _harmonic_basis_t_2d(m1, dtype=torch.float64)
    B2 = _harmonic_basis_t_2d(m2, dtype=torch.float64)
    M_poly = _build_poly_mult_matrix_2d(m1, m2, M)

    outer = torch.einsum("im,jn->ijmn", B1, B2)
    outer_flat = outer.reshape(B1.shape[0] * B2.shape[0], so2_irrep_dim(m1) * so2_irrep_dim(m2))
    tM = M_poly @ outer_flat
    c3 = P @ tM
    return c3.T.reshape(so2_irrep_dim(m1), so2_irrep_dim(m2), so2_irrep_dim(m3)).contiguous()


def _so2_valid_intermediates(m1: int, m2: int, m3: int, *, m_out: int) -> List[int]:
    mids: List[int] = []
    for mid in _so2_valid_outputs(m1, m2):
        if m_out in _so2_valid_outputs(mid, m3):
            mids.append(mid)
    return mids


@lru_cache(maxsize=None)
def build_recoupling_tensor_so2(
    m_a: int,
    m_b: int,
    m_c: int,
    m_out: int,
    coupling_tree: str = "left",
) -> torch.Tensor:
    """
    Fixed triple-product recoupling tensor in the real SO(2) basis.

    Returns T with shape [dim(a), dim(b), dim(c), dim(out)] such that:
      y[out] = sum_{a,b,c} x_a x_b x_c T[a,b,c,out]
    for the chosen binary coupling tree.
    """
    m_a = int(m_a)
    m_b = int(m_b)
    m_c = int(m_c)
    m_out = int(m_out)
    if min(m_a, m_b, m_c, m_out) < 0:
        raise ValueError(f"frequencies must be >=0, got {(m_a, m_b, m_c, m_out)}")
    tree = str(coupling_tree).strip().lower()
    da, db, dc, do = (so2_irrep_dim(m) for m in (m_a, m_b, m_c, m_out))
    T = torch.zeros(da, db, dc, do, dtype=torch.float64)
    if tree == "left":
        for mid in _so2_valid_intermediates(m_a, m_b, m_c, m_out=m_out):
            C1 = build_cg_tensor_so2(m_a, m_b, mid)
            C2 = build_cg_tensor_so2(mid, m_c, m_out)
            T = T + torch.einsum("abu,ucd->abcd", C1, C2)
    elif tree == "right":
        for mid in _so2_valid_intermediates(m_b, m_c, m_a, m_out=m_out):
            C1 = build_cg_tensor_so2(m_b, m_c, mid)
            C2 = build_cg_tensor_so2(m_a, mid, m_out)
            T = T + torch.einsum("bcv,avd->abcd", C1, C2)
    else:
        raise ValueError(f"coupling_tree must be 'left' or 'right', got {coupling_tree!r}")
    return T.contiguous()


@lru_cache(maxsize=None)
def build_associator_tensor_so2(
    m_a: int,
    m_b: int,
    m_c: int,
    m_out: int,
) -> torch.Tensor:
    """
    Change-of-coupling tensor between left and right SO(2) recoupling trees.

    In the current real basis this is typically close to identity, but we build it
    explicitly from the two direct triple-product maps.
    """
    left = build_recoupling_tensor_so2(m_a, m_b, m_c, m_out, "left").reshape(-1, so2_irrep_dim(m_out))
    right = build_recoupling_tensor_so2(m_a, m_b, m_c, m_out, "right").reshape(-1, so2_irrep_dim(m_out))
    return torch.linalg.lstsq(left, right).solution.contiguous()


def cg_tensor_sparsity_so2(C: torch.Tensor, threshold: float = 1e-10) -> Tuple[int, int, float]:
    n = C.numel()
    nz = int((C.abs() > threshold).sum().item())
    return n, nz, 1.0 - (nz / max(n, 1))


def _normalize_cg_tensor(C: torch.Tensor, output_dim: int, normalization: str) -> torch.Tensor:
    C_fn = C.norm().item()
    if normalization == "component" and C_fn > 1e-30:
        return C * (math.sqrt(output_dim) / C_fn)
    if normalization == "norm" and C_fn > 1e-30:
        return C * (1.0 / C_fn)
    return C


def _so2_valid_outputs(m1: int, m2: int, mmax: int | None = None) -> List[int]:
    outs = []
    diff = abs(int(m1) - int(m2))
    summ = int(m1) + int(m2)
    for cand in (diff, summ):
        if mmax is not None and cand > int(mmax):
            continue
        if cand not in outs:
            outs.append(cand)
    return outs


def _expand_path_gate(gates: torch.Tensor, p_idx: int, batch_ndim: int, dtype: torch.dtype) -> torch.Tensor:
    gate = gates[..., int(p_idx)].to(dtype=dtype)
    return gate.view(*gate.shape, *([1] * (2 - batch_ndim if batch_ndim > 2 else 0)))


class HarmonicElementwiseProductSO2(nn.Module):
    """
    Same-frequency elementwise product in the SO(2) ICTD basis.
    """

    def __init__(
        self,
        mmax: int,
        mul: int,
        irreps_out: str | None = "0",
        normalization: str = "component",
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.mmax = int(mmax)
        self.mul = int(mul)
        self.internal_compute_dtype = _resolve_internal_compute_dtype(internal_compute_dtype)
        self._normalization = normalization
        self._irreps_out = irreps_out.strip().lower() if (irreps_out and isinstance(irreps_out, str)) else "full"
        self._output_scalar_only = self._irreps_out == "0"
        self._filter_m: Optional[List[int]] = None
        if self._irreps_out not in ("0", "full"):
            self._filter_m = _parse_so2_irreps_string(self._irreps_out)

        self.paths: List[Tuple[int, int]] = []
        self._cg_cache_cpu_f64: List[torch.Tensor] = []
        for m in range(self.mmax + 1):
            for m3 in _so2_valid_outputs(m, m):
                self.paths.append((m, m3))
                C = build_cg_tensor_so2(m, m, m3)
                self._cg_cache_cpu_f64.append(_normalize_cg_tensor(C, so2_irrep_dim(m3), normalization))
        self._cg_cache_by_dev_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}

    def _get_cg_list(self, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        key = (str(device), str(self.internal_compute_dtype))
        cached = self._cg_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        cg_list = [C.to(device=device, dtype=self.internal_compute_dtype) for C in self._cg_cache_cpu_f64]
        self._cg_cache_by_dev_dtype[key] = cg_list
        return cg_list

    def forward(self, x1: Dict[int, torch.Tensor], x2: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor] | torch.Tensor:
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        cg_list = self._get_cg_list(device=device, dtype=dtype)

        if self._output_scalar_only:
            outs = []
            for p_idx, (m, m3) in enumerate(self.paths):
                if m3 != 0:
                    continue
                y = torch.einsum(
                    "...cm,...cn,mnk->...ck",
                    x1[m].to(dtype=self.internal_compute_dtype),
                    x2[m].to(dtype=self.internal_compute_dtype),
                    cg_list[p_idx],
                )
                outs.append(y[..., 0].to(dtype=dtype))
            return torch.cat(outs, dim=-1)

        out: Dict[int, List[torch.Tensor]] = {}
        for p_idx, (m, m3) in enumerate(self.paths):
            y = torch.einsum(
                "...cm,...cn,mnk->...ck",
                x1[m].to(dtype=self.internal_compute_dtype),
                x2[m].to(dtype=self.internal_compute_dtype),
                cg_list[p_idx],
            ).to(dtype=dtype)
            out.setdefault(m3, []).append(y)

        if self._filter_m is not None:
            parts = []
            for m3 in self._filter_m:
                if m3 in out:
                    parts.append(torch.cat(out[m3], dim=-2).reshape(*batch_shape, -1))
            if not parts:
                return torch.zeros(*batch_shape, 0, device=device, dtype=dtype)
            return torch.cat(parts, dim=-1)

        return {m3: torch.cat(parts, dim=-2) for m3, parts in out.items()}


class HarmonicFullyConnectedTensorProductSO2(nn.Module):
    """
    Fully-connected SO(2) tensor product in the 2D ICTD basis.
    """

    def __init__(
        self,
        mul_in1: int,
        mul_in2: int,
        mul_out: int,
        mmax: int,
        internal_weights: bool = True,
        *,
        allowed_paths: List[Tuple[int, int, int]] | None = None,
        path_policy: str = "full",
        max_rank_other: int | None = None,
        normalization: str = "component",
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.mul_in1 = int(mul_in1)
        self.mul_in2 = int(mul_in2)
        self.mul_out = int(mul_out)
        self.mmax = int(mmax)
        self.internal_weights = bool(internal_weights)
        self._normalization = normalization
        self.internal_compute_dtype = _resolve_internal_compute_dtype(internal_compute_dtype)

        all_paths: List[Tuple[int, int, int]] = []
        for m1 in range(self.mmax + 1):
            for m2 in range(self.mmax + 1):
                for m3 in _so2_valid_outputs(m1, m2, mmax=self.mmax):
                    all_paths.append((m1, m2, m3))

        if allowed_paths is not None:
            allowed_set = {tuple(map(int, p)) for p in allowed_paths}
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
        self.weight_numel = self.num_paths * self.mul_out * self.mul_in1 * self.mul_in2
        if self.internal_weights:
            self.weight = nn.Parameter(torch.randn(self.num_paths, self.mul_out, self.mul_in1, self.mul_in2) * 0.02)
        else:
            self.register_parameter("weight", None)

        self._cg_cpu_f64: List[torch.Tensor] = []
        for m1, m2, m3 in self.paths:
            C = build_cg_tensor_so2(m1, m2, m3)
            self._cg_cpu_f64.append(_normalize_cg_tensor(C, so2_irrep_dim(m3), normalization))
        self._cg_cache_by_dev_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}

    def _get_cg_list(self, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        key = (str(device), str(self.internal_compute_dtype))
        cached = self._cg_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        cg_list = [C.to(device=device, dtype=self.internal_compute_dtype) for C in self._cg_cpu_f64]
        self._cg_cache_by_dev_dtype[key] = cg_list
        return cg_list

    def _path_weight(
        self,
        weights: torch.Tensor | None,
        p_idx: int,
        batch_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.internal_weights:
            return self.weight[p_idx].to(device=device, dtype=dtype)
        if weights is None:
            raise ValueError("weights must be provided when internal_weights=False")
        expected_tail = (self.num_paths, self.mul_out, self.mul_in1, self.mul_in2)
        if tuple(weights.shape[-4:]) != expected_tail:
            raise ValueError(
                f"weights must end with {expected_tail}, got {tuple(weights.shape)}"
            )
        return weights[..., p_idx, :, :, :].to(device=device, dtype=dtype)

    def forward(
        self,
        x1: Dict[int, torch.Tensor],
        x2: Dict[int, torch.Tensor],
        weights: torch.Tensor | None = None,
    ) -> Dict[int, torch.Tensor]:
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        compute_dtype = self.internal_compute_dtype
        cg_list = self._get_cg_list(device=device, dtype=dtype)

        out = {
            m: torch.zeros(*batch_shape, self.mul_out, so2_irrep_dim(m), device=device, dtype=dtype)
            for m in range(self.mmax + 1)
        }

        path_gates: torch.Tensor | None = None
        if self.internal_weights and weights is not None:
            if weights.shape[-1] != self.num_paths:
                raise ValueError(f"path gates must have last dim {self.num_paths}, got {weights.shape[-1]}")
            path_gates = weights.to(device=device, dtype=compute_dtype)

        for p_idx, (m1, m2, m3) in enumerate(self.paths):
            Wp = self._path_weight(weights if not self.internal_weights else None, p_idx, batch_shape, device, compute_dtype)
            a = x1[m1].to(dtype=compute_dtype)
            b = x2[m2].to(dtype=compute_dtype)
            C = cg_list[p_idx]
            if self.internal_weights:
                y = torch.einsum("...im,...jn,mnk,oij->...ok", a, b, C, Wp)
                if path_gates is not None:
                    gate = path_gates[..., p_idx].view(*batch_shape, 1, 1)
                    y = y * gate
            else:
                y = torch.einsum("...im,...jn,mnk,...oij->...ok", a, b, C, Wp)
            out[m3] = out[m3] + y.to(dtype=dtype)
        return out


class HarmonicPathWeightedTensorProductSO2(nn.Module):
    def __init__(
        self,
        channels: int,
        mmax: int,
        *,
        allowed_paths: List[Tuple[int, int, int]] | None = None,
        path_policy: str = "full",
        max_rank_other: int | None = None,
        normalization: str = "component",
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.mmax = int(mmax)
        self._normalization = normalization
        self.internal_compute_dtype = internal_compute_dtype

        all_paths: List[Tuple[int, int, int]] = []
        for m1 in range(self.mmax + 1):
            for m2 in range(self.mmax + 1):
                for m3 in _so2_valid_outputs(m1, m2, mmax=self.mmax):
                    all_paths.append((m1, m2, m3))

        if allowed_paths is not None:
            allowed_set = {tuple(map(int, p)) for p in allowed_paths}
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
        self.weight = nn.Parameter(torch.randn(self.num_paths, self.channels) * 0.02)

        self._cg_cpu_f64: List[torch.Tensor] = []
        for m1, m2, m3 in self.paths:
            C = build_cg_tensor_so2(m1, m2, m3)
            self._cg_cpu_f64.append(_normalize_cg_tensor(C, so2_irrep_dim(m3), normalization))
        self._cg_cache_by_dev_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}

    def _get_cg_list(self, device: torch.device, compute_dtype: torch.dtype) -> List[torch.Tensor]:
        key = (str(device), str(compute_dtype))
        cached = self._cg_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        cg_list = [C.to(device=device, dtype=compute_dtype) for C in self._cg_cpu_f64]
        self._cg_cache_by_dev_dtype[key] = cg_list
        return cg_list

    def forward(self, x1: Dict[int, torch.Tensor], x2: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        compute_dtype = dtype if self.internal_compute_dtype is None else self.internal_compute_dtype
        cg_list = self._get_cg_list(device=device, compute_dtype=compute_dtype)
        w = self.weight.to(device=device, dtype=compute_dtype)

        out = {
            m: torch.zeros(*batch_shape, self.channels, so2_irrep_dim(m), device=device, dtype=dtype)
            for m in range(self.mmax + 1)
        }
        for p_idx, (m1, m2, m3) in enumerate(self.paths):
            y = torch.einsum(
                "...cm,...cn,mnk->...ck",
                x1[m1].to(dtype=compute_dtype),
                x2[m2].to(dtype=compute_dtype),
                cg_list[p_idx],
            )
            y = y * w[p_idx].view(*([1] * len(batch_shape)), self.channels, 1)
            out[m3] = out[m3] + y.to(dtype=dtype)
        return out


@lru_cache(maxsize=None)
def _build_o2_scalar_odd_cg(m: int) -> torch.Tensor:
    m = int(m)
    if m <= 0:
        raise ValueError(f"odd scalar coupling requires m>=1, got {m}")
    if so2_irrep_dim(m) != 2:
        raise AssertionError("odd scalar coupling only exists for 2D frequency blocks")
    C = torch.zeros(2, 2, 1, dtype=torch.float64)
    # Antisymmetric bilinear form; this is the O(2)-odd scalar branch.
    C[0, 1, 0] = 1.0
    C[1, 0, 0] = -1.0
    return C


@lru_cache(maxsize=None)
def _build_cg_tensor_o2(key1: O2IrrepKey, key2: O2IrrepKey, key3: O2IrrepKey) -> torch.Tensor:
    key1 = _normalize_o2_irrep_key(key1)
    key2 = _normalize_o2_irrep_key(key2)
    key3 = _normalize_o2_irrep_key(key3)

    d1 = _o2_irrep_dim(key1)
    d2 = _o2_irrep_dim(key2)
    d3 = _o2_irrep_dim(key3)

    if key1[0] == "scalar" and key2[0] == "scalar":
        if key3 == ("scalar", key1[1] * key2[1]):
            return torch.ones(d1, d2, d3, dtype=torch.float64)
        return torch.zeros(d1, d2, d3, dtype=torch.float64)

    if key1[0] == "scalar" and key2[0] == "freq":
        if key3 == key2:
            return torch.eye(2, dtype=torch.float64).unsqueeze(0)
        return torch.zeros(d1, d2, d3, dtype=torch.float64)

    if key1[0] == "freq" and key2[0] == "scalar":
        if key3 == key1:
            return torch.eye(2, dtype=torch.float64).unsqueeze(1)
        return torch.zeros(d1, d2, d3, dtype=torch.float64)

    if key1[0] == "freq" and key2[0] == "freq":
        m1 = key1[1]
        m2 = key2[1]
        if key3[0] == "freq":
            return build_cg_tensor_so2(m1, m2, key3[1])
        if key3 == ("scalar", 1) and m1 == m2:
            return build_cg_tensor_so2(m1, m2, 0)
        if key3 == ("scalar", -1) and m1 == m2:
            return _build_o2_scalar_odd_cg(m1)
        return torch.zeros(d1, d2, d3, dtype=torch.float64)

    return torch.zeros(d1, d2, d3, dtype=torch.float64)


def _o2_valid_output_keys(key1: O2IrrepKey, key2: O2IrrepKey) -> List[O2IrrepKey]:
    key1 = _normalize_o2_irrep_key(key1)
    key2 = _normalize_o2_irrep_key(key2)

    if key1[0] == "scalar" and key2[0] == "scalar":
        return [("scalar", key1[1] * key2[1])]

    if key1[0] == "scalar" and key2[0] == "freq":
        return [key2]
    if key1[0] == "freq" and key2[0] == "scalar":
        return [key1]

    m1 = key1[1]
    m2 = key2[1]
    outs: List[O2IrrepKey] = []
    if m1 == m2:
        outs.extend([("scalar", 1), ("scalar", -1)])
    else:
        outs.append(("freq", abs(m1 - m2)))
    outs.append(("freq", m1 + m2))
    dedup: List[O2IrrepKey] = []
    for key in outs:
        if key not in dedup:
            dedup.append(key)
    return dedup


class HarmonicElementwiseProductO2(nn.Module):
    def __init__(
        self,
        active_irreps: Sequence[str | O2IrrepKey],
        mul: int,
        irreps_out: str | None = "0e",
        normalization: str = "component",
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.active_irreps = parse_o2_active_irreps(active_irreps)
        self.active_set = set(self.active_irreps)
        self.mul = int(mul)
        self.internal_compute_dtype = _resolve_internal_compute_dtype(internal_compute_dtype)
        self._normalization = normalization
        self._irreps_out = irreps_out.strip().lower() if (irreps_out and isinstance(irreps_out, str)) else "full"
        self._output_0e_only = self._irreps_out == "0e"

        self.paths: List[Tuple[O2IrrepKey, O2IrrepKey]] = []
        self._cg_cpu_f64: List[torch.Tensor] = []
        for key in self.active_irreps:
            for out_key in _o2_valid_output_keys(key, key):
                if out_key not in self.active_set:
                    continue
                self.paths.append((key, out_key))
                C = _build_cg_tensor_o2(key, key, out_key)
                self._cg_cpu_f64.append(_normalize_cg_tensor(C, _o2_irrep_dim(out_key), normalization))
        self._cg_cache_by_dev_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}

    def _get_cg_list(self, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        key = (str(device), str(self.internal_compute_dtype))
        cached = self._cg_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        cg_list = [C.to(device=device, dtype=self.internal_compute_dtype) for C in self._cg_cpu_f64]
        self._cg_cache_by_dev_dtype[key] = cg_list
        return cg_list

    def forward(
        self,
        x1: Dict[O2IrrepKey, torch.Tensor],
        x2: Dict[O2IrrepKey, torch.Tensor],
    ) -> Dict[O2IrrepKey, torch.Tensor] | torch.Tensor:
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        cg_list = self._get_cg_list(device=device, dtype=dtype)

        if self._output_0e_only:
            outs = []
            for p_idx, (key_in, key_out) in enumerate(self.paths):
                if key_out != ("scalar", 1):
                    continue
                y = torch.einsum(
                    "...cm,...cn,mnk->...ck",
                    x1[key_in].to(dtype=self.internal_compute_dtype),
                    x2[key_in].to(dtype=self.internal_compute_dtype),
                    cg_list[p_idx],
                )
                outs.append(y[..., 0].to(dtype=dtype))
            return torch.cat(outs, dim=-1)

        out: Dict[O2IrrepKey, List[torch.Tensor]] = {}
        for p_idx, (key_in, key_out) in enumerate(self.paths):
            y = torch.einsum(
                "...cm,...cn,mnk->...ck",
                x1[key_in].to(dtype=self.internal_compute_dtype),
                x2[key_in].to(dtype=self.internal_compute_dtype),
                cg_list[p_idx],
            ).to(dtype=dtype)
            out.setdefault(key_out, []).append(y)
        return {key: torch.cat(parts, dim=-2) for key, parts in out.items()}


class HarmonicFullyConnectedTensorProductO2(nn.Module):
    def __init__(
        self,
        mul_in1: int,
        mul_in2: int,
        mul_out: int,
        active_irreps: Sequence[str | O2IrrepKey],
        internal_weights: bool = True,
        *,
        allowed_paths: List[Tuple[O2IrrepKey, O2IrrepKey, O2IrrepKey]] | None = None,
        path_policy: str = "full",
        max_rank_other: int | None = None,
        normalization: str = "component",
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.mul_in1 = int(mul_in1)
        self.mul_in2 = int(mul_in2)
        self.mul_out = int(mul_out)
        self.internal_weights = bool(internal_weights)
        self.active_irreps = parse_o2_active_irreps(active_irreps)
        self.active_set = set(self.active_irreps)
        self._normalization = normalization
        self.internal_compute_dtype = _resolve_internal_compute_dtype(internal_compute_dtype)

        all_paths: List[Tuple[O2IrrepKey, O2IrrepKey, O2IrrepKey]] = []
        for key1 in self.active_irreps:
            for key2 in self.active_irreps:
                for key3 in _o2_valid_output_keys(key1, key2):
                    if key3 in self.active_set:
                        all_paths.append((key1, key2, key3))

        if allowed_paths is not None:
            allowed_set = {
                (_normalize_o2_irrep_key(k1), _normalize_o2_irrep_key(k2), _normalize_o2_irrep_key(k3))
                for k1, k2, k3 in allowed_paths
            }
            self.paths = [p for p in all_paths if p in allowed_set]
        else:
            if path_policy == "full":
                self.paths = all_paths
            elif path_policy == "max_rank_other":
                if max_rank_other is None:
                    raise ValueError("path_policy='max_rank_other' requires max_rank_other")
                max_rank_other = int(max_rank_other)

                def _rank(key: O2IrrepKey) -> int:
                    return 0 if key[0] == "scalar" else key[1]

                self.paths = [p for p in all_paths if min(_rank(p[0]), _rank(p[1])) <= max_rank_other]
            else:
                raise ValueError(f"Unknown path_policy={path_policy!r}")

        self.num_paths = len(self.paths)
        self.weight_numel = self.num_paths * self.mul_out * self.mul_in1 * self.mul_in2
        if self.internal_weights:
            self.weight = nn.Parameter(torch.randn(self.num_paths, self.mul_out, self.mul_in1, self.mul_in2) * 0.02)
        else:
            self.register_parameter("weight", None)

        self._cg_cpu_f64: List[torch.Tensor] = []
        for key1, key2, key3 in self.paths:
            C = _build_cg_tensor_o2(key1, key2, key3)
            self._cg_cpu_f64.append(_normalize_cg_tensor(C, _o2_irrep_dim(key3), normalization))
        self._cg_cache_by_dev_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}

    def _get_cg_list(self, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        key = (str(device), str(self.internal_compute_dtype))
        cached = self._cg_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        cg_list = [C.to(device=device, dtype=self.internal_compute_dtype) for C in self._cg_cpu_f64]
        self._cg_cache_by_dev_dtype[key] = cg_list
        return cg_list

    def _path_weight(self, weights: torch.Tensor | None, p_idx: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.internal_weights:
            return self.weight[p_idx].to(device=device, dtype=dtype)
        if weights is None:
            raise ValueError("weights must be provided when internal_weights=False")
        expected_tail = (self.num_paths, self.mul_out, self.mul_in1, self.mul_in2)
        if tuple(weights.shape[-4:]) != expected_tail:
            raise ValueError(f"weights must end with {expected_tail}, got {tuple(weights.shape)}")
        return weights[..., p_idx, :, :, :].to(device=device, dtype=dtype)

    def forward(
        self,
        x1: Dict[O2IrrepKey, torch.Tensor],
        x2: Dict[O2IrrepKey, torch.Tensor],
        weights: torch.Tensor | None = None,
    ) -> Dict[O2IrrepKey, torch.Tensor]:
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        compute_dtype = self.internal_compute_dtype
        cg_list = self._get_cg_list(device=device, dtype=dtype)

        out = {
            key: torch.zeros(*batch_shape, self.mul_out, _o2_irrep_dim(key), device=device, dtype=dtype)
            for key in self.active_irreps
        }

        path_gates: torch.Tensor | None = None
        if self.internal_weights and weights is not None:
            if weights.shape[-1] != self.num_paths:
                raise ValueError(f"path gates must have last dim {self.num_paths}, got {weights.shape[-1]}")
            path_gates = weights.to(device=device, dtype=compute_dtype)

        for p_idx, (key1, key2, key3) in enumerate(self.paths):
            Wp = self._path_weight(weights if not self.internal_weights else None, p_idx, device, compute_dtype)
            a = x1[key1].to(dtype=compute_dtype)
            b = x2[key2].to(dtype=compute_dtype)
            C = cg_list[p_idx]
            if self.internal_weights:
                y = torch.einsum("...im,...jn,mnk,oij->...ok", a, b, C, Wp)
                if path_gates is not None:
                    gate = path_gates[..., p_idx].view(*batch_shape, 1, 1)
                    y = y * gate
            else:
                y = torch.einsum("...im,...jn,mnk,...oij->...ok", a, b, C, Wp)
            out[key3] = out[key3] + y.to(dtype=dtype)
        return out


class HarmonicPathWeightedTensorProductO2(nn.Module):
    def __init__(
        self,
        channels: int,
        active_irreps: Sequence[str | O2IrrepKey],
        *,
        allowed_paths: List[Tuple[O2IrrepKey, O2IrrepKey, O2IrrepKey]] | None = None,
        path_policy: str = "full",
        max_rank_other: int | None = None,
        normalization: str = "component",
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.active_irreps = parse_o2_active_irreps(active_irreps)
        self.active_set = set(self.active_irreps)
        self._normalization = normalization
        self.internal_compute_dtype = _resolve_internal_compute_dtype(internal_compute_dtype)

        all_paths: List[Tuple[O2IrrepKey, O2IrrepKey, O2IrrepKey]] = []
        for key1 in self.active_irreps:
            for key2 in self.active_irreps:
                for key3 in _o2_valid_output_keys(key1, key2):
                    if key3 in self.active_set:
                        all_paths.append((key1, key2, key3))

        if allowed_paths is not None:
            allowed_set = {
                (_normalize_o2_irrep_key(k1), _normalize_o2_irrep_key(k2), _normalize_o2_irrep_key(k3))
                for k1, k2, k3 in allowed_paths
            }
            self.paths = [p for p in all_paths if p in allowed_set]
        else:
            if path_policy == "full":
                self.paths = all_paths
            elif path_policy == "max_rank_other":
                if max_rank_other is None:
                    raise ValueError("path_policy='max_rank_other' requires max_rank_other")
                max_rank_other = int(max_rank_other)

                def _rank(key: O2IrrepKey) -> int:
                    return 0 if key[0] == "scalar" else key[1]

                self.paths = [p for p in all_paths if min(_rank(p[0]), _rank(p[1])) <= max_rank_other]
            else:
                raise ValueError(f"Unknown path_policy={path_policy!r}")

        self.num_paths = len(self.paths)
        self.weight = nn.Parameter(torch.randn(self.num_paths, self.channels) * 0.02)

        self._cg_cpu_f64: List[torch.Tensor] = []
        for key1, key2, key3 in self.paths:
            C = _build_cg_tensor_o2(key1, key2, key3)
            self._cg_cpu_f64.append(_normalize_cg_tensor(C, _o2_irrep_dim(key3), normalization))
        self._cg_cache_by_dev_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}

    def _get_cg_list(self, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        key = (str(device), str(self.internal_compute_dtype))
        cached = self._cg_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        cg_list = [C.to(device=device, dtype=self.internal_compute_dtype) for C in self._cg_cpu_f64]
        self._cg_cache_by_dev_dtype[key] = cg_list
        return cg_list

    def forward(
        self,
        x1: Dict[O2IrrepKey, torch.Tensor],
        x2: Dict[O2IrrepKey, torch.Tensor],
    ) -> Dict[O2IrrepKey, torch.Tensor]:
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        compute_dtype = self.internal_compute_dtype
        cg_list = self._get_cg_list(device=device, dtype=dtype)
        w = self.weight.to(device=device, dtype=compute_dtype)

        out = {
            key: torch.zeros(*batch_shape, self.channels, _o2_irrep_dim(key), device=device, dtype=dtype)
            for key in self.active_irreps
        }
        for p_idx, (key1, key2, key3) in enumerate(self.paths):
            y = torch.einsum(
                "...cm,...cn,mnk->...ck",
                x1[key1].to(dtype=compute_dtype),
                x2[key2].to(dtype=compute_dtype),
                cg_list[p_idx],
            )
            y = y * w[p_idx].view(*([1] * len(batch_shape)), self.channels, 1)
            out[key3] = out[key3] + y.to(dtype=dtype)
        return out


class MultipleContractionSO2(nn.Module):
    """
    Higher-order contraction block for the flattened SO(2) irreps layout.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        mmax: int,
        correlation: int = 3,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        normalization: str = "component",
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.mmax = int(mmax)
        self.correlation = int(correlation)
        if self.correlation < 1:
            raise ValueError(f"correlation must be >= 1, got {self.correlation}")

        self.reduce = EquivariantChannelLinearSO2Rect(
            self.in_channels,
            self.hidden_channels,
            self.mmax,
            bias=False,
        )
        self.order_mix = nn.ModuleList(
            [
                EquivariantChannelLinearSO2(self.hidden_channels, self.mmax, bias=False)
                for _ in range(self.correlation)
            ]
        )
        self.tp_layers = nn.ModuleList(
            [
                HarmonicPathWeightedTensorProductSO2(
                    channels=self.hidden_channels,
                    mmax=self.mmax,
                    path_policy=ictd_tp_path_policy,
                    max_rank_other=ictd_tp_max_rank_other,
                    normalization=normalization,
                    internal_compute_dtype=internal_compute_dtype,
                )
                for _ in range(max(self.correlation - 1, 0))
            ]
        )
        self.out_linear = EquivariantChannelLinearSO2(
            self.hidden_channels,
            self.mmax,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.reduce(x)
        accum = self.order_mix[0](base)
        if self.correlation == 1:
            return self.out_linear(accum)

        base_blocks = split_flat_irreps_so2(base, self.hidden_channels, self.mmax)
        current_blocks = base_blocks
        for order_idx, tp in enumerate(self.tp_layers, start=1):
            current_blocks = tp(current_blocks, base_blocks)
            current_flat = merge_flat_irreps_so2(current_blocks, self.hidden_channels, self.mmax)
            current_flat = self.order_mix[order_idx](current_flat)
            accum = accum + current_flat
        return self.out_linear(accum)


class MultipleContractionO2(nn.Module):
    """
    Higher-order contraction block for the flattened O(2) irreps layout.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        active_irreps: Sequence[str | O2IrrepKey],
        correlation: int = 3,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        normalization: str = "component",
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.active_irreps = parse_o2_active_irreps(active_irreps)
        self.correlation = int(correlation)
        if self.correlation < 1:
            raise ValueError(f"correlation must be >= 1, got {self.correlation}")

        self.reduce = EquivariantChannelLinearO2Rect(
            self.in_channels,
            self.hidden_channels,
            self.active_irreps,
            bias=False,
        )
        self.order_mix = nn.ModuleList(
            [
                EquivariantChannelLinearO2(self.hidden_channels, self.active_irreps, bias=False)
                for _ in range(self.correlation)
            ]
        )
        self.tp_layers = nn.ModuleList(
            [
                HarmonicPathWeightedTensorProductO2(
                    channels=self.hidden_channels,
                    active_irreps=self.active_irreps,
                    path_policy=ictd_tp_path_policy,
                    max_rank_other=ictd_tp_max_rank_other,
                    normalization=normalization,
                    internal_compute_dtype=internal_compute_dtype,
                )
                for _ in range(max(self.correlation - 1, 0))
            ]
        )
        self.out_linear = EquivariantChannelLinearO2(
            self.hidden_channels,
            self.active_irreps,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.reduce(x)
        accum = self.order_mix[0](base)
        if self.correlation == 1:
            return self.out_linear(accum)

        base_blocks = split_flat_irreps_o2(base, self.hidden_channels, self.active_irreps)
        current_blocks = base_blocks
        for order_idx, tp in enumerate(self.tp_layers, start=1):
            current_blocks = tp(current_blocks, base_blocks)
            current_flat = merge_flat_irreps_o2(current_blocks, self.hidden_channels, self.active_irreps)
            current_flat = self.order_mix[order_idx](current_flat)
            accum = accum + current_flat
        return self.out_linear(accum)
