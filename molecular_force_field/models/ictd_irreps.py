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
from molecular_force_field.models.ictd_irreps_cuda import (
    bucketed_tp_forward as _tp_cuda_ext_bucket_forward,
    ensure_grouped_tp_cuda_ext_supported,
    normalize_ictd_tp_backend,
)
from molecular_force_field.models import ictd_disk_cache

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
# Channelwise fused TP is experimentally useful only on some workloads.
_USE_TRITON_CHANNELWISE_TP = os.environ.get("ICTD_USE_TRITON_CHANNELWISE_TP", "0") == "1"
# Experimental pure PyTorch batching for channelwise per-path mixing by same-kdim buckets.
_USE_BUCKETED_CHANNELWISE_MIX = os.environ.get("ICTD_USE_BUCKETED_CHANNELWISE_MIX", "0") == "1"


def _resolve_internal_compute_dtype(internal_compute_dtype: torch.dtype | None) -> torch.dtype:
    return torch.get_default_dtype() if internal_compute_dtype is None else internal_compute_dtype


def _resolve_irrep_normalization(
    irrep_normalization: str | None,
    legacy_normalization: str | None,
) -> str:
    value = irrep_normalization if irrep_normalization is not None else legacy_normalization
    value = "component" if value is None else str(value)
    if value not in ("component", "norm", "none"):
        raise ValueError(
            "irrep_normalization/normalization must be one of "
            f"'component', 'norm', or 'none', got {value!r}"
        )
    return value


def _resolve_path_normalization(path_normalization: str | None) -> str:
    value = "element" if path_normalization is None else str(path_normalization)
    if value not in ("element", "path", "none"):
        raise ValueError(
            "path_normalization must be one of 'element', 'path', or 'none', "
            f"got {value!r}"
        )
    return value


def _path_normalization_scales(
    paths: List[Tuple[int, ...]],
    *,
    output_key_index: int | Tuple[int, ...],
    num_elements: float,
    path_normalization: str,
) -> List[float]:
    """e3nn-style path normalization, separate from irrep/CG normalization."""
    if path_normalization == "none":
        return [1.0 for _ in paths]
    if isinstance(output_key_index, int):
        def out_key(path: Tuple[int, ...]) -> Tuple[int, ...]:
            return (int(path[output_key_index]),)
    else:
        def out_key(path: Tuple[int, ...]) -> Tuple[int, ...]:
            return tuple(int(path[i]) for i in output_key_index)

    count_by_out: Dict[Tuple[int, ...], int] = {}
    element_sum_by_out: Dict[Tuple[int, ...], float] = {}
    for path in paths:
        key = out_key(path)
        count_by_out[key] = count_by_out.get(key, 0) + 1
        element_sum_by_out[key] = element_sum_by_out.get(key, 0.0) + float(num_elements)

    scales: List[float] = []
    for path in paths:
        key = out_key(path)
        if path_normalization == "element":
            denom = element_sum_by_out[key]
        else:
            denom = float(num_elements) * float(count_by_out[key])
        scales.append(1.0 / math.sqrt(denom) if denom > 1e-30 else 1.0)
    return scales


def _apply_cg_normalization(C: torch.Tensor, l_out: int, irrep_normalization: str) -> torch.Tensor:
    C_fn = C.norm().item()
    if irrep_normalization == "component" and C_fn > 1e-30:
        return C * (math.sqrt(2 * int(l_out) + 1) / C_fn)
    if irrep_normalization == "norm" and C_fn > 1e-30:
        return C * (1.0 / C_fn)
    return C


def split_flat_irreps_so3(x: torch.Tensor, channels: int, lmax: int) -> Dict[int, torch.Tensor]:
    """
    Split a flattened SO(3) irreps layout into per-l blocks.

    x: (..., channels * (lmax+1)^2)
    returns: dict l -> (..., channels, 2l+1)
    """
    out: Dict[int, torch.Tensor] = {}
    idx = 0
    for l in range(int(lmax) + 1):
        d = int(channels) * (2 * l + 1)
        blk = x[..., idx : idx + d]
        idx += d
        out[l] = blk.view(*x.shape[:-1], int(channels), 2 * l + 1)
    return out


def merge_flat_irreps_so3(blocks: Dict[int, torch.Tensor], channels: int, lmax: int) -> torch.Tensor:
    """
    Merge per-l SO(3) irreps blocks back into the flattened layout.
    """
    parts = []
    for l in range(int(lmax) + 1):
        parts.append(blocks[l].reshape(*blocks[l].shape[:-2], int(channels) * (2 * l + 1)))
    return torch.cat(parts, dim=-1)


def apply_channel_adapter_per_l(x_l: torch.Tensor, adapter: nn.Module) -> torch.Tensor:
    """
    Apply a channel adapter Linear/Identity to one irreps block.

    x_l: (..., Cin, 2l+1)
    adapter: maps Cin -> Cout (or Identity)
    returns: (..., Cout, 2l+1)
    """
    if isinstance(adapter, nn.Identity):
        return x_l
    y = adapter(x_l.movedim(-2, -1))
    return y.movedim(-1, -2)


class EquivariantChannelLinearSO3(nn.Module):
    """
    Block-diagonal equivariant linear map for a flattened SO(3) irreps layout.

    Each l-block keeps its (2l+1) angular components untouched and only mixes the
    channel/multiplicity dimension, so equivariance is preserved.
    """

    def __init__(self, channels: int, lmax: int, bias: bool = False):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.adapters = nn.ModuleDict(
            {str(l): nn.Linear(self.channels, self.channels, bias=bias) for l in range(self.lmax + 1)}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = split_flat_irreps_so3(x, self.channels, self.lmax)
        for l in range(self.lmax + 1):
            blocks[l] = apply_channel_adapter_per_l(blocks[l], self.adapters[str(l)])
        return merge_flat_irreps_so3(blocks, self.channels, self.lmax)


class EquivariantChannelLinearSO3Rect(nn.Module):
    """
    Block-diagonal equivariant linear map between two flattened SO(3) irreps
    layouts with the same lmax but different channel counts.
    """

    def __init__(self, in_channels: int, out_channels: int, lmax: int, bias: bool = False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.lmax = int(lmax)
        self.adapters = nn.ModuleDict(
            {str(l): nn.Linear(self.in_channels, self.out_channels, bias=bias) for l in range(self.lmax + 1)}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = split_flat_irreps_so3(x, self.in_channels, self.lmax)
        for l in range(self.lmax + 1):
            blocks[l] = apply_channel_adapter_per_l(blocks[l], self.adapters[str(l)])
        return merge_flat_irreps_so3(blocks, self.out_channels, self.lmax)


class MultipleContractionSO3(nn.Module):
    """
    Lightweight higher-order contraction block for the flattened SO(3) irreps
    layout used by ICTD-save-multiple.

    Pipeline:
      1) Equivariantly reduce concatenated features from in_channels to
         hidden_channels.
      2) Build higher-order product features by repeated path-weighted tensor
         products against the reduced base features.
      3) Mix the order-wise contributions and project back to the same hidden
         irreps layout.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        lmax: int,
        correlation: int = 3,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.lmax = int(lmax)
        self.correlation = int(correlation)
        if self.correlation < 1:
            raise ValueError(f"correlation must be >= 1, got {self.correlation}")

        self.reduce = EquivariantChannelLinearSO3Rect(
            self.in_channels, self.hidden_channels, self.lmax, bias=False
        )
        self.order_mix = nn.ModuleList(
            [
                EquivariantChannelLinearSO3(self.hidden_channels, self.lmax, bias=False)
                for _ in range(self.correlation)
            ]
        )
        self.tp_layers = nn.ModuleList(
            [
                HarmonicPathWeightedTensorProduct(
                    channels=self.hidden_channels,
                    lmax=self.lmax,
                    path_policy=ictd_tp_path_policy,
                    max_rank_other=ictd_tp_max_rank_other,
                    internal_compute_dtype=internal_compute_dtype,
                )
                for _ in range(max(self.correlation - 1, 0))
            ]
        )
        self.out_linear = EquivariantChannelLinearSO3(
            self.hidden_channels, self.lmax, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.reduce(x)
        accum = self.order_mix[0](base)
        if self.correlation == 1:
            return self.out_linear(accum)

        base_blocks = split_flat_irreps_so3(base, self.hidden_channels, self.lmax)
        current_blocks = base_blocks
        for order_idx, tp in enumerate(self.tp_layers, start=1):
            current_blocks = tp(current_blocks, base_blocks)
            current_flat = merge_flat_irreps_so3(current_blocks, self.hidden_channels, self.lmax)
            current_flat = self.order_mix[order_idx](current_flat)
            accum = accum + current_flat
        return self.out_linear(accum)


def split_flat_irreps_o3(
    x: torch.Tensor,
    channels: int,
    active_irreps: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Split a flattened O(3) irreps layout into per-(l, parity) blocks.
    """
    out: Dict[Tuple[int, int], torch.Tensor] = {}
    idx = 0
    for l, p in active_irreps:
        d = int(channels) * (2 * int(l) + 1)
        blk = x[..., idx : idx + d]
        idx += d
        out[_normalize_irrep_key(l, p)] = blk.view(*x.shape[:-1], int(channels), 2 * int(l) + 1)
    return out


def merge_flat_irreps_o3(
    blocks: Dict[Tuple[int, int], torch.Tensor],
    channels: int,
    active_irreps: List[Tuple[int, int]],
) -> torch.Tensor:
    """
    Merge per-(l, parity) O(3) irreps blocks back into the flattened layout.
    """
    parts = []
    for l, p in active_irreps:
        key = _normalize_irrep_key(l, p)
        parts.append(blocks[key].reshape(*blocks[key].shape[:-2], int(channels) * (2 * int(l) + 1)))
    return torch.cat(parts, dim=-1)


def apply_channel_adapter_per_irrep_o3(x_lp: torch.Tensor, adapter: nn.Module) -> torch.Tensor:
    """
    Apply a channel adapter Linear/Identity to one O(3) irrep block.

    x_lp: (..., Cin, 2l+1)
    adapter: maps Cin -> Cout (or Identity)
    returns: (..., Cout, 2l+1)
    """
    if isinstance(adapter, nn.Identity):
        return x_lp
    y = adapter(x_lp.movedim(-2, -1))
    return y.movedim(-1, -2)


class EquivariantChannelLinearO3(nn.Module):
    """
    Block-diagonal equivariant linear map for a flattened O(3) irreps layout.

    Each (l, parity) block keeps its angular components untouched and only mixes
    the channel dimension, so equivariance is preserved.
    """

    def __init__(self, channels: int, active_irreps: List[Tuple[int, int]], bias: bool = False):
        super().__init__()
        self.channels = int(channels)
        self.active_irreps = [_normalize_irrep_key(l, p) for l, p in active_irreps]
        self.adapters = nn.ModuleDict(
            {
                f"{l}{parity_sign_to_letter(p)}": nn.Linear(self.channels, self.channels, bias=bias)
                for l, p in self.active_irreps
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = split_flat_irreps_o3(x, self.channels, self.active_irreps)
        for key in self.active_irreps:
            blocks[key] = apply_channel_adapter_per_irrep_o3(
                blocks[key],
                self.adapters[f"{key[0]}{parity_sign_to_letter(key[1])}"],
            )
        return merge_flat_irreps_o3(blocks, self.channels, self.active_irreps)


class EquivariantChannelLinearO3Rect(nn.Module):
    """
    Block-diagonal equivariant linear map between two flattened O(3) irreps
    layouts with the same active irreps but different channel counts.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        active_irreps: List[Tuple[int, int]],
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.active_irreps = [_normalize_irrep_key(l, p) for l, p in active_irreps]
        self.adapters = nn.ModuleDict(
            {
                f"{l}{parity_sign_to_letter(p)}": nn.Linear(self.in_channels, self.out_channels, bias=bias)
                for l, p in self.active_irreps
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = split_flat_irreps_o3(x, self.in_channels, self.active_irreps)
        for key in self.active_irreps:
            blocks[key] = apply_channel_adapter_per_irrep_o3(
                blocks[key],
                self.adapters[f"{key[0]}{parity_sign_to_letter(key[1])}"],
            )
        return merge_flat_irreps_o3(blocks, self.out_channels, self.active_irreps)


def _segment_offsets_from_segments(segments: List[Tuple[int, ...]]) -> torch.Tensor:
    offsets = [int(seg[-2]) for seg in segments]
    offsets.append(int(segments[-1][-1]) if segments else 0)
    return torch.tensor(offsets, dtype=torch.long)


def _stack_group_weights(
    *,
    w_param: torch.Tensor,
    segments: List[Tuple[int, ...]],
    compute_dtype: torch.dtype,
    mul_scale: float = 1.0,
) -> torch.Tensor:
    path_indices = [int(seg[0]) for seg in segments]
    start = path_indices[0]
    if path_indices == list(range(start, start + len(path_indices))):
        stacked = w_param.narrow(0, start, len(path_indices))
    else:
        idx_t = torch.tensor(path_indices, device=w_param.device, dtype=torch.long)
        stacked = w_param.index_select(0, idx_t)
    if stacked.dtype != compute_dtype:
        stacked = stacked.to(dtype=compute_dtype)
    if abs(float(mul_scale) - 1.0) > 1e-12:
        stacked = stacked * float(mul_scale)
    return stacked.contiguous().view(stacked.shape[0], stacked.shape[1], -1)


def _slice_or_index_lastdim(t: torch.Tensor, path_indices: List[int]) -> torch.Tensor:
    start = int(path_indices[0])
    if path_indices == list(range(start, start + len(path_indices))):
        return t.narrow(-1, start, len(path_indices))
    idx_t = torch.tensor(path_indices, device=t.device, dtype=torch.long)
    return t.index_select(-1, idx_t)


def _build_kdim_buckets(
    *,
    segments: List[Tuple[int, ...]],
    U: torch.Tensor,
) -> List[Dict[str, object]]:
    buckets_by_kdim: Dict[int, List[Tuple[int, ...]]] = {}
    for seg in segments:
        start = int(seg[-2])
        end = int(seg[-1])
        buckets_by_kdim.setdefault(end - start, []).append(seg)

    buckets: List[Dict[str, object]] = []
    for kdim, bucket_segments in sorted(buckets_by_kdim.items()):
        starts = [int(seg[-2]) for seg in bucket_segments]
        path_indices = [int(seg[0]) for seg in bucket_segments]
        U_bucket = torch.cat([U[:, s : s + kdim] for s in starts], dim=1).contiguous()
        bucket: Dict[str, object] = {
            "kdim": kdim,
            "segments": bucket_segments,
            "path_indices": path_indices,
            "U_bucket": U_bucket,
        }
        buckets.append(bucket)
    return buckets


def _detect_scalar_identity_group_meta(
    U: torch.Tensor,
    *,
    l1: int,
    l2: int,
    segments: List[Tuple[int, ...]],
) -> Dict[str, object] | None:
    if l1 != 0 and l2 != 0:
        return None
    side = "rhs" if l2 == 0 else "lhs"
    ref_l = int(l1 if l2 == 0 else l2)
    m = 2 * ref_l + 1
    eye = torch.eye(m, device=U.device, dtype=U.dtype)
    if U.shape[0] != m:
        return None
    if U.dtype in (torch.float16, torch.bfloat16, torch.float32):
        atol = 1e-5
        rtol = 1e-4
    else:
        atol = 1e-8
        rtol = 1e-6
    meta_segments: List[Tuple[int, object, float]] = []
    for seg in segments:
        if len(seg) == 4:
            p_idx, l3, s, e = seg
            key_ir: object = int(l3)
        elif len(seg) == 5:
            p_idx, l3, p3, s, e = seg
            key_ir = (int(l3), int(p3))
        else:
            return None
        s_i = int(s)
        e_i = int(e)
        if int(l3) != ref_l or (e_i - s_i) != m:
            return None
        U_seg = U[:, s_i:e_i]
        alpha = torch.trace(U_seg) / float(m)
        if not torch.allclose(U_seg, alpha * eye, atol=atol, rtol=rtol):
            return None
        meta_segments.append((int(p_idx), key_ir, float(alpha.item())))
    return {
        "side": side,
        "segments": meta_segments,
    }


def _detect_scalar_output_split_meta(
    U: torch.Tensor,
    *,
    l1: int,
    l2: int,
    segments: List[Tuple[int, ...]],
) -> Dict[str, object] | None:
    if l1 != l2:
        return None
    m = 2 * l1 + 1
    if m <= 1:
        return None
    eye = torch.eye(m, device=U.device, dtype=U.dtype)
    if U.shape[0] != m * m:
        return None
    if U.dtype in (torch.float16, torch.bfloat16, torch.float32):
        atol = 1e-5
        rtol = 1e-4
    else:
        atol = 1e-8
        rtol = 1e-6
    scalar_entries: List[Tuple[int, object, float]] = []
    rem_segments: List[Tuple[int, object, int, int]] = []
    rem_cols: List[torch.Tensor] = []
    rem_start = 0
    for seg in segments:
        if len(seg) == 4:
            p_idx, l3, s, e = seg
            key_ir: object = int(l3)
        elif len(seg) == 5:
            p_idx, l3, p3, s, e = seg
            key_ir = (int(l3), int(p3))
        else:
            return None
        s_i = int(s)
        e_i = int(e)
        if int(l3) == 0 and (e_i - s_i) == 1:
            U_seg = U[:, s_i:e_i].reshape(m, m)
            alpha = torch.trace(U_seg) / float(m)
            if not torch.allclose(U_seg, alpha * eye, atol=atol, rtol=rtol):
                return None
            scalar_entries.append((int(p_idx), key_ir, float(alpha.item())))
        else:
            kdim = e_i - s_i
            rem_segments.append((int(p_idx), key_ir, rem_start, rem_start + kdim))
            rem_cols.append(U[:, s_i:e_i])
            rem_start += kdim
    if not scalar_entries:
        return None
    U_rem = torch.cat(rem_cols, dim=1).contiguous() if rem_cols else None
    return {
        "scalar_entries": scalar_entries,
        "rem_segments": rem_segments,
        "U_rem": U_rem,
    }


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


def _sphere_monomial_moment_3d(a: int, b: int, c: int) -> float:
    """Moment E[x^a y^b z^c] for a uniform unit vector on S^2."""
    a = int(a)
    b = int(b)
    c = int(c)
    if (a % 2) or (b % 2) or (c % 2):
        return 0.0
    p = a // 2
    q = b // 2
    r = c // 2
    return float(
        _double_factorial(2 * p - 1)
        * _double_factorial(2 * q - 1)
        * _double_factorial(2 * r - 1)
        / _double_factorial(2 * (p + q + r) + 1)
    )


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
    return c * _direction_component_scale_cpu_f64(l)


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


@lru_cache(maxsize=None)
def _direction_component_scale_cpu_f64(l: int) -> float:
    """
    Scale ICTD direction harmonics to e3nn's component normalization.

    The ICTD harmonic basis is orthonormal under isotropic Gaussian polynomial
    moments. MACE/e3nn edge spherical harmonics use component normalization,
    i.e. each component has RMS 1 over directions on the unit sphere.
    """
    l = int(l)
    if l == 0:
        return 1.0
    counts = _counts_list(l)
    coefs = [
        math.factorial(l) / (math.factorial(a) * math.factorial(b) * math.factorial(c))
        for a, b, c in counts
    ]
    d = len(counts)
    sphere_gram = torch.zeros(d, d, dtype=torch.float64)
    for i, (a, b, c) in enumerate(counts):
        for j, (a2, b2, c2) in enumerate(counts):
            sphere_gram[i, j] = (
                float(coefs[i])
                * float(coefs[j])
                * _sphere_monomial_moment_3d(a + a2, b + b2, c + c2)
            )
    P = _dir_proj_cpu_f64(l)
    mean_square = torch.trace(P.T @ sphere_gram @ P) / float(2 * l + 1)
    return float(mean_square.clamp_min(1e-30).rsqrt().item())


_dir_proj_cache_by_dev_dtype: Dict[Tuple[str, str, int], torch.Tensor] = {}
_dir_exps_cache_by_dev: Dict[Tuple[str, int], torch.Tensor] = {}
_dir_coefs_cache_by_dev_dtype: Dict[Tuple[str, str, int], torch.Tensor] = {}
_dir_component_scale_cache_by_dev_dtype: Dict[Tuple[str, str, int], torch.Tensor] = {}


def _integer_power_table(x: torch.Tensor, max_power: int) -> torch.Tensor:
    """
    Build [x^0, x^1, ..., x^max_power] using repeated multiplication.

    This avoids autograd's generic PowBackward path, which can produce NaNs for
    signed bases even when the exponents are mathematically integers.
    """
    if max_power < 0:
        raise ValueError(f"max_power must be >= 0, got {max_power}")
    powers = [torch.ones_like(x)]
    cur = torch.ones_like(x)
    for _ in range(int(max_power)):
        cur = cur * x
        powers.append(cur)
    return torch.stack(powers, dim=-1)

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

    exps_key = (str(n.device), int(l))
    exps = _dir_exps_cache_by_dev.get(exps_key)
    if exps is None:
        exps = _dir_monomial_exps_coefs(l)[0].to(device=n.device)
        _dir_exps_cache_by_dev[exps_key] = exps
    coefs_key = (str(n.device), str(n.dtype), int(l))
    coefs = _dir_coefs_cache_by_dev_dtype.get(coefs_key)
    if coefs is None:
        coefs = _dir_monomial_exps_coefs(l)[1].to(device=n.device, dtype=n.dtype)
        _dir_coefs_cache_by_dev_dtype[coefs_key] = coefs

    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    a = exps[:, 0]
    b = exps[:, 1]
    c = exps[:, 2]
    max_power = int(l)
    x_pows = _integer_power_table(nx, max_power)
    y_pows = _integer_power_table(ny, max_power)
    z_pows = _integer_power_table(nz, max_power)
    # (..., Dsym) with integer-power lookup instead of generic pow backward
    t = x_pows[..., a] * y_pows[..., b] * z_pows[..., c]
    t = t * coefs
    scale_key = (str(n.device), str(n.dtype), int(l))
    scale = _dir_component_scale_cache_by_dev_dtype.get(scale_key)
    if scale is None:
        scale = torch.tensor(
            _direction_component_scale_cpu_f64(l),
            device=n.device,
            dtype=n.dtype,
        )
        _dir_component_scale_cache_by_dev_dtype[scale_key] = scale
    # (..., 2l+1)
    return (t @ P) * scale


def parity_letter_to_sign(parity: str) -> int:
    p = str(parity).strip().lower()
    if p == "e":
        return 1
    if p == "o":
        return -1
    raise ValueError(f"parity must be 'e' or 'o', got {parity!r}")


def parity_sign_to_letter(parity: int) -> str:
    if int(parity) == 1:
        return "e"
    if int(parity) == -1:
        return "o"
    raise ValueError(f"parity sign must be +1 or -1, got {parity!r}")


def canonical_irrep_parity_sign(l: int) -> int:
    return 1 if (int(l) % 2 == 0) else -1


def parse_irreps_string(irreps: str) -> List[Tuple[int, int, int]]:
    """
    Parse e3nn-style irreps string into (mul, l, parity_sign) list.
    """
    out: List[Tuple[int, int, int]] = []
    for part in irreps.replace(",", " ").split("+"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d*)x?(\d+)(e|o)$", part.strip(), re.IGNORECASE)
        if not m:
            raise ValueError(f"Invalid irreps token: {part!r}")
        mul_s, l_s, p_s = m.groups()
        mul = int(mul_s) if mul_s else 1
        l_val = int(l_s)
        out.append((mul, l_val, parity_letter_to_sign(p_s)))
    return out


def parse_irreps_string_l_only(irreps: str) -> List[Tuple[int, int]]:
    return [(mul, l_val) for mul, l_val, _ in parse_irreps_string(irreps)]


def parse_irreps_to_l3_list(irreps: str, allowed_l3: Optional[List[int]] = None) -> List[int]:
    """
    Parse e3nn-style irreps string to an ordered list of l values (e.g. for output filtering).
    Examples: "0e + 2e" -> [0, 2]; "2e + 0e" -> [2, 0].
    If allowed_l3 is provided, only l that are in allowed_l3 are included (e.g. l⊗l gives only even l3).
    """
    parts = parse_irreps_string(irreps)
    out: List[int] = []
    seen: Set[int] = set()
    for _, l_val, _ in parts:
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
    for mul, l_val, parity in parts:
        canonical = canonical_irrep_parity_sign(l_val)
        if parity != canonical:
            raise ValueError(
                f"direction_harmonics_irreps only supports geometric parity {l_val}{parity_sign_to_letter(canonical)}; "
                f"got {l_val}{parity_sign_to_letter(parity)}"
            )
        h = direction_harmonics_fast(n, l_val)  # (..., 2l+1)
        chunks.append(h.unsqueeze(-2).expand(*h.shape[:-1], mul, 2 * l_val + 1).reshape(*h.shape[:-1], mul * (2 * l_val + 1)))
    return torch.cat(chunks, dim=-1)


def direction_harmonics_all(n: torch.Tensor, lmax: int) -> List[torch.Tensor]:
    """
    Compute direction harmonics for all l=0..lmax.
    Returns a list Y where Y[l] has shape (..., 2l+1).
    """
    lmax = int(lmax)
    if lmax < 0:
        raise ValueError(f"lmax must be >= 0, got {lmax}")
    out: List[torch.Tensor] = [
        torch.ones(*n.shape[:-1], 1, device=n.device, dtype=n.dtype)
    ]
    if lmax == 0:
        return out

    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    x_pows = _integer_power_table(nx, lmax)
    y_pows = _integer_power_table(ny, lmax)
    z_pows = _integer_power_table(nz, lmax)
    dev_key = str(n.device)
    dtype_key = str(n.dtype)

    for l in range(1, lmax + 1):
        proj_key = (dev_key, dtype_key, l)
        P = _dir_proj_cache_by_dev_dtype.get(proj_key)
        if P is None:
            P = _dir_proj_cpu_f64(l).to(device=n.device, dtype=n.dtype)
            _dir_proj_cache_by_dev_dtype[proj_key] = P

        exps_key = (dev_key, l)
        exps = _dir_exps_cache_by_dev.get(exps_key)
        if exps is None:
            exps = _dir_monomial_exps_coefs(l)[0].to(device=n.device)
            _dir_exps_cache_by_dev[exps_key] = exps

        coefs_key = (dev_key, dtype_key, l)
        coefs = _dir_coefs_cache_by_dev_dtype.get(coefs_key)
        if coefs is None:
            coefs = _dir_monomial_exps_coefs(l)[1].to(device=n.device, dtype=n.dtype)
            _dir_coefs_cache_by_dev_dtype[coefs_key] = coefs
        scale_key = (dev_key, dtype_key, l)
        scale = _dir_component_scale_cache_by_dev_dtype.get(scale_key)
        if scale is None:
            scale = torch.tensor(
                _direction_component_scale_cpu_f64(l),
                device=n.device,
                dtype=n.dtype,
            )
            _dir_component_scale_cache_by_dev_dtype[scale_key] = scale

        a = exps[:, 0]
        b = exps[:, 1]
        c = exps[:, 2]
        t = x_pows[..., a] * y_pows[..., b] * z_pows[..., c]
        t = t * coefs
        out.append((t @ P) * scale)
    return out


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
    # Parity-forbidden / out-of-range paths are exact zeros and trivially cheap;
    # compute inline rather than store them in the on-disk cache.
    if not (abs(l1 - l2) <= l3 <= l1 + l2) or ((l1 + l2 + l3) % 2 == 1):
        return torch.zeros(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1, dtype=torch.float64)
    # L2 on-disk cache (float64 canonical) sitting under the L1 lru_cache above.
    return ictd_disk_cache.load_or_compute(
        "cg", (int(l1), int(l2), int(l3)), lambda: _build_cg_tensor_compute(l1, l2, l3)
    )


def _build_cg_tensor_compute(l1: int, l2: int, l3: int) -> torch.Tensor:
    """Exact float64 computation behind build_cg_tensor (see its docstring)."""
    L = l1 + l2
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


@lru_cache(maxsize=None)
def _monomial_rotation_generator_cpu_f64(L: int, axis: str) -> torch.Tensor:
    """Infinitesimal SO(3) generator on degree-L monomial coefficients."""
    counts = _counts_list(int(L))
    idx = {t: i for i, t in enumerate(counts)}
    out = torch.zeros(len(counts), len(counts), dtype=torch.float64)
    for j, (a, b, c) in enumerate(counts):
        terms: List[Tuple[float, Tuple[int, int, int]]] = []
        if axis == "x":
            if c:
                terms.append((float(c), (a, b + 1, c - 1)))
            if b:
                terms.append((float(-b), (a, b - 1, c + 1)))
        elif axis == "y":
            if a:
                terms.append((float(a), (a - 1, b, c + 1)))
            if c:
                terms.append((float(-c), (a + 1, b, c - 1)))
        elif axis == "z":
            if b:
                terms.append((float(b), (a + 1, b - 1, c)))
            if a:
                terms.append((float(-a), (a - 1, b + 1, c)))
        else:
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
        for coef, key in terms:
            out[idx[key], j] += coef
    return out


@lru_cache(maxsize=None)
def _harmonic_rotation_generator_cpu_f64(l: int, axis: str) -> torch.Tensor:
    """Infinitesimal SO(3) generator in the ICTD harmonic basis."""
    B = _harmonic_basis_t(int(l), dtype=torch.float64)
    G = _gram_gaussian(int(l))
    return (B.T @ G @ _monomial_rotation_generator_cpu_f64(int(l), str(axis)) @ B).contiguous()


@lru_cache(maxsize=None)
def build_full_cg_tensor_so3(l1: int, l2: int, l3: int) -> torch.Tensor:
    """
    Full SO(3) Clebsch-Gordan tensor in the ICTD harmonic basis.

    `build_cg_tensor` comes from harmonic polynomial multiplication and therefore
    only covers the natural-parity/symmetric paths where l1+l2+l3 is even. MACE's
    higher-order contraction also uses intermediate irreps with independent O(3)
    parity, which requires the full SO(3) tensor product, including antisymmetric
    paths such as 1 x 1 -> 1. This routine constructs the unique intertwiner by
    solving the infinitesimal equivariance equations in the ICTD basis, without
    calling e3nn wigner/CG code.
    """
    l1 = int(l1)
    l2 = int(l2)
    l3 = int(l3)
    if not (abs(l1 - l2) <= l3 <= l1 + l2):
        return torch.zeros(2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1, dtype=torch.float64)
    return ictd_disk_cache.load_or_compute(
        "cg_full", (l1, l2, l3), lambda: _build_full_cg_tensor_so3_compute(l1, l2, l3)
    )


def _robust_svd_vh(A: torch.Tensor) -> torch.Tensor:
    """Vh of SVD(A), robust to LAPACK gesdd non-convergence.

    torch's default gesdd driver can raise "failed to converge ... ill-conditioned"
    on these equivariance-constraint matrices for larger l (observed on macOS
    Accelerate); fall back to numpy (LAPACK gesvd) -- slower but reliably convergent.
    Pure float64; the caller sign-fixes the null vector, so the result is canonical
    regardless of which backend produced it.
    """
    try:
        _, _, vh = torch.linalg.svd(A)
        return vh
    except Exception:
        import numpy as _np
        _, _, vh_np = _np.linalg.svd(A.detach().cpu().double().numpy(), full_matrices=True)
        return torch.from_numpy(vh_np).to(A)


def _build_full_cg_tensor_so3_compute(l1: int, l2: int, l3: int) -> torch.Tensor:
    """Exact float64 computation behind build_full_cg_tensor_so3 (see its docstring)."""
    d1, d2, d3 = 2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1
    n_unknown = d3 * d1 * d2
    equations: List[torch.Tensor] = []
    for axis in ("x", "y", "z"):
        J1 = _harmonic_rotation_generator_cpu_f64(l1, axis)
        J2 = _harmonic_rotation_generator_cpu_f64(l2, axis)
        J3 = _harmonic_rotation_generator_cpu_f64(l3, axis)
        Jin = torch.kron(J1, torch.eye(d2, dtype=torch.float64)) + torch.kron(
            torch.eye(d1, dtype=torch.float64), J2
        )
        A = torch.zeros(d3 * d1 * d2, n_unknown, dtype=torch.float64)
        col = 0
        for q in range(d3):
            for r in range(d1 * d2):
                Cmat = torch.zeros(d3, d1 * d2, dtype=torch.float64)
                Cmat[q, r] = 1.0
                A[:, col] = (J3 @ Cmat - Cmat @ Jin).reshape(-1)
                col += 1
        equations.append(A)

    A = torch.cat(equations, dim=0)
    vh = _robust_svd_vh(A)
    Cmat = vh[-1].reshape(d3, d1 * d2)
    C = Cmat.T.reshape(d1, d2, d3).contiguous()
    max_idx = int(C.abs().argmax().item())
    if C.flatten()[max_idx] < 0:
        C = -C
    return C.contiguous()


def cg_tensor_sparsity(C: torch.Tensor, threshold: float = 1e-10) -> Tuple[int, int, float]:
    """
    Return (numel, num_nonzero, zero_fraction) for an ICTD CG tensor.
    Many (l1,l2,l3) triples yield 60--85%% zeros (exact or |x|<=threshold); useful for sparse kernels.
    """
    n = C.numel()
    nz = (C.abs() > threshold).sum().item()
    return n, nz, 1.0 - (nz / n)


def _ictd_so3_coupled_basis(
    *,
    lmax: int,
    correlation: int,
    irrep_normalization: str = "component",
    dtype: torch.dtype | None = None,
) -> List[Tuple[int, int, torch.Tensor]]:
    """
    Build n-body SO(3) coupling basis tensors in the ICTD harmonic basis.

    Returns a list of `(l_out, parity, tensor)` where tensor has shape
    `(2*l_out+1, D, ..., D)` with `correlation` copies of the flattened
    one-copy SO(3) input dimension `D = sum_l (2*l+1)`.
    """
    lmax = int(lmax)
    correlation = int(correlation)
    dtype = torch.float64 if dtype is None else dtype
    if correlation < 1:
        raise ValueError(f"correlation must be >= 1, got {correlation}")
    dim = sum(2 * l + 1 for l in range(lmax + 1))
    starts: Dict[int, int] = {}
    offset = 0
    for l in range(lmax + 1):
        starts[l] = offset
        offset += 2 * l + 1

    basis: List[Tuple[int, int, torch.Tensor]] = []
    eye = torch.eye(dim, dtype=dtype)
    for l in range(lmax + 1):
        start = starts[l]
        stop = start + 2 * l + 1
        basis.append((l, canonical_irrep_parity_sign(l), eye[start:stop]))
    if correlation == 1:
        return basis

    for _order in range(2, correlation + 1):
        next_basis: List[Tuple[int, int, torch.Tensor]] = []
        for l_left, p_left, left_tensor in basis:
            left_tensor = left_tensor.to(dtype=dtype)
            for l_right in range(lmax + 1):
                right_start = starts[l_right]
                right_stop = right_start + 2 * l_right + 1
                for l_out in range(abs(l_left - l_right), l_left + l_right + 1):
                    p_out = int(p_left) * canonical_irrep_parity_sign(l_right)
                    cg = build_full_cg_tensor_so3(l_left, l_right, l_out)
                    cg = _apply_cg_normalization(cg, l_out, irrep_normalization).to(dtype=dtype)
                    flat_left = left_tensor.reshape(2 * l_left + 1, -1)
                    coupled = torch.einsum("ar,abq->qrb", flat_left, cg)
                    full = torch.zeros(
                        2 * l_out + 1,
                        flat_left.shape[1],
                        dim,
                        dtype=dtype,
                    )
                    full[:, :, right_start:right_stop] = coupled
                    full = full.reshape(2 * l_out + 1, *left_tensor.shape[1:], dim)
                    next_basis.append((l_out, p_out, full))
        basis = sorted(next_basis, key=lambda item: (item[0], item[1]))
    return basis


@lru_cache(maxsize=None)
def _ictd_u_matrix_so3_cached(
    lmax: int,
    output_l: int,
    correlation: int,
    irrep_normalization: str,
) -> torch.Tensor:
    """Float64 canonical U matrix: L1 lru_cache over the L2 on-disk cache."""
    return ictd_disk_cache.load_or_compute(
        "u_so3",
        (int(lmax), int(output_l), int(correlation), str(irrep_normalization)),
        lambda: _ictd_u_matrix_so3_compute(
            int(lmax), int(output_l), int(correlation), str(irrep_normalization)
        ),
    )


def _ictd_u_matrix_so3_compute(
    lmax: int,
    output_l: int,
    correlation: int,
    irrep_normalization: str,
) -> torch.Tensor:
    """Exact float64 build of the MACE-style symmetric-contraction U from ICTD CG."""
    dtype = torch.float64
    basis = _ictd_so3_coupled_basis(
        lmax=int(lmax),
        correlation=int(correlation),
        irrep_normalization=str(irrep_normalization),
        dtype=dtype,
    )
    target_parity = canonical_irrep_parity_sign(int(output_l))
    tensors = [
        tensor
        for l_out, parity, tensor in basis
        if int(l_out) == int(output_l) and int(parity) == int(target_parity)
    ]
    if not tensors:
        dim = sum(2 * l + 1 for l in range(int(lmax) + 1))
        if int(output_l) == 0:
            return torch.zeros(*([dim] * int(correlation)), 1, dtype=dtype)
        return torch.zeros(2 * int(output_l) + 1, *([dim] * int(correlation)), 1, dtype=dtype)
    if int(output_l) == 0:
        tensors = [tensor.squeeze(0) for tensor in tensors]
    return torch.stack(tensors, dim=-1).contiguous()


def ictd_u_matrix_so3(
    *,
    lmax: int,
    output_l: int,
    correlation: int,
    irrep_normalization: str = "component",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """MACE-style symmetric-contraction U matrix generated purely from ICTD CG tensors.

    Built and cached in float64 (the canonical high-precision value); the requested
    dtype is produced by a final cast. float64 -> bit-identical to the previous
    behaviour; float32 -> a downcast of the float64 build (>= as accurate as the old
    native-float32 path).
    """
    dtype = torch.get_default_dtype() if dtype is None else dtype
    u64 = _ictd_u_matrix_so3_cached(
        int(lmax),
        int(output_l),
        int(correlation),
        str(irrep_normalization),
    )
    return u64.to(dtype=dtype)


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
        irrep_normalization: str | None = None,
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.lmax = int(lmax)
        self.mul = int(mul)
        self.internal_compute_dtype = _resolve_internal_compute_dtype(internal_compute_dtype)
        self._normalization = _resolve_irrep_normalization(irrep_normalization, normalization)
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
            C = _apply_cg_normalization(C, l3, self._normalization)
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
        irrep_normalization: str | None = None,
        path_normalization: str = "element",
        # Internal computation dtype for CG tensors and projections (default: float64 for stability)
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
    ):
        super().__init__()
        self.mul_in1 = mul_in1
        self.mul_in2 = mul_in2
        self.mul_out = mul_out
        self.lmax = lmax
        self.internal_weights = internal_weights
        self._normalization = _resolve_irrep_normalization(irrep_normalization, normalization)
        self._path_normalization = _resolve_path_normalization(path_normalization)
        self._mul_path_scale = 1.0
        self.internal_compute_dtype = _resolve_internal_compute_dtype(internal_compute_dtype)
        self.ictd_tp_backend = normalize_ictd_tp_backend(ictd_tp_backend)

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
        self._path_normalization_scales = _path_normalization_scales(
            [tuple(p) for p in self.paths],
            output_key_index=2,
            num_elements=float(mul_in1 * mul_in2),
            path_normalization=self._path_normalization,
        )

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
        self._proj_group_view_cache_by_dev_dtype: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
        # Sparse U per group when zero_frac >= _SPARSE_MIN_ZERO_FRAC: list of None or (d_idx, k_idx, vals)
        self._proj_sparse_cache_by_dev_dtype: Dict[Tuple[str, str], List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]] = {}
        # Packed same-kdim buckets for the custom CUDA backend.
        self._proj_bucket_cache_by_dev_dtype: Dict[Tuple[str, str], List[List[Dict[str, object]]]] = {}
        self._proj_group_view_cache_by_dev_dtype: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
        self._scalar_identity_group_cache_by_dev_dtype: Dict[Tuple[str, str], List[Optional[Dict[str, object]]]] = {}
        self._scalar_split_group_cache_by_dev_dtype: Dict[Tuple[str, str], List[Optional[Dict[str, object]]]] = {}

    def _weight_to_compute(self, w: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        return w.to(dtype=compute_dtype) if w.dtype != compute_dtype else w


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
            for path_idx, (l1, l2, l3) in enumerate(self.paths):
                C = build_cg_tensor(l1, l2, l3)
                C = _apply_cg_normalization(C, l3, self._normalization)
                C = C * float(self._path_normalization_scales[path_idx])
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
    def _get_proj_bucket_list(self, device: torch.device, dtype: torch.dtype) -> List[List[Dict[str, object]]]:
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._proj_bucket_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        proj_list = self._get_proj_group_list(device=device, dtype=dtype)
        buckets = [
            _build_kdim_buckets(segments=g["segments"], U=proj_list[g_idx])  # type: ignore[arg-type]
            for g_idx, g in enumerate(self._groups)
        ]
        self._proj_bucket_cache_by_dev_dtype[key] = buckets
        return buckets

    @_dynamo_disable
    def _get_proj_group_view_list(self, device: torch.device, dtype: torch.dtype) -> List[Dict[str, object]]:
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._proj_group_view_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        proj_list = self._get_proj_group_list(device=device, dtype=dtype)
        views: List[Dict[str, object]] = []
        for g_idx, g in enumerate(self._groups):
            l1 = int(g["l1"])  # type: ignore[arg-type]
            l2 = int(g["l2"])  # type: ignore[arg-type]
            segments = g["segments"]  # type: ignore[assignment]
            U3 = proj_list[g_idx].reshape(2 * l1 + 1, 2 * l2 + 1, -1)
            seg_views = [(seg, U3[:, :, int(seg[-2]): int(seg[-1])]) for seg in segments]
            views.append(
                {
                    "U3": U3,
                    "seg_views": seg_views,
                }
            )
        self._proj_group_view_cache_by_dev_dtype[key] = views
        return views


    @_dynamo_disable
    def _get_scalar_identity_group_list(self, device: torch.device, dtype: torch.dtype) -> List[Optional[Dict[str, object]]]:
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._scalar_identity_group_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        proj_list = self._get_proj_group_list(device=device, dtype=dtype)
        metas = [
            _detect_scalar_identity_group_meta(
                proj_list[g_idx],
                l1=int(g["l1"]),
                l2=int(g["l2"]),
                segments=g["segments"],  # type: ignore[arg-type]
            )
            for g_idx, g in enumerate(self._groups)
        ]
        self._scalar_identity_group_cache_by_dev_dtype[key] = metas
        return metas

    @_dynamo_disable
    def _get_scalar_split_group_list(self, device: torch.device, dtype: torch.dtype) -> List[Optional[Dict[str, object]]]:
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._scalar_split_group_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        proj_list = self._get_proj_group_list(device=device, dtype=dtype)
        metas = [
            _detect_scalar_output_split_meta(
                proj_list[g_idx],
                l1=int(g["l1"]),
                l2=int(g["l2"]),
                segments=g["segments"],  # type: ignore[arg-type]
            )
            for g_idx, g in enumerate(self._groups)
        ]
        self._scalar_split_group_cache_by_dev_dtype[key] = metas
        return metas

    @_dynamo_disable
    def prewarm_caches(self, device: torch.device, dtype: torch.dtype) -> None:
        """Pre-build internal caches on (device, dtype).

        This keeps one-time Python-side work (and `.item()` calls) out of torch.compile tracing.
        Safe to call multiple times.
        """
        _ = self._get_cg_list(device=device, dtype=dtype)
        _ = self._get_proj_group_list(device=device, dtype=dtype)
        _ = self._get_proj_sparse_list(device=device, dtype=dtype)
        _ = self._get_proj_bucket_list(device=device, dtype=dtype)

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
            bucket_list = self._get_proj_bucket_list(device=device, dtype=dtype)
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
                num_paths_in_group = len(segments)
                # Convert inputs to compute_dtype for numerical stability
                a_comp = a.to(dtype=compute_dtype) if a.dtype != compute_dtype else a
                b_comp = b.to(dtype=compute_dtype) if b.dtype != compute_dtype else b
                B_flat = 1
                for s in batch_shape:
                    B_flat *= int(s)
                a_flat = a_comp.reshape(B_flat, self.mul_in1, m1)
                b_flat = b_comp.reshape(B_flat, self.mul_in2, m2)
                if a_comp.dim() >= 2:
                    bucket_outputs: list[tuple[int, torch.Tensor]] = []
                    cuda_supported = True
                    for bucket in bucket_list[g_idx]:
                        bucket_segments = bucket["segments"]  # type: ignore[assignment]
                        bucket_path_indices = bucket["path_indices"]  # type: ignore[assignment]
                        U_bucket = bucket["U_bucket"]  # type: ignore[assignment]
                        kdim = int(bucket["kdim"])  # type: ignore[arg-type]
                        W_bucket = _stack_group_weights(
                            w_param=w_param,
                            segments=bucket_segments,
                            compute_dtype=compute_dtype,
                            mul_scale=self._mul_path_scale,
                        ).to(device=a_comp.device)
                        gates_bucket = None
                        if weights is not None:
                            gates_bucket = _slice_or_index_lastdim(weights, [int(p_idx) for p_idx in bucket_path_indices])
                            gates_bucket = gates_bucket.reshape(B_flat, len(bucket_path_indices)).to(dtype=compute_dtype)
                        bucket_out = _tp_cuda_ext_bucket_forward(
                            backend=self.ictd_tp_backend,
                            a=a_flat,
                            b=b_flat,
                            U_bucket=U_bucket,
                            W_stack=W_bucket,
                            gates=gates_bucket,
                            compute_dtype=compute_dtype,
                        )
                        if bucket_out is None:
                            cuda_supported = False
                            break
                        bucket_out = bucket_out.reshape(*batch_shape, self.mul_out, kdim)
                        bucket_outputs.append((int(bucket_segments[0][1]), bucket_out))
                    if cuda_supported and bucket_outputs:
                        for l3, bucket_out in bucket_outputs:
                            bucket_out = bucket_out.to(dtype=dtype) if bucket_out.dtype != dtype else bucket_out
                            out[l3] = out[l3] + bucket_out
                        continue
                # FlashTP-style: fused projection+channel-mix (one kernel), or sparse/dense TP then per-path mix
                y = None
                used_fused_mix = False
                if a_comp.is_cuda and a_comp.dim() >= 2:
                    # Try fused outer-product + projection + channel mixing (one kernel, no y write-back)
                    if (
                        _tp_fused_outer_proj_channel_mix is not None
                        and num_paths_in_group <= 16
                    ):
                        W_stack = _stack_group_weights(
                            w_param=w_param,
                            segments=segments,
                            compute_dtype=compute_dtype,
                            mul_scale=self._mul_path_scale,
                        ).to(device=a_comp.device)
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
                        # PyTorch fallback: outer product + matmul projection
                        U = proj_list[g_idx]  # (m1*m2, K_total)
                        t_mn = (a_comp.unsqueeze(-2).unsqueeze(-1) * b_comp.unsqueeze(-3).unsqueeze(-2))
                        t_flat = t_mn.reshape(*batch_shape, self.mul_in1, self.mul_in2, m1 * m2)
                        if not t_flat.is_contiguous():
                            t_flat = t_flat.contiguous()
                        y = torch.matmul(t_flat, U)

                    # Per-path channel mixing
                    i, j = self.mul_in1, self.mul_in2
                    ij = i * j
                    for p_idx, l3, s, e in segments:  # type: ignore[misc]
                        Wp = w_param[int(p_idx)]  # (o,i,j)
                        Wp_comp = self._weight_to_compute(Wp, compute_dtype)
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
                    U = proj_list[g_idx]  # (m1*m2, K_total)
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
                
                # Batched channel mixing over the flattened (i,j) dimension.
                #   y_reshaped: (..., ij, K)
                #   W_reshaped: (..., P_g, o, ij)
                # We need sum_{ij} W[p,o,ij] * y[ij,k] -> (..., P_g, o, K).
                # Arrange the inputs as (..., 1, K, ij) @ (..., P_g, ij, o) -> (..., P_g, K, o),
                # then move the last two dims back to (..., P_g, o, K).
                y_expanded = y_reshaped.transpose(-2, -1).unsqueeze(-3)  # (..., 1, K, ij)
                W_transposed = W_reshaped.transpose(-2, -1)  # (..., P_g, ij, o)
                out_group = torch.matmul(y_expanded, W_transposed)  # (..., P_g, K, o) in compute_dtype
                out_group = out_group.movedim(-1, -2)  # (..., P_g, o, K_total)
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


class HarmonicPathWeightedTensorProduct(nn.Module):
    """
    Lightweight path-weighted tensor product in the ICTD SO(3) basis.

    Compared with HarmonicFullyConnectedTensorProduct, this module does not learn
    a full `(mul_out, mul_in1, mul_in2)` kernel per path. Instead, it:

    - assumes aligned input/output channels (`mul_in = mul_out = channels`)
    - reuses the CG / projection cache already available in ICTD irreps
    - learns only one scalar weight per `(path, channel)`

    This matches the "few path weights + fixed coupling basis" idea used by
    MACE-style symmetric contractions while staying fully within the ICTD
    operator family.
    """

    def __init__(
        self,
        channels: int,
        lmax: int,
        *,
        allowed_paths: List[Tuple[int, int, int]] | None = None,
        path_policy: str = "full",
        max_rank_other: int | None = None,
        normalization: str = "component",
        irrep_normalization: str | None = None,
        path_normalization: str = "element",
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self._normalization = _resolve_irrep_normalization(irrep_normalization, normalization)
        self._path_normalization = _resolve_path_normalization(path_normalization)
        self.internal_compute_dtype = _resolve_internal_compute_dtype(internal_compute_dtype)

        all_paths: List[Tuple[int, int, int]] = []
        for l1 in range(self.lmax + 1):
            for l2 in range(self.lmax + 1):
                for l3 in range(abs(l1 - l2), min(l1 + l2, self.lmax) + 1):
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
        self._path_normalization_scales = _path_normalization_scales(
            [tuple(p) for p in self.paths],
            output_key_index=2,
            num_elements=1.0,
            path_normalization=self._path_normalization,
        )
        self.weight = nn.Parameter(torch.randn(self.num_paths, self.channels) * 0.02)

        self._cg_cpu_f64: List[torch.Tensor] | None = None
        self._cg_cache_by_dev_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}

        self._groups: List[Dict[str, object]] = []
        groups_tmp: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for p_idx, (l1, l2, l3) in enumerate(self.paths):
            groups_tmp.setdefault((l1, l2), []).append((p_idx, l3))
        for (l1, l2), items in sorted(groups_tmp.items()):
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
                    "segments": segments,
                    "k_total": start,
                }
            )

        self._proj_group_cache_by_dev_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}
        self._proj_group_view_cache_by_dev_dtype: Dict[Tuple[str, str], List[Dict[str, object]]] = {}

    @_dynamo_disable
    def _get_cg_list(self, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._cg_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached

        if self._cg_cpu_f64 is None:
            self._cg_cpu_f64 = []
            for path_idx, (l1, l2, l3) in enumerate(self.paths):
                C = build_cg_tensor(l1, l2, l3)
                C = _apply_cg_normalization(C, l3, self._normalization)
                C = C * float(self._path_normalization_scales[path_idx])
                self._cg_cpu_f64.append(C)

        cg_list = [C.to(device=device, dtype=compute_dtype) for C in self._cg_cpu_f64]
        self._cg_cache_by_dev_dtype[key] = cg_list
        return cg_list

    @_dynamo_disable
    def _get_proj_group_list(self, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._proj_group_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached

        cg_list = self._get_cg_list(device=device, dtype=dtype)
        proj_list: List[torch.Tensor] = []
        for g in self._groups:
            l1 = int(g["l1"])
            l2 = int(g["l2"])
            segments = g["segments"]
            k_total = int(g["k_total"])
            m1 = 2 * l1 + 1
            m2 = 2 * l2 + 1
            U = torch.zeros(m1 * m2, k_total, device=device, dtype=compute_dtype)
            for p_idx, _l3, s, e in segments:
                C = cg_list[int(p_idx)]
                U[:, int(s): int(e)] = C.reshape(m1 * m2, int(e) - int(s))
            proj_list.append(U)

        self._proj_group_cache_by_dev_dtype[key] = proj_list
        return proj_list

    def forward(
        self,
        x1: Dict[int, torch.Tensor],
        x2: Dict[int, torch.Tensor],
        path_channel_weights: torch.Tensor | None = None,
    ) -> Dict[int, torch.Tensor]:
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        compute_dtype = self.internal_compute_dtype

        out: Dict[int, torch.Tensor] = {
            l: torch.zeros(*batch_shape, self.channels, 2 * l + 1, device=device, dtype=dtype)
            for l in range(self.lmax + 1)
        }

        proj_list = self._get_proj_group_list(device=device, dtype=dtype)
        w = self.weight.to(device=device, dtype=compute_dtype)
        if path_channel_weights is not None:
            path_channel_weights = path_channel_weights.to(device=device, dtype=compute_dtype)

        for g_idx, g in enumerate(self._groups):
            l1 = int(g["l1"])
            l2 = int(g["l2"])
            segments = g["segments"]

            a = x1.get(l1)
            b = x2.get(l2)
            if a is None or b is None:
                continue

            a_comp = a.to(dtype=compute_dtype) if a.dtype != compute_dtype else a
            b_comp = b.to(dtype=compute_dtype) if b.dtype != compute_dtype else b
            m1 = 2 * l1 + 1
            m2 = 2 * l2 + 1
            U = proj_list[g_idx]

            pair = (
                a_comp.unsqueeze(-1) * b_comp.unsqueeze(-2)
            ).reshape(*batch_shape, self.channels, m1 * m2)
            y = torch.matmul(pair, U)
            for p_idx, l3, s, e in segments:
                seg = y[..., int(s): int(e)]
                weight = w[int(p_idx)].view(*([1] * len(batch_shape)), self.channels, 1)
                if path_channel_weights is not None:
                    weight = weight * path_channel_weights[..., int(p_idx), :].unsqueeze(-1)
                seg = seg * weight
                seg = seg.to(dtype=dtype) if seg.dtype != dtype else seg
                out[int(l3)] = out[int(l3)] + seg

        return out


class EdgeWeightedPathTensorProduct(HarmonicPathWeightedTensorProduct):
    """
    Thin message-passing tensor product with edge-dependent path gates.

    This keeps the ICTD SO(3) path basis and path-weighted channel scaling from
    `HarmonicPathWeightedTensorProduct`, while additionally accepting a
    per-edge/per-sample gate tensor of shape `(..., num_paths)`.
    """

    def forward(
        self,
        x1: Dict[int, torch.Tensor],
        x2: Dict[int, torch.Tensor],
        gates: torch.Tensor | None = None,
    ) -> Dict[int, torch.Tensor]:
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        compute_dtype = self.internal_compute_dtype

        if gates is not None and (gates.device != device or gates.dtype != dtype):
            gates = gates.to(device=device, dtype=dtype)

        out: Dict[int, torch.Tensor] = {
            l: torch.zeros(*batch_shape, self.channels, 2 * l + 1, device=device, dtype=dtype)
            for l in range(self.lmax + 1)
        }

        proj_list = self._get_proj_group_list(device=device, dtype=dtype)
        w = self.weight.to(device=device, dtype=compute_dtype)

        for g_idx, g in enumerate(self._groups):
            l1 = int(g["l1"])
            l2 = int(g["l2"])
            segments = g["segments"]

            a = x1.get(l1)
            b = x2.get(l2)
            if a is None or b is None:
                continue

            a_comp = a.to(dtype=compute_dtype) if a.dtype != compute_dtype else a
            b_comp = b.to(dtype=compute_dtype) if b.dtype != compute_dtype else b
            m1 = 2 * l1 + 1
            m2 = 2 * l2 + 1
            U = proj_list[g_idx]

            pair = (
                a_comp.unsqueeze(-1) * b_comp.unsqueeze(-2)
            ).reshape(*batch_shape, self.channels, m1 * m2)
            y = torch.matmul(pair, U)
            for p_idx, l3, s, e in segments:
                seg = y[..., int(s): int(e)]
                seg = seg * w[int(p_idx)].view(*([1] * len(batch_shape)), self.channels, 1)
                seg = seg.to(dtype=dtype) if seg.dtype != dtype else seg
                if gates is not None:
                    seg = seg * gates[..., int(p_idx)].unsqueeze(-1).unsqueeze(-1)
                out[int(l3)] = out[int(l3)] + seg

        return out


class EdgeWeightedPathPreservingTensorProduct(EdgeWeightedPathTensorProduct):
    """
    Path-preserving ICTD SO(3) tensor product for MACE-style interactions.

    The standard EdgeWeightedPathTensorProduct sums all paths that land in the
    same output l. MACE/e3nn keeps each path as a separate multiplicity block
    and only mixes them in the following equivariant linear. This class preserves
    that layout while using the same ICTD CG/projector tensors as the compressed
    variant.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path_l3 = [0 for _ in range(self.num_paths)]
        path_offset = [0 for _ in range(self.num_paths)]
        counts = {l: 0 for l in range(self.lmax + 1)}
        for group in self._groups:
            for p_idx, l3, _s, _e in group["segments"]:
                l3 = int(l3)
                p_idx = int(p_idx)
                path_l3[p_idx] = l3
                path_offset[p_idx] = counts[l3]
                counts[l3] += 1
        self.path_l3 = path_l3
        self.path_offset = path_offset
        self.path_counts_by_l = counts

    def forward(
        self,
        x1: Dict[int, torch.Tensor],
        x2: Dict[int, torch.Tensor],
        gates: torch.Tensor | None = None,
    ) -> Dict[int, torch.Tensor]:
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        compute_dtype = self.internal_compute_dtype

        if gates is not None:
            if gates.device != device or gates.dtype != dtype:
                gates = gates.to(device=device, dtype=dtype)
            if gates.shape[-1] == self.num_paths * self.channels:
                gates = gates.view(*gates.shape[:-1], self.num_paths, self.channels)
            elif gates.shape[-2:] != (self.num_paths, self.channels):
                raise ValueError(
                    f"Expected gates shape (..., {self.num_paths * self.channels}) or "
                    f"(..., {self.num_paths}, {self.channels}), got {tuple(gates.shape)}"
                )

        out: Dict[int, torch.Tensor] = {}
        for l in range(self.lmax + 1):
            out_channels_l = self.channels * int(self.path_counts_by_l.get(l, 0))
            out[l] = torch.zeros(*batch_shape, out_channels_l, 2 * l + 1, device=device, dtype=dtype)

        proj_list = self._get_proj_group_list(device=device, dtype=dtype)
        w = self.weight.to(device=device, dtype=compute_dtype)

        for g_idx, group in enumerate(self._groups):
            l1 = int(group["l1"])
            l2 = int(group["l2"])
            segments = group["segments"]

            a = x1.get(l1)
            b = x2.get(l2)
            if a is None or b is None:
                continue

            a_comp = a.to(dtype=compute_dtype) if a.dtype != compute_dtype else a
            b_comp = b.to(dtype=compute_dtype) if b.dtype != compute_dtype else b
            m1 = 2 * l1 + 1
            m2 = 2 * l2 + 1
            U = proj_list[g_idx]

            pair = (a_comp.unsqueeze(-1) * b_comp.unsqueeze(-2)).reshape(
                *batch_shape, self.channels, m1 * m2
            )
            y = torch.matmul(pair, U)
            for p_idx, l3, s, e in segments:
                p_idx = int(p_idx)
                l3 = int(l3)
                seg = y[..., int(s) : int(e)]
                seg = seg * w[p_idx].view(*([1] * len(batch_shape)), self.channels, 1)
                if gates is not None:
                    seg = seg * gates[..., p_idx, :].unsqueeze(-1)
                seg = seg.to(dtype=dtype) if seg.dtype != dtype else seg
                c0 = int(self.path_offset[p_idx]) * self.channels
                c1 = c0 + self.channels
                out[l3][..., c0:c1, :] = out[l3][..., c0:c1, :] + seg

        return out


def _normalize_irrep_key(l: int, parity: int) -> Tuple[int, int]:
    return (int(l), 1 if int(parity) >= 0 else -1)


def _is_canonical_irrep_list(active_irreps: List[Tuple[int, int]], *, lmax: int | None = None) -> bool:
    keys = [_normalize_irrep_key(l, p) for l, p in active_irreps]
    if lmax is None:
        lmax = max((int(l) for l, _ in keys), default=-1)
    expected = [(int(l), canonical_irrep_parity_sign(int(l))) for l in range(int(lmax) + 1)]
    return keys == expected


def o3_irrep_keys(lmax: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for l in range(int(lmax) + 1):
        out.append((l, 1))
        out.append((l, -1))
    return out


class HarmonicElementwiseProductO3(nn.Module):
    """
    O(3) version of HarmonicElementwiseProduct with parity-aware keys.

    Inputs are dict[(l, p)] -> (..., mul, 2l+1). For irreps_out="0e", only
    same-(l,p) self-pair invariants are produced.
    """

    def __init__(
        self,
        active_irreps: List[Tuple[int, int]],
        mul: int,
        irreps_out: str | None = "0e",
        normalization: str = "component",
        irrep_normalization: str | None = None,
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.active_irreps = [_normalize_irrep_key(l, p) for l, p in active_irreps]
        self.mul = int(mul)
        self.internal_compute_dtype = _resolve_internal_compute_dtype(internal_compute_dtype)
        self._normalization = _resolve_irrep_normalization(irrep_normalization, normalization)
        self._irreps_out = irreps_out.strip().lower() if (irreps_out and isinstance(irreps_out, str)) else "full"
        self._output_0e_only = self._irreps_out == "0e"

        self._paths: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for key in self.active_irreps:
            l, p = key
            if p * p != 1:
                continue
            self._paths.append((key, (0, 1)))

        self._0e_factors: Dict[Tuple[int, int], float] = {}
        for l, p in self.active_irreps:
            if (l, p) not in self._0e_factors:
                C = build_cg_tensor(l, l, 0)
                C = _apply_cg_normalization(C, 0, self._normalization)
                self._0e_factors[(l, p)] = float(C[0, 0, 0].item())
    def forward(
        self,
        x1: Dict[Tuple[int, int], torch.Tensor],
        x2: Dict[Tuple[int, int], torch.Tensor],
    ) -> torch.Tensor:
        if not self._output_0e_only:
            raise NotImplementedError("HarmonicElementwiseProductO3 currently supports irreps_out='0e' only")
        sample = next(iter(x1.values()))
        outs = []
        for key in self.active_irreps:
            a = x1[key]
            b = x2[key]
            outs.append((a * b).sum(dim=-1) * self._0e_factors[key])
        return torch.cat(outs, dim=-1)


class HarmonicFullyConnectedTensorProductO3(nn.Module):
    """
    Parity-aware O(3) fully-connected tensor product in ICTD basis.

    Inputs / outputs are keyed by (l, parity_sign), where parity_sign is +1/-1.
    CG tensors are reused from the SO(3) ICTD basis; parity only filters valid paths.
    """

    def __init__(
        self,
        mul_in1: int,
        mul_in2: int,
        mul_out: int,
        lmax: int,
        active_irreps: List[Tuple[int, int]] | None = None,
        internal_weights: bool = True,
        *,
        allowed_paths: List[Tuple[int, int, int, int, int, int]] | None = None,
        path_policy: str = "full",
        max_rank_other: int | None = None,
        normalization: str = "component",
        irrep_normalization: str | None = None,
        path_normalization: str = "element",
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
    ):
        super().__init__()
        self.mul_in1 = int(mul_in1)
        self.mul_in2 = int(mul_in2)
        self.mul_out = int(mul_out)
        self.lmax = int(lmax)
        self.internal_weights = bool(internal_weights)
        self._normalization = _resolve_irrep_normalization(irrep_normalization, normalization)
        self._path_normalization = _resolve_path_normalization(path_normalization)
        # Keep O(3) on the same effective weight parameterization as the SO(3)
        # save path for an apples-to-apples training comparison.
        self._mul_path_scale = 1.0
        self.internal_compute_dtype = _resolve_internal_compute_dtype(internal_compute_dtype)
        self.ictd_tp_backend = normalize_ictd_tp_backend(ictd_tp_backend)
        self.active_irreps = (
            [_normalize_irrep_key(l, p) for l, p in active_irreps]
            if active_irreps is not None
            else o3_irrep_keys(self.lmax)
        )
        active_set = set(self.active_irreps)

        all_paths: List[Tuple[int, int, int, int, int, int]] = []
        for l1, p1 in self.active_irreps:
            for l2, p2 in self.active_irreps:
                for l3 in range(abs(l1 - l2), min(l1 + l2, self.lmax) + 1):
                    if (l1 + l2 + l3) % 2 == 1:
                        continue
                    p3 = p1 * p2
                    if (l3, p3) not in active_set:
                        continue
                    all_paths.append((l1, p1, l2, p2, l3, p3))

        if allowed_paths is not None:
            allowed_set = {tuple(map(int, p)) for p in allowed_paths}
            self.paths = [p for p in all_paths if p in allowed_set]
        else:
            if path_policy == "full":
                self.paths = all_paths
            elif path_policy == "max_rank_other":
                if max_rank_other is None:
                    raise ValueError("path_policy='max_rank_other' requires max_rank_other")
                self.paths = [p for p in all_paths if min(p[0], p[2]) <= int(max_rank_other)]
            else:
                raise ValueError(f"Unknown path_policy={path_policy!r}")

        self.num_paths = len(self.paths)
        self.weight_numel = self.num_paths * self.mul_out * self.mul_in1 * self.mul_in2
        self._path_normalization_scales = _path_normalization_scales(
            [tuple(p) for p in self.paths],
            output_key_index=(4, 5),
            num_elements=float(self.mul_in1 * self.mul_in2),
            path_normalization=self._path_normalization,
        )
        if self.internal_weights:
            self.weight = nn.Parameter(torch.randn(self.num_paths, self.mul_out, self.mul_in1, self.mul_in2) * 0.02)
        else:
            self.register_parameter("weight", None)

        self._cg_cpu_f64: List[torch.Tensor] | None = None
        self._cg_cache_by_dev_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}
        self._groups: List[Dict[str, object]] = []
        groups_tmp: Dict[Tuple[int, int, int, int], List[Tuple[int, int, int]]] = {}
        for p_idx, (l1, p1, l2, p2, l3, p3) in enumerate(self.paths):
            groups_tmp.setdefault((l1, p1, l2, p2), []).append((p_idx, l3, p3))
        for (l1, p1, l2, p2), items in sorted(groups_tmp.items()):
            segments = []
            start = 0
            for p_idx, l3, p3 in items:
                kdim = 2 * l3 + 1
                segments.append((p_idx, l3, p3, start, start + kdim))
                start += kdim
            self._groups.append(
                {
                    "l1": l1,
                    "p1": p1,
                    "l2": l2,
                    "p2": p2,
                    "segments": segments,
                    "k_total": start,
                }
            )
        self._proj_group_cache_by_dev_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}
        self._proj_sparse_cache_by_dev_dtype: Dict[Tuple[str, str], List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]] = {}
        self._proj_bucket_cache_by_dev_dtype: Dict[Tuple[str, str], List[List[Dict[str, object]]]] = {}
        self._proj_group_view_cache_by_dev_dtype: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
        self._scalar_identity_group_cache_by_dev_dtype: Dict[Tuple[str, str], List[Optional[Dict[str, object]]]] = {}
        self._scalar_split_group_cache_by_dev_dtype: Dict[Tuple[str, str], List[Optional[Dict[str, object]]]] = {}

    def _weight_to_compute(self, w: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
        w_comp = w.to(dtype=compute_dtype) if w.dtype != compute_dtype else w
        if abs(float(self._mul_path_scale) - 1.0) > 1e-12:
            w_comp = w_comp * float(self._mul_path_scale)
        return w_comp

    @_dynamo_disable
    def _get_cg_list(self, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._cg_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        if self._cg_cpu_f64 is None:
            self._cg_cpu_f64 = []
            for path_idx, (l1, _p1, l2, _p2, l3, _p3) in enumerate(self.paths):
                C = build_cg_tensor(l1, l2, l3)
                C = _apply_cg_normalization(C, l3, self._normalization)
                C = C * float(self._path_normalization_scales[path_idx])
                self._cg_cpu_f64.append(C)
        cg_list = [C.to(device=device, dtype=compute_dtype) for C in self._cg_cpu_f64]
        self._cg_cache_by_dev_dtype[key] = cg_list
        return cg_list

    @_dynamo_disable
    def _get_proj_group_list(self, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._proj_group_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        cg_list = self._get_cg_list(device=device, dtype=dtype)
        proj_list: List[torch.Tensor] = []
        sparse_list: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
        for g in self._groups:
            l1 = int(g["l1"])
            l2 = int(g["l2"])
            segments = g["segments"]
            k_total = int(g["k_total"])
            m1 = 2 * l1 + 1
            m2 = 2 * l2 + 1
            U = torch.zeros(m1 * m2, k_total, device=device, dtype=compute_dtype)
            for p_idx, _l3, _p3, s, e in segments:
                C = cg_list[int(p_idx)]
                U[:, int(s): int(e)] = C.reshape(m1 * m2, int(e) - int(s))
            proj_list.append(U)
            if device.type == "cuda":
                n = U.numel()
                nz = (U.abs() > _SPARSE_ZERO_THRESHOLD).sum().item()
                zero_frac = 1.0 - (nz / n) if n else 0.0
                if zero_frac >= _SPARSE_MIN_ZERO_FRAC:
                    mask = U.abs() > _SPARSE_ZERO_THRESHOLD
                    nz_flat = mask.nonzero(as_tuple=False)
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
        self._get_proj_group_list(device=device, dtype=dtype)
        key = (str(device), str(self.internal_compute_dtype))
        return self._proj_sparse_cache_by_dev_dtype[key]

    @_dynamo_disable
    def _get_proj_bucket_list(self, device: torch.device, dtype: torch.dtype) -> List[List[Dict[str, object]]]:
        key = (str(device), str(self.internal_compute_dtype))
        cached = self._proj_bucket_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        proj_list = self._get_proj_group_list(device=device, dtype=dtype)
        buckets = [
            _build_kdim_buckets(segments=g["segments"], U=proj_list[g_idx])  # type: ignore[arg-type]
            for g_idx, g in enumerate(self._groups)
        ]
        self._proj_bucket_cache_by_dev_dtype[key] = buckets
        return buckets

    @_dynamo_disable
    def _get_proj_group_view_list(self, device: torch.device, dtype: torch.dtype) -> List[Dict[str, object]]:
        compute_dtype = self.internal_compute_dtype
        key = (str(device), str(compute_dtype))
        cached = self._proj_group_view_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        proj_list = self._get_proj_group_list(device=device, dtype=dtype)
        views: List[Dict[str, object]] = []
        for g_idx, g in enumerate(self._groups):
            l1 = int(g["l1"])
            l2 = int(g["l2"])
            segments = g["segments"]
            U3 = proj_list[g_idx].reshape(2 * l1 + 1, 2 * l2 + 1, -1)
            seg_views = [(seg, U3[:, :, int(seg[-2]): int(seg[-1])]) for seg in segments]
            seg2_stack = None
            seg2_kdims = None
            if len(seg_views) == 2:
                kdims = [int(seg[-1]) - int(seg[-2]) for seg, _ in seg_views]
                kmax = max(kdims)
                seg2_stack = torch.zeros(2, 2 * l1 + 1, 2 * l2 + 1, kmax, device=U3.device, dtype=U3.dtype)
                for seg_idx, (_seg, U_seg) in enumerate(seg_views):
                    seg2_stack[seg_idx, :, :, : U_seg.shape[-1]] = U_seg
                seg2_kdims = tuple(kdims)
            views.append({"U3": U3, "seg_views": seg_views, "seg2_stack": seg2_stack, "seg2_kdims": seg2_kdims})
        self._proj_group_view_cache_by_dev_dtype[key] = views
        return views

    @_dynamo_disable
    def _get_scalar_identity_group_list(self, device: torch.device, dtype: torch.dtype) -> List[Optional[Dict[str, object]]]:
        key = (str(device), str(self.internal_compute_dtype))
        cached = self._scalar_identity_group_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        proj_list = self._get_proj_group_list(device=device, dtype=dtype)
        metas = [
            _detect_scalar_identity_group_meta(
                proj_list[g_idx],
                l1=int(g["l1"]),
                l2=int(g["l2"]),
                segments=g["segments"],  # type: ignore[arg-type]
            )
            for g_idx, g in enumerate(self._groups)
        ]
        self._scalar_identity_group_cache_by_dev_dtype[key] = metas
        return metas

    @_dynamo_disable
    def _get_scalar_split_group_list(self, device: torch.device, dtype: torch.dtype) -> List[Optional[Dict[str, object]]]:
        key = (str(device), str(self.internal_compute_dtype))
        cached = self._scalar_split_group_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        proj_list = self._get_proj_group_list(device=device, dtype=dtype)
        metas = [
            _detect_scalar_output_split_meta(
                proj_list[g_idx],
                l1=int(g["l1"]),
                l2=int(g["l2"]),
                segments=g["segments"],  # type: ignore[arg-type]
            )
            for g_idx, g in enumerate(self._groups)
        ]
        self._scalar_split_group_cache_by_dev_dtype[key] = metas
        return metas

    @_dynamo_disable
    def prewarm_caches(self, device: torch.device, dtype: torch.dtype) -> None:
        _ = self._get_cg_list(device=device, dtype=dtype)
        _ = self._get_proj_group_list(device=device, dtype=dtype)
        _ = self._get_proj_sparse_list(device=device, dtype=dtype)
        _ = self._get_proj_bucket_list(device=device, dtype=dtype)
        _ = self._get_proj_group_view_list(device=device, dtype=dtype)
        _ = self._get_scalar_identity_group_list(device=device, dtype=dtype)
        _ = self._get_scalar_split_group_list(device=device, dtype=dtype)

    def forward(
        self,
        x1: Dict[Tuple[int, int], torch.Tensor],
        x2: Dict[Tuple[int, int], torch.Tensor],
        weights: torch.Tensor | None = None,
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        compute_dtype = self.internal_compute_dtype

        if self.internal_weights:
            w_param = self.weight
        else:
            assert weights is not None
            if weights.shape[-1] not in (self.weight_numel, self.num_paths):
                raise ValueError(f"weights last-dim must be weight_numel={self.weight_numel} or num_paths={self.num_paths}, got {weights.shape[-1]}")
        if weights is not None and (weights.device != device or weights.dtype != dtype):
            weights = weights.to(device=device, dtype=dtype)

        out: Dict[Tuple[int, int], torch.Tensor] = {}
        for key_ir in self.active_irreps:
            l, _p = key_ir
            out[key_ir] = torch.zeros(*batch_shape, self.mul_out, 2 * l + 1, device=device, dtype=dtype)

        if self.internal_weights and (weights is None or weights.shape[-1] == self.num_paths):
            proj_list = self._get_proj_group_list(device=device, dtype=dtype)
            bucket_list = self._get_proj_bucket_list(device=device, dtype=dtype)
            sparse_list = self._get_proj_sparse_list(device=device, dtype=dtype) if device.type == "cuda" else None
            for g_idx, g in enumerate(self._groups):
                l1 = int(g["l1"])
                p1 = int(g["p1"])
                l2 = int(g["l2"])
                p2 = int(g["p2"])
                segments = g["segments"]
                k_total = int(g["k_total"])
                a = x1.get((l1, p1))
                b = x2.get((l2, p2))
                if a is None or b is None:
                    continue
                m1 = 2 * l1 + 1
                m2 = 2 * l2 + 1
                U = proj_list[g_idx]
                num_paths_in_group = len(segments)
                a_comp = a.to(dtype=compute_dtype) if a.dtype != compute_dtype else a
                b_comp = b.to(dtype=compute_dtype) if b.dtype != compute_dtype else b
                B_flat = 1
                for s in batch_shape:
                    B_flat *= int(s)
                a_flat = a_comp.reshape(B_flat, self.mul_in1, m1)
                b_flat = b_comp.reshape(B_flat, self.mul_in2, m2)
                if a_comp.dim() >= 2:
                    bucket_outputs: list[tuple[tuple[int, int], torch.Tensor]] = []
                    cuda_supported = True
                    for bucket in bucket_list[g_idx]:
                        bucket_segments = bucket["segments"]  # type: ignore[assignment]
                        bucket_path_indices = bucket["path_indices"]  # type: ignore[assignment]
                        U_bucket = bucket["U_bucket"]  # type: ignore[assignment]
                        kdim = int(bucket["kdim"])  # type: ignore[arg-type]
                        W_bucket = _stack_group_weights(
                            w_param=w_param,
                            segments=bucket_segments,
                            compute_dtype=compute_dtype,
                        ).to(device=a_comp.device)
                        gates_bucket = None
                        if weights is not None:
                            gates_bucket = _slice_or_index_lastdim(weights, [int(p_idx) for p_idx in bucket_path_indices])
                            gates_bucket = gates_bucket.reshape(B_flat, len(bucket_path_indices)).to(dtype=compute_dtype)
                        bucket_out = _tp_cuda_ext_bucket_forward(
                            backend=self.ictd_tp_backend,
                            a=a_flat,
                            b=b_flat,
                            U_bucket=U_bucket,
                            W_stack=W_bucket,
                            gates=gates_bucket,
                            compute_dtype=compute_dtype,
                        )
                        if bucket_out is None:
                            cuda_supported = False
                            break
                        bucket_out = bucket_out.reshape(*batch_shape, self.mul_out, kdim)
                        bucket_outputs.append(((int(bucket_segments[0][1]), int(bucket_segments[0][2])), bucket_out))
                    if cuda_supported and bucket_outputs:
                        for key_ir, bucket_out in bucket_outputs:
                            bucket_out = bucket_out.to(dtype=dtype) if bucket_out.dtype != dtype else bucket_out
                            out[key_ir] = out[key_ir] + bucket_out
                        continue
                y = None
                used_fused_mix = False
                if a_comp.is_cuda and a_comp.dim() >= 2:
                    if _tp_fused_outer_proj_channel_mix is not None and num_paths_in_group <= 16:
                        W_stack = _stack_group_weights(
                            w_param=w_param,
                            segments=segments,
                            compute_dtype=compute_dtype,
                        ).to(device=a_comp.device)
                        fused_segments = [(int(p_idx), int(l3), int(s), int(e)) for p_idx, l3, _p3, s, e in segments]
                        out_buf = _tp_fused_outer_proj_channel_mix(
                            a_flat, b_flat, U, W_stack, fused_segments, k_total, self.mul_out, m1, m2
                        )
                        if out_buf is not None:
                            out_buf = out_buf.to(dtype=dtype) if out_buf.dtype != dtype else out_buf
                            for seg_idx, (p_idx, l3, p3, s, e) in enumerate(segments):
                                seg_out = out_buf[:, seg_idx, :, int(s): int(e)]
                                if weights is not None:
                                    seg_out = seg_out * weights[..., int(p_idx), None, None]
                                out[(int(l3), int(p3))] = out[(int(l3), int(p3))] + seg_out
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
                        U = proj_list[g_idx]  # (m1*m2, K_total)
                        t_mn = (a_comp.unsqueeze(-2).unsqueeze(-1) * b_comp.unsqueeze(-3).unsqueeze(-2))
                        t_flat = t_mn.reshape(*batch_shape, self.mul_in1, self.mul_in2, m1 * m2)
                        if not t_flat.is_contiguous():
                            t_flat = t_flat.contiguous()
                        y = torch.matmul(t_flat, U)
                    ij = self.mul_in1 * self.mul_in2
                    for p_idx, l3, p3, s, e in segments:
                        Wp = w_param[int(p_idx)]
                        Wp_comp = self._weight_to_compute(Wp, compute_dtype)
                        y_seg = y[..., :, :, int(s): int(e)]
                        kdim = int(e) - int(s)
                        y2 = y_seg.movedim(-1, -3).contiguous().view(*y_seg.shape[:-3], kdim, ij)
                        W2 = Wp_comp.contiguous().view(Wp_comp.shape[0], ij)
                        out_seg = torch.matmul(y2, W2.transpose(0, 1)).movedim(-1, -2)
                        out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                        if weights is not None:
                            out_seg = out_seg * weights[..., int(p_idx), None, None]
                        out[(int(l3), int(p3))] = out[(int(l3), int(p3))] + out_seg
        else:
            # Fallback path: supports external full per-example weights
            # (..., weight_numel) as well as the internal-weight case.
            cg_list = self._get_cg_list(device=device, dtype=dtype)
            idx = 0
            for p_idx, (l1, p1, l2, p2, l3, p3) in enumerate(self.paths):
                if self.internal_weights:
                    gate = 1.0
                    if weights is not None and weights.shape[-1] == self.num_paths:
                        gate = weights[..., p_idx]
                    elif weights is not None and weights.shape[-1] == self.weight_numel:
                        block = self.mul_out * self.mul_in1 * self.mul_in2
                        Wp = weights[..., idx: idx + block].view(
                            *batch_shape, self.mul_out, self.mul_in1, self.mul_in2
                        )
                        idx += block
                    else:
                        Wp = w_param[p_idx]
                else:
                    assert weights is not None
                    block = self.mul_out * self.mul_in1 * self.mul_in2
                    Wp = weights[..., idx: idx + block].view(
                        *batch_shape, self.mul_out, self.mul_in1, self.mul_in2
                    )
                    idx += block
                    gate = 1.0

                a = x1.get((l1, p1))
                b = x2.get((l2, p2))
                if a is None or b is None:
                    continue

                a_comp = a.to(dtype=compute_dtype) if a.dtype != compute_dtype else a
                b_comp = b.to(dtype=compute_dtype) if b.dtype != compute_dtype else b
                Wp_comp = self._weight_to_compute(Wp, compute_dtype)
                C = cg_list[p_idx]
                out_l3 = torch.einsum("...im,...jn,mnk,...oij->...ok", a_comp, b_comp, C, Wp_comp)
                out_l3 = out_l3.to(dtype=dtype) if out_l3.dtype != dtype else out_l3
                if not isinstance(gate, float):
                    out_l3 = out_l3 * gate[..., None, None]
                out[(l3, p3)] = out[(l3, p3)] + out_l3
        return out


class MultipleContractionO3(nn.Module):
    """
    Lightweight higher-order contraction block for the flattened O(3) irreps
    layout used by parity-aware ICTD models.

    This mirrors MultipleContractionSO3 but operates on (l, parity) blocks and
    reuses the existing O(3) fully-connected ICTD tensor product.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        lmax: int,
        active_irreps: List[Tuple[int, int]] | None = None,
        correlation: int = 3,
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.lmax = int(lmax)
        self.active_irreps = (
            [_normalize_irrep_key(l, p) for l, p in active_irreps]
            if active_irreps is not None
            else o3_irrep_keys(self.lmax)
        )
        self.correlation = int(correlation)
        if self.correlation < 1:
            raise ValueError(f"correlation must be >= 1, got {self.correlation}")

        self.reduce = EquivariantChannelLinearO3Rect(
            self.in_channels, self.hidden_channels, self.active_irreps, bias=False
        )
        self.order_mix = nn.ModuleList(
            [
                EquivariantChannelLinearO3(self.hidden_channels, self.active_irreps, bias=False)
                for _ in range(self.correlation)
            ]
        )
        self.tp_layers = nn.ModuleList(
            [
                HarmonicFullyConnectedTensorProductO3(
                    mul_in1=self.hidden_channels,
                    mul_in2=self.hidden_channels,
                    mul_out=self.hidden_channels,
                    lmax=self.lmax,
                    active_irreps=self.active_irreps,
                    internal_weights=True,
                    path_policy=ictd_tp_path_policy,
                    max_rank_other=ictd_tp_max_rank_other,
                    internal_compute_dtype=internal_compute_dtype,
                    ictd_tp_backend=ictd_tp_backend,
                )
                for _ in range(max(self.correlation - 1, 0))
            ]
        )
        self.out_linear = EquivariantChannelLinearO3(
            self.hidden_channels, self.active_irreps, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.reduce(x)
        accum = self.order_mix[0](base)
        if self.correlation == 1:
            return self.out_linear(accum)

        base_blocks = split_flat_irreps_o3(base, self.hidden_channels, self.active_irreps)
        current_blocks = base_blocks
        for order_idx, tp in enumerate(self.tp_layers, start=1):
            current_blocks = tp(current_blocks, base_blocks)
            current_flat = merge_flat_irreps_o3(current_blocks, self.hidden_channels, self.active_irreps)
            current_flat = self.order_mix[order_idx](current_flat)
            accum = accum + current_flat
        return self.out_linear(accum)


class HarmonicChannelWiseTensorProductO3(HarmonicFullyConnectedTensorProductO3):
    """
    O(3) parity-aware channel-wise ICTD tensor product specialized for convolution-style usage.

    Differences from HarmonicFullyConnectedTensorProductO3:
      - only supports internal learnable weights + optional per-path scalar gates
      - channel mixing is restricted to channel-wise pairing:
          * mul_in2 == 1: geometry/scalar broadcast over input channels
          * mul_in2 == mul_in1: elementwise channel pairing
      - does not support generic external full weights (..., weight_numel)
    """

    def __init__(
        self,
        mul_in1: int,
        mul_in2: int,
        mul_out: int,
        lmax: int,
        active_irreps: List[Tuple[int, int]] | None = None,
        internal_weights: bool = True,
        *,
        allowed_paths: List[Tuple[int, int, int, int, int, int]] | None = None,
        path_policy: str = "full",
        max_rank_other: int | None = None,
        normalization: str = "component",
        irrep_normalization: str | None = None,
        path_normalization: str = "element",
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
    ):
        if not internal_weights:
            raise ValueError("HarmonicChannelWiseTensorProductO3 only supports internal_weights=True")
        if mul_in2 not in (1, mul_in1):
            raise ValueError(
                f"HarmonicChannelWiseTensorProductO3 requires mul_in2 in {{1, mul_in1={mul_in1}}}, got {mul_in2}"
            )
        super().__init__(
            mul_in1=mul_in1,
            mul_in2=mul_in2,
            mul_out=mul_out,
            lmax=lmax,
            active_irreps=active_irreps,
            internal_weights=True,
            allowed_paths=allowed_paths,
            path_policy=path_policy,
            max_rank_other=max_rank_other,
            normalization=normalization,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            internal_compute_dtype=internal_compute_dtype,
            ictd_tp_backend=ictd_tp_backend,
        )
        self.channel_mul = int(mul_in1)
        self.channel_mode = "broadcast_rhs" if int(mul_in2) == 1 else "paired"
        self.weight_numel = self.num_paths * self.mul_out * self.channel_mul
        self.weight = nn.Parameter(torch.randn(self.num_paths, self.mul_out, self.channel_mul) * 0.02)
        self._canonical_only = _is_canonical_irrep_list(self.active_irreps, lmax=self.lmax)
        self._canonical_keys_by_l = {int(l): (int(l), canonical_irrep_parity_sign(int(l))) for l in range(self.lmax + 1)}
        out_offsets: List[int] = []
        start = 0
        for l in range(self.lmax + 1):
            out_offsets.append(start)
            start += 2 * l + 1
        self._out_offsets = tuple(out_offsets)
        self._out_total_dim = start
        out_offsets: List[int] = []
        start = 0
        for l in range(self.lmax + 1):
            out_offsets.append(start)
            start += 2 * l + 1
        self._out_offsets = tuple(out_offsets)
        self._out_total_dim = start

    @_dynamo_disable
    def _forward_canonical(
        self,
        x1: Dict[Tuple[int, int], torch.Tensor],
        x2: Dict[Tuple[int, int], torch.Tensor],
        weights: torch.Tensor | None = None,
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        compute_dtype = self.internal_compute_dtype
        validate_shapes = not (torch.jit.is_tracing() or torch.jit.is_scripting())

        if weights is not None:
            if validate_shapes and weights.shape[-1] != self.num_paths:
                raise ValueError(
                    f"HarmonicChannelWiseTensorProductO3 only accepts path gates with last-dim num_paths={self.num_paths}, "
                    f"got {weights.shape[-1]}"
                )
            if weights.device != device or weights.dtype != dtype:
                weights = weights.to(device=device, dtype=dtype)

        out_by_l: Dict[int, torch.Tensor] = {
            l: torch.zeros(*batch_shape, self.mul_out, 2 * l + 1, device=device, dtype=dtype)
            for l in range(self.lmax + 1)
        }
        proj_list = self._get_proj_group_list(device=device, dtype=dtype)
        bucket_list = self._get_proj_bucket_list(device=device, dtype=dtype)
        proj_view_list = self._get_proj_group_view_list(device=device, dtype=dtype)
        scalar_meta_list = self._get_scalar_identity_group_list(device=device, dtype=dtype)
        scalar_split_list = self._get_scalar_split_group_list(device=device, dtype=dtype)

        for g_idx, g in enumerate(self._groups):
            l1 = int(g["l1"])
            p1 = int(g["p1"])
            l2 = int(g["l2"])
            p2 = int(g["p2"])
            segments = g["segments"]
            k_total = int(g["k_total"])
            a = x1.get((l1, p1))
            b = x2.get((l2, p2))
            if a is None or b is None:
                continue

            if validate_shapes and a.shape[-2] != self.channel_mul:
                raise ValueError(f"x1[{(l1, p1)}] channel dim must be {self.channel_mul}, got {a.shape[-2]}")
            if self.channel_mode == "broadcast_rhs":
                if validate_shapes and b.shape[-2] != 1:
                    raise ValueError(
                        f"x2[{(l2, p2)}] channel dim must be 1 for broadcast_rhs mode, got {b.shape[-2]}"
                    )
            else:
                if validate_shapes and b.shape[-2] != self.channel_mul:
                    raise ValueError(
                        f"x2[{(l2, p2)}] channel dim must be {self.channel_mul} for paired mode, got {b.shape[-2]}"
                    )

            a_comp = a.to(dtype=compute_dtype) if a.dtype != compute_dtype else a
            b_comp = b.to(dtype=compute_dtype) if b.dtype != compute_dtype else b
            scalar_meta = scalar_meta_list[g_idx]
            if scalar_meta is not None:
                side = str(scalar_meta["side"])
                if side == "rhs":
                    base = a_comp * b_comp[..., 0, 0][..., None, None]
                else:
                    lhs = a_comp[..., :, 0]
                    if self.channel_mode == "broadcast_rhs":
                        base = lhs.unsqueeze(-1) * b_comp[..., 0, :].unsqueeze(-2)
                    else:
                        base = lhs.unsqueeze(-1) * b_comp
                for p_idx, key_ir, alpha in scalar_meta["segments"]:  # type: ignore[index]
                    Wp = self.weight[int(p_idx)]
                    Wp_comp = self._weight_to_compute(Wp, compute_dtype)
                    out_seg = torch.matmul(
                        (base * float(alpha)).movedim(-1, -2).contiguous(),
                        Wp_comp.transpose(0, 1),
                    ).movedim(-1, -2)
                    out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                    if weights is not None:
                        out_seg = out_seg * weights[..., int(p_idx), None, None]
                    out_by_l[int(key_ir[0])] = out_by_l[int(key_ir[0])] + out_seg
                continue

            scalar_split_meta = scalar_split_list[g_idx] if self.channel_mode == "paired" else None
            active_segments = segments
            active_U = proj_list[g_idx]
            if scalar_split_meta is not None:
                base_scalar = (a_comp * b_comp).sum(dim=-1, keepdim=True)
                for p_idx, key_ir, alpha in scalar_split_meta["scalar_entries"]:  # type: ignore[index]
                    Wp = self.weight[int(p_idx)]
                    Wp_comp = self._weight_to_compute(Wp, compute_dtype)
                    out_seg = torch.matmul(
                        (base_scalar * float(alpha)).movedim(-1, -2).contiguous(),
                        Wp_comp.transpose(0, 1),
                    ).movedim(-1, -2)
                    out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                    if weights is not None:
                        out_seg = out_seg * weights[..., int(p_idx), None, None]
                    out_by_l[int(key_ir[0])] = out_by_l[int(key_ir[0])] + out_seg
                active_segments = scalar_split_meta["rem_segments"]  # type: ignore[assignment]
                active_U = scalar_split_meta["U_rem"]  # type: ignore[assignment]
                if active_U is None:
                    continue

            used_fused_mix = False
            if self.channel_mode == "broadcast_rhs" and a_comp.dim() >= 2:
                B_flat = 1
                for s in batch_shape:
                    B_flat *= int(s)
                a_flat = a_comp.reshape(B_flat, self.channel_mul, 2 * l1 + 1)
                b_flat = b_comp.reshape(B_flat, 1, 2 * l2 + 1)
                if _USE_TRITON_CHANNELWISE_TP and _tp_fused_outer_proj_channel_mix is not None and len(segments) <= 16:
                    W_stack = _stack_group_weights(
                        w_param=self.weight,
                        segments=segments,
                        compute_dtype=compute_dtype,
                    ).to(device=a_comp.device)
                    fused_segments = [(int(p_idx), int(l3), int(s), int(e)) for p_idx, l3, _p3, s, e in segments]
                    out_buf = _tp_fused_outer_proj_channel_mix(
                        a_flat,
                        b_flat,
                        proj_list[g_idx],
                        W_stack,
                        fused_segments,
                        k_total,
                        self.mul_out,
                        2 * l1 + 1,
                        2 * l2 + 1,
                    )
                    if out_buf is not None:
                        out_buf = out_buf.to(dtype=dtype) if out_buf.dtype != dtype else out_buf
                        for seg_idx, (p_idx, l3, _p3, s, e) in enumerate(segments):
                            seg_out = out_buf[:, seg_idx, :, int(s): int(e)].reshape(*batch_shape, self.mul_out, int(e) - int(s))
                            if weights is not None:
                                seg_out = seg_out * weights[..., int(p_idx), None, None]
                            out_by_l[int(l3)] = out_by_l[int(l3)] + seg_out
                        used_fused_mix = True
                if used_fused_mix:
                    continue
                gates_all = (
                    weights.reshape(B_flat, weights.shape[-1]).to(dtype=compute_dtype)
                    if weights is not None
                    else None
                )
                if ensure_grouped_tp_cuda_ext_supported(
                    backend=self.ictd_tp_backend,
                    sample=a_flat,
                    compute_dtype=compute_dtype,
                    internal_weights=True,
                    weights=gates_all,
                ):
                    bucket_outputs: list[tuple[int, torch.Tensor]] = []
                    assert bucket_list is not None
                    for bucket in bucket_list[g_idx]:
                        bucket_segments = bucket["segments"]  # type: ignore[assignment]
                        bucket_path_indices = bucket["path_indices"]  # type: ignore[assignment]
                        U_bucket = bucket["U_bucket"]  # type: ignore[assignment]
                        kdim = int(bucket["kdim"])  # type: ignore[arg-type]
                        W_bucket = _stack_group_weights(
                            w_param=self.weight,
                            segments=bucket_segments,
                            compute_dtype=compute_dtype,
                        ).to(device=a_comp.device)
                        gates_bucket = None
                        if gates_all is not None:
                            gates_bucket = _slice_or_index_lastdim(gates_all, [int(p_idx) for p_idx in bucket_path_indices])
                        bucket_out = _tp_cuda_ext_bucket_forward(
                            backend=self.ictd_tp_backend,
                            a=a_flat,
                            b=b_flat,
                            U_bucket=U_bucket,
                            W_stack=W_bucket,
                            gates=gates_bucket,
                            compute_dtype=compute_dtype,
                        )
                        if bucket_out is None:
                            bucket_outputs = []
                            break
                        bucket_outputs.append((int(bucket_segments[0][1]), bucket_out.reshape(*batch_shape, self.mul_out, kdim)))
                    if bucket_outputs:
                        for l3, bucket_out in bucket_outputs:
                            bucket_out = bucket_out.to(dtype=dtype) if bucket_out.dtype != dtype else bucket_out
                            out_by_l[l3] = out_by_l[l3] + bucket_out
                        continue

            m1 = 2 * l1 + 1
            m2 = 2 * l2 + 1
            if active_segments is segments and active_U is proj_list[g_idx]:
                view_meta = proj_view_list[g_idx]
                U3 = view_meta["U3"]  # type: ignore[assignment]
                seg_views = view_meta["seg_views"]  # type: ignore[assignment]
            else:
                U3 = active_U.reshape(m1, m2, -1)
                seg_views = [(seg, U3[:, :, int(seg[-2]): int(seg[-1])]) for seg in active_segments]
            if len(active_segments) <= 2:
                if self.channel_mode == "broadcast_rhs":
                    b_rhs = b_comp[..., 0, :]
                    if (not self.training) and active_segments is segments and active_U is proj_list[g_idx] and len(active_segments) == 2:
                        seg2_stack = view_meta.get("seg2_stack")
                        seg2_kdims = view_meta.get("seg2_kdims")
                        if seg2_stack is not None and seg2_kdims is not None:
                            W_stack = _stack_group_weights(
                                w_param=self.weight,
                                segments=segments,
                                compute_dtype=compute_dtype,
                                mul_scale=self._mul_path_scale,
                            )
                            out_stack = torch.einsum("...cm,...n,pmnk,poc->...pok", a_comp, b_rhs, seg2_stack, W_stack)
                            out_stack = out_stack.to(dtype=dtype) if out_stack.dtype != dtype else out_stack
                            for seg_idx, seg in enumerate(segments):
                                if len(seg) == 5:
                                    p_idx, l3, _p3, s, e = seg  # type: ignore[misc]
                                else:
                                    p_idx, key_ir, s, e = seg  # type: ignore[misc]
                                    l3 = int(key_ir[0])
                                kdim = int(e) - int(s)
                                out_seg = out_stack[..., seg_idx, :, :kdim]
                                if weights is not None:
                                    out_seg = out_seg * weights[..., int(p_idx), None, None]
                                out_by_l[int(l3)] = out_by_l[int(l3)] + out_seg
                            continue
                    for seg, U_seg in seg_views:
                        if len(seg) == 5:
                            p_idx, l3, _p3, _s, _e = seg  # type: ignore[misc]
                        else:
                            p_idx, key_ir, _s, _e = seg  # type: ignore[misc]
                            l3 = int(key_ir[0])
                        Wp = self.weight[int(p_idx)]
                        Wp_comp = self._weight_to_compute(Wp, compute_dtype)
                        out_seg = torch.einsum("...cm,...n,mnk,oc->...ok", a_comp, b_rhs, U_seg, Wp_comp)
                        out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                        if weights is not None:
                            out_seg = out_seg * weights[..., int(p_idx), None, None]
                        out_by_l[int(l3)] = out_by_l[int(l3)] + out_seg
                else:
                    for seg, U_seg in seg_views:
                        if len(seg) == 5:
                            p_idx, l3, _p3, _s, _e = seg  # type: ignore[misc]
                        else:
                            p_idx, key_ir, _s, _e = seg  # type: ignore[misc]
                            l3 = int(key_ir[0])
                        Wp = self.weight[int(p_idx)]
                        Wp_comp = self._weight_to_compute(Wp, compute_dtype)
                        out_seg = torch.einsum("...cm,...cn,mnk,oc->...ok", a_comp, b_comp, U_seg, Wp_comp)
                        out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                        if weights is not None:
                            out_seg = out_seg * weights[..., int(p_idx), None, None]
                        out_by_l[int(l3)] = out_by_l[int(l3)] + out_seg
                continue

            if self.channel_mode == "broadcast_rhs":
                b_rhs = b_comp[..., 0, :]
                y = torch.einsum("...cm,...n,mnk->...ck", a_comp, b_rhs, U3)
            else:
                y = torch.einsum("...cm,...cn,mnk->...ck", a_comp, b_comp, U3)
            if _USE_BUCKETED_CHANNELWISE_MIX:
                for bucket in bucket_list[g_idx]:
                    bucket_segments = bucket["segments"]  # type: ignore[assignment]
                    if active_segments is not segments:
                        continue
                    starts = [int(seg[-2]) for seg in bucket_segments]
                    kdim = int(bucket["kdim"])  # type: ignore[arg-type]
                    y_bucket = torch.stack([y[..., :, s: s + kdim] for s in starts], dim=-3)
                    W_bucket = _stack_group_weights(
                        w_param=self.weight,
                        segments=bucket_segments,
                        compute_dtype=compute_dtype,
                    )
                    out_bucket = torch.einsum("...pck,poc->...pok", y_bucket, W_bucket)
                    out_bucket = out_bucket.to(dtype=dtype) if out_bucket.dtype != dtype else out_bucket
                    if weights is not None:
                        path_indices = [int(seg[0]) for seg in bucket_segments]
                        gates_bucket = _slice_or_index_lastdim(weights, path_indices)
                        out_bucket = out_bucket * gates_bucket[..., :, None, None]
                    out_by_l[int(bucket_segments[0][1])] = out_by_l[int(bucket_segments[0][1])] + out_bucket.sum(dim=-3)
            else:
                for seg in active_segments:
                    if len(seg) == 5:
                        p_idx, l3, _p3, s, e = seg  # type: ignore[misc]
                    else:
                        p_idx, key_ir, s, e = seg  # type: ignore[misc]
                        l3 = int(key_ir[0])
                    Wp = self.weight[int(p_idx)]
                    Wp_comp = self._weight_to_compute(Wp, compute_dtype)
                    y_seg = y[..., :, int(s): int(e)]
                    out_seg = torch.matmul(
                        y_seg.movedim(-1, -2).contiguous(),
                        Wp_comp.transpose(0, 1),
                    ).movedim(-1, -2)
                    out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                    if weights is not None:
                        out_seg = out_seg * weights[..., int(p_idx), None, None]
                    out_by_l[int(l3)] = out_by_l[int(l3)] + out_seg

        return {self._canonical_keys_by_l[l]: out_by_l[l] for l in range(self.lmax + 1)}

    def forward(
        self,
        x1: Dict[Tuple[int, int], torch.Tensor],
        x2: Dict[Tuple[int, int], torch.Tensor],
        weights: torch.Tensor | None = None,
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        if self._canonical_only:
            return self._forward_canonical(x1, x2, weights)
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        compute_dtype = self.internal_compute_dtype
        validate_shapes = not (torch.jit.is_tracing() or torch.jit.is_scripting())

        if weights is not None:
            if validate_shapes and weights.shape[-1] != self.num_paths:
                raise ValueError(
                    f"HarmonicChannelWiseTensorProductO3 only accepts path gates with last-dim num_paths={self.num_paths}, "
                    f"got {weights.shape[-1]}"
                )
            if weights.device != device or weights.dtype != dtype:
                weights = weights.to(device=device, dtype=dtype)

        out: Dict[Tuple[int, int], torch.Tensor] = {
            key_ir: torch.zeros(*batch_shape, self.mul_out, 2 * key_ir[0] + 1, device=device, dtype=dtype)
            for key_ir in self.active_irreps
        }
        proj_list = self._get_proj_group_list(device=device, dtype=dtype)
        bucket_list = self._get_proj_bucket_list(device=device, dtype=dtype)
        proj_view_list = self._get_proj_group_view_list(device=device, dtype=dtype)
        scalar_meta_list = self._get_scalar_identity_group_list(device=device, dtype=dtype)
        scalar_split_list = self._get_scalar_split_group_list(device=device, dtype=dtype)

        for g_idx, g in enumerate(self._groups):
            l1 = int(g["l1"])
            p1 = int(g["p1"])
            l2 = int(g["l2"])
            p2 = int(g["p2"])
            segments = g["segments"]
            k_total = int(g["k_total"])
            a = x1.get((l1, p1))
            b = x2.get((l2, p2))
            if a is None or b is None:
                continue

            if validate_shapes and a.shape[-2] != self.channel_mul:
                raise ValueError(
                    f"x1[{(l1, p1)}] channel dim must be {self.channel_mul}, got {a.shape[-2]}"
                )
            if self.channel_mode == "broadcast_rhs":
                if validate_shapes and b.shape[-2] != 1:
                    raise ValueError(
                        f"x2[{(l2, p2)}] channel dim must be 1 for broadcast_rhs mode, got {b.shape[-2]}"
                    )
            else:
                if validate_shapes and b.shape[-2] != self.channel_mul:
                    raise ValueError(
                        f"x2[{(l2, p2)}] channel dim must be {self.channel_mul} for paired mode, got {b.shape[-2]}"
                    )

            a_comp = a.to(dtype=compute_dtype) if a.dtype != compute_dtype else a
            b_comp = b.to(dtype=compute_dtype) if b.dtype != compute_dtype else b
            scalar_meta = scalar_meta_list[g_idx]
            if scalar_meta is not None:
                side = str(scalar_meta["side"])
                if side == "rhs":
                    base = a_comp * b_comp[..., 0, 0][..., None, None]
                else:
                    lhs = a_comp[..., :, 0]
                    if self.channel_mode == "broadcast_rhs":
                        base = lhs.unsqueeze(-1) * b_comp[..., 0, :].unsqueeze(-2)
                    else:
                        base = lhs.unsqueeze(-1) * b_comp
                for p_idx, key_ir, alpha in scalar_meta["segments"]:  # type: ignore[index]
                    Wp = self.weight[int(p_idx)]
                    Wp_comp = self._weight_to_compute(Wp, compute_dtype)
                    out_seg = torch.matmul(
                        (base * float(alpha)).movedim(-1, -2).contiguous(),
                        Wp_comp.transpose(0, 1),
                    ).movedim(-1, -2)
                    out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                    if weights is not None:
                        out_seg = out_seg * weights[..., int(p_idx), None, None]
                    out[key_ir] = out[key_ir] + out_seg
                continue
            scalar_split_meta = scalar_split_list[g_idx] if self.channel_mode == "paired" else None
            active_segments = segments
            active_U = proj_list[g_idx]
            if scalar_split_meta is not None:
                base_scalar = (a_comp * b_comp).sum(dim=-1, keepdim=True)
                for p_idx, key_ir, alpha in scalar_split_meta["scalar_entries"]:  # type: ignore[index]
                    Wp = self.weight[int(p_idx)]
                    Wp_comp = self._weight_to_compute(Wp, compute_dtype)
                    out_seg = torch.matmul(
                        (base_scalar * float(alpha)).movedim(-1, -2).contiguous(),
                        Wp_comp.transpose(0, 1),
                    ).movedim(-1, -2)
                    out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                    if weights is not None:
                        out_seg = out_seg * weights[..., int(p_idx), None, None]
                    out[key_ir] = out[key_ir] + out_seg
                active_segments = scalar_split_meta["rem_segments"]  # type: ignore[assignment]
                active_U = scalar_split_meta["U_rem"]  # type: ignore[assignment]
                if active_U is None:
                    continue
            used_fused_mix = False
            if self.channel_mode == "broadcast_rhs" and a_comp.dim() >= 2:
                B_flat = 1
                for s in batch_shape:
                    B_flat *= int(s)
                a_flat = a_comp.reshape(B_flat, self.channel_mul, 2 * l1 + 1)
                b_flat = b_comp.reshape(B_flat, 1, 2 * l2 + 1)
                if _USE_TRITON_CHANNELWISE_TP and _tp_fused_outer_proj_channel_mix is not None and len(segments) <= 16:
                    W_stack = _stack_group_weights(
                        w_param=self.weight,
                        segments=segments,
                        compute_dtype=compute_dtype,
                        mul_scale=self._mul_path_scale,
                    ).to(device=a_comp.device)
                    fused_segments = [(int(p_idx), int(l3), int(s), int(e)) for p_idx, l3, _p3, s, e in segments]
                    out_buf = _tp_fused_outer_proj_channel_mix(
                        a_flat,
                        b_flat,
                        proj_list[g_idx],
                        W_stack,
                        fused_segments,
                        k_total,
                        self.mul_out,
                        2 * l1 + 1,
                        2 * l2 + 1,
                    )
                    if out_buf is not None:
                        out_buf = out_buf.to(dtype=dtype) if out_buf.dtype != dtype else out_buf
                        for seg_idx, (p_idx, l3, p3, s, e) in enumerate(segments):
                            seg_out = out_buf[:, seg_idx, :, int(s): int(e)].reshape(*batch_shape, self.mul_out, int(e) - int(s))
                            if weights is not None:
                                seg_out = seg_out * weights[..., int(p_idx), None, None]
                            out[(int(l3), int(p3))] = out[(int(l3), int(p3))] + seg_out
                        used_fused_mix = True
                if used_fused_mix:
                    continue
                gates_all = (
                    weights.reshape(B_flat, weights.shape[-1]).to(dtype=compute_dtype)
                    if weights is not None
                    else None
                )
                if ensure_grouped_tp_cuda_ext_supported(
                    backend=self.ictd_tp_backend,
                    sample=a_flat,
                    compute_dtype=compute_dtype,
                    internal_weights=True,
                    weights=gates_all,
                ):
                    bucket_outputs: list[tuple[tuple[int, int], torch.Tensor]] = []
                    assert bucket_list is not None
                    for bucket in bucket_list[g_idx]:
                        bucket_segments = bucket["segments"]  # type: ignore[assignment]
                        bucket_path_indices = bucket["path_indices"]  # type: ignore[assignment]
                        U_bucket = bucket["U_bucket"]  # type: ignore[assignment]
                        kdim = int(bucket["kdim"])  # type: ignore[arg-type]
                        W_bucket = _stack_group_weights(
                            w_param=self.weight,
                            segments=bucket_segments,
                            compute_dtype=compute_dtype,
                            mul_scale=self._mul_path_scale,
                        ).to(device=a_comp.device)
                        gates_bucket = None
                        if gates_all is not None:
                            gates_bucket = _slice_or_index_lastdim(gates_all, [int(p_idx) for p_idx in bucket_path_indices])
                        bucket_out = _tp_cuda_ext_bucket_forward(
                            backend=self.ictd_tp_backend,
                            a=a_flat,
                            b=b_flat,
                            U_bucket=U_bucket,
                            W_stack=W_bucket,
                            gates=gates_bucket,
                            compute_dtype=compute_dtype,
                        )
                        if bucket_out is None:
                            bucket_outputs = []
                            break
                        bucket_outputs.append(
                            (
                                (int(bucket_segments[0][1]), int(bucket_segments[0][2])),
                                bucket_out.reshape(*batch_shape, self.mul_out, kdim),
                            )
                        )
                    if bucket_outputs:
                        for key_ir, bucket_out in bucket_outputs:
                            bucket_out = bucket_out.to(dtype=dtype) if bucket_out.dtype != dtype else bucket_out
                            out[key_ir] = out[key_ir] + bucket_out
                        continue
            m1 = 2 * l1 + 1
            m2 = 2 * l2 + 1
            U3 = active_U.reshape(m1, m2, -1)
            if len(active_segments) <= 2:
                if self.channel_mode == "broadcast_rhs":
                    b_rhs = b_comp[..., 0, :]
                    for seg in active_segments:
                        if len(seg) == 5:
                            p_idx, l3, p3, s, e = seg
                            key_ir = (int(l3), int(p3))
                        else:
                            p_idx, key_ir, s, e = seg
                        Wp = self.weight[int(p_idx)]
                        Wp_comp = self._weight_to_compute(Wp, compute_dtype)
                        out_seg = torch.einsum(
                            "...cm,...n,mnk,oc->...ok",
                            a_comp,
                            b_rhs,
                            U3[:, :, int(s): int(e)],
                            Wp_comp,
                        )
                        out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                        if weights is not None:
                            out_seg = out_seg * weights[..., int(p_idx), None, None]
                        out[key_ir] = out[key_ir] + out_seg
                else:
                    for seg in active_segments:
                        if len(seg) == 5:
                            p_idx, l3, p3, s, e = seg
                            key_ir = (int(l3), int(p3))
                        else:
                            p_idx, key_ir, s, e = seg
                        Wp = self.weight[int(p_idx)]
                        Wp_comp = self._weight_to_compute(Wp, compute_dtype)
                        out_seg = torch.einsum(
                            "...cm,...cn,mnk,oc->...ok",
                            a_comp,
                            b_comp,
                            U3[:, :, int(s): int(e)],
                            Wp_comp,
                        )
                        out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                        if weights is not None:
                            out_seg = out_seg * weights[..., int(p_idx), None, None]
                        out[key_ir] = out[key_ir] + out_seg
                continue
            if self.channel_mode == "broadcast_rhs":
                b_rhs = b_comp[..., 0, :]
                y = torch.einsum("...cm,...n,mnk->...ck", a_comp, b_rhs, U3)
            else:
                y = torch.einsum("...cm,...cn,mnk->...ck", a_comp, b_comp, U3)
            if _USE_BUCKETED_CHANNELWISE_MIX:
                for bucket in bucket_list[g_idx]:
                    bucket_segments = bucket["segments"]  # type: ignore[assignment]
                    if active_segments is not segments:
                        continue
                    starts = [int(seg[-2]) for seg in bucket_segments]
                    kdim = int(bucket["kdim"])  # type: ignore[arg-type]
                    y_bucket = torch.stack([y[..., :, s : s + kdim] for s in starts], dim=-3)
                    W_bucket = _stack_group_weights(
                        w_param=self.weight,
                        segments=bucket_segments,
                        compute_dtype=compute_dtype,
                        mul_scale=self._mul_path_scale,
                    )
                    out_bucket = torch.einsum("...pck,poc->...pok", y_bucket, W_bucket)
                    out_bucket = out_bucket.to(dtype=dtype) if out_bucket.dtype != dtype else out_bucket
                    if weights is not None:
                        path_indices = [int(seg[0]) for seg in bucket_segments]
                        gates_bucket = _slice_or_index_lastdim(weights, path_indices)
                        out_bucket = out_bucket * gates_bucket[..., :, None, None]
                    key_to_local: Dict[Tuple[int, int], List[int]] = {}
                    for local_idx, seg in enumerate(bucket_segments):
                        key_to_local.setdefault((int(seg[1]), int(seg[2])), []).append(local_idx)
                    for key_ir, local_indices in key_to_local.items():
                        out[key_ir] = out[key_ir] + out_bucket[..., local_indices, :, :].sum(dim=-3)
            else:
                for p_idx, l3, p3, s, e in active_segments:
                    Wp = self.weight[int(p_idx)]
                    Wp_comp = self._weight_to_compute(Wp, compute_dtype)
                    y_seg = y[..., :, int(s): int(e)]
                    out_seg = torch.matmul(
                        y_seg.movedim(-1, -2).contiguous(),
                        Wp_comp.transpose(0, 1),
                    ).movedim(-1, -2)
                    out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                    if weights is not None:
                        out_seg = out_seg * weights[..., int(p_idx), None, None]
                    out[(int(l3), int(p3))] = out[(int(l3), int(p3))] + out_seg
        return out


class HarmonicChannelWiseTensorProduct(HarmonicFullyConnectedTensorProduct):
    """
    Channel-wise ICTD tensor product specialized for convolution-style usage.

    Differences from HarmonicFullyConnectedTensorProduct:
      - only supports internal learnable weights + optional per-path scalar gates
      - channel mixing is restricted to channel-wise pairing:
          * mul_in2 == 1: geometry/scalar broadcast over input channels
          * mul_in2 == mul_in1: elementwise channel pairing
      - does not support generic external full weights (..., weight_numel)
    """

    def __init__(
        self,
        mul_in1: int,
        mul_in2: int,
        mul_out: int,
        lmax: int,
        internal_weights: bool = True,
        *,
        allowed_paths: List[Tuple[int, int, int]] | None = None,
        path_policy: str = "full",
        max_rank_other: int | None = None,
        normalization: str = "component",
        irrep_normalization: str | None = None,
        path_normalization: str = "element",
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "pytorch",
    ):
        if not internal_weights:
            raise ValueError("HarmonicChannelWiseTensorProduct only supports internal_weights=True")
        if mul_in2 not in (1, mul_in1):
            raise ValueError(
                f"HarmonicChannelWiseTensorProduct requires mul_in2 in {{1, mul_in1={mul_in1}}}, got {mul_in2}"
            )
        super().__init__(
            mul_in1=mul_in1,
            mul_in2=mul_in2,
            mul_out=mul_out,
            lmax=lmax,
            internal_weights=True,
            allowed_paths=allowed_paths,
            path_policy=path_policy,
            max_rank_other=max_rank_other,
            normalization=normalization,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            internal_compute_dtype=internal_compute_dtype,
            ictd_tp_backend=ictd_tp_backend,
        )
        self.channel_mul = int(mul_in1)
        self.channel_mode = "broadcast_rhs" if int(mul_in2) == 1 else "paired"
        self.weight_numel = self.num_paths * self.mul_out * self.channel_mul
        self.weight = nn.Parameter(torch.randn(self.num_paths, self.mul_out, self.channel_mul) * 0.02)
        out_offsets: List[int] = []
        start = 0
        for l in range(self.lmax + 1):
            out_offsets.append(start)
            start += 2 * l + 1
        self._out_offsets = tuple(out_offsets)
        self._out_total_dim = start

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
        validate_shapes = not (torch.jit.is_tracing() or torch.jit.is_scripting())

        if weights is not None:
            if validate_shapes and weights.shape[-1] != self.num_paths:
                raise ValueError(
                    f"HarmonicChannelWiseTensorProduct only accepts path gates with last-dim num_paths={self.num_paths}, "
                    f"got {weights.shape[-1]}"
                )
            if weights.device != device or weights.dtype != dtype:
                weights = weights.to(device=device, dtype=dtype)

        out: Dict[int, torch.Tensor] = {
            l: torch.zeros(*batch_shape, self.mul_out, 2 * l + 1, device=device, dtype=dtype)
            for l in range(self.lmax + 1)
        }
        proj_list = self._get_proj_group_list(device=device, dtype=dtype)
        bucket_list = self._get_proj_bucket_list(device=device, dtype=dtype)
        proj_view_list = self._get_proj_group_view_list(device=device, dtype=dtype)
        scalar_meta_list = self._get_scalar_identity_group_list(device=device, dtype=dtype)
        scalar_split_list = self._get_scalar_split_group_list(device=device, dtype=dtype)

        for g_idx, g in enumerate(self._groups):
            l1 = int(g["l1"])  # type: ignore[arg-type]
            l2 = int(g["l2"])  # type: ignore[arg-type]
            segments = g["segments"]  # type: ignore[assignment]
            k_total = int(g["k_total"])  # type: ignore[arg-type]
            a = x1.get(l1)
            b = x2.get(l2)
            if a is None or b is None:
                continue

            if validate_shapes and a.shape[-2] != self.channel_mul:
                raise ValueError(
                    f"x1[{l1}] channel dim must be {self.channel_mul}, got {a.shape[-2]}"
                )
            if self.channel_mode == "broadcast_rhs":
                if validate_shapes and b.shape[-2] != 1:
                    raise ValueError(f"x2[{l2}] channel dim must be 1 for broadcast_rhs mode, got {b.shape[-2]}")
            else:
                if validate_shapes and b.shape[-2] != self.channel_mul:
                    raise ValueError(
                        f"x2[{l2}] channel dim must be {self.channel_mul} for paired mode, got {b.shape[-2]}"
                    )

            a_comp = a.to(dtype=compute_dtype) if a.dtype != compute_dtype else a
            b_comp = b.to(dtype=compute_dtype) if b.dtype != compute_dtype else b
            scalar_meta = scalar_meta_list[g_idx]
            if scalar_meta is not None:
                side = str(scalar_meta["side"])
                if side == "rhs":
                    base = a_comp * b_comp[..., 0, 0][..., None, None]
                else:
                    lhs = a_comp[..., :, 0]
                    if self.channel_mode == "broadcast_rhs":
                        base = lhs.unsqueeze(-1) * b_comp[..., 0, :].unsqueeze(-2)
                    else:
                        base = lhs.unsqueeze(-1) * b_comp
                for p_idx, key_ir, alpha in scalar_meta["segments"]:  # type: ignore[index]
                    Wp = self.weight[int(p_idx)]
                    Wp_comp = self._weight_to_compute(Wp, compute_dtype)
                    out_seg = torch.matmul(
                        (base * float(alpha)).movedim(-1, -2).contiguous(),
                        Wp_comp.transpose(0, 1),
                    ).movedim(-1, -2)
                    out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                    if weights is not None:
                        out_seg = out_seg * weights[..., int(p_idx), None, None]
                    out[int(key_ir)] = out[int(key_ir)] + out_seg
                continue
            scalar_split_meta = scalar_split_list[g_idx] if self.channel_mode == "paired" else None
            active_segments = segments
            active_U = proj_list[g_idx]
            if scalar_split_meta is not None:
                base_scalar = (a_comp * b_comp).sum(dim=-1, keepdim=True)
                for p_idx, key_ir, alpha in scalar_split_meta["scalar_entries"]:  # type: ignore[index]
                    Wp = self.weight[int(p_idx)]
                    Wp_comp = Wp.to(dtype=compute_dtype) if Wp.dtype != compute_dtype else Wp
                    out_seg = torch.matmul(
                        (base_scalar * float(alpha)).movedim(-1, -2).contiguous(),
                        Wp_comp.transpose(0, 1),
                    ).movedim(-1, -2)
                    out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                    if weights is not None:
                        out_seg = out_seg * weights[..., int(p_idx), None, None]
                    out[int(key_ir)] = out[int(key_ir)] + out_seg
                active_segments = scalar_split_meta["rem_segments"]  # type: ignore[assignment]
                active_U = scalar_split_meta["U_rem"]  # type: ignore[assignment]
                if active_U is None:
                    continue
            used_fused_mix = False
            if self.channel_mode == "broadcast_rhs" and a_comp.dim() >= 2:
                B_flat = 1
                for s in batch_shape:
                    B_flat *= int(s)
                a_flat = a_comp.reshape(B_flat, self.channel_mul, 2 * l1 + 1)
                b_flat = b_comp.reshape(B_flat, 1, 2 * l2 + 1)
                if _USE_TRITON_CHANNELWISE_TP and _tp_fused_outer_proj_channel_mix is not None and len(segments) <= 16:
                    W_stack = _stack_group_weights(
                        w_param=self.weight,
                        segments=segments,
                        compute_dtype=compute_dtype,
                    ).to(device=a_comp.device)
                    fused_segments = [(int(p_idx), int(l3), int(s), int(e)) for p_idx, l3, s, e in segments]
                    out_buf = _tp_fused_outer_proj_channel_mix(
                        a_flat,
                        b_flat,
                        proj_list[g_idx],
                        W_stack,
                        fused_segments,
                        k_total,
                        self.mul_out,
                        2 * l1 + 1,
                        2 * l2 + 1,
                    )
                    if out_buf is not None:
                        out_buf = out_buf.to(dtype=dtype) if out_buf.dtype != dtype else out_buf
                        for seg_idx, (p_idx, l3, s, e) in enumerate(segments):
                            seg_out = out_buf[:, seg_idx, :, int(s): int(e)].reshape(*batch_shape, self.mul_out, int(e) - int(s))
                            if weights is not None:
                                seg_out = seg_out * weights[..., int(p_idx), None, None]
                            out[int(l3)] = out[int(l3)] + seg_out
                        used_fused_mix = True
                if used_fused_mix:
                    continue
                gates_all = (
                    weights.reshape(B_flat, weights.shape[-1]).to(dtype=compute_dtype)
                    if weights is not None
                    else None
                )
                if ensure_grouped_tp_cuda_ext_supported(
                    backend=self.ictd_tp_backend,
                    sample=a_flat,
                    compute_dtype=compute_dtype,
                    internal_weights=True,
                    weights=gates_all,
                ):
                    bucket_outputs: list[tuple[int, torch.Tensor]] = []
                    assert bucket_list is not None
                    for bucket in bucket_list[g_idx]:
                        bucket_segments = bucket["segments"]  # type: ignore[assignment]
                        bucket_path_indices = bucket["path_indices"]  # type: ignore[assignment]
                        U_bucket = bucket["U_bucket"]  # type: ignore[assignment]
                        kdim = int(bucket["kdim"])  # type: ignore[arg-type]
                        W_bucket = _stack_group_weights(
                            w_param=self.weight,
                            segments=bucket_segments,
                            compute_dtype=compute_dtype,
                        ).to(device=a_comp.device)
                        gates_bucket = None
                        if gates_all is not None:
                            gates_bucket = _slice_or_index_lastdim(gates_all, [int(p_idx) for p_idx in bucket_path_indices])
                        bucket_out = _tp_cuda_ext_bucket_forward(
                            backend=self.ictd_tp_backend,
                            a=a_flat,
                            b=b_flat,
                            U_bucket=U_bucket,
                            W_stack=W_bucket,
                            gates=gates_bucket,
                            compute_dtype=compute_dtype,
                        )
                        if bucket_out is None:
                            bucket_outputs = []
                            break
                        bucket_outputs.append(
                            (int(bucket_segments[0][1]), bucket_out.reshape(*batch_shape, self.mul_out, kdim))
                        )
                    if bucket_outputs:
                        for l3, bucket_out in bucket_outputs:
                            bucket_out = bucket_out.to(dtype=dtype) if bucket_out.dtype != dtype else bucket_out
                            out[l3] = out[l3] + bucket_out
                        continue
            m1 = 2 * l1 + 1
            m2 = 2 * l2 + 1
            if active_segments is segments and active_U is proj_list[g_idx]:
                view_meta = proj_view_list[g_idx]
                U3 = view_meta["U3"]  # type: ignore[assignment]
                seg_views = view_meta["seg_views"]  # type: ignore[assignment]
            else:
                U3 = active_U.reshape(m1, m2, -1)
                seg_views = [(seg, U3[:, :, int(seg[-2]): int(seg[-1])]) for seg in active_segments]
            if len(active_segments) <= 2:
                if self.channel_mode == "broadcast_rhs":
                    b_rhs = b_comp[..., 0, :]
                    for seg, U_seg in seg_views:
                        p_idx, l3, s, e = seg  # type: ignore[misc]
                        Wp = self.weight[int(p_idx)]
                        Wp_comp = self._weight_to_compute(Wp, compute_dtype)
                        out_seg = torch.einsum(
                            "...cm,...n,mnk,oc->...ok",
                            a_comp,
                            b_rhs,
                            U_seg,
                            Wp_comp,
                        )
                        out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                        if weights is not None:
                            out_seg = out_seg * weights[..., int(p_idx), None, None]
                        out[int(l3)] = out[int(l3)] + out_seg
                else:
                    for seg, U_seg in seg_views:
                        p_idx, l3, s, e = seg  # type: ignore[misc]
                        Wp = self.weight[int(p_idx)]
                        Wp_comp = self._weight_to_compute(Wp, compute_dtype)
                        out_seg = torch.einsum(
                            "...cm,...cn,mnk,oc->...ok",
                            a_comp,
                            b_comp,
                            U_seg,
                            Wp_comp,
                        )
                        out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                        if weights is not None:
                            out_seg = out_seg * weights[..., int(p_idx), None, None]
                        out[int(l3)] = out[int(l3)] + out_seg
                continue
            if self.channel_mode == "broadcast_rhs":
                b_rhs = b_comp[..., 0, :]
                y = torch.einsum("...cm,...n,mnk->...ck", a_comp, b_rhs, U3)
            else:
                y = torch.einsum("...cm,...cn,mnk->...ck", a_comp, b_comp, U3)
            if _USE_BUCKETED_CHANNELWISE_MIX:
                for bucket in bucket_list[g_idx]:
                    bucket_segments = bucket["segments"]  # type: ignore[assignment]
                    if active_segments is not segments:
                        continue
                    starts = [int(seg[-2]) for seg in bucket_segments]
                    kdim = int(bucket["kdim"])  # type: ignore[arg-type]
                    y_bucket = torch.stack([y[..., :, s : s + kdim] for s in starts], dim=-3)
                    W_bucket = _stack_group_weights(
                        w_param=self.weight,
                        segments=bucket_segments,
                        compute_dtype=compute_dtype,
                        mul_scale=self._mul_path_scale,
                    )
                    out_bucket = torch.einsum("...pck,poc->...pok", y_bucket, W_bucket)
                    out_bucket = out_bucket.to(dtype=dtype) if out_bucket.dtype != dtype else out_bucket
                    if weights is not None:
                        path_indices = [int(seg[0]) for seg in bucket_segments]
                        gates_bucket = _slice_or_index_lastdim(weights, path_indices)
                        out_bucket = out_bucket * gates_bucket[..., :, None, None]
                    out[int(bucket_segments[0][1])] = out[int(bucket_segments[0][1])] + out_bucket.sum(dim=-3)
            else:
                for p_idx, l3, s, e in active_segments:  # type: ignore[misc]
                    Wp = self.weight[int(p_idx)]
                    Wp_comp = self._weight_to_compute(Wp, compute_dtype)
                    y_seg = y[..., :, int(s): int(e)]
                    out_seg = torch.matmul(
                        y_seg.movedim(-1, -2).contiguous(),
                        Wp_comp.transpose(0, 1),
                    ).movedim(-1, -2)
                    out_seg = out_seg.to(dtype=dtype) if out_seg.dtype != dtype else out_seg
                    if weights is not None:
                        out_seg = out_seg * weights[..., int(p_idx), None, None]
                    out[int(l3)] = out[int(l3)] + out_seg
        return out
