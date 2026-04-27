"""
eSCN-inspired local SO(2) acceleration for pure_cartesian_ictd_layers.

This module keeps the public interface and readout stack of
`pure_cartesian_ictd_layers.PureCartesianICTDTransformerLayer`, but replaces
the expensive SO(3) edge tensor products in interaction layers 2..N with a
fast SO(2)-inspired equivariant approximation:

  1) split global ICTD SO(3) blocks by l
  2) keep the expensive local-frame basis changes available for O(2) paths and
     auxiliary contractions
  3) use a scalar-gated diagonal SO(2) surrogate in the main interaction path
  4) use a fast scalar-powered surrogate for SO(2) multiple-contraction readout

Conv1 remains the original scalar -> SO(3) ICTD embedding layer to minimize
integration risk and preserve the original external-field / physical-head stack.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from molecular_force_field.models.pure_cartesian_ictd_layers import (
    EquivariantChannelLinearSO3,
    EquivariantChannelLinearSO3Rect,
    PureCartesianICTDTransformerLayer,
    _merge_irreps,
    _split_irreps,
    _sym_rank_linear_permutation_indices,
)
from molecular_force_field.models.ictd_irreps import (
    build_harmonic_projectors,
    build_harmonic_reconstructors,
    direction_harmonics_fast,
)
from molecular_force_field.models.ictd_irreps_2d import (
    _build_cg_tensor_o2,
    _normalize_o2_irrep_key,
    _o2_irrep_dim,
    _o2_valid_output_keys,
    build_cg_tensor_so2,
    parse_o2_active_irreps,
    so2_irrep_dim,
)

def _apply_channel_linear(block: torch.Tensor, linear: nn.Module) -> torch.Tensor:
    if isinstance(linear, nn.Identity):
        return block
    param = next(linear.parameters(), None)
    linear_dtype = block.dtype if param is None else param.dtype
    y = linear(block.to(dtype=linear_dtype).movedim(-2, -1))
    return y.movedim(-1, -2).to(dtype=block.dtype)


@lru_cache(maxsize=None)
def _sym_to_full_cpu(L: int) -> torch.Tensor:
    perms, coefs = _sym_rank_linear_permutation_indices(int(L))
    nfull = 1 if int(L) == 0 else (3 ** int(L))
    H = torch.zeros(len(perms), nfull, dtype=torch.float64)
    if int(L) == 0:
        H[0, 0] = 1.0
        return H
    for j, idx in enumerate(perms):
        H[j, idx] = 1.0 / float(coefs[j].item())
    return H.contiguous()


@lru_cache(maxsize=None)
def _full_to_sym_cpu(L: int) -> torch.Tensor:
    perms, _ = _sym_rank_linear_permutation_indices(int(L))
    nfull = 1 if int(L) == 0 else (3 ** int(L))
    K = torch.zeros(nfull, len(perms), dtype=torch.float64)
    if int(L) == 0:
        K[0, 0] = 1.0
        return K
    for j, idx in enumerate(perms):
        K[idx, j] = 1.0
    return K.contiguous()


@lru_cache(maxsize=None)
def _harmonic_to_full_cpu(l: int) -> torch.Tensor:
    l = int(l)
    recon = build_harmonic_reconstructors(l)
    return (recon.V[(l, l)].T @ _sym_to_full_cpu(l)).contiguous()


@lru_cache(maxsize=None)
def _full_to_harmonic_cpu(l: int) -> torch.Tensor:
    l = int(l)
    proj = build_harmonic_projectors(l)
    return (_full_to_sym_cpu(l) @ proj.P[(l, l)].T).contiguous()


def _kron_power_rt(R: torch.Tensor, L: int) -> torch.Tensor:
    L = int(L)
    if L < 0:
        raise ValueError(f"L must be >= 0, got {L}")
    if L == 0:
        shape = (*R.shape[:-2], 1, 1)
        return torch.ones(shape, device=R.device, dtype=R.dtype)
    K = R.transpose(-1, -2)
    for _ in range(L - 1):
        K = torch.einsum("...ab,...cd->...acbd", K, R.transpose(-1, -2))
        K = K.reshape(*K.shape[:-4], K.shape[-4] * K.shape[-3], K.shape[-2] * K.shape[-1])
    return K.contiguous()


_harmonic_to_full_cache: Dict[Tuple[str, str, int], torch.Tensor] = {}
_full_to_harmonic_cache: Dict[Tuple[str, str, int], torch.Tensor] = {}


def _get_harmonic_full_maps(
    l: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (str(device), str(dtype), int(l))
    H = _harmonic_to_full_cache.get(key)
    C = _full_to_harmonic_cache.get(key)
    if H is None:
        H = _harmonic_to_full_cpu(int(l)).to(device=device, dtype=dtype)
        _harmonic_to_full_cache[key] = H
    if C is None:
        C = _full_to_harmonic_cpu(int(l)).to(device=device, dtype=dtype)
        _full_to_harmonic_cache[key] = C
    return H, C


def _harmonic_row_rotation(R: torch.Tensor, l: int) -> torch.Tensor:
    l = int(l)
    if l == 0:
        shape = (*R.shape[:-2], 1, 1)
        return torch.ones(shape, device=R.device, dtype=R.dtype)
    H, C = _get_harmonic_full_maps(l, R.device, R.dtype)
    K = _kron_power_rt(R, l)
    return torch.einsum("ab,...bc,cd->...ad", H, K, C).contiguous()


def _z_rotation(angle: float) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )


@lru_cache(maxsize=None)
def _so3_local_so2_basis_cpu(l: int) -> Tuple[torch.Tensor, Tuple[Tuple[int, int], ...]]:
    """
    Return Q_l such that x_freq = x_ictd @ Q_l groups the l irrep into
    [m=0 | m=1 pair | ... | m=l pair].
    """
    l = int(l)
    if l == 0:
        return torch.ones(1, 1, dtype=torch.float64), ((0, 1),)

    phi = 0.1
    D = _harmonic_row_rotation(_z_rotation(phi), l).to(torch.float64)
    S = 0.5 * (D + D.T)
    evals, evecs = torch.linalg.eigh(S)

    cols: List[torch.Tensor] = []
    slices: List[Tuple[int, int]] = []
    offset = 0
    for m in range(l + 1):
        target = math.cos(m * phi)
        inds = torch.nonzero((evals - target).abs() < 1e-6).flatten()
        if m == 0:
            if inds.numel() != 1:
                raise RuntimeError(f"Failed to isolate m=0 eigenspace for l={l}")
            q = evecs[:, inds[0]]
            lead = int(torch.argmax(q.abs()).item())
            if float(q[lead].item()) < 0.0:
                q = -q
            cols.append(q)
            slices.append((offset, offset + 1))
            offset += 1
            continue

        if inds.numel() != 2:
            raise RuntimeError(f"Failed to isolate m={m} eigenspace for l={l}")
        q1 = evecs[:, inds[0]]
        s = math.sin(m * phi)
        q2 = (D @ q1 - target * q1) / s
        q2 = q2 - q1 * (q1 @ q2)
        q2 = q2 / q2.norm()
        A = torch.stack([q1, q2], dim=1).T @ D @ torch.stack([q1, q2], dim=1)
        if float(A[0, 1].item()) < 0.0:
            q2 = -q2
        cols.extend([q1, q2])
        slices.append((offset, offset + 2))
        offset += 2

    Q = torch.stack(cols, dim=1).contiguous()
    return Q, tuple(slices)


_local_q_cache: Dict[Tuple[str, str, int], torch.Tensor] = {}


def _get_local_q(l: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), str(dtype), int(l))
    Q = _local_q_cache.get(key)
    if Q is None:
        Q = _so3_local_so2_basis_cpu(int(l))[0].to(device=device, dtype=dtype)
        _local_q_cache[key] = Q
    return Q


def _reference_axis_from_so3_blocks(
    blocks: Dict[int, torch.Tensor],
    lmax: int,
    *,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    sample = next(iter(blocks.values()))
    device = sample.device
    if int(lmax) < 1:
        return torch.tensor([0.0, 0.0, 1.0], device=device, dtype=compute_dtype).expand(sample.shape[0], 3)
    v = blocks[1].to(dtype=compute_dtype).mean(dim=-2)
    n = torch.stack([v[..., 2], v[..., 1], v[..., 0]], dim=-1)
    norm = n.norm(dim=-1, keepdim=True)
    fallback = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=compute_dtype).expand_as(n)
    return torch.where(norm > 1e-8, n / norm.clamp(min=1e-8), fallback)


def _so3_blocks_to_local_freq(
    blocks: Dict[int, torch.Tensor],
    G: torch.Tensor,
    lmax: int,
    *,
    compute_dtype: torch.dtype,
) -> Dict[Tuple[int, int], torch.Tensor]:
    local: Dict[Tuple[int, int], torch.Tensor] = {}
    device = G.device
    for l in range(int(lmax) + 1):
        blk = blocks[l].to(dtype=compute_dtype)
        if l == 0:
            blk_local = blk
        elif l in (1, 2):
            H, C = _get_harmonic_full_maps(l, device, compute_dtype)
            full = torch.matmul(blk, H)
            if l == 1:
                vec_local = torch.einsum("...ai,...ci->...ca", G.to(dtype=compute_dtype), full)
                blk_local = torch.matmul(vec_local, C)
            else:
                tens = full.reshape(*full.shape[:-1], 3, 3)
                tens_local = torch.einsum("...ai,...cij,...bj->...cab", G.to(dtype=compute_dtype), tens, G.to(dtype=compute_dtype))
                blk_local = torch.matmul(tens_local.reshape(*tens_local.shape[:-2], 9), C)
        else:
            D = _harmonic_row_rotation(G.to(dtype=compute_dtype), l)
            blk_local = torch.matmul(blk, D)
        Q = _get_local_q(l, device, compute_dtype)
        blk_freq = torch.matmul(blk_local, Q)
        for m, (s, e) in enumerate(_so3_local_so2_basis_cpu(l)[1]):
            local[(l, m)] = blk_freq[..., s:e]
    return local


def _local_freq_to_so3_blocks(
    pieces_by_lm: Dict[Tuple[int, int], torch.Tensor],
    G: torch.Tensor,
    lmax: int,
    *,
    out_dtype: torch.dtype,
    compute_dtype: torch.dtype,
) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    device = G.device
    for l in range(int(lmax) + 1):
        blk_freq = torch.cat([pieces_by_lm[(l, m)] for m in range(l + 1)], dim=-1).to(dtype=compute_dtype)
        Q = _get_local_q(l, device, compute_dtype)
        blk_local = torch.matmul(blk_freq, Q.transpose(-1, -2))
        if l == 0:
            blk_global = blk_local
        elif l in (1, 2):
            H, C = _get_harmonic_full_maps(l, device, compute_dtype)
            full_local = torch.matmul(blk_local, H)
            GT = G.transpose(-1, -2).to(dtype=compute_dtype)
            if l == 1:
                vec_global = torch.einsum("...ia,...ca->...ci", GT, full_local)
                blk_global = torch.matmul(vec_global, C)
            else:
                tens_local = full_local.reshape(*full_local.shape[:-1], 3, 3)
                tens_global = torch.einsum("...ia,...cab,...jb->...cij", GT, tens_local, GT)
                blk_global = torch.matmul(tens_global.reshape(*tens_global.shape[:-2], 9), C)
        else:
            D = _harmonic_row_rotation(G.to(dtype=compute_dtype), l)
            blk_global = torch.matmul(blk_local, D.transpose(-1, -2))
        out[l] = blk_global.to(dtype=out_dtype)
    return out


def _build_edge_local_frames(n: torch.Tensor) -> torch.Tensor:
    """
    Global->local frame matrix G such that G @ n = e_z.
    """
    z_axis = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ref_a = torch.tensor([0.0, 0.0, 1.0], device=n.device, dtype=n.dtype).expand_as(z_axis)
    ref_b = torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype).expand_as(z_axis)
    use_b = z_axis[..., 2].abs() > 0.9
    ref = torch.where(use_b.unsqueeze(-1), ref_b, ref_a)
    x_axis = torch.cross(ref, z_axis, dim=-1)
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    det = torch.det(torch.stack([x_axis, y_axis, z_axis], dim=-1))
    flip = det < 0.0
    if bool(flip.any()):
        x_axis = torch.where(flip.unsqueeze(-1), -x_axis, x_axis)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    local_to_global = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return local_to_global.transpose(-1, -2).contiguous()


def _normalize_local_so2_cg(C: torch.Tensor, output_dim: int, normalization: str) -> torch.Tensor:
    C_fn = C.norm().item()
    if normalization == "component" and C_fn > 1e-30:
        return C * (math.sqrt(output_dim) / C_fn)
    if normalization == "norm" and C_fn > 1e-30:
        return C * (1.0 / C_fn)
    return C


def _so2_valid_outputs_local(m1: int, m2: int, mmax: int | None = None) -> List[int]:
    outs: List[int] = []
    for cand in (abs(int(m1) - int(m2)), int(m1) + int(m2)):
        if mmax is not None and cand > int(mmax):
            continue
        if cand not in outs:
            outs.append(cand)
    return outs


class HarmonicLocalSO2ContractionTensorProduct(nn.Module):
    """
    Local-frame bilinear SO(2) tensor product for flattened SO(3) ICTD blocks.

    A node-local reference axis is extracted from the l=1 block of the second
    input. Features are rotated into that local frame, regrouped by SO(2)
    frequency m, multiplied via SO(2) CG tensors, mixed across contributor
    l-blocks, and rotated back to the global SO(3) ICTD basis.
    """

    def __init__(
        self,
        channels: int,
        lmax: int,
        *,
        path_policy: str = "full",
        max_rank_other: int | None = None,
        normalization: str = "component",
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self._normalization = str(normalization)
        self.internal_compute_dtype = torch.get_default_dtype() if internal_compute_dtype is None else internal_compute_dtype

        self._contributors_by_m: Dict[int, List[int]] = {
            m: [l for l in range(m, self.lmax + 1)]
            for m in range(self.lmax + 1)
        }
        self._segments_per_m = {m: len(ls) for m, ls in self._contributors_by_m.items()}

        all_paths: List[Tuple[int, int, int]] = []
        for m1 in range(self.lmax + 1):
            for m2 in range(self.lmax + 1):
                for m3 in _so2_valid_outputs_local(m1, m2, mmax=self.lmax):
                    all_paths.append((m1, m2, m3))
        if path_policy == "full":
            self.paths = all_paths
        elif path_policy == "max_rank_other":
            if max_rank_other is None:
                raise ValueError("path_policy='max_rank_other' requires max_rank_other")
            self.paths = [p for p in all_paths if min(p[0], p[1]) <= int(max_rank_other)]
        else:
            raise ValueError(f"Unknown path_policy={path_policy!r}")
        self.num_paths = len(self.paths)

        self.path_weights = nn.Parameter(torch.randn(self.num_paths, self.channels) * 0.02)
        self._paths_by_out_m: Dict[int, List[Tuple[int, int, int, int]]] = {m: [] for m in range(self.lmax + 1)}
        for p_idx, (m1, m2, m3) in enumerate(self.paths):
            self._paths_by_out_m[m3].append((p_idx, m1, m2, m3))

        self.mixers = nn.ModuleDict(
            {
                str(m): nn.Linear(self.channels, self.channels, bias=False)
                for m in range(self.lmax + 1)
                if len(self._contributors_by_m[m]) > 0
            }
        )

        self._cg_cpu_f64: List[torch.Tensor] = [
            _normalize_local_so2_cg(build_cg_tensor_so2(m1, m2, m3), so2_irrep_dim(m3), self._normalization)
            for (m1, m2, m3) in self.paths
        ]
        self._cg_cache_by_dev_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}

    def _get_cg_list(self, device: torch.device) -> List[torch.Tensor]:
        key = (str(device), str(self.internal_compute_dtype))
        cached = self._cg_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        cg_list = [C.to(device=device, dtype=self.internal_compute_dtype) for C in self._cg_cpu_f64]
        self._cg_cache_by_dev_dtype[key] = cg_list
        return cg_list

    def forward(self, x1: Dict[int, torch.Tensor], x2: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        sample = next(iter(x1.values()))
        device = sample.device
        dtype = sample.dtype
        cg_list = self._get_cg_list(device)
        axis = _reference_axis_from_so3_blocks(x2, self.lmax, compute_dtype=self.internal_compute_dtype)
        G = _build_edge_local_frames(axis)
        local1 = _so3_blocks_to_local_freq(x1, G, self.lmax, compute_dtype=self.internal_compute_dtype)
        local2 = _so3_blocks_to_local_freq(x2, G, self.lmax, compute_dtype=self.internal_compute_dtype)

        packed1 = {
            m: torch.stack([local1[(l, m)] for l in self._contributors_by_m[m]], dim=1)
            for m in range(self.lmax + 1)
        }
        packed2 = {
            m: torch.stack([local2[(l, m)] for l in self._contributors_by_m[m]], dim=1)
            for m in range(self.lmax + 1)
        }

        pieces_by_lm: Dict[Tuple[int, int], torch.Tensor] = {}
        path_w = self.path_weights.to(device=device, dtype=self.internal_compute_dtype)
        batch_shape = sample.shape[:-2]
        for m3 in range(self.lmax + 1):
            out_segments = self._contributors_by_m[m3]
            dim3 = so2_irrep_dim(m3)
            agg = torch.zeros(
                *batch_shape,
                len(out_segments),
                self.channels,
                dim3,
                device=device,
                dtype=self.internal_compute_dtype,
            )
            for p_idx, m1, m2, _ in self._paths_by_out_m[m3]:
                valid_pos = [i for i, l in enumerate(out_segments) if l >= m1 and l >= m2]
                if not valid_pos:
                    continue
                idx1 = torch.tensor([out_segments[i] - m1 for i in valid_pos], device=device, dtype=torch.long)
                idx2 = torch.tensor([out_segments[i] - m2 for i in valid_pos], device=device, dtype=torch.long)
                pos = torch.tensor(valid_pos, device=device, dtype=torch.long)
                C = cg_list[p_idx]
                w = path_w[p_idx].view(*([1] * len(batch_shape)), 1, self.channels, 1)
                y = torch.einsum(
                    "...scm,...scn,mnk->...sck",
                    packed1[m1].index_select(-3, idx1),
                    packed2[m2].index_select(-3, idx2),
                    C,
                )
                agg[..., pos, :, :] = agg[..., pos, :, :] + y * w

            mixed = _apply_channel_linear(agg, self.mixers[str(m3)])
            for seg_idx, l in enumerate(out_segments):
                pieces_by_lm[(l, m3)] = mixed[..., seg_idx, :, :]

        return _local_freq_to_so3_blocks(
            pieces_by_lm,
            G,
            self.lmax,
            out_dtype=dtype,
            compute_dtype=self.internal_compute_dtype,
        )


class LocalMultipleContractionSO2(nn.Module):
    """
    Fast higher-order surrogate for the SO(2) readout path.

    The public SO(3) flat layout is preserved, but higher orders are generated
    by repeatedly gating each irrep block with the hidden scalar block. This
    keeps the readout strictly equivariant while removing the expensive local
    frequency tensor products from the hot path.
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
        normalization: str = "component",
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.lmax = int(lmax)
        self.correlation = int(correlation)
        if self.correlation < 1:
            raise ValueError(f"correlation must be >= 1, got {self.correlation}")

        self.reduce = EquivariantChannelLinearSO3Rect(
            self.in_channels,
            self.hidden_channels,
            self.lmax,
            bias=False,
        )
        self.order_mix = nn.ModuleList(
            [
                EquivariantChannelLinearSO3(self.hidden_channels, self.lmax, bias=False)
                for _ in range(self.correlation)
            ]
        )
        self.tp_layers = nn.ModuleList(
            [
                HarmonicLocalSO2ContractionTensorProduct(
                    channels=self.hidden_channels,
                    lmax=self.lmax,
                    path_policy=ictd_tp_path_policy,
                    max_rank_other=ictd_tp_max_rank_other,
                    normalization=normalization,
                    internal_compute_dtype=internal_compute_dtype,
                )
                for _ in range(max(self.correlation - 1, 0))
            ]
        )
        self.out_linear = EquivariantChannelLinearSO3(
            self.hidden_channels,
            self.lmax,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.reduce(x)
        accum = self.order_mix[0](base)
        if self.correlation == 1:
            return self.out_linear(accum)

        base_blocks = _split_irreps(base, self.hidden_channels, self.lmax)
        scalar = base_blocks[0]
        current_blocks = base_blocks
        for order_idx, _tp in enumerate(self.tp_layers, start=1):
            current_blocks = {
                l: current_blocks[l] * scalar
                for l in range(self.lmax + 1)
            }
            current_flat = _merge_irreps(current_blocks, self.hidden_channels, self.lmax)
            current_flat = self.order_mix[order_idx](current_flat)
            accum = accum + current_flat
        return self.out_linear(accum)


class HarmonicLocalO2ContractionTensorProduct(nn.Module):
    """
    Local-frame O(2) tensor product used inside the O(2) contraction stack.

    The hidden local state keeps canonical SO(3)->O(2) blocks:
      - 0e from every local m=0 slice
      - m=1..lmax frequency blocks
      - an extra latent 0o block that is absent in one-body SO(3) features but
        can be generated and consumed by higher-order O(2) products.
    """

    def __init__(
        self,
        channels: int,
        lmax: int,
        *,
        path_policy: str = "full",
        max_rank_other: int | None = None,
        normalization: str = "component",
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self._normalization = str(normalization)
        self.internal_compute_dtype = torch.get_default_dtype() if internal_compute_dtype is None else internal_compute_dtype

        self.hidden_active_irreps = parse_o2_active_irreps(["0e", "0o"] + [str(m) for m in range(1, self.lmax + 1)])
        self._contributors_by_m: Dict[int, List[int]] = {
            m: [l for l in range(m, self.lmax + 1)]
            for m in range(self.lmax + 1)
        }
        self._num_segments_by_key: Dict[Tuple[str, int], int] = {
            ("scalar", 1): len(self._contributors_by_m[0]),
            ("scalar", -1): len(self._contributors_by_m[0]),
        }
        for m in range(1, self.lmax + 1):
            self._num_segments_by_key[("freq", m)] = len(self._contributors_by_m[m])

        all_paths: List[Tuple[Tuple[str, int], Tuple[str, int], Tuple[str, int]]] = []
        for key1 in self.hidden_active_irreps:
            for key2 in self.hidden_active_irreps:
                for key3 in _o2_valid_output_keys(key1, key2):
                    if key3 in self.hidden_active_irreps:
                        all_paths.append((key1, key2, key3))
        if path_policy == "full":
            self.paths = all_paths
        elif path_policy == "max_rank_other":
            if max_rank_other is None:
                raise ValueError("path_policy='max_rank_other' requires max_rank_other")

            def _rank(key: Tuple[str, int]) -> int:
                return 0 if key[0] == "scalar" else key[1]

            self.paths = [p for p in all_paths if min(_rank(p[0]), _rank(p[1])) <= int(max_rank_other)]
        else:
            raise ValueError(f"Unknown path_policy={path_policy!r}")
        self.num_paths = len(self.paths)
        self.path_weights = nn.Parameter(torch.randn(self.num_paths, self.channels) * 0.02)

        self._paths_by_out_key: Dict[Tuple[str, int], List[Tuple[int, Tuple[str, int], Tuple[str, int], Tuple[str, int]]]] = {
            key: [] for key in self.hidden_active_irreps
        }
        for p_idx, (key1, key2, key3) in enumerate(self.paths):
            self._paths_by_out_key[key3].append((p_idx, key1, key2, key3))

        self.mixers = nn.ModuleDict(
            {
                f"{key3[0]}_{key3[1]}": nn.Linear(self.channels, self.channels, bias=False)
                for key3 in self.hidden_active_irreps
                if self._num_segments_by_key[key3] > 0
            }
        )

        self._cg_cpu_f64: List[torch.Tensor] = [
            _normalize_local_so2_cg(
                _build_cg_tensor_o2(key1, key2, key3),
                _o2_irrep_dim(key3),
                self._normalization,
            )
            for (key1, key2, key3) in self.paths
        ]
        self._cg_cache_by_dev_dtype: Dict[Tuple[str, str], List[torch.Tensor]] = {}

    def _get_cg_list(self, device: torch.device) -> List[torch.Tensor]:
        key = (str(device), str(self.internal_compute_dtype))
        cached = self._cg_cache_by_dev_dtype.get(key)
        if cached is not None:
            return cached
        cg_list = [C.to(device=device, dtype=self.internal_compute_dtype) for C in self._cg_cpu_f64]
        self._cg_cache_by_dev_dtype[key] = cg_list
        return cg_list

    def _zero_state_like(self, sample: torch.Tensor) -> Dict[Tuple[str, int], torch.Tensor]:
        batch_shape = sample.shape[:-2]
        device = sample.device
        return {
            key: torch.zeros(
                *batch_shape,
                self.channels * self._num_segments_by_key[key],
                _o2_irrep_dim(key),
                device=device,
                dtype=self.internal_compute_dtype,
            )
            for key in self.hidden_active_irreps
        }

    @staticmethod
    def _rank_from_key(key: Tuple[str, int]) -> int:
        return 0 if key[0] == "scalar" else key[1]

    def _contributors_for_key(self, key: Tuple[str, int]) -> List[int]:
        return self._contributors_by_m[self._rank_from_key(key)]

    def forward(
        self,
        x1: Dict[Tuple[str, int], torch.Tensor],
        x2: Dict[Tuple[str, int], torch.Tensor],
    ) -> Dict[Tuple[str, int], torch.Tensor]:
        sample = next(iter(x1.values()))
        device = sample.device
        batch_shape = sample.shape[:-2]
        cg_list = self._get_cg_list(device)
        path_w = self.path_weights.to(device=device, dtype=self.internal_compute_dtype)
        packed1 = {
            key: x1[key].to(dtype=self.internal_compute_dtype).reshape(
                *batch_shape,
                self._num_segments_by_key[key],
                self.channels,
                _o2_irrep_dim(key),
            )
            for key in self.hidden_active_irreps
        }
        packed2 = {
            key: x2[key].to(dtype=self.internal_compute_dtype).reshape(
                *batch_shape,
                self._num_segments_by_key[key],
                self.channels,
                _o2_irrep_dim(key),
            )
            for key in self.hidden_active_irreps
        }

        out: Dict[Tuple[str, int], torch.Tensor] = {}
        for key3 in self.hidden_active_irreps:
            dim3 = _o2_irrep_dim(key3)
            contributors = self._contributors_for_key(key3)
            agg = torch.zeros(
                *batch_shape,
                len(contributors),
                self.channels,
                dim3,
                device=device,
                dtype=self.internal_compute_dtype,
            )
            for p_idx, key1, key2, _ in self._paths_by_out_key[key3]:
                rank1 = self._rank_from_key(key1)
                rank2 = self._rank_from_key(key2)
                valid_pos = [i for i, l in enumerate(contributors) if l >= rank1 and l >= rank2]
                if not valid_pos:
                    continue
                idx1 = torch.tensor([contributors[i] - rank1 for i in valid_pos], device=device, dtype=torch.long)
                idx2 = torch.tensor([contributors[i] - rank2 for i in valid_pos], device=device, dtype=torch.long)
                pos = torch.tensor(valid_pos, device=device, dtype=torch.long)
                C = cg_list[p_idx]
                w = path_w[p_idx].view(*([1] * len(batch_shape)), 1, self.channels, 1)
                y = torch.einsum(
                    "...scm,...scn,mnk->...sck",
                    packed1[key1].index_select(-3, idx1),
                    packed2[key2].index_select(-3, idx2),
                    C,
                )
                agg[..., pos, :, :] = agg[..., pos, :, :] + y * w

            mixed = _apply_channel_linear(agg, self.mixers[f"{key3[0]}_{key3[1]}"])
            out[key3] = mixed.reshape(*batch_shape, self.channels * len(contributors), dim3).to(dtype=sample.dtype)
        return out


class LocalMultipleContractionO2(nn.Module):
    """
    O(2)-complete local contraction stack for flattened SO(3) ICTD features.

    Unlike the SO(2) version, the hidden local state explicitly carries a 0o
    branch so higher-order contractions can use O(2)-odd scalar paths.
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
        normalization: str = "component",
        internal_compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.lmax = int(lmax)
        self.correlation = int(correlation)
        if self.correlation < 1:
            raise ValueError(f"correlation must be >= 1, got {self.correlation}")

        self.reduce = EquivariantChannelLinearSO3Rect(
            self.in_channels,
            self.hidden_channels,
            self.lmax,
            bias=False,
        )
        self.order_mix = nn.ModuleList(
            [
                EquivariantChannelLinearSO3(self.hidden_channels, self.lmax, bias=False)
                for _ in range(self.correlation)
            ]
        )
        self.tp_layers = nn.ModuleList(
            [
                HarmonicLocalO2ContractionTensorProduct(
                    channels=self.hidden_channels,
                    lmax=self.lmax,
                    path_policy=ictd_tp_path_policy,
                    max_rank_other=ictd_tp_max_rank_other,
                    normalization=normalization,
                    internal_compute_dtype=internal_compute_dtype,
                )
                for _ in range(max(self.correlation - 1, 0))
            ]
        )
        self.out_linear = EquivariantChannelLinearSO3(
            self.hidden_channels,
            self.lmax,
            bias=False,
        )

        self.hidden_active_irreps = parse_o2_active_irreps(["0e", "0o"] + [str(m) for m in range(1, self.lmax + 1)])
        self._contributors_by_m: Dict[int, List[int]] = {
            m: [l for l in range(m, self.lmax + 1)]
            for m in range(self.lmax + 1)
        }

    def _local_freq_to_hidden_o2(self, local_freq: Dict[Tuple[int, int], torch.Tensor]) -> Dict[Tuple[str, int], torch.Tensor]:
        sample = next(iter(local_freq.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        out: Dict[Tuple[str, int], torch.Tensor] = {}

        zero_shape = (*batch_shape, self.hidden_channels * len(self._contributors_by_m[0]), 1)
        out[("scalar", -1)] = torch.zeros(*zero_shape, device=device, dtype=dtype)
        out[("scalar", 1)] = torch.cat([local_freq[(l, 0)] for l in self._contributors_by_m[0]], dim=-2)
        for m in range(1, self.lmax + 1):
            out[("freq", m)] = torch.cat([local_freq[(l, m)] for l in self._contributors_by_m[m]], dim=-2)
        return out

    def _hidden_o2_to_blocks(
        self,
        hidden: Dict[Tuple[str, int], torch.Tensor],
        G: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> Dict[int, torch.Tensor]:
        pieces_by_lm: Dict[Tuple[int, int], torch.Tensor] = {}
        compute_dtype = self.tp_layers[0].internal_compute_dtype if len(self.tp_layers) > 0 else self.reduce.adapters["0"].weight.dtype

        scalar_even = hidden[("scalar", 1)].to(dtype=compute_dtype)
        step0 = self.hidden_channels
        for seg_idx, l in enumerate(self._contributors_by_m[0]):
            s = seg_idx * step0
            e = s + step0
            pieces_by_lm[(l, 0)] = scalar_even[..., s:e, :]

        for m in range(1, self.lmax + 1):
            blk = hidden[("freq", m)].to(dtype=compute_dtype)
            step = self.hidden_channels
            for seg_idx, l in enumerate(self._contributors_by_m[m]):
                s = seg_idx * step
                e = s + step
                pieces_by_lm[(l, m)] = blk[..., s:e, :]
        return _local_freq_to_so3_blocks(
            pieces_by_lm,
            G,
            self.lmax,
            out_dtype=out_dtype,
            compute_dtype=compute_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.reduce(x)
        accum = self.order_mix[0](base)
        if self.correlation == 1:
            return self.out_linear(accum)

        base_blocks = _split_irreps(base, self.hidden_channels, self.lmax)
        compute_dtype = self.tp_layers[0].internal_compute_dtype if len(self.tp_layers) > 0 else base.dtype
        axis = _reference_axis_from_so3_blocks(base_blocks, self.lmax, compute_dtype=compute_dtype)
        G = _build_edge_local_frames(axis)
        base_local = _so3_blocks_to_local_freq(base_blocks, G, self.lmax, compute_dtype=compute_dtype)
        base_hidden = self._local_freq_to_hidden_o2(base_local)
        current_hidden = base_hidden

        for order_idx, tp in enumerate(self.tp_layers, start=1):
            current_hidden = tp(current_hidden, base_hidden)
            current_blocks = self._hidden_o2_to_blocks(current_hidden, G, base.dtype)
            current_flat = _merge_irreps(current_blocks, self.hidden_channels, self.lmax)
            current_flat = self.order_mix[order_idx](current_flat)
            accum = accum + current_flat
        return self.out_linear(accum)


class HarmonicLocalSO2TensorProduct(nn.Module):
    """
    Fast SO(2)-inspired diagonal mixer with the same call shape as the ICTD TP modules.

    Inputs:
      x1: dict l -> (E, C, 2l+1)
      x2: dict l -> kept for API compatibility
      weights: optional per-edge gates of shape (E, lmax+1)

    Output:
      dict l -> (E, C, 2l+1)
    """

    def __init__(self, channels: int, lmax: int, *, internal_compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.internal_compute_dtype = torch.get_default_dtype() if internal_compute_dtype is None else internal_compute_dtype
        self.num_paths = self.lmax + 1

        self._contributors_by_m: Dict[int, List[int]] = {
            m: [l for l in range(m, self.lmax + 1)]
            for m in range(self.lmax + 1)
        }
        self.mixers = nn.ModuleDict(
            {
                str(m): nn.Linear(
                    self.channels,
                    self.channels,
                    bias=False,
                )
                for m in range(self.lmax + 1)
            }
        )

    def forward(
        self,
        x1: Dict[int, torch.Tensor],
        x2: Dict[int, torch.Tensor],
        weights: torch.Tensor | None = None,
    ) -> Dict[int, torch.Tensor]:
        for l in range(self.lmax + 1):
            if str(l) not in self.mixers:
                self.mixers[str(l)] = nn.Linear(self.channels, self.channels, bias=False)
        out: Dict[int, torch.Tensor] = {}
        dtype = next(iter(x1.values())).dtype
        compute_dtype = self.internal_compute_dtype
        del x2
        for l in range(self.lmax + 1):
            mixed = _apply_channel_linear(x1[l].to(dtype=compute_dtype), self.mixers[str(l)])
            if weights is not None:
                gate = weights[..., : l + 1].to(dtype=compute_dtype).mean(dim=-1, keepdim=True).view(*weights.shape[:-1], 1, 1)
                mixed = mixed * gate
            out[l] = mixed.to(dtype=dtype)
        return out


class PureCartesianICTDSO2TransformerLayer(PureCartesianICTDTransformerLayer):
    """
    Drop-in variant of PureCartesianICTDTransformerLayer whose interaction
    layers 2..N use local-frame SO(2) mixing instead of full ICTD SO(3) TP.
    """

    def __init__(self, *args, internal_compute_dtype: torch.dtype | None = None, **kwargs):
        ictd_tp_path_policy = kwargs.get("ictd_tp_path_policy", "full")
        ictd_tp_max_rank_other = kwargs.get("ictd_tp_max_rank_other", None)
        super().__init__(*args, internal_compute_dtype=internal_compute_dtype, **kwargs)
        self.local_group = "so2"
        self._local_ictd_tp_path_policy = ictd_tp_path_policy
        self._local_ictd_tp_max_rank_other = ictd_tp_max_rank_other

        new_tp_layers = nn.ModuleList()
        new_fc_layers = nn.ModuleList()
        for _ in range(max(self.num_interaction - 1, 0)):
            tp2 = HarmonicLocalSO2TensorProduct(
                channels=self.channels,
                lmax=self.lmax,
                internal_compute_dtype=internal_compute_dtype,
            )
            fc2 = nn.Sequential(
                nn.Linear(self.number_of_basis, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, tp2.num_paths),
            )
            new_tp_layers.append(tp2)
            new_fc_layers.append(fc2)
        self.tp2_layers = new_tp_layers
        self.fc2_layers = new_fc_layers

        if self.save_readout_mode == "multiple-contraction":
            self.multiple_contraction_last = LocalMultipleContractionSO2(
                in_channels=self.channels,
                hidden_channels=self.channels,
                lmax=self.lmax,
                correlation=self.save_contraction_order,
                ictd_tp_path_policy=ictd_tp_path_policy,
                ictd_tp_max_rank_other=ictd_tp_max_rank_other,
                internal_compute_dtype=internal_compute_dtype,
            )
            self.multiple_contraction_mix = LocalMultipleContractionSO2(
                in_channels=self.channels * self.num_interaction,
                hidden_channels=self.save_multiple_mix_channels,
                lmax=self.lmax,
                correlation=self.save_contraction_order,
                ictd_tp_path_policy=ictd_tp_path_policy,
                ictd_tp_max_rank_other=ictd_tp_max_rank_other,
                internal_compute_dtype=internal_compute_dtype,
            )
