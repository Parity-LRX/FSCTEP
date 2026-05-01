from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from molecular_force_field.models.ictd_irreps import (
    build_harmonic_projectors,
    build_harmonic_reconstructors,
)
from molecular_force_field.models.ictd_irreps_2d import so2_irrep_dim
from molecular_force_field.models.pure_cartesian_ictd_layers import (
    _sym_rank_linear_permutation_indices,
)
from molecular_force_field.utils.scatter import scatter


def apply_channel_linear(block: torch.Tensor, linear: nn.Module) -> torch.Tensor:
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
    if L == 0:
        return torch.ones((*R.shape[:-2], 1, 1), device=R.device, dtype=R.dtype)
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


def harmonic_row_rotation(R: torch.Tensor, l: int) -> torch.Tensor:
    l = int(l)
    if l == 0:
        return torch.ones((*R.shape[:-2], 1, 1), device=R.device, dtype=R.dtype)
    H, C = _get_harmonic_full_maps(l, R.device, R.dtype)
    K = _kron_power_rt(R, l)
    return torch.einsum("ab,...bc,cd->...ad", H, K, C).contiguous()


def _rotation_chunk_size(
    l: int,
    batch: int,
    dtype: torch.dtype,
    *,
    target_mb: int = 128,
) -> int:
    l = int(l)
    batch = int(batch)
    if l <= 2 or batch <= 1:
        return batch
    full_dim = 3 ** l
    element_size = torch.empty((), dtype=dtype).element_size()
    bytes_per_sample = full_dim * full_dim * element_size
    target_bytes = int(target_mb) * 1024 * 1024
    chunk = max(1, target_bytes // max(bytes_per_sample, 1))
    return max(16, min(batch, int(chunk)))


def _z_rotation(angle: float) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )


@lru_cache(maxsize=None)
def so3_local_so2_basis_cpu(l: int) -> Tuple[torch.Tensor, Tuple[Tuple[int, int], ...]]:
    l = int(l)
    if l == 0:
        return torch.ones(1, 1, dtype=torch.float64), ((0, 1),)

    phi = 0.1
    D = harmonic_row_rotation(_z_rotation(phi), l).to(torch.float64)
    S = 0.5 * (D + D.T)
    evals, evecs = torch.linalg.eigh(S)

    cols: List[torch.Tensor] = []
    slices: List[Tuple[int, int]] = []
    offset = 0
    for m in range(l + 1):
        target = math.cos(m * phi)
        inds = torch.nonzero((evals - target).abs() < 1e-6).flatten()
        if m == 0:
            q = evecs[:, inds[0]]
            lead = int(torch.argmax(q.abs()).item())
            if float(q[lead].item()) < 0.0:
                q = -q
            cols.append(q)
            slices.append((offset, offset + 1))
            offset += 1
            continue
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
    return torch.stack(cols, dim=1).contiguous(), tuple(slices)


_local_q_cache: Dict[Tuple[str, str, int], torch.Tensor] = {}


def get_local_q(l: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), str(dtype), int(l))
    Q = _local_q_cache.get(key)
    if Q is None:
        Q = so3_local_so2_basis_cpu(int(l))[0].to(device=device, dtype=dtype)
        _local_q_cache[key] = Q
    return Q


def build_node_local_frames(
    edge_vec: torch.Tensor,
    edge_dst: torch.Tensor,
    num_nodes: int,
    *,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    device = edge_vec.device
    vec = edge_vec.to(dtype=compute_dtype)
    cov = scatter(
        vec.unsqueeze(-1) * vec.unsqueeze(-2),
        edge_dst,
        dim=0,
        dim_size=int(num_nodes),
        reduce="sum",
    )
    mean = scatter(
        vec,
        edge_dst,
        dim=0,
        dim_size=int(num_nodes),
        reduce="mean",
    )
    eye = torch.eye(3, device=device, dtype=compute_dtype).unsqueeze(0)
    cov = cov + 1e-8 * eye
    evals, evecs = torch.linalg.eigh(cov)
    z_axis = evecs[..., :, 2]
    x_axis = evecs[..., :, 1]

    mean_norm = mean.norm(dim=-1, keepdim=True)
    z_sign = torch.where(
        mean_norm > 1e-8,
        torch.sign((z_axis * mean).sum(dim=-1, keepdim=True)).clamp(min=0.0) * 2.0 - 1.0,
        torch.ones_like(mean_norm),
    )
    z_axis = z_axis * z_sign

    mean_proj = mean - (mean * z_axis).sum(dim=-1, keepdim=True) * z_axis
    proj_norm = mean_proj.norm(dim=-1, keepdim=True)
    x_sign = torch.where(
        proj_norm > 1e-8,
        torch.sign((x_axis * mean_proj).sum(dim=-1, keepdim=True)).clamp(min=0.0) * 2.0 - 1.0,
        torch.ones_like(proj_norm),
    )
    x_axis = x_axis * x_sign

    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    y_norm = y_axis.norm(dim=-1, keepdim=True)

    fallback_z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=compute_dtype).expand(int(num_nodes), 3)
    fallback_x = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=compute_dtype).expand(int(num_nodes), 3)
    fallback_y = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=compute_dtype).expand(int(num_nodes), 3)

    good = (evals[..., 2] - evals[..., 1] > 1e-8).unsqueeze(-1) & (y_norm > 1e-8)
    z_axis = torch.where(good, z_axis, fallback_z)
    x_axis = torch.where(good, x_axis, fallback_x)
    y_axis = torch.where(good, y_axis / y_norm.clamp(min=1e-8), fallback_y)

    local_to_global = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    det = torch.det(local_to_global)
    flip = det < 0.0
    if bool(flip.any()):
        x_axis = torch.where(flip.unsqueeze(-1), -x_axis, x_axis)
        y_axis = torch.cross(z_axis, x_axis, dim=-1)
        y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        local_to_global = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return local_to_global.transpose(-1, -2).contiguous()


def so3_blocks_to_node_local_so2(
    blocks: Dict[int, torch.Tensor],
    G: torch.Tensor,
    lmax: int,
    *,
    compute_dtype: torch.dtype,
) -> Dict[Tuple[int, int], torch.Tensor]:
    local: Dict[Tuple[int, int], torch.Tensor] = {}
    device = G.device
    Gc = G.to(dtype=compute_dtype)
    for l in range(int(lmax) + 1):
        blk = blocks[l].to(dtype=compute_dtype)
        slices = so3_local_so2_basis_cpu(l)[1]
        if l == 0:
            blk_freq = blk
            for m, (s, e) in enumerate(slices):
                local[(l, m)] = blk_freq[..., s:e]
            continue

        Q = get_local_q(l, device, compute_dtype)
        D = harmonic_row_rotation(Gc, l)
        blk_local = torch.matmul(blk, D)
        blk_freq = torch.matmul(blk_local, Q)
        for m, (s, e) in enumerate(slices):
            local[(l, m)] = blk_freq[..., s:e]
    return local


def node_local_so2_change_frame(
    pieces_by_lm: Dict[Tuple[int, int], torch.Tensor],
    R: torch.Tensor,
    lmax: int,
    *,
    compute_dtype: torch.dtype,
) -> Dict[Tuple[int, int], torch.Tensor]:
    out: Dict[Tuple[int, int], torch.Tensor] = {}
    device = R.device
    Rc = R.to(dtype=compute_dtype)
    for l in range(int(lmax) + 1):
        slices = so3_local_so2_basis_cpu(l)[1]
        blk_freq = torch.cat([pieces_by_lm[(l, m)] for m in range(l + 1)], dim=-1).to(dtype=compute_dtype)
        if l == 0:
            for m, (s, e) in enumerate(slices):
                out[(l, m)] = blk_freq[..., s:e]
            continue

        Q = get_local_q(l, device, compute_dtype)
        blk_local = torch.matmul(blk_freq, Q.transpose(-1, -2))
        if l in (1, 2):
            H, C = _get_harmonic_full_maps(l, device, compute_dtype)
            full = torch.matmul(blk_local, H)
            if l == 1:
                vec_local = torch.einsum("...ai,...ci->...ca", Rc, full)
                blk_target_local = torch.matmul(vec_local, C)
            else:
                tens = full.reshape(*full.shape[:-1], 3, 3)
                tens_local = torch.einsum("...ai,...cij,...bj->...cab", Rc, tens, Rc)
                blk_target_local = torch.matmul(tens_local.reshape(*tens_local.shape[:-2], 9), C)
        else:
            chunk = _rotation_chunk_size(l, blk_local.shape[0], compute_dtype)
            if chunk >= blk_local.shape[0]:
                D = harmonic_row_rotation(Rc, l)
                blk_target_local = torch.matmul(blk_local, D)
            else:
                chunks: List[torch.Tensor] = []
                for start in range(0, blk_local.shape[0], chunk):
                    end = min(blk_local.shape[0], start + chunk)
                    D = harmonic_row_rotation(Rc[start:end], l)
                    chunks.append(torch.matmul(blk_local[start:end], D))
                blk_target_local = torch.cat(chunks, dim=0)
        blk_target_freq = torch.matmul(blk_target_local, Q)
        for m, (s, e) in enumerate(slices):
            out[(l, m)] = blk_target_freq[..., s:e]
    return out


def node_local_so2_to_so3_blocks(
    pieces_by_lm: Dict[Tuple[int, int], torch.Tensor],
    G: torch.Tensor,
    lmax: int,
    *,
    out_dtype: torch.dtype,
    compute_dtype: torch.dtype,
) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    device = G.device
    Gc = G.to(dtype=compute_dtype)
    GT = Gc.transpose(-1, -2)
    for l in range(int(lmax) + 1):
        blk_freq = torch.cat([pieces_by_lm[(l, m)] for m in range(l + 1)], dim=-1).to(dtype=compute_dtype)
        Q = get_local_q(l, device, compute_dtype)
        blk_local = torch.matmul(blk_freq, Q.transpose(-1, -2))
        if l == 0:
            blk_global = blk_local
        elif l in (1, 2):
            H, C = _get_harmonic_full_maps(l, device, compute_dtype)
            full_local = torch.matmul(blk_local, H)
            if l == 1:
                vec_global = torch.einsum("...ia,...ca->...ci", GT, full_local)
                blk_global = torch.matmul(vec_global, C)
            else:
                tens_local = full_local.reshape(*full_local.shape[:-1], 3, 3)
                tens_global = torch.einsum("...ia,...cab,...jb->...cij", GT, tens_local, GT)
                blk_global = torch.matmul(tens_global.reshape(*tens_global.shape[:-2], 9), C)
        else:
            chunk = _rotation_chunk_size(l, blk_freq.shape[0], compute_dtype)
            if chunk >= blk_freq.shape[0]:
                D = harmonic_row_rotation(Gc, l)
                blk_global = torch.matmul(blk_local, D.transpose(-1, -2))
            else:
                chunks: List[torch.Tensor] = []
                for start in range(0, blk_freq.shape[0], chunk):
                    end = min(blk_freq.shape[0], start + chunk)
                    D = harmonic_row_rotation(Gc[start:end], l)
                    chunks.append(torch.matmul(blk_local[start:end], D.transpose(-1, -2)))
                blk_global = torch.cat(chunks, dim=0)
        out[l] = blk_global.to(dtype=out_dtype)
    return out


def pack_node_local_so2(
    local: Dict[Tuple[int, int], torch.Tensor],
    lmax: int,
) -> Dict[int, torch.Tensor]:
    packed: Dict[int, torch.Tensor] = {}
    for m in range(int(lmax) + 1):
        packed[m] = torch.stack([local[(l, m)] for l in range(m, int(lmax) + 1)], dim=1)
    return packed


def unpack_node_local_so2(
    packed: Dict[int, torch.Tensor],
    lmax: int,
) -> Dict[Tuple[int, int], torch.Tensor]:
    local: Dict[Tuple[int, int], torch.Tensor] = {}
    for m in range(int(lmax) + 1):
        for seg_idx, l in enumerate(range(m, int(lmax) + 1)):
            local[(l, m)] = packed[m][..., seg_idx, :, :]
    return local


def concat_node_local_so2_states(
    states: List[Dict[Tuple[int, int], torch.Tensor]],
    lmax: int,
) -> Dict[Tuple[int, int], torch.Tensor]:
    return {
        (l, m): torch.cat([state[(l, m)] for state in states], dim=-2)
        for l in range(int(lmax) + 1)
        for m in range(l + 1)
    }


def node_local_so2_invariants(
    local: Dict[Tuple[int, int], torch.Tensor],
    lmax: int,
) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for m in range(int(lmax) + 1):
        for l in range(m, int(lmax) + 1):
            block = local[(l, m)]
            parts.append(block.square().sum(dim=-1) / math.sqrt(float(so2_irrep_dim(m))))
    return torch.cat(parts, dim=-1)


def node_local_segments_total(lmax: int) -> int:
    total = 0
    for m in range(int(lmax) + 1):
        total += int(lmax) - m + 1
    return total
