from __future__ import annotations

import math

import torch
import torch.nn as nn

from molecular_force_field.models.ictd_irreps import direction_harmonics_all
from molecular_force_field.models.node_local_so2_helpers import (
    _get_harmonic_full_maps,
    _rotation_chunk_size,
    get_local_q,
    harmonic_row_rotation,
    node_local_so2_change_frame,
    node_local_so2_invariants,
    so3_local_so2_basis_cpu,
)
from molecular_force_field.models.pure_cartesian_ictd_fix_so2 import (
    NodeLocalL0ReadoutSO2,
    _resolve_internal_compute_dtype,
)
from molecular_force_field.utils.scatter import scatter


def _so2_lm_keys(lmax: int) -> list[tuple[int, int]]:
    return [(l, m) for l in range(int(lmax) + 1) for m in range(l + 1)]


def _so2_m_keys(lmax: int, mmax: int | None = None) -> dict[int, list[tuple[int, int]]]:
    lmax = int(lmax)
    mmax = lmax if mmax is None else min(int(mmax), lmax)
    return {m: [(l, m) for l in range(m, lmax + 1)] for m in range(mmax + 1)}


def _pack_m_major_so2(
    local: dict[tuple[int, int], torch.Tensor],
    lmax: int,
    mmax: int | None = None,
) -> dict[int, torch.Tensor]:
    """Pack local SO2 blocks as m-major tensors: m=0..mmax, l=m..lmax."""
    packed: dict[int, torch.Tensor] = {}
    for m, keys in _so2_m_keys(lmax, mmax).items():
        packed[m] = torch.stack([local[key] for key in keys], dim=1)
    return packed


def _unpack_m_major_so2(
    packed: dict[int, torch.Tensor],
    lmax: int,
    channels: int,
    mmax: int | None = None,
) -> dict[tuple[int, int], torch.Tensor]:
    """Inverse of _pack_m_major_so2."""
    del channels
    local: dict[tuple[int, int], torch.Tensor] = {}
    for m, keys in _so2_m_keys(lmax, mmax).items():
        block = packed[m]
        if block.shape[1] != len(keys):
            raise ValueError(f"Packed m={m} has {block.shape[1]} l-blocks, expected {len(keys)}")
        for idx, key in enumerate(keys):
            local[key] = block[:, idx, :, :].contiguous()
    return local


def _local_add(
    a: dict[tuple[int, int], torch.Tensor],
    b: dict[tuple[int, int], torch.Tensor],
) -> dict[tuple[int, int], torch.Tensor]:
    return {key: a[key] + b[key] for key in a}


def _concat_packed_channels(
    a: dict[int, torch.Tensor],
    b: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    return {m: torch.cat([a[m], b[m]], dim=2) for m in a}


def _change_frame_freq(
    blk_freq: torch.Tensor,
    R: torch.Tensor,
    l: int,
    *,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    if int(l) == 0:
        return blk_freq.to(dtype=compute_dtype)
    device = R.device
    Rc = R.to(dtype=compute_dtype)
    Q = get_local_q(int(l), device, compute_dtype)
    blk_local = torch.matmul(blk_freq.to(dtype=compute_dtype), Q.transpose(-1, -2))
    if int(l) in (1, 2):
        H, C = _get_harmonic_full_maps(int(l), device, compute_dtype)
        full = torch.matmul(blk_local, H)
        if int(l) == 1:
            vec_local = torch.einsum("...ai,...ci->...ca", Rc, full)
            blk_target_local = torch.matmul(vec_local, C)
        else:
            tens = full.reshape(*full.shape[:-1], 3, 3)
            tens_local = torch.einsum("...ai,...cij,...bj->...cab", Rc, tens, Rc)
            blk_target_local = torch.matmul(tens_local.reshape(*tens_local.shape[:-2], 9), C)
    else:
        chunk = _rotation_chunk_size(int(l), blk_local.shape[0], compute_dtype)
        if chunk >= blk_local.shape[0]:
            D = harmonic_row_rotation(Rc, int(l))
            blk_target_local = torch.matmul(blk_local, D)
        else:
            chunks = []
            for start in range(0, blk_local.shape[0], chunk):
                end = min(blk_local.shape[0], start + chunk)
                D = harmonic_row_rotation(Rc[start:end], int(l))
                chunks.append(torch.matmul(blk_local[start:end], D))
            blk_target_local = torch.cat(chunks, dim=0)
    return torch.matmul(blk_target_local, Q)


def _change_frame_indexed_to_packed(
    local: dict[tuple[int, int], torch.Tensor],
    index: torch.Tensor,
    R: torch.Tensor,
    lmax: int,
    mmax: int,
    *,
    compute_dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    by_m: dict[int, list[torch.Tensor]] = {m: [] for m in range(int(mmax) + 1)}
    for l in range(int(lmax) + 1):
        slices = so3_local_so2_basis_cpu(l)[1]
        blk_freq = torch.cat([local[(l, m)][index] for m in range(l + 1)], dim=-1)
        target_freq = _change_frame_freq(blk_freq, R, l, compute_dtype=compute_dtype)
        for m, (s, e) in enumerate(slices):
            if m <= int(mmax):
                by_m[m].append(target_freq[..., s:e].to(dtype=blk_freq.dtype))
    return {m: torch.stack(parts, dim=1).contiguous() for m, parts in by_m.items()}


def _packed_l_freq(
    packed: dict[int, torch.Tensor],
    l: int,
) -> torch.Tensor:
    return torch.cat([packed[m][:, int(l) - m, :, :] for m in range(int(l) + 1)], dim=-1)


def _packed_to_so3_blocks(
    packed: dict[int, torch.Tensor],
    lmax: int,
    *,
    compute_dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    blocks: dict[int, torch.Tensor] = {}
    device = packed[0].device
    for l in range(int(lmax) + 1):
        freq = _packed_l_freq(packed, l).to(dtype=compute_dtype)
        if l == 0:
            blocks[l] = freq
        else:
            q = get_local_q(l, device, compute_dtype)
            blocks[l] = torch.matmul(freq, q.transpose(-1, -2))
    return blocks


def _so3_blocks_to_packed(
    blocks: dict[int, torch.Tensor],
    lmax: int,
    mmax: int,
    *,
    out_dtype: torch.dtype,
    compute_dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    by_m: dict[int, list[torch.Tensor]] = {m: [] for m in range(int(mmax) + 1)}
    sample = next(iter(blocks.values()))
    device = sample.device
    for l in range(int(lmax) + 1):
        block = blocks[l].to(dtype=compute_dtype)
        if l == 0:
            freq = block
        else:
            q = get_local_q(l, device, compute_dtype)
            freq = torch.matmul(block, q)
        slices = so3_local_so2_basis_cpu(l)[1]
        for m, (s, e) in enumerate(slices):
            if m <= int(mmax):
                by_m[m].append(freq[..., s:e].to(dtype=out_dtype))
    return {m: torch.stack(parts, dim=1).contiguous() for m, parts in by_m.items()}


def _change_frame_packed_to_packed(
    packed: dict[int, torch.Tensor],
    R: torch.Tensor,
    lmax: int,
    mmax: int,
    *,
    compute_dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    by_m: dict[int, list[torch.Tensor]] = {m: [] for m in range(int(mmax) + 1)}
    out_dtype = packed[0].dtype
    for l in range(int(lmax) + 1):
        slices = so3_local_so2_basis_cpu(l)[1]
        blk_freq = _packed_l_freq(packed, l)
        target_freq = _change_frame_freq(blk_freq, R, l, compute_dtype=compute_dtype)
        for m, (s, e) in enumerate(slices):
            if m <= int(mmax):
                by_m[m].append(target_freq[..., s:e].to(dtype=out_dtype))
    return {m: torch.stack(parts, dim=1).contiguous() for m, parts in by_m.items()}


def _scatter_packed_so2(
    edge_packed: dict[int, torch.Tensor],
    *,
    edge_dst: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_count: torch.Tensor,
    num_nodes: int,
) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for m, block in edge_packed.items():
        mask = edge_mask.to(dtype=block.dtype).view(-1, 1, 1, 1)
        denom = neighbor_count.to(dtype=block.dtype).sqrt().view(-1, 1, 1, 1)
        out[m] = scatter(block * mask, edge_dst, dim=0, dim_size=int(num_nodes), reduce="sum") / denom
    return out


def _fibonacci_sphere(num_points: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    idx = torch.arange(int(num_points), dtype=dtype)
    z = 1.0 - 2.0 * (idx + 0.5) / float(num_points)
    radius = torch.sqrt((1.0 - z.square()).clamp(min=0.0))
    phi = idx * (math.pi * (3.0 - math.sqrt(5.0)))
    x = radius * torch.cos(phi)
    y = radius * torch.sin(phi)
    return torch.stack([x, y, z], dim=-1).contiguous()


def _build_pack_so3_maps(lmax: int) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, int, int]]]:
    lmax = int(lmax)
    total = (lmax + 1) ** 2
    pack_to_so3 = torch.zeros(total, total, dtype=torch.float64)
    so3_to_pack = torch.zeros(total, total, dtype=torch.float64)
    m_meta: list[tuple[int, int, int, int]] = []
    pack_offsets: dict[int, int] = {}
    offset = 0
    for m in range(lmax + 1):
        d = 1 if m == 0 else 2
        lm = lmax - m + 1
        pack_offsets[m] = offset
        m_meta.append((m, offset, lm, d))
        offset += lm * d
    so3_offsets: dict[int, int] = {}
    offset = 0
    for l in range(lmax + 1):
        so3_offsets[l] = offset
        offset += 2 * l + 1
    for l in range(lmax + 1):
        q, slices = so3_local_so2_basis_cpu(l)
        so3_start = so3_offsets[l]
        for m, (s, e) in enumerate(slices):
            d = e - s
            pack_start = pack_offsets[m] + (l - m) * d
            for k in range(d):
                pack_idx = pack_start + k
                local_idx = s + k
                pack_to_so3[pack_idx, so3_start : so3_start + 2 * l + 1] = q.T[local_idx]
                so3_to_pack[so3_start : so3_start + 2 * l + 1, pack_idx] = q[:, local_idx]
    return pack_to_so3.contiguous(), so3_to_pack.contiguous(), m_meta


def _build_edge_local_frames(edge_vec: torch.Tensor, *, compute_dtype: torch.dtype) -> torch.Tensor:
    """Return global-to-edge-local frames with local z aligned to each edge."""
    vec = edge_vec.to(dtype=compute_dtype)
    z_axis = vec / vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ref_x = torch.tensor([1.0, 0.0, 0.0], device=edge_vec.device, dtype=compute_dtype).expand_as(z_axis)
    ref_y = torch.tensor([0.0, 1.0, 0.0], device=edge_vec.device, dtype=compute_dtype).expand_as(z_axis)
    ref = torch.where(z_axis[..., :1].abs() < 0.9, ref_x, ref_y)
    x_axis = torch.cross(ref, z_axis, dim=-1)
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    local_to_global = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return local_to_global.transpose(-1, -2).contiguous()


def _edge_radial_basis(
    *,
    edge_length: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_mask: torch.Tensor,
    num_nodes: int,
    number_of_basis: int,
    max_radius: float,
    function_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    from molecular_force_field.models.radial_basis import soft_one_hot_linspace_mace_cutoff as soft_one_hot_linspace

    dtype = edge_length.dtype
    radial = (
        soft_one_hot_linspace(
            edge_length,
            0.0,
            float(max_radius),
            int(number_of_basis),
            basis=str(function_type),
            cutoff=True,
        )
        .mul(int(number_of_basis) ** 0.5)
        .to(dtype=dtype)
    )
    neighbor_count = scatter(
        edge_mask.to(dtype=dtype).view(-1),
        edge_dst,
        dim=0,
        dim_size=int(num_nodes),
        reduce="sum",
    ).clamp(min=1.0)
    return radial, neighbor_count


class _SO2ChannelLayerNorm(nn.Module):
    """Shared channel LayerNorm applied independently to every SO2 coefficient."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.norm = nn.LayerNorm(self.channels)

    def forward(self, local: dict[tuple[int, int], torch.Tensor]) -> dict[tuple[int, int], torch.Tensor]:
        return {
            key: self.norm(value.movedim(-2, -1)).movedim(-1, -2).contiguous()
            for key, value in local.items()
        }


class ICTDSO2Convolution(nn.Module):
    """FairChem/eSCN-style m-block SO2 convolution in ICTD-SO2 local basis."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        lmax: int,
        mmax: int | None = None,
        number_of_basis: int | None = None,
        external_weights: bool = False,
        extra_m0_output_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.lmax = int(lmax)
        self.mmax = self.lmax if mmax is None else min(int(mmax), self.lmax)
        self.external_weights = bool(external_weights)
        self.number_of_basis = None if number_of_basis is None else int(number_of_basis)
        self.extra_m0_output_channels = int(extra_m0_output_channels)
        self._flat_sizes = [(self.lmax - m + 1) * self.in_channels for m in range(self.mmax + 1)]
        self.fc_m0 = nn.Linear(
            self._flat_sizes[0],
            (self.lmax + 1) * self.out_channels + self.extra_m0_output_channels,
            bias=True,
        )
        self.weight_re = nn.ParameterList()
        self.weight_im = nn.ParameterList()
        for m in range(1, self.mmax + 1):
            in_size = self._flat_sizes[m]
            out_size = (self.lmax - m + 1) * self.out_channels
            wr = nn.Parameter(torch.empty(in_size, out_size))
            wi = nn.Parameter(torch.empty(in_size, out_size))
            nn.init.normal_(wr, mean=0.0, std=1.0 / math.sqrt(max(in_size, 1)))
            nn.init.normal_(wi, mean=0.0, std=1.0 / math.sqrt(max(in_size, 1)))
            self.weight_re.append(wr)
            self.weight_im.append(wi)
        if self.external_weights:
            if self.number_of_basis is None:
                raise ValueError("number_of_basis is required when external_weights=True")
            radial_dim = sum(self._flat_sizes)
            self._radial_slices: list[tuple[int, int, int]] = []
            offset = 0
            for m, width in enumerate(self._flat_sizes):
                self._radial_slices.append((m, offset, offset + width))
                offset += width
            self.radial_mlp = nn.Sequential(
                nn.Linear(self.number_of_basis, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, radial_dim),
            )
            last = self.radial_mlp[-1]
            if isinstance(last, nn.Linear):
                with torch.no_grad():
                    last.weight.zero_()
                    last.bias.fill_(1.0)
        else:
            self.radial_mlp = None
            self._radial_slices = []

    def forward(
        self,
        local: dict[tuple[int, int], torch.Tensor],
        *,
        edge_radial: torch.Tensor | None = None,
    ) -> dict[tuple[int, int], torch.Tensor] | tuple[dict[tuple[int, int], torch.Tensor], torch.Tensor]:
        out = self.forward_packed(
            _pack_m_major_so2(local, self.lmax, self.mmax),
            edge_radial=edge_radial,
        )
        if isinstance(out, tuple):
            packed_out, gating = out
            return _unpack_m_major_so2(packed_out, self.lmax, self.out_channels, self.mmax), gating
        return _unpack_m_major_so2(out, self.lmax, self.out_channels, self.mmax)

    def forward_packed(
        self,
        packed: dict[int, torch.Tensor],
        *,
        edge_radial: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor] | tuple[dict[int, torch.Tensor], torch.Tensor]:
        out_packed: dict[int, torch.Tensor] = {}
        dtype = packed[0].dtype
        radial_by_m: dict[int, torch.Tensor] = {}

        if self.external_weights:
            if edge_radial is None:
                raise ValueError("edge_radial is required for external ICTDSO2Convolution")
            radial = self.radial_mlp(edge_radial.to(dtype=dtype))  # type: ignore[operator]
            for m, start, end in self._radial_slices:
                radial_by_m[m] = radial[:, start:end]

        x0 = packed[0][..., 0].reshape(packed[0].shape[0], -1)
        if radial_by_m:
            x0 = x0 * radial_by_m[0]
        y0 = self.fc_m0(x0)
        gating = None
        if self.extra_m0_output_channels > 0:
            gating, y0 = y0.split(
                (self.extra_m0_output_channels, y0.shape[-1] - self.extra_m0_output_channels),
                dim=-1,
            )
        out_packed[0] = y0.reshape(x0.shape[0], self.lmax + 1, self.out_channels, 1).contiguous()

        for m in range(1, self.mmax + 1):
            x = packed[m]
            xr = x[..., 0].reshape(x.shape[0], -1)
            xi = x[..., 1].reshape(x.shape[0], -1)
            if radial_by_m:
                gate = radial_by_m[m]
                xr = xr * gate
                xi = xi * gate
            wr = self.weight_re[m - 1].to(dtype=x.dtype, device=x.device)
            wi = self.weight_im[m - 1].to(dtype=x.dtype, device=x.device)
            yr = xr.matmul(wr) - xi.matmul(wi)
            yi = xr.matmul(wi) + xi.matmul(wr)
            lm = self.lmax - m + 1
            out_packed[m] = torch.stack(
                [
                    yr.reshape(x.shape[0], lm, self.out_channels),
                    yi.reshape(x.shape[0], lm, self.out_channels),
                ],
                dim=-1,
            ).contiguous()

        if self.extra_m0_output_channels > 0:
            if gating is None:
                raise RuntimeError("Expected m=0 gating output")
            return out_packed, gating
        return out_packed


class ICTDS2GridActivation(nn.Module):
    """Project ICTD local harmonics to an S2 grid, apply pointwise MLP, and project back."""

    def __init__(
        self,
        *,
        channels: int,
        lmax: int,
        grid_points: int = 64,
        hidden_channels: int | None = None,
        pointwise: str = "silu",
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.grid_points = int(grid_points)
        self.pointwise = str(pointwise)
        self.compute_dtype = torch.get_default_dtype() if compute_dtype is None else compute_dtype
        hidden_channels = self.channels if hidden_channels is None else int(hidden_channels)
        dirs = _fibonacci_sphere(self.grid_points, dtype=torch.float64)
        y_blocks = direction_harmonics_all(dirs, self.lmax)
        y_flat = torch.cat(y_blocks, dim=-1).contiguous()
        y_pinv = torch.linalg.pinv(y_flat).contiguous()
        pack_to_so3, so3_to_pack, m_meta = _build_pack_so3_maps(self.lmax)
        self.register_buffer("grid_dirs", dirs.to(dtype=torch.get_default_dtype()), persistent=False)
        self.register_buffer("grid_y", y_flat.to(dtype=torch.get_default_dtype()), persistent=False)
        self.register_buffer("grid_y_pinv", y_pinv.to(dtype=torch.get_default_dtype()), persistent=False)
        self.register_buffer("pack_to_so3", pack_to_so3.to(dtype=torch.get_default_dtype()), persistent=False)
        self.register_buffer("so3_to_pack", so3_to_pack.to(dtype=torch.get_default_dtype()), persistent=False)
        self._m_meta = m_meta
        if self.pointwise == "silu":
            self.grid_mlp = nn.SiLU()
        elif self.pointwise == "mlp":
            self.grid_mlp = nn.Sequential(
                nn.Linear(self.channels, hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_channels, hidden_channels, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_channels, self.channels, bias=False),
            )
        else:
            raise ValueError(f"Unknown ICTDS2GridActivation pointwise={self.pointwise!r}")

    def forward(
        self,
        local: dict[tuple[int, int], torch.Tensor],
        input_scalars: torch.Tensor | None = None,
    ) -> dict[tuple[int, int], torch.Tensor]:
        out = self.forward_packed(
            _pack_m_major_so2(local, self.lmax, self.lmax),
            input_scalars=input_scalars,
        )
        return _unpack_m_major_so2(out, self.lmax, self.channels, self.lmax)

    def forward_packed(
        self,
        packed: dict[int, torch.Tensor],
        input_scalars: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor]:
        sample = packed[0]
        out_dtype = sample.dtype
        compute_dtype = self.compute_dtype if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
        freq_pack = torch.cat(
            [
                packed[m].permute(0, 2, 1, 3).reshape(sample.shape[0], sample.shape[2], -1)
                for m in range(self.lmax + 1)
            ],
            dim=-1,
        ).to(dtype=compute_dtype)
        coeff = torch.matmul(
            freq_pack,
            self.pack_to_so3.to(device=sample.device, dtype=compute_dtype),
        )
        y = self.grid_y.to(device=sample.device, dtype=compute_dtype)
        grid = torch.einsum("ncd,pd->npc", coeff, y).to(dtype=out_dtype)
        activated = self.grid_mlp(grid)
        y_pinv = self.grid_y_pinv.to(device=sample.device, dtype=activated.dtype)
        coeff_new = torch.einsum("npc,dp->ncd", activated, y_pinv)
        freq_new = torch.matmul(
            coeff_new.to(dtype=compute_dtype),
            self.so3_to_pack.to(device=sample.device, dtype=compute_dtype),
        )
        out: dict[int, torch.Tensor] = {}
        for m, start, lm, d in self._m_meta:
            part = freq_new[..., start : start + lm * d]
            out[m] = part.reshape(sample.shape[0], sample.shape[2], lm, d).permute(0, 2, 1, 3).to(dtype=out_dtype).contiguous()
        if input_scalars is not None:
            out[0] = out[0].clone()
            out[0][:, 0, :, 0] = torch.nn.functional.silu(input_scalars).to(dtype=out_dtype)
        return out


def _scatter_local_so2(
    edge_local: dict[tuple[int, int], torch.Tensor],
    *,
    edge_dst: torch.Tensor,
    edge_mask: torch.Tensor,
    neighbor_count: torch.Tensor,
    num_nodes: int,
    lmax: int,
) -> dict[tuple[int, int], torch.Tensor]:
    out: dict[tuple[int, int], torch.Tensor] = {}
    for key in _so2_lm_keys(lmax):
        block = edge_local[key]
        mask = edge_mask.to(dtype=block.dtype).view(-1, 1, 1)
        denom = neighbor_count.to(dtype=block.dtype).sqrt().view(-1, 1, 1)
        out[key] = scatter(block * mask, edge_dst, dim=0, dim_size=int(num_nodes), reduce="sum") / denom
    return out


class _ESCNInitialSO2Interaction(nn.Module):
    """eSCN edge-degree embedding: edge scalar features -> full node-local SO2 state."""

    def __init__(
        self,
        *,
        channels: int,
        lmax: int,
        edge_feature_dim: int,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.edge_feature_dim = int(edge_feature_dim)
        self.compute_dtype = compute_dtype
        self.radial_mlp = nn.Sequential(
            nn.Linear(self.edge_feature_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, (self.lmax + 1) * self.channels),
        )

    def forward(
        self,
        *,
        edge_features: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_mask: torch.Tensor,
        neighbor_count: torch.Tensor,
        edge_frame: torch.Tensor,
        target_frame: torch.Tensor,
    ) -> dict[tuple[int, int], torch.Tensor]:
        del edge_src
        dtype = edge_features.dtype
        num_nodes = target_frame.shape[0]
        radial = self.radial_mlp(edge_features.to(dtype=dtype)).view(-1, self.lmax + 1, self.channels)

        edge_local: dict[tuple[int, int], torch.Tensor] = {}
        for l in range(self.lmax + 1):
            edge_local[(l, 0)] = radial[:, l, :].unsqueeze(-1)
            for m in range(1, l + 1):
                edge_local[(l, m)] = radial.new_zeros(radial.shape[0], self.channels, 2)

        R_edge_to_target = torch.matmul(target_frame[edge_dst], edge_frame.transpose(-1, -2))
        target_edge_local = node_local_so2_change_frame(
            edge_local,
            R_edge_to_target,
            self.lmax,
            compute_dtype=dtype if self.compute_dtype is None else self.compute_dtype,
        )
        message = _scatter_local_so2(
            target_edge_local,
            edge_dst=edge_dst,
            edge_mask=edge_mask,
            neighbor_count=neighbor_count,
            num_nodes=num_nodes,
            lmax=self.lmax,
        )
        return message


class _ESCNSO2InteractionBlock(nn.Module):
    """eSCN-style edge-frame SO2 convolution + grid atomwise residual block."""

    def __init__(
        self,
        *,
        channels: int,
        hidden_channels: int,
        lmax: int,
        edge_feature_dim: int,
        grid_points: int = 64,
        mmax: int | None = None,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.lmax = int(lmax)
        self.mmax = self.lmax if mmax is None else min(int(mmax), self.lmax)
        self.edge_feature_dim = int(edge_feature_dim)
        self.compute_dtype = compute_dtype
        self.keys = _so2_lm_keys(self.lmax)
        self.norm1 = _SO2ChannelLayerNorm(self.channels)
        self.conv1 = ICTDSO2Convolution(
            in_channels=2 * self.channels,
            out_channels=self.hidden_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            number_of_basis=self.edge_feature_dim,
            external_weights=True,
            extra_m0_output_channels=self.hidden_channels,
        )
        self.grid_activation = ICTDS2GridActivation(
            channels=self.hidden_channels,
            lmax=self.lmax,
            grid_points=int(grid_points),
            hidden_channels=self.hidden_channels,
            pointwise="silu",
            compute_dtype=compute_dtype,
        )
        self.conv2 = ICTDSO2Convolution(
            in_channels=self.hidden_channels,
            out_channels=self.channels,
            lmax=self.lmax,
            mmax=self.mmax,
            external_weights=False,
        )
        self.norm2 = _SO2ChannelLayerNorm(self.channels)
        self.grid_atomwise = ICTDS2GridActivation(
            channels=self.channels,
            lmax=self.lmax,
            grid_points=int(grid_points),
            hidden_channels=self.channels,
            pointwise="mlp",
            compute_dtype=compute_dtype,
        )

    def forward(
        self,
        *,
        local: dict[tuple[int, int], torch.Tensor],
        edge_features: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_mask: torch.Tensor,
        neighbor_count: torch.Tensor,
        node_frame: torch.Tensor,
        edge_frame: torch.Tensor,
    ) -> dict[tuple[int, int], torch.Tensor]:
        dtype = next(iter(local.values())).dtype
        num_nodes = next(iter(local.values())).shape[0]
        normed = self.norm1(local)
        compute_dtype = dtype if self.compute_dtype is None else self.compute_dtype
        R_source_to_edge = torch.matmul(edge_frame, node_frame[edge_src].transpose(-1, -2))
        source_edge = _change_frame_indexed_to_packed(
            normed,
            edge_src,
            R_source_to_edge,
            self.lmax,
            self.mmax,
            compute_dtype=compute_dtype,
        )
        R_target_to_edge = torch.matmul(edge_frame, node_frame[edge_dst].transpose(-1, -2))
        target_edge = _change_frame_indexed_to_packed(
            normed,
            edge_dst,
            R_target_to_edge,
            self.lmax,
            self.mmax,
            compute_dtype=compute_dtype,
        )

        edge_pair = _concat_packed_channels(source_edge, target_edge)
        hidden_edge, hidden_scalars = self.conv1.forward_packed(edge_pair, edge_radial=edge_features)
        hidden_edge = self.grid_activation.forward_packed(hidden_edge, input_scalars=hidden_scalars)
        edge_message = self.conv2.forward_packed(hidden_edge)
        if isinstance(edge_message, tuple):
            raise RuntimeError("conv2 must not produce m=0 gating")

        R_edge_to_target = torch.matmul(node_frame[edge_dst], edge_frame.transpose(-1, -2))
        target_edge_message = _change_frame_packed_to_packed(
            edge_message,
            R_edge_to_target,
            self.lmax,
            self.mmax,
            compute_dtype=compute_dtype,
        )
        message_packed = _scatter_packed_so2(
            target_edge_message,
            edge_dst=edge_dst,
            edge_mask=edge_mask,
            neighbor_count=neighbor_count,
            num_nodes=num_nodes,
        )
        message = _unpack_m_major_so2(message_packed, self.lmax, self.channels, self.mmax)
        residual = _local_add(local, message)

        atomwise_delta_packed = self.grid_atomwise.forward_packed(
            _pack_m_major_so2(self.norm2(residual), self.lmax, self.mmax)
        )
        atomwise_delta = _unpack_m_major_so2(atomwise_delta_packed, self.lmax, self.channels, self.mmax)
        return _local_add(residual, atomwise_delta)


class PureCartesianICTDESCNSO2(nn.Module):
    """eSCN-style SO2-native backbone/readout in the ICTD-SO2 local basis."""

    def __init__(
        self,
        max_embed_radius: float = 5.0,
        main_max_radius: float = 5.0,
        main_number_of_basis: int = 8,
        hidden_dim_conv: int = 64,
        hidden_dim_sh: int = 64,
        hidden_dim: int = 64,
        channel_in2: int | None = None,
        embedding_dim: int = 16,
        max_atomvalue: int = 10,
        atomic_numbers: list[int] | tuple[int, ...] | None = None,
        output_size: int = 1,
        embed_size: list[int] | tuple[int, ...] | int = 64,
        main_hidden_sizes3: list[int] | tuple[int, ...] | None = None,
        num_layers: int = 2,
        num_interaction: int = 2,
        function_type_main: str = "gaussian",
        lmax: int = 3,
        ictd_fix_route: str = "baseline",
        ictd_fix_contraction_combine: str = "softmax",
        ictd_fix_product_backend: str = "ictd-pure-u",
        ictd_fix_interaction_scale: str = "none",
        ictd_fix_fusion_scale_init: float = 0.1,
        ictd_fix_fusion_heads: int = 1,
        ictd_fix_fusion_head_weight_mode: str = "softmax",
        ictd_fix_readout_head_scale_init: float = 1.0,
        ictd_fix_readout_head_scale_trainable: bool = False,
        save_contraction_order: int = 3,
        save_multiple_mix_channels: int | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        interaction_compute_dtype: torch.dtype | None = None,
        contraction_compute_dtype: torch.dtype | None = None,
        avg_num_neighbors: float | None = None,
        device: torch.device | str | None = None,
        **unused,
    ):
        super().__init__()
        grid_points = int(unused.pop("grid_points", 64))
        if "mmax" in unused and int(unused["mmax"]) != int(lmax):
            raise ValueError("PureCartesianICTDESCNSO2 v2 fixes mmax=lmax")
        del (
            max_embed_radius,
            hidden_dim_sh,
            channel_in2,
            embedding_dim,
            output_size,
            embed_size,
            ictd_fix_contraction_combine,
            contraction_compute_dtype,
            device,
            unused,
        )
        del (
            main_hidden_sizes3,
            ictd_fix_product_backend,
            ictd_fix_route,
            ictd_fix_fusion_scale_init,
            ictd_fix_fusion_heads,
            ictd_fix_fusion_head_weight_mode,
            ictd_fix_readout_head_scale_init,
            ictd_fix_readout_head_scale_trainable,
            save_contraction_order,
            save_multiple_mix_channels,
            avg_num_neighbors,
        )
        if ictd_fix_interaction_scale not in {"none", "mace-rms"}:
            raise ValueError(f"ictd_fix_interaction_scale must be 'none' or 'mace-rms', got {ictd_fix_interaction_scale!r}")
        if int(num_interaction) < 1:
            raise ValueError(f"pure-cartesian-ictd-escn-so2 requires num_interaction >= 1, got {num_interaction}")

        self.channels = int(hidden_dim_conv)
        self.lmax = int(lmax)
        self.escn_mmax = self.lmax
        self.escn_grid_points = grid_points
        self.edge_hidden_channels = int(hidden_dim) if int(hidden_dim) > 0 else self.channels
        self.num_layers = int(num_interaction)
        self.max_radius = float(main_max_radius)
        self.number_of_basis = int(main_number_of_basis)
        self.function_type = str(function_type_main)
        self.max_atomvalue = int(max_atomvalue)
        self.internal_compute_dtype = _resolve_internal_compute_dtype(internal_compute_dtype)
        self.interaction_compute_dtype = interaction_compute_dtype
        self.ictd_fix_interaction_scale = str(ictd_fix_interaction_scale)
        if atomic_numbers is None:
            atomic_numbers = tuple(range(self.max_atomvalue))
        else:
            atomic_numbers = tuple(sorted({int(z) for z in atomic_numbers}))
            if len(atomic_numbers) == 0:
                raise ValueError("atomic_numbers must not be empty")
        self.atomic_numbers = atomic_numbers
        self.num_elements = len(self.atomic_numbers)
        map_size = max(self.max_atomvalue, max(self.atomic_numbers) + 1)
        atomic_number_to_index = torch.full((map_size,), -1, dtype=torch.long)
        for idx, z in enumerate(self.atomic_numbers):
            if z < 0:
                raise ValueError(f"atomic_numbers must be non-negative, got {z}")
            atomic_number_to_index[z] = idx
        self.register_buffer("atomic_number_to_index", atomic_number_to_index, persistent=False)

        self.node_embedding = nn.Embedding(self.num_elements, self.channels)
        self.edge_channels = self.edge_hidden_channels
        self.source_embedding = nn.Embedding(self.num_elements, self.edge_channels)
        self.target_embedding = nn.Embedding(self.num_elements, self.edge_channels)
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
        self.edge_feature_dim = self.number_of_basis + 2 * self.edge_channels

        self.initial_interaction = _ESCNInitialSO2Interaction(
            channels=self.channels,
            lmax=self.lmax,
            edge_feature_dim=self.edge_feature_dim,
            compute_dtype=self.interaction_compute_dtype,
        )
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(
                _ESCNSO2InteractionBlock(
                    channels=self.channels,
                    hidden_channels=self.edge_hidden_channels,
                    lmax=self.lmax,
                    edge_feature_dim=self.edge_feature_dim,
                    grid_points=self.escn_grid_points,
                    mmax=self.escn_mmax,
                    compute_dtype=self.interaction_compute_dtype,
                )
            )
        self.final_norm = _SO2ChannelLayerNorm(self.channels)
        self.energy_readout = NodeLocalL0ReadoutSO2(self.channels, output_init_std=0.003)

    def _compact_atom_types(self, A: torch.Tensor) -> torch.Tensor:
        A_long = A.long()
        if int(A_long.max().item()) >= self.atomic_number_to_index.numel():
            raise ValueError(
                f"Encountered atomic number {int(A_long.max().item())}, but compact mapping supports only up to "
                f"{self.atomic_number_to_index.numel() - 1}. atomic_numbers={self.atomic_numbers}"
            )
        compact_idx = self.atomic_number_to_index[A_long]
        if torch.any(compact_idx < 0):
            bad = torch.unique(A_long[compact_idx < 0]).tolist()
            raise ValueError(
                f"Encountered atomic numbers without compact mapping: {bad}. "
                f"Configured atomic_numbers={self.atomic_numbers}"
            )
        return compact_idx

    def _combined_local_summary(
        self,
        layer_states: list[dict[tuple[int, int], torch.Tensor]],
        *,
        scalar_last: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [node_local_so2_invariants(local, self.lmax) for local in layer_states]
        if scalar_last is not None:
            parts.append(scalar_last)
        return torch.cat(parts, dim=-1)

    def _make_scalar_local(self, scalar_feats: torch.Tensor) -> dict[tuple[int, int], torch.Tensor]:
        local: dict[tuple[int, int], torch.Tensor] = {}
        for l in range(self.lmax + 1):
            if l == 0:
                local[(0, 0)] = scalar_feats.unsqueeze(-1)
            else:
                local[(l, 0)] = scalar_feats.new_zeros(scalar_feats.shape[0], self.channels, 1)
            for m in range(1, l + 1):
                local[(l, m)] = scalar_feats.new_zeros(scalar_feats.shape[0], self.channels, 2)
        return local

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
        return_physical_tensors: bool = False,
        return_reciprocal_source: bool = False,
    ):
        del sync_after_scatter
        if return_physical_tensors:
            raise ValueError("pure-cartesian-ictd-escn-so2 does not currently support return_physical_tensors=True")

        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        sort_idx = torch.argsort(edge_dst)
        edge_src = edge_src[sort_idx]
        edge_dst = edge_dst[sort_idx]
        edge_shifts = edge_shifts[sort_idx]
        if precomputed_edge_vec is not None:
            edge_vec = precomputed_edge_vec[sort_idx].to(dtype=dtype)
        else:
            edge_batch_idx = batch[edge_src]
            edge_cells = cell[edge_batch_idx]
            shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
            edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)
        edge_mask = (edge_length <= self.max_radius).to(dtype=pos.dtype).unsqueeze(-1)
        num_nodes = pos.size(0)

        edge_radial, neighbor_count = _edge_radial_basis(
            edge_length=edge_length,
            edge_dst=edge_dst,
            edge_mask=edge_mask,
            num_nodes=num_nodes,
            number_of_basis=self.number_of_basis,
            max_radius=self.max_radius,
            function_type=self.function_type,
        )
        edge_frame = _build_edge_local_frames(edge_vec, compute_dtype=self.internal_compute_dtype)
        from molecular_force_field.models.node_local_so2_helpers import build_node_local_frames

        node_frame = build_node_local_frames(
            edge_vec,
            edge_dst,
            num_nodes,
            compute_dtype=self.internal_compute_dtype,
        )

        compact_idx = self._compact_atom_types(A)
        scalar_feats = self.node_embedding(compact_idx)
        edge_features = torch.cat(
            [
                edge_radial.to(dtype=dtype),
                self.source_embedding(compact_idx[edge_src]),
                self.target_embedding(compact_idx[edge_dst]),
            ],
            dim=-1,
        )
        local = self._make_scalar_local(scalar_feats)
        edge_degree = self.initial_interaction(
            edge_features=edge_features,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_mask=edge_mask,
            neighbor_count=neighbor_count,
            edge_frame=edge_frame,
            target_frame=node_frame,
        )
        local = _local_add(local, edge_degree)
        layer_states = [local]

        for block in self.blocks:
            local = block(
                local=layer_states[-1],
                edge_features=edge_features,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_mask=edge_mask,
                neighbor_count=neighbor_count,
                node_frame=node_frame,
                edge_frame=edge_frame,
            )
            layer_states.append(local)

        local = self.final_norm(local)
        layer_states[-1] = local
        node_energy = self.energy_readout(local)
        out = node_energy.sum(dim=-1, keepdim=True)
        if return_combined_features:
            combined_features = self._combined_local_summary(layer_states)
            if return_reciprocal_source:
                return out, combined_features, out.new_empty((out.size(0), 0))
            return out, combined_features
        if return_reciprocal_source:
            return out, out.new_empty((out.size(0), 0))
        return out
