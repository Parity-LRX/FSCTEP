from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from molecular_force_field.utils.checkpoint_metadata import (
    derive_long_range_far_max_radius_multiplier,
)


def _build_integer_k_lattice(kmax: int, include_k0: bool) -> torch.Tensor:
    values: list[list[float]] = []
    for i in range(-int(kmax), int(kmax) + 1):
        for j in range(-int(kmax), int(kmax) + 1):
            for k in range(-int(kmax), int(kmax) + 1):
                if not include_k0 and i == 0 and j == 0 and k == 0:
                    continue
                values.append([float(i), float(j), float(k)])
    if not values:
        return torch.zeros((0, 3), dtype=torch.float32)
    return torch.tensor(values, dtype=torch.float32)


class LatentSourceHead(nn.Module):
    """Map node invariant features to latent reciprocal-space sources."""

    def __init__(self, feature_dim: int, hidden_dim: int, source_channels: int = 1):
        super().__init__()
        self.source_channels = int(source_channels)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.source_channels),
        )

    def forward(self, invariant_features: torch.Tensor) -> torch.Tensor:
        return self.net(invariant_features)


def _fft_integer_frequencies(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.fft.fftfreq(size, d=1.0 / float(size), device=device).to(dtype=dtype)


def _effective_cell_for_boundary(
    cell: torch.Tensor,
    *,
    boundary: str,
    slab_padding_factor: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    effective_cell = cell.to(dtype=dtype)
    if boundary == "slab":
        effective_cell = effective_cell.clone()
        effective_cell[2] = effective_cell[2] * float(max(int(slab_padding_factor), 1))
    return effective_cell


def _prepare_frac_for_boundary(
    pos: torch.Tensor,
    cell: torch.Tensor,
    *,
    boundary: str,
    slab_padding_factor: int,
) -> torch.Tensor:
    inv_cell = torch.linalg.inv(cell)
    frac = torch.einsum("ni,ij->nj", pos, inv_cell)
    if boundary == "periodic":
        return frac - torch.floor(frac)

    frac = frac.clone()
    frac[:, :2] = frac[:, :2] - torch.floor(frac[:, :2])
    pad = float(max(int(slab_padding_factor), 1))
    z_offset = 0.5 * (pad - 1.0) / pad
    frac[:, 2] = frac[:, 2] / pad + z_offset
    return frac


def _corner_weights_from_frac(frac: torch.Tensor) -> torch.Tensor:
    wx0 = 1.0 - frac[:, 0]
    wy0 = 1.0 - frac[:, 1]
    wz0 = 1.0 - frac[:, 2]
    wx1 = frac[:, 0]
    wy1 = frac[:, 1]
    wz1 = frac[:, 2]
    return torch.stack(
        [
            wx0 * wy0 * wz0,
            wx0 * wy0 * wz1,
            wx0 * wy1 * wz0,
            wx0 * wy1 * wz1,
            wx1 * wy0 * wz0,
            wx1 * wy0 * wz1,
            wx1 * wy1 * wz0,
            wx1 * wy1 * wz1,
        ],
        dim=1,
    )


def _apply_mesh_boundary(idx: torch.Tensor, *, mesh_size: int, boundary: str) -> torch.Tensor:
    if boundary == "periodic":
        return torch.remainder(idx, mesh_size)
    idx_wrapped = idx.clone()
    idx_wrapped[:, 0] = torch.remainder(idx_wrapped[:, 0], mesh_size)
    idx_wrapped[:, 1] = torch.remainder(idx_wrapped[:, 1], mesh_size)
    idx_wrapped[:, 2] = idx_wrapped[:, 2].clamp(0, mesh_size - 1)
    return idx_wrapped


def _spread_source_to_mesh(
    frac: torch.Tensor,
    source: torch.Tensor,
    *,
    mesh_size: int,
    corner_offsets: torch.Tensor,
    boundary: str,
) -> torch.Tensor:
    channels = int(source.size(1))
    mesh = source.new_zeros((mesh_size, mesh_size, mesh_size, channels))
    flat_mesh = mesh.view(-1, channels)
    scaled = frac * float(mesh_size)
    base = torch.floor(scaled).to(dtype=torch.long)
    frac_offset = scaled - base.to(dtype=scaled.dtype)
    corner_weights = _corner_weights_from_frac(frac_offset)
    for corner in range(8):
        idx = _apply_mesh_boundary(base + corner_offsets[corner], mesh_size=mesh_size, boundary=boundary)
        flat_idx = ((idx[:, 0] * mesh_size) + idx[:, 1]) * mesh_size + idx[:, 2]
        flat_mesh.scatter_add_(
            0,
            flat_idx.unsqueeze(-1).expand(-1, channels),
            source * corner_weights[:, corner].unsqueeze(-1),
        )
    return mesh


def _gather_source_from_mesh(
    frac: torch.Tensor,
    mesh: torch.Tensor,
    *,
    mesh_size: int,
    corner_offsets: torch.Tensor,
    boundary: str,
) -> torch.Tensor:
    channels = int(mesh.size(-1))
    flat_mesh = mesh.view(-1, channels)
    scaled = frac * float(mesh_size)
    base = torch.floor(scaled).to(dtype=torch.long)
    frac_offset = scaled - base.to(dtype=scaled.dtype)
    corner_weights = _corner_weights_from_frac(frac_offset)
    gathered = frac.new_zeros((frac.size(0), channels))
    for corner in range(8):
        idx = _apply_mesh_boundary(base + corner_offsets[corner], mesh_size=mesh_size, boundary=boundary)
        flat_idx = ((idx[:, 0] * mesh_size) + idx[:, 1]) * mesh_size + idx[:, 2]
        gathered = gathered + flat_mesh.index_select(0, flat_idx) * corner_weights[:, corner].unsqueeze(-1)
    return gathered


class FeatureSpectralFilterGrid(nn.Module):
    """Apply a learnable radial spectral filter on a regular periodic/slab mesh."""

    def __init__(
        self,
        *,
        mesh_size: int,
        channels: int,
        hidden_dim: int,
        boundary: str = "periodic",
        slab_padding_factor: int = 2,
        include_k0: bool = False,
        k_norm_floor: float = 1.0e-6,
    ):
        super().__init__()
        self.mesh_size = int(mesh_size)
        self.channels = int(channels)
        self.boundary = str(boundary)
        if self.boundary not in {"periodic", "slab"}:
            raise ValueError(f"Unsupported feature spectral boundary: {self.boundary!r}")
        self.slab_padding_factor = max(int(slab_padding_factor), 1)
        self.include_k0 = bool(include_k0)
        self.k_norm_floor = float(k_norm_floor)
        self.radial_filter = RadialSpectralFilter(hidden_dim=hidden_dim, k_norm_floor=k_norm_floor)
        self.channel_scale_raw = nn.Parameter(torch.zeros(self.channels))

    def _effective_cell(self, cell: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return _effective_cell_for_boundary(
            cell,
            boundary=self.boundary,
            slab_padding_factor=self.slab_padding_factor,
            dtype=dtype,
        )

    def build_k_norms(self, cell: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        freq = _fft_integer_frequencies(self.mesh_size, device=cell.device, dtype=dtype)
        kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing="ij")
        integer_k = torch.stack([kx, ky, kz], dim=-1).reshape(-1, 3)
        effective_cell = self._effective_cell(cell, dtype=dtype)
        inv_cell = torch.linalg.inv(effective_cell)
        k_cart = 2.0 * math.pi * torch.matmul(integer_k, inv_cell)
        return torch.linalg.vector_norm(k_cart, dim=-1).reshape(self.mesh_size, self.mesh_size, self.mesh_size)

    def forward(self, mesh: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        mesh_dtype = mesh.dtype
        mesh_complex = torch.fft.fftn(mesh, dim=(0, 1, 2))
        k_norms = self.build_k_norms(cell, dtype=mesh_dtype)
        spectral_weights = self.radial_filter(k_norms)
        if not self.include_k0:
            spectral_weights = torch.where(
                k_norms > self.k_norm_floor,
                spectral_weights,
                torch.zeros_like(spectral_weights),
            )
        channel_scale = torch.nn.functional.softplus(self.channel_scale_raw).to(dtype=mesh_dtype)
        filtered = torch.fft.ifftn(
            mesh_complex * spectral_weights.unsqueeze(-1) * channel_scale.view(1, 1, 1, -1),
            dim=(0, 1, 2),
        )
        return filtered.real


class FeatureSpectralResidualBlock(nn.Module):
    """Low-rank feature-space spectral filter with FFT mesh projection."""

    def __init__(
        self,
        *,
        feature_dim: int,
        bottleneck_dim: int = 8,
        mesh_size: int = 16,
        filter_hidden_dim: int = 64,
        boundary: str = "periodic",
        slab_padding_factor: int = 2,
        neutralize: bool = True,
        include_k0: bool = False,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.mesh_size = int(mesh_size)
        self.boundary = str(boundary)
        if self.boundary not in {"periodic", "slab"}:
            raise ValueError(f"Unsupported feature spectral boundary: {self.boundary!r}")
        self.slab_padding_factor = max(int(slab_padding_factor), 1)
        self.neutralize = bool(neutralize)
        self.include_k0 = bool(include_k0)
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.in_proj = nn.Linear(self.feature_dim, self.bottleneck_dim)
        self.out_proj = nn.Linear(self.bottleneck_dim, self.feature_dim, bias=False)
        self.mesh_filter = FeatureSpectralFilterGrid(
            mesh_size=self.mesh_size,
            channels=self.bottleneck_dim,
            hidden_dim=int(filter_hidden_dim),
            boundary=self.boundary,
            slab_padding_factor=self.slab_padding_factor,
            include_k0=self.include_k0,
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        corner_offsets = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=torch.long,
        )
        self.register_buffer("corner_offsets", corner_offsets, persistent=False)

    def _neutralize_source(self, source: torch.Tensor) -> torch.Tensor:
        if not self.neutralize:
            return source
        return source - source.mean(dim=0, keepdim=True)

    def _corner_weights(self, frac: torch.Tensor) -> torch.Tensor:
        return _corner_weights_from_frac(frac)

    def _effective_cell(self, cell: torch.Tensor) -> torch.Tensor:
        return _effective_cell_for_boundary(
            cell,
            boundary=self.boundary,
            slab_padding_factor=self.slab_padding_factor,
            dtype=cell.dtype,
        )

    def _prepare_frac(self, pos: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        return _prepare_frac_for_boundary(
            pos,
            cell,
            boundary=self.boundary,
            slab_padding_factor=self.slab_padding_factor,
        )

    def _apply_boundary(self, idx: torch.Tensor) -> torch.Tensor:
        return _apply_mesh_boundary(idx, mesh_size=self.mesh_size, boundary=self.boundary)

    def _spread_to_mesh(self, frac: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return _spread_source_to_mesh(
            frac,
            source,
            mesh_size=self.mesh_size,
            corner_offsets=self.corner_offsets,
            boundary=self.boundary,
        )

    def _gather_from_mesh(self, frac: torch.Tensor, mesh: torch.Tensor) -> torch.Tensor:
        return _gather_source_from_mesh(
            frac,
            mesh,
            mesh_size=self.mesh_size,
            corner_offsets=self.corner_offsets,
            boundary=self.boundary,
        )

    def _filter_single_graph(self, pos: torch.Tensor, cell: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        frac = self._prepare_frac(pos, cell)
        mesh = self._spread_to_mesh(frac, source)
        filtered_mesh = self.mesh_filter(mesh, cell)
        return self._gather_from_mesh(frac, filtered_mesh)

    def forward(
        self,
        invariant_features: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.in_proj(self.input_norm(invariant_features))
        filtered_source = torch.zeros_like(source)

        for graph_idx in range(cell.size(0)):
            node_index = torch.nonzero(batch == graph_idx, as_tuple=False).view(-1)
            if node_index.numel() == 0:
                continue
            filtered_source.index_copy_(
                0,
                node_index,
                self._filter_single_graph(
                    pos.index_select(0, node_index),
                    cell[graph_idx],
                    self._neutralize_source(source.index_select(0, node_index)),
                ),
            )

        residual = self.out_proj(filtered_source)
        gated_residual = torch.tanh(self.gate).to(dtype=residual.dtype) * residual
        return invariant_features + gated_residual, source


class RadialSpectralFilter(nn.Module):
    """Learnable radial filter that modulates a Coulomb-like k-space kernel."""

    def __init__(self, hidden_dim: int, k_norm_floor: float = 1.0e-6):
        super().__init__()
        self.k_norm_floor = float(k_norm_floor)
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, k_norms: torch.Tensor) -> torch.Tensor:
        safe_k = k_norms.clamp_min(self.k_norm_floor)
        x0 = torch.log1p(safe_k)
        x1 = x0 * x0
        x2 = torch.reciprocal(safe_k)
        x = torch.stack([x0, x1, x2], dim=-1)
        learned_scale = torch.nn.functional.softplus(self.net(x).squeeze(-1))
        base_kernel = 4.0 * math.pi / (safe_k * safe_k)
        return base_kernel * learned_scale


class ReciprocalGreenKernel(nn.Module):
    """Base Poisson kernel with an optional learnable radial modifier."""

    def __init__(self, *, green_mode: str, hidden_dim: int, k_norm_floor: float = 1.0e-6):
        super().__init__()
        if green_mode not in {"poisson", "learned_poisson"}:
            raise ValueError(f"Unsupported long-range green mode: {green_mode!r}")
        self.green_mode = str(green_mode)
        self.k_norm_floor = float(k_norm_floor)
        self.learned_filter = (
            RadialSpectralFilter(hidden_dim=hidden_dim, k_norm_floor=k_norm_floor)
            if self.green_mode == "learned_poisson"
            else None
        )

    def forward(self, k_norms: torch.Tensor) -> torch.Tensor:
        if self.learned_filter is not None:
            return self.learned_filter(k_norms)
        safe_k = k_norms.clamp_min(self.k_norm_floor)
        return 4.0 * math.pi / (safe_k * safe_k)


class ReciprocalSpectralKernel3D(nn.Module):
    """Direct k-space prototype for periodic reciprocal-space long-range energy."""

    def __init__(
        self,
        *,
        kmax: int = 2,
        filter_hidden_dim: int = 64,
        include_k0: bool = False,
        reciprocal_backend: str = "direct_kspace",
        energy_partition: str = "potential",
        k_norm_floor: float = 1.0e-6,
    ):
        super().__init__()
        if int(kmax) < 0:
            raise ValueError(f"kmax must be >= 0, got {kmax}")
        if reciprocal_backend != "direct_kspace":
            raise ValueError(f"Unsupported reciprocal backend: {reciprocal_backend!r}")
        if energy_partition not in ("potential", "uniform"):
            raise ValueError(f"Unsupported reciprocal energy partition: {energy_partition!r}")
        self.kmax = int(kmax)
        self.include_k0 = bool(include_k0)
        self.reciprocal_backend = str(reciprocal_backend)
        self.energy_partition = str(energy_partition)
        self.k_norm_floor = float(k_norm_floor)
        self.register_buffer("integer_k_lattice", _build_integer_k_lattice(self.kmax, self.include_k0))
        if self.integer_k_lattice.shape[0] == 0:
            raise ValueError("ReciprocalSpectralKernel3D produced an empty k-lattice; increase kmax or enable include_k0")
        self.spectral_filter = RadialSpectralFilter(hidden_dim=filter_hidden_dim, k_norm_floor=k_norm_floor)

    @property
    def num_k(self) -> int:
        return int(self.integer_k_lattice.shape[0])

    def build_k_lattice(self, cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inv_cells = torch.linalg.inv(cell)
        k_lattice = self.integer_k_lattice.to(device=cell.device, dtype=cell.dtype)
        k_cart = 2.0 * math.pi * torch.einsum("kd,bdh->bkh", k_lattice, inv_cells)
        k_norms = torch.linalg.vector_norm(k_cart, dim=-1)
        volumes = torch.abs(torch.linalg.det(cell)).clamp_min(self.k_norm_floor)
        return k_cart, k_norms, volumes

    def compute_structure_factor(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inv_cells = torch.linalg.inv(cell)
        atom_inv_cells = inv_cells.index_select(0, batch)
        frac = torch.einsum("ni,nij->nj", pos, atom_inv_cells)
        k_lattice = self.integer_k_lattice.to(device=pos.device, dtype=pos.dtype)
        phases = 2.0 * math.pi * torch.matmul(frac, k_lattice.transpose(0, 1))
        cos_phase = torch.cos(phases)
        sin_phase = torch.sin(phases)

        graph_ids = torch.arange(cell.size(0), device=batch.device, dtype=batch.dtype)
        graph_mask = (batch.unsqueeze(1) == graph_ids.unsqueeze(0)).to(dtype=pos.dtype)
        weighted_cos = source.unsqueeze(1) * cos_phase.unsqueeze(-1)
        weighted_sin = source.unsqueeze(1) * sin_phase.unsqueeze(-1)
        structure_cos = torch.einsum("nb,nkc->bkc", graph_mask, weighted_cos)
        structure_sin = torch.einsum("nb,nkc->bkc", graph_mask, weighted_sin)
        return structure_cos, structure_sin, cos_phase, sin_phase

    def apply_spectral_filter(self, structure_cos: torch.Tensor, structure_sin: torch.Tensor, k_norms: torch.Tensor) -> torch.Tensor:
        spectral_weights = self.spectral_filter(k_norms)
        if not self.include_k0:
            spectral_weights = torch.where(
                k_norms > self.k_norm_floor,
                spectral_weights,
                torch.zeros_like(spectral_weights),
            )
        return spectral_weights

    def forward(self, pos: torch.Tensor, batch: torch.Tensor, cell: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(batch, minlength=cell.size(0)).to(dtype=pos.dtype).clamp_min(1.0)
        k_cart, k_norms, volumes = self.build_k_lattice(cell)
        structure_cos, structure_sin, cos_phase, sin_phase = self.compute_structure_factor(pos, batch, cell, source)
        del k_cart
        spectral_weights = self.apply_spectral_filter(structure_cos, structure_sin, k_norms)

        source_cos = structure_cos.index_select(0, batch)
        source_sin = structure_sin.index_select(0, batch)
        spectral_per_atom = spectral_weights.index_select(0, batch)
        volume_per_atom = volumes.index_select(0, batch).unsqueeze(-1)
        potential = (
            spectral_per_atom.unsqueeze(-1)
            * (source_cos * cos_phase.unsqueeze(-1) + source_sin * sin_phase.unsqueeze(-1))
        ).sum(dim=1) / volume_per_atom

        if self.energy_partition == "potential":
            atom_energy = 0.5 * (source * potential).sum(dim=-1, keepdim=True)
            return atom_energy

        graph_total = 0.5 * (
            spectral_weights.unsqueeze(-1) * (structure_cos.square() + structure_sin.square())
        ).sum(dim=(1, 2)) / volumes
        atom_energy = graph_total.index_select(0, batch).unsqueeze(-1) / counts.index_select(0, batch).unsqueeze(-1)
        return atom_energy


def _screened_coulomb_kernel(distance: torch.Tensor, *, screening: torch.Tensor) -> torch.Tensor:
    return torch.exp(-screening * distance) / distance


def _neutralize_scalar_source(source: torch.Tensor, batch: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    graph_ids = torch.arange(cell.size(0), device=batch.device, dtype=batch.dtype)
    graph_mask = (batch.unsqueeze(1) == graph_ids.unsqueeze(0)).to(dtype=source.dtype)
    counts = graph_mask.sum(dim=0).clamp_min(1.0)
    graph_mean = torch.einsum("nb,n->b", graph_mask, source) / counts
    return source - graph_mean.index_select(0, batch)


def _apply_periodic_minimum_image(delta: torch.Tensor, *, batch: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    atom_cells = cell.index_select(0, batch)
    atom_inv_cells = torch.linalg.inv(atom_cells)
    frac = torch.einsum("nij,njk->nik", delta, atom_inv_cells)
    shifts = -torch.round(frac)
    return delta + torch.einsum("nij,njk->nik", shifts, atom_cells)


def _segment_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size,) + src.shape[1:])
    return out.index_add(0, index, src)


class DenseRealSpaceLongRangeKernel(nn.Module):
    """Dense pairwise screened-Coulomb kernel."""

    def __init__(self, *, boundary: str, energy_partition: str = "potential"):
        super().__init__()
        if boundary not in {"nonperiodic", "periodic"}:
            raise ValueError(f"Unsupported dense long-range boundary mode: {boundary!r}")
        if energy_partition not in {"potential", "uniform"}:
            raise ValueError(f"Unsupported long-range energy partition: {energy_partition!r}")
        self.boundary = str(boundary)
        self.energy_partition = str(energy_partition)

    def forward(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        source: torch.Tensor,
        *,
        screening: torch.Tensor,
        softening: torch.Tensor,
    ) -> torch.Tensor:
        same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
        delta = pos.unsqueeze(0) - pos.unsqueeze(1)
        if self.boundary == "periodic":
            delta = _apply_periodic_minimum_image(delta, batch=batch, cell=cell)

        r2 = (delta * delta).sum(dim=-1)
        r = torch.sqrt(r2 + softening * softening)
        kernel = _screened_coulomb_kernel(r, screening=screening)
        n = pos.size(0)
        diag_mask = torch.arange(n, device=pos.device, dtype=torch.long)
        kernel = kernel * same_graph.to(kernel.dtype)
        kernel = kernel * (1.0 - (diag_mask.unsqueeze(0) == diag_mask.unsqueeze(1)).to(kernel.dtype))

        pair_energy = 0.5 * (source.unsqueeze(0) * source.unsqueeze(1)) * kernel
        atom_energy = pair_energy.sum(dim=1, keepdim=True)
        if self.energy_partition == "uniform":
            counts = torch.bincount(batch, minlength=cell.size(0)).to(dtype=source.dtype).clamp_min(1.0)
            graph_ids = torch.arange(cell.size(0), device=batch.device, dtype=batch.dtype)
            graph_mask = (batch.unsqueeze(1) == graph_ids.unsqueeze(0)).to(dtype=atom_energy.dtype)
            graph_total = torch.einsum("nb,nc->bc", graph_mask, atom_energy).sum(dim=-1, keepdim=True)
            atom_energy = graph_total.index_select(0, batch) / counts.index_select(0, batch).unsqueeze(-1)
        return atom_energy


class _TreeCell:
    def __init__(self, indices: list[int], center: torch.Tensor, half_extent: float):
        self.indices = indices
        self.center = center
        self.half_extent = float(half_extent)
        self.children: list[_TreeCell] = []
        self.index_tensor: torch.Tensor | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def contains(self, atom_idx: int) -> bool:
        return atom_idx in self.indices


class OpenBoundaryTreeLongRangeKernel(nn.Module):
    """Barnes-Hut style open-boundary approximation for scalar latent charges."""

    def __init__(
        self,
        *,
        boundary: str,
        theta: float = 0.5,
        leaf_size: int = 32,
        multipole_order: int = 0,
        energy_partition: str = "potential",
    ):
        super().__init__()
        if boundary != "nonperiodic":
            raise ValueError("tree_fmm long-range backend currently requires boundary='nonperiodic'")
        if energy_partition not in {"potential", "uniform"}:
            raise ValueError(f"Unsupported long-range energy partition: {energy_partition!r}")
        if int(multipole_order) != 0:
            raise ValueError("tree_fmm correctness-first backend currently supports only multipole_order=0")
        self.boundary = str(boundary)
        self.theta = float(theta)
        self.leaf_size = max(int(leaf_size), 1)
        self.multipole_order = int(multipole_order)
        self.energy_partition = str(energy_partition)

    def _build_tree(self, pos: torch.Tensor, indices: list[int]) -> _TreeCell:
        local = pos.index_select(0, torch.tensor(indices, device=pos.device, dtype=torch.long))
        coord_min = local.amin(dim=0)
        coord_max = local.amax(dim=0)
        center = 0.5 * (coord_min + coord_max)
        half_extent = float((coord_max - coord_min).amax().item() * 0.5)
        node = _TreeCell(indices=indices, center=center, half_extent=half_extent)
        node.index_tensor = torch.tensor(indices, device=pos.device, dtype=torch.long)
        if len(indices) <= self.leaf_size:
            return node

        children_bins: dict[tuple[int, int, int], list[int]] = {}
        for atom_idx in indices:
            atom_pos = pos[atom_idx]
            key = (
                int(atom_pos[0] >= center[0]),
                int(atom_pos[1] >= center[1]),
                int(atom_pos[2] >= center[2]),
            )
            children_bins.setdefault(key, []).append(atom_idx)
        if len(children_bins) <= 1:
            return node
        node.children = [self._build_tree(pos, child_indices) for child_indices in children_bins.values()]
        return node

    def _approximate_node_potential(
        self,
        *,
        pos: torch.Tensor,
        source: torch.Tensor,
        target_idx: int,
        target_pos: torch.Tensor,
        node: _TreeCell,
        screening: torch.Tensor,
        softening: torch.Tensor,
    ) -> torch.Tensor:
        assert node.index_tensor is not None
        if node.index_tensor.numel() == 0:
            return target_pos.new_zeros(())

        contains_target = node.contains(target_idx)
        local_pos = pos.index_select(0, node.index_tensor)
        local_source = source.index_select(0, node.index_tensor)

        if node.is_leaf:
            if contains_target:
                mask = node.index_tensor != target_idx
                local_pos = local_pos[mask]
                local_source = local_source[mask]
            if local_source.numel() == 0:
                return target_pos.new_zeros(())
            delta = target_pos.unsqueeze(0) - local_pos
            r = torch.sqrt((delta * delta).sum(dim=-1) + softening * softening)
            kernel = _screened_coulomb_kernel(r, screening=screening)
            return torch.sum(local_source * kernel)

        delta_center = target_pos - node.center
        dist_center = torch.sqrt((delta_center * delta_center).sum() + softening * softening)
        cell_diameter = max(2.0 * node.half_extent, 1.0e-9)
        if (not contains_target) and (cell_diameter / dist_center.item() < self.theta):
            cluster_charge = local_source.sum()
            cluster_center = local_pos.mean(dim=0)
            delta = target_pos - cluster_center
            r = torch.sqrt((delta * delta).sum() + softening * softening)
            return cluster_charge * _screened_coulomb_kernel(r, screening=screening)

        total = target_pos.new_zeros(())
        for child in node.children:
            total = total + self._approximate_node_potential(
                pos=pos,
                source=source,
                target_idx=target_idx,
                target_pos=target_pos,
                node=child,
                screening=screening,
                softening=softening,
            )
        return total

    def forward(
        self,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        source: torch.Tensor,
        *,
        screening: torch.Tensor,
        softening: torch.Tensor,
    ) -> torch.Tensor:
        atom_energy = source.new_zeros((source.size(0), 1))
        counts = torch.bincount(batch, minlength=cell.size(0)).to(dtype=source.dtype).clamp_min(1.0)
        for graph_idx in range(cell.size(0)):
            node_index = torch.nonzero(batch == graph_idx, as_tuple=False).view(-1)
            if node_index.numel() == 0:
                continue
            local_pos = pos.index_select(0, node_index)
            local_source = source.index_select(0, node_index)
            root = self._build_tree(local_pos.detach(), list(range(int(node_index.numel()))))
            local_potential = local_source.new_zeros((node_index.numel(),))
            for local_idx in range(int(node_index.numel())):
                local_potential[local_idx] = self._approximate_node_potential(
                    pos=local_pos,
                    source=local_source,
                    target_idx=local_idx,
                    target_pos=local_pos[local_idx],
                    node=root,
                    screening=screening,
                    softening=softening,
                )
            atom_energy_local = 0.5 * (local_source * local_potential).unsqueeze(-1)
            if self.energy_partition == "uniform":
                graph_total = atom_energy_local.sum()
                atom_energy_local = graph_total.expand_as(atom_energy_local) / counts[graph_idx]
            atom_energy.index_copy_(0, node_index, atom_energy_local)
        return atom_energy


class LatentCoulombLongRange(nn.Module):
    """Real-space latent-charge prototype kept for backward compatibility."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        *,
        boundary: str = "nonperiodic",
        neutralize: bool = True,
        backend: str = "dense_pairwise",
        energy_partition: str = "potential",
        screening_init: float = 0.0,
        softening_init: float = 0.5,
        theta: float = 0.5,
        leaf_size: int = 32,
        multipole_order: int = 0,
    ):
        super().__init__()
        if boundary not in ("nonperiodic", "periodic"):
            raise ValueError(f"Unsupported long-range boundary mode: {boundary!r}")
        if backend not in {"dense_pairwise", "tree_fmm"}:
            raise ValueError(f"Unsupported latent-coulomb backend: {backend!r}")
        self.boundary = str(boundary)
        self.neutralize = bool(neutralize)
        self.backend = str(backend)
        self.energy_partition = str(energy_partition)
        self.theta = float(theta)
        self.leaf_size = max(int(leaf_size), 1)
        self.multipole_order = int(multipole_order)
        self.charge_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.source_channels = 1
        self.exports_reciprocal_source = self.backend == "tree_fmm"
        self.source_kind = "latent_charge"
        self.source_layout = "scalar_per_atom"
        self.runtime_backend = "tree_fmm" if self.backend == "tree_fmm" else "none"
        self.num_k = None
        self.energy_scale = nn.Parameter(torch.tensor(0.0))
        self.screening_raw = nn.Parameter(torch.tensor(float(screening_init)))
        self.softening_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(max(float(softening_init), 1.0e-3))))
        )
        if self.backend == "tree_fmm":
            self.kernel = OpenBoundaryTreeLongRangeKernel(
                boundary=self.boundary,
                theta=self.theta,
                leaf_size=self.leaf_size,
                multipole_order=self.multipole_order,
                energy_partition=self.energy_partition,
            )
        else:
            self.kernel = DenseRealSpaceLongRangeKernel(
                boundary=self.boundary,
                energy_partition=self.energy_partition,
            )

    def forward(
        self,
        invariant_features: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        *,
        edge_src: torch.Tensor | None = None,
        edge_dst: torch.Tensor | None = None,
        return_source: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        q = self.charge_head(invariant_features).squeeze(-1)
        softening = torch.nn.functional.softplus(self.softening_raw) + 1.0e-6
        screening = torch.nn.functional.softplus(self.screening_raw)
        if self.neutralize:
            q = _neutralize_scalar_source(q, batch, cell)

        source = q.unsqueeze(-1)
        if return_source and self.exports_reciprocal_source:
            atom_energy = source.new_zeros((source.size(0), 1))
            return atom_energy, source

        atom_energy = self.kernel(
            pos,
            batch,
            cell,
            q,
            screening=screening,
            softening=softening,
        )
        atom_energy = self.energy_scale * atom_energy
        if return_source:
            return atom_energy, source
        return atom_energy


class IsolatedFarFieldLongRange(nn.Module):
    """Nonperiodic isolated-system far-field correction from global-minus-local latent statistics."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        *,
        boundary: str = "nonperiodic",
        neutralize: bool = True,
        energy_partition: str = "potential",
        filter_hidden_dim: int = 64,
        source_channels: int = 1,
    ):
        super().__init__()
        if boundary != "nonperiodic":
            raise ValueError("isolated-far-field-v1 currently requires boundary='nonperiodic'")
        if energy_partition not in {"potential", "uniform"}:
            raise ValueError(f"Unsupported long-range energy partition: {energy_partition!r}")
        latent_dim = max(int(source_channels), min(int(hidden_dim), 16))
        self.boundary = str(boundary)
        self.neutralize = bool(neutralize)
        self.energy_partition = str(energy_partition)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.filter_hidden_dim = int(filter_hidden_dim)
        self.source_channels = 0
        self.exports_reciprocal_source = False
        self.source_kind = "none"
        self.source_layout = "none"
        self.runtime_backend = "none"
        self.num_k = None
        self.input_norm = nn.LayerNorm(feature_dim)
        self.source_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )
        self.shell_stat_mlp = nn.Sequential(
            nn.Linear(3, self.filter_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.filter_hidden_dim, self.latent_dim),
        )
        self.energy_head = nn.Sequential(
            nn.LayerNorm(4 * self.latent_dim + 3),
            nn.Linear(4 * self.latent_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.energy_scale = nn.Parameter(torch.tensor(0.0))

    def _neutralize_source(self, source: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        if not self.neutralize:
            return source
        graph_ids = torch.arange(num_graphs, device=batch.device, dtype=batch.dtype)
        graph_mask = (batch.unsqueeze(1) == graph_ids.unsqueeze(0)).to(dtype=source.dtype)
        counts = graph_mask.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        graph_mean = torch.einsum("nb,nc->bc", graph_mask, source) / counts
        return source - graph_mean.index_select(0, batch)

    def forward(
        self,
        invariant_features: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        *,
        edge_src: torch.Tensor | None = None,
        edge_dst: torch.Tensor | None = None,
        return_source: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        num_graphs = cell.size(0)
        source = self.source_head(self.input_norm(invariant_features))
        source = self._neutralize_source(source, batch, num_graphs)
        n_atoms = source.size(0)

        graph_sum = _segment_sum(source, batch, num_graphs)
        graph_sq_sum = _segment_sum(source.square(), batch, num_graphs)
        graph_counts = _segment_sum(
            torch.ones((n_atoms, 1), device=source.device, dtype=source.dtype),
            batch,
            num_graphs,
        )
        graph_center = _segment_sum(pos, batch, num_graphs) / graph_counts.clamp_min(1.0)
        atom_graph_sum = graph_sum.index_select(0, batch)
        atom_graph_sq_sum = graph_sq_sum.index_select(0, batch)
        atom_graph_counts = graph_counts.index_select(0, batch)
        atom_graph_center = graph_center.index_select(0, batch)

        local_sum = source.clone()
        local_sq_sum = source.square()
        local_count = torch.ones((n_atoms, 1), device=source.device, dtype=source.dtype)
        if edge_src is not None and edge_dst is not None:
            local_sum = local_sum + _segment_sum(source.index_select(0, edge_src), edge_dst, n_atoms)
            local_sq_sum = local_sq_sum + _segment_sum(source.index_select(0, edge_src).square(), edge_dst, n_atoms)
            local_count = local_count + _segment_sum(
                torch.ones((edge_dst.numel(), 1), device=source.device, dtype=source.dtype),
                edge_dst,
                n_atoms,
            )

        far_count = (atom_graph_counts - local_count).clamp_min(0.0)
        far_denom = far_count.clamp_min(1.0)
        far_sum = atom_graph_sum - local_sum
        far_sq_sum = atom_graph_sq_sum - local_sq_sum
        far_mean = far_sum / far_denom
        far_var = (far_sq_sum / far_denom - far_mean.square()).clamp_min(0.0)
        far_std = torch.sqrt(far_var + 1.0e-8)

        radial = torch.linalg.vector_norm(pos - atom_graph_center, dim=-1, keepdim=True)
        graph_radial_mean = _segment_sum(radial, batch, num_graphs) / graph_counts.clamp_min(1.0)
        radial_scale = graph_radial_mean.index_select(0, batch).clamp_min(1.0e-6)
        radial_feature = radial / radial_scale

        shell_stats = torch.cat(
            [
                torch.log1p(far_count),
                far_count / atom_graph_counts.clamp_min(1.0),
                radial_feature,
            ],
            dim=-1,
        )
        shell_gate = torch.sigmoid(self.shell_stat_mlp(shell_stats))
        fused = torch.cat(
            [
                source,
                far_mean,
                source * far_mean,
                far_std * shell_gate,
                shell_stats,
            ],
            dim=-1,
        )
        atom_energy = self.energy_scale * self.energy_head(fused)
        if self.energy_partition == "uniform":
            graph_total = _segment_sum(atom_energy, batch, num_graphs)
            atom_energy = graph_total.index_select(0, batch) / atom_graph_counts.clamp_min(1.0)
        if return_source:
            return atom_energy, atom_energy.new_empty((atom_energy.size(0), 0))
        return atom_energy


class IsolatedFarFieldLongRangeV2(nn.Module):
    """Nonperiodic isolated-system far-field correction with explicit shells and optional tail."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        *,
        boundary: str = "nonperiodic",
        neutralize: bool = True,
        energy_partition: str = "potential",
        far_source_dim: int = 16,
        far_num_shells: int = 3,
        far_shell_growth: float = 2.0,
        far_tail: bool = True,
        far_tail_bins: int = 2,
        far_stats: str = "mean,count,mean_r,rms_r",
        far_max_radius_multiplier: float | None = None,
        far_source_norm: bool = True,
        far_gate_init: float = 0.0,
        cutoff_radius: float = 5.0,
    ):
        super().__init__()
        if boundary != "nonperiodic":
            raise ValueError("isolated-far-field-v2 currently requires boundary='nonperiodic'")
        if energy_partition not in {"potential", "uniform"}:
            raise ValueError(f"Unsupported long-range energy partition: {energy_partition!r}")
        if int(far_source_dim) < 1:
            raise ValueError("isolated-far-field-v2 requires far_source_dim >= 1")
        if int(far_num_shells) < 1:
            raise ValueError("isolated-far-field-v2 requires far_num_shells >= 1")
        if float(far_shell_growth) <= 1.0:
            raise ValueError("isolated-far-field-v2 requires far_shell_growth > 1")
        if int(far_tail_bins) < 1:
            raise ValueError("isolated-far-field-v2 requires far_tail_bins >= 1")
        if far_max_radius_multiplier is None:
            far_max_radius_multiplier = derive_long_range_far_max_radius_multiplier(
                int(far_num_shells),
                float(far_shell_growth),
            )
        if float(far_max_radius_multiplier) <= 1.0:
            raise ValueError("isolated-far-field-v2 requires far_max_radius_multiplier > 1")
        if float(far_shell_growth) ** int(far_num_shells) > float(far_max_radius_multiplier):
            raise ValueError(
                "isolated-far-field-v2 requires far_shell_growth ** far_num_shells <= "
                "far_max_radius_multiplier to avoid empty explicit shells"
            )
        if str(far_stats) != "mean,count,mean_r,rms_r":
            raise ValueError("isolated-far-field-v2 currently supports only far_stats='mean,count,mean_r,rms_r'")
        self.boundary = str(boundary)
        self.neutralize = bool(neutralize)
        self.energy_partition = str(energy_partition)
        self.hidden_dim = int(hidden_dim)
        self.far_source_dim = int(far_source_dim)
        self.far_num_shells = int(far_num_shells)
        self.far_shell_growth = float(far_shell_growth)
        self.far_tail = bool(far_tail)
        self.far_tail_bins = int(far_tail_bins)
        self.tail_num_bins = self.far_tail_bins if self.far_tail else 0
        self.far_stats = str(far_stats)
        self.far_max_radius_multiplier = float(far_max_radius_multiplier)
        self.far_source_norm = bool(far_source_norm)
        self.cutoff_radius = float(cutoff_radius)
        self.total_shells = self.far_num_shells + self.tail_num_bins
        self.shell_feature_dim = self.far_source_dim + 3
        self.source_channels = 0
        self.exports_reciprocal_source = False
        self.source_kind = "none"
        self.source_layout = "none"
        self.runtime_backend = "none"
        self.num_k = None
        self.input_norm = nn.LayerNorm(feature_dim)
        self.source_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.far_source_dim),
        )
        self.source_norm = nn.LayerNorm(self.far_source_dim) if self.far_source_norm else nn.Identity()
        self.shell_encoder = nn.Sequential(
            nn.LayerNorm(self.total_shells * self.shell_feature_dim),
            nn.Linear(self.total_shells * self.shell_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.far_source_dim),
        )
        self.energy_head = nn.Sequential(
            nn.LayerNorm(4 * self.far_source_dim + self.total_shells * self.shell_feature_dim),
            nn.Linear(4 * self.far_source_dim + self.total_shells * self.shell_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer(
            "cell_neighbor_offsets",
            torch.cartesian_prod(
                torch.tensor([-1, 0, 1], dtype=torch.long),
                torch.tensor([-1, 0, 1], dtype=torch.long),
                torch.tensor([-1, 0, 1], dtype=torch.long),
            ),
            persistent=False,
        )
        kernel_rbf_centers = torch.linspace(
            self.cutoff_radius,
            self.cutoff_radius * self.far_max_radius_multiplier,
            steps=8,
            dtype=torch.float32,
        )
        kernel_rbf_spacing = float(kernel_rbf_centers[1] - kernel_rbf_centers[0]) if kernel_rbf_centers.numel() > 1 else 1.0
        self.register_buffer("kernel_rbf_centers", kernel_rbf_centers, persistent=False)
        self.register_buffer(
            "kernel_rbf_gamma",
            torch.tensor(1.0 / max(kernel_rbf_spacing * kernel_rbf_spacing, 1.0e-6), dtype=torch.float32),
            persistent=False,
        )
        self.kernel_screening_raw = nn.Parameter(torch.tensor(-8.0))
        self.kernel_gate_mlp = nn.Sequential(
            nn.Linear(self.kernel_rbf_centers.numel() + 2, min(self.hidden_dim, 32)),
            nn.SiLU(),
            nn.Linear(min(self.hidden_dim, 32), 1),
        )
        nn.init.zeros_(self.kernel_gate_mlp[-1].weight)
        nn.init.zeros_(self.kernel_gate_mlp[-1].bias)
        self.far_gate = nn.Parameter(torch.tensor(float(far_gate_init)))
        self.energy_scale = nn.Parameter(torch.tensor(0.0))

    def _neutralize_source(self, source: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        if not self.neutralize:
            return source
        graph_ids = torch.arange(num_graphs, device=batch.device, dtype=batch.dtype)
        graph_mask = (batch.unsqueeze(1) == graph_ids.unsqueeze(0)).to(dtype=source.dtype)
        counts = graph_mask.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        graph_mean = torch.einsum("nb,nc->bc", graph_mask, source) / counts
        return source - graph_mean.index_select(0, batch)

    def _shell_bounds(self, *, device: torch.device, dtype: torch.dtype) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = []
        lower = self.cutoff_radius
        explicit_max = self.cutoff_radius * self.far_max_radius_multiplier
        for shell_idx in range(self.far_num_shells):
            upper = self.cutoff_radius * (self.far_shell_growth ** float(shell_idx + 1))
            upper = min(upper, explicit_max)
            bounds.append((lower, upper))
            lower = upper
        return bounds

    def _tail_bin_bounds(self) -> list[tuple[float, float]]:
        if self.tail_num_bins == 0:
            return []
        explicit_max = self.cutoff_radius * self.far_max_radius_multiplier
        bounds: list[tuple[float, float]] = []
        lower = explicit_max
        for bin_idx in range(self.tail_num_bins):
            if bin_idx == self.tail_num_bins - 1:
                upper = float("inf")
            else:
                upper = explicit_max * (self.far_shell_growth ** float(bin_idx + 1))
            bounds.append((lower, upper))
            lower = upper
        return bounds

    def _physical_kernel(self, distance: torch.Tensor) -> torch.Tensor:
        screening = F.softplus(self.kernel_screening_raw)
        base_kernel = torch.exp(-screening * distance) / distance.clamp_min(1.0e-6)
        centers = self.kernel_rbf_centers.to(device=distance.device, dtype=distance.dtype)
        gamma = self.kernel_rbf_gamma.to(device=distance.device, dtype=distance.dtype)
        distance_col = distance.unsqueeze(-1)
        rbf = torch.exp(-gamma * (distance_col - centers.unsqueeze(0)).square())
        gate_input = torch.cat(
            [
                distance_col / max(self.cutoff_radius, 1.0e-6),
                distance_col / max(self.cutoff_radius * self.far_max_radius_multiplier, 1.0e-6),
                rbf,
            ],
            dim=-1,
        )
        learned_gate = 1.0 + torch.tanh(self.kernel_gate_mlp(gate_input)).squeeze(-1)
        return base_kernel * learned_gate

    def _tail_proxy_distance(self, radial_to_center: torch.Tensor) -> torch.Tensor:
        explicit_max = self.cutoff_radius * self.far_max_radius_multiplier
        return explicit_max + radial_to_center

    def _build_sparse_far_edges(
        self,
        local_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_local = local_pos.size(0)
        if n_local == 0:
            return (
                local_pos.new_empty((0,), dtype=torch.long),
                local_pos.new_empty((0,), dtype=torch.long),
                local_pos.new_empty((0,), dtype=local_pos.dtype),
            )
        explicit_max = self.cutoff_radius * self.far_max_radius_multiplier
        cell_size = max(explicit_max, self.cutoff_radius + 1.0e-6)
        origin = local_pos.min(dim=0).values
        cell_coords = torch.floor((local_pos - origin) / cell_size).to(dtype=torch.long)
        grid_shape = cell_coords.max(dim=0).values + 1
        cell_hash = cell_coords[:, 0] + grid_shape[0] * (
            cell_coords[:, 1] + grid_shape[1] * cell_coords[:, 2]
        )
        atom_order = torch.argsort(cell_hash)
        sorted_hash = cell_hash.index_select(0, atom_order)
        sorted_coords = cell_coords.index_select(0, atom_order)
        unique_hash, counts = torch.unique_consecutive(sorted_hash, return_counts=True)
        cell_ptr = torch.cat(
            [
                counts.new_zeros((1,)),
                counts.cumsum(dim=0),
            ],
            dim=0,
        )
        unique_coords = sorted_coords.index_select(0, cell_ptr[:-1])
        offsets = self.cell_neighbor_offsets.to(device=local_pos.device)
        query_coords = unique_coords.unsqueeze(1) + offsets.unsqueeze(0)
        valid_query = (
            (query_coords[..., 0] >= 0)
            & (query_coords[..., 1] >= 0)
            & (query_coords[..., 2] >= 0)
            & (query_coords[..., 0] < grid_shape[0])
            & (query_coords[..., 1] < grid_shape[1])
            & (query_coords[..., 2] < grid_shape[2])
        )
        query_hash = query_coords[..., 0] + grid_shape[0] * (
            query_coords[..., 1] + grid_shape[1] * query_coords[..., 2]
        )
        flat_query_hash = query_hash.reshape(-1)
        search_idx = torch.searchsorted(unique_hash, flat_query_hash)
        max_search_idx = torch.full_like(search_idx, unique_hash.size(0) - 1)
        safe_search_idx = torch.minimum(search_idx, max_search_idx)
        matched = (search_idx < unique_hash.numel()) & valid_query.reshape(-1)
        matched = matched & (
            unique_hash.index_select(0, safe_search_idx) == flat_query_hash
        )
        if not bool(matched.any()):
            return (
                local_pos.new_empty((0,), dtype=torch.long),
                local_pos.new_empty((0,), dtype=torch.long),
                local_pos.new_empty((0,), dtype=local_pos.dtype),
            )
        src_cell = torch.arange(unique_hash.numel(), device=local_pos.device, dtype=torch.long)
        src_cell = src_cell.unsqueeze(1).expand(-1, offsets.size(0)).reshape(-1)
        src_cell = src_cell[matched]
        dst_cell = search_idx[matched]
        num_pairs_per_cell_pair = counts.index_select(0, src_cell) * counts.index_select(0, dst_cell)
        pair_ptr = torch.cat(
            [
                num_pairs_per_cell_pair.new_zeros((1,)),
                num_pairs_per_cell_pair.cumsum(dim=0),
            ],
            dim=0,
        )
        total_pairs = pair_ptr[-1]
        if total_pairs == 0:
            return (
                local_pos.new_empty((0,), dtype=torch.long),
                local_pos.new_empty((0,), dtype=torch.long),
                local_pos.new_empty((0,), dtype=local_pos.dtype),
            )
        cell_pair_ids = torch.repeat_interleave(
            torch.arange(src_cell.numel(), device=local_pos.device, dtype=torch.long),
            num_pairs_per_cell_pair,
        )
        flat_pair_index = torch.arange(total_pairs, device=local_pos.device, dtype=torch.long)
        local_pair_offset = flat_pair_index - pair_ptr[:-1].index_select(0, cell_pair_ids)
        recv_cell = src_cell.index_select(0, cell_pair_ids)
        send_cell = dst_cell.index_select(0, cell_pair_ids)
        send_count = counts.index_select(0, send_cell)
        recv_local = torch.div(local_pair_offset, send_count, rounding_mode="floor")
        send_local = torch.remainder(local_pair_offset, send_count)
        recv_atoms = atom_order.index_select(0, cell_ptr.index_select(0, recv_cell) + recv_local)
        send_atoms = atom_order.index_select(0, cell_ptr.index_select(0, send_cell) + send_local)
        disp = local_pos.index_select(0, recv_atoms) - local_pos.index_select(0, send_atoms)
        dist = torch.linalg.vector_norm(disp, dim=-1)
        keep = (recv_atoms != send_atoms) & (dist > self.cutoff_radius) & (dist <= explicit_max)
        return send_atoms[keep], recv_atoms[keep], dist[keep]

    def _summarize_sparse_edges(
        self,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_dist: torch.Tensor,
        source: torch.Tensor,
        n_local: int,
        *,
        lower: float,
        upper: float,
    ) -> torch.Tensor:
        shell_mask = (edge_dist > lower) & (edge_dist <= upper)
        masked_src = edge_src[shell_mask]
        masked_dst = edge_dst[shell_mask]
        masked_dist = edge_dist[shell_mask].unsqueeze(-1)
        kernel_weight = self._physical_kernel(masked_dist.squeeze(-1)).unsqueeze(-1).to(dtype=source.dtype)
        counts = _segment_sum(
            kernel_weight,
            masked_dst,
            n_local,
        )
        denom = counts.clamp_min(1.0e-8)
        mean_source = _segment_sum(
            source.index_select(0, masked_src) * kernel_weight,
            masked_dst,
            n_local,
        ) / denom
        mean_r = _segment_sum(masked_dist.to(dtype=source.dtype) * kernel_weight, masked_dst, n_local) / denom
        rms_r = torch.sqrt(
            _segment_sum(
                masked_dist.square().to(dtype=source.dtype) * kernel_weight,
                masked_dst,
                n_local,
            ) / denom + 1.0e-8
        )
        present = (counts > 0).to(dtype=source.dtype)
        return torch.cat([mean_source, counts, mean_r, rms_r], dim=-1) * present

    def _accumulate_tail_bin_stats(
        self,
        source: torch.Tensor,
        proxy_distance: torch.Tensor,
        dst_index: torch.Tensor,
        dim_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source_sum = source.new_zeros((dim_size, self.tail_num_bins, self.far_source_dim))
        weight_sum = source.new_zeros((dim_size, self.tail_num_bins, 1))
        r_sum = source.new_zeros((dim_size, self.tail_num_bins, 1))
        r2_sum = source.new_zeros((dim_size, self.tail_num_bins, 1))
        if self.tail_num_bins == 0 or source.size(0) == 0:
            return source_sum, weight_sum, r_sum, r2_sum
        proxy_flat = proxy_distance.squeeze(-1)
        for bin_idx, (lower, upper) in enumerate(self._tail_bin_bounds()):
            mask = proxy_flat > lower
            if math.isfinite(upper):
                mask = mask & (proxy_flat <= upper)
            if not bool(mask.any()):
                continue
            selected_source = source[mask]
            selected_proxy = proxy_distance[mask]
            selected_dst = dst_index[mask]
            selected_weight = self._physical_kernel(selected_proxy.squeeze(-1)).unsqueeze(-1).to(dtype=source.dtype)
            source_sum[:, bin_idx, :] = _segment_sum(selected_source * selected_weight, selected_dst, dim_size)
            weight_sum[:, bin_idx, :] = _segment_sum(selected_weight, selected_dst, dim_size)
            r_sum[:, bin_idx, :] = _segment_sum(selected_proxy.to(dtype=source.dtype) * selected_weight, selected_dst, dim_size)
            r2_sum[:, bin_idx, :] = _segment_sum(
                selected_proxy.square().to(dtype=source.dtype) * selected_weight,
                selected_dst,
                dim_size,
            )
        return source_sum, weight_sum, r_sum, r2_sum

    def _compute_graph_tail_features(
        self,
        local_pos: torch.Tensor,
        local_source: torch.Tensor,
        local_edge_src: torch.Tensor | None,
        local_edge_dst: torch.Tensor | None,
        far_src: torch.Tensor,
        far_dst: torch.Tensor,
    ) -> torch.Tensor:
        n_local = local_pos.size(0)
        out = local_source.new_zeros((n_local, self.tail_num_bins, self.shell_feature_dim))
        if self.tail_num_bins == 0 or n_local == 0:
            return out
        graph_center = local_pos.mean(dim=0, keepdim=True)
        radial_to_center = torch.linalg.vector_norm(local_pos - graph_center, dim=-1, keepdim=True)
        proxy_distance = self._tail_proxy_distance(radial_to_center)

        graph_dst = torch.zeros((n_local,), device=local_pos.device, dtype=torch.long)
        graph_source_sum, graph_weight_sum, graph_r_sum, graph_r2_sum = self._accumulate_tail_bin_stats(
            local_source,
            proxy_distance,
            graph_dst,
            1,
        )

        local_contrib_src = torch.arange(n_local, device=local_pos.device, dtype=torch.long)
        local_contrib_dst = local_contrib_src
        if local_edge_src is not None and local_edge_dst is not None and local_edge_src.numel() > 0:
            local_contrib_src = torch.cat([local_contrib_src, local_edge_src], dim=0)
            local_contrib_dst = torch.cat([local_contrib_dst, local_edge_dst], dim=0)
        local_source_sum, local_weight_sum, local_r_sum, local_r2_sum = self._accumulate_tail_bin_stats(
            local_source.index_select(0, local_contrib_src),
            proxy_distance.index_select(0, local_contrib_src),
            local_contrib_dst,
            n_local,
        )

        explicit_source_sum, explicit_weight_sum, explicit_r_sum, explicit_r2_sum = self._accumulate_tail_bin_stats(
            local_source.index_select(0, far_src),
            proxy_distance.index_select(0, far_src),
            far_dst,
            n_local,
        )

        residual_source_sum = graph_source_sum.expand(n_local, -1, -1) - local_source_sum - explicit_source_sum
        residual_weight_sum = (graph_weight_sum.expand(n_local, -1, -1) - local_weight_sum - explicit_weight_sum).clamp_min(0.0)
        residual_r_sum = (graph_r_sum.expand(n_local, -1, -1) - local_r_sum - explicit_r_sum).clamp_min(0.0)
        residual_r2_sum = (graph_r2_sum.expand(n_local, -1, -1) - local_r2_sum - explicit_r2_sum).clamp_min(0.0)
        denom = residual_weight_sum.clamp_min(1.0e-8)
        mean_source = residual_source_sum / denom
        mean_r = residual_r_sum / denom
        rms_r = torch.sqrt(residual_r2_sum / denom + 1.0e-8)
        present = (residual_weight_sum > 0).to(dtype=local_source.dtype)
        out = torch.cat(
            [
                mean_source,
                residual_weight_sum,
                mean_r,
                rms_r,
            ],
            dim=-1,
        ) * present
        return out

    def _compute_graph_shell_features_from_edges(
        self,
        local_source: torch.Tensor,
        far_src: torch.Tensor,
        far_dst: torch.Tensor,
        far_dist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_local = local_source.size(0)
        out = local_source.new_zeros((n_local, self.far_num_shells, self.shell_feature_dim))
        explicit_count = _segment_sum(
            torch.ones((far_dst.numel(), 1), device=local_source.device, dtype=local_source.dtype),
            far_dst,
            n_local,
        )
        explicit_sum = _segment_sum(local_source.index_select(0, far_src), far_dst, n_local)
        shell_bounds = self._shell_bounds(device=local_source.device, dtype=local_source.dtype)
        for shell_idx, (lower, upper) in enumerate(shell_bounds):
            out[:, shell_idx, :] = self._summarize_sparse_edges(
                far_src,
                far_dst,
                far_dist,
                local_source,
                n_local,
                lower=lower,
                upper=upper,
            )
        return out, explicit_sum, explicit_count

    def _compute_graph_shell_features(
        self,
        local_pos: torch.Tensor,
        local_source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        far_src, far_dst, far_dist = self._build_sparse_far_edges(local_pos)
        return self._compute_graph_shell_features_from_edges(local_source, far_src, far_dst, far_dist)

    def forward(
        self,
        invariant_features: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        *,
        edge_src: torch.Tensor | None = None,
        edge_dst: torch.Tensor | None = None,
        return_source: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        num_graphs = cell.size(0)
        source = self.source_head(self.input_norm(invariant_features))
        source = self.source_norm(source)
        source = self._neutralize_source(source, batch, num_graphs)
        n_atoms = source.size(0)
        graph_counts = _segment_sum(
            torch.ones((n_atoms, 1), device=source.device, dtype=source.dtype),
            batch,
            num_graphs,
        )
        shell_features = source.new_zeros((n_atoms, self.total_shells, self.shell_feature_dim))
        for graph_idx in range(num_graphs):
            node_index = torch.nonzero(batch == graph_idx, as_tuple=False).view(-1)
            if node_index.numel() == 0:
                continue
            local_pos = pos.index_select(0, node_index)
            local_source = source.index_select(0, node_index)
            local_edge_src = None
            local_edge_dst = None
            if edge_src is not None and edge_dst is not None:
                edge_mask = (batch.index_select(0, edge_dst) == graph_idx)
                graph_edge_src = edge_src[edge_mask]
                graph_edge_dst = edge_dst[edge_mask]
                global_to_local = torch.full((n_atoms,), -1, device=batch.device, dtype=torch.long)
                global_to_local[node_index] = torch.arange(node_index.numel(), device=batch.device, dtype=torch.long)
                local_edge_src = global_to_local.index_select(0, graph_edge_src)
                local_edge_dst = global_to_local.index_select(0, graph_edge_dst)
            far_src, far_dst, far_dist = self._build_sparse_far_edges(local_pos)
            explicit_shells, _, _ = self._compute_graph_shell_features_from_edges(local_source, far_src, far_dst, far_dist)
            shell_features[node_index, :self.far_num_shells, :] = explicit_shells
            if self.tail_num_bins > 0:
                tail_features = self._compute_graph_tail_features(
                    local_pos,
                    local_source,
                    local_edge_src,
                    local_edge_dst,
                    far_src,
                    far_dst,
                )
                shell_features[node_index, self.far_num_shells:, :] = tail_features

        shell_flat = shell_features.reshape(n_atoms, self.total_shells * self.shell_feature_dim)
        shell_encoded = self.shell_encoder(shell_flat)
        gated_shell = self.far_gate * shell_encoded
        fused = torch.cat(
            [
                source,
                gated_shell,
                source * gated_shell,
                source - gated_shell,
                shell_flat,
            ],
            dim=-1,
        )
        atom_energy = self.energy_scale * self.energy_head(fused)
        if self.energy_partition == "uniform":
            graph_counts = _segment_sum(
                torch.ones((n_atoms, 1), device=source.device, dtype=source.dtype),
                batch,
                num_graphs,
            )
            graph_total = _segment_sum(atom_energy, batch, num_graphs)
            atom_energy = graph_total.index_select(0, batch) / graph_counts.index_select(0, batch).clamp_min(1.0)
        if return_source:
            return atom_energy, atom_energy.new_empty((atom_energy.size(0), 0))
        return atom_energy


class MeshLongRangeKernel3D(nn.Module):
    """Mesh/FFT reciprocal kernel with periodic/slab boundary support."""

    def __init__(
        self,
        *,
        mesh_size: int = 16,
        filter_hidden_dim: int = 64,
        boundary: str = "periodic",
        slab_padding_factor: int = 2,
        include_k0: bool = False,
        energy_partition: str = "potential",
        green_mode: str = "poisson",
        assignment: str = "cic",
        k_norm_floor: float = 1.0e-6,
    ):
        super().__init__()
        if boundary not in {"periodic", "slab"}:
            raise ValueError(f"Unsupported reciprocal mesh boundary: {boundary!r}")
        if energy_partition not in {"potential", "uniform"}:
            raise ValueError(f"Unsupported reciprocal energy partition: {energy_partition!r}")
        if assignment != "cic":
            raise ValueError(f"Unsupported mesh assignment: {assignment!r}")
        self.mesh_size = int(mesh_size)
        self.boundary = str(boundary)
        self.slab_padding_factor = max(int(slab_padding_factor), 1)
        self.include_k0 = bool(include_k0)
        self.energy_partition = str(energy_partition)
        self.green_mode = str(green_mode)
        self.assignment = str(assignment)
        self.k_norm_floor = float(k_norm_floor)
        self.green_kernel = ReciprocalGreenKernel(
            green_mode=self.green_mode,
            hidden_dim=int(filter_hidden_dim),
            k_norm_floor=self.k_norm_floor,
        )
        corner_offsets = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=torch.long,
        )
        self.register_buffer("corner_offsets", corner_offsets, persistent=False)

    @property
    def num_k(self) -> int:
        total = self.mesh_size * self.mesh_size * self.mesh_size
        return total if self.include_k0 else max(total - 1, 0)

    def build_k_norms(self, cell: torch.Tensor, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        effective_cell = _effective_cell_for_boundary(
            cell,
            boundary=self.boundary,
            slab_padding_factor=self.slab_padding_factor,
            dtype=dtype,
        )
        freq = _fft_integer_frequencies(self.mesh_size, device=cell.device, dtype=dtype)
        kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing="ij")
        integer_k = torch.stack([kx, ky, kz], dim=-1).reshape(-1, 3)
        inv_cell = torch.linalg.inv(effective_cell)
        k_cart = 2.0 * math.pi * torch.matmul(integer_k, inv_cell)
        k_norms = torch.linalg.vector_norm(k_cart, dim=-1).reshape(self.mesh_size, self.mesh_size, self.mesh_size)
        volume = torch.abs(torch.linalg.det(effective_cell)).clamp_min(self.k_norm_floor)
        return k_norms, volume

    def apply_green_kernel(self, mesh: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        mesh_dtype = mesh.dtype
        mesh_complex = torch.fft.fftn(mesh, dim=(0, 1, 2))
        k_norms, volume = self.build_k_norms(cell, dtype=mesh_dtype)
        spectral_weights = self.green_kernel(k_norms) / volume
        if not self.include_k0:
            spectral_weights = torch.where(
                k_norms > self.k_norm_floor,
                spectral_weights,
                torch.zeros_like(spectral_weights),
            )
        filtered = torch.fft.ifftn(mesh_complex * spectral_weights.unsqueeze(-1), dim=(0, 1, 2))
        return filtered.real

    def forward(self, pos: torch.Tensor, batch: torch.Tensor, cell: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        atom_energy = source.new_zeros((source.size(0), 1))
        counts = torch.bincount(batch, minlength=cell.size(0)).to(dtype=source.dtype).clamp_min(1.0)
        for graph_idx in range(cell.size(0)):
            node_index = torch.nonzero(batch == graph_idx, as_tuple=False).view(-1)
            if node_index.numel() == 0:
                continue
            local_pos = pos.index_select(0, node_index)
            local_source = source.index_select(0, node_index)
            local_frac = _prepare_frac_for_boundary(
                local_pos,
                cell[graph_idx],
                boundary=self.boundary,
                slab_padding_factor=self.slab_padding_factor,
            )
            mesh = _spread_source_to_mesh(
                local_frac,
                local_source,
                mesh_size=self.mesh_size,
                corner_offsets=self.corner_offsets,
                boundary=self.boundary,
            )
            potential_mesh = self.apply_green_kernel(mesh, cell[graph_idx])
            potential = _gather_source_from_mesh(
                local_frac,
                potential_mesh,
                mesh_size=self.mesh_size,
                corner_offsets=self.corner_offsets,
                boundary=self.boundary,
            )
            atom_energy_local = 0.5 * (local_source * potential).sum(dim=-1, keepdim=True)
            if self.energy_partition == "uniform":
                graph_total = atom_energy_local.sum()
                atom_energy_local = graph_total.expand_as(atom_energy_local) / counts[graph_idx]
            atom_energy.index_copy_(0, node_index, atom_energy_local)
        return atom_energy


class LatentReciprocalLongRange(nn.Module):
    """Periodic reciprocal-space prototype closer to a learnable Green's function."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        *,
        source_channels: int = 1,
        boundary: str = "periodic",
        neutralize: bool = True,
        kmax: int = 2,
        mesh_size: int = 16,
        filter_hidden_dim: int = 64,
        slab_padding_factor: int = 2,
        include_k0: bool = False,
        reciprocal_backend: str = "direct_kspace",
        energy_partition: str = "potential",
        green_mode: str = "poisson",
        assignment: str = "cic",
    ):
        super().__init__()
        if reciprocal_backend not in {"direct_kspace", "mesh_fft"}:
            raise ValueError(f"Unsupported reciprocal backend: {reciprocal_backend!r}")
        if reciprocal_backend == "direct_kspace" and boundary != "periodic":
            raise ValueError("direct_kspace reciprocal backend currently requires boundary='periodic'")
        if boundary not in {"periodic", "slab"}:
            raise ValueError(f"Unsupported long-range boundary mode: {boundary!r}")
        self.source_channels = int(source_channels)
        self.boundary = str(boundary)
        self.neutralize = bool(neutralize)
        self.kmax = int(kmax)
        self.mesh_size = int(mesh_size)
        self.slab_padding_factor = max(int(slab_padding_factor), 1)
        self.include_k0 = bool(include_k0)
        self.reciprocal_backend = str(reciprocal_backend)
        self.energy_partition = str(energy_partition)
        self.green_mode = str(green_mode)
        self.assignment = str(assignment)
        self.source_head = LatentSourceHead(feature_dim, hidden_dim, source_channels=self.source_channels)
        if self.reciprocal_backend == "mesh_fft":
            self.kernel = MeshLongRangeKernel3D(
                mesh_size=self.mesh_size,
                filter_hidden_dim=int(filter_hidden_dim),
                boundary=self.boundary,
                slab_padding_factor=self.slab_padding_factor,
                include_k0=self.include_k0,
                energy_partition=self.energy_partition,
                green_mode=self.green_mode,
                assignment=self.assignment,
            )
            self.exports_reciprocal_source = True
            final_linear = self.source_head.net[-1]
            if isinstance(final_linear, nn.Linear):
                nn.init.zeros_(final_linear.weight)
                nn.init.zeros_(final_linear.bias)
            self.energy_scale = None
        else:
            self.kernel = ReciprocalSpectralKernel3D(
                kmax=self.kmax,
                filter_hidden_dim=int(filter_hidden_dim),
                include_k0=self.include_k0,
                reciprocal_backend=self.reciprocal_backend,
                energy_partition=self.energy_partition,
            )
            self.exports_reciprocal_source = False
            # Keep the default contribution near zero so the module can be enabled
            # in existing workflows without destabilizing outputs before training.
            self.energy_scale = nn.Parameter(torch.tensor(0.0))

    @property
    def num_k(self) -> int:
        return self.kernel.num_k

    def _neutralize_source(self, source: torch.Tensor, batch: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        if not self.neutralize:
            return source
        graph_ids = torch.arange(cell.size(0), device=batch.device, dtype=batch.dtype)
        graph_mask = (batch.unsqueeze(1) == graph_ids.unsqueeze(0)).to(dtype=source.dtype)
        counts = graph_mask.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        graph_mean = torch.einsum("nb,nc->bc", graph_mask, source) / counts
        return source - graph_mean.index_select(0, batch)

    def forward(
        self,
        invariant_features: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        cell: torch.Tensor,
        *,
        edge_src: torch.Tensor | None = None,
        edge_dst: torch.Tensor | None = None,
        return_source: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        source = self.source_head(invariant_features)
        source = self._neutralize_source(source, batch, cell)
        atom_energy = self.kernel(pos, batch, cell, source)
        if self.energy_scale is not None:
            atom_energy = self.energy_scale * atom_energy
        if return_source:
            return atom_energy, source
        return atom_energy


def build_feature_spectral_module(
    *,
    mode: str,
    feature_dim: int,
    bottleneck_dim: int = 8,
    mesh_size: int = 16,
    filter_hidden_dim: int = 64,
    boundary: str = "periodic",
    slab_padding_factor: int = 2,
    neutralize: bool = True,
    include_k0: bool = False,
    gate_init: float = 0.0,
) -> nn.Module | None:
    if mode == "none":
        return None
    if mode == "fft":
        return FeatureSpectralResidualBlock(
            feature_dim=feature_dim,
            bottleneck_dim=bottleneck_dim,
            mesh_size=mesh_size,
            filter_hidden_dim=filter_hidden_dim,
            boundary=boundary,
            slab_padding_factor=slab_padding_factor,
            neutralize=neutralize,
            include_k0=include_k0,
            gate_init=gate_init,
        )
    raise ValueError(f"Unsupported feature_spectral_mode: {mode!r}")


def build_long_range_module(
    *,
    mode: str,
    feature_dim: int,
    hidden_dim: int = 64,
    boundary: str = "nonperiodic",
    neutralize: bool = True,
    filter_hidden_dim: int = 64,
    kmax: int = 2,
    mesh_size: int = 16,
    slab_padding_factor: int = 2,
    include_k0: bool = False,
    source_channels: int = 1,
    backend: str = "dense_pairwise",
    reciprocal_backend: str = "direct_kspace",
    energy_partition: str = "potential",
    green_mode: str = "poisson",
    assignment: str = "cic",
    theta: float = 0.5,
    leaf_size: int = 32,
    multipole_order: int = 0,
    far_source_dim: int = 16,
    far_num_shells: int = 3,
    far_shell_growth: float = 2.0,
    far_tail: bool = True,
    far_tail_bins: int = 2,
    far_stats: str = "mean,count,mean_r,rms_r",
    far_max_radius_multiplier: float | None = None,
    far_source_norm: bool = True,
    far_gate_init: float = 0.0,
    cutoff_radius: float = 5.0,
) -> nn.Module | None:
    if mode == "none":
        return None
    if mode == "latent-coulomb":
        return LatentCoulombLongRange(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            boundary=boundary,
            neutralize=neutralize,
            backend=backend,
            energy_partition=energy_partition,
            theta=theta,
            leaf_size=leaf_size,
            multipole_order=multipole_order,
        )
    if mode == "isolated-far-field-v1":
        return IsolatedFarFieldLongRange(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            boundary=boundary,
            neutralize=neutralize,
            energy_partition=energy_partition,
            filter_hidden_dim=filter_hidden_dim,
            source_channels=source_channels,
        )
    if mode == "isolated-far-field-v2":
        return IsolatedFarFieldLongRangeV2(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            boundary=boundary,
            neutralize=neutralize,
            energy_partition=energy_partition,
            far_source_dim=far_source_dim,
            far_num_shells=far_num_shells,
            far_shell_growth=far_shell_growth,
            far_tail=far_tail,
            far_tail_bins=far_tail_bins,
            far_stats=far_stats,
            far_max_radius_multiplier=far_max_radius_multiplier,
            far_source_norm=far_source_norm,
            far_gate_init=far_gate_init,
            cutoff_radius=cutoff_radius,
        )
    if mode == "reciprocal-spectral-v1":
        return LatentReciprocalLongRange(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            source_channels=source_channels,
            boundary=boundary,
            neutralize=neutralize,
            kmax=kmax,
            mesh_size=mesh_size,
            filter_hidden_dim=filter_hidden_dim,
            slab_padding_factor=slab_padding_factor,
            include_k0=include_k0,
            reciprocal_backend=reciprocal_backend,
            energy_partition=energy_partition,
            green_mode=green_mode,
            assignment=assignment,
        )
    raise ValueError(f"Unsupported long_range_mode: {mode!r}")


def configure_long_range_modules(
    owner: nn.Module,
    *,
    feature_dim: int,
    cutoff_radius: float,
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
) -> None:
    owner.long_range_mode = str(long_range_mode)
    owner.long_range_hidden_dim = int(long_range_hidden_dim)
    owner.long_range_boundary = str(long_range_boundary)
    owner.long_range_neutralize = bool(long_range_neutralize)
    owner.long_range_filter_hidden_dim = int(long_range_filter_hidden_dim)
    owner.long_range_kmax = int(long_range_kmax)
    owner.long_range_mesh_size = int(long_range_mesh_size)
    owner.long_range_slab_padding_factor = int(long_range_slab_padding_factor)
    owner.long_range_include_k0 = bool(long_range_include_k0)
    owner.long_range_source_channels = int(long_range_source_channels)
    owner.long_range_backend = str(long_range_backend)
    owner.long_range_reciprocal_backend = str(long_range_reciprocal_backend)
    owner.long_range_energy_partition = str(long_range_energy_partition)
    owner.long_range_green_mode = str(long_range_green_mode)
    owner.long_range_assignment = str(long_range_assignment)
    owner.long_range_theta = float(long_range_theta)
    owner.long_range_leaf_size = int(long_range_leaf_size)
    owner.long_range_multipole_order = int(long_range_multipole_order)
    owner.long_range_far_source_dim = int(long_range_far_source_dim)
    owner.long_range_far_num_shells = int(long_range_far_num_shells)
    owner.long_range_far_shell_growth = float(long_range_far_shell_growth)
    owner.long_range_far_tail = bool(long_range_far_tail)
    owner.long_range_far_tail_bins = int(long_range_far_tail_bins)
    owner.long_range_far_stats = str(long_range_far_stats)
    owner.long_range_far_max_radius_multiplier = (
        None if long_range_far_max_radius_multiplier is None else float(long_range_far_max_radius_multiplier)
    )
    owner.long_range_far_source_norm = bool(long_range_far_source_norm)
    owner.long_range_far_gate_init = float(long_range_far_gate_init)
    owner.feature_spectral_mode = str(feature_spectral_mode)
    owner.feature_spectral_bottleneck_dim = int(feature_spectral_bottleneck_dim)
    owner.feature_spectral_mesh_size = int(feature_spectral_mesh_size)
    owner.feature_spectral_filter_hidden_dim = int(feature_spectral_filter_hidden_dim)
    owner.feature_spectral_boundary = str(feature_spectral_boundary)
    owner.feature_spectral_slab_padding_factor = int(feature_spectral_slab_padding_factor)
    owner.feature_spectral_neutralize = bool(feature_spectral_neutralize)
    owner.feature_spectral_include_k0 = bool(feature_spectral_include_k0)
    owner.feature_spectral_gate_init = float(feature_spectral_gate_init)

    owner.long_range_module = build_long_range_module(
        mode=owner.long_range_mode,
        feature_dim=int(feature_dim),
        hidden_dim=owner.long_range_hidden_dim,
        boundary=owner.long_range_boundary,
        neutralize=owner.long_range_neutralize,
        filter_hidden_dim=owner.long_range_filter_hidden_dim,
        kmax=owner.long_range_kmax,
        mesh_size=owner.long_range_mesh_size,
        slab_padding_factor=owner.long_range_slab_padding_factor,
        include_k0=owner.long_range_include_k0,
        source_channels=owner.long_range_source_channels,
        backend=owner.long_range_backend,
        reciprocal_backend=owner.long_range_reciprocal_backend,
        energy_partition=owner.long_range_energy_partition,
        green_mode=owner.long_range_green_mode,
        assignment=owner.long_range_assignment,
        theta=owner.long_range_theta,
        leaf_size=owner.long_range_leaf_size,
        multipole_order=owner.long_range_multipole_order,
        far_source_dim=owner.long_range_far_source_dim,
        far_num_shells=owner.long_range_far_num_shells,
        far_shell_growth=owner.long_range_far_shell_growth,
        far_tail=owner.long_range_far_tail,
        far_tail_bins=owner.long_range_far_tail_bins,
        far_stats=owner.long_range_far_stats,
        far_max_radius_multiplier=owner.long_range_far_max_radius_multiplier,
        far_source_norm=owner.long_range_far_source_norm,
        far_gate_init=owner.long_range_far_gate_init,
        cutoff_radius=float(cutoff_radius),
    )
    owner.long_range_num_k = (
        getattr(owner.long_range_module, "num_k", None) if owner.long_range_module is not None else None
    )
    owner.feature_spectral_module = build_feature_spectral_module(
        mode=owner.feature_spectral_mode,
        feature_dim=int(feature_dim),
        bottleneck_dim=owner.feature_spectral_bottleneck_dim,
        mesh_size=owner.feature_spectral_mesh_size,
        filter_hidden_dim=owner.feature_spectral_filter_hidden_dim,
        boundary=owner.feature_spectral_boundary,
        slab_padding_factor=owner.feature_spectral_slab_padding_factor,
        neutralize=owner.feature_spectral_neutralize,
        include_k0=owner.feature_spectral_include_k0,
        gate_init=owner.feature_spectral_gate_init,
    )
    owner.long_range_runtime_backend = "none"
    owner.long_range_runtime_source_kind = "none"
    owner.long_range_runtime_source_channels = 0
    owner.long_range_runtime_source_layout = "none"
    owner.long_range_runtime_source_boundary = owner.long_range_boundary
    owner.long_range_runtime_source_slab_padding_factor = owner.long_range_slab_padding_factor
    if owner.long_range_module is not None and bool(getattr(owner.long_range_module, "exports_reciprocal_source", False)):
        owner.long_range_runtime_backend = str(getattr(owner.long_range_module, "runtime_backend", "none"))
        owner.long_range_runtime_source_kind = str(getattr(owner.long_range_module, "source_kind", "latent_charge"))
        owner.long_range_runtime_source_channels = int(
            getattr(owner.long_range_module, "source_channels", owner.long_range_source_channels)
        )
        owner.long_range_runtime_source_layout = str(getattr(owner.long_range_module, "source_layout", "channels_last"))
        owner.long_range_runtime_source_boundary = owner.long_range_boundary
        owner.long_range_runtime_source_slab_padding_factor = owner.long_range_slab_padding_factor
        owner.reciprocal_source_channels = owner.long_range_runtime_source_channels
        owner.reciprocal_source_boundary = owner.long_range_runtime_source_boundary
        owner.reciprocal_source_slab_padding_factor = owner.long_range_runtime_source_slab_padding_factor
    elif owner.feature_spectral_module is not None:
        owner.long_range_runtime_backend = "mesh_fft"
        owner.long_range_runtime_source_kind = "feature_bottleneck"
        owner.long_range_runtime_source_channels = owner.feature_spectral_bottleneck_dim
        owner.long_range_runtime_source_layout = "channels_last"
        owner.long_range_runtime_source_boundary = owner.feature_spectral_boundary
        owner.long_range_runtime_source_slab_padding_factor = owner.feature_spectral_slab_padding_factor
        owner.reciprocal_source_channels = owner.feature_spectral_bottleneck_dim
        owner.reciprocal_source_boundary = owner.feature_spectral_boundary
        owner.reciprocal_source_slab_padding_factor = owner.feature_spectral_slab_padding_factor
    else:
        owner.reciprocal_source_channels = 0
        owner.reciprocal_source_boundary = "periodic"
        owner.reciprocal_source_slab_padding_factor = 2


def apply_long_range_modules(
    owner: nn.Module,
    invariant_features: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    cell: torch.Tensor,
    *,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    return_reciprocal_source: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]:
    feature_reciprocal_source = None
    if getattr(owner, "feature_spectral_module", None) is not None:
        invariant_features, feature_reciprocal_source = owner.feature_spectral_module(
            invariant_features,
            pos,
            batch,
            cell,
        )

    long_range_energy = None
    reciprocal_source = None
    defer_long_range_to_runtime = False
    if getattr(owner, "long_range_module", None) is not None:
        if return_reciprocal_source and bool(getattr(owner.long_range_module, "exports_reciprocal_source", False)):
            long_range_energy, reciprocal_source = owner.long_range_module(
                invariant_features,
                pos,
                batch,
                cell,
                edge_src=edge_src,
                edge_dst=edge_dst,
                return_source=True,
            )
            defer_long_range_to_runtime = reciprocal_source.numel() > 0
        else:
            long_range_energy = owner.long_range_module(
                invariant_features,
                pos,
                batch,
                cell,
                edge_src=edge_src,
                edge_dst=edge_dst,
            )
    if reciprocal_source is None:
        reciprocal_source = feature_reciprocal_source
    return invariant_features, long_range_energy, reciprocal_source, defer_long_range_to_runtime
