from __future__ import annotations

import argparse
import json
import math
import time

import torch

from molecular_force_field.models.long_range import (
    MeshLongRangeKernel3D,
    ReciprocalGreenKernel,
    _build_assignment_offsets,
    _effective_cell_for_boundary,
    _fft_integer_frequencies,
    _gather_source_from_mesh,
    _prepare_frac_for_boundary,
    _spread_source_to_mesh,
)


class LegacyFullEwaldMeshLongRangeKernel3D(torch.nn.Module):
    """Pre-optimization full-Ewald mesh FFT path used for regression benchmarks."""

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
    ) -> None:
        super().__init__()
        self.mesh_size = int(mesh_size)
        self.boundary = str(boundary)
        self.slab_padding_factor = max(int(slab_padding_factor), 1)
        self.include_k0 = bool(include_k0)
        self.energy_partition = str(energy_partition)
        self.green_mode = str(green_mode)
        self.assignment = str(assignment)
        self.k_norm_floor = float(k_norm_floor)
        self.ewald_alpha_prefactor = 5.0
        self.assignment_window_floor = 1.0e-6
        self.green_kernel = ReciprocalGreenKernel(
            green_mode=self.green_mode,
            hidden_dim=int(filter_hidden_dim),
            k_norm_floor=self.k_norm_floor,
        )
        assignment_offsets = _build_assignment_offsets(self.assignment)
        self.register_buffer("assignment_offsets", assignment_offsets, persistent=False)

    def _periodic_axes(self) -> tuple[bool, bool, bool]:
        if self.boundary == "periodic":
            return True, True, True
        return True, True, False

    def _estimate_real_cutoff(self, cell: torch.Tensor) -> torch.Tensor:
        periodic_axes = torch.tensor(self._periodic_axes(), device=cell.device, dtype=torch.bool)
        periodic_vectors = cell[periodic_axes]
        periodic_lengths = torch.linalg.vector_norm(periodic_vectors, dim=-1)
        return 0.5 * periodic_lengths.min().clamp_min(self.k_norm_floor)

    def _estimate_ewald_alpha(self, real_cutoff: torch.Tensor) -> torch.Tensor:
        return real_cutoff.new_tensor(self.ewald_alpha_prefactor) / real_cutoff.clamp_min(self.k_norm_floor)

    def _build_assignment_window(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        freq = _fft_integer_frequencies(self.mesh_size, device=device, dtype=dtype)
        sinc = torch.sinc(freq / float(self.mesh_size)).square()
        wx, wy, wz = torch.meshgrid(sinc, sinc, sinc, indexing="ij")
        return wx * wy * wz

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

    def _build_reciprocal_spectral_weights(
        self,
        cell: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real_cutoff = self._estimate_real_cutoff(cell.to(dtype=dtype))
        alpha = self._estimate_ewald_alpha(real_cutoff)
        k_norms, volume = self.build_k_norms(cell, dtype=dtype)
        spectral_weights = self.green_kernel(k_norms) / volume
        spectral_weights = spectral_weights * torch.exp(-(k_norms.square()) / (4.0 * alpha * alpha))
        if self.assignment == "cic":
            assignment_window = self._build_assignment_window(device=cell.device, dtype=dtype)
            assignment_scale = torch.reciprocal(assignment_window.clamp_min(self.assignment_window_floor).square())
            spectral_weights = spectral_weights * assignment_scale
        spectral_weights = torch.where(
            k_norms > self.k_norm_floor,
            spectral_weights,
            torch.zeros_like(spectral_weights),
        )
        return spectral_weights, alpha, volume

    def apply_green_kernel(self, mesh: torch.Tensor, cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mesh_dtype = mesh.dtype
        mesh_complex = torch.fft.fftn(mesh, dim=(0, 1, 2))
        spectral_weights, alpha, volume = self._build_reciprocal_spectral_weights(cell, dtype=mesh_dtype)
        filtered = torch.fft.ifftn(mesh_complex * spectral_weights.unsqueeze(-1), dim=(0, 1, 2))
        return filtered.real, alpha, volume

    def _build_real_space_shifts(self, cell: torch.Tensor) -> torch.Tensor:
        ranges: list[torch.Tensor] = []
        for is_periodic in self._periodic_axes():
            if is_periodic:
                ranges.append(torch.tensor([-1, 0, 1], device=cell.device, dtype=torch.long))
            else:
                ranges.append(torch.tensor([0], device=cell.device, dtype=torch.long))
        return torch.cartesian_prod(*ranges)

    def _compute_real_space_potential(
        self,
        pos: torch.Tensor,
        source: torch.Tensor,
        cell: torch.Tensor,
        *,
        alpha: torch.Tensor,
        real_cutoff: torch.Tensor,
    ) -> torch.Tensor:
        if pos.size(0) == 0:
            return source.new_zeros(source.shape)
        shift_index = self._build_real_space_shifts(cell)
        shift_cart = torch.matmul(shift_index.to(dtype=pos.dtype), cell.to(dtype=pos.dtype))
        disp = pos.unsqueeze(1).unsqueeze(2) - pos.unsqueeze(0).unsqueeze(2) - shift_cart.unsqueeze(0).unsqueeze(0)
        distance = torch.linalg.vector_norm(disp, dim=-1)
        kernel = torch.special.erfc(alpha * distance) / distance.clamp_min(self.k_norm_floor)
        valid = distance <= real_cutoff
        zero_shift = (shift_index == 0).all(dim=1)
        self_mask = torch.eye(pos.size(0), device=pos.device, dtype=torch.bool).unsqueeze(-1) & zero_shift.view(1, 1, -1)
        kernel = kernel * (valid & (~self_mask)).to(dtype=source.dtype)
        pair_kernel = kernel.sum(dim=-1)
        return torch.matmul(pair_kernel, source)

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
                assignment=self.assignment,
                assignment_offsets=self.assignment_offsets,
                boundary=self.boundary,
            )
            potential_mesh, alpha, volume = self.apply_green_kernel(mesh, cell[graph_idx])
            potential = _gather_source_from_mesh(
                local_frac,
                potential_mesh,
                mesh_size=self.mesh_size,
                assignment=self.assignment,
                assignment_offsets=self.assignment_offsets,
                boundary=self.boundary,
            )
            real_cutoff = self._estimate_real_cutoff(cell[graph_idx].to(dtype=local_pos.dtype))
            real_space_potential = self._compute_real_space_potential(
                local_pos,
                local_source,
                cell[graph_idx],
                alpha=alpha.to(dtype=local_pos.dtype),
                real_cutoff=real_cutoff,
            )
            self_potential = (-2.0 * alpha.to(dtype=local_source.dtype) / math.sqrt(math.pi)) * local_source
            background_potential = -math.pi * local_source.sum(dim=0, keepdim=True) / (
                alpha.to(dtype=local_source.dtype).square() * volume.to(dtype=local_source.dtype)
            )
            atom_energy_local = 0.5 * (
                local_source * (potential + real_space_potential + self_potential + background_potential)
            ).sum(dim=-1, keepdim=True)
            if self.energy_partition == "uniform":
                graph_total = atom_energy_local.sum()
                atom_energy_local = graph_total.expand_as(atom_energy_local) / counts[graph_idx]
            atom_energy.index_copy_(0, node_index, atom_energy_local)
        return atom_energy


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench(module: torch.nn.Module, pos: torch.Tensor, batch: torch.Tensor, cell: torch.Tensor, source: torch.Tensor, *, iters: int, warmup: int) -> float:
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            module(pos, batch, cell, source)
        _sync(pos.device)
        t0 = time.perf_counter()
        for _ in range(iters):
            module(pos, batch, cell, source)
        _sync(pos.device)
        t1 = time.perf_counter()
    return (t1 - t0) / float(iters)


def build_inputs(*, n_atoms: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * 24.0
    coords = torch.linspace(0.0, 18.0, steps=n_atoms, dtype=dtype, device=device)
    pos = torch.stack(
        [
            torch.remainder(coords * 1.0, 20.0) + 1.0,
            torch.remainder(coords * 1.7, 20.0) + 1.0,
            torch.remainder(coords * 2.3, 20.0) + 1.0,
        ],
        dim=-1,
    )
    batch = torch.zeros(n_atoms, dtype=torch.long, device=device)
    source = torch.randn(n_atoms, 1, dtype=dtype, device=device)
    return pos, batch, cell, source


def run_once(
    device: torch.device,
    *,
    n_atoms: int,
    mesh_size: int,
    iters: int,
    warmup: int,
    assignment: str,
) -> dict[str, float | int | str]:
    dtype = torch.float64
    pos, batch, cell, source = build_inputs(n_atoms=n_atoms, dtype=dtype, device=device)
    reciprocal_mod = MeshLongRangeKernel3D(mesh_size=mesh_size, assignment=assignment, full_ewald=False).to(device=device, dtype=dtype)
    legacy_ewald_mod = LegacyFullEwaldMeshLongRangeKernel3D(mesh_size=mesh_size, assignment=assignment).to(device=device, dtype=dtype)
    optimized_ewald_mod = MeshLongRangeKernel3D(mesh_size=mesh_size, assignment=assignment, full_ewald=True).to(device=device, dtype=dtype)
    reciprocal_sec = bench(reciprocal_mod, pos, batch, cell, source, iters=iters, warmup=warmup)
    legacy_ewald_sec = bench(legacy_ewald_mod, pos, batch, cell, source, iters=iters, warmup=warmup)
    optimized_ewald_sec = bench(optimized_ewald_mod, pos, batch, cell, source, iters=iters, warmup=warmup)
    return {
        "device": str(device),
        "assignment": assignment,
        "n_atoms": n_atoms,
        "mesh_size": mesh_size,
        "iters": iters,
        "reciprocal_only_seconds_per_forward": reciprocal_sec,
        "legacy_full_ewald_seconds_per_forward": legacy_ewald_sec,
        "optimized_full_ewald_seconds_per_forward": optimized_ewald_sec,
        "ewald_optimization_delta_seconds": optimized_ewald_sec - legacy_ewald_sec,
        "ewald_optimization_delta_percent": (optimized_ewald_sec - legacy_ewald_sec) / legacy_ewald_sec * 100.0,
        "full_ewald_vs_reciprocal_delta_seconds": optimized_ewald_sec - reciprocal_sec,
        "full_ewald_vs_reciprocal_delta_percent": (optimized_ewald_sec - reciprocal_sec) / reciprocal_sec * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-atoms", type=int, default=128)
    parser.add_argument("--mesh-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--assignment", type=str, default="cic", choices=["cic", "tsc", "pcs"])
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)

    results = [
        run_once(
            torch.device("cpu"),
            n_atoms=args.n_atoms,
            mesh_size=args.mesh_size,
            iters=args.iters,
            warmup=args.warmup,
            assignment=args.assignment,
        )
    ]
    if (not args.cpu_only) and torch.cuda.is_available():
        results.append(
            run_once(
                torch.device("cuda"),
                n_atoms=args.n_atoms,
                mesh_size=args.mesh_size,
                iters=args.iters,
                warmup=args.warmup,
                assignment=args.assignment,
            )
        )

    for row in results:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
