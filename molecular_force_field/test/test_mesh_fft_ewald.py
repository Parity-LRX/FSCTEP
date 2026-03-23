from __future__ import annotations

import math

import pytest
import torch

from molecular_force_field.models.long_range import (
    FeatureSpectralResidualBlock,
    MeshLongRangeKernel3D,
    _gather_source_from_mesh,
    _prepare_frac_for_boundary,
    _spread_source_to_mesh,
)


def _devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def test_mesh_fft_default_stays_reciprocal_only() -> None:
    kernel = MeshLongRangeKernel3D(
        mesh_size=8,
        boundary="periodic",
        energy_partition="potential",
        green_mode="poisson",
        assignment="cic",
    ).to(dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64) * 8.0

    spectral_weights, alpha, volume, real_cutoff = kernel._build_reciprocal_spectral_weights(cell, dtype=torch.float64)
    k_norms, _ = kernel.build_k_norms(cell, dtype=torch.float64)

    assert alpha is None
    assert real_cutoff is None
    assert spectral_weights[0, 0, 0].item() == 0.0

    probe = (1, 0, 0)
    k_norm = k_norms[probe]
    expected = (4.0 * math.pi / (k_norm * k_norm)) / volume
    assert torch.allclose(spectral_weights[probe], expected, atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.parametrize(
    ("assignment", "expected_points", "expected_exponent"),
    [
        ("cic", 8, 2),
        ("tsc", 27, 3),
        ("pcs", 64, 4),
    ],
)
def test_mesh_fft_assignment_stencil_and_window(
    assignment: str,
    expected_points: int,
    expected_exponent: int,
) -> None:
    kernel = MeshLongRangeKernel3D(
        mesh_size=8,
        boundary="periodic",
        energy_partition="potential",
        green_mode="poisson",
        assignment=assignment,
        full_ewald=True,
    ).to(dtype=torch.float64)
    frac = torch.tensor([[0.37, 0.29, 0.83]], dtype=torch.float64)
    source = torch.tensor([[1.25, -0.5]], dtype=torch.float64)
    mesh = _spread_source_to_mesh(
        frac,
        source,
        mesh_size=kernel.mesh_size,
        assignment=kernel.assignment,
        assignment_offsets=kernel.assignment_offsets,
        boundary=kernel.boundary,
    )
    gathered = _gather_source_from_mesh(
        frac,
        torch.ones_like(mesh),
        mesh_size=kernel.mesh_size,
        assignment=kernel.assignment,
        assignment_offsets=kernel.assignment_offsets,
        boundary=kernel.boundary,
    )
    assignment_window = kernel._build_assignment_window(device=frac.device, dtype=torch.float64)
    expected_1d = torch.sinc(torch.tensor(1.0 / kernel.mesh_size, dtype=torch.float64)).pow(expected_exponent)

    assert kernel.assignment_offsets.shape == (expected_points, 3)
    assert torch.allclose(mesh.sum(dim=(0, 1, 2)), source.squeeze(0), atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(gathered, torch.ones_like(gathered), atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(assignment_window[1, 0, 0], expected_1d, atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.parametrize("assignment", ["cic", "tsc", "pcs"])
def test_feature_spectral_fft_supports_high_order_assignment(assignment: str) -> None:
    block = FeatureSpectralResidualBlock(
        feature_dim=6,
        bottleneck_dim=3,
        mesh_size=8,
        filter_hidden_dim=8,
        boundary="periodic",
        assignment=assignment,
    ).to(dtype=torch.float64)
    invariant = torch.randn(5, 6, dtype=torch.float64)
    pos = torch.tensor(
        [
            [0.2, 0.4, 0.6],
            [1.0, 1.4, 1.8],
            [2.1, 2.3, 2.7],
            [3.4, 3.2, 1.5],
            [4.6, 0.8, 2.9],
        ],
        dtype=torch.float64,
    )
    batch = torch.zeros(5, dtype=torch.long)
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 8.0

    out, source = block(invariant, pos, batch, cell)

    assert out.shape == invariant.shape
    assert source.shape == (5, 3)
    assert torch.isfinite(out).all()
    assert torch.isfinite(source).all()


@pytest.mark.parametrize("device", _devices())
def test_mesh_fft_reciprocal_weights_match_ewald_formula(device: torch.device) -> None:
    kernel = MeshLongRangeKernel3D(
        mesh_size=8,
        boundary="periodic",
        energy_partition="potential",
        green_mode="poisson",
        assignment="cic",
        full_ewald=True,
    ).to(device=device, dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64, device=device) * 8.0

    spectral_weights, alpha, volume, _ = kernel._build_reciprocal_spectral_weights(cell, dtype=torch.float64)
    k_norms, _ = kernel.build_k_norms(cell, dtype=torch.float64)
    assignment_window = kernel._build_assignment_window(device=cell.device, dtype=torch.float64)

    assert spectral_weights[0, 0, 0].item() == 0.0

    probe = (1, 0, 0)
    k_norm = k_norms[probe]
    expected = (
        (4.0 * math.pi / (k_norm * k_norm))
        * torch.exp(-(k_norm * k_norm) / (4.0 * alpha * alpha))
        / volume
        / assignment_window[probe].square()
    )
    assert torch.allclose(spectral_weights[probe], expected, atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.parametrize("device", _devices())
def test_mesh_fft_forward_adds_real_self_and_background_terms(device: torch.device) -> None:
    kernel = MeshLongRangeKernel3D(
        mesh_size=16,
        boundary="periodic",
        energy_partition="potential",
        green_mode="poisson",
        assignment="cic",
        full_ewald=True,
    ).to(device=device, dtype=torch.float64)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
        device=device,
    )
    batch = torch.zeros(pos.size(0), dtype=torch.long)
    batch = batch.to(device=device)
    cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 8.0
    source = torch.tensor([[1.0], [0.25]], dtype=torch.float64, device=device)

    frac = _prepare_frac_for_boundary(pos, cell[0], boundary="periodic", slab_padding_factor=2)
    mesh = _spread_source_to_mesh(
        frac,
        source,
        mesh_size=kernel.mesh_size,
        assignment=kernel.assignment,
        assignment_offsets=kernel.assignment_offsets,
        boundary=kernel.boundary,
    )
    potential_mesh, alpha, volume, real_cutoff = kernel.apply_green_kernel(mesh, cell[0])
    assert alpha is not None
    assert real_cutoff is not None
    reciprocal_potential = _gather_source_from_mesh(
        frac,
        potential_mesh,
        mesh_size=kernel.mesh_size,
        assignment=kernel.assignment,
        assignment_offsets=kernel.assignment_offsets,
        boundary=kernel.boundary,
    )
    real_space_potential = kernel._compute_real_space_potential(
        pos,
        source,
        cell[0],
        alpha=alpha,
        real_cutoff=real_cutoff,
    )
    self_potential = (-2.0 * alpha / math.sqrt(math.pi)) * source
    background_potential = -math.pi * source.sum(dim=0, keepdim=True) / (alpha.square() * volume)
    manual = 0.5 * (
        source * (reciprocal_potential + real_space_potential + self_potential + background_potential)
    ).sum(dim=-1, keepdim=True)

    atom_energy = kernel(pos, batch, cell, source)

    assert torch.allclose(atom_energy, manual, atol=1.0e-12, rtol=1.0e-12)
    assert torch.linalg.vector_norm(real_space_potential).item() > 0.0
    assert torch.linalg.vector_norm(self_potential).item() > 0.0
    assert torch.linalg.vector_norm(background_potential).item() > 0.0


@pytest.mark.parametrize("device", _devices())
def test_mesh_fft_slab_real_space_uses_only_xy_images(device: torch.device) -> None:
    kernel = MeshLongRangeKernel3D(
        mesh_size=8,
        boundary="slab",
        slab_padding_factor=2,
        energy_partition="potential",
        green_mode="poisson",
        assignment="cic",
        full_ewald=True,
    ).to(device=device, dtype=torch.float64)
    cell = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 20.0]],
        dtype=torch.float64,
        device=device,
    )

    shift_index = kernel.real_space_shift_index.to(device=cell.device)

    assert shift_index.shape == (9, 3)
    assert torch.equal(torch.unique(shift_index[:, 2]), torch.tensor([0], dtype=torch.long, device=device))
