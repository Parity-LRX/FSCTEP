import torch

from molecular_force_field.models.radial_basis import (
    mace_polynomial_cutoff,
    mace_radial_embedding,
    soft_one_hot_linspace_mace_cutoff,
)


def test_mace_polynomial_cutoff_boundary_values():
    r = torch.tensor([0.0, 4.999, 5.0, 5.001], dtype=torch.float64)
    cutoff = mace_polynomial_cutoff(r, 5.0, p=6)

    assert torch.allclose(cutoff[0], torch.tensor(1.0, dtype=torch.float64))
    assert torch.allclose(cutoff[2:], torch.zeros(2, dtype=torch.float64))
    assert cutoff[1] > 0.0


def test_radial_embedding_uses_mace_cutoff_not_e3nn_cutoff():
    r = torch.tensor([0.25, 5.0, 5.25], dtype=torch.float64)
    emb = mace_radial_embedding(
        r,
        r_max=5.0,
        number_of_basis=8,
        function_type="bessel",
    )

    assert emb.shape == (3, 8)
    assert torch.isfinite(emb).all()
    assert torch.allclose(emb[1:], torch.zeros_like(emb[1:]))
    assert emb[0].abs().sum() > 0.0


def test_e3nn_compatible_wrapper_keeps_shape_and_cutoff():
    r = torch.linspace(0.0, 6.0, 13, dtype=torch.float64)
    emb = soft_one_hot_linspace_mace_cutoff(
        r,
        0.0,
        5.0,
        6,
        basis="gaussian",
        cutoff=True,
    )

    assert emb.shape == (13, 6)
    assert torch.allclose(emb[r >= 5.0], torch.zeros_like(emb[r >= 5.0]))
