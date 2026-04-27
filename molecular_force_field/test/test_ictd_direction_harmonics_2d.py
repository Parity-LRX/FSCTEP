from __future__ import annotations

import math

import pytest
import torch

from molecular_force_field.models.ictd_irreps_2d import (
    direction_harmonics_all_so2,
    direction_harmonics_irreps_o2,
    direction_harmonics_so2,
)


def _rotate_2d(v: torch.Tensor, angle: float) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    R = torch.tensor([[c, -s], [s, c]], dtype=v.dtype, device=v.device)
    return v @ R.T


def _so2_row_rotation(m: int, angle: float, *, dtype: torch.dtype) -> torch.Tensor:
    c = math.cos(m * angle)
    s = math.sin(m * angle)
    return torch.tensor([[c, s], [-s, c]], dtype=dtype)


def test_direction_harmonics_all_so2_shapes() -> None:
    n = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    out = direction_harmonics_all_so2(n, 4)
    assert len(out) == 5
    assert out[0].shape == (2, 1)
    for m in range(1, 5):
        assert out[m].shape == (2, 2)


def test_direction_harmonics_so2_rotates_by_frequency() -> None:
    raw = torch.tensor(
        [
            [0.8, 0.3],
            [0.2, 0.9],
            [-0.6, 0.5],
        ],
        dtype=torch.float64,
    )
    n = raw / raw.norm(dim=-1, keepdim=True)
    angle = 0.37
    n_rot = _rotate_2d(n, angle)

    for m in range(1, 5):
        y = direction_harmonics_so2(n, m)
        y_rot = direction_harmonics_so2(n_rot, m)
        expected = y @ _so2_row_rotation(m, angle, dtype=y.dtype)
        torch.testing.assert_close(y_rot, expected, rtol=1e-10, atol=1e-10)


def test_direction_harmonics_so2_reflection_is_diag_one_minus_one() -> None:
    raw = torch.tensor(
        [
            [0.8, 0.3],
            [0.2, 0.9],
            [-0.6, 0.5],
        ],
        dtype=torch.float64,
    )
    n = raw / raw.norm(dim=-1, keepdim=True)
    n_ref = torch.stack([n[..., 0], -n[..., 1]], dim=-1)
    D = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float64)

    for m in range(1, 5):
        y = direction_harmonics_so2(n, m)
        y_ref = direction_harmonics_so2(n_ref, m)
        expected = y @ D
        torch.testing.assert_close(y_ref, expected, rtol=1e-10, atol=1e-10)


def test_direction_harmonics_o2_rejects_odd_scalar() -> None:
    n = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="0o"):
        direction_harmonics_irreps_o2(n, "0o")


def test_direction_harmonics_second_order_backward_is_finite_2d() -> None:
    raw = torch.tensor(
        [
            [-0.8, 0.3],
            [0.2, -0.9],
            [-0.6, -0.1],
            [0.5, 0.4],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    n = raw / raw.norm(dim=-1, keepdim=True)
    harmonics = direction_harmonics_all_so2(n, 3)

    first_seed = sum(block.square().sum() for block in harmonics[1:])
    first_grad = torch.autograd.grad(first_seed, raw, create_graph=True)[0]
    second_seed = first_grad.square().sum()
    second_seed.backward()

    assert raw.grad is not None
    assert torch.isfinite(first_grad).all()
    assert torch.isfinite(raw.grad).all()

