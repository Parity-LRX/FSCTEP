from __future__ import annotations

import torch

from molecular_force_field.models.ictd_irreps import direction_harmonics_all


def test_direction_harmonics_second_order_backward_is_finite() -> None:
    raw = torch.tensor(
        [
            [-0.8, 0.3, 0.5],
            [0.2, -0.9, 0.4],
            [-0.6, -0.1, 0.7],
            [0.5, 0.4, -0.75],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    n = raw / raw.norm(dim=-1, keepdim=True)
    harmonics = direction_harmonics_all(n, 2)

    first_order_seed = sum(block.square().sum() for block in harmonics[1:])
    first_grad = torch.autograd.grad(first_order_seed, raw, create_graph=True)[0]
    second_order_seed = first_grad.square().sum()
    second_order_seed.backward()

    assert raw.grad is not None
    assert torch.isfinite(first_grad).all()
    assert torch.isfinite(raw.grad).all()
