"""Shared radial basis utilities with MACE-style polynomial cutoff."""

from __future__ import annotations

import math

import torch
from e3nn.math import soft_one_hot_linspace as _e3nn_soft_one_hot_linspace


def mace_polynomial_cutoff(
    x: torch.Tensor,
    r_max: float | torch.Tensor,
    p: int = 50,
) -> torch.Tensor:
    """MACE polynomial envelope, C^(p-1) smooth at the cutoff.

    The polynomial is 1 at r=0 and has zero value and zero derivatives up to
    order p - 1 at r=r_max.  This mirrors MACE's PolynomialCutoff formula.
    """
    p_int = int(p)
    p_f = float(p_int)
    # Use Python scalars instead of host->device tensors (torch.as_tensor) so this
    # forward is CUDA-graph capturable. tensor/scalar ops cast the scalar to
    # x.dtype, so the result is numerically identical to the tensor version.
    if isinstance(r_max, torch.Tensor):
        r = x / r_max
        within = (x < r_max).to(dtype=x.dtype)
    else:
        r_max_f = float(r_max)
        r = x / r_max_f
        within = (x < r_max_f).to(dtype=x.dtype)
    envelope = (
        1.0
        - ((p_f + 1.0) * (p_f + 2.0) / 2.0) * torch.pow(r, p_int)
        + p_f * (p_f + 2.0) * torch.pow(r, p_int + 1)
        - (p_f * (p_f + 1.0) / 2.0) * torch.pow(r, p_int + 2)
    )
    return envelope * within


def mace_radial_embedding(
    edge_length: torch.Tensor,
    *,
    r_max: float,
    number_of_basis: int,
    function_type: str,
    polynomial_cutoff_p: int | None = None,
) -> torch.Tensor:
    """e3nn radial basis, optionally multiplied by MACE polynomial envelope.

    polynomial_cutoff_p=None (default): raw e3nn bessel basis. Best for
        MD17 single-point regression accuracy (E sub-1 meV/atom).
    polynomial_cutoff_p=int: apply MACE polynomial envelope of that order.
        Adds C^(p-1) smoothness at r_max (useful for MD inference / energy
        conservation in long NVE trajectories) at the cost of ~2x worse E MAE
        in training due to plateau-schedule delay from gradient noise.
    """
    emb = _e3nn_soft_one_hot_linspace(
        edge_length,
        0.0,
        float(r_max),
        int(number_of_basis),
        basis=str(function_type),
        cutoff=True,
    )
    if polynomial_cutoff_p is not None:
        envelope = mace_polynomial_cutoff(edge_length, float(r_max), int(polynomial_cutoff_p))
        emb = emb * envelope.unsqueeze(-1)
    return emb.mul(math.sqrt(int(number_of_basis)))


def soft_one_hot_linspace_mace_cutoff(
    x: torch.Tensor,
    start: float,
    end: float,
    number: int,
    *,
    basis: str = "gaussian",
    cutoff: bool = True,
    polynomial_cutoff_p: int = 50,
) -> torch.Tensor:
    """e3nn-compatible radial basis with MACE polynomial cutoff when requested."""
    emb = _e3nn_soft_one_hot_linspace(
        x,
        float(start),
        float(end),
        int(number),
        basis=str(basis),
        cutoff=False,
    )
    if not cutoff:
        return emb
    envelope = mace_polynomial_cutoff(x, float(end), int(polynomial_cutoff_p))
    return emb * envelope.unsqueeze(-1)
