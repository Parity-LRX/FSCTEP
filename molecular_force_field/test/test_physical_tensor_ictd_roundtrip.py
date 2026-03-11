#!/usr/bin/env python3
from __future__ import annotations

import torch

from molecular_force_field.models.pure_cartesian_ictd_layers import (
    PhysicalTensorICTDEmbedding,
    PhysicalTensorICTDRecovery,
)


def _assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-6) -> None:
    if not torch.allclose(got, ref, atol=atol, rtol=rtol):
        diff = (got - ref).abs().max().item()
        raise AssertionError(f"{name} mismatch: max_abs_diff={diff:.3e}")


def test_rank1_roundtrip(device: torch.device, dtype: torch.dtype) -> None:
    x = torch.tensor(
        [[0.1, -0.2, 0.3], [1.2, 0.0, -0.7]],
        device=device,
        dtype=dtype,
    )
    embed = PhysicalTensorICTDEmbedding(
        rank=1,
        lmax_out=2,
        channels_in=1,
        channels_out=1,
        input_repr="cartesian",
        include_trace_chain=False,
    ).to(device=device, dtype=dtype)
    recover = PhysicalTensorICTDRecovery(
        rank=1,
        channels_in=1,
        lmax_in=2,
        include_trace_chain=False,
    ).to(device=device)

    blocks = embed(x, return_blocks=True)
    x_rec = recover(blocks)
    _assert_close("rank1 roundtrip", x_rec, x)
    print("PASS: rank-1 Cartesian <-> ICTD roundtrip")


def test_rank2_roundtrip(device: torch.device, dtype: torch.dtype) -> None:
    x = torch.tensor(
        [
            [[2.0, 0.3, -0.4], [0.3, 1.5, 0.8], [-0.4, 0.8, -0.2]],
            [[-1.0, 0.1, 0.2], [0.1, 0.5, -0.6], [0.2, -0.6, 0.7]],
        ],
        device=device,
        dtype=dtype,
    )
    x = 0.5 * (x + x.transpose(-1, -2))

    embed = PhysicalTensorICTDEmbedding(
        rank=2,
        lmax_out=2,
        channels_in=1,
        channels_out=1,
        input_repr="cartesian",
        include_trace_chain=True,
    ).to(device=device, dtype=dtype)
    recover = PhysicalTensorICTDRecovery(
        rank=2,
        channels_in=1,
        lmax_in=2,
        include_trace_chain=True,
    ).to(device=device)

    blocks = embed(x, return_blocks=True)
    x_rec = recover(blocks)
    _assert_close("rank2 roundtrip", x_rec, x, atol=1e-5, rtol=1e-5)
    print("PASS: rank-2 symmetric Cartesian <-> ICTD roundtrip")


def test_rank2_traceless_from_l2(device: torch.device, dtype: torch.dtype) -> None:
    x = torch.tensor(
        [
            [[2.0, 0.3, -0.4], [0.3, 1.5, 0.8], [-0.4, 0.8, -0.2]],
            [[-1.0, 0.1, 0.2], [0.1, 0.5, -0.6], [0.2, -0.6, 0.7]],
        ],
        device=device,
        dtype=dtype,
    )
    x = 0.5 * (x + x.transpose(-1, -2))
    trace = x.diagonal(dim1=-2, dim2=-1).sum(dim=-1) / 3.0
    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    x_traceless = x - trace[:, None, None] * eye

    embed = PhysicalTensorICTDEmbedding(
        rank=2,
        lmax_out=2,
        channels_in=1,
        channels_out=1,
        input_repr="cartesian",
        include_trace_chain=False,
    ).to(device=device, dtype=dtype)
    recover = PhysicalTensorICTDRecovery(
        rank=2,
        channels_in=1,
        lmax_in=2,
        include_trace_chain=False,
    ).to(device=device)

    blocks = embed(x_traceless, return_blocks=True)
    x_rec = recover({2: blocks[2]})
    _assert_close("rank2 traceless from l=2", x_rec, x_traceless, atol=1e-5, rtol=1e-5)
    print("PASS: rank-2 traceless symmetric recovery from l=2")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"Testing physical tensor ICTD roundtrip on device={device}, dtype={dtype}")
    test_rank1_roundtrip(device, dtype)
    test_rank2_roundtrip(device, dtype)
    test_rank2_traceless_from_l2(device, dtype)
    print("All physical tensor ICTD roundtrip tests passed.")


if __name__ == "__main__":
    main()
