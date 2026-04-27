from __future__ import annotations

import torch

from molecular_force_field.models.ictd_irreps_2d import (
    HarmonicFullyConnectedTensorProductO2,
    HarmonicFullyConnectedTensorProductSO2,
    build_associator_tensor_so2,
    build_cg_tensor_so2,
    build_harmonic_projectors_2d,
    build_harmonic_reconstructors_2d,
    build_recoupling_tensor_so2,
    merge_flat_irreps_o2,
    merge_flat_irreps_so2,
    parse_o2_active_irreps,
    split_flat_irreps_o2,
    split_flat_irreps_so2,
    so2_irrep_dim,
)


def test_split_merge_so2_roundtrip() -> None:
    batch = 3
    channels = 4
    mmax = 3
    total_dim = channels * sum(so2_irrep_dim(m) for m in range(mmax + 1))
    x = torch.randn(batch, total_dim, dtype=torch.float64)
    blocks = split_flat_irreps_so2(x, channels, mmax)
    y = merge_flat_irreps_so2(blocks, channels, mmax)
    torch.testing.assert_close(y, x)


def test_split_merge_o2_roundtrip() -> None:
    batch = 2
    channels = 3
    active = ["0e", "0o", "1", "2"]
    total_dim = channels * (1 + 1 + 2 + 2)
    x = torch.randn(batch, total_dim, dtype=torch.float64)
    blocks = split_flat_irreps_o2(x, channels, active)
    y = merge_flat_irreps_o2(blocks, channels, active)
    torch.testing.assert_close(y, x)


def test_trace_chain_roundtrip_and_dims_up_to_l8() -> None:
    Lmax = 8
    proj = build_harmonic_projectors_2d(Lmax)
    recon = build_harmonic_reconstructors_2d(Lmax)
    g = torch.Generator().manual_seed(1234)

    for L in range(Lmax + 1):
        expected_ms = list(range(L, -1, -2))
        found_ms = sorted(m for LL, m in proj.P if LL == L)
        assert found_ms == sorted(expected_ms)

        total_dim = sum(1 if m == 0 else 2 for m in expected_ms)
        assert total_dim == L + 1

        t = torch.randn(L + 1, generator=g, dtype=torch.float64)
        rebuilt = torch.zeros_like(t)
        for m in expected_ms:
            P = proj.P[(L, m)]
            V = recon.V[(L, m)]
            rebuilt = rebuilt + V @ (P @ t)
        torch.testing.assert_close(rebuilt, t, rtol=1e-10, atol=1e-10)


def test_so2_path_rules_match_sum_and_difference() -> None:
    tp = HarmonicFullyConnectedTensorProductSO2(
        mul_in1=1,
        mul_in2=1,
        mul_out=1,
        mmax=2,
        internal_weights=True,
    )
    expected = {
        (0, 0, 0),
        (0, 1, 1),
        (0, 2, 2),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 2),
        (1, 2, 1),
        (2, 0, 2),
        (2, 1, 1),
        (2, 2, 0),
    }
    assert set(tp.paths) == expected


def _make_o2_blocks(
    active: list[str],
    *,
    batch: int = 2,
    mul: int = 1,
    dtype: torch.dtype = torch.float64,
):
    keys = parse_o2_active_irreps(active)
    blocks = {
        key: torch.zeros(batch, mul, 1 if key[0] == "scalar" else 2, dtype=dtype)
        for key in keys
    }
    return keys, blocks


def test_o2_scalar_and_frequency_rules() -> None:
    active = ["0e", "0o", "1", "2"]
    keys, x1 = _make_o2_blocks(active)
    _, x2 = _make_o2_blocks(active)
    key_0e = ("scalar", 1)
    key_0o = ("scalar", -1)
    key_1 = ("freq", 1)

    tp = HarmonicFullyConnectedTensorProductO2(
        mul_in1=1,
        mul_in2=1,
        mul_out=1,
        active_irreps=active,
        internal_weights=True,
        normalization="none",
    )
    with torch.no_grad():
        tp.weight.fill_(1.0)

    x1[key_0o][..., 0] = 1.0
    x2[key_0o][..., 0] = 1.0
    out = tp(x1, x2)
    assert out[key_0e].abs().sum().item() > 0.0
    assert out[key_0o].abs().sum().item() == 0.0
    assert out[key_1].abs().sum().item() == 0.0

    _, x1 = _make_o2_blocks(active)
    _, x2 = _make_o2_blocks(active)
    x1[key_0o][..., 0] = 1.0
    x2[key_1][..., 0] = 1.0
    out = tp(x1, x2)
    assert out[key_1].abs().sum().item() > 0.0
    assert out[key_0e].abs().sum().item() == 0.0
    assert out[key_0o].abs().sum().item() == 0.0


def test_o2_same_frequency_pair_produces_even_and_odd_scalars() -> None:
    active = ["0e", "0o", "1", "2"]
    key_0e = ("scalar", 1)
    key_0o = ("scalar", -1)
    key_1 = ("freq", 1)
    key_2 = ("freq", 2)
    _, x1 = _make_o2_blocks(active)
    _, x2 = _make_o2_blocks(active)

    tp = HarmonicFullyConnectedTensorProductO2(
        mul_in1=1,
        mul_in2=1,
        mul_out=1,
        active_irreps=active,
        internal_weights=True,
        normalization="none",
    )
    with torch.no_grad():
        tp.weight.fill_(1.0)

    x1[key_1][..., 0] = 1.0
    x2[key_1][..., 0] = 1.0
    x2[key_1][..., 1] = 1.0
    out = tp(x1, x2)

    assert out[key_0e].abs().sum().item() > 0.0
    assert out[key_0o].abs().sum().item() > 0.0
    assert out[key_2].abs().sum().item() > 0.0


def test_so2_recoupling_left_right_match_explicit_sequential_cg() -> None:
    for ma, mb, mc, mout in [(1, 1, 1, 1), (1, 2, 1, 2), (2, 2, 1, 1), (2, 1, 1, 0)]:
        left = build_recoupling_tensor_so2(ma, mb, mc, mout, "left")
        right = build_recoupling_tensor_so2(ma, mb, mc, mout, "right")
        torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-10)

        xa = torch.randn(so2_irrep_dim(ma), dtype=torch.float64)
        xb = torch.randn(so2_irrep_dim(mb), dtype=torch.float64)
        xc = torch.randn(so2_irrep_dim(mc), dtype=torch.float64)
        y_left = torch.einsum("a,b,c,abcd->d", xa, xb, xc, left)
        y_right = torch.einsum("a,b,c,abcd->d", xa, xb, xc, right)
        torch.testing.assert_close(y_left, y_right, rtol=1e-10, atol=1e-10)


def test_so2_associator_maps_left_to_right() -> None:
    for ma, mb, mc, mout in [(1, 1, 1, 1), (2, 1, 1, 0), (1, 2, 1, 2)]:
        left = build_recoupling_tensor_so2(ma, mb, mc, mout, "left").reshape(-1, so2_irrep_dim(mout))
        right = build_recoupling_tensor_so2(ma, mb, mc, mout, "right").reshape(-1, so2_irrep_dim(mout))
        A = build_associator_tensor_so2(ma, mb, mc, mout)
        torch.testing.assert_close(left @ A, right, rtol=1e-10, atol=1e-10)
