from __future__ import annotations

import math

import torch

from molecular_force_field.models.ictd_irreps_2d import (
    HarmonicElementwiseProductO2,
    HarmonicElementwiseProductSO2,
    HarmonicFullyConnectedTensorProductO2,
    HarmonicFullyConnectedTensorProductSO2,
    MultipleContractionO2,
    MultipleContractionSO2,
    build_cg_tensor_so2,
    merge_flat_irreps_o2,
    merge_flat_irreps_so2,
    parse_o2_active_irreps,
    so2_irrep_dim,
)


def _make_so2_inputs(batch: int, mul: int, mmax: int, dtype: torch.dtype):
    g = torch.Generator().manual_seed(1234)
    return {
        m: torch.randn(batch, mul, so2_irrep_dim(m), generator=g, dtype=dtype)
        for m in range(mmax + 1)
    }


def _make_o2_inputs(batch: int, mul: int, active: list[str], dtype: torch.dtype):
    g = torch.Generator().manual_seed(4321)
    keys = parse_o2_active_irreps(active)
    return {
        key: torch.randn(batch, mul, 1 if key[0] == "scalar" else 2, generator=g, dtype=dtype)
        for key in keys
    }


def test_so2_cg_component_vs_norm_relation() -> None:
    tp_comp = HarmonicFullyConnectedTensorProductSO2(
        mul_in1=2,
        mul_in2=2,
        mul_out=2,
        mmax=3,
        normalization="component",
        internal_compute_dtype=torch.float64,
    )
    tp_norm = HarmonicFullyConnectedTensorProductSO2(
        mul_in1=2,
        mul_in2=2,
        mul_out=2,
        mmax=3,
        normalization="norm",
        internal_compute_dtype=torch.float64,
    )
    cg_comp = tp_comp._get_cg_list(torch.device("cpu"), torch.float64)
    cg_norm = tp_norm._get_cg_list(torch.device("cpu"), torch.float64)
    for idx, (_, _, m3) in enumerate(tp_comp.paths):
        ratio = math.sqrt(so2_irrep_dim(m3))
        torch.testing.assert_close(cg_comp[idx], ratio * cg_norm[idx], rtol=1e-12, atol=1e-12)


def test_so2_fctp_none_matches_manual_raw_cg() -> None:
    batch = 7
    mul_in1 = 3
    mul_in2 = 2
    mul_out = 4
    mmax = 2
    dtype = torch.float64

    tp = HarmonicFullyConnectedTensorProductSO2(
        mul_in1=mul_in1,
        mul_in2=mul_in2,
        mul_out=mul_out,
        mmax=mmax,
        internal_weights=True,
        normalization="none",
        internal_compute_dtype=torch.float64,
    )
    x1 = _make_so2_inputs(batch, mul_in1, mmax, dtype)
    x2 = _make_so2_inputs(batch, mul_in2, mmax, dtype)
    out = tp(x1, x2)

    manual = {m: torch.zeros_like(out[m]) for m in out}
    w = tp.weight.data.to(dtype=dtype)
    for p_idx, (m1, m2, m3) in enumerate(tp.paths):
        C = build_cg_tensor_so2(m1, m2, m3).to(dtype=dtype)
        seg = torch.einsum("...im,...jn,mnk,oij->...ok", x1[m1], x2[m2], C, w[p_idx])
        manual[m3] = manual[m3] + seg

    for m in manual:
        torch.testing.assert_close(out[m], manual[m], rtol=1e-10, atol=1e-10)


def test_so2_ewp_none_matches_manual_raw_cg_full() -> None:
    batch = 6
    mul = 3
    mmax = 3
    dtype = torch.float64

    ewp = HarmonicElementwiseProductSO2(
        mmax=mmax,
        mul=mul,
        irreps_out="full",
        normalization="none",
        internal_compute_dtype=torch.float64,
    )
    x1 = _make_so2_inputs(batch, mul, mmax, dtype)
    x2 = _make_so2_inputs(batch, mul, mmax, dtype)
    out = ewp(x1, x2)

    manual_parts: dict[int, list[torch.Tensor]] = {}
    for m, m3 in ewp.paths:
        C = build_cg_tensor_so2(m, m, m3).to(dtype=dtype)
        seg = torch.einsum("...cm,...cn,mnk->...ck", x1[m], x2[m], C)
        manual_parts.setdefault(m3, []).append(seg)

    manual = {m3: torch.cat(parts, dim=-2) for m3, parts in manual_parts.items()}
    for m3 in manual:
        torch.testing.assert_close(out[m3], manual[m3], rtol=1e-10, atol=1e-10)


def test_o2_cg_component_vs_norm_relation() -> None:
    active = ["0e", "0o", "1", "2"]
    tp_comp = HarmonicFullyConnectedTensorProductO2(
        mul_in1=2,
        mul_in2=2,
        mul_out=2,
        active_irreps=active,
        normalization="component",
        internal_compute_dtype=torch.float64,
    )
    tp_norm = HarmonicFullyConnectedTensorProductO2(
        mul_in1=2,
        mul_in2=2,
        mul_out=2,
        active_irreps=active,
        normalization="norm",
        internal_compute_dtype=torch.float64,
    )
    cg_comp = tp_comp._get_cg_list(torch.device("cpu"), torch.float64)
    cg_norm = tp_norm._get_cg_list(torch.device("cpu"), torch.float64)
    for idx, (_, _, key3) in enumerate(tp_comp.paths):
        ratio = math.sqrt(1 if key3[0] == "scalar" else 2)
        torch.testing.assert_close(cg_comp[idx], ratio * cg_norm[idx], rtol=1e-12, atol=1e-12)


def test_o2_fctp_none_matches_manual_raw_cg() -> None:
    active = ["0e", "0o", "1", "2"]
    batch = 5
    mul_in1 = 2
    mul_in2 = 2
    mul_out = 3
    dtype = torch.float64

    tp = HarmonicFullyConnectedTensorProductO2(
        mul_in1=mul_in1,
        mul_in2=mul_in2,
        mul_out=mul_out,
        active_irreps=active,
        internal_weights=True,
        normalization="none",
        internal_compute_dtype=torch.float64,
    )
    x1 = _make_o2_inputs(batch, mul_in1, active, dtype)
    x2 = _make_o2_inputs(batch, mul_in2, active, dtype)
    out = tp(x1, x2)

    manual = {key: torch.zeros_like(out[key]) for key in out}
    w = tp.weight.data.to(dtype=dtype)
    cg_list = tp._get_cg_list(torch.device("cpu"), torch.float64)
    for p_idx, (key1, key2, key3) in enumerate(tp.paths):
        seg = torch.einsum("...im,...jn,mnk,oij->...ok", x1[key1], x2[key2], cg_list[p_idx], w[p_idx])
        manual[key3] = manual[key3] + seg

    for key in manual:
        torch.testing.assert_close(out[key], manual[key], rtol=1e-10, atol=1e-10)


def test_o2_ewp_none_matches_manual_raw_cg_full() -> None:
    active = ["0e", "0o", "1", "2"]
    batch = 4
    mul = 2
    dtype = torch.float64

    ewp = HarmonicElementwiseProductO2(
        active_irreps=active,
        mul=mul,
        irreps_out="full",
        normalization="none",
        internal_compute_dtype=torch.float64,
    )
    x1 = _make_o2_inputs(batch, mul, active, dtype)
    x2 = _make_o2_inputs(batch, mul, active, dtype)
    out = ewp(x1, x2)

    manual_parts: dict[tuple[str, int], list[torch.Tensor]] = {}
    cg_list = ewp._get_cg_list(torch.device("cpu"), torch.float64)
    for p_idx, (key_in, key_out) in enumerate(ewp.paths):
        seg = torch.einsum("...cm,...cn,mnk->...ck", x1[key_in], x2[key_in], cg_list[p_idx])
        manual_parts.setdefault(key_out, []).append(seg)

    manual = {key: torch.cat(parts, dim=-2) for key, parts in manual_parts.items()}
    for key in manual:
        torch.testing.assert_close(out[key], manual[key], rtol=1e-10, atol=1e-10)


def test_multiple_contraction_so2_shape_and_grad() -> None:
    batch = 5
    in_channels = 3
    hidden_channels = 2
    mmax = 3
    dtype = torch.float64

    x_blocks = _make_so2_inputs(batch, in_channels, mmax, dtype)
    x = merge_flat_irreps_so2(x_blocks, in_channels, mmax).requires_grad_(True)
    mod = MultipleContractionSO2(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        mmax=mmax,
        correlation=3,
        internal_compute_dtype=torch.float64,
    )
    out = mod(x)
    expected_dim = hidden_channels * (1 + 2 * mmax)
    assert out.shape == (batch, expected_dim)
    loss = out.square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(out).all()
    assert torch.isfinite(x.grad).all()


def test_multiple_contraction_o2_shape_and_grad() -> None:
    batch = 4
    in_channels = 3
    hidden_channels = 2
    active = ["0e", "0o", "1", "2"]
    dtype = torch.float64

    x_blocks = _make_o2_inputs(batch, in_channels, active, dtype)
    x = merge_flat_irreps_o2(x_blocks, in_channels, active).requires_grad_(True)
    mod = MultipleContractionO2(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        active_irreps=active,
        correlation=3,
        internal_compute_dtype=torch.float64,
    )
    out = mod(x)
    expected_dim = hidden_channels * (1 + 1 + 2 + 2)
    assert out.shape == (batch, expected_dim)
    loss = out.square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(out).all()
    assert torch.isfinite(x.grad).all()
