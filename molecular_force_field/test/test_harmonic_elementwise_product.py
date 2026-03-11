"""
Test HarmonicElementwiseProduct: irreps_out="0e", "full", and filter strings like "0e + 2e".
"""
import torch
from molecular_force_field.models.ictd_irreps import (
    HarmonicElementwiseProduct,
    parse_irreps_to_l3_list,
)


def make_input(lmax: int, mul: int, batch: int = 4, device=None):
    """x1/x2: dict l -> (batch, mul, 2l+1)."""
    d = {}
    for l in range(lmax + 1):
        d[l] = torch.randn(batch, mul, 2 * l + 1, device=device)
    return d


def test_parse_irreps_to_l3_list():
    # No filter: all l in order
    assert parse_irreps_to_l3_list("0e + 1o + 2e") == [0, 1, 2]
    assert parse_irreps_to_l3_list("2e + 0e") == [2, 0]
    # Only even l3 allowed (l⊗l case)
    allowed = [0, 2, 4]
    assert parse_irreps_to_l3_list("0e + 1o + 2e", allowed_l3=allowed) == [0, 2]
    assert parse_irreps_to_l3_list("2e + 0e + 4e", allowed_l3=allowed) == [2, 0, 4]
    print("parse_irreps_to_l3_list: ok")


def test_0e_only():
    lmax, mul = 2, 3
    batch = 4
    m = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="0e")
    x1 = make_input(lmax, mul, batch)
    x2 = make_input(lmax, mul, batch)
    out = m(x1, x2)
    assert isinstance(out, torch.Tensor)
    expected_dim = mul * (lmax + 1)  # 3 * 3 = 9
    assert out.shape == (batch, expected_dim), f"got {out.shape}"
    print("irreps_out='0e': ok")


def test_full():
    lmax, mul = 2, 3
    batch = 4
    m = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="full")
    x1 = make_input(lmax, mul, batch)
    x2 = make_input(lmax, mul, batch)
    out = m(x1, x2)
    assert isinstance(out, dict)
    # lmax=2: l=0->0, l=1->0,2, l=2->0,2,4
    for l3 in [0, 2, 4]:
        assert l3 in out
        # mul_l3 = mul * num of l that contribute to l3
        if l3 == 0:
            n_l = 3  # l=0,1,2
        elif l3 == 2:
            n_l = 2  # l=1,2
        else:
            n_l = 1  # l=2
        assert out[l3].shape == (batch, mul * n_l, 2 * l3 + 1), f"l3={l3} got {out[l3].shape}"
    print("irreps_out='full': ok")


def test_filter_0e_2e():
    lmax, mul = 2, 3
    batch = 4
    m_full = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="full")
    m_filter = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="0e + 2e")
    x1 = make_input(lmax, mul, batch)
    x2 = make_input(lmax, mul, batch)
    full = m_full(x1, x2)
    filtered = m_filter(x1, x2)
    assert isinstance(filtered, torch.Tensor)
    # Same as concat full[0] and full[2] in that order
    dim_expected = full[0].reshape(batch, -1).shape[-1] + full[2].reshape(batch, -1).shape[-1]
    assert filtered.shape == (batch, dim_expected), f"got {filtered.shape} expected last dim {dim_expected}"
    ref = torch.cat([full[0].reshape(batch, -1), full[2].reshape(batch, -1)], dim=-1)
    torch.testing.assert_close(filtered, ref)
    print("irreps_out='0e + 2e': ok (matches full)")


def test_filter_2e_0e_order():
    lmax, mul = 2, 3
    batch = 4
    m_full = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="full")
    m_filter = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="2e + 0e")
    x1 = make_input(lmax, mul, batch)
    x2 = make_input(lmax, mul, batch)
    full = m_full(x1, x2)
    filtered = m_filter(x1, x2)
    ref = torch.cat([full[2].reshape(batch, -1), full[0].reshape(batch, -1)], dim=-1)
    torch.testing.assert_close(filtered, ref)
    print("irreps_out='2e + 0e': ok (order 2e then 0e)")


def test_filter_ignores_odd():
    # "0e + 1o + 2e" with l⊗l: only 0e, 2e exist -> same as "0e + 2e"
    lmax, mul = 2, 3
    batch = 4
    m1 = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="0e + 2e")
    m2 = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="0e + 1o + 2e")
    x1 = make_input(lmax, mul, batch)
    x2 = make_input(lmax, mul, batch)
    out1 = m1(x1, x2)
    out2 = m2(x1, x2)
    assert out1.shape == out2.shape
    torch.testing.assert_close(out1, out2)
    print("irreps_out='0e + 1o + 2e': ok (1o ignored, same as 0e+2e)")


if __name__ == "__main__":
    test_parse_irreps_to_l3_list()
    test_0e_only()
    test_full()
    test_filter_0e_2e()
    test_filter_2e_0e_order()
    test_filter_ignores_odd()
    print("All tests passed.")
