"""
Test that HarmonicElementwiseProduct("0e") produces identical results to
the original _irreps_elementwise_tensor_product_0e / _elementwise_tensor_product_0e_blocks.
"""
import math
import torch

from molecular_force_field.models.ictd_irreps import HarmonicElementwiseProduct


def _split_irreps(x: torch.Tensor, channels: int, lmax: int) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    idx = 0
    for l in range(lmax + 1):
        d = channels * (2 * l + 1)
        blk = x[..., idx: idx + d]
        idx += d
        out[l] = blk.view(*x.shape[:-1], channels, 2 * l + 1)
    return out


def _irreps_elementwise_tensor_product_0e_ref(x1, x2, channels, lmax):
    """Original reference implementation."""
    b1 = _split_irreps(x1, channels, lmax)
    b2 = _split_irreps(x2, channels, lmax)
    outs = []
    for l in range(lmax + 1):
        outs.append((b1[l] * b2[l]).sum(dim=-1) / ((2 * l + 1) ** 0.5))
    return torch.cat(outs, dim=-1)


def test_matches_reference():
    lmax = 2
    channels = 6
    batch = 8
    dim = channels * (lmax + 1) ** 2

    torch.manual_seed(42)
    x1 = torch.randn(batch, dim)
    x2 = torch.randn(batch, dim)

    ref = _irreps_elementwise_tensor_product_0e_ref(x1, x2, channels, lmax)

    product_5 = HarmonicElementwiseProduct(lmax=lmax, mul=channels, irreps_out="0e")
    b1 = _split_irreps(x1, channels, lmax)
    b2 = _split_irreps(x2, channels, lmax)
    out = product_5(b1, b2)

    assert ref.shape == out.shape, f"Shape mismatch: {ref.shape} vs {out.shape}"
    torch.testing.assert_close(ref, out, atol=1e-12, rtol=1e-8)
    print(f"test_matches_reference: ok  shape={out.shape}, max_diff={torch.abs(ref - out).max().item():.2e}")


def test_self_product_matches():
    """Verify self-product (f, f) path — the most common usage in PureCartesianICTDTransformerLayer."""
    lmax = 2
    channels = 8
    batch = 4
    dim = channels * (lmax + 1) ** 2

    torch.manual_seed(0)
    f = torch.randn(batch, dim)

    ref = _irreps_elementwise_tensor_product_0e_ref(f, f, channels, lmax)

    product_5 = HarmonicElementwiseProduct(lmax=lmax, mul=channels, irreps_out="0e")
    b = _split_irreps(f, channels, lmax)
    out = product_5(b, b)

    torch.testing.assert_close(ref, out, atol=1e-12, rtol=1e-8)
    print(f"test_self_product_matches: ok  max_diff={torch.abs(ref - out).max().item():.2e}")


def test_non_uniform_mul():
    """
    When adapters change mul per l, the 0e path still works correctly
    because it doesn't use self.mul internally.
    """
    lmax = 2
    batch = 4
    product_5 = HarmonicElementwiseProduct(lmax=lmax, mul=999, irreps_out="0e")

    muls = {0: 3, 1: 5, 2: 7}
    b1, b2 = {}, {}
    ref_parts = []
    for l in range(lmax + 1):
        m = muls[l]
        a = torch.randn(batch, m, 2 * l + 1)
        b = torch.randn(batch, m, 2 * l + 1)
        b1[l] = a
        b2[l] = b
        ref_parts.append((a * b).sum(dim=-1) / math.sqrt(2 * l + 1))

    ref = torch.cat(ref_parts, dim=-1)
    out = product_5(b1, b2)
    assert ref.shape == out.shape
    torch.testing.assert_close(ref, out, atol=1e-12, rtol=1e-8)
    print(f"test_non_uniform_mul: ok  shape={out.shape}")


def test_combined_features():
    """
    Verify that computing per-feature then concatenating == computing on combined features.
    This validates the pattern used in PureCartesianICTDTransformerLayer.
    """
    lmax = 2
    channels = 4
    num_interaction = 3
    batch = 8
    dim = channels * (lmax + 1) ** 2

    torch.manual_seed(123)
    features = [torch.randn(batch, dim) for _ in range(num_interaction)]
    product_5 = HarmonicElementwiseProduct(lmax=lmax, mul=channels, irreps_out="0e")

    invs = []
    for f in features:
        b = _split_irreps(f, channels, lmax)
        invs.append(product_5(b, b))
    out_per_feature = torch.cat(invs, dim=-1)

    refs = []
    for f in features:
        refs.append(_irreps_elementwise_tensor_product_0e_ref(f, f, channels, lmax))
    out_ref = torch.cat(refs, dim=-1)

    assert out_per_feature.shape == out_ref.shape
    torch.testing.assert_close(out_per_feature, out_ref, atol=1e-12, rtol=1e-8)
    print(f"test_combined_features: ok  shape={out_per_feature.shape}")


if __name__ == "__main__":
    test_matches_reference()
    test_self_product_matches()
    test_non_uniform_mul()
    test_combined_features()
    print("\nAll product_5 replacement tests passed.")
