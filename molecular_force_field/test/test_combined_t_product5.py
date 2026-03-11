"""
Test: combined T = cat(features, scalars) per l → product_5(T, T) → 0e.

Verifies:
  1) Output dimension matches proj_total input expectation
  2) Scalars in l=0 block get the same normalization factor (=1 for l=0 component)
  3) Equivalence: combined T gives same values as separate invs + inv3 (reordered)
  4) Component normalization factor for l=0 is exactly 1.0
"""
import torch
from molecular_force_field.models.ictd_irreps import HarmonicElementwiseProduct


def _split_irreps(x, channels, lmax):
    out = {}
    idx = 0
    for l in range(lmax + 1):
        d = channels * (2 * l + 1)
        blk = x[..., idx: idx + d]
        idx += d
        out[l] = blk.view(*x.shape[:-1], channels, 2 * l + 1)
    return out


def test_output_dim():
    """Output dim of combined T product_5 matches proj_total input."""
    print("Test 1: output dimension")
    lmax = 2
    channels = 8
    num_interaction = 2
    scalar_channels = 32
    batch = 16
    dim = channels * (lmax + 1) ** 2

    torch.manual_seed(0)
    features = [torch.randn(batch, dim) for _ in range(num_interaction)]
    scalars = torch.randn(batch, scalar_channels)

    product_5 = HarmonicElementwiseProduct(lmax=lmax, mul=channels, irreps_out="0e")

    splits = [_split_irreps(f, channels, lmax) for f in features]
    T_blocks = {}
    for l in range(lmax + 1):
        T_blocks[l] = torch.cat([splits[i][l] for i in range(num_interaction)], dim=-2)
    T_blocks[0] = torch.cat([T_blocks[0], scalars.unsqueeze(-1)], dim=-2)
    f_prod5 = product_5(T_blocks, T_blocks)

    expected_dim = num_interaction * channels * (lmax + 1) + scalar_channels
    assert f_prod5.shape == (batch, expected_dim), f"got {f_prod5.shape}, expected (_, {expected_dim})"
    print(f"  PASS: shape={f_prod5.shape}, expected_dim={expected_dim}")
    return True


def test_scalar_normalization():
    """Scalar part in l=0 block gets component normalization (factor=1 for l=0)."""
    print("\nTest 2: scalar normalization via product_5")
    batch = 1024
    lmax = 2
    channels = 4
    scalar_channels = 8

    product_5 = HarmonicElementwiseProduct(lmax=lmax, mul=channels, irreps_out="0e")

    torch.manual_seed(42)
    T_blocks = {}
    for l in range(lmax + 1):
        T_blocks[l] = torch.zeros(batch, channels, 2 * l + 1)
    scalars = torch.randn(batch, scalar_channels)
    T_blocks[0] = torch.cat([T_blocks[0], scalars.unsqueeze(-1)], dim=-2)

    out = product_5(T_blocks, T_blocks)
    scalar_start = channels
    scalar_part = out[:, scalar_start:scalar_start + scalar_channels]
    ref = scalars * scalars * product_5._0e_factors[0]
    diff = (scalar_part - ref).abs().max().item()
    ok = diff < 1e-12
    print(f"  scalar part max_diff={diff:.2e} {'PASS' if ok else 'FAIL'}")
    return ok


def test_equivalence_with_old():
    """Combined T product_5 gives same values as separate invs + inv3 (just reordered)."""
    print("\nTest 3: equivalence with old (separate) computation")
    lmax = 2
    channels = 8
    num_interaction = 2
    scalar_channels = 32
    batch = 16
    dim = channels * (lmax + 1) ** 2

    torch.manual_seed(7)
    features = [torch.randn(batch, dim) for _ in range(num_interaction)]
    scalars = torch.randn(batch, scalar_channels)

    product_5 = HarmonicElementwiseProduct(lmax=lmax, mul=channels, irreps_out="0e")

    # --- Old approach: per-feature product_5 + separate inv3 ---
    old_invs = []
    for f in features:
        b = _split_irreps(f, channels, lmax)
        old_invs.append(product_5(b, b))
    old_inv3 = scalars * scalars

    # --- New approach: combined T per l ---
    splits = [_split_irreps(f, channels, lmax) for f in features]
    T_blocks = {}
    for l in range(lmax + 1):
        T_blocks[l] = torch.cat([splits[i][l] for i in range(num_interaction)], dim=-2)
    T_blocks[0] = torch.cat([T_blocks[0], scalars.unsqueeze(-1)], dim=-2)
    new_f_prod5 = product_5(T_blocks, T_blocks)

    old_total_dim = sum(inv.shape[-1] for inv in old_invs) + scalar_channels
    assert new_f_prod5.shape[-1] == old_total_dim, \
        f"dim mismatch: new={new_f_prod5.shape[-1]} old_total={old_total_dim}"

    # New output layout (per l, with channels from all features + scalars for l=0):
    #   [feat0_l0(C), feat1_l0(C), scalars(S), feat0_l1(C), feat1_l1(C), feat0_l2(C), feat1_l2(C)]
    # Old output layout:
    #   [feat0(l0,l1,l2), feat1(l0,l1,l2), inv3(S)]
    # We verify element-by-element.

    idx = 0
    all_ok = True
    for l in range(lmax + 1):
        if l == 0:
            mul_l = num_interaction * channels + scalar_channels
        else:
            mul_l = num_interaction * channels
        new_block = new_f_prod5[:, idx:idx + mul_l]
        idx += mul_l

        # Features part: new has [feat0_l, feat1_l, ...]
        feat_parts_new = new_block[:, :num_interaction * channels]
        for i in range(num_interaction):
            new_fi_l = feat_parts_new[:, i * channels:(i + 1) * channels]
            old_fi_l = old_invs[i][:, l * channels:(l + 1) * channels]
            diff = (new_fi_l - old_fi_l).abs().max().item()
            ok = diff < 1e-12
            all_ok = all_ok and ok
            if not ok:
                print(f"  feat[{i}] l={l} diff={diff:.2e} FAIL")

        if l == 0:
            scalar_part = new_block[:, num_interaction * channels:]
            diff_s = (scalar_part - old_inv3).abs().max().item()
            ok_s = diff_s < 1e-12
            all_ok = all_ok and ok_s
            if not ok_s:
                print(f"  scalars l=0 diff={diff_s:.2e} FAIL")

    print(f"  {'PASS' if all_ok else 'FAIL'}: all elements match (reordered)")
    return all_ok


def test_component_factor_l0():
    """Component normalization factor for l=0 is exactly 1.0."""
    print("\nTest 4: l=0 component normalization factor")
    product_5 = HarmonicElementwiseProduct(lmax=3, mul=8, irreps_out="0e")
    factor = product_5._0e_factors[0]
    ok = abs(factor - 1.0) < 1e-10
    print(f"  factor_l0={factor:.8f} {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = []
    results.append(test_output_dim())
    results.append(test_scalar_normalization())
    results.append(test_equivalence_with_old())
    results.append(test_component_factor_l0())
    print(f"\n{'=' * 60}")
    print(f"{'All tests PASSED.' if all(results) else 'Some tests FAILED!'}")
