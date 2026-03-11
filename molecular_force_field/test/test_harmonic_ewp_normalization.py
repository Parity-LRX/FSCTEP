"""
Test HarmonicElementwiseProduct normalization modes: component, norm, none.
Verifies:
  1) component: each output m3-component has unit variance (from unit-variance i.i.d. inputs)
  2) norm: output squared-norm has expected value 1
  3) 0e path backward compat: component normalization gives identical results to old 1/sqrt(2l+1)
  4) none: matches raw CG (no rescaling)
"""
import math
import torch
from molecular_force_field.models.ictd_irreps import HarmonicElementwiseProduct, build_cg_tensor

SEP = "─" * 80


def make_input(lmax, mul, batch, dtype=torch.float64):
    d = {}
    for l in range(lmax + 1):
        d[l] = torch.randn(batch, mul, 2 * l + 1, dtype=dtype)
    return d


def test_0e_backward_compat():
    """component normalization 0e path must match old 1/sqrt(2l+1) formula."""
    print(f"\n{SEP}\nTest: 0e backward compatibility (component vs old)\n{SEP}")
    ok = True
    for lmax in [2, 3, 4]:
        mul, batch = 8, 256
        torch.manual_seed(42)
        x1 = make_input(lmax, mul, batch)
        x2 = make_input(lmax, mul, batch)

        m_comp = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="0e", normalization="component")
        out_comp = m_comp(x1, x2)

        # Old formula: (a*b).sum(-1) / sqrt(2l+1) per l
        old_parts = []
        for l in range(lmax + 1):
            old_parts.append((x1[l] * x2[l]).sum(dim=-1) / math.sqrt(2 * l + 1))
        out_old = torch.cat(old_parts, dim=-1)

        max_diff = (out_comp - out_old).abs().max().item()
        match = max_diff < 1e-12
        ok = ok and match
        print(f"  lmax={lmax}: max_diff={max_diff:.2e} {'PASS' if match else 'FAIL'}")
    return ok


def test_component_variance():
    """Under component normalization, each output m3-component should have unit variance."""
    print(f"\n{SEP}\nTest: component normalization → per-component unit variance\n{SEP}")
    ok = True
    for lmax in [2, 3]:
        mul = 1
        batch = 100000
        m = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="full", normalization="component")
        torch.manual_seed(0)
        x1 = make_input(lmax, mul, batch)
        x2 = make_input(lmax, mul, batch)
        result = m(x1, x2)
        for l3, tensor in result.items():
            # tensor: (batch, mul_l3, 2l3+1)
            # compute variance over batch for each (mul_channel, m3) component
            var = tensor.var(dim=0)  # (mul_l3, 2l3+1)
            mean_var = var.mean().item()
            max_dev = (var - 1.0).abs().max().item()
            pass_ = max_dev < 0.05  # statistical tolerance for 100k samples
            ok = ok and pass_
            print(f"  lmax={lmax} l3={l3}: mean_var={mean_var:.4f}, max_dev_from_1={max_dev:.4f} {'PASS' if pass_ else 'FAIL'}")
    return ok


def test_norm_variance():
    """Under norm normalization, output squared-norm per (mul_channel) should have expected value 1."""
    print(f"\n{SEP}\nTest: norm normalization → unit expected squared-norm\n{SEP}")
    ok = True
    for lmax in [2, 3]:
        mul = 1
        batch = 100000
        m = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="full", normalization="norm")
        torch.manual_seed(0)
        x1 = make_input(lmax, mul, batch)
        x2 = make_input(lmax, mul, batch)
        result = m(x1, x2)
        for l3, tensor in result.items():
            # tensor: (batch, mul_l3, 2l3+1)
            sq_norm = (tensor ** 2).sum(dim=-1)  # (batch, mul_l3)
            mean_sq = sq_norm.mean(dim=0)  # (mul_l3,)
            mean_val = mean_sq.mean().item()
            max_dev = (mean_sq - 1.0).abs().max().item()
            pass_ = max_dev < 0.05
            ok = ok and pass_
            print(f"  lmax={lmax} l3={l3}: mean_||out||^2={mean_val:.4f}, max_dev_from_1={max_dev:.4f} {'PASS' if pass_ else 'FAIL'}")
    return ok


def test_none_matches_raw():
    """normalization='none' should use raw CG tensors."""
    print(f"\n{SEP}\nTest: normalization='none' matches raw CG einsum\n{SEP}")
    ok = True
    for lmax in [2, 3]:
        mul = 4
        batch = 16
        m = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="full", normalization="none")
        torch.manual_seed(7)
        x1 = make_input(lmax, mul, batch)
        x2 = make_input(lmax, mul, batch)
        result = m(x1, x2)

        # Manually compute with raw CG
        manual: dict = {}
        for l in range(lmax + 1):
            for l3 in range(0, 2 * l + 1):
                if (2 * l + l3) % 2 != 0:
                    continue
                C = build_cg_tensor(l, l, l3).to(dtype=torch.float64)
                a = x1[l]
                b = x2[l]
                out_l3 = torch.einsum("bcm,bcn,mno->bco", a, b, C)
                manual.setdefault(l3, []).append(out_l3)
        manual_result = {l3: torch.cat(v, dim=-2) for l3, v in manual.items()}

        for l3 in result:
            diff = (result[l3] - manual_result[l3]).abs().max().item()
            match = diff < 1e-12
            ok = ok and match
            print(f"  lmax={lmax} l3={l3}: max_diff={diff:.2e} {'PASS' if match else 'FAIL'}")
    return ok


def test_component_vs_norm_relation():
    """component output should be sqrt(2l3+1) times norm output for each l3."""
    print(f"\n{SEP}\nTest: component = sqrt(2l3+1) * norm\n{SEP}")
    ok = True
    lmax, mul, batch = 3, 4, 16
    torch.manual_seed(99)
    x1 = make_input(lmax, mul, batch)
    x2 = make_input(lmax, mul, batch)

    m_comp = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="full", normalization="component")
    m_norm = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="full", normalization="norm")

    r_comp = m_comp(x1, x2)
    r_norm = m_norm(x1, x2)

    for l3 in r_comp:
        ratio = math.sqrt(2 * l3 + 1)
        diff = (r_comp[l3] - ratio * r_norm[l3]).abs().max().item()
        match = diff < 1e-10
        ok = ok and match
        print(f"  l3={l3}: ratio=sqrt({2*l3+1})={ratio:.4f}, max_diff={diff:.2e} {'PASS' if match else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = []
    results.append(("0e backward compat", test_0e_backward_compat()))
    results.append(("component variance", test_component_variance()))
    results.append(("norm variance", test_norm_variance()))
    results.append(("none = raw CG", test_none_matches_raw()))
    results.append(("component/norm relation", test_component_vs_norm_relation()))

    print(f"\n{'=' * 80}\nSummary:\n{'=' * 80}")
    all_ok = True
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        all_ok = all_ok and ok
    print(f"\n{'All tests PASSED.' if all_ok else 'Some tests FAILED!'}")
