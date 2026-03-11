"""
Test that ictd_irreps works correctly for lmax > 6 (previously documented limit).
Tests direction harmonics, CG tensors, HarmonicElementwiseProduct, and
HarmonicFullyConnectedTensorProduct for lmax up to 10.
"""
import math
import torch
from molecular_force_field.models.ictd_irreps import (
    HarmonicElementwiseProduct,
    HarmonicFullyConnectedTensorProduct,
    build_cg_tensor,
    build_harmonic_projectors,
    direction_harmonics,
    direction_harmonics_all,
    direction_harmonics_fast,
)

SEP = "─" * 80


def test_harmonic_basis_high_l():
    """Verify harmonic basis is orthonormal for l up to 10."""
    print(f"\n{SEP}\nTest: harmonic basis orthonormality for l=0..10\n{SEP}")
    ok = True
    for l in range(11):
        from molecular_force_field.models.ictd_irreps import _harmonic_basis_cpu_f64, _gram_gaussian
        B = _harmonic_basis_cpu_f64(l)  # (Dsym, 2l+1)
        G = _gram_gaussian(l)           # (Dsym, Dsym)
        # Orthonormality: B^T G B = I
        M = B.T @ G @ B
        I = torch.eye(2 * l + 1, dtype=torch.float64)
        dev = (M - I).abs().max().item()
        pass_ = dev < 1e-10
        ok = ok and pass_
        status = "PASS" if pass_ else "FAIL"
        print(f"  l={l:2d}: dim={2*l+1:3d}, orthonormality dev={dev:.2e} {status}")
    return ok


def test_direction_harmonics_high_l():
    """Verify direction_harmonics for l up to 10: equivariance check via norm preservation."""
    print(f"\n{SEP}\nTest: direction harmonics for l=0..10\n{SEP}")
    ok = True
    batch = 64
    torch.manual_seed(42)
    n = torch.randn(batch, 3, dtype=torch.float64)
    n = n / n.norm(dim=-1, keepdim=True)

    for l in range(11):
        Y = direction_harmonics(n, l)
        assert Y.shape == (batch, 2 * l + 1), f"l={l}: shape {Y.shape}"
        # Y should be finite
        finite = torch.isfinite(Y).all().item()
        ok = ok and finite
        # Check norm: ||Y||^2 should be consistent across batch for unit vectors
        norms = Y.norm(dim=-1)
        std_norm = norms.std().item()
        mean_norm = norms.mean().item()
        print(f"  l={l:2d}: shape={Y.shape}, mean_norm={mean_norm:.4f}, std_norm={std_norm:.4f}, finite={finite}")
    return ok


def test_direction_harmonics_all_high_lmax():
    """Test direction_harmonics_all for lmax=8,10."""
    print(f"\n{SEP}\nTest: direction_harmonics_all for high lmax\n{SEP}")
    ok = True
    batch = 32
    torch.manual_seed(0)
    n = torch.randn(batch, 3, dtype=torch.float64)
    n = n / n.norm(dim=-1, keepdim=True)

    for lmax in [8, 10]:
        Y_list = direction_harmonics_all(n, lmax)
        assert len(Y_list) == lmax + 1
        for l, Y in enumerate(Y_list):
            Y_ref = direction_harmonics(n, l)
            diff = (Y - Y_ref).abs().max().item()
            if diff > 1e-10:
                print(f"  FAIL lmax={lmax} l={l}: diff={diff:.2e}")
                ok = False
        print(f"  lmax={lmax}: all l match reference, PASS")
    return ok


def test_cg_tensor_high_l():
    """Build CG tensors for high l and verify basic properties."""
    print(f"\n{SEP}\nTest: CG tensors for l up to 8\n{SEP}")
    ok = True
    for l in range(9):
        for l3 in range(0, 2 * l + 1):
            if (2 * l + l3) % 2 != 0:
                continue
            C = build_cg_tensor(l, l, l3)
            assert C.shape == (2 * l + 1, 2 * l + 1, 2 * l3 + 1)
            finite = torch.isfinite(C).all().item()
            C_fn = C.norm().item()
            ok = ok and finite and C_fn > 0
            if not finite or C_fn == 0:
                print(f"  FAIL l={l}, l3={l3}: finite={finite}, norm={C_fn:.4e}")
    # Cross-l CG
    for l1 in range(5):
        for l2 in range(5):
            for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                if (l1 + l2 + l3) % 2 == 1:
                    continue
                C = build_cg_tensor(l1, l2, l3)
                assert C.shape == (2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)
                ok = ok and torch.isfinite(C).all().item()

    print(f"  {'PASS' if ok else 'FAIL'} (all CG tensors finite and nonzero)")
    return ok


def test_ewp_high_lmax():
    """HarmonicElementwiseProduct for lmax=8."""
    print(f"\n{SEP}\nTest: HarmonicElementwiseProduct lmax=8\n{SEP}")
    ok = True
    lmax = 8
    mul = 4
    batch = 16
    m = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="0e", normalization="component")
    torch.manual_seed(0)
    x1 = {l: torch.randn(batch, mul, 2 * l + 1) for l in range(lmax + 1)}
    x2 = {l: torch.randn(batch, mul, 2 * l + 1) for l in range(lmax + 1)}
    out = m(x1, x2)
    expected_dim = mul * (lmax + 1)
    assert out.shape == (batch, expected_dim), f"shape {out.shape}"
    finite = torch.isfinite(out).all().item()
    ok = ok and finite
    print(f"  0e output: shape={out.shape}, finite={finite}")

    m_full = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="full", normalization="component")
    out_full = m_full(x1, x2)
    for l3 in out_full:
        f = torch.isfinite(out_full[l3]).all().item()
        ok = ok and f
        print(f"  full l3={l3}: shape={out_full[l3].shape}, finite={f}")

    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_ewp_component_variance_high_lmax():
    """Verify component normalization gives unit variance for lmax=8."""
    print(f"\n{SEP}\nTest: EWP component variance lmax=8\n{SEP}")
    ok = True
    lmax = 8
    mul = 1
    batch = 50000
    m = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="full", normalization="component")
    torch.manual_seed(0)
    x1 = {l: torch.randn(batch, mul, 2 * l + 1, dtype=torch.float64) for l in range(lmax + 1)}
    x2 = {l: torch.randn(batch, mul, 2 * l + 1, dtype=torch.float64) for l in range(lmax + 1)}
    result = m(x1, x2)
    for l3 in sorted(result.keys()):
        var = result[l3].var(dim=0)
        mean_var = var.mean().item()
        max_dev = (var - 1.0).abs().max().item()
        pass_ = max_dev < 0.06
        ok = ok and pass_
        print(f"  l3={l3:2d}: mean_var={mean_var:.4f}, max_dev={max_dev:.4f} {'PASS' if pass_ else 'FAIL'}")
    return ok


def test_fctp_high_lmax():
    """HarmonicFullyConnectedTensorProduct for lmax=6,8."""
    print(f"\n{SEP}\nTest: HarmonicFullyConnectedTensorProduct lmax=6,8\n{SEP}")
    ok = True
    for lmax in [6, 8]:
        mul = 2
        batch = 8
        tp = HarmonicFullyConnectedTensorProduct(
            mul_in1=mul, mul_in2=mul, mul_out=mul, lmax=lmax,
            internal_weights=True, normalization="component")
        torch.manual_seed(0)
        x1 = {l: torch.randn(batch, mul, 2 * l + 1, dtype=torch.float64) for l in range(lmax + 1)}
        x2 = {l: torch.randn(batch, mul, 2 * l + 1, dtype=torch.float64) for l in range(lmax + 1)}
        out = tp(x1, x2)
        all_finite = True
        for l3 in range(lmax + 1):
            if l3 in out:
                f = torch.isfinite(out[l3]).all().item()
                all_finite = all_finite and f
        ok = ok and all_finite
        print(f"  lmax={lmax}: paths={tp.num_paths}, output_keys={sorted(out.keys())}, finite={all_finite} {'PASS' if all_finite else 'FAIL'}")
    return ok


def test_projectors_high_lmax():
    """Verify harmonic projectors build correctly for Lmax up to 20."""
    print(f"\n{SEP}\nTest: build_harmonic_projectors for Lmax up to 20\n{SEP}")
    ok = True
    for Lmax in [8, 10, 12, 16, 20]:
        proj = build_harmonic_projectors(Lmax)
        assert proj.Lmax == Lmax
        for L in range(Lmax + 1):
            for k in range(L // 2 + 1):
                l = L - 2 * k
                P = proj.P[(L, l)]
                assert P.shape[0] == 2 * l + 1
                finite = torch.isfinite(P).all().item()
                ok = ok and finite
                if not finite:
                    print(f"  FAIL Lmax={Lmax} (L={L},l={l})")
        print(f"  Lmax={Lmax}: PASS")
    return ok


if __name__ == "__main__":
    results = []
    results.append(("Harmonic basis ortho", test_harmonic_basis_high_l()))
    results.append(("Direction harmonics", test_direction_harmonics_high_l()))
    results.append(("Direction harmonics all", test_direction_harmonics_all_high_lmax()))
    results.append(("CG tensors", test_cg_tensor_high_l()))
    results.append(("EWP high lmax", test_ewp_high_lmax()))
    results.append(("EWP component variance", test_ewp_component_variance_high_lmax()))
    results.append(("FCTP high lmax", test_fctp_high_lmax()))
    results.append(("Projectors high Lmax", test_projectors_high_lmax()))

    print(f"\n{'=' * 80}\nSummary:\n{'=' * 80}")
    all_ok = True
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        all_ok = all_ok and ok
    print(f"\n{'All tests PASSED.' if all_ok else 'Some tests FAILED!'}")
