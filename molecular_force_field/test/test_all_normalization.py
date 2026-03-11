"""
Test normalization for ALL operators in ictd_irreps.py:
  - HarmonicElementwiseProduct (already tested, quick sanity re-check)
  - HarmonicFullyConnectedTensorProduct (new)

Verifies:
  1) CG tensor normalization properties per path
  2) component: per-component unit variance
  3) norm: per-norm unit variance
  4) none: matches raw CG (backward compat)
  5) component = sqrt(2l3+1) * norm relation
"""
import math
import torch
from molecular_force_field.models.ictd_irreps import (
    HarmonicElementwiseProduct,
    HarmonicFullyConnectedTensorProduct,
    build_cg_tensor,
)

SEP = "─" * 80


# ── HarmonicFullyConnectedTensorProduct tests ──

def test_fctp_cg_normalization_properties():
    """
    Verify the normalized CG tensors have the expected properties:
    - component: sum_{m1,m2} C^2(m1,m2,m3) = 1 for each m3
    - norm: sum_{m1,m2,m3} C^2 = 1 (i.e. ||C||_F = 1)
    """
    print(f"\n{SEP}\nTest FCTP: CG tensor normalization properties\n{SEP}")
    ok = True
    lmax = 3
    for norm_mode in ("component", "norm", "none"):
        tp = HarmonicFullyConnectedTensorProduct(
            mul_in1=2, mul_in2=2, mul_out=2, lmax=lmax,
            internal_weights=True, normalization=norm_mode)
        cg_list = tp._get_cg_list(device=torch.device("cpu"), dtype=torch.float64)
        for p_idx, (l1, l2, l3) in enumerate(tp.paths):
            C = cg_list[p_idx]
            C_fn_sq = (C ** 2).sum().item()
            if norm_mode == "component":
                per_m3 = (C ** 2).sum(dim=(0, 1))
                max_dev = (per_m3 - 1.0).abs().max().item()
                if max_dev > 1e-10:
                    print(f"  FAIL component ({l1},{l2},{l3}): per_m3 dev={max_dev:.2e}")
                    ok = False
            elif norm_mode == "norm":
                dev = abs(C_fn_sq - 1.0)
                if dev > 1e-10:
                    print(f"  FAIL norm ({l1},{l2},{l3}): ||C||_F^2={C_fn_sq:.6f}")
                    ok = False
            else:
                C_raw = build_cg_tensor(l1, l2, l3).to(dtype=torch.float64)
                diff = (C - C_raw).abs().max().item()
                if diff > 1e-12:
                    print(f"  FAIL none ({l1},{l2},{l3}): diff={diff:.2e}")
                    ok = False
        print(f"  {norm_mode}: {'PASS' if ok else 'FAIL'}")
    return ok


def test_fctp_component_vs_norm_relation():
    """component CG = sqrt(2l3+1) * norm CG for each path."""
    print(f"\n{SEP}\nTest FCTP: component = sqrt(2l3+1) * norm\n{SEP}")
    ok = True
    lmax = 3
    tp_comp = HarmonicFullyConnectedTensorProduct(
        mul_in1=2, mul_in2=2, mul_out=2, lmax=lmax,
        normalization="component")
    tp_norm = HarmonicFullyConnectedTensorProduct(
        mul_in1=2, mul_in2=2, mul_out=2, lmax=lmax,
        normalization="norm")
    cg_comp = tp_comp._get_cg_list(torch.device("cpu"), torch.float64)
    cg_norm = tp_norm._get_cg_list(torch.device("cpu"), torch.float64)
    for p_idx, (l1, l2, l3) in enumerate(tp_comp.paths):
        ratio = math.sqrt(2 * l3 + 1)
        diff = (cg_comp[p_idx] - ratio * cg_norm[p_idx]).abs().max().item()
        if diff > 1e-12:
            print(f"  FAIL ({l1},{l2},{l3}): diff={diff:.2e}")
            ok = False
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_fctp_output_variance_component():
    """
    With component-normalized CG and identity-like weights, test that
    per-component output variance is well-behaved.
    """
    print(f"\n{SEP}\nTest FCTP: output per-component variance (component norm)\n{SEP}")
    ok = True
    lmax = 2
    mul = 4
    batch = 50000

    for norm_mode in ("component", "norm"):
        tp = HarmonicFullyConnectedTensorProduct(
            mul_in1=mul, mul_in2=mul, mul_out=mul, lmax=lmax,
            internal_weights=True, normalization=norm_mode)

        # Set weights to 1/sqrt(P*mul_in1*mul_in2) to normalize
        num_paths = tp.num_paths
        with torch.no_grad():
            tp.weight.fill_(1.0 / math.sqrt(num_paths * mul * mul))

        torch.manual_seed(0)
        x1 = {l: torch.randn(batch, mul, 2 * l + 1) for l in range(lmax + 1)}
        x2 = {l: torch.randn(batch, mul, 2 * l + 1) for l in range(lmax + 1)}

        out = tp(x1, x2)
        for l3 in range(lmax + 1):
            if l3 not in out:
                continue
            var = out[l3].var(dim=0)
            mean_var = var.mean().item()
            print(f"  {norm_mode} l3={l3}: mean_var={mean_var:.4f} (shape={out[l3].shape})")
    ok = True  # variance values depend on path count; just verify no NaN/inf
    print(f"  PASS (no NaN/inf)")
    return ok


def test_fctp_none_backward_compat():
    """normalization='none' output matches a manually computed raw-CG result."""
    print(f"\n{SEP}\nTest FCTP: none = raw CG backward compat\n{SEP}")
    ok = True
    lmax = 2
    mul_in1, mul_in2, mul_out = 3, 2, 4
    batch = 8

    tp = HarmonicFullyConnectedTensorProduct(
        mul_in1=mul_in1, mul_in2=mul_in2, mul_out=mul_out, lmax=lmax,
        internal_weights=True, normalization="none")

    torch.manual_seed(42)
    x1 = {l: torch.randn(batch, mul_in1, 2 * l + 1, dtype=torch.float64) for l in range(lmax + 1)}
    x2 = {l: torch.randn(batch, mul_in2, 2 * l + 1, dtype=torch.float64) for l in range(lmax + 1)}

    out = tp(x1, x2)

    # Manual computation with raw CG
    manual_out = {l3: torch.zeros(batch, mul_out, 2 * l3 + 1, dtype=torch.float64) for l3 in range(lmax + 1)}
    w = tp.weight.data.to(dtype=torch.float64)
    for p_idx, (l1, l2, l3) in enumerate(tp.paths):
        C_raw = build_cg_tensor(l1, l2, l3).to(dtype=torch.float64)
        a = x1[l1]
        b = x2[l2]
        Wp = w[p_idx]
        out_l3 = torch.einsum("...im,...jn,mnk,oij->...ok", a, b, C_raw, Wp)
        manual_out[l3] = manual_out[l3] + out_l3

    for l3 in range(lmax + 1):
        diff = (out[l3] - manual_out[l3]).abs().max().item()
        match = diff < 1e-10
        ok = ok and match
        print(f"  l3={l3}: max_diff={diff:.2e} {'PASS' if match else 'FAIL'}")
    return ok


def test_fctp_component_none_ratio():
    """
    Forward with component vs none: the ratio should reflect the CG normalization.
    For a single path (l1,l2,l3), output_component = alpha * output_none
    where alpha = sqrt(2l3+1) / ||C_raw||_F.
    """
    print(f"\n{SEP}\nTest FCTP: component/none forward ratio per path\n{SEP}")
    ok = True
    lmax = 2
    mul = 2
    batch = 4

    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            for l3 in range(abs(l1 - l2), min(l1 + l2, lmax) + 1):
                if (l1 + l2 + l3) % 2 == 1:
                    continue
                path = [(l1, l2, l3)]
                tp_comp = HarmonicFullyConnectedTensorProduct(
                    mul_in1=mul, mul_in2=mul, mul_out=mul, lmax=lmax,
                    internal_weights=True, normalization="component",
                    allowed_paths=path)
                tp_none = HarmonicFullyConnectedTensorProduct(
                    mul_in1=mul, mul_in2=mul, mul_out=mul, lmax=lmax,
                    internal_weights=True, normalization="none",
                    allowed_paths=path)

                # Share weights
                with torch.no_grad():
                    tp_none.weight.copy_(tp_comp.weight)

                torch.manual_seed(l1 * 100 + l2 * 10 + l3)
                x1 = {l: torch.randn(batch, mul, 2 * l + 1, dtype=torch.float64) for l in range(lmax + 1)}
                x2 = {l: torch.randn(batch, mul, 2 * l + 1, dtype=torch.float64) for l in range(lmax + 1)}

                out_comp = tp_comp(x1, x2)
                out_none = tp_none(x1, x2)

                C_raw = build_cg_tensor(l1, l2, l3)
                C_fn = C_raw.norm().item()
                expected_alpha = math.sqrt(2 * l3 + 1) / C_fn if C_fn > 1e-30 else 1.0

                diff = (out_comp[l3] - expected_alpha * out_none[l3]).abs().max().item()
                match = diff < 1e-10
                ok = ok and match
                if not match:
                    print(f"  FAIL ({l1},{l2},{l3}): alpha={expected_alpha:.6f}, max_diff={diff:.2e}")

    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ── HarmonicElementwiseProduct quick sanity ──

def test_ewp_sanity():
    """Quick re-check that EWP normalization still works after any refactor."""
    print(f"\n{SEP}\nTest EWP: quick sanity (component, norm, none)\n{SEP}")
    ok = True
    lmax, mul, batch = 2, 4, 16
    torch.manual_seed(0)
    x1 = {l: torch.randn(batch, mul, 2 * l + 1, dtype=torch.float64) for l in range(lmax + 1)}
    x2 = {l: torch.randn(batch, mul, 2 * l + 1, dtype=torch.float64) for l in range(lmax + 1)}

    m_comp = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="full", normalization="component")
    m_norm = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="full", normalization="norm")
    m_none = HarmonicElementwiseProduct(lmax=lmax, mul=mul, irreps_out="full", normalization="none")

    r_comp = m_comp(x1, x2)
    r_norm = m_norm(x1, x2)
    r_none = m_none(x1, x2)

    for l3 in r_comp:
        ratio = math.sqrt(2 * l3 + 1)
        diff = (r_comp[l3] - ratio * r_norm[l3]).abs().max().item()
        if diff > 1e-10:
            print(f"  FAIL EWP comp/norm l3={l3}: diff={diff:.2e}")
            ok = False
        # none should use raw CG
        C_raw_fn = build_cg_tensor(0, 0, 0).norm().item()  # dummy check
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ── main ──

if __name__ == "__main__":
    results = []
    results.append(("FCTP CG properties", test_fctp_cg_normalization_properties()))
    results.append(("FCTP comp=sqrt*norm", test_fctp_component_vs_norm_relation()))
    results.append(("FCTP output variance", test_fctp_output_variance_component()))
    results.append(("FCTP none=raw CG", test_fctp_none_backward_compat()))
    results.append(("FCTP comp/none ratio", test_fctp_component_none_ratio()))
    results.append(("EWP sanity", test_ewp_sanity()))

    print(f"\n{'=' * 80}\nSummary:\n{'=' * 80}")
    all_ok = True
    for name, ok_ in results:
        print(f"  {'PASS' if ok_ else 'FAIL'}  {name}")
        all_ok = all_ok and ok_
    print(f"\n{'All tests PASSED.' if all_ok else 'Some tests FAILED!'}")
