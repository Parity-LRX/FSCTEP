#!/usr/bin/env python3
"""Validate the extracted PureCartesianICTDMaceBaseline reproduces
PureCartesianICTDFix(ictd_fix_route="baseline", ictd_fix_product_backend="native-mace")
bit-for-bit.

Builds both with matching hyper-parameters, transfers the fix model's weights
into the extracted baseline (state_dict copy), and asserts:
  (1) the key sets match exactly (no baseline key missing from fix; no fix-only
      key absent from baseline => the extraction captures the whole path),
  (2) energy + forces match to machine precision (fp64),
  (3) the extracted model is equivariant (energy invariant, forces covariant).

Run: python -m molecular_force_field.test._mace_baseline_equiv
"""
from __future__ import annotations

import torch

from molecular_force_field.test.bench_ictd_fix_trainstep import (
    build_model,
    make_fixed_graph,
    random_rotation,
    SPECIES,
)
from molecular_force_field.models.pure_cartesian_ictd_mace_baseline import (
    PureCartesianICTDMaceBaseline,
)


def _ef(model, graph):
    pos = graph[0].detach().clone().requires_grad_(True)
    g = (pos,) + tuple(graph[1:])
    out = model(*g)
    e = (out[0] if isinstance(out, tuple) else out).sum()
    f = -torch.autograd.grad(e, pos, create_graph=False)[0]
    return e, f


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    channels, lmax, L, corr, atoms = 16, 2, 2, 3, 48

    torch.manual_seed(0)
    fix = build_model(
        channels=channels, lmax=lmax, num_interaction=L, route="baseline",
        product_backend="native-mace", dtype=dtype, device=device, correlation=corr,
    )
    fix.eval()

    base = PureCartesianICTDMaceBaseline(
        max_embed_radius=5.0, main_number_of_basis=8, hidden_dim_conv=channels,
        atomic_numbers=list(SPECIES), max_atomvalue=10, num_interaction=L, lmax=lmax,
        function_type_main="bessel", polynomial_cutoff_p=6, save_contraction_order=corr,
        avg_num_neighbors=24.0, internal_compute_dtype=dtype, device=device,
    ).to(device=device, dtype=dtype)
    base.eval()

    # ---- (1) key-set check + weight transfer ----
    fix_sd, base_sd = fix.state_dict(), base.state_dict()
    missing = [k for k in base_sd if k not in fix_sd]          # base keys not in fix
    fix_only = [k for k in fix_sd if k not in base_sd]          # fix keys base lacks
    print(f"[equiv] base #params={len(base_sd)}  fix(baseline,native-mace) #params={len(fix_sd)}")
    print(f"[equiv] base keys missing from fix : {len(missing)} {missing[:5]}")
    print(f"[equiv] fix-only keys (want 0)      : {len(fix_only)} {fix_only[:8]}")
    base.load_state_dict({k: v for k, v in fix_sd.items() if k in base_sd}, strict=True)

    # ---- (2) numerical equivalence ----
    graph = make_fixed_graph(num_nodes=atoms, avg_degree=16, dtype=dtype, device=device, seed=7)
    e_fix, f_fix = _ef(fix, graph)
    e_base, f_base = _ef(base, graph)
    dE = (e_base - e_fix).abs().item()
    dF = (f_base - f_fix).abs().max().item()
    escale = abs(e_fix.item()) + 1e-30
    fscale = f_fix.abs().max().item() + 1e-30
    print(f"[equiv] E fix={e_fix.item():.10e} base={e_base.item():.10e} |dE|={dE:.3e} rel={dE/escale:.3e}")
    print(f"[equiv] F max|d|={dF:.3e} scale={fscale:.3e} rel={dF/fscale:.3e}")
    tol = 1e-10
    ok_equiv = (dE / escale <= tol) and (dF / fscale <= tol) and not missing and not fix_only

    # ---- (3) equivariance of the extracted model ----
    R = random_rotation(dtype=dtype).to(device)
    graph_rot = (graph[0] @ R.T,) + tuple(graph[1:])
    e0, f0 = _ef(base, graph)
    er, fr = _ef(base, graph_rot)
    e_err = (er - e0).abs().item()
    f_err = (fr - f0 @ R.T).abs().max().item()
    print(f"[equiv] equivariance |E(Rx)-E(x)|={e_err:.3e}  |F(Rx)-F(x)R^T|={f_err:.3e}")
    ok_eq = (e_err <= 1e-8 * max(1.0, abs(e0.item()))) and (f_err <= 1e-8 * max(1.0, f0.abs().max().item()))

    ok = ok_equiv and ok_eq
    print(f"[equiv] {'PASS' if ok else 'FAIL'} (equivalence={ok_equiv}, equivariance={ok_eq})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
