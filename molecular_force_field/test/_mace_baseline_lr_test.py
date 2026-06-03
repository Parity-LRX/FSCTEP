#!/usr/bin/env python3
"""Long-range integration test for PureCartesianICTDMaceBaseline.

Covers:
  (1) mode="none" is a true no-op  -> see _mace_baseline_equiv (still bit-identical
      to fix(baseline,native-mace) after the scaffold was added).
  (2) with a long-range backend ON (latent-coulomb / dense_pairwise / nonperiodic),
      the energy_scale parameter inits to 0, so at init the LR term is EXACTLY zero
      -> energy & forces identical to the mode="none" baseline (proves the wiring is
      correct AND adds zero init perturbation), equivariance is preserved, and the
      force double-backward (create_graph=True then backward a force loss) runs.

Run: python -m molecular_force_field.test._mace_baseline_lr_test
"""
from __future__ import annotations

import torch

from molecular_force_field.test.bench_ictd_fix_trainstep import (
    make_fixed_graph,
    random_rotation,
    SPECIES,
)
from molecular_force_field.models.pure_cartesian_ictd_mace_baseline import (
    PureCartesianICTDMaceBaseline,
)


def _build(dtype, device, **lr):
    return PureCartesianICTDMaceBaseline(
        max_embed_radius=5.0, main_number_of_basis=8, hidden_dim_conv=16,
        atomic_numbers=list(SPECIES), max_atomvalue=10, num_interaction=2, lmax=2,
        function_type_main="bessel", polynomial_cutoff_p=6, save_contraction_order=3,
        avg_num_neighbors=24.0, internal_compute_dtype=dtype, device=device, **lr,
    ).to(device=device, dtype=dtype)


def _ef(model, graph, create_graph=False):
    pos = graph[0].detach().clone().requires_grad_(True)
    out = model(pos, *graph[1:])
    e = (out[0] if isinstance(out, tuple) else out).sum()
    f = -torch.autograd.grad(e, pos, create_graph=create_graph)[0]
    return e, f


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    ok = True

    torch.manual_seed(0); base_none = _build(dtype, device); base_none.eval()
    torch.manual_seed(0); base_lr = _build(
        dtype, device,
        long_range_mode="latent-coulomb",
        long_range_backend="dense_pairwise",
        long_range_boundary="nonperiodic",
    ); base_lr.eval()

    # share the non-LR weights so ONLY the (init) LR module differs
    base_lr.load_state_dict(
        {k: v for k, v in base_none.state_dict().items() if k in base_lr.state_dict()},
        strict=False,
    )
    n_extra = len(base_lr.state_dict()) - len(base_none.state_dict())
    print(f"[lr] base #params={len(base_none.state_dict())}  base+LR #params={len(base_lr.state_dict())} (LR adds {n_extra})")
    print(f"[lr] long_range_module = {type(base_lr.long_range_module).__name__}")

    graph = make_fixed_graph(num_nodes=48, avg_degree=16, dtype=dtype, device=device, seed=7)
    e0, f0 = _ef(base_none, graph)
    e1, f1 = _ef(base_lr, graph)
    dE = (e1 - e0).abs().item(); dF = (f1 - f0).abs().max().item()
    print(f"[lr] init zero-contribution (energy_scale init=0): |dE|={dE:.3e} |dF|={dF:.3e} (expect ~0)")
    ok &= (dE <= 1e-18) and (dF <= 1e-18)

    R = random_rotation(dtype=dtype).to(device)
    graph_rot = (graph[0] @ R.T,) + tuple(graph[1:])
    er, fr = _ef(base_lr, graph_rot)
    e_err = (er - e1).abs().item(); f_err = (fr - f1 @ R.T).abs().max().item()
    print(f"[lr] equivariance(LR-on): |E(Rx)-E(x)|={e_err:.3e} |F(Rx)-F(x)R^T|={f_err:.3e}")
    ok &= (e_err <= 1e-8 * max(1.0, abs(e1.item()))) and (f_err <= 1e-8 * max(1.0, f1.abs().max().item()))

    # force double-backward through the LR module must run
    try:
        pos = graph[0].detach().clone().requires_grad_(True)
        out = base_lr(pos, *graph[1:])
        e = (out[0] if isinstance(out, tuple) else out).sum()
        forces = -torch.autograd.grad(e, pos, create_graph=True)[0]
        (forces ** 2).mean().backward()
        gsum = sum(p.grad.abs().sum().item() for p in base_lr.parameters() if p.grad is not None)
        db_ok = (gsum == gsum) and (gsum < 1e30)
        print(f"[lr] force double-backward OK (sum|param.grad|={gsum:.3e})")
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"[lr] force double-backward FAILED: {type(ex).__name__}: {ex}")
        db_ok = False
    ok &= db_ok

    print(f"[lr] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
