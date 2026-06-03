#!/usr/bin/env python3
"""Comprehensive self-validation of PureCartesianICTDMaceBaseline.

Per request: (A) trains normally, (B) compiled-autograd usable in training,
(C) equivariance guaranteed. Each tested with long-range OFF (default) and ON
(latent-coulomb). LAMMPS interface is tested separately.

Run: python -m molecular_force_field.test._mace_baseline_full_validate
"""
from __future__ import annotations

import tempfile

import torch

from molecular_force_field.test.bench_ictd_fix_trainstep import (
    make_fixed_graph, random_rotation, _apply_strain, _det3x3, SPECIES,
)
from molecular_force_field.test._trainer_smoke import build_trainer
from molecular_force_field.models.pure_cartesian_ictd_mace_baseline import (
    PureCartesianICTDMaceBaseline,
)

LR_ON = dict(long_range_mode="latent-coulomb", long_range_backend="dense_pairwise",
             long_range_boundary="nonperiodic")


def _build(dtype, device, **lr):
    return PureCartesianICTDMaceBaseline(
        max_embed_radius=5.0, main_number_of_basis=8, hidden_dim_conv=16,
        atomic_numbers=list(SPECIES), max_atomvalue=10, num_interaction=2, lmax=2,
        function_type_main="bessel", polynomial_cutoff_p=6, save_contraction_order=3,
        avg_num_neighbors=24.0, internal_compute_dtype=dtype, device=device, **lr,
    ).to(device=device, dtype=dtype)


def _ef(model, graph, create_graph=False):
    pos = graph[0].detach().clone().requires_grad_(True)
    e = model(pos, *graph[1:]).sum()
    f = -torch.autograd.grad(e, pos, create_graph=create_graph)[0]
    return e, f


# ---------- (A) training ----------
def test_train(device, dtype, lr, tag):
    torch.manual_seed(0)
    base = _build(dtype, device, **lr)
    with tempfile.TemporaryDirectory() as ck:
        tr = build_trainer("baseline", dtype, device, 48, ck, stress=True, e3=base)
        losses = []
        for ep in range(5):
            m = tr.train_epoch(ep)
            losses.append(float(m["energy_loss"]) + float(m["force_loss"]))
    finite = all((l == l) and abs(l) < 1e30 for l in losses)
    decreasing = losses[-1] <= losses[0]
    ok = finite and decreasing
    print(f"[A train {tag:10s}] e+f loss " + " ".join(f"{l:.3e}" for l in losses)
          + f"  finite={finite} decreasing={decreasing} {'PASS' if ok else 'FAIL'}")
    return ok


# ---------- (B) compiled-autograd ----------
def test_ca(device, dtype, lr, tag):
    torch.manual_seed(0)
    base = _build(dtype, device, **lr)
    with tempfile.TemporaryDirectory() as ck:
        tr = build_trainer("baseline", dtype, device, 48, ck, e3=base,
                           train_compiled_autograd=True)
        trainable = [p for p in tr.e3trans.parameters() if p.requires_grad]
        p0 = [p.detach().clone() for p in trainable]
        m = tr.train_epoch(0)
        el, fl = float(m["energy_loss"]), float(m["force_loss"])
        finite = (el == el) and (fl == fl)
        ca_used = bool(getattr(tr, "_ca_checked", False)) and not bool(getattr(tr, "_ca_disabled", True))
        moved = any(not torch.equal(a, b) for a, b in
                    zip(p0, [p for p in tr.e3trans.parameters() if p.requires_grad]))
    ok = ca_used and finite and moved
    print(f"[B  CA   {tag:10s}] CA_used={ca_used} el={el:.4e} fl={fl:.4e} finite={finite} "
          f"moved={moved} {'PASS' if ok else 'FALLBACK/FAIL'}")
    return ok


# ---------- (C) equivariance ----------
def test_equiv(device, dtype, lr, tag):
    torch.manual_seed(0)
    base = _build(dtype, device, **lr); base.eval()
    g = make_fixed_graph(num_nodes=48, avg_degree=16, dtype=dtype, device=device, seed=7)
    R = random_rotation(dtype=dtype).to(device)
    e0, f0 = _ef(base, g)
    grot = (g[0] @ R.T,) + tuple(g[1:])
    er, fr = _ef(base, grot)
    e_err = (er - e0).abs().item()
    f_err = (fr - f0 @ R.T).abs().max().item()
    # stress equivariance: S(Rx) = R S(x) R^T (rotate pos AND cell)
    def _stress(graph):
        p = graph[0].detach().clone().requires_grad_(True)
        A_, b_, es_, ed_, esh_, c_ = graph[1], graph[2], graph[3], graph[4], graph[5], graph[6]
        nm = int(b_.max().item()) + 1
        strn = torch.zeros(nm, 3, 3, device=device, dtype=dtype, requires_grad=True)
        pin, cin = _apply_strain(p, c_, b_, strn)
        out = base(pin, A_, b_, es_, ed_, esh_, cin).sum()
        sg = torch.autograd.grad(out, strn, create_graph=False)[0]
        vol = _det3x3(c_).abs().clamp_min(1e-10)
        return sg / vol.view(-1, 1, 1)
    s0 = _stress(g)
    grot_full = (g[0] @ R.T, g[1], g[2], g[3], g[4], g[5], g[6] @ R.T)
    sr = _stress(grot_full)
    s_err = (sr - torch.matmul(torch.matmul(R, s0), R.transpose(-1, -2))).abs().max().item()
    ok = (e_err <= 1e-8 * max(1.0, abs(e0.item()))) and \
         (f_err <= 1e-8 * max(1.0, f0.abs().max().item())) and \
         (s_err <= 1e-7 * max(1.0, s0.abs().max().item()))
    print(f"[C equiv {tag:10s}] |E(Rx)-E(x)|={e_err:.2e} |F-cov|={f_err:.2e} |S-cov|={s_err:.2e} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    ok = True
    for lr, tag in [({}, "LR-off"), (LR_ON, "LR-on")]:
        ok &= test_train(device, dtype, lr, tag)
        ok &= test_ca(device, dtype, lr, tag)
        ok &= test_equiv(device, dtype, lr, tag)
    print(f"[full-validate] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
