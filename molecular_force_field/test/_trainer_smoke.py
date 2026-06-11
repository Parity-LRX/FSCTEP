#!/usr/bin/env python3
"""Trainer smoke test on synthetic fixed-shape data: constructs a real Trainer and
runs a couple of train_epoch steps, asserting losses are finite and weights move.

This is the regression harness for the CUDA-graph trainer work: it lets us verify
(1) the train_epoch refactor keeps behavior identical, and (2) the new
train_epoch_cuda_graph path matches eager. Uses synthetic data so it runs without
the real dataset. Run: python -m molecular_force_field.test._trainer_smoke
"""
from __future__ import annotations

import os
import tempfile

import torch
import torch.nn as nn

from molecular_force_field.training.trainer import Trainer
from molecular_force_field.test.bench_ictd_fix_trainstep import build_model, make_fixed_graph, SPECIES


class _DummyDataset:
    def __init__(self, n):
        self.n = n

    def restore_force(self, x):
        return x

    def __len__(self):
        return self.n


def _make_batch(atoms, degree, dtype, seed):
    g = make_fixed_graph(num_nodes=atoms, avg_degree=degree, dtype=dtype, device=torch.device("cpu"), seed=seed)
    pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell = g
    gen = torch.Generator().manual_seed(seed + 1)
    force_ref = torch.randn(atoms, 3, dtype=dtype, generator=gen) * 0.1
    target_energies = torch.randn(1, dtype=dtype, generator=gen)
    stress_ref = torch.zeros(1, 3, 3, dtype=dtype)
    return (pos, A, batch_idx, force_ref, target_energies, edge_src, edge_dst, edge_shifts, cell, stress_ref)


def build_trainer(route, dtype, device, atoms, ckpt_dir, *, stress=False, e3=None, **extra):
    if e3 is None:
        e3 = build_model(channels=16, lmax=2, num_interaction=2, route=route,
                         product_backend="ictd-pure-u", dtype=dtype, device=device)
    batches = [_make_batch(atoms, 16, dtype, seed=i) for i in range(4)]
    ds = _DummyDataset(len(batches))
    trainer = Trainer(
        model=nn.Identity(),
        e3trans=e3,
        train_loader=batches,
        val_loader=[],
        train_dataset=ds,
        val_dataset=ds,
        device=device,
        atomic_energy_keys=torch.tensor(list(SPECIES)),
        atomic_energy_values=torch.zeros(len(SPECIES)),
        epoch_numbers=2,
        learning_rate=1e-3,
        checkpoint_path=os.path.join(ckpt_dir, "ckpt.pth"),
        c=(1.0 if stress else 0.0),
        **extra,
    )
    return trainer


def run_epoch(label, route, dtype, device, atoms, ckpt_dir, **extra):
    torch.manual_seed(0)
    trainer = build_trainer(route, dtype, device, atoms, ckpt_dir, **extra)
    p0 = [p.detach().clone() for p in trainer.e3trans.parameters()]
    m = trainer.train_epoch(0)
    el = float(m["energy_loss"]); fl = float(m["force_loss"])
    finite = (el == el) and (fl == fl) and abs(el) < 1e30 and abs(fl) < 1e30
    moved = any(not torch.equal(a, b) for a, b in zip(p0, trainer.e3trans.parameters()))
    print(f"[smoke] {label:28s} energy_loss={el:.4e} force_loss={fl:.4e} finite={finite} weights_moved={moved}")
    return finite and moved


def _weight_vec(trainer):
    return torch.cat([p.detach().reshape(-1).double().cpu() for p in trainer.e3trans.parameters()])


def run_cg_compare(label, route, dtype, device, atoms, ckpt_dir, *, stress=False):
    """Train one epoch eagerly and one via the CUDA-graph path on identical init
    weights + identical data; assert the resulting weights match (within tol) and moved.
    On non-CUDA the cuda-graph path falls back to eager, so this still holds."""
    torch.manual_seed(0)
    te = build_trainer(route, dtype, device, atoms, ckpt_dir, stress=stress)
    w0 = _weight_vec(te)
    me = te.train_epoch(0)
    we = _weight_vec(te)

    torch.manual_seed(0)
    tg = build_trainer(route, dtype, device, atoms, ckpt_dir, stress=stress, train_cuda_graph=True)
    mg = tg.train_epoch_cuda_graph(0)
    wg = _weight_vec(tg)

    dE = abs(me["energy_loss"] - mg["energy_loss"])
    dF = abs(me["force_loss"] - mg["force_loss"])
    dw = (we - wg).abs().max().item()
    wscale = we.abs().max().item() + 1e-30
    moved = (we - w0).abs().max().item() > 0
    tol = 1e-9 if dtype == torch.float64 else 5e-3
    ok = (dw / wscale) <= tol and moved
    print(f"[cg-smoke] {label:30s} dE={dE:.2e} dF={dF:.2e} d_w(rel)={dw/wscale:.2e} "
          f"moved={moved} {'PASS' if ok else 'FAIL'}")
    return ok


def run_ca_check(label, route, dtype, device, atoms, ckpt_dir, *, stress=False):
    """Build a real Trainer with --train-compiled-autograd ON, run one epoch, and verify:
    compiled-autograd actually ENGAGED (did not fall back to eager), the fusion_readouts
    biases froze (4 for fusion / 0 for baseline), losses finite, and trainable weights moved."""
    torch.manual_seed(0)
    trainer = build_trainer(route, dtype, device, atoms, ckpt_dir, stress=stress,
                            train_compiled_autograd=True)
    frozen = [n for n, p in trainer.e3trans.named_parameters()
              if (not p.requires_grad) and ("fusion_readouts" in n) and n.endswith(".bias")]
    w0 = [p.detach().clone() for p in trainer.e3trans.parameters() if p.requires_grad]
    m = trainer.train_epoch(0)
    el, fl = float(m["energy_loss"]), float(m["force_loss"])
    finite = (el == el) and (fl == fl) and abs(el) < 1e30 and abs(fl) < 1e30
    ca_used = bool(getattr(trainer, "_ca_checked", False)) and not bool(getattr(trainer, "_ca_disabled", True))
    moved = any(not torch.equal(a, b) for a, b in
                zip(w0, [p for p in trainer.e3trans.parameters() if p.requires_grad]))
    exp_frozen = 4 if route == "fusion" else 0
    ok = finite and ca_used and moved and (len(frozen) == exp_frozen)
    print(f"[ca-smoke] {label:30s} CA_used={ca_used} frozen_bias={len(frozen)}(exp {exp_frozen}) "
          f"el={el:.4e} fl={fl:.4e} finite={finite} moved={moved} {'PASS' if ok else 'FAIL'}")
    return ok


def run_makefx_compare(label, route, dtype, device, atoms, ckpt_dir):
    """Train one epoch eagerly and one via --train-makefx-compile on identical init
    weights + data; assert weights match within tol and moved. make_fx flattens the
    forward + force-autograd then torch.compile's it, so Inductor REORDERS float ops:
    this is machine-precision close, NOT the bit-identical replay cuda-graph gives, so
    the tolerance is looser. Also asserts the path actually engaged (didn't fall back)."""
    torch.manual_seed(0)
    te = build_trainer(route, dtype, device, atoms, ckpt_dir)
    w0 = _weight_vec(te)
    me = te.train_epoch(0)
    we = _weight_vec(te)

    torch.manual_seed(0)
    tm = build_trainer(route, dtype, device, atoms, ckpt_dir, train_makefx_compile=True)
    mm = tm.train_epoch(0)
    wm = _weight_vec(tm)
    used = bool(tm.train_makefx_compile) and not bool(getattr(tm, "_makefx_disabled", True))

    dE = abs(me["energy_loss"] - mm["energy_loss"])
    dF = abs(me["force_loss"] - mm["force_loss"])
    dw = (we - wm).abs().max().item()
    wscale = we.abs().max().item() + 1e-30
    moved = (we - w0).abs().max().item() > 0
    tol = 1e-6 if dtype == torch.float64 else 5e-3
    ok = used and (dw / wscale) <= tol and moved
    print(f"[mfx-smoke] {label:30s} used={used} dE={dE:.2e} dF={dF:.2e} d_w(rel)={dw/wscale:.2e} "
          f"moved={moved} {'PASS' if ok else 'FAIL'}")
    return ok


def _pad_batch(batch, n_max):
    """Append dummy atoms to a collated batch to reach n_max nodes, emitting atom_mask in extras.
    Dummies: pos=0 (no edges reference them), species copies a real one, force 0, batch_idx=0.
    Mirrors H5Dataset.pad_nodes_to_max + collate so we can test the trainer's masking end-to-end."""
    pos, A, batch_idx, force_ref, target_e, src, dst, shifts, cell, stress = batch
    N = pos.shape[0]
    npad = n_max - N
    if npad <= 0:
        extras = {"atom_mask": torch.ones(N, dtype=pos.dtype)}
        return (pos, A, batch_idx, force_ref, target_e, src, dst, shifts, cell, stress, extras)
    pos_p = torch.cat([pos, pos.new_zeros(npad, 3)])
    A_p = torch.cat([A, A[:1].expand(npad).clone()])
    f_p = torch.cat([force_ref, force_ref.new_zeros(npad, 3)])
    bi_p = torch.cat([batch_idx, batch_idx.new_zeros(npad)])  # dummies belong to molecule 0 (energy masked out)
    mask = torch.cat([torch.ones(N, dtype=pos.dtype), torch.zeros(npad, dtype=pos.dtype)])
    return (pos_p, A_p, bi_p, f_p, target_e, src, dst, shifts, cell, stress, {"atom_mask": mask})


def run_padnodes_compare(label, route, dtype, device, atoms, ckpt_dir, *, stress=False):
    """Train one epoch on the plain batches and one on the SAME batches padded with dummy atoms
    (+ atom_mask); assert weights + losses are bit-identical (dummies must not perturb anything).
    This is the numerical gate for pad_nodes_to_max: a padded batch must equal the unpadded one."""
    torch.manual_seed(0)
    tu = build_trainer(route, dtype, device, atoms, ckpt_dir, stress=stress)
    w0 = _weight_vec(tu)
    mu = tu.train_epoch(0)
    wu = _weight_vec(tu)

    torch.manual_seed(0)
    tp = build_trainer(route, dtype, device, atoms, ckpt_dir, stress=stress)
    n_max = atoms + 7  # pad every frame to atoms+7 dummy-padded nodes
    tp.train_loader = [_pad_batch(b, n_max) for b in tp.train_loader]
    mp = tp.train_epoch(0)
    wp = _weight_vec(tp)

    dE = abs(mu["energy_loss"] - mp["energy_loss"])
    dF = abs(mu["force_loss"] - mp["force_loss"])
    dw = (wu - wp).abs().max().item()
    wscale = wu.abs().max().item() + 1e-30
    moved = (wu - w0).abs().max().item() > 0
    tol = 1e-9 if dtype == torch.float64 else 1e-4
    ok = (dE <= tol) and (dF <= tol) and (dw / wscale <= tol) and moved
    print(f"[padnodes] {label:28s} dE={dE:.2e} dF={dF:.2e} d_w(rel)={dw/wscale:.2e} "
          f"moved={moved} {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # MFF_SMOKE_SUITE: "all" (default, original suite) / "makefx" (only make_fx vs eager) /
    # "all+makefx" (everything). make_fx is gated off the default run because it triggers
    # an Inductor compile (slow on CPU) and is the newest, opt-in path.
    suite = os.environ.get("MFF_SMOKE_SUITE", "all").strip().lower()
    run_base = suite in ("all", "all+makefx")
    run_mfx = suite in ("makefx", "all+makefx")
    run_padnodes = suite in ("padnodes", "all+makefx")
    with tempfile.TemporaryDirectory() as ckpt_dir:
        ok = True
        if run_base:
            ok &= run_epoch("fusion energy+force", "fusion", torch.float64, device, 48, ckpt_dir)
            ok &= run_epoch("fusion energy+force+stress", "fusion", torch.float64, device, 48, ckpt_dir, stress=True)
            ok &= run_epoch("baseline energy+force", "baseline", torch.float64, device, 48, ckpt_dir)
            # CUDA-graph path must match eager (numerically identical replay).
            ok &= run_cg_compare("fusion energy+force", "fusion", torch.float64, device, 48, ckpt_dir)
            ok &= run_cg_compare("fusion energy+force+stress", "fusion", torch.float64, device, 48, ckpt_dir, stress=True)
            ok &= run_cg_compare("baseline energy+force", "baseline", torch.float64, device, 48, ckpt_dir)
            # --train-compiled-autograd must actually engage (not silently fall back to eager).
            ok &= run_ca_check("fusion +compiled-autograd", "fusion", torch.float64, device, 48, ckpt_dir)
            ok &= run_ca_check("fusion+stress +CA", "fusion", torch.float64, device, 48, ckpt_dir, stress=True)
            ok &= run_ca_check("baseline +CA", "baseline", torch.float64, device, 48, ckpt_dir)
        if run_padnodes:
            # pad_nodes_to_max must be bit-identical to unpadded (dummy atoms perturb nothing).
            ok &= run_padnodes_compare("fusion energy+force", "fusion", torch.float64, device, 48, ckpt_dir)
            ok &= run_padnodes_compare("fusion energy+force+stress", "fusion", torch.float64, device, 48, ckpt_dir, stress=True)
            ok &= run_padnodes_compare("baseline energy+force", "baseline", torch.float64, device, 48, ckpt_dir)
        if run_mfx:
            # make_fx-compile path must match eager (machine-precision, Inductor reorders floats).
            ok &= run_makefx_compare("fusion energy+force", "fusion", torch.float64, device, 48, ckpt_dir)
            ok &= run_makefx_compare("baseline energy+force", "baseline", torch.float64, device, 48, ckpt_dir)
    print(f"[smoke] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
