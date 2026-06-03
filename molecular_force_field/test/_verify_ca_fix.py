#!/usr/bin/env python3
"""Verify the steady-state compiled-autograd fallback.

VARYING-shape batches + compiled-autograd previously CRASHED on a later-batch recompile
(torch CopySlices/symbolic-sizes), for BOTH mae and smooth-l1. With the fix, the
steady-state backward catches it -> disables CA -> eager for the rest, no crash.
Expected after fix: every case 'OK no-crash  ca_disabled=True'.
"""
import tempfile
import torch
from molecular_force_field.test._trainer_smoke import build_trainer, _make_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[fix] cuda={torch.cuda.is_available()} torch={torch.__version__}")
for route in ("baseline", "fusion"):
    for lf in ("mae", "smooth-l1"):
        with tempfile.TemporaryDirectory() as ck:
            torch.manual_seed(0)
            tr = build_trainer(route, torch.float64, device, 48, ck,
                               loss_function=lf, train_compiled_autograd=True)
            tr.train_loader = [_make_batch(a, 16, torch.float64, seed=i)
                               for i, a in enumerate([48, 56, 64, 72, 80])]
            try:
                tr.train_epoch(0)
                dis = bool(getattr(tr, "_ca_disabled", False))
                print(f"[fix] {route} {lf:10s} varying-shape: OK no-crash  ca_disabled={dis}")
            except Exception as e:
                print(f"[fix] {route} {lf:10s} varying-shape: STILL-CRASH {type(e).__name__}: {str(e)[:160]}")
