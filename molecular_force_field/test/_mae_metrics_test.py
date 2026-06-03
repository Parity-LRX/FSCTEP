#!/usr/bin/env python3
"""Verify MAE is surfaced into the MAIN metric pipeline (train_epoch return dict,
the fast-path train metrics, and loss.csv) -- not just per-fidelity logging.

Two checks:
  1. train_epoch(...) return dict now carries real (nonzero) energy/force/stress MAE
     (energy_mae, energy_mae_avg, force_mae, stress_mae), alongside the existing RMSE.
  2. Driving validate() via a small dump_frequency (the real fast-path, train_eval_
     sample_ratio=0) writes loss.csv with BOTH train_*_mae and val_*_mae columns, and
     the train MAE is a real number (not the old 0.0 placeholder).

Runs on CPU/float64 without the real dataset. On CUDA the cuda-graph compare in
_trainer_smoke covers the replay path; here we only need the eager metric pipeline.

Run: python -m molecular_force_field.test._mae_metrics_test
"""
from __future__ import annotations

import os
import shutil
import tempfile

import pandas as pd
import torch
import torch.nn as nn

from molecular_force_field.training.trainer import Trainer
from molecular_force_field.test.bench_ictd_fix_trainstep import build_model, SPECIES
from molecular_force_field.test._trainer_smoke import _make_batch


class _Dataset:
    """Identity unit-restoration so validate() can run on synthetic data."""
    def __init__(self, n):
        self.n = n

    def restore_force(self, x):
        return x

    def restore_energy(self, x):
        return x

    def __len__(self):
        return self.n


def _build(ckpt_dir, dtype, device, *, dump_frequency):
    e3 = build_model(channels=16, lmax=2, num_interaction=2, route="baseline",
                     product_backend="ictd-pure-u", dtype=dtype, device=device)
    batches = [_make_batch(48, 16, dtype, seed=i) for i in range(4)]
    # nonzero stress reference so stress MAE is a real, distinguishable number
    batches = [(*b[:9], torch.randn(1, 3, 3, dtype=dtype) * 0.05) for b in batches]
    ds = _Dataset(len(batches))
    tr = Trainer(
        model=nn.Identity(),
        e3trans=e3,
        train_loader=batches,
        val_loader=batches,            # reuse train batches as a synthetic val set
        train_dataset=ds,
        val_dataset=ds,
        device=device,
        atomic_energy_keys=torch.tensor(list(SPECIES)),
        atomic_energy_values=torch.zeros(len(SPECIES)),
        epoch_numbers=1,
        learning_rate=1e-3,
        checkpoint_path=os.path.join(ckpt_dir, "ckpt.pth"),
        c=1.0,                          # enable stress
        dump_frequency=dump_frequency,  # fire validate() inside train_epoch
        train_eval_sample_ratio=0.0,    # force the fast-path full_train_metrics
        save_val_csv=False,
    )
    # Trainer hardcodes checkpoint_dir='checkpoint'; redirect outputs into the
    # per-check tempdir so loss.csv lands where we read it and checks don't collide.
    tr.checkpoint_dir = ckpt_dir
    tr.loss_csv_path = os.path.join(ckpt_dir, "loss.csv")
    return tr


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    ok = True

    # ---- Check 1: train_epoch return dict carries real MAE ----
    with tempfile.TemporaryDirectory() as ck:
        torch.manual_seed(0)
        tr = _build(ck, dtype, device, dump_frequency=10_000)  # no validate noise
        m = tr.train_epoch(0)
    need = ["energy_mae", "energy_mae_avg", "force_mae", "stress_mae"]
    have = {k: float(m.get(k, float("nan"))) for k in need}
    c1 = all(k in m for k in need) and all(v == v for v in have.values()) \
        and have["energy_mae"] > 0 and have["force_mae"] > 0
    print(f"[mae-metrics] train_epoch return: "
          + " ".join(f"{k}={have[k]:.4e}" for k in need)
          + f"  -> {'OK' if c1 else 'FAIL'}")
    ok &= c1

    # ---- Check 2: loss.csv has real train_*_mae AND val_*_mae columns ----
    with tempfile.TemporaryDirectory() as ck:
        torch.manual_seed(0)
        tr = _build(ck, dtype, device, dump_frequency=2)  # validate fires mid-epoch
        tr.train_epoch(0)
        csv_path = os.path.join(ck, "loss.csv")
        exists = os.path.exists(csv_path)
        df = pd.read_csv(csv_path) if exists else pd.DataFrame()
    cols = list(df.columns)
    want_cols = ["train_energy_mae", "train_force_mae", "train_stress_mae",
                 "val_energy_mae", "val_force_mae", "val_stress_mae",
                 "train_energy_mae_avg", "val_energy_mae_avg"]
    have_cols = [c for c in want_cols if c in cols]
    missing = [c for c in want_cols if c not in cols]
    # train MAE must be a real number now (the old placeholder was a constant 0.0)
    last = df.iloc[-1] if len(df) else {}
    train_e_mae = float(last.get("train_energy_mae", 0.0)) if len(df) else 0.0
    train_f_mae = float(last.get("train_force_mae", 0.0)) if len(df) else 0.0
    val_e_mae = float(last.get("val_energy_mae", 0.0)) if len(df) else 0.0
    val_f_mae = float(last.get("val_force_mae", 0.0)) if len(df) else 0.0
    c2 = exists and not missing and train_e_mae > 0 and train_f_mae > 0 \
        and val_e_mae > 0 and val_f_mae > 0
    print(f"[mae-metrics] loss.csv rows={len(df)} mae_cols_present={len(have_cols)}/{len(want_cols)}"
          + (f" missing={missing}" if missing else ""))
    print(f"[mae-metrics]   train_energy_mae={train_e_mae:.4e} train_force_mae={train_f_mae:.4e} "
          f"val_energy_mae={val_e_mae:.4e} val_force_mae={val_f_mae:.4e}  -> {'OK' if c2 else 'FAIL'}")
    ok &= c2

    # Trainer.__init__ unconditionally makedirs('checkpoint') and writes a stale
    # header there before our redirect; clean it so the test leaves no repo droppings.
    shutil.rmtree("checkpoint", ignore_errors=True)
    print(f"[mae-metrics] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
