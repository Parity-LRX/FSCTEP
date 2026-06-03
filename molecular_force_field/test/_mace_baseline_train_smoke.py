#!/usr/bin/env python3
"""Trainability smoke for the extracted PureCartesianICTDMaceBaseline.

Builds the standalone baseline, wraps it in the production Trainer (reusing
_trainer_smoke.build_trainer via the e3 override), and runs a train_epoch on
synthetic fixed-shape data with energy+force (+stress). Asserts losses are
finite and weights move -> confirms the extracted model trains end-to-end
(forward -> force double-backward -> optimizer) through the real Trainer.

Run: python -m molecular_force_field.test._mace_baseline_train_smoke
"""
from __future__ import annotations

import tempfile

import torch

from molecular_force_field.test._trainer_smoke import build_trainer, SPECIES
from molecular_force_field.models.pure_cartesian_ictd_mace_baseline import (
    PureCartesianICTDMaceBaseline,
)


def _build_baseline(dtype, device):
    return PureCartesianICTDMaceBaseline(
        max_embed_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=16,
        atomic_numbers=list(SPECIES),
        max_atomvalue=10,
        num_interaction=2,
        lmax=2,
        function_type_main="bessel",
        polynomial_cutoff_p=6,
        save_contraction_order=3,
        avg_num_neighbors=24.0,
        internal_compute_dtype=dtype,
        device=device,
    ).to(device=device, dtype=dtype)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    ok = True
    for stress in (False, True):
        torch.manual_seed(0)
        base = _build_baseline(dtype, device)
        with tempfile.TemporaryDirectory() as ckpt:
            trainer = build_trainer("baseline", dtype, device, 48, ckpt, stress=stress, e3=base)
            p0 = [p.detach().clone() for p in trainer.e3trans.parameters()]
            m = trainer.train_epoch(0)
            el, fl = float(m["energy_loss"]), float(m["force_loss"])
            finite = (el == el) and (fl == fl) and abs(el) < 1e30 and abs(fl) < 1e30
            moved = any(
                not torch.equal(a, b)
                for a, b in zip(p0, trainer.e3trans.parameters())
            )
        tag = "energy+force+stress" if stress else "energy+force"
        print(f"[base-train] {tag:20s} energy_loss={el:.4e} force_loss={fl:.4e} finite={finite} moved={moved}")
        ok &= finite and moved
    print(f"[base-train] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
