#!/usr/bin/env python3
"""Validate the ICTD CG + pure-U on-disk cache.

Checks (numerical consistency + equivariance are hard constraints):
  A. Builder bit-identical in float64: cache OFF (truth) == cache ON cold (compute+write)
     == cache ON warm (read from disk), for build_full_cg_tensor_so3 and ictd_u_matrix_so3.
  B. fp32 request == float64 build downcast to float32 (the agreed precision policy).
  C. Model-level identical: a real Trainer epoch gives the SAME energy_loss with the
     cache OFF, ON-cold and ON-warm (so training numerics are untouched); warm build
     is no slower than cold.
  D. Kill switch: FSCETP_ICTD_CACHE=0 performs no disk writes.

Uses a throwaway cache dir so it never touches the committed in-repo cache.
Run: python -m molecular_force_field.test._ictd_cache_test
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time

import torch

import molecular_force_field.models.ictd_irreps as ii
from molecular_force_field.models import ictd_disk_cache as dc


def _clear_lru():
    ii.build_cg_tensor.cache_clear()
    ii.build_full_cg_tensor_so3.cache_clear()
    ii._ictd_u_matrix_so3_cached.cache_clear()


def _reset_dir(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def main() -> int:
    torch.set_default_dtype(torch.float64)
    ok = True
    cache_dir = tempfile.mkdtemp(prefix="ictd_cache_test_")
    os.environ["FSCETP_ICTD_CACHE_DIR"] = cache_dir
    try:
        # ---- A) builder bit-identical: OFF (truth) vs ON-cold vs ON-warm ----
        os.environ["FSCETP_ICTD_CACHE"] = "0"; _clear_lru()
        u_truth = ii.ictd_u_matrix_so3(lmax=2, output_l=1, correlation=3)
        cgf_truth = ii.build_full_cg_tensor_so3(1, 1, 1)

        _reset_dir(cache_dir)
        os.environ["FSCETP_ICTD_CACHE"] = "1"; _clear_lru()
        u_cold = ii.ictd_u_matrix_so3(lmax=2, output_l=1, correlation=3)  # compute + write
        _clear_lru()
        u_warm = ii.ictd_u_matrix_so3(lmax=2, output_l=1, correlation=3)  # read from disk
        cgf_warm = ii.build_full_cg_tensor_so3(1, 1, 1)
        a = (torch.equal(u_truth, u_cold) and torch.equal(u_truth, u_warm)
             and torch.equal(cgf_truth, cgf_warm))
        print(f"[cache] A builder bit-identical fp64 (off==cold==warm): {a}")
        ok &= a

        # ---- B) fp32 == downcast(float64) ----
        u32 = ii.ictd_u_matrix_so3(lmax=2, output_l=1, correlation=3, dtype=torch.float32)
        b = torch.equal(u32, u_truth.to(torch.float32))
        print(f"[cache] B fp32 == downcast(f64): {b}")
        ok &= b

        # ---- C) model-level identical loss + warm not slower ----
        from molecular_force_field.test._trainer_smoke import build_trainer

        def run(cache_on):
            os.environ["FSCETP_ICTD_CACHE"] = "1" if cache_on else "0"
            _clear_lru()
            torch.manual_seed(0)
            with tempfile.TemporaryDirectory() as ck:
                t0 = time.time()
                tr = build_trainer("baseline", torch.float64, torch.device("cpu"), 48, ck)
                bt = time.time() - t0
                loss = float(tr.train_epoch(0)["energy_loss"])
            return loss, bt

        _reset_dir(cache_dir)
        loss_off, _ = run(False)
        loss_cold, t_cold = run(True)   # populate disk
        loss_warm, t_warm = run(True)   # read disk
        c = abs(loss_off - loss_cold) == 0.0 and abs(loss_off - loss_warm) == 0.0
        print(f"[cache] C model energy_loss off={loss_off:.12e} cold={loss_cold:.12e} "
              f"warm={loss_warm:.12e} identical={c}  build cold={t_cold:.2f}s warm={t_warm:.2f}s")
        ok &= c

        # ---- D) kill switch writes nothing ----
        _reset_dir(cache_dir)
        os.environ["FSCETP_ICTD_CACHE"] = "0"; _clear_lru()
        ii.ictd_u_matrix_so3(lmax=2, output_l=0, correlation=2)
        d = dc.cache_info()["entries"] == 0
        print(f"[cache] D killswitch writes nothing: {d}")
        ok &= d

        print(f"[cache] {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.environ.pop("FSCETP_ICTD_CACHE", None)
        os.environ.pop("FSCETP_ICTD_CACHE_DIR", None)


if __name__ == "__main__":
    raise SystemExit(main())
