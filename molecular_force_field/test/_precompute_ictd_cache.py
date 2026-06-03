#!/usr/bin/env python3
"""Precompute & bake the ICTD Clebsch-Gordan + pure-U symmetric-contraction cache.

Builds small PureCartesianICTDFix models (pure-U backend) across a grid of
(lmax, correlation). That exercises every cached float64 builder --
build_cg_tensor, build_full_cg_tensor_so3, ictd_u_matrix_so3 -- and writes the
results into the in-repo on-disk cache (molecular_force_field/models/_ictd_cache/).
Commit that directory so every checkout (4090 / Parity / local) reads the tensors
directly instead of recomputing (lmax=3 build ~24s -> <1s on a cache hit).

Default grid: lmax in {1,2,3}, correlation in {2,3} (the common contraction orders).

Run:
  python -m molecular_force_field.test._precompute_ictd_cache
  python -m molecular_force_field.test._precompute_ictd_cache --lmax 1 2 3 4 --corr 2 3
"""
from __future__ import annotations

import argparse
import time

import torch

from molecular_force_field.models import ictd_disk_cache as dc
from molecular_force_field.test.bench_ictd_fix_trainstep import build_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmax", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--corr", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--channels", type=int, default=16)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    print(f"[precompute] cache root: {dc.cache_root()}")
    print(f"[precompute] grid: lmax={args.lmax} x corr={args.corr} (pure-U)")
    before = dc.cache_info()["entries"]
    for lmax in args.lmax:
        for corr in args.corr:
            t0 = time.time()
            build_model(channels=args.channels, lmax=lmax, num_interaction=2,
                        route="fusion", product_backend="ictd-pure-u",
                        correlation=corr, dtype=torch.float64, device=torch.device("cpu"))
            print(f"[precompute] lmax={lmax} corr={corr} : {time.time() - t0:6.2f}s")
    info = dc.cache_info()
    print(f"[precompute] entries {before} -> {info['entries']}  {info['by_namespace']}  "
          f"total={info['total_bytes'] / 1e6:.1f} MB")
    print("[precompute] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
