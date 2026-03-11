#!/usr/bin/env python3
"""
Benchmark build_cg_tensor (vectorized matrix ops).

Usage:
  python -m molecular_force_field.test.benchmark_cg_build
  python -m molecular_force_field.test.benchmark_cg_build --lmax 6 --repeats 20
"""

from __future__ import annotations

import argparse
import time

from molecular_force_field.models.ictd_irreps import build_cg_tensor


def collect_paths(lmax: int) -> list[tuple[int, int, int]]:
    """All (l1,l2,l3) paths for lmax."""
    paths = []
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            for l3 in range(abs(l1 - l2), min(l1 + l2, lmax) + 1):
                if (l1 + l2 + l3) % 2 == 1:
                    continue
                paths.append((l1, l2, l3))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmax", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    paths = collect_paths(args.lmax)
    print(f"lmax={args.lmax}, paths={len(paths)}")
    print()

    # Cold build (cache cleared each run)
    from molecular_force_field.models.ictd_irreps import _build_poly_mult_matrix

    def run_cold():
        build_cg_tensor.cache_clear()
        _build_poly_mult_matrix.cache_clear()
        t0 = time.perf_counter()
        for l1, l2, l3 in paths:
            _ = build_cg_tensor(l1, l2, l3)
        return time.perf_counter() - t0

    cold_times = [run_cold() for _ in range(args.repeats)]
    cold_mean = sum(cold_times) / len(cold_times) * 1000
    cold_std = (sum((t * 1000 - cold_mean) ** 2 for t in cold_times) / len(cold_times)) ** 0.5
    print("Cold build (cache cleared each run):")
    print(f"  {cold_mean:.2f} ± {cold_std:.2f} ms")
    print()

    # Cached (warm)
    build_cg_tensor.cache_clear()
    _build_poly_mult_matrix.cache_clear()
    for l1, l2, l3 in paths:
        _ = build_cg_tensor(l1, l2, l3)

    t0 = time.perf_counter()
    for _ in range(args.repeats):
        for l1, l2, l3 in paths:
            _ = build_cg_tensor(l1, l2, l3)
    cached_ms = (time.perf_counter() - t0) / args.repeats * 1000
    print("Cached (warm, cache hit):")
    print(f"  {cached_ms:.4f} ms")


if __name__ == "__main__":
    main()
