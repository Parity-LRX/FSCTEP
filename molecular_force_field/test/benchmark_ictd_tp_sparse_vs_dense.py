#!/usr/bin/env python3
"""
Benchmark: sparse Triton vs dense Triton vs PyTorch for HarmonicFullyConnectedTensorProduct.

Compares three projection paths:
  1. Sparse + Triton (ICTD_USE_SPARSE_TP=1, ICTD_USE_TRITON_TP=1) — sparse CG when zero_frac >= 40%
  2. Dense Triton only (ICTD_USE_SPARSE_TP=0, ICTD_USE_TRITON_TP=1)
  3. PyTorch only (ICTD_USE_TRITON_TP=0)

Usage:
  python -m molecular_force_field.benchmark_ictd_tp_sparse_vs_dense
  python -m molecular_force_field.benchmark_ictd_tp_sparse_vs_dense --device cuda --batch 64 --lmax 4 --repeat 50
  python -m molecular_force_field.benchmark_ictd_tp_sparse_vs_dense --numerics  # include numerical comparison
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import torch

# We will reload ictd_irreps per mode to pick up env vars
import molecular_force_field.models.ictd_irreps as ictd_irreps_module


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _run_forward(
    tp: Any,
    x1: Dict[int, torch.Tensor],
    x2: Dict[int, torch.Tensor],
    device: torch.device,
    warmup: int,
    repeat: int,
) -> Tuple[float, Dict[int, torch.Tensor]]:
    """Run forward warmup + repeat; return mean time (ms) and last output (for numerics)."""
    tp.train()
    for _ in range(warmup):
        _ = tp(x1, x2)
    _sync(device)
    times_ms: List[float] = []
    out_last: Dict[int, torch.Tensor] = {}
    for _ in range(repeat):
        _sync(device)
        t0 = time.perf_counter()
        out_last = tp(x1, x2)
        _sync(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = sum(times_ms) / len(times_ms)
    return mean_ms, out_last


def _max_abs_diff(out_a: Dict[int, torch.Tensor], out_b: Dict[int, torch.Tensor]) -> float:
    """Max absolute difference over all l outputs."""
    diffs = [torch.abs(out_a[l] - out_b[l]).max().item() for l in out_a if l in out_b]
    return max(diffs) if diffs else 0.0


def run_benchmark(
    device: torch.device,
    dtype: torch.dtype,
    *,
    batch: int,
    mul_in1: int,
    mul_in2: int,
    mul_out: int,
    lmax: int,
    path_policy: str,
    max_rank_other: int | None,
    warmup: int,
    repeat: int,
    check_numerics: bool,
    seed: int,
) -> Dict[str, Any]:
    HarmonicFullyConnectedTensorProduct = ictd_irreps_module.HarmonicFullyConnectedTensorProduct

    # Build reference TP and inputs once (will clone state for each mode)
    torch.manual_seed(seed)
    tp_ref = HarmonicFullyConnectedTensorProduct(
        mul_in1=mul_in1,
        mul_in2=mul_in2,
        mul_out=mul_out,
        lmax=lmax,
        internal_weights=True,
        path_policy=path_policy,
        max_rank_other=max_rank_other,
    ).to(device=device, dtype=dtype)
    state_dict = {k: v.clone() for k, v in tp_ref.state_dict().items()}

    x1 = {l: torch.randn(batch, mul_in1, 2 * l + 1, device=device, dtype=dtype) for l in range(lmax + 1)}
    x2 = {l: torch.randn(batch, mul_in2, 2 * l + 1, device=device, dtype=dtype) for l in range(lmax + 1)}

    modes = [
        ("sparse_triton", {"ICTD_USE_SPARSE_TP": "1", "ICTD_USE_TRITON_TP": "1"}),
        ("dense_triton", {"ICTD_USE_SPARSE_TP": "0", "ICTD_USE_TRITON_TP": "1"}),
        ("pytorch", {"ICTD_USE_TRITON_TP": "0"}),
    ]

    results: Dict[str, Any] = {"batch": batch, "lmax": lmax, "device": str(device), "modes": {}}

    ref_out: Dict[int, torch.Tensor] | None = None

    for name, env_overrides in modes:
        for k, v in env_overrides.items():
            os.environ[k] = v
        importlib.reload(ictd_irreps_module)
        HarmonicFullyConnectedTensorProduct = ictd_irreps_module.HarmonicFullyConnectedTensorProduct

        tp = HarmonicFullyConnectedTensorProduct(
            mul_in1=mul_in1,
            mul_in2=mul_in2,
            mul_out=mul_out,
            lmax=lmax,
            internal_weights=True,
            path_policy=path_policy,
            max_rank_other=max_rank_other,
        ).to(device=device, dtype=dtype)
        tp.load_state_dict(state_dict)

        try:
            mean_ms, out = _run_forward(tp, x1, x2, device, warmup=warmup, repeat=repeat)
            results["modes"][name] = {"time_ms": mean_ms, "out": out if check_numerics else None}
            if name == "pytorch" and check_numerics:
                ref_out = out
        except Exception as e:
            results["modes"][name] = {"time_ms": None, "error": str(e), "out": None}

    # Numerics: compare sparse and dense Triton to PyTorch reference
    if check_numerics and ref_out is not None:
        for name in ("sparse_triton", "dense_triton"):
            if name in results["modes"] and results["modes"][name].get("out") is not None:
                diff = _max_abs_diff(ref_out, results["modes"][name]["out"])
                results["modes"][name]["max_abs_diff_vs_pytorch"] = diff
        results["ref_out_sum"] = sum(ref_out[l].sum().item() for l in ref_out)

    return results


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Benchmark ICTD TP: sparse vs dense Triton vs PyTorch")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "float64"))
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--mul", type=int, default=8, help="mul_in1, mul_in2, mul_out")
    parser.add_argument("--lmax", type=int, default=3, help="lmax")
    parser.add_argument("--path-policy", type=str, default="max_rank_other", choices=("full", "max_rank_other"))
    parser.add_argument("--max-rank-other", type=int, default=3, help="Used when path_policy=max_rank_other")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--numerics", action="store_true", help="Compare outputs to PyTorch reference")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    mul = args.mul
    max_rank = args.max_rank_other if args.path_policy == "max_rank_other" else None

    print("[benchmark_ictd_tp_sparse_vs_dense]", flush=True)
    print(f"  device={device}, dtype={args.dtype}, batch={args.batch}, mul={mul}, lmax={args.lmax}", flush=True)
    print(f"  path_policy={args.path_policy}, max_rank_other={max_rank}", flush=True)
    print(f"  warmup={args.warmup}, repeat={args.repeat}, check_numerics={args.numerics}", flush=True)

    out = run_benchmark(
        device,
        dtype,
        batch=args.batch,
        mul_in1=mul,
        mul_in2=mul,
        mul_out=mul,
        lmax=args.lmax,
        path_policy=args.path_policy,
        max_rank_other=max_rank,
        warmup=args.warmup,
        repeat=args.repeat,
        check_numerics=args.numerics,
        seed=args.seed,
    )

    # Report
    print("", flush=True)
    times: Dict[str, float] = {}
    for name, data in out["modes"].items():
        t = data.get("time_ms")
        if t is not None:
            times[name] = t
            err = data.get("error")
            diff = data.get("max_abs_diff_vs_pytorch")
            extra = ""
            if err:
                extra = f"  [error: {err}]"
            elif diff is not None:
                extra = f"  max|diff vs PyTorch| = {diff:.2e}"
            print(f"  {name:18s}  {t:.3f} ms/iter{extra}", flush=True)
        else:
            print(f"  {name:18s}  FAILED  {data.get('error', '')}", flush=True)

    if len(times) >= 2 and "pytorch" in times:
        t_ref = times["pytorch"]
        print("", flush=True)
        if device.type != "cuda":
            print("  (On CPU, Triton paths fall back to PyTorch; run with --device cuda for real comparison.)", flush=True)
        print("  Speedup vs PyTorch:", flush=True)
        for name in ("sparse_triton", "dense_triton"):
            if name in times and t_ref > 0:
                ratio = t_ref / times[name]
                print(f"    {name}: {ratio:.2f}x", flush=True)
    if args.numerics and "ref_out_sum" in out:
        print(f"  ref_out_sum (PyTorch): {out['ref_out_sum']:.6f}", flush=True)


if __name__ == "__main__":
    main()
