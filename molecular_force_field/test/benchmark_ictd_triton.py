#!/usr/bin/env python3
"""
Benchmark: optimized vs baseline for direction_harmonics_all and full ICTD layer on CUDA.

Usage:
  python -m molecular_force_field.benchmark_ictd_triton
  python -m molecular_force_field.benchmark_ictd_triton --sizes 1000 50000 200000 --lmax 4 --repeat 50
  python -m molecular_force_field.benchmark_ictd_triton --full-layer --atoms 64 128 256
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from typing import Dict, List

import torch
import torch.nn as nn

os.environ.setdefault("ICTD_USE_TRITON", "0")

from molecular_force_field.models.ictd_fast import _counts_list
from molecular_force_field.models import ictd_irreps
from molecular_force_field.models.ictd_irreps import (
    direction_harmonics_all,
    direction_harmonics_fast,
    _dir_proj_cpu_f64,
    _dir_monomial_exps_coefs,
    _dir_proj_cache_by_dev_dtype,
    HarmonicFullyConnectedTensorProduct,
)


# ============ Baseline (old) direction_harmonics implementations ============

def _direction_harmonics_fast_baseline(n: torch.Tensor, l: int) -> torch.Tensor:
    """Old implementation: generic ** operator, no power table."""
    if l == 0:
        return torch.ones(*n.shape[:-1], 1, device=n.device, dtype=n.dtype)
    key = (str(n.device), str(n.dtype), int(l))
    P = _dir_proj_cache_by_dev_dtype.get(key)
    if P is None:
        P = _dir_proj_cpu_f64(l).to(device=n.device, dtype=n.dtype)
        _dir_proj_cache_by_dev_dtype[key] = P
    exps, coefs = _dir_monomial_exps_coefs(l)
    exps = exps.to(device=n.device)
    coefs = coefs.to(device=n.device, dtype=n.dtype)
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    a, b, c = exps[:, 0], exps[:, 1], exps[:, 2]
    t = (nx.unsqueeze(-1) ** a) * (ny.unsqueeze(-1) ** b) * (nz.unsqueeze(-1) ** c) * coefs
    return t @ P


def _direction_harmonics_all_baseline(n: torch.Tensor, lmax: int) -> List[torch.Tensor]:
    """Old implementation: separate call per l, generic pow."""
    return [_direction_harmonics_fast_baseline(n, l) for l in range(int(lmax) + 1)]


# ============ Timing helpers ============

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_fn(fn, warmup: int, repeat: int, device) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(repeat):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    mean = sum(times) / len(times)
    std = (sum((x - mean) ** 2 for x in times) / len(times)) ** 0.5
    return mean, std


# ============ Full layer benchmark helpers ============

def make_dummy_graph(device, dtype, num_nodes=128, avg_degree=24, seed=42):
    torch.manual_seed(seed)
    pos = torch.randn(num_nodes, 3, device=device, dtype=dtype) * 2.0
    A = torch.randint(1, 6, (num_nodes,), device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    num_edges = num_nodes * avg_degree
    edge_dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_src = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_shifts = torch.zeros(num_edges, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(num_nodes, -1, -1)
    return pos, A, batch, edge_src, edge_dst, edge_shifts, cell


def _time_layer_forward(layer, pos, A, batch, edge_src, edge_dst, edge_shifts, cell,
                        warmup=5, repeat=30, device=None):
    layer.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        _sync(device or pos.device)
        times = []
        for _ in range(repeat):
            _sync(device or pos.device)
            t0 = time.perf_counter()
            _ = layer(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
            _sync(device or pos.device)
            times.append((time.perf_counter() - t0) * 1000.0)
    mean = sum(times) / len(times)
    std = (sum((x - mean) ** 2 for x in times) / len(times)) ** 0.5
    return mean, std


def _time_layer_forward_backward(layer, pos, A, batch, edge_src, edge_dst, edge_shifts, cell,
                                 warmup=5, repeat=20, device=None):
    layer.train()
    pos_req = pos.detach().requires_grad_(True)
    for _ in range(warmup):
        layer.zero_grad(set_to_none=True)
        if pos_req.grad is not None:
            pos_req.grad = None
        out = layer(pos_req, A, batch, edge_src, edge_dst, edge_shifts, cell)
        out.sum().backward()
    _sync(device or pos.device)
    times = []
    for _ in range(repeat):
        _sync(device or pos.device)
        layer.zero_grad(set_to_none=True)
        if pos_req.grad is not None:
            pos_req.grad = None
        t0 = time.perf_counter()
        out = layer(pos_req, A, batch, edge_src, edge_dst, edge_shifts, cell)
        out.sum().backward()
        _sync(device or pos.device)
        times.append((time.perf_counter() - t0) * 1000.0)
    mean = sum(times) / len(times)
    std = (sum((x - mean) ** 2 for x in times) / len(times)) ** 0.5
    return mean, std



# ============ Main ============

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark optimized vs baseline ICTD: direction harmonics + full layer"
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[5_000, 20_000, 100_000],
                        help="N for direction_harmonics benchmark")
    parser.add_argument("--lmax", type=int, default=2, help="lmax")
    parser.add_argument("--repeat", type=int, default=30, help="Repeat count")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--full-layer", action="store_true",
                        help="Benchmark full PureCartesianICTDTransformerLayer")
    parser.add_argument("--atoms", type=int, nargs="+", default=[64, 128, 256],
                        help="Number of atoms for full-layer benchmark")
    parser.add_argument("--avg-degree", type=int, default=24, help="Average edges per node")
    parser.add_argument("--backward", action="store_true",
                        help="Also benchmark forward+backward for full layer")
    parser.add_argument("--profile", action="store_true",
                        help="Run CUDA profiler on last atom count and print top kernels")
    parser.add_argument("--tf32", action="store_true",
                        help="Enable TF32 tensor cores (set matmul precision to 'high')")
    args = parser.parse_args()

    if args.tf32:
        torch.set_float32_matmul_precision("high")
        print("TF32 tensor cores: ENABLED (matmul precision = high)")
    else:
        print("TF32 tensor cores: disabled (add --tf32 to enable)")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    if device.type != "cuda":
        print("WARNING: CUDA not available; benchmarking on CPU.", file=sys.stderr)

    print(f"Device: {device}")
    print(f"lmax: {args.lmax}, repeat: {args.repeat}, warmup: {args.warmup}")
    print()

    # ========== Part 1: direction_harmonics_all ==========
    print("=" * 70)
    print("  direction_harmonics_all: Optimized vs Baseline")
    print("=" * 70)

    n_check = torch.randn(1000, 3, device=device, dtype=torch.float32)
    n_check = n_check / n_check.norm(dim=-1, keepdim=True)
    ref = _direction_harmonics_all_baseline(n_check, args.lmax)
    opt = direction_harmonics_all(n_check, args.lmax)
    max_diff = max((r - o).abs().max().item() for r, o in zip(ref, opt))
    print(f"Correctness: max diff = {max_diff:.2e}")
    print()

    print(f"{'N (dirs)':<12} {'Optimized (ms)':<16} {'Baseline (ms)':<16} {'Speedup':<10}")
    print("-" * 56)

    for N in args.sizes:
        n = torch.randn(N, 3, device=device)
        n = n / n.norm(dim=-1, keepdim=True)
        mean_o, std_o = _time_fn(lambda: direction_harmonics_all(n, args.lmax), args.warmup, args.repeat, device)
        mean_b, std_b = _time_fn(lambda: _direction_harmonics_all_baseline(n, args.lmax), args.warmup, args.repeat, device)
        speedup = mean_b / mean_o if mean_o > 0 else 0.0
        print(f"{N:<12} {mean_o:.2f} ± {std_o:.2f}    {mean_b:.2f} ± {std_b:.2f}    {speedup:.2f}x")

    # ========== Part 2: Full layer ==========
    if args.full_layer:
        print()
        print("=" * 70)
        print("  PureCartesianICTDTransformerLayer: float32 vs float64 internal compute")
        print("=" * 70)

        from molecular_force_field.models.pure_cartesian_ictd_layers import PureCartesianICTDTransformerLayer

        base_cfg = dict(
            max_embed_radius=5.0,
            main_max_radius=5.0,
            main_number_of_basis=8,
            hidden_dim_conv=32,
            hidden_dim_sh=32,
            hidden_dim=64,
            channel_in2=32,
            embedding_dim=16,
            max_atomvalue=10,
            output_size=8,
            num_interaction=3,
            lmax=args.lmax,
            ictd_tp_path_policy="full",
        )

        layer_f64 = PureCartesianICTDTransformerLayer(
            **base_cfg, internal_compute_dtype=torch.float64
        ).to(device=device, dtype=torch.float32)

        layer_f32 = PureCartesianICTDTransformerLayer(
            **base_cfg, internal_compute_dtype=torch.float32
        ).to(device=device, dtype=torch.float32)
        layer_f32.load_state_dict(layer_f64.state_dict())

        print()
        params = sum(p.numel() for p in layer_f64.parameters())
        print(f"  Model params: {params:,}")
        print(f"  num_interaction: {base_cfg['num_interaction']}, lmax: {base_cfg['lmax']}, channels: {base_cfg['hidden_dim_conv']}")

        # Correctness: float32 vs float64 internal
        num_atoms = args.atoms[0]
        graph = make_dummy_graph(device, torch.float32, num_nodes=num_atoms, avg_degree=args.avg_degree)
        with torch.no_grad():
            out_f64 = layer_f64(*graph).clone()
            out_f32 = layer_f32(*graph).clone()
        diff = (out_f64 - out_f32).abs().max().item()
        rdiff = diff / (out_f64.abs().max().item() + 1e-12)
        print(f"  Correctness (f32 vs f64 internal): max abs diff = {diff:.2e}, max rel diff = {rdiff:.2e}")
        print()

        # Inference benchmark
        header_inf = f"{'Atoms':<8} {'Edges':<10} {'f32 intern (ms)':<18} {'f64 intern (ms)':<18} {'Speedup':<10}"
        print("  --- Inference (forward only) ---")
        print(f"  {header_inf}")
        print(f"  {'-' * len(header_inf)}")

        for num_atoms in args.atoms:
            graph = make_dummy_graph(device, torch.float32, num_nodes=num_atoms, avg_degree=args.avg_degree)
            num_edges = num_atoms * args.avg_degree

            mean_32, std_32 = _time_layer_forward(layer_f32, *graph, warmup=args.warmup, repeat=args.repeat, device=device)
            mean_64, std_64 = _time_layer_forward(layer_f64, *graph, warmup=args.warmup, repeat=args.repeat, device=device)

            speedup = mean_64 / mean_32 if mean_32 > 0 else 0.0
            print(f"  {num_atoms:<8} {num_edges:<10} {mean_32:.2f} ± {std_32:.2f}      {mean_64:.2f} ± {std_64:.2f}      {speedup:.2f}x")

        # Forward+backward benchmark
        if args.backward:
            print()
            print("  --- Training (forward + backward) ---")
            print(f"  {header_inf}")
            print(f"  {'-' * len(header_inf)}")

            for num_atoms in args.atoms:
                graph = make_dummy_graph(device, torch.float32, num_nodes=num_atoms, avg_degree=args.avg_degree)
                num_edges = num_atoms * args.avg_degree

                mean_32, std_32 = _time_layer_forward_backward(
                    layer_f32, *graph, warmup=args.warmup, repeat=args.repeat, device=device
                )
                mean_64, std_64 = _time_layer_forward_backward(
                    layer_f64, *graph, warmup=args.warmup, repeat=args.repeat, device=device
                )

                speedup = mean_64 / mean_32 if mean_32 > 0 else 0.0
                print(f"  {num_atoms:<8} {num_edges:<10} {mean_32:.2f} ± {std_32:.2f}      {mean_64:.2f} ± {std_64:.2f}      {speedup:.2f}x")

        # ========== Part 3: torch.compile ==========
        if device.type == "cuda":
            print()
            print("=" * 70)
            print("  torch.compile: f32 eager vs f32 compiled")
            print("=" * 70)
            try:
                layer_compiled = torch.compile(
                    copy.deepcopy(layer_f32), mode="reduce-overhead", dynamic=False
                )
                # Correctness
                graph_check = make_dummy_graph(device, torch.float32, num_nodes=args.atoms[0], avg_degree=args.avg_degree)
                with torch.no_grad():
                    # Warm up compile
                    for _ in range(3):
                        _ = layer_compiled(*graph_check)
                    out_compiled = layer_compiled(*graph_check).clone()
                    out_eager = layer_f32(*graph_check).clone()
                diff_c = (out_compiled - out_eager).abs().max().item()
                print(f"  Correctness (compiled vs eager): max diff = {diff_c:.2e}")
                print()

                header_c = f"{'Atoms':<8} {'Edges':<10} {'Compiled (ms)':<16} {'Eager (ms)':<16} {'Speedup':<10}"
                print("  --- Inference (forward only) ---")
                print(f"  {header_c}")
                print(f"  {'-' * len(header_c)}")

                for num_atoms in args.atoms:
                    graph = make_dummy_graph(device, torch.float32, num_nodes=num_atoms, avg_degree=args.avg_degree)
                    num_edges = num_atoms * args.avg_degree
                    mean_c, std_c = _time_layer_forward(
                        layer_compiled, *graph, warmup=max(args.warmup, 10), repeat=args.repeat, device=device
                    )
                    mean_e, std_e = _time_layer_forward(
                        layer_f32, *graph, warmup=args.warmup, repeat=args.repeat, device=device
                    )
                    speedup_c = mean_e / mean_c if mean_c > 0 else 0.0
                    print(f"  {num_atoms:<8} {num_edges:<10} {mean_c:.2f} ± {std_c:.2f}    {mean_e:.2f} ± {std_e:.2f}    {speedup_c:.2f}x")
            except Exception as e:
                print(f"  torch.compile failed: {e}")

        # ========== Part 4: CUDA profiling ==========
        if device.type == "cuda" and args.profile:
            print()
            print("=" * 70)
            print("  CUDA Profiling: top kernels in full layer forward")
            print("=" * 70)
            graph_prof = make_dummy_graph(device, torch.float32, num_nodes=args.atoms[-1], avg_degree=args.avg_degree)
            layer_f32.eval()
            with torch.no_grad():
                for _ in range(3):
                    _ = layer_f32(*graph_prof)
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                ) as prof:
                    for _ in range(5):
                        torch.cuda.synchronize()
                        _ = layer_f32(*graph_prof)
                        torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))

    print()
    print("Done.")


if __name__ == "__main__":
    main()
