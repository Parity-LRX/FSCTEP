#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import time

import torch

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from molecular_force_field.models.pure_cartesian import (
    PureCartesianTensorProductO3Sparse,
    total_dim_o3,
)
from molecular_force_field.models.triton_sparse_tp import sparse_tp_fused


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _median(times):
    s = sorted(times)
    return s[len(s) // 2]


def _bench(fn, device, warmup=5, repeat=15):
    for _ in range(warmup):
        fn()
    _sync(device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(repeat):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000)

    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    return _median(times), peak_mb


def make_tp_inputs(E, C1, C2, Cout, Lmax, device, dtype):
    dim1 = total_dim_o3(C1, Lmax)
    dim2 = total_dim_o3(C2, Lmax)
    t1 = torch.randn(E, dim1, device=device, dtype=dtype)
    t2 = torch.randn(E, dim2, device=device, dtype=dtype)
    t1[:, C1 * 13:] = 0
    t2[:, C2 * 13:] = 0
    w_numel = 6 * Cout * C1 * C2
    w = torch.randn(E, w_numel, device=device, dtype=dtype)
    return t1, t2, w


def make_dummy_graph(num_nodes, avg_degree, device, dtype):
    torch.manual_seed(42)
    E = num_nodes * avg_degree
    pos = torch.randn(num_nodes, 3, device=device, dtype=dtype) * 2.0
    A = torch.randint(1, 6, (num_nodes,), device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    edge_dst = torch.randint(0, num_nodes, (E,), device=device)
    edge_src = torch.randint(0, num_nodes, (E,), device=device)
    edge_shifts = torch.zeros(E, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(num_nodes, -1, -1)
    return pos, A, batch, edge_src, edge_dst, edge_shifts, cell


def bench_tp(args):
    device = torch.device(args.device)
    dtype = torch.float32
    E = args.atoms * args.avg_degree
    C1, C2, Cout, Lmax = 8, 8, 64, 2

    print(f"\n{'=' * 70}")
    print(f"TP Benchmark: E={E}, C1={C1}, C2={C2}, Cout={Cout}, Lmax={Lmax}")
    print(f"device={device}, dtype={dtype}")
    print(f"{'=' * 70}")

    t1, t2, w = make_tp_inputs(E, C1, C2, Cout, Lmax, device, dtype)

    tp_ref = PureCartesianTensorProductO3Sparse(
        C1,
        C2,
        Cout,
        Lmax,
        max_rank_other=1,
        allow_epsilon=False,
        k_policy="k0",
        share_parity_weights=True,
        assume_pseudo_zero=True,
        internal_weights=False,
    ).to(device=device, dtype=dtype)
    tp_ref.use_triton_fast_path = False

    print("\n--- 0) 数值正确性 ---")
    with torch.no_grad():
        ref = tp_ref(t1, t2, weights=w)
        out = sparse_tp_fused(t1, t2, w, C1, C2, Cout)
        fwd_diff = (ref - out).abs().max().item()
    fwd_rtol = fwd_diff / (ref.abs().max().item() + 1e-8)
    print(f"  Forward max diff: {fwd_diff:.2e} (rtol={fwd_rtol:.2e})  {'PASS' if fwd_rtol < 1e-4 else 'FAIL'}")

    t1_g = t1.clone().requires_grad_(True)
    t2_g = t2.clone().requires_grad_(True)
    w_g = w.clone().requires_grad_(True)
    sparse_tp_fused(t1_g, t2_g, w_g, C1, C2, Cout).sum().backward()

    t1_r = t1.clone().requires_grad_(True)
    t2_r = t2.clone().requires_grad_(True)
    w_r = w.clone().requires_grad_(True)
    tp_ref(t1_r, t2_r, weights=w_r).sum().backward()

    g1 = (t1_g.grad - t1_r.grad).abs().max().item()
    g2 = (t2_g.grad - t2_r.grad).abs().max().item()
    gw = (w_g.grad - w_r.grad).abs().max().item()
    g_ref_max = max(t1_r.grad.abs().max().item(), t2_r.grad.abs().max().item(), w_r.grad.abs().max().item(), 1e-8)
    grad_rtol = max(g1, g2, gw) / g_ref_max
    print(f"  Grad diffs: t1={g1:.2e} t2={g2:.2e} w={gw:.2e} (rtol={grad_rtol:.2e})  {'PASS' if grad_rtol < 1e-3 else 'FAIL'}")

    t1_2 = t1.clone().requires_grad_(True)
    w_2 = w.clone().requires_grad_(True)
    out_2 = sparse_tp_fused(t1_2, t2, w_2, C1, C2, Cout)
    g1_2 = torch.autograd.grad(out_2.sum(), t1_2, create_graph=True)[0]
    g1_2.sum().backward()
    print(f"  2nd-order grad: w_2.grad exists={w_2.grad is not None}  PASS")

    print("\n--- 1) Forward only ---")

    def fwd_ref():
        with torch.no_grad():
            tp_ref(t1, t2, weights=w)

    def fwd_fused():
        with torch.no_grad():
            sparse_tp_fused(t1, t2, w, C1, C2, Cout)

    t_ref, _ = _bench(fwd_ref, device, warmup=args.warmup, repeat=args.repeat)
    t_fused, _ = _bench(fwd_fused, device, warmup=args.warmup, repeat=args.repeat)
    print(f"  原始 einsum:   {t_ref:.2f} ms")
    print(f"  Fast path:     {t_fused:.2f} ms")
    print(f"  加速比: {t_ref / t_fused:.2f}x")

    print("\n--- 2) Forward + Backward ---")

    def fwd_bwd_ref():
        t1_ = t1.clone().requires_grad_(True)
        t2_ = t2.clone().requires_grad_(True)
        w_ = w.clone().requires_grad_(True)
        tp_ref(t1_, t2_, weights=w_).sum().backward()

    def fwd_bwd_fused():
        t1_ = t1.clone().requires_grad_(True)
        t2_ = t2.clone().requires_grad_(True)
        w_ = w.clone().requires_grad_(True)
        sparse_tp_fused(t1_, t2_, w_, C1, C2, Cout).sum().backward()

    t_ref, _ = _bench(fwd_bwd_ref, device, warmup=args.warmup, repeat=args.repeat)
    t_fused, _ = _bench(fwd_bwd_fused, device, warmup=args.warmup, repeat=args.repeat)
    print(f"  原始 einsum:   {t_ref:.2f} ms")
    print(f"  Fast path:     {t_fused:.2f} ms")
    print(f"  加速比: {t_ref / t_fused:.2f}x")

    print("\n--- 3) Forward + Backward (create_graph=True, 训练场景) ---")

    def fwd_bwd_2nd_ref():
        t1_ = t1.clone().requires_grad_(True)
        w_ = w.clone().requires_grad_(True)
        out = tp_ref(t1_, t2, weights=w_)
        g = torch.autograd.grad(out.sum(), t1_, create_graph=True)[0]
        g.sum().backward()

    def fwd_bwd_2nd_fused():
        t1_ = t1.clone().requires_grad_(True)
        w_ = w.clone().requires_grad_(True)
        out = sparse_tp_fused(t1_, t2, w_, C1, C2, Cout)
        g = torch.autograd.grad(out.sum(), t1_, create_graph=True)[0]
        g.sum().backward()

    t_ref, _ = _bench(fwd_bwd_2nd_ref, device, warmup=args.warmup, repeat=args.repeat)
    t_fused, _ = _bench(fwd_bwd_2nd_fused, device, warmup=args.warmup, repeat=args.repeat)
    print(f"  原始 einsum:   {t_ref:.2f} ms")
    print(f"  Fast path:     {t_fused:.2f} ms")
    print(f"  加速比: {t_ref / t_fused:.2f}x")

    print(f"\n--- 4) TP配置 C1=64, C2=1, Cout=64 ---")
    C1b, C2b = 64, 1
    t1b, t2b, wb = make_tp_inputs(E, C1b, C2b, Cout, Lmax, device, dtype)

    tp_ref2 = PureCartesianTensorProductO3Sparse(
        C1b,
        C2b,
        Cout,
        Lmax,
        max_rank_other=1,
        allow_epsilon=False,
        k_policy="k0",
        share_parity_weights=True,
        assume_pseudo_zero=True,
        internal_weights=False,
    ).to(device=device, dtype=dtype)
    tp_ref2.use_triton_fast_path = False

    with torch.no_grad():
        ref2 = tp_ref2(t1b, t2b, weights=wb)
        out2 = sparse_tp_fused(t1b, t2b, wb, C1b, C2b, Cout)
        d2 = (ref2 - out2).abs().max().item()
    d2_rtol = d2 / (ref2.abs().max().item() + 1e-8)
    print(f"  Forward diff: {d2:.2e} (rtol={d2_rtol:.2e})  {'PASS' if d2_rtol < 1e-4 else 'FAIL'}")

    def fwd_ref2():
        with torch.no_grad():
            tp_ref2(t1b, t2b, weights=wb)

    def fwd_fused2():
        with torch.no_grad():
            sparse_tp_fused(t1b, t2b, wb, C1b, C2b, Cout)

    t_r2, _ = _bench(fwd_ref2, device, warmup=args.warmup, repeat=args.repeat)
    t_f2, _ = _bench(fwd_fused2, device, warmup=args.warmup, repeat=args.repeat)
    print(f"  Forward 原始:  {t_r2:.2f} ms")
    print(f"  Forward fast:  {t_f2:.2f} ms")
    print(f"  加速比: {t_r2 / t_f2:.2f}x")

    def fwd_bwd_ref2():
        t1_ = t1b.clone().requires_grad_(True)
        t2_ = t2b.clone().requires_grad_(True)
        w_ = wb.clone().requires_grad_(True)
        tp_ref2(t1_, t2_, weights=w_).sum().backward()

    def fwd_bwd_fused2():
        t1_ = t1b.clone().requires_grad_(True)
        t2_ = t2b.clone().requires_grad_(True)
        w_ = wb.clone().requires_grad_(True)
        sparse_tp_fused(t1_, t2_, w_, C1b, C2b, Cout).sum().backward()

    t_r2b, _ = _bench(fwd_bwd_ref2, device, warmup=args.warmup, repeat=args.repeat)
    t_f2b, _ = _bench(fwd_bwd_fused2, device, warmup=args.warmup, repeat=args.repeat)
    print(f"  Fwd+Bwd 原始:  {t_r2b:.2f} ms")
    print(f"  Fwd+Bwd fast:  {t_f2b:.2f} ms")
    print(f"  加速比: {t_r2b / t_f2b:.2f}x")


def bench_full_layer(args):
    from molecular_force_field.utils.config import ModelConfig
    from molecular_force_field.models.pure_cartesian_sparse_layers import PureCartesianSparseTransformerLayer

    device = torch.device(args.device)
    dtype = torch.float32
    N = args.atoms

    print(f"\n{'=' * 70}")
    print(f"Full Layer Benchmark: atoms={N}, avg_degree={args.avg_degree}")
    print(f"device={device}, dtype={dtype}")
    print(f"{'=' * 70}")

    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = make_dummy_graph(N, args.avg_degree, device, dtype)

    cfg = ModelConfig()
    layer = PureCartesianSparseTransformerLayer(
        max_embed_radius=cfg.max_radius,
        main_max_radius=cfg.max_radius_main,
        main_number_of_basis=cfg.number_of_basis_main,
        hidden_dim_conv=cfg.channel_in,
        hidden_dim_sh=cfg.get_hidden_dim_sh(),
        hidden_dim=cfg.emb_number_main_2,
        channel_in2=cfg.channel_in2,
        embedding_dim=cfg.embedding_dim,
        max_atomvalue=cfg.max_atomvalue,
        output_size=cfg.output_size,
        embed_size=cfg.embed_size,
        main_hidden_sizes3=cfg.main_hidden_sizes3,
        num_layers=cfg.num_layers,
        num_interaction=2,
        function_type_main=cfg.function_type,
        lmax=cfg.lmax,
        max_rank_other=1,
        k_policy="k0",
        device=device,
    ).to(device=device, dtype=dtype)

    def run_layer():
        pos_r = pos.clone().requires_grad_(True)
        out = layer(pos_r, A, batch, edge_src, edge_dst, edge_shifts, cell)
        out.sum().backward()

    t_layer, peak_mb = _bench(run_layer, device, warmup=args.warmup, repeat=args.repeat)
    print(f"\n  整层 fwd+bwd median: {t_layer:.1f} ms")
    if peak_mb > 0:
        print(f"  Peak GPU memory: {peak_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Triton sparse TP benchmark")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--atoms", type=int, default=256)
    parser.add_argument("--avg-degree", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--full-layer", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.device == "cpu":
        print("NOTE: CPU 模式下 Triton 不可用，fused 会走组合式 fallback。")

    torch.manual_seed(42)
    bench_tp(args)
    if args.full_layer:
        bench_full_layer(args)
    print("\n完成。")


if __name__ == "__main__":
    main()
