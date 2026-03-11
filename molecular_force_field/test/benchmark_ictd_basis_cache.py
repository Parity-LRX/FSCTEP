#!/usr/bin/env python3
"""
Benchmark: 基缓存 (_harmonic_basis_cpu_f64) 是否加快首次构建与 forward。

- 无缓存时：同一 L 在 build_cg_tensor / build_harmonic_projectors 中会被多次计算。
- 有缓存后：每个 L 只算一次，后续用 lru_cache 返回。
"""
from __future__ import annotations

import argparse
import time

import torch

import molecular_force_field.models.ictd_irreps as ictd_irreps


def get_paths_for_lmax(lmax: int, path_policy: str = "max_rank_other", max_rank_other: int = 5):
    """与 HarmonicFullyConnectedTensorProduct 相同的 path 列表。"""
    paths = []
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            for l3 in range(abs(l1 - l2), min(lmax, l1 + l2) + 1):
                if (l1 + l2 + l3) % 2 != 0:
                    continue
                if path_policy == "max_rank_other" and max_rank_other is not None:
                    if min(l1, l2) > max_rank_other:
                        continue
                paths.append((l1, l2, l3))
    return paths


def bench_basis_gpu_vs_cpu(device: torch.device | None = None):
    """基已改为仅 CPU 算 + .to(device)，不再提供 GPU 算基；此处仅测 CPU+传输耗时。"""
    if not torch.cuda.is_available():
        print("  (无 CUDA，跳过)", flush=True)
        return None, None
    if device is None:
        device = torch.device("cuda")
    L = 4
    n_calls = 21
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_calls):
        ictd_irreps._harmonic_basis_cpu_f64(L).to(device=device, dtype=torch.float32)
    torch.cuda.synchronize()
    t_ms = (time.perf_counter() - start) * 1000
    print(f"  [CPU 算 + .to(GPU)] L={L}, {n_calls} 次: {t_ms:.2f} ms", flush=True)
    return t_ms, None


def bench_basis_direct():
    """直接测 _harmonic_basis_t：同一 L 重复调用时，缓存带来的加速。"""
    ictd_irreps._harmonic_basis_cpu_f64.cache_clear()
    L = 4
    n_calls = 40

    # 第一次调用：真正计算
    start = time.perf_counter()
    for _ in range(n_calls):
        ictd_irreps._harmonic_basis_t(L, dtype=torch.float64)
    t_first = (time.perf_counter() - start) * 1000
    # 后续调用：全部命中缓存（仅 .to()）
    start = time.perf_counter()
    for _ in range(n_calls):
        ictd_irreps._harmonic_basis_t(L, dtype=torch.float64)
    t_cached = (time.perf_counter() - start) * 1000
    # 第一次 40 次里只有第 1 次是真实计算，其余 39 次已命中
    # 所以「无缓存时 40 次」≈ 40 * t_first_per_call，这里用 t_first 表示「含 1 次计算 + 39 次缓存」的耗时
    # 若完全无缓存，40 次全算：估计约 40 * (t_first/40) 的 40 倍？不对。更简单：比较「1 次」vs「第 2 次起」
    t_one_compute = t_first / n_calls  # 平均每次（实际第一次贵，后面便宜）
    t_one_cached = t_cached / n_calls
    print(f"  L={L}, 调用 {n_calls} 次: 首轮 {t_first:.2f} ms (含 1 次计算), 纯缓存轮 {t_cached:.2f} ms")
    if t_one_cached > 0:
        print(f"  => 单次取缓存约 {t_one_compute/t_one_cached:.0f}x 快于「平均含 1 次计算」")
    return t_first, t_cached


def bench_basis_cache():
    ictd_irreps.build_cg_tensor.cache_clear()
    ictd_irreps.build_harmonic_projectors.cache_clear()
    ictd_irreps._harmonic_basis_cpu_f64.cache_clear()

    lmax = 5
    paths = get_paths_for_lmax(lmax)
    n_paths = len(paths)

    # 1) 冷启动：清空所有缓存，计时「为所有 path 构建 CG」
    start = time.perf_counter()
    for l1, l2, l3 in paths:
        ictd_irreps.build_cg_tensor(l1, l2, l3)
    t_cold = (time.perf_counter() - start) * 1000
    print(f"  [冷启动] 构建 {n_paths} 个 CG 张量（基按 L 只算一次）: {t_cold:.2f} ms")

    # 2) 仅清空 build_cg_tensor 缓存，保留基缓存，再次构建所有 CG
    ictd_irreps.build_cg_tensor.cache_clear()
    start = time.perf_counter()
    for l1, l2, l3 in paths:
        ictd_irreps.build_cg_tensor(l1, l2, l3)
    t_warm_basis = (time.perf_counter() - start) * 1000
    print(f"  [基已缓存] 再次构建 {n_paths} 个 CG 张量:                 {t_warm_basis:.2f} ms")

    if t_warm_basis > 0:
        speedup = t_cold / t_warm_basis
        print(f"  => CG 构建阶段加速比: {speedup:.2f}x (构建主要耗时在卷积+投影，基只占一部分)")
    return t_cold, t_warm_basis


def bench_tp_first_forward(device: torch.device | None = None):
    """计时：创建 TP + 首次 forward（会触发 CG/projector 构建，基缓存生效）。"""
    ictd_irreps.build_cg_tensor.cache_clear()
    ictd_irreps.build_harmonic_projectors.cache_clear()
    ictd_irreps._harmonic_basis_cpu_f64.cache_clear()

    batch = 32
    mul = 8
    lmax = 5
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    tp = ictd_irreps.HarmonicFullyConnectedTensorProduct(
        mul_in1=mul, mul_in2=mul, mul_out=mul, lmax=lmax,
        internal_weights=True, path_policy="max_rank_other", max_rank_other=5,
    ).to(device=device, dtype=dtype)

    x1 = {l: torch.randn(batch, mul, 2 * l + 1, device=device, dtype=dtype) for l in range(lmax + 1)}
    x2 = {l: torch.randn(batch, mul, 2 * l + 1, device=device, dtype=dtype) for l in range(lmax + 1)}

    # 首次 forward（含 CG/proj 构建）
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    _ = tp(x1, x2)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_first = (time.perf_counter() - start) * 1000
    print(f"  [首次 forward] device={device}, batch={batch}, lmax={lmax}: {t_first:.2f} ms")

    # 后续 forward（仅算力，无构建）
    n_warm = 50
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    for _ in range(n_warm):
        _ = tp(x1, x2)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_warm = (time.perf_counter() - start) / n_warm * 1000
    print(f"  [后续 forward] 平均 {n_warm} 次: {t_warm:.3f} ms/iter")
    return t_first, t_warm


def main():
    parser = argparse.ArgumentParser(description="ICTD 基缓存 / GPU 算基 benchmark")
    parser.add_argument("--device", type=str, default=None, help="设备，如 cuda, cuda:0, cuda:1（默认: cuda 或 cpu）")
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[benchmark_ictd_basis_cache] 基缓存效果", flush=True)
    print(f"  device = {device}", flush=True)
    print("", flush=True)
    print("1) 基：CPU 算 + .to(device)（无 GPU 算基）", flush=True)
    bench_basis_gpu_vs_cpu(device)
    print("", flush=True)
    print("2) 直接测基：同一 L 重复调用（有缓存后仅 .to()，无缓存则每次 SVD+eigh）", flush=True)
    bench_basis_direct()
    print("", flush=True)
    print("3) CG 构建：69 个 path 共享 L=0..5，基按 L 只算一次", flush=True)
    bench_basis_cache()
    print("", flush=True)
    print("4) TP 首次 forward（含 CG/投影构建）", flush=True)
    bench_tp_first_forward(device)
    print("", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
