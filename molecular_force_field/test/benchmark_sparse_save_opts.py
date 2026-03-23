#!/usr/bin/env python3
"""
pure-cartesian-sparse-save 相关优化方向的性能对比（不修改库源码）。

对比项：
0) 整体：pure-cartesian-sparse vs pure-cartesian-sparse-save（同图 forward+backward）
1) 微基准：torch.argsort(edge_dst) 单独耗时，及对整层 forward 的占比估计
2) 微基准：当前 avg 邻居数写法 new_ones().sum() vs 标量 E/N
3) 整层 forward：基类（含 argsort）vs 测试内子类（跳过 argsort，边列表需已按 edge_dst 有序）
4) 可选：torch.compile 整层 forward / forward+backward
5) 首层边特征构造：save 首层（大零块 + merge）vs sparse 首层（Ai⊗e^L + merge），不含 TP

用法:
  python -m molecular_force_field.test.benchmark_sparse_save_opts
  python -m molecular_force_field.test.benchmark_sparse_save_opts --device cuda --atoms 512
  python -m molecular_force_field.test.benchmark_sparse_save_opts --compile
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from typing import Callable

import torch
import torch.nn as nn

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from molecular_force_field.models.long_range import apply_long_range_modules
from molecular_force_field.models.pure_cartesian import (
    edge_rank_powers,
    merge_by_rank_o3,
    split_by_rank_o3,
)
from molecular_force_field.models.pure_cartesian_sparse_layers import (
    PureCartesianSparseTransformerLayer,
    _apply_channel_adapter_cartesian,
)
from molecular_force_field.models.pure_cartesian_sparse_layers_save import (
    PureCartesianSparseTransformerLayerSave,
)
from molecular_force_field.utils.config import ModelConfig
from molecular_force_field.utils.fidelity import apply_delta_energy_heads
from molecular_force_field.utils.scatter import scatter


def _median_ms(times: list[float]) -> float:
    s = sorted(times)
    return float(s[len(s) // 2])


def make_dummy_graph(
    device: torch.device,
    dtype: torch.dtype,
    num_nodes: int,
    avg_degree: int = 24,
    seed: int = 42,
):
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


def sort_edges_by_dst(edge_src, edge_dst, edge_shifts):
    sort_idx = torch.argsort(edge_dst)
    return edge_src[sort_idx], edge_dst[sort_idx], edge_shifts[sort_idx]


def build_sparse_layer(device: torch.device, dtype: torch.dtype, config: ModelConfig) -> nn.Module:
    return PureCartesianSparseTransformerLayer(
        max_embed_radius=config.max_radius,
        main_max_radius=config.max_radius_main,
        main_number_of_basis=config.number_of_basis_main,
        hidden_dim_conv=config.channel_in,
        hidden_dim_sh=config.get_hidden_dim_sh(),
        hidden_dim=config.emb_number_main_2,
        channel_in2=config.channel_in2,
        embedding_dim=config.embedding_dim,
        max_atomvalue=config.max_atomvalue,
        output_size=config.output_size,
        embed_size=config.embed_size,
        main_hidden_sizes3=config.main_hidden_sizes3,
        num_layers=config.num_layers,
        num_interaction=2,
        function_type_main=config.function_type,
        lmax=config.lmax,
        max_rank_other=1,
        k_policy="k0",
        device=device,
    ).to(device=device, dtype=dtype)


def build_save_layer(device: torch.device, dtype: torch.dtype, config: ModelConfig) -> nn.Module:
    return PureCartesianSparseTransformerLayerSave(
        max_embed_radius=config.max_radius,
        main_max_radius=config.max_radius_main,
        main_number_of_basis=config.number_of_basis_main,
        hidden_dim_conv=config.channel_in,
        hidden_dim_sh=config.get_hidden_dim_sh(),
        hidden_dim=config.emb_number_main_2,
        channel_in2=config.channel_in2,
        embedding_dim=config.embedding_dim,
        max_atomvalue=config.max_atomvalue,
        output_size=config.output_size,
        embed_size=config.embed_size,
        main_hidden_sizes3=config.main_hidden_sizes3,
        num_layers=config.num_layers,
        num_interaction=2,
        function_type_main=config.function_type,
        lmax=config.lmax,
        max_rank_other=1,
        k_policy="k0",
        device=device,
    ).to(device=device, dtype=dtype)


class PureCartesianSparseTransformerLayerSaveNoEdgeSort(PureCartesianSparseTransformerLayerSave):
    """与库中实现相同，但不在 forward 内对边做 argsort（用于估计去掉排序的收益）。"""

    def forward(
        self,
        pos,
        A,
        batch,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        *,
        precomputed_edge_vec=None,
        external_tensor: torch.Tensor | None = None,
        fidelity_ids: torch.Tensor | None = None,
        return_physical_tensors: bool = False,
        return_reciprocal_source: bool = False,
        **_unused,
    ):
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        features = []
        f_prev = self.e3_conv_layers[0](
            pos,
            A,
            batch,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
            external_tensor=external_tensor,
            fidelity_ids=fidelity_ids,
        )
        features.append(f_prev)
        for conv in self.e3_conv_layers[1:]:
            f_prev = conv(f_prev, pos, batch, edge_src, edge_dst, edge_shifts, cell)
            features.append(f_prev)

        blocks_list = [split_by_rank_o3(f, self.channels, self.Lmax) for f in features]
        combined_blocks = {}
        for s in (0, 1):
            for L in range(self.Lmax + 1):
                combined_blocks[(s, L)] = torch.cat([blocks[(s, L)] for blocks in blocks_list], dim=1)
        f_combine = merge_by_rank_o3(combined_blocks, self.combined_channels, self.Lmax)

        physical_out = None
        if return_physical_tensors:
            if self.physical_tensor_heads is None or self._physical_tensor_specs is None:
                raise ValueError("return_physical_tensors=True but physical_tensor_outputs was not set in __init__")
            physical_out = {}
            true_blocks = {(0, L): combined_blocks[(0, L)] for L in range(self.Lmax + 1)}
            num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
            counts = None
            for name, per_l in self.physical_tensor_heads.items():
                spec = self._physical_tensor_specs[name]
                out_blocks: dict[int, torch.Tensor] = {}
                for l in spec["ls"]:
                    y_l = _apply_channel_adapter_cartesian(true_blocks[(0, l)], per_l[str(l)])
                    if l == 0:
                        y_l = y_l.unsqueeze(-1)
                    if spec["reduce"] == "none":
                        out_blocks[l] = y_l
                    else:
                        yg = scatter(y_l, batch, dim=0, dim_size=num_graphs, reduce="sum")
                        if spec["reduce"] == "mean":
                            if counts is None:
                                counts = scatter(
                                    torch.ones_like(batch, dtype=yg.dtype),
                                    batch,
                                    dim=0,
                                    dim_size=num_graphs,
                                    reduce="sum",
                                ).clamp(min=1.0)
                            view_shape = (counts.shape[0],) + (1,) * (yg.dim() - 1)
                            yg = yg / counts.view(view_shape)
                        out_blocks[l] = yg
                physical_out[name] = out_blocks

        f_prod3 = self.product_3(f_combine)
        invs = [self.product_5_o3(f, f) for f in features]
        inv3 = f_prod3 * f_prod3
        f_prod5 = torch.cat(invs + [inv3], dim=-1)
        f_prod5, long_range_energy, reciprocal_source, defer_long_range_to_runtime = apply_long_range_modules(
            self,
            f_prod5,
            pos,
            batch,
            cell,
            edge_src=edge_src,
            edge_dst=edge_dst,
            return_reciprocal_source=return_reciprocal_source,
        )

        product_proj = self.proj_total(f_prod5)
        e_out = self.weighted_sum(product_proj)
        atom_energies = e_out.sum(dim=-1, keepdim=True)
        if self.multi_fidelity_mode == "delta-baseline":
            atom_energies = apply_delta_energy_heads(
                atom_energies,
                f_prod5,
                batch,
                fidelity_ids,
                self.delta_proj_total,
                self.delta_weighted_sum,
            )
        if long_range_energy is not None and not defer_long_range_to_runtime:
            atom_energies = atom_energies + long_range_energy
        if reciprocal_source is None and return_reciprocal_source:
            reciprocal_source = atom_energies.new_empty((atom_energies.size(0), 0))
        if return_physical_tensors and return_reciprocal_source:
            assert physical_out is not None
            return atom_energies, physical_out, reciprocal_source
        if return_physical_tensors:
            assert physical_out is not None
            return atom_energies, physical_out
        if return_reciprocal_source:
            return atom_energies, reciprocal_source
        return atom_energies


def bench_argsort_only(device: torch.device, edge_dst: torch.Tensor, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        torch.argsort(edge_dst)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.argsort(edge_dst)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return _median_ms(times)


def bench_avg_neighbor_norm(
    device: torch.device,
    dtype: torch.dtype,
    E: int,
    N: int,
    D: int,
    warmup: int,
    repeat: int,
) -> tuple[float, float]:
    """返回 (current_style_ms, scalar_style_ms) 的中位数。"""
    edge_features = torch.randn(E, D, device=device, dtype=dtype)
    out = torch.randn(N, D, device=device, dtype=dtype)

    def run_current():
        num_edges_t = edge_features.new_ones(edge_features.shape[0]).sum()
        num_nodes_t = out.new_ones(out.shape[0]).sum().clamp_min(1.0)
        avg = (num_edges_t / num_nodes_t).clamp_min(1e-8)
        return out / avg

    def run_scalar():
        avg = max(float(E) / float(N), 1e-8)
        return out * (1.0 / avg)

    for _ in range(warmup):
        run_current()
        run_scalar()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t_cur: list[float] = []
    t_sca: list[float] = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = run_current()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_cur.append((time.perf_counter() - t0) * 1000.0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = run_scalar()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_sca.append((time.perf_counter() - t0) * 1000.0)

    if device.type == "cuda":
        _ = x + y
    return _median_ms(t_cur), _median_ms(t_sca)


def bench_first_layer_block_prep(
    device: torch.device,
    dtype: torch.dtype,
    Ai: torch.Tensor,
    edge_src: torch.Tensor,
    edge_vec: torch.Tensor,
    output_size: int,
    Lmax: int,
    warmup: int,
    repeat: int,
) -> tuple[float, float]:
    """
    仅统计 merge 前的块构造 + merge_by_rank_o3(x_side) + merge(e_side)（save 与 sparse 首层公共几何侧相同）。
    save: x1 侧为大零块；sparse: x1 侧为 Ai[src] * e^L。
    """
    e_list = edge_rank_powers(edge_vec, Lmax, normalize=True)
    E = edge_src.shape[0]
    batch_shape = (E,)

    def save_x1_and_merge():
        x1_blocks = {(0, 0): Ai[edge_src], (1, 0): torch.zeros_like(Ai[edge_src])}
        for L in range(1, Lmax + 1):
            z = torch.zeros(E, output_size, *([3] * L), device=device, dtype=dtype)
            x1_blocks[(0, L)] = z
            x1_blocks[(1, L)] = torch.zeros_like(z)
        x1 = merge_by_rank_o3(x1_blocks, output_size, Lmax)
        e_blocks = {(0, 0): e_list[0].view(-1, 1), (1, 0): torch.zeros_like(e_list[0].view(-1, 1))}
        for L in range(1, Lmax + 1):
            base = e_list[L].view(-1, 1, *([3] * L))
            e_blocks[(0, L)] = base
            e_blocks[(1, L)] = torch.zeros_like(base)
        e_flat = merge_by_rank_o3(e_blocks, 1, Lmax)
        return x1, e_flat

    def sparse_x1_and_merge():
        f1_blocks = {(0, 0): Ai[edge_src], (1, 0): torch.zeros_like(Ai[edge_src])}
        for L in range(1, Lmax + 1):
            base = Ai[edge_src].view(*batch_shape, output_size, *([1] * L)) * e_list[L].view(
                *batch_shape, 1, *([3] * L)
            )
            f1_blocks[(0, L)] = base
            f1_blocks[(1, L)] = torch.zeros_like(base)
        f_in = merge_by_rank_o3(f1_blocks, output_size, Lmax)
        e_blocks = {(0, 0): e_list[0].view(-1, 1), (1, 0): torch.zeros_like(e_list[0].view(-1, 1))}
        for L in range(1, Lmax + 1):
            base = e_list[L].view(-1, 1, *([3] * L))
            e_blocks[(0, L)] = base
            e_blocks[(1, L)] = torch.zeros_like(base)
        e_flat = merge_by_rank_o3(e_blocks, 1, Lmax)
        return f_in, e_flat

    for _ in range(warmup):
        save_x1_and_merge()
        sparse_x1_and_merge()
    if device.type == "cuda":
        torch.cuda.synchronize()

    ts: list[float] = []
    tp: list[float] = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        save_x1_and_merge()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        sparse_x1_and_merge()
        if device.type == "cuda":
            torch.cuda.synchronize()
        tp.append((time.perf_counter() - t0) * 1000.0)

    return _median_ms(ts), _median_ms(tp)


def measure_forward_ms(
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    repeat: int,
    stable_tail: int,
) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    tail = sorted(times[-stable_tail:])
    return float(tail[len(tail) // 2])


def measure_forward_backward_ms(
    layer: nn.Module,
    graph: tuple,
    device: torch.device,
    warmup: int,
    repeat: int,
    stable_tail: int,
) -> float:
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
    pos = pos.detach().clone().requires_grad_(True)
    graph2 = (pos, A, batch, edge_src, edge_dst, edge_shifts, cell)

    layer.train()
    with torch.enable_grad():
        for _ in range(warmup):
            layer.zero_grad(set_to_none=True)
            out = layer(*graph2)
            loss = out.sum()
            loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()

        times: list[float] = []
        for _ in range(repeat):
            layer.zero_grad(set_to_none=True)
            if pos.grad is not None:
                pos.grad = None
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = layer(*graph2)
            loss = out.sum()
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    tail = sorted(times[-stable_tail:])
    return float(tail[len(tail) // 2])


def main():
    p = argparse.ArgumentParser(description="pure-cartesian-sparse-save 优化方向性能对比")
    p.add_argument("--device", type=str, default=None, help="cuda | cpu（默认自动）")
    p.add_argument("--atoms", type=int, default=256)
    p.add_argument("--avg-degree", type=int, default=24)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=40)
    p.add_argument("--stable-tail", type=int, default=15)
    p.add_argument("--compile", action="store_true", help="额外测 torch.compile 整层 forward")
    p.add_argument("--compile-full", action="store_true", help="torch.compile forward+backward（较慢）")
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        try:
            torch.zeros(1, device=device)
        except Exception as e:
            print(f"CUDA 不可用（{e}），改用 CPU。", flush=True)
            device = torch.device("cpu")

    dtype = torch.float32
    config = ModelConfig(dtype=dtype)

    print(f"device={device}, dtype={dtype}, atoms={args.atoms}, avg_degree={args.avg_degree}", flush=True)

    graph = make_dummy_graph(device, dtype, args.atoms, args.avg_degree)
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
    E = int(edge_dst.numel())
    N = int(pos.size(0))
    edge_src_s, edge_dst_s, edge_shifts_s = sort_edges_by_dst(edge_src, edge_dst, edge_shifts)

    # --- 0) sparse vs sparse-save：整层 forward+backward（同图、同超参）---
    layer_sparse = build_sparse_layer(device, dtype, config)
    ms_fb_sparse = measure_forward_backward_ms(
        layer_sparse, graph, device, args.warmup, args.repeat, args.stable_tail
    )
    del layer_sparse
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    layer_save_fb = build_save_layer(device, dtype, config)
    ms_fb_save = measure_forward_backward_ms(
        layer_save_fb, graph, device, args.warmup, args.repeat, args.stable_tail
    )
    del layer_save_fb
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print("\n=== 0) 整体速度：pure-cartesian-sparse vs pure-cartesian-sparse-save ===", flush=True)
    print(f"  forward+backward median: sparse={ms_fb_sparse:.3f} ms, sparse-save={ms_fb_save:.3f} ms", flush=True)
    if ms_fb_sparse > 0 and ms_fb_save > 0:
        print(
            f"  比值 sparse-save / sparse = {ms_fb_save / ms_fb_sparse:.3f}x "
            f"（>1 表示 save 更慢）| sparse / sparse-save = {ms_fb_sparse / ms_fb_save:.3f}x",
            flush=True,
        )

    # --- 1) argsort 微基准 ---
    t_sort = bench_argsort_only(device, edge_dst, args.warmup, args.repeat)
    print("\n=== 1) torch.argsort(edge_dst) 单独耗时 ===", flush=True)
    print(f"  E={E}  median: {t_sort:.4f} ms", flush=True)

    # --- 2) avg neighbor 归一化微基准 ---
    D = 256
    t_cur, t_sca = bench_avg_neighbor_norm(device, dtype, E, N, D, max(20, args.warmup), max(50, args.repeat))
    print("\n=== 2) scatter 前除法：new_ones().sum() vs 标量 E/N ===", flush=True)
    print(f"  当前写法 median: {t_cur:.5f} ms", flush=True)
    print(f"  标量写法 median: {t_sca:.5f} ms", flush=True)
    if t_cur > 0:
        print(f"  标量相对加速: {t_cur / max(t_sca, 1e-9):.2f}x", flush=True)

    # --- 3) 首层块构造（无 TP）---
    torch.manual_seed(0)
    Ai = torch.randn(N, config.output_size, device=device, dtype=dtype)
    edge_vec = torch.randn(E, 3, device=device, dtype=dtype)
    t_save_blk, t_sparse_blk = bench_first_layer_block_prep(
        device, dtype, Ai, edge_src, edge_vec, config.output_size, config.lmax, args.warmup, args.repeat
    )
    print("\n=== 3) 首层边侧块构造 + merge（不含张量积）===", flush=True)
    print(f"  save 首层（零填充 x1）median: {t_save_blk:.3f} ms", flush=True)
    print(f"  sparse 首层（Ai⊗e^L）median: {t_sparse_blk:.3f} ms", flush=True)
    if t_save_blk > 0:
        print(
            f"  耗时比 sparse/save: {t_sparse_blk / max(t_save_blk, 1e-9):.2f}x（>1 表示 sparse 的块构造更慢）",
            flush=True,
        )

    # --- 4) 整层 forward：基类 vs 无 argsort（边已排序）---
    layer_base = build_save_layer(device, dtype, config)
    layer_nosort = PureCartesianSparseTransformerLayerSaveNoEdgeSort(
        max_embed_radius=config.max_radius,
        main_max_radius=config.max_radius_main,
        main_number_of_basis=config.number_of_basis_main,
        hidden_dim_conv=config.channel_in,
        hidden_dim_sh=config.get_hidden_dim_sh(),
        hidden_dim=config.emb_number_main_2,
        channel_in2=config.channel_in2,
        embedding_dim=config.embedding_dim,
        max_atomvalue=config.max_atomvalue,
        output_size=config.output_size,
        embed_size=config.embed_size,
        main_hidden_sizes3=config.main_hidden_sizes3,
        num_layers=config.num_layers,
        num_interaction=2,
        function_type_main=config.function_type,
        lmax=config.lmax,
        max_rank_other=1,
        k_policy="k0",
        device=device,
    ).to(device=device, dtype=dtype)
    layer_base.load_state_dict(layer_nosort.state_dict())

    layer_base.eval()
    layer_nosort.eval()

    def fwd_base():
        with torch.no_grad():
            return layer_base(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)

    def fwd_nosort_sorted_edges():
        with torch.no_grad():
            return layer_nosort(pos, A, batch, edge_src_s, edge_dst_s, edge_shifts_s, cell)

    ms_base = measure_forward_ms(fwd_base, device, args.warmup, args.repeat, args.stable_tail)
    ms_nosort = measure_forward_ms(fwd_nosort_sorted_edges, device, args.warmup, args.repeat, args.stable_tail)

    print("\n=== 4) 整层 forward：含 argsort vs 跳过 argsort（边已按 dst 排好）===", flush=True)
    print(f"  基类（库实现，内部 argsort）median: {ms_base:.3f} ms", flush=True)
    print(f"  测试子类（无 argsort）median: {ms_nosort:.3f} ms", flush=True)
    pct_est = 100.0 * (ms_base - ms_nosort) / max(ms_base, 1e-9)
    print(f"  估计去掉 argsort 的 forward 变化: {pct_est:+.2f}%（CPU 上噪声大，GPU 更有意义）", flush=True)
    print(f"  单独 argsort 耗时 / forward: {100.0 * t_sort / max(ms_base, 1e-9):.2f}%", flush=True)

    del layer_base, layer_nosort
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # --- 5) torch.compile ---
    if args.compile or args.compile_full:
        if not hasattr(torch, "compile"):
            print("\n=== 5) torch.compile：当前 PyTorch 不支持 ===", flush=True)
        else:
            layer_c = build_save_layer(device, dtype, config)
            layer_c.eval()
            pos_c = pos.clone()
            example = (pos_c, A, batch, edge_src_s, edge_dst_s, edge_shifts_s, cell)

            def fwd_c():
                with torch.no_grad():
                    return layer_c(*example)

            ms_before = measure_forward_ms(fwd_c, device, max(3, args.warmup // 2), args.repeat, args.stable_tail)
            try:
                layer_c = torch.compile(layer_c, mode="reduce-overhead")
            except Exception as e:
                print(f"\n=== 5) torch.compile 失败: {e} ===", flush=True)
            else:
                for _ in range(max(5, args.warmup)):
                    with torch.no_grad():
                        layer_c(*example)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ms_after = measure_forward_ms(fwd_c, device, args.warmup, args.repeat, args.stable_tail)
                print("\n=== 5) torch.compile（整层 forward，边已排序，无图内 argsort 对比意义时仍用随机边）===", flush=True)
                print(f"  compile 前 median: {ms_before:.3f} ms", flush=True)
                print(f"  compile 后 median: {ms_after:.3f} ms", flush=True)
                if ms_after > 0:
                    print(f"  比值 (before/after): {ms_before / ms_after:.3f}x", flush=True)

            if args.compile_full:
                layer_cb = build_save_layer(device, dtype, config)
                layer_cb.train()
                pos_t = pos.detach().clone().requires_grad_(True)
                g = (pos_t, A, batch, edge_src, edge_dst, edge_shifts, cell)
                try:
                    layer_cb = torch.compile(layer_cb, mode="reduce-overhead")
                except Exception as e:
                    print(f"  compile full 失败: {e}", flush=True)
                else:
                    ms_fb = measure_forward_backward_ms(
                        layer_cb, g, device, max(5, args.warmup), args.repeat, args.stable_tail
                    )
                    print(f"  forward+backward（compile）median: {ms_fb:.3f} ms（与未 compile 可自行对比）", flush=True)

    print("\n完成。", flush=True)


if __name__ == "__main__":
    main()
