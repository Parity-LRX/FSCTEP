#!/usr/bin/env python3
"""
所有张量积模式速度对比 benchmark，基准为 spherical。

特性：
- 支持 GPU 和 CPU
- 多轮 warmup + repeat，取稳定段中位数
- l=0..6：spherical、spherical-save、spherical-save-cue、partial-cartesian、partial-cartesian-loose
- l=0..3：pure-cartesian、pure-cartesian-sparse、pure-cartesian-ictd、pure-cartesian-ictd-save（>3 易 OOM）
- 典型配置：默认模型大小（lmax=2, channels=64）下所有模式对比
- 输出相对 spherical 的加速比（spherical 为 1.0x）

用法:
  python -m molecular_force_field.test.benchmark_tp_modes_speed
  python -m molecular_force_field.test.benchmark_tp_modes_speed --device cpu
  python -m molecular_force_field.test.benchmark_tp_modes_speed --atoms 512 --warmup 15 --repeat 30
  python -m molecular_force_field.test.benchmark_tp_modes_speed --typical-only
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from typing import Any

import torch

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from molecular_force_field.models import (
    CartesianTransformerLayer,
    CartesianTransformerLayerLoose,
    E3_TransformerLayer_multi,
    PureCartesianSparseTransformerLayer,
    PureCartesianTransformerLayer,
)
from molecular_force_field.models.e3nn_layers_channelwise import (
    E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise,
)
from molecular_force_field.models.pure_cartesian_ictd_layers import (
    PureCartesianICTDTransformerLayer as PureCartesianICTDSave,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
    PureCartesianICTDTransformerLayer as PureCartesianICTDTransformerLayerFull,
)
from molecular_force_field.utils.config import ModelConfig

# 非 pure_cartesian 系列：lmax 0..6
MODES_LMAX_0_6 = [
    "spherical",
    "spherical-save",
    "spherical-save-cue",
    "partial-cartesian",
    "partial-cartesian-loose",
]
# pure_cartesian 系列：lmax 0..3（>3 易 OOM）
PURE_CARTESIAN_MODES = [
    "pure-cartesian",
    "pure-cartesian-sparse",
    "pure-cartesian-ictd",
    "pure-cartesian-ictd-save",
]
PURE_CARTESIAN_MAX_L = 3
ALL_MODES = MODES_LMAX_0_6 + PURE_CARTESIAN_MODES


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


def build_layer(
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    channels: int = 64,
    lmax: int = 2,
    num_interaction: int = 2,
) -> torch.nn.Module:
    cfg = ModelConfig(dtype=dtype)
    cfg.channel_in = channels
    cfg.irreps_output_conv_channels = channels
    cfg.lmax = lmax

    k_spherical = dict(
        max_embed_radius=cfg.max_radius,
        main_max_radius=cfg.max_radius_main,
        main_number_of_basis=cfg.number_of_basis_main,
        irreps_input=cfg.get_irreps_output_conv(),
        irreps_query=cfg.get_irreps_query_main(),
        irreps_key=cfg.get_irreps_key_main(),
        irreps_value=cfg.get_irreps_value_main(),
        irreps_output=cfg.get_irreps_output_conv_2(),
        irreps_sh=cfg.get_irreps_sh_transformer(),
        hidden_dim_sh=cfg.get_hidden_dim_sh(),
        hidden_dim=cfg.emb_number_main_2,
        channel_in2=cfg.channel_in2,
        embedding_dim=cfg.embedding_dim,
        max_atomvalue=cfg.max_atomvalue,
        output_size=cfg.output_size,
        embed_size=cfg.embed_size,
        main_hidden_sizes3=cfg.main_hidden_sizes3,
        num_layers=cfg.num_layers,
        num_interaction=num_interaction,
        function_type_main=cfg.function_type,
        device=device,
    )

    k_cartesian = dict(
        max_embed_radius=cfg.max_radius,
        main_max_radius=cfg.max_radius_main,
        main_number_of_basis=cfg.number_of_basis_main,
        hidden_dim_conv=channels,
        hidden_dim_sh=cfg.get_hidden_dim_sh(),
        hidden_dim=cfg.emb_number_main_2,
        channel_in2=cfg.channel_in2,
        embedding_dim=cfg.embedding_dim,
        max_atomvalue=cfg.max_atomvalue,
        output_size=cfg.output_size,
        embed_size=cfg.embed_size,
        main_hidden_sizes3=cfg.main_hidden_sizes3,
        num_layers=cfg.num_layers,
        num_interaction=num_interaction,
        function_type_main=cfg.function_type,
        lmax=lmax,
        device=device,
    )

    if mode == "spherical":
        layer = E3_TransformerLayer_multi(**k_spherical)
    elif mode == "spherical-save":
        layer = E3_TransformerLayer_multi_channelwise(**k_spherical)
    elif mode == "spherical-save-cue":
        try:
            from molecular_force_field.models.cue_layers_channelwise import (
                E3_TransformerLayer_multi as E3_TransformerLayer_multi_cue,
            )
        except Exception as e:
            raise ImportError(
                f"spherical-save-cue 需要 cuEquivariance: {e}\n"
                "安装: pip install cuequivariance-torch cuequivariance-ops-torch-cu12"
            ) from e
        layer = E3_TransformerLayer_multi_cue(**k_spherical)
    elif mode == "partial-cartesian":
        layer = CartesianTransformerLayer(**k_cartesian)
    elif mode == "partial-cartesian-loose":
        layer = CartesianTransformerLayerLoose(**k_cartesian)
    elif mode == "pure-cartesian":
        layer = PureCartesianTransformerLayer(**k_cartesian)
    elif mode == "pure-cartesian-sparse":
        layer = PureCartesianSparseTransformerLayer(
            **k_cartesian,
            max_rank_other=1,
            k_policy="k0",
        )
    elif mode == "pure-cartesian-ictd":
        layer = PureCartesianICTDTransformerLayerFull(
            **k_cartesian,
            ictd_tp_path_policy="full",
            ictd_tp_max_rank_other=None,
            internal_compute_dtype=dtype,
        )
    elif mode == "pure-cartesian-ictd-save":
        layer = PureCartesianICTDSave(
            **k_cartesian,
            ictd_tp_path_policy="full",
            ictd_tp_max_rank_other=None,
            internal_compute_dtype=dtype,
        )
    else:
        raise ValueError(f"未知模式: {mode}")

    return layer.to(device=device, dtype=dtype)


def measure_forward_backward_ms(
    layer: torch.nn.Module,
    graph: tuple,
    device: torch.device,
    warmup: int,
    repeat: int,
    stable_tail: int,
) -> float:
    """测量 forward+backward 平均耗时 (ms)，取最后 stable_tail 次的中位数。"""
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
    pos = pos.detach().clone().requires_grad_(True)
    graph = (pos, A, batch, edge_src, edge_dst, edge_shifts, cell)

    layer.train()
    with torch.enable_grad():
        for _ in range(warmup):
            layer.zero_grad(set_to_none=True)
            out = layer(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
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
            out = layer(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
            loss = out.sum()
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    if len(times) >= stable_tail:
        tail = sorted(times[-stable_tail:])
        return float(tail[len(tail) // 2])
    return sum(times) / len(times)


def run_one(
    mode: str,
    lmax: int,
    device: torch.device,
    dtype: torch.dtype,
    graph: tuple,
    channels: int,
    num_interaction: int,
    warmup: int,
    repeat: int,
    stable_tail: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"mode": mode, "lmax": lmax, "time_ms": None, "error": None}
    try:
        layer = build_layer(mode, device, dtype, channels=channels, lmax=lmax, num_interaction=num_interaction)
        t_ms = measure_forward_backward_ms(layer, graph, device, warmup, repeat, stable_tail)
        result["time_ms"] = t_ms
        del layer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        result["error"] = str(e).replace("\n", " ")[:100]
    return result


def run_lmax_scan(
    device: torch.device,
    dtype: torch.dtype,
    atoms: int,
    channels: int,
    num_interaction: int,
    warmup: int,
    repeat: int,
    stable_tail: int,
    avg_degree: int,
    modes: list[str],
) -> dict[str, dict[int, float | None]]:
    """扫描 lmax，返回 {mode: {lmax: time_ms}}。"""
    graph = make_dummy_graph(device, dtype, atoms, avg_degree)
    results: dict[str, dict[int, float | None]] = {m: {} for m in modes}

    for mode in modes:
        lmax_range = range(PURE_CARTESIAN_MAX_L + 1) if mode in PURE_CARTESIAN_MODES else range(7)
        for lmax in lmax_range:
            res = run_one(mode, lmax, device, dtype, graph, channels, num_interaction, warmup, repeat, stable_tail)
            if res["error"]:
                print(f"  [{mode}] l={lmax}: FAIL {res['error'][:60]}", flush=True)
                results[mode][lmax] = None
            else:
                results[mode][lmax] = res["time_ms"]
                print(f"  [{mode}] l={lmax}: {res['time_ms']:.2f} ms", flush=True)

    return results


def run_typical_config(
    device: torch.device,
    dtype: torch.dtype,
    atoms: int,
    channels: int,
    lmax: int,
    num_interaction: int,
    warmup: int,
    repeat: int,
    stable_tail: int,
    avg_degree: int,
    modes: list[str],
) -> dict[str, float | None]:
    """典型配置（默认模型大小）下所有模式速度。"""
    graph = make_dummy_graph(device, dtype, atoms, avg_degree)
    results: dict[str, float | None] = {}

    for mode in modes:
        res = run_one(mode, lmax, device, dtype, graph, channels, num_interaction, warmup, repeat, stable_tail)
        if res["error"]:
            print(f"  [{mode}]: FAIL {res['error'][:60]}", flush=True)
            results[mode] = None
        else:
            results[mode] = res["time_ms"]
            print(f"  [{mode}]: {res['time_ms']:.2f} ms", flush=True)

    return results


def print_lmax_table(
    results: dict[str, dict[int, float | None]],
    modes: list[str],
    baseline_mode: str = "spherical",
):
    """打印 lmax 扫描表格，含相对 spherical 的加速比。"""
    lmax_0_6 = list(range(7))

    print("\n" + "=" * 100)
    print("lmax 扫描 (forward+backward ms/call) | 基准: spherical = 1.0x")
    print("=" * 100)

    # 非 pure 模式：l=0..6
    header = "mode".ljust(28) + "".join([f"l={l}".rjust(10) for l in lmax_0_6]) + " | vs spherical"
    print(header)
    print("-" * len(header))
    base_l2 = results.get(baseline_mode, {}).get(2)
    for mode in modes:
        if mode in PURE_CARTESIAN_MODES:
            continue
        row = mode.ljust(28)
        for l in lmax_0_6:
            t = results.get(mode, {}).get(l)
            cell = "N/A" if t is None else f"{t:.1f}"
            row += cell.rjust(10)
        t_this = results.get(mode, {}).get(2)
        if t_this is not None and base_l2 is not None and base_l2 > 0:
            ratio = base_l2 / t_this
            row += f" | l=2: {ratio:.2f}x"
        elif mode == baseline_mode:
            row += " | 1.0x"
        else:
            row += " | -"
        print(row)

    # pure 模式：l=0..3
    print("-" * len(header))
    for mode in modes:
        if mode not in PURE_CARTESIAN_MODES:
            continue
        row = mode.ljust(28)
        for l in lmax_0_6:
            if l <= PURE_CARTESIAN_MAX_L:
                t = results.get(mode, {}).get(l)
                cell = "N/A" if t is None else f"{t:.1f}"
            else:
                cell = "-"
            row += cell.rjust(10)
        t2 = results.get(mode, {}).get(2)
        base = results.get(baseline_mode, {}).get(2)
        if t2 is not None and base is not None and base > 0:
            row += f" | l=2: {base/t2:.2f}x"
        elif mode == baseline_mode and t2 is not None:
            row += " | 1.0x"
        else:
            row += " | -"
        print(row)
    print("=" * 100)


def print_typical_table(
    results: dict[str, float | None],
    modes: list[str],
    baseline_mode: str = "spherical",
):
    """打印典型配置表格。"""
    print("\n" + "=" * 80)
    print("典型配置 (lmax=2, channels=64, num_interaction=2) | 基准: spherical = 1.0x")
    print("=" * 80)
    baseline_ms = results.get(baseline_mode)
    if baseline_ms is None:
        baseline_ms = 1.0
    print(f"{'mode':<30} | {'time (ms)':>12} | vs spherical")
    print("-" * 80)
    for mode in modes:
        t = results.get(mode)
        cell = "N/A" if t is None else f"{t:.2f}"
        ratio = "N/A" if t is None or baseline_ms is None or baseline_ms <= 0 else f"{baseline_ms/t:.2f}x"
        print(f"{mode:<30} | {cell:>12} | {ratio}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="所有张量积模式速度对比，基准 spherical，支持 GPU/CPU"
    )
    parser.add_argument("--device", type=str, default=None, help="cpu 或 cuda，默认自动")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--atoms", type=int, default=256, help="原子数")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--num-interaction", type=int, default=2)
    parser.add_argument("--avg-degree", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=10, help="warmup 轮数")
    parser.add_argument("--repeat", type=int, default=25, help="repeat 轮数")
    parser.add_argument("--stable-tail", type=int, default=15, help="取最后 N 次中位数")
    parser.add_argument("--typical-only", action="store_true", help="仅运行典型配置，不扫描 lmax")
    parser.add_argument("--lmax-only", action="store_true", help="仅运行 lmax 扫描，不运行典型配置")
    parser.add_argument("--modes", type=str, nargs="+", default=None, help="指定模式，默认全部")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    modes = args.modes or ALL_MODES

    if device.type == "cuda" and dtype == torch.float32:
        torch.set_float32_matmul_precision("high")

    print("=" * 80)
    print("张量积模式速度对比 | 基准: spherical")
    print("=" * 80)
    print(f"Device: {device} | dtype: {args.dtype} | atoms: {args.atoms}")
    print(f"channels={args.channels}, num_interaction={args.num_interaction}")
    print(f"warmup={args.warmup}, repeat={args.repeat}, stable_tail={args.stable_tail}")
    print(f"Modes: {modes}")
    print()

    if not args.typical_only:
        print("[1/2] lmax 扫描 (l=0..6 或 l=0..3 for pure_cartesian)...", flush=True)
        results_lmax = run_lmax_scan(
            device, dtype,
            atoms=args.atoms,
            channels=args.channels,
            num_interaction=args.num_interaction,
            warmup=args.warmup,
            repeat=args.repeat,
            stable_tail=args.stable_tail,
            avg_degree=args.avg_degree,
            modes=modes,
        )
        print_lmax_table(results_lmax, modes)

    if not args.lmax_only:
        print("\n[2/2] 典型配置 (lmax=2, 默认模型大小)...", flush=True)
        results_typical = run_typical_config(
            device, dtype,
            atoms=args.atoms,
            channels=args.channels,
            lmax=2,
            num_interaction=args.num_interaction,
            warmup=args.warmup,
            repeat=args.repeat,
            stable_tail=args.stable_tail,
            avg_degree=args.avg_degree,
            modes=modes,
        )
        print_typical_table(results_typical, modes)

    print("\nDone.")


if __name__ == "__main__":
    main()
