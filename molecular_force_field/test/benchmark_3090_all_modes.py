#!/usr/bin/env python3
"""
3090 单卡 benchmark：测试所有 tensor_product_mode 的最大原子数、推理速度（含力），并与 MACE 对比。

特性：
- 二分法查找最大原子数，OOM 时继续缩小范围
- 多轮 warmup + repeat，取后面稳定值（中位数）
- 可选 torch.compile
- 测量 forward+backward（能量+力）耗时

用法:
  python -m molecular_force_field.test.benchmark_3090_all_modes
  python -m molecular_force_field.test.benchmark_3090_all_modes --compile
  python -m molecular_force_field.test.benchmark_3090_all_modes --no-mace
  python -m molecular_force_field.test.benchmark_3090_all_modes --modes spherical pure-cartesian-ictd
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import warnings
import time
from typing import Any

# Suppress TorchScript type annotation warnings from torch.compile
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit")

import torch

# Fix PyTorch 2.6 weights_only issue with e3nn
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# Add repo root
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

# 所有支持的 tensor_product_mode（spherical-save-cue 需 cuEquivariance）
ALL_TP_MODES = [
    "spherical",
    "spherical-save",
    "spherical-save-cue",
    "partial-cartesian",
    "partial-cartesian-loose",
    "pure-cartesian",
    "pure-cartesian-sparse",
    "pure-cartesian-ictd",
    "pure-cartesian-ictd-save",
]


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


def _common_cfg(
    channels: int = 64,
    lmax: int = 2,
    num_interaction: int = 2,
    dtype: torch.dtype = torch.float32,
) -> dict:
    cfg = ModelConfig(dtype=dtype)
    cfg.channel_in = channels
    cfg.irreps_output_conv_channels = channels
    cfg.lmax = lmax
    return dict(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=channels,
        hidden_dim_sh=cfg.get_hidden_dim_sh(),
        hidden_dim=64,
        channel_in2=32,
        embedding_dim=16,
        max_atomvalue=10,
        output_size=8,
        embed_size=cfg.embed_size,
        main_hidden_sizes3=cfg.main_hidden_sizes3,
        num_interaction=num_interaction,
        lmax=lmax,
        function_type_main="gaussian",
        device=torch.device("cpu"),  # 会在 .to(device) 时覆盖
    )


def build_fscetp_layer(
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    channels: int = 64,
    lmax: int = 2,
    num_interaction: int = 2,
    max_rank_other: int = 1,
    k_policy: str = "k0",
    ictd_tp_path_policy: str = "full",
) -> torch.nn.Module:
    """Build FSCETP layer. All tensor_product_mode use same channels, lmax, num_interaction."""
    cfg = ModelConfig(dtype=dtype)
    cfg.channel_in = channels
    cfg.irreps_output_conv_channels = channels
    cfg.lmax = lmax

    # spherical 系列用 irreps，不用 hidden_dim_conv / lmax
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

    # Cartesian 系列用 hidden_dim_conv, lmax
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
            import cuequivariance_torch  # noqa: F401
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
            max_rank_other=max_rank_other,
            k_policy=k_policy,
        )
    elif mode == "pure-cartesian-ictd":
        layer = PureCartesianICTDTransformerLayerFull(
            **k_cartesian,
            ictd_tp_path_policy=ictd_tp_path_policy,
            ictd_tp_max_rank_other=None,
            internal_compute_dtype=dtype,
        )
    elif mode == "pure-cartesian-ictd-save":
        layer = PureCartesianICTDSave(
            **k_cartesian,
            ictd_tp_path_policy=ictd_tp_path_policy,
            ictd_tp_max_rank_other=None,
            internal_compute_dtype=dtype,
        )
    else:
        raise ValueError(f"未知模式: {mode}")

    return layer.to(device=device, dtype=dtype)


def build_mace_layer(
    device: torch.device,
    dtype: torch.dtype,
    channels: int = 64,
    lmax: int = 2,
    num_interaction: int = 2,
    correlation: int = 3,
    avg_degree: int = 24,
) -> torch.nn.Module:
    """MACE 默认: num_interaction=2, correlation=3, lmax=2, channels=64"""
    import numpy as np
    import e3nn.o3 as o3
    import mace.modules

    layer = mace.modules.MACE(
        r_max=5.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=lmax,
        interaction_cls=mace.modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=mace.modules.interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=num_interaction,
        num_elements=1,
        hidden_irreps=o3.Irreps(f"{channels}x0e + {channels}x1o + {channels}x2e"),
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=np.zeros(1),
        avg_num_neighbors=avg_degree,
        atomic_numbers=[1],
        correlation=correlation,
        gate=torch.nn.functional.silu,
    ).to(device=device, dtype=dtype)
    return layer


def _to_mace_data(graph: tuple, device: torch.device) -> tuple:
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
    num_nodes = pos.shape[0]
    mace_data = {
        "positions": pos,
        "node_attrs": torch.ones(num_nodes, 1, device=device, dtype=pos.dtype),
        "edge_index": torch.vstack([edge_src, edge_dst]),
        "shifts": edge_shifts,
        "unit_shifts": edge_shifts,
        "cell": cell,
        "batch": batch,
        "ptr": torch.tensor([0, num_nodes], device=device, dtype=torch.long),
    }
    return (mace_data,)


def measure_inference_ms(
    layer: torch.nn.Module,
    graph: tuple,
    warmup: int,
    repeat: int,
    device: torch.device,
    is_mace: bool = False,
    backward: bool = True,
    stable_tail: int | None = None,
) -> float:
    """测量 forward+backward 平均耗时 (ms)。backward=True 时包含力的计算。
    stable_tail: 取最后 N 次测量的中位数作为稳定值；None 则用全部均值。
    """
    if is_mace:
        pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
        if backward:
            pos = pos.clone().requires_grad_(True)
            graph = (pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        graph = _to_mace_data(graph, device)
    else:
        pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
        if backward:
            pos = pos.clone().requires_grad_(True)
            graph = (pos, A, batch, edge_src, edge_dst, edge_shifts, cell)

    if backward:
        layer.train()
    else:
        layer.eval()

    context = torch.enable_grad() if backward else torch.no_grad()
    with context:
        for _ in range(warmup):
            if backward:
                layer.zero_grad(set_to_none=True)
            if is_mace:
                out = layer(*graph, compute_force=False)
                if backward:
                    loss = out["energy"].sum()
                    loss.backward()
            else:
                out = layer(*graph)
                if backward:
                    loss = out.sum()
                    loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()

        times: list[float] = []
        for _ in range(repeat):
            if backward:
                layer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if is_mace:
                out = layer(*graph, compute_force=False)
                if backward:
                    loss = out["energy"].sum()
                    loss.backward()
            else:
                out = layer(*graph)
                if backward:
                    loss = out.sum()
                    loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    if stable_tail is not None and len(times) >= stable_tail:
        tail = times[-stable_tail:]
        tail.sort()
        return float(tail[len(tail) // 2])  # 中位数
    return sum(times) / len(times)


def binary_search_max_atoms(
    layer: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    low: int,
    high: int,
    warmup: int,
    repeat: int,
    avg_degree: int,
    is_mace: bool = False,
    stable_tail: int | None = None,
) -> tuple[int, float]:
    """二分法查找最大原子数，OOM 时 high=mid-1 继续。返回 (max_atoms, time_ms)。"""
    best_n = 0
    best_time = 0.0

    while low <= high:
        mid = (low + high) // 2
        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        try:
            graph = make_dummy_graph(device, dtype, mid, avg_degree)
            t_ms = measure_inference_ms(
                layer, graph, warmup=warmup, repeat=repeat,
                device=device, is_mace=is_mace, backward=True,
                stable_tail=stable_tail,
            )
            best_n = mid
            best_time = t_ms
            low = mid + 1
        except RuntimeError as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or "cuda" in err_str and "memory" in err_str:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                high = mid - 1
            else:
                raise
    return best_n, best_time


def run_benchmark_one_mode(
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    channels: int = 64,
    lmax: int = 2,
    num_interaction: int = 2,
    avg_degree: int = 24,
    warmup: int = 10,
    repeat: int = 15,
    stable_tail: int = 10,
    use_compile: bool = False,
    max_atoms_cap: int = 500_000,
) -> dict[str, Any]:
    """对单个模式运行 benchmark，返回 max_atoms, time_ms, atoms_per_sec 等。"""
    result: dict[str, Any] = {
        "mode": mode,
        "max_atoms": 0,
        "time_ms": 0.0,
        "atoms_per_sec": 0.0,
        "error": None,
    }

    try:
        layer = build_fscetp_layer(
            mode, device, dtype,
            channels=channels, lmax=lmax, num_interaction=num_interaction,
        )
        layer.eval()
    except ImportError as e:
        result["error"] = str(e)
        return result

    if use_compile and device.type == "cuda":
        try:
            layer = torch.compile(layer, mode="reduce-overhead", dynamic=False)
            # 预热
            _ = measure_inference_ms(
                layer, make_dummy_graph(device, dtype, 256, avg_degree),
                warmup=5, repeat=3, device=device, is_mace=False, backward=True,
            )
        except Exception as e:
            result["error"] = f"compile 失败: {e}"
            return result

    max_atoms, time_ms = binary_search_max_atoms(
        layer, device, dtype,
        low=1, high=max_atoms_cap,
        warmup=warmup, repeat=repeat, avg_degree=avg_degree,
        is_mace=False, stable_tail=stable_tail,
    )
    result["max_atoms"] = max_atoms
    result["time_ms"] = time_ms
    result["atoms_per_sec"] = max_atoms / (time_ms / 1000.0) if time_ms > 0 else 0.0
    del layer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def run_benchmark_mace(
    device: torch.device,
    dtype: torch.dtype,
    *,
    channels: int = 64,
    lmax: int = 2,
    num_interaction: int = 2,
    correlation: int = 3,
    avg_degree: int = 24,
    warmup: int = 10,
    repeat: int = 15,
    stable_tail: int = 10,
    use_compile: bool = False,
    max_atoms_cap: int = 500_000,
) -> dict[str, Any]:
    """对 MACE 运行 benchmark。"""
    result: dict[str, Any] = {
        "mode": "MACE",
        "max_atoms": 0,
        "time_ms": 0.0,
        "atoms_per_sec": 0.0,
        "error": None,
    }
    try:
        layer = build_mace_layer(
            device, dtype,
            channels=channels, lmax=lmax, num_interaction=num_interaction,
            correlation=correlation, avg_degree=avg_degree,
        )
        layer.eval()
    except ImportError as e:
        result["error"] = f"MACE 未安装: {e}"
        return result

    if use_compile and device.type == "cuda":
        try:
            layer = torch.compile(layer, mode="reduce-overhead", dynamic=False)
            _ = measure_inference_ms(
                layer, make_dummy_graph(device, dtype, 256, avg_degree),
                warmup=5, repeat=3, device=device, is_mace=True, backward=True,
            )
        except Exception as e:
            result["error"] = f"MACE compile 失败: {e}"
            return result

    max_atoms, time_ms = binary_search_max_atoms(
        layer, device, dtype,
        low=1, high=max_atoms_cap,
        warmup=warmup, repeat=repeat, avg_degree=avg_degree,
        is_mace=True, stable_tail=stable_tail,
    )
    result["max_atoms"] = max_atoms
    result["time_ms"] = time_ms
    result["atoms_per_sec"] = max_atoms / (time_ms / 1000.0) if time_ms > 0 else 0.0
    del layer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="3090 单卡：所有 tensor_product_mode 最大原子数、推理速度（含力）、与 MACE 对比"
    )
    parser.add_argument("--modes", type=str, nargs="+", default=None,
                        help=f"指定模式，默认全部。可选: {ALL_TP_MODES}")
    parser.add_argument("--compile", action="store_true", help="使用 torch.compile")
    parser.add_argument("--no-mace", action="store_true", help="不运行 MACE 对比")
    parser.add_argument("--device", type=str, default=None,
                        help="设备，默认 cuda")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--num-interaction", type=int, default=2)
    parser.add_argument("--correlation", type=int, default=3,
                        help="MACE correlation / many-body order (default: 3)")
    parser.add_argument("--avg-degree", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=10,
                        help="每次测量的 warmup 轮数")
    parser.add_argument("--repeat", type=int, default=15,
                        help="每次测量的 repeat 轮数")
    parser.add_argument("--stable-tail", type=int, default=10,
                        help="取最后 N 次测量的中位数作为稳定值")
    parser.add_argument("--max-atoms-cap", type=int, default=500_000,
                        help="二分上界")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"],
                        help="Compute dtype (default: float32)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)  # Required for cuequivariance Fused TP math_dtype
    if device.type == "cuda" and dtype == torch.float32:
        torch.set_float32_matmul_precision("high")

    modes = args.modes or ALL_TP_MODES
    use_compile = args.compile

    # All FSCETP modes and MACE use the same: lmax, channels, num_interaction
    ch, lmax_val, ni = args.channels, args.lmax, args.num_interaction
    print("=" * 80)
    print("3090 Single-GPU Benchmark: tensor_product_mode max_atoms + inference (energy+forces)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Config (all modes + MACE): lmax={lmax_val}, channels={ch}, num_interaction={ni}, dtype={args.dtype}")
    print(f"  MACE only: correlation={args.correlation}")
    print(f"compile: {use_compile}, warmup={args.warmup}, repeat={args.repeat}, stable_tail={args.stable_tail}")
    print(f"Modes: {modes}")
    print()

    results: list[dict[str, Any]] = []

    for mode in modes:
        print(f"[{mode}] running...", flush=True)
        res = run_benchmark_one_mode(
            mode, device, dtype,
            channels=args.channels,
            lmax=args.lmax,
            num_interaction=args.num_interaction,
            avg_degree=args.avg_degree,
            warmup=args.warmup,
            repeat=args.repeat,
            stable_tail=args.stable_tail,
            use_compile=use_compile,
            max_atoms_cap=args.max_atoms_cap,
        )
        results.append(res)
        if res.get("error"):
            print(f"  -> skip: {res['error']}")
        else:
            print(f"  -> max_atoms={res['max_atoms']}, time={res['time_ms']:.2f} ms, "
                  f"{res['atoms_per_sec']:.0f} atoms/s")

    if not args.no_mace:
        print("[MACE] running...", flush=True)
        mace_res = run_benchmark_mace(
            device, dtype,
            channels=args.channels,
            lmax=args.lmax,
            num_interaction=args.num_interaction,
            correlation=args.correlation,
            avg_degree=args.avg_degree,
            warmup=args.warmup,
            repeat=args.repeat,
            stable_tail=args.stable_tail,
            use_compile=use_compile,
            max_atoms_cap=args.max_atoms_cap,
        )
        results.append(mace_res)
        if mace_res.get("error"):
            print(f"  -> skip: {mace_res['error']}")
        else:
            print(f"  -> max_atoms={mace_res['max_atoms']}, time={mace_res['time_ms']:.2f} ms, "
                  f"{mace_res['atoms_per_sec']:.0f} atoms/s")

    # Summary table: all modes + MACE, one table
    # Order: FSCETP modes (ALL_TP_MODES first, then any extras), MACE last
    mace_atoms = None
    mace_time = None
    results_by_mode = {r["mode"]: r for r in results}
    for r in results:
        if r["mode"] == "MACE" and not r.get("error"):
            mace_atoms = r["max_atoms"]
            mace_time = r["time_ms"]
            break

    table_order = list(modes) + (["MACE"] if not args.no_mace else [])
    # Dedupe preserving order
    seen = set()
    table_order = [m for m in table_order if not (m in seen or seen.add(m))]

    print()
    print("=" * 95)
    print("Summary (all modes + MACE) | lmax=%d channels=%d num_interaction=%d %s" % (args.lmax, args.channels, args.num_interaction, args.dtype))
    print("=" * 95)
    print(f"{'Mode':<28} | {'max_atoms':>10} | {'time(ms)':>12} | {'atoms/s':>12} | {'vs MACE':>8}")
    print("-" * 95)

    for mode in table_order:
        r = results_by_mode.get(mode)
        if r is None:
            continue
        err = r.get("error")
        if err:
            print(f"{mode:<28} | {'N/A':>10} | {'N/A':>12} | {'N/A':>12} | {err[:20]:>8}")
            continue
        max_a = r["max_atoms"]
        t_ms = r["time_ms"]
        aps = r["atoms_per_sec"]
        vs_mace = ""
        if mode != "MACE" and mace_atoms is not None and mace_time is not None:
            ratio = max_a / mace_atoms if mace_atoms > 0 else 0
            vs_mace = f"{ratio:.2f}x"
        print(f"{mode:<28} | {max_a:>10} | {t_ms:>12.2f} | {aps:>12.0f} | {vs_mace:>8}")
    print("=" * 95)
    print("Note: vs MACE = max_atoms of this mode / MACE max_atoms")
    print("      time = forward+backward (energy+forces), median of stable tail")


if __name__ == "__main__":
    main()
