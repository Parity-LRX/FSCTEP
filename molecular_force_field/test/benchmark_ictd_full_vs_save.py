#!/usr/bin/env python3
"""
对比 pure-cartesian-ictd (Full) 与 pure-cartesian-ictd-save (Save) 的参数量和推理/反向速度。

用法:
  python -m molecular_force_field.benchmark_ictd_full_vs_save
  python -m molecular_force_field.benchmark_ictd_full_vs_save --device cuda --sizes 1000 10000 50000
"""
from __future__ import annotations

import argparse
import time

import torch

from molecular_force_field.models.pure_cartesian_ictd_layers import PureCartesianICTDTransformerLayer as SaveLayer
from molecular_force_field.models.pure_cartesian_ictd_layers_full import PureCartesianICTDTransformerLayer as FullLayer
from molecular_force_field.utils.config import ModelConfig


def param_count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def make_dummy_graph(device: torch.device, dtype: torch.dtype, num_nodes: int, avg_degree: int = 24, seed: int = 42):
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


def build_layer(layer_cls, config: ModelConfig, device: torch.device, dtype: torch.dtype, num_interaction: int = 2):
    return layer_cls(
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
        num_interaction=num_interaction,
        function_type_main=config.function_type,
        lmax=config.lmax,
        ictd_tp_path_policy="full",
        ictd_tp_max_rank_other=None,
        internal_compute_dtype=dtype,
        device=device,
    ).to(device=device, dtype=dtype)


def time_forward(layer: torch.nn.Module, graph, warmup: int, repeat: int, device: torch.device) -> float:
    """返回单次前向的毫秒数。"""
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
    layer.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = layer(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            _ = layer(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / repeat * 1000.0


def time_forward_backward(layer: torch.nn.Module, graph, warmup: int, repeat: int, device: torch.device) -> float:
    """返回单次前向+反向（算力）的毫秒数。"""
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
    pos = pos.detach().requires_grad_(True)
    layer.eval()
    for _ in range(warmup):
        layer.zero_grad(set_to_none=True)
        out = layer(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        loss = out.sum()
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        layer.zero_grad(set_to_none=True)
        if pos.grad is not None:
            pos.grad = None
        out = layer(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        loss = out.sum()
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000.0


def main():
    parser = argparse.ArgumentParser(description="Compare pure-cartesian-ictd (Full) vs pure-cartesian-ictd-save (Save)")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--num-interaction", type=int, default=2)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000],
                        help="Atom counts for speed test (default: 1000 10000 50000)")
    parser.add_argument("--avg-degree", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    config = ModelConfig(dtype=dtype)
    print("Config: lmax={}, channel_in={}, num_interaction={}, dtype={}".format(
        config.lmax, config.channel_in, args.num_interaction, args.dtype), flush=True)
    print("Device: {}".format(device), flush=True)
    print(flush=True)

    # Build both models
    full_layer = build_layer(FullLayer, config, device, dtype, args.num_interaction)
    save_layer = build_layer(SaveLayer, config, device, dtype, args.num_interaction)

    n_full = param_count(full_layer)
    n_save = param_count(save_layer)
    print("=" * 60, flush=True)
    print("参数量 (Parameters)", flush=True)
    print("=" * 60, flush=True)
    print("  pure-cartesian-ictd      (Full): {:>12,}".format(n_full), flush=True)
    print("  pure-cartesian-ictd-save (Save): {:>12,}".format(n_save), flush=True)
    if n_save != 0:
        diff_pct = (n_full - n_save) / n_save * 100
        print("  差异: Full - Save = {:,} ({:+.1f}%)".format(n_full - n_save, diff_pct))
    print()

    # Speed: forward only
    print("=" * 60)
    print("速度：前向 only (ms/call)")
    print("=" * 60)
    print("{:>10} | {:>18} | {:>18} | {:>10}".format(
        "N (atoms)", "Full (ms)", "Save (ms)", "Full/Save"))
    print("-" * 60)
    for n in args.sizes:
        try:
            graph = make_dummy_graph(device, dtype, n, args.avg_degree)
            t_full = time_forward(full_layer, graph, args.warmup, args.repeat, device)
            t_save = time_forward(save_layer, graph, args.warmup, args.repeat, device)
            ratio = t_full / t_save if t_save > 0 else 0
            print("{:>10} | {:>18.2f} | {:>18.2f} | {:>10.2f}x".format(n, t_full, t_save, ratio))
        except Exception as e:
            print("{:>10} | Error: {}".format(n, e))
    print()

    # Speed: forward + backward (forces)
    print("=" * 60)
    print("速度：前向+反向(力) (ms/call)")
    print("=" * 60)
    print("{:>10} | {:>18} | {:>18} | {:>10}".format(
        "N (atoms)", "Full (ms)", "Save (ms)", "Full/Save"))
    print("-" * 60)
    for n in args.sizes:
        try:
            graph = make_dummy_graph(device, dtype, n, args.avg_degree)
            t_full = time_forward_backward(full_layer, graph, args.warmup, args.repeat, device)
            t_save = time_forward_backward(save_layer, graph, args.warmup, args.repeat, device)
            ratio = t_full / t_save if t_save > 0 else 0
            print("{:>10} | {:>18.2f} | {:>18.2f} | {:>10.2f}x".format(n, t_full, t_save, ratio))
        except Exception as e:
            print("{:>10} | Error: {}".format(n, e))
    print()
    print("Done.")


if __name__ == "__main__":
    main()
