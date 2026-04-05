#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time

import torch

from molecular_force_field.models.pure_cartesian_ictd_layers import (
    PureCartesianICTDTransformerLayer as PureCartesianICTDSave,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_o3 import (
    PureCartesianICTDO3TransformerLayer,
)


def make_graph(num_nodes: int, avg_degree: int, device: torch.device, dtype: torch.dtype):
    pos = torch.randn(num_nodes, 3, device=device, dtype=dtype)
    A = torch.randint(1, 6, (num_nodes,), device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    num_edges = num_nodes * avg_degree
    edge_dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_src = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_shifts = torch.zeros(num_edges, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    return pos, A, batch, edge_src, edge_dst, edge_shifts, cell


def bench(model: torch.nn.Module, graph, *, warmup: int, repeat: int, do_backward: bool):
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
    for _ in range(warmup):
        p = pos.detach().clone().requires_grad_(do_backward)
        out = model(p, A, batch, edge_src, edge_dst, edge_shifts, cell)
        if do_backward:
            out.sum().backward()
    if pos.is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(pos.device)
    t0 = time.perf_counter()
    for _ in range(repeat):
        p = pos.detach().clone().requires_grad_(do_backward)
        out = model(p, A, batch, edge_src, edge_dst, edge_shifts, cell)
        if do_backward:
            out.sum().backward()
    if pos.is_cuda:
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated(pos.device) / (1024**2)
    else:
        peak_mb = 0.0
    ms = (time.perf_counter() - t0) / repeat * 1000.0
    return ms, peak_mb


def build_models(device: torch.device, dtype: torch.dtype):
    common = dict(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=64,
        hidden_dim_sh=32,
        hidden_dim=64,
        channel_in2=32,
        embedding_dim=16,
        max_atomvalue=10,
        output_size=8,
        embed_size=[128, 128, 128],
        main_hidden_sizes3=[64, 32],
        num_layers=1,
        num_interaction=2,
        lmax=2,
    )
    models = {
        "ictd-save-so3": PureCartesianICTDSave(**common).to(device=device, dtype=dtype),
        "ictd-save-o3-auto": PureCartesianICTDO3TransformerLayer(
            **common, o3_irrep_preset="auto"
        ).to(device=device, dtype=dtype),
        "ictd-save-o3-min": PureCartesianICTDO3TransformerLayer(
            **common, o3_irrep_preset="minimal"
        ).to(device=device, dtype=dtype),
        "ictd-save-o3-full": PureCartesianICTDO3TransformerLayer(
            **common, o3_irrep_preset="full"
        ).to(device=device, dtype=dtype),
    }
    return models


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--nodes", type=int, default=512)
    p.add_argument("--avg-degree", type=int, default=24)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    results = {name: {"fwd": [], "train": [], "fwd_peak": [], "train_peak": []} for name in build_models(device, dtype)}
    for run_idx in range(args.runs):
        torch.manual_seed(1234 + run_idx)
        graph = make_graph(args.nodes, args.avg_degree, device, dtype)
        models = build_models(device, dtype)
        for name, model in models.items():
            model.eval()
            fwd_ms, fwd_peak = bench(model, graph, warmup=args.warmup, repeat=args.repeat, do_backward=False)
            train_ms, train_peak = bench(model, graph, warmup=args.warmup, repeat=args.repeat, do_backward=True)
            results[name]["fwd"].append(fwd_ms)
            results[name]["train"].append(train_ms)
            results[name]["fwd_peak"].append(fwd_peak)
            results[name]["train_peak"].append(train_peak)
            params = sum(p.numel() for p in model.parameters())
            print(
                f"run={run_idx+1} variant={name} params={params} "
                f"fwd_ms={fwd_ms:.3f} train_ms={train_ms:.3f} "
                f"fwd_peak_mb={fwd_peak:.1f} train_peak_mb={train_peak:.1f}",
                flush=True,
            )
            del model
        del graph
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("== medians ==", flush=True)
    for name, vals in results.items():
        print(
            f"variant={name} runs={args.runs} "
            f"fwd_ms_med={statistics.median(vals['fwd']):.3f} "
            f"train_ms_med={statistics.median(vals['train']):.3f} "
            f"fwd_peak_mb_med={statistics.median(vals['fwd_peak']):.1f} "
            f"train_peak_mb_med={statistics.median(vals['train_peak']):.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
