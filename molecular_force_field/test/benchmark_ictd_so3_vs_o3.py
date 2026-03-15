#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import torch

from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
    PureCartesianICTDTransformerLayer as PureCartesianICTDSO3,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_o3 import (
    PureCartesianICTDO3TransformerLayer,
)


def _make_graph(num_nodes: int, avg_degree: int, device: torch.device, dtype: torch.dtype):
    pos = torch.randn(num_nodes, 3, device=device, dtype=dtype)
    A = torch.randint(1, 6, (num_nodes,), device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    num_edges = num_nodes * avg_degree
    edge_dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_src = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_shifts = torch.zeros(num_edges, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    return pos, A, batch, edge_src, edge_dst, edge_shifts, cell


def _bench(model: torch.nn.Module, graph, *, repeat: int, do_backward: bool) -> float:
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
    for _ in range(5):
        p = pos.detach().clone().requires_grad_(do_backward)
        out = model(p, A, batch, edge_src, edge_dst, edge_shifts, cell)
        if do_backward:
            out.sum().backward()
    if pos.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        p = pos.detach().clone().requires_grad_(do_backward)
        out = model(p, A, batch, edge_src, edge_dst, edge_shifts, cell)
        if do_backward:
            out.sum().backward()
    if pos.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1000.0


def _maybe_compile(model: torch.nn.Module, *, mode: str, dynamic: bool, fullgraph: bool):
    if not hasattr(torch, "compile"):
        return model
    prewarm = getattr(model, "prewarm_caches", None)
    if callable(prewarm):
        try:
            prewarm(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        except Exception:
            pass
    return torch.compile(model, mode=mode, dynamic=dynamic, fullgraph=fullgraph)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare pure-cartesian-ictd (SO3) vs pure-cartesian-ictd-o3 presets")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--nodes", type=int, default=128)
    p.add_argument("--avg-degree", type=int, default=16)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--include-full", action="store_true",
                   help="Also benchmark the full O(3) trunk. Default output compares SO(3), O(3)-auto, and O(3)-minimal.")
    p.add_argument("--compile", action="store_true", help="Apply torch.compile before benchmarking.")
    p.add_argument("--compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--compile-dynamic", action="store_true")
    p.add_argument("--compile-fullgraph", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    graph = _make_graph(args.nodes, args.avg_degree, device, dtype)

    common = dict(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=32,
        hidden_dim_sh=16,
        hidden_dim=32,
        channel_in2=32,
        embedding_dim=16,
        max_atomvalue=10,
        output_size=8,
        embed_size=[64, 64],
        main_hidden_sizes3=[64, 64],
        num_layers=1,
        num_interaction=2,
        lmax=2,
    )
    models = {
        "pure-cartesian-ictd": PureCartesianICTDSO3(**common).to(device=device, dtype=dtype),
        "pure-cartesian-ictd-o3-auto": PureCartesianICTDO3TransformerLayer(
            **common,
            o3_irrep_preset="auto",
        ).to(device=device, dtype=dtype),
        "pure-cartesian-ictd-o3-min": PureCartesianICTDO3TransformerLayer(
            **common,
            o3_irrep_preset="minimal",
        ).to(device=device, dtype=dtype),
    }
    if args.include_full:
        models["pure-cartesian-ictd-o3-full"] = PureCartesianICTDO3TransformerLayer(
            **common,
            o3_irrep_preset="full",
        ).to(device=device, dtype=dtype)

    for name, model in models.items():
        if args.compile:
            model = _maybe_compile(
                model,
                mode=args.compile_mode,
                dynamic=args.compile_dynamic,
                fullgraph=args.compile_fullgraph,
            )
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        fwd_ms = _bench(model, graph, repeat=args.repeat, do_backward=False)
        bwd_ms = _bench(model, graph, repeat=args.repeat, do_backward=True)
        print(f"{name:24s} params={n_params:10d}  fwd_ms={fwd_ms:8.3f}  fwd+bwd_ms={bwd_ms:8.3f}")


if __name__ == "__main__":
    main()
