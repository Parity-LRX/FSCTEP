"""
Benchmark inference speed for a single configuration (e.g. 40 atoms, num_interaction=3, 64 channels, lmax=2).

Useful for comparing with MACE or other potentials. Example:
  python -m molecular_force_field.benchmark_inference --atoms 40 --num-interaction 3 --channels 64 --lmax 2
  python -m molecular_force_field.benchmark_inference --atoms 40 --device cuda --repeat 200
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

from molecular_force_field.models.pure_cartesian_ictd_layers import PureCartesianICTDTransformerLayer


def make_dummy_graph(
    device: torch.device,
    dtype: torch.dtype,
    num_nodes: int = 40,
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


def time_inference(
    module: nn.Module,
    pos,
    A,
    batch,
    edge_src,
    edge_dst,
    edge_shifts,
    cell,
    *,
    warmup: int = 15,
    repeat: int = 100,
) -> float:
    """Returns average time per forward (seconds)."""
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    if next(module.parameters()).is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            _ = module(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    if next(module.parameters()).is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference (ICTD mode) for given config.")
    parser.add_argument("--atoms", type=int, default=40, help="Number of atoms (default: 40)")
    parser.add_argument("--num-interaction", type=int, default=3, help="num_interaction (default: 3, MACE-style correction)")
    parser.add_argument("--channels", type=int, default=64, help="Channel size (hidden_dim_conv, default: 64)")
    parser.add_argument("--lmax", type=int, default=2, help="lmax (default: 2)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--warmup", type=int, default=15, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=100, help="Timed iterations")
    parser.add_argument("--avg-degree", type=int, default=24, help="Average neighbors per atom for dummy graph")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    cfg = dict(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=args.channels,
        hidden_dim_sh=32,
        hidden_dim=64,
        channel_in2=32,
        embedding_dim=16,
        max_atomvalue=10,
        output_size=8,
        num_interaction=args.num_interaction,
        lmax=args.lmax,
        ictd_tp_path_policy="full",
        ictd_tp_max_rank_other=None,
        internal_compute_dtype=torch.float64,
    )

    model = PureCartesianICTDTransformerLayer(**cfg).to(device=device, dtype=dtype)
    num_atoms = args.atoms
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = make_dummy_graph(
        device, dtype, num_nodes=num_atoms, avg_degree=args.avg_degree
    )

    t_s = time_inference(
        model, pos, A, batch, edge_src, edge_dst, edge_shifts, cell,
        warmup=args.warmup, repeat=args.repeat,
    )
    t_ms = t_s * 1000.0
    atoms_per_sec = num_atoms / t_s if t_s > 0 else 0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Config: {num_atoms} atoms, num_interaction={args.num_interaction}, channels={args.channels}, lmax={args.lmax}")
    print(f"Device: {args.device}, dtype: {args.dtype}")
    print(f"Parameters: {n_params:,}")
    print(f"Inference: {t_ms:.3f} ms/call  ({atoms_per_sec:.0f} atoms/s)")
    print(f"(MACE-style: correction={args.num_interaction}, 64ch, lmax=2 — run this script to compare with your MACE build)")


if __name__ == "__main__":
    main()
