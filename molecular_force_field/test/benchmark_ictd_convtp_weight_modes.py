#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import statistics
import time

import torch
from torch import nn

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])

from molecular_force_field.models.pure_cartesian_ictd_layers import (
    ICTDIrrepsE3Conv,
    direction_harmonics_all,
    soft_one_hot_linspace,
    _merge_irreps,
)
from molecular_force_field.models.ictd_irreps import HarmonicFullyConnectedTensorProduct


class ICTDIrrepsE3ConvFullExternal(nn.Module):
    def __init__(
        self,
        max_radius: float,
        number_of_basis: int,
        channels_out: int,
        embedding_dim: int = 16,
        max_atomvalue: int = 10,
        output_size: int = 8,
        lmax: int = 2,
        function_type: str = "gaussian",
        ictd_tp_path_policy: str = "full",
        ictd_tp_max_rank_other: int | None = None,
        avg_num_neighbors: float | None = None,
        internal_compute_dtype: torch.dtype | None = None,
        ictd_tp_backend: str = "auto",
    ):
        super().__init__()
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.channels_out = channels_out
        self.output_size = output_size
        self.lmax = lmax
        self.function_type = function_type
        self.avg_num_neighbors = avg_num_neighbors

        self.atom_embedding = nn.Embedding(max_atomvalue, embedding_dim)
        self.atom_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, output_size),
        )
        self.tp2 = HarmonicFullyConnectedTensorProduct(
            mul_in1=output_size,
            mul_in2=1,
            mul_out=channels_out,
            lmax=lmax,
            internal_weights=False,
            path_policy=ictd_tp_path_policy,
            max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=internal_compute_dtype,
            ictd_tp_backend=ictd_tp_backend,
        )
        self.fc = nn.Sequential(
            nn.Linear(number_of_basis, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, self.tp2.weight_numel),
        )
        self.output_dim = sum(channels_out * (2 * l + 1) for l in range(lmax + 1))

    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, *, precomputed_n=None, precomputed_edge_length=None, precomputed_Y_list=None):
        dtype = next(self.parameters()).dtype
        pos = pos.to(dtype=dtype)
        cell = cell.to(dtype=dtype)
        edge_shifts = edge_shifts.to(dtype=dtype)

        if precomputed_n is None or precomputed_edge_length is None:
            edge_batch_idx = batch[edge_src]
            edge_cells = cell[edge_batch_idx]
            shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
            edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
            edge_length = edge_vec.norm(dim=1)
            n = edge_vec / edge_length.clamp(min=1e-8).unsqueeze(-1)
        else:
            n = precomputed_n
            edge_length = precomputed_edge_length

        Ai = self.atom_mlp(self.atom_embedding(A.long()))
        n = n.to(dtype=Ai.dtype)
        edge_length = edge_length.to(dtype=Ai.dtype)
        if precomputed_Y_list is None:
            Y_list = direction_harmonics_all(n, self.lmax)
        else:
            Y_list = precomputed_Y_list

        f_in = {l: Ai[edge_src].unsqueeze(-1) * Y_list[l].unsqueeze(-2) for l in range(self.lmax + 1)}
        x2 = {l: Y_list[l].unsqueeze(-2) for l in range(self.lmax + 1)}
        emb = soft_one_hot_linspace(edge_length, 0.0, self.max_radius, self.number_of_basis, basis=self.function_type, cutoff=True)
        emb = emb.mul(self.number_of_basis ** 0.5).to(dtype=Ai.dtype)
        weights = self.fc(emb)
        out_blocks = self.tp2(f_in, x2, weights)
        edge_features = _merge_irreps(out_blocks, self.channels_out, self.lmax)

        num_nodes = pos.size(0)
        out = torch.zeros(num_nodes, edge_features.shape[-1], device=edge_features.device, dtype=edge_features.dtype)
        out.index_add_(0, edge_dst, edge_features)
        avg = self.avg_num_neighbors if self.avg_num_neighbors is not None else (float(edge_src.numel()) / float(max(num_nodes, 1)))
        return out / max(avg, 1e-8)


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def clone_module_weights(src: nn.Module, dst: nn.Module):
    dst.atom_embedding.load_state_dict(copy.deepcopy(src.atom_embedding.state_dict()))
    dst.atom_mlp.load_state_dict(copy.deepcopy(src.atom_mlp.state_dict()))
    src_fc = src.fc.state_dict()
    dst_fc = dst.fc.state_dict()
    for key in dst_fc:
        if key in src_fc and dst_fc[key].shape == src_fc[key].shape:
            dst_fc[key] = copy.deepcopy(src_fc[key])
    dst.fc.load_state_dict(dst_fc)


def make_graph(num_nodes: int, avg_degree: int, max_atomvalue: int, max_radius: float, device: torch.device, dtype: torch.dtype, lmax: int):
    num_edges = num_nodes * avg_degree
    edge_src = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    pos = torch.randn(num_nodes, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    edge_shifts = torch.zeros(num_edges, 3, device=device, dtype=dtype)
    A = torch.randint(0, max_atomvalue, (num_nodes,), device=device)

    n = torch.randn(num_edges, 3, device=device, dtype=dtype)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    edge_length = torch.rand(num_edges, device=device, dtype=dtype) * (max_radius - 0.5) + 0.5
    Y_list = direction_harmonics_all(n, lmax)
    return {
        "pos": pos,
        "A": A,
        "batch": batch,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_shifts": edge_shifts,
        "cell": cell,
        "precomputed_n": n,
        "precomputed_edge_length": edge_length,
        "precomputed_Y_list": Y_list,
    }


def zero_grads(module: nn.Module):
    module.zero_grad(set_to_none=True)
    for p in module.parameters():
        if p.grad is not None:
            p.grad = None


def measure(fn, warmup: int, repeat: int, device: torch.device):
    for _ in range(warmup):
        fn()
    sync(device)
    reset_peak(device)
    vals = []
    for _ in range(repeat):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        vals.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(vals), peak_mb(device)


def bench_model(model: nn.Module, graph: dict, mode: str, warmup: int, repeat: int, device: torch.device):
    if mode == "fwd":
        def run():
            zero_grads(model)
            with torch.no_grad():
                _ = model(**graph)
        return measure(run, warmup, repeat, device)

    def run():
        zero_grads(model)
        out = model(**graph)
        loss = out.square().mean()
        loss.backward()
    return measure(run, warmup, repeat, device)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--nodes", type=int, default=1024)
    p.add_argument("--avg-degree", type=int, default=24)
    p.add_argument("--channels-out", type=int, default=64)
    p.add_argument("--output-size", type=int, default=8)
    p.add_argument("--lmax", type=int, default=2)
    p.add_argument("--max-radius", type=float, default=5.0)
    p.add_argument("--number-of-basis", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=5)
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if device.type == "cuda" and dtype == torch.float32:
        torch.set_float32_matmul_precision("high")

    scalar = ICTDIrrepsE3Conv(
        max_radius=args.max_radius,
        number_of_basis=args.number_of_basis,
        channels_out=args.channels_out,
        output_size=args.output_size,
        lmax=args.lmax,
        max_atomvalue=10,
        internal_compute_dtype=dtype,
    ).to(device=device, dtype=dtype)
    full = ICTDIrrepsE3ConvFullExternal(
        max_radius=args.max_radius,
        number_of_basis=args.number_of_basis,
        channels_out=args.channels_out,
        output_size=args.output_size,
        lmax=args.lmax,
        max_atomvalue=10,
        internal_compute_dtype=dtype,
    ).to(device=device, dtype=dtype)
    clone_module_weights(scalar, full)

    graph = make_graph(args.nodes, args.avg_degree, 10, args.max_radius, device, dtype, args.lmax)

    results = []
    for name, model in (("scalar_gates", scalar), ("full_external", full)):
        fwd_ms, fwd_mb = bench_model(model, graph, "fwd", args.warmup, args.repeat, device)
        train_ms, train_mb = bench_model(model, graph, "train", args.warmup, args.repeat, device)
        row = {
            "name": name,
            "nodes": args.nodes,
            "avg_degree": args.avg_degree,
            "lmax": args.lmax,
            "channels_out": args.channels_out,
            "output_size": args.output_size,
            "params": count_params(model),
            "tp_num_paths": int(model.tp2.num_paths),
            "tp_weight_numel": int(model.tp2.weight_numel),
            "fc_out_dim": int(model.fc[-1].out_features),
            "fwd_ms": fwd_ms,
            "fwd_peak_mb": fwd_mb,
            "train_ms": train_ms,
            "train_peak_mb": train_mb,
        }
        results.append(row)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
