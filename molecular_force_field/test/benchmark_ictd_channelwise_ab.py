#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time

import torch

from molecular_force_field.models.ictd_irreps import (
    HarmonicChannelWiseTensorProduct,
    HarmonicChannelWiseTensorProductO3,
    canonical_irrep_parity_sign,
    direction_harmonics_all,
)
from molecular_force_field.models.pure_cartesian_ictd_layers import (
    ICTDIrrepsE3Conv as SaveConv1,
    PureCartesianICTDTransformerLayer as SaveModel,
    _merge_irreps as merge_so3_irreps,
    _split_irreps as split_so3_irreps,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
    ICTDIrrepsE3Conv as FullConv1,
    PureCartesianICTDTransformerLayer as FullModel,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_full_o3 import (
    PureCartesianICTDO3TransformerLayer as O3Model,
    _merge_irreps_o3 as merge_o3_irreps,
    _o3_active_irreps,
    _split_irreps_o3 as split_o3_irreps,
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


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size, src.shape[-1]))
    out.index_add_(0, index, src)
    return out


class SO3NextConv(torch.nn.Module):
    def __init__(self, *, channels: int, lmax: int, num_basis: int, divide_by_neighbors: bool):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.divide_by_neighbors = bool(divide_by_neighbors)
        self.tp2 = HarmonicChannelWiseTensorProduct(
            mul_in1=self.channels,
            mul_in2=1,
            mul_out=self.channels,
            lmax=self.lmax,
            internal_weights=True,
            internal_compute_dtype=torch.float32,
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(num_basis, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, self.tp2.num_paths),
        )

    def forward(self, pos, _A, _batch, edge_src, edge_dst, edge_shifts, cell, node_feat):
        edge_cells = cell[torch.zeros_like(edge_src)]
        shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
        edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)
        n = edge_vec / edge_length.clamp(min=1e-8).unsqueeze(-1)
        Y_list = direction_harmonics_all(n.to(dtype=node_feat.dtype), self.lmax)
        Y = {l: Y_list[l].to(dtype=node_feat.dtype).unsqueeze(-2) for l in range(self.lmax + 1)}
        emb = torch.nn.functional.silu(
            torch.nn.functional.one_hot(torch.clamp((edge_length / 5.0 * 7).long(), min=0, max=7), num_classes=8).to(node_feat.dtype)
        )
        gates = self.fc(emb)
        x1 = split_so3_irreps(node_feat, self.channels, self.lmax)
        x1e = {l: x1[l][edge_src] for l in range(self.lmax + 1)}
        edge_blocks = self.tp2(x1e, Y, gates)
        edge_flat = merge_so3_irreps(edge_blocks, self.channels, self.lmax)
        out = scatter_sum(edge_flat, edge_dst, pos.size(0))
        if self.divide_by_neighbors:
            neighbor_count = torch.zeros(pos.size(0), device=pos.device, dtype=edge_flat.dtype)
            neighbor_count.index_add_(0, edge_dst, torch.ones_like(edge_dst, dtype=edge_flat.dtype))
            out = out / neighbor_count.clamp(min=1).unsqueeze(-1)
        else:
            avg_num_neighbors = float(edge_src.numel()) / float(max(pos.size(0), 1))
            out = out / max(avg_num_neighbors, 1e-8)
        return out


class O3NextConv(torch.nn.Module):
    def __init__(self, *, channels: int, lmax: int, num_basis: int):
        super().__init__()
        self.channels = int(channels)
        self.lmax = int(lmax)
        self.active_irreps = _o3_active_irreps(self.lmax)
        self.tp2 = HarmonicChannelWiseTensorProductO3(
            mul_in1=self.channels,
            mul_in2=1,
            mul_out=self.channels,
            lmax=self.lmax,
            active_irreps=self.active_irreps,
            internal_weights=True,
            internal_compute_dtype=torch.float32,
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(num_basis, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, self.tp2.num_paths),
        )

    def forward(self, pos, _A, _batch, edge_src, edge_dst, edge_shifts, cell, node_feat):
        edge_cells = cell[torch.zeros_like(edge_src)]
        shift_vecs = torch.einsum("ni,nij->nj", edge_shifts, edge_cells)
        edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs
        edge_length = edge_vec.norm(dim=1)
        n = edge_vec / edge_length.clamp(min=1e-8).unsqueeze(-1)
        Y_list = direction_harmonics_all(n.to(dtype=node_feat.dtype), self.lmax)
        Y = {}
        for l in range(self.lmax + 1):
            key = (l, canonical_irrep_parity_sign(l))
            Y[key] = Y_list[l].to(dtype=node_feat.dtype).unsqueeze(-2)
        emb = torch.nn.functional.silu(
            torch.nn.functional.one_hot(torch.clamp((edge_length / 5.0 * 7).long(), min=0, max=7), num_classes=8).to(node_feat.dtype)
        )
        gates = self.fc(emb)
        x1 = split_o3_irreps(node_feat, self.channels, self.active_irreps)
        x1e = {k: v[edge_src] for k, v in x1.items()}
        edge_blocks = self.tp2(x1e, Y, gates)
        edge_flat = merge_o3_irreps(edge_blocks, self.channels, self.active_irreps)
        neighbor_count = torch.zeros(pos.size(0), device=pos.device, dtype=edge_flat.dtype)
        neighbor_count.index_add_(0, edge_dst, torch.ones_like(edge_dst, dtype=edge_flat.dtype))
        return scatter_sum(edge_flat, edge_dst, pos.size(0)) / neighbor_count.clamp(min=1).unsqueeze(-1)


def bench(fn, *, warmup: int, repeat: int, do_backward: bool, device: torch.device):
    for _ in range(warmup):
        out = fn(do_backward)
        if do_backward:
            out.sum().backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(do_backward)
        if do_backward:
            out.sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    return statistics.median(times), peak_mb


def model_factory(kind: str, *, device: torch.device, dtype: torch.dtype, channels: int, lmax: int, num_interaction: int):
    common = dict(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=channels,
        hidden_dim_sh=channels,
        hidden_dim=64,
        channel_in2=32,
        embedding_dim=16,
        max_atomvalue=10,
        output_size=8,
        embed_size=[128, 128, 128],
        main_hidden_sizes3=[64, 32],
        num_layers=1,
        num_interaction=num_interaction,
        lmax=lmax,
        internal_compute_dtype=dtype,
    )
    if kind == "ictd":
        return FullModel(**common).to(device=device, dtype=dtype)
    if kind == "ictd-save-so3":
        return SaveModel(**common).to(device=device, dtype=dtype)
    if kind == "ictd-save-o3-auto":
        return O3Model(**common, o3_irrep_preset="auto").to(device=device, dtype=dtype)
    raise ValueError(kind)


def run_model(model, graph, do_backward: bool):
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
    p = pos.detach().clone().requires_grad_(do_backward)
    model.zero_grad(set_to_none=True)
    return model(p, A, batch, edge_src, edge_dst, edge_shifts, cell)


def make_layer_node_feat(kind: str, *, num_nodes: int, channels: int, lmax: int, device: torch.device, dtype: torch.dtype):
    if kind == "o3":
        active_irreps = _o3_active_irreps(lmax)
        total_dim = channels * sum(2 * l + 1 for l, _ in active_irreps)
    else:
        total_dim = channels * (lmax + 1) ** 2
    return torch.randn(num_nodes, total_dim, device=device, dtype=dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--nodes", type=int, nargs="+", default=[512, 1024])
    p.add_argument("--avg-degree", type=int, default=24)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--lmax", type=int, default=2)
    p.add_argument("--num-interaction", type=int, default=2)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if device.type == "cuda" and dtype == torch.float32:
        torch.set_float32_matmul_precision("high")
    torch.set_default_dtype(dtype)

    for nodes in args.nodes:
        print(f"== layer atoms={nodes} ==", flush=True)
        layer_summary = {}
        for run_idx in range(args.runs):
            torch.manual_seed(1000 + run_idx)
            graph = make_graph(nodes, args.avg_degree, device, dtype)
            so3_feat = make_layer_node_feat("so3", num_nodes=nodes, channels=args.channels, lmax=args.lmax, device=device, dtype=dtype)
            o3_feat = make_layer_node_feat("o3", num_nodes=nodes, channels=args.channels, lmax=args.lmax, device=device, dtype=dtype)
            variants = {
                "ictd-conv1": FullConv1(max_radius=5.0, number_of_basis=8, channels_out=args.channels, output_size=8, lmax=args.lmax, internal_compute_dtype=dtype).to(device=device, dtype=dtype),
                "ictd-convnext": SO3NextConv(channels=args.channels, lmax=args.lmax, num_basis=8, divide_by_neighbors=True).to(device=device, dtype=dtype),
                "ictd-save-conv1": SaveConv1(max_radius=5.0, number_of_basis=8, channels_out=args.channels, output_size=8, lmax=args.lmax, internal_compute_dtype=dtype).to(device=device, dtype=dtype),
                "ictd-save-convnext": SO3NextConv(channels=args.channels, lmax=args.lmax, num_basis=8, divide_by_neighbors=False).to(device=device, dtype=dtype),
                "ictd-save-o3-convnext": O3NextConv(channels=args.channels, lmax=args.lmax, num_basis=8).to(device=device, dtype=dtype),
            }
            for name, mod in variants.items():
                mod.train()
                if "convnext" in name:
                    node_feat = o3_feat if "o3" in name else so3_feat
                    def runner(do_backward, m=mod, nf=node_feat):
                        m.zero_grad(set_to_none=True)
                        return m(
                            graph[0].detach().clone().requires_grad_(do_backward),
                            graph[1],
                            graph[2],
                            graph[3],
                            graph[4],
                            graph[5],
                            graph[6],
                            nf.detach().clone(),
                        )
                else:
                    def runner(do_backward, m=mod):
                        m.zero_grad(set_to_none=True)
                        return m(
                            graph[0].detach().clone().requires_grad_(do_backward),
                            graph[1],
                            graph[2],
                            graph[3],
                            graph[4],
                            graph[5],
                            graph[6],
                        )
                fwd_ms, fwd_peak = bench(runner, warmup=args.warmup, repeat=args.repeat, do_backward=False, device=device)
                train_ms, train_peak = bench(runner, warmup=args.warmup, repeat=args.repeat, do_backward=True, device=device)
                params = sum(p_.numel() for p_ in mod.parameters())
                bucket = layer_summary.setdefault(name, {"fwd": [], "train": [], "fwd_peak": [], "train_peak": [], "params": params})
                bucket["fwd"].append(fwd_ms)
                bucket["train"].append(train_ms)
                bucket["fwd_peak"].append(fwd_peak)
                bucket["train_peak"].append(train_peak)
                print(f"atoms={nodes} run={run_idx+1} layer={name} params={params} fwd_ms={fwd_ms:.3f} train_ms={train_ms:.3f} fwd_peak_mb={fwd_peak:.1f} train_peak_mb={train_peak:.1f}", flush=True)
                del mod
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print("-- layer medians --", flush=True)
        for name, vals in sorted(layer_summary.items()):
            print(
                f"atoms={nodes} layer={name} params={vals['params']} "
                f"fwd_ms_med={statistics.median(vals['fwd']):.3f} train_ms_med={statistics.median(vals['train']):.3f} "
                f"fwd_peak_mb_med={statistics.median(vals['fwd_peak']):.1f} train_peak_mb_med={statistics.median(vals['train_peak']):.1f}",
                flush=True,
            )

        print(f"== models atoms={nodes} ==", flush=True)
        model_summary = {}
        for run_idx in range(args.runs):
            torch.manual_seed(2000 + run_idx)
            graph = make_graph(nodes, args.avg_degree, device, dtype)
            for kind in ("ictd", "ictd-save-so3", "ictd-save-o3-auto"):
                name = kind
                model = model_factory(
                    kind,
                    device=device,
                    dtype=dtype,
                    channels=args.channels,
                    lmax=args.lmax,
                    num_interaction=args.num_interaction,
                )
                fwd_ms, fwd_peak = bench(
                    lambda do_backward, m=model: run_model(m, graph, do_backward),
                    warmup=args.warmup,
                    repeat=args.repeat,
                    do_backward=False,
                    device=device,
                )
                train_ms, train_peak = bench(
                    lambda do_backward, m=model: run_model(m, graph, do_backward),
                    warmup=args.warmup,
                    repeat=args.repeat,
                    do_backward=True,
                    device=device,
                )
                params = sum(p_.numel() for p_ in model.parameters())
                bucket = model_summary.setdefault(name, {"fwd": [], "train": [], "fwd_peak": [], "train_peak": [], "params": params})
                bucket["fwd"].append(fwd_ms)
                bucket["train"].append(train_ms)
                bucket["fwd_peak"].append(fwd_peak)
                bucket["train_peak"].append(train_peak)
                print(f"atoms={nodes} run={run_idx+1} model={name} params={params} fwd_ms={fwd_ms:.3f} train_ms={train_ms:.3f} fwd_peak_mb={fwd_peak:.1f} train_peak_mb={train_peak:.1f}", flush=True)
                del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print("-- model medians --", flush=True)
        for name, vals in sorted(model_summary.items()):
            print(
                f"atoms={nodes} model={name} params={vals['params']} "
                f"fwd_ms_med={statistics.median(vals['fwd']):.3f} train_ms_med={statistics.median(vals['train']):.3f} "
                f"fwd_peak_mb_med={statistics.median(vals['fwd_peak']):.1f} train_peak_mb_med={statistics.median(vals['train_peak']):.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
