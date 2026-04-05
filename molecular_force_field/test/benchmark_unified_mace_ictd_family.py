#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])
from e3nn import o3

from molecular_force_field.models.pure_cartesian_ictd_layers import (
    PureCartesianICTDTransformerLayer as PureCartesianICTDSave,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
    PureCartesianICTDTransformerLayer as PureCartesianICTD,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_o3 import (
    PureCartesianICTDO3TransformerLayer,
)
from molecular_force_field.models.cue_layers_channelwise import (
    E3_TransformerLayer_multi as CueLayer,
)

import mace.modules
from mace.modules import ICTDConfig


def make_graph(device, dtype, num_nodes: int, avg_degree: int, num_elements: int, seed: int):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    pos = torch.randn(num_nodes, 3, generator=g, device=device, dtype=dtype) * 2.0
    A = torch.randint(1, num_elements + 1, (num_nodes,), generator=g, device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    num_edges = num_nodes * avg_degree
    edge_dst = torch.randint(0, num_nodes, (num_edges,), generator=g, device=device)
    edge_src = torch.randint(0, num_nodes, (num_edges,), generator=g, device=device)
    edge_shifts = torch.zeros(num_edges, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(num_nodes, -1, -1)
    return pos, A, batch, edge_src, edge_dst, edge_shifts, cell


def build_models(device, dtype, *, channels: int, lmax: int, num_interaction: int, correction: int, num_elements: int):
    common_cfg = dict(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=channels,
        hidden_dim_sh=channels,
        hidden_dim=64,
        channel_in2=32,
        embedding_dim=16,
        max_atomvalue=max(10, num_elements + 1),
        output_size=8,
        num_interaction=num_interaction,
        lmax=lmax,
        ictd_tp_path_policy="full",
        internal_compute_dtype=dtype,
    )
    hidden_irreps = o3.Irreps(
        " + ".join(f"{channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1))
    )
    models = {
        "mace": mace.modules.MACE(
            r_max=5.0,
            num_bessel=8,
            num_polynomial_cutoff=5,
            max_ell=lmax,
            interaction_cls=mace.modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
            interaction_cls_first=mace.modules.interaction_classes["RealAgnosticInteractionBlock"],
            num_interactions=num_interaction,
            num_elements=num_elements,
            hidden_irreps=hidden_irreps,
            MLP_irreps=o3.Irreps("16x0e"),
            atomic_energies=np.zeros(num_elements),
            avg_num_neighbors=24.0,
            atomic_numbers=list(range(1, num_elements + 1)),
            correlation=correction,
            gate=torch.nn.functional.silu,
        ).to(device=device, dtype=dtype),
        "mace-ictd": mace.modules.MACE(
            r_max=5.0,
            num_bessel=8,
            num_polynomial_cutoff=5,
            max_ell=lmax,
            interaction_cls=mace.modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
            interaction_cls_first=mace.modules.interaction_classes["RealAgnosticInteractionBlock"],
            num_interactions=num_interaction,
            num_elements=num_elements,
            hidden_irreps=hidden_irreps,
            MLP_irreps=o3.Irreps("16x0e"),
            atomic_energies=np.zeros(num_elements),
            avg_num_neighbors=24.0,
            atomic_numbers=list(range(1, num_elements + 1)),
            correlation=correction,
            gate=torch.nn.functional.silu,
            ictd_config=ICTDConfig(
                enabled=True,
                internal_compute_dtype=dtype,
                tp_backend="auto",
                group="o3",
                channelwise_scalar_gates=True,
            ),
        ).to(device=device, dtype=dtype),
        "ictd": PureCartesianICTD(**common_cfg).to(device=device, dtype=dtype),
        "ictd-save-so3": PureCartesianICTDSave(**common_cfg).to(device=device, dtype=dtype),
        "ictd-save-o3-auto": PureCartesianICTDO3TransformerLayer(
            **common_cfg,
            o3_irrep_preset="auto",
        ).to(device=device, dtype=dtype),
        "ictd-save-o3-min": PureCartesianICTDO3TransformerLayer(
            **common_cfg,
            o3_irrep_preset="minimal",
        ).to(device=device, dtype=dtype),
        "cue": CueLayer(
            max_embed_radius=5.0,
            main_max_radius=5.0,
            main_number_of_basis=8,
            embedding_dim=16,
            max_atomvalue=max(10, num_elements + 1),
            output_size=8,
            embed_size=[128, 128, 128],
            main_hidden_sizes3=[64, 32],
            num_layers=1,
            num_interaction=num_interaction,
            irreps_input=" + ".join(
                f"{channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1)
            ),
            irreps_query=" + ".join(
                f"{channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1)
            ),
            irreps_key=" + ".join(
                f"{channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1)
            ),
            irreps_value=" + ".join(
                f"{channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1)
            ),
            irreps_output=" + ".join(
                f"{channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1)
            ),
            irreps_sh=o3.Irreps.spherical_harmonics(lmax=lmax),
            hidden_dim_sh=channels,
            hidden_dim=64,
            channel_in2=32,
        ).to(device=device, dtype=dtype),
    }
    return models


def to_mace_graph(graph, *, dtype, num_elements: int):
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
    num_nodes = pos.shape[0]
    node_attrs = torch.nn.functional.one_hot(A.long() - 1, num_classes=num_elements).to(dtype=dtype)
    return {
        "positions": pos,
        "node_attrs": node_attrs,
        "edge_index": torch.vstack([edge_src, edge_dst]),
        "shifts": edge_shifts,
        "unit_shifts": edge_shifts,
        "cell": cell,
        "batch": batch,
        "ptr": torch.tensor([0, num_nodes], device=pos.device, dtype=torch.long),
    }


def clone_graph(graph, *, for_mace: bool, dtype, num_elements: int, requires_grad: bool):
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
    p = pos.detach().clone().requires_grad_(requires_grad)
    if for_mace:
        return to_mace_graph((p, A, batch, edge_src, edge_dst, edge_shifts, cell), dtype=dtype, num_elements=num_elements)
    return (p, A, batch, edge_src, edge_dst, edge_shifts, cell)


def run_once(model, graph, *, is_mace: bool, do_backward: bool, dtype, num_elements: int):
    batch = clone_graph(graph, for_mace=is_mace, dtype=dtype, num_elements=num_elements, requires_grad=do_backward)
    model.zero_grad(set_to_none=True)
    out = model(batch, compute_force=False) if is_mace else model(*batch)
    loss = out["energy"].sum() if is_mace else out.sum()
    if do_backward:
        loss.backward()


def measure(model, graph, *, warmup: int, repeat: int, is_mace: bool, do_backward: bool, device, dtype, num_elements: int):
    for _ in range(warmup):
        run_once(model, graph, is_mace=is_mace, do_backward=do_backward, dtype=dtype, num_elements=num_elements)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once(model, graph, is_mace=is_mace, do_backward=do_backward, dtype=dtype, num_elements=num_elements)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    return statistics.median(times), peak_mb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--atoms", type=int, nargs="+", default=[512, 1024])
    p.add_argument("--avg-degree", type=int, default=24)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--lmax", type=int, default=2)
    p.add_argument("--num-interaction", type=int, default=2)
    p.add_argument("--correction", type=int, default=3)
    p.add_argument("--num-elements", type=int, default=5)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if device.type == "cuda" and dtype == torch.float32:
        torch.set_float32_matmul_precision("high")
    torch.set_default_dtype(dtype)

    for atoms in args.atoms:
        summary = {}
        for run_idx in range(args.runs):
            graph = make_graph(device, dtype, atoms, args.avg_degree, args.num_elements, seed=1234 + run_idx)
            models = build_models(
                device,
                dtype,
                channels=args.channels,
                lmax=args.lmax,
                num_interaction=args.num_interaction,
                correction=args.correction,
                num_elements=args.num_elements,
            )
            for name, model in models.items():
                is_mace = name in {"mace", "mace-ictd"}
                fwd_ms, fwd_peak = measure(
                    model, graph, warmup=args.warmup, repeat=args.repeat, is_mace=is_mace,
                    do_backward=False, device=device, dtype=dtype, num_elements=args.num_elements
                )
                train_ms, train_peak = measure(
                    model, graph, warmup=args.warmup, repeat=args.repeat, is_mace=is_mace,
                    do_backward=True, device=device, dtype=dtype, num_elements=args.num_elements
                )
                params = sum(p.numel() for p in model.parameters())
                print(
                    f"atoms={atoms} run={run_idx+1} variant={name} params={params} "
                    f"fwd_ms={fwd_ms:.3f} train_ms={train_ms:.3f} "
                    f"fwd_peak_mb={fwd_peak:.1f} train_peak_mb={train_peak:.1f}",
                    flush=True,
                )
                bucket = summary.setdefault(name, {"fwd": [], "train": [], "fwd_peak": [], "train_peak": [], "params": params})
                bucket["fwd"].append(fwd_ms)
                bucket["train"].append(train_ms)
                bucket["fwd_peak"].append(fwd_peak)
                bucket["train_peak"].append(train_peak)
                del model
            del graph
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"== medians atoms={atoms} ==", flush=True)
        for name, vals in summary.items():
            print(
                f"atoms={atoms} variant={name} runs={args.runs} params={vals['params']} "
                f"fwd_ms_med={statistics.median(vals['fwd']):.3f} "
                f"train_ms_med={statistics.median(vals['train']):.3f} "
                f"fwd_peak_mb_med={statistics.median(vals['fwd_peak']):.1f} "
                f"train_peak_mb_med={statistics.median(vals['train_peak']):.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
