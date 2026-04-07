#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])

ROOT = Path("/home/rebuild")
sys.path.insert(0, str(ROOT / "mace-ictd"))
sys.path.insert(0, str(ROOT / "nequip-ictd"))

from e3nn import o3

import mace.modules
from mace.modules import ICTDConfig

from molecular_force_field.models.pure_cartesian_ictd_layers import (
    PureCartesianICTDTransformerLayer as PureCartesianICTDSave,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
    PureCartesianICTDTransformerLayer as PureCartesianICTD,
)

from nequip.data import AtomicDataDict
from nequip.model import NequIPGNNModel
from nequip.nn._tp_scatter_base import TensorProductScatter
import nequip.utils.global_state as nequip_global_state


def _median(xs):
    return statistics.median(xs)


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


def build_common_cfg(channels: int, lmax: int, num_interaction: int, dtype, num_elements: int):
    return dict(
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


def build_ictd_models(device, dtype, *, channels: int, lmax: int, num_interaction: int, correlation: int, num_elements: int):
    common_cfg = build_common_cfg(channels, lmax, num_interaction, dtype, num_elements)
    hidden_irreps = o3.Irreps(
        " + ".join(f"{channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1))
    )
    return {
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
            correlation=correlation,
            gate=torch.nn.functional.silu,
        ).to(device=device, dtype=dtype),
        "ictd": PureCartesianICTD(**common_cfg).to(device=device, dtype=dtype),
        "ictd-save": PureCartesianICTDSave(**common_cfg).to(device=device, dtype=dtype),
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
            correlation=correlation,
            gate=torch.nn.functional.silu,
            ictd_config=ICTDConfig(
                enabled=True,
                internal_compute_dtype=dtype,
                tp_backend="auto",
                group="o3",
                channelwise_scalar_gates=True,
            ),
        ).to(device=device, dtype=dtype),
    }


def initialize_nequip_global_state():
    try:
        nequip_global_state.set_global_state()
    except Exception:
        # Current local e3nn/nequip mix rejects one optimization flag in set_global_state().
        # For benchmarking we only need the builder guard to be satisfied.
        nequip_global_state._GLOBAL_STATE_INITIALIZED = True
        nequip_global_state._latest_global_config[nequip_global_state.TF32_KEY] = False


def build_nequip_model(device, *, channels: int, lmax: int, num_elements: int, avg_degree: int, dtype_name: str, enable_ictd: bool):
    initialize_nequip_global_state()
    model = NequIPGNNModel(
        seed=123,
        model_dtype=dtype_name,
        type_names=[f"T{i}" for i in range(num_elements)],
        r_max=5.0,
        avg_num_neighbors=float(avg_degree),
        num_layers=2,
        l_max=lmax,
        parity=True,
        num_features=channels,
        radial_mlp_depth=1,
        radial_mlp_width=64,
        per_type_energy_shifts={f"T{i}": 0.0 for i in range(num_elements)},
        do_derivatives=False,
    ).to(device)
    if enable_ictd:
        model = TensorProductScatter.enable_ICTD(model)
    return model.to(device)


def to_nequip_data(graph, *, num_elements: int):
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
    del edge_shifts, cell
    num_nodes = pos.shape[0]
    return {
        AtomicDataDict.POSITIONS_KEY: pos.detach().clone(),
        AtomicDataDict.ATOM_TYPE_KEY: (A.long() - 1).view(num_nodes, 1),
        AtomicDataDict.EDGE_INDEX_KEY: torch.stack([edge_src, edge_dst], dim=0),
        AtomicDataDict.BATCH_KEY: batch.detach().clone(),
        AtomicDataDict.NUM_NODES_KEY: torch.tensor([num_nodes], device=pos.device, dtype=torch.long),
    }


def clone_nequip_data(data, *, requires_grad: bool):
    out = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            out[key] = value.detach().clone()
            if key == AtomicDataDict.POSITIONS_KEY and requires_grad:
                out[key].requires_grad_(requires_grad)
        else:
            out[key] = value
    return out


def run_once_generic(model, graph, *, is_mace: bool, dtype, num_elements: int, do_backward: bool):
    batch = clone_graph(graph, for_mace=is_mace, dtype=dtype, num_elements=num_elements, requires_grad=do_backward)
    model.zero_grad(set_to_none=True)
    out = model(batch, compute_force=False) if is_mace else model(*batch)
    loss = out["energy"].sum() if is_mace else out.sum()
    if do_backward:
        loss.backward()


def run_once_nequip(model, data, *, do_backward: bool):
    local = clone_nequip_data(data, requires_grad=do_backward)
    model.zero_grad(set_to_none=True)
    out = model(local)[AtomicDataDict.TOTAL_ENERGY_KEY]
    loss = out.sum()
    if do_backward:
        loss.backward()


def measure(run_fn, *, warmup: int, repeat: int, device):
    for _ in range(warmup):
        run_fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        run_fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    return _median(times), peak_mb


def benchmark_config(device, dtype, *, nodes: int, avg_degree: int, channels: int, lmax: int, num_interaction: int, correlation: int, num_elements: int, edge_basis: int, warmup: int, repeat: int, runs: int):
    summary = {}
    for run_idx in range(runs):
        seed = 1234 + run_idx
        graph = make_graph(device, dtype, nodes, avg_degree, num_elements, seed=seed)
        models = build_ictd_models(
            device, dtype, channels=channels, lmax=lmax, num_interaction=num_interaction,
            correlation=correlation, num_elements=num_elements
        )
        nequip_model = build_nequip_model(
            device, channels=channels, lmax=lmax,
            num_elements=num_elements, avg_degree=avg_degree,
            dtype_name="float32" if dtype == torch.float32 else "float64", enable_ictd=False,
        )
        nequip_ictd_model = build_nequip_model(
            device, channels=channels, lmax=lmax,
            num_elements=num_elements, avg_degree=avg_degree,
            dtype_name="float32" if dtype == torch.float32 else "float64", enable_ictd=True,
        )
        nequip_data = to_nequip_data(graph, num_elements=num_elements)
        models["nequip"] = nequip_model
        models["nequip-ictd"] = nequip_ictd_model

        for name, model in models.items():
            if name in {"nequip", "nequip-ictd"}:
                fwd_ms, fwd_peak = measure(
                    lambda: run_once_nequip(model, nequip_data, do_backward=False),
                    warmup=warmup, repeat=repeat, device=device,
                )
                train_ms, train_peak = measure(
                    lambda: run_once_nequip(model, nequip_data, do_backward=True),
                    warmup=warmup, repeat=repeat, device=device,
                )
            else:
                is_mace = name in {"mace", "mace-ictd"}
                fwd_ms, fwd_peak = measure(
                    lambda m=model, g=graph, im=is_mace: run_once_generic(
                        m, g, is_mace=im, dtype=dtype, num_elements=num_elements, do_backward=False
                    ),
                    warmup=warmup, repeat=repeat, device=device,
                )
                train_ms, train_peak = measure(
                    lambda m=model, g=graph, im=is_mace: run_once_generic(
                        m, g, is_mace=im, dtype=dtype, num_elements=num_elements, do_backward=True
                    ),
                    warmup=warmup, repeat=repeat, device=device,
                )
            params = sum(p.numel() for p in model.parameters())
            bucket = summary.setdefault(name, {"fwd": [], "train": [], "fwd_peak": [], "train_peak": [], "params": params})
            bucket["fwd"].append(fwd_ms)
            bucket["train"].append(train_ms)
            bucket["fwd_peak"].append(fwd_peak)
            bucket["train_peak"].append(train_peak)
            print(
                f"nodes={nodes} lmax={lmax} run={run_idx+1} variant={name} params={params} "
                f"fwd_ms={fwd_ms:.3f} train_ms={train_ms:.3f} "
                f"fwd_peak_mb={fwd_peak:.1f} train_peak_mb={train_peak:.1f}",
                flush=True,
            )

    result = []
    for name, bucket in summary.items():
        result.append(
            {
                "variant": name,
                "params": bucket["params"],
                "fwd_ms": _median(bucket["fwd"]),
                "train_ms": _median(bucket["train"]),
                "fwd_peak_mb": _median(bucket["fwd_peak"]),
                "train_peak_mb": _median(bucket["train_peak"]),
            }
        )
    return sorted(result, key=lambda x: x["train_ms"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--nodes", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--lmax", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--avg-degree", type=int, default=24)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--num-interaction", type=int, default=2)
    parser.add_argument("--correlation", type=int, default=3)
    parser.add_argument("--num-elements", type=int, default=10)
    parser.add_argument("--edge-basis", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if device.type == "cuda" and dtype == torch.float32:
        torch.set_float32_matmul_precision("high")
    torch.set_default_dtype(dtype)

    payload = {"config": vars(args), "results": []}
    for nodes in args.nodes:
        for lmax in args.lmax:
            ranked = benchmark_config(
                device,
                dtype,
                nodes=nodes,
                avg_degree=args.avg_degree,
                channels=args.channels,
                lmax=lmax,
                num_interaction=args.num_interaction,
                correlation=args.correlation,
                num_elements=args.num_elements,
                edge_basis=args.edge_basis,
                warmup=args.warmup,
                repeat=args.repeat,
                runs=args.runs,
            )
            payload["results"].append({"nodes": nodes, "lmax": lmax, "ranked_by_train": ranked})
            print(json.dumps({"nodes": nodes, "lmax": lmax, "ranked_by_train": ranked}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
