#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import torch

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])


def make_dummy_graph(
    device,
    dtype,
    num_nodes: int,
    avg_degree: int = 24,
    seed: int = 42,
    num_elements: int = 5,
):
    torch.manual_seed(seed)
    pos = torch.randn(num_nodes, 3, device=device, dtype=dtype) * 2.0
    A = torch.randint(1, num_elements + 1, (num_nodes,), device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    num_edges = num_nodes * avg_degree
    edge_dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_src = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_shifts = torch.zeros(num_edges, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(num_nodes, -1, -1)
    return pos, A, batch, edge_src, edge_dst, edge_shifts, cell


def measure_train_ms(layer, graph, warmup: int, repeat: int, device, is_mace: bool, num_elements: int):
    if is_mace:
        pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
        pos = pos.detach().clone().requires_grad_(True)
        num_nodes = pos.shape[0]
        node_attrs = torch.nn.functional.one_hot(A.long() - 1, num_classes=num_elements).to(dtype=pos.dtype)
        mace_data = {
            "positions": pos,
            "node_attrs": node_attrs,
            "edge_index": torch.vstack([edge_src, edge_dst]),
            "shifts": edge_shifts,
            "unit_shifts": edge_shifts,
            "cell": cell,
            "batch": batch,
            "ptr": torch.tensor([0, num_nodes], device=device, dtype=torch.long),
        }
        graph = (mace_data,)
    else:
        pos = graph[0].detach().clone().requires_grad_(True)
        graph = (pos,) + tuple(graph[1:])

    layer.train()
    for _ in range(warmup):
        layer.zero_grad(set_to_none=True)
        out = layer(*graph, compute_force=False) if is_mace else layer(*graph)
        loss = out["energy"].sum() if is_mace else out.sum()
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        layer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = layer(*graph, compute_force=False) if is_mace else layer(*graph)
        loss = out["energy"].sum() if is_mace else out.sum()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times)


def build_model(args, device, dtype):
    common_cfg = dict(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=args.channels,
        hidden_dim_sh=args.channels,
        hidden_dim=64,
        channel_in2=32,
        embedding_dim=16,
        # Dummy graphs use 1-based atomic labels in [1, num_elements].
        # Embedding tables therefore need at least num_elements + 1 slots.
        max_atomvalue=max(10, args.num_elements + 1),
        output_size=8,
        num_interaction=args.num_interaction,
        lmax=args.lmax,
        ictd_tp_path_policy="full",
        internal_compute_dtype=dtype,
    )

    if args.model == "mace":
        import numpy as np
        import e3nn.o3 as o3
        import mace.modules

        return mace.modules.MACE(
            r_max=5.0,
            num_bessel=8,
            num_polynomial_cutoff=5,
            max_ell=args.lmax,
            interaction_cls=mace.modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
            interaction_cls_first=mace.modules.interaction_classes["RealAgnosticInteractionBlock"],
            num_interactions=args.num_interaction,
            num_elements=args.num_elements,
            hidden_irreps=o3.Irreps(
                " + ".join(f"{args.channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(args.lmax + 1))
            ),
            MLP_irreps=o3.Irreps("16x0e"),
            atomic_energies=np.zeros(args.num_elements),
            avg_num_neighbors=args.avg_degree,
            atomic_numbers=list(range(1, args.num_elements + 1)),
            correlation=args.correction,
            gate=torch.nn.functional.silu,
        ).to(device=device, dtype=dtype)

    if args.model == "ictd":
        from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
            PureCartesianICTDTransformerLayer,
        )

        return PureCartesianICTDTransformerLayer(**common_cfg).to(device=device, dtype=dtype)

    if args.model == "ictd-save":
        from molecular_force_field.models.pure_cartesian_ictd_layers import (
            PureCartesianICTDTransformerLayer,
        )

        return PureCartesianICTDTransformerLayer(**common_cfg).to(device=device, dtype=dtype)

    if args.model in {"cue", "e3nn", "e3nn-save"}:
        import e3nn.o3 as o3
        if args.model == "cue":
            from molecular_force_field.models.cue_layers_channelwise import (
                E3_TransformerLayer_multi,
            )
        elif args.model == "e3nn":
            from molecular_force_field.models.e3nn_layers import (
                E3_TransformerLayer_multi,
            )
        else:
            from molecular_force_field.models.e3nn_layers_channelwise import (
                E3_TransformerLayer_multi,
            )

        cue_cfg = dict(common_cfg)
        irreps_in = " + ".join(
            f"{args.channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(args.lmax + 1)
        )
        cue_cfg.update(
            {
                "irreps_input": irreps_in,
                "irreps_query": irreps_in,
                "irreps_key": irreps_in,
                "irreps_value": irreps_in,
                "irreps_output": irreps_in,
                "irreps_sh": o3.Irreps.spherical_harmonics(lmax=args.lmax),
                "hidden_dim_sh": args.channels,
                "hidden_dim": 64,
                "channel_in2": 32,
            }
        )
        cue_cfg.pop("hidden_dim_conv", None)
        cue_cfg.pop("lmax", None)
        cue_cfg.pop("ictd_tp_path_policy", None)
        cue_cfg.pop("internal_compute_dtype", None)
        return E3_TransformerLayer_multi(**cue_cfg).to(device=device, dtype=dtype)

    raise ValueError(f"unknown model: {args.model}")


def main():
    parser = argparse.ArgumentParser(description="Fixed-size training speed compare benchmark")
    parser.add_argument(
        "--model",
        choices=["mace", "ictd", "ictd-save", "cue", "e3nn", "e3nn-save"],
        required=True,
    )
    parser.add_argument("--atoms", type=int, default=512)
    parser.add_argument("--avg-degree", type=int, default=24)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--num-interaction", type=int, default=2)
    parser.add_argument("--correction", type=int, default=3)
    parser.add_argument("--num-elements", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    if device.type == "cuda" and dtype == torch.float32:
        torch.set_float32_matmul_precision("high")

    layer = build_model(args, device, dtype)
    graph = make_dummy_graph(
        device,
        dtype,
        num_nodes=args.atoms,
        avg_degree=args.avg_degree,
        num_elements=args.num_elements,
    )
    t_ms = measure_train_ms(
        layer,
        graph,
        args.warmup,
        args.repeat,
        device,
        is_mace=(args.model == "mace"),
        num_elements=args.num_elements,
    )
    steps_per_s = 1000.0 / t_ms
    atom_steps_per_s = args.atoms * steps_per_s
    print(f"model={args.model}")
    print(f"atoms={args.atoms}")
    print(f"avg_degree={args.avg_degree}")
    print(f"channels={args.channels}")
    print(f"lmax={args.lmax}")
    print(f"num_interaction={args.num_interaction}")
    print(f"correction={args.correction}")
    print(f"num_elements={args.num_elements}")
    print(f"dtype={args.dtype}")
    print(f"train_ms={t_ms:.3f}")
    print(f"steps_per_s={steps_per_s:.3f}")
    print(f"atom_steps_per_s={atom_steps_per_s:.3f}")


if __name__ == "__main__":
    main()
