#!/usr/bin/env python3
"""
Benchmark all tensor-product modes across lmax = 0..6.
`pure-cartesian` is capped to lmax <= 4 in this benchmark.

Note:
  `spherical-cue` uses `cue_layers.py` — a full cuEquivariance backend strictly
  equivalent to `e3nn_layers.py` (same FullTensorProduct + FullyConnectedTP arch).
  `spherical-save-cue` uses `cue_layers_channelwise.py` (channelwise TP arch).

Usage:
  python -m molecular_force_field.benchmark_all_tp_modes_l0_l6
  python -m molecular_force_field.benchmark_all_tp_modes_l0_l6 --device cuda --dtype float32
  python -m molecular_force_field.benchmark_all_tp_modes_l0_l6 --atoms 256 --warmup 2 --repeat 5
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from molecular_force_field.models import (
    CartesianTransformerLayer,
    CartesianTransformerLayerLoose,
    E3_TransformerLayer_multi,
    PureCartesianICTDTransformerLayer,
    PureCartesianSparseTransformerLayer,
    PureCartesianTransformerLayer,
)
from molecular_force_field.models.e3nn_layers_channelwise import (
    E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
    PureCartesianICTDTransformerLayer as PureCartesianICTDTransformerLayerFull,
)
from molecular_force_field.utils.config import ModelConfig


ALL_MODES: List[str] = [
    "spherical",
    "spherical-cue",
    "spherical-save",
    "spherical-save-cue",
    "partial-cartesian",
    "partial-cartesian-loose",
    "pure-cartesian",
    "pure-cartesian-sparse",
    "pure-cartesian-ictd",
    "pure-cartesian-ictd-save",
]
PURE_CARTESIAN_MAX_L = 4


def make_dummy_graph(
    device: torch.device,
    dtype: torch.dtype,
    num_nodes: int = 256,
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


def _common_kwargs(cfg: ModelConfig, num_interaction: int, device: torch.device) -> Dict:
    return dict(
        max_embed_radius=cfg.max_radius,
        main_max_radius=cfg.max_radius_main,
        main_number_of_basis=cfg.number_of_basis_main,
        hidden_dim_conv=cfg.channel_in,
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
        lmax=cfg.lmax,
        device=device,
    )


def build_layer(
    mode: str,
    cfg: ModelConfig,
    device: torch.device,
    dtype: torch.dtype,
    num_interaction: int,
    max_rank_other: int,
    k_policy: str,
    ictd_tp_path_policy: str,
    ictd_tp_max_rank_other: int | None,
) -> nn.Module:
    k = _common_kwargs(cfg, num_interaction, device)

    if mode == "pure-cartesian":
        layer = PureCartesianTransformerLayer(**k)
    elif mode == "pure-cartesian-ictd":
        layer = PureCartesianICTDTransformerLayerFull(
            **k,
            ictd_tp_path_policy=ictd_tp_path_policy,
            ictd_tp_max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=dtype,
        )
    elif mode == "pure-cartesian-ictd-save":
        layer = PureCartesianICTDTransformerLayer(
            **k,
            ictd_tp_path_policy=ictd_tp_path_policy,
            ictd_tp_max_rank_other=ictd_tp_max_rank_other,
            internal_compute_dtype=dtype,
        )
    elif mode == "pure-cartesian-sparse":
        layer = PureCartesianSparseTransformerLayer(
            **k,
            max_rank_other=max_rank_other,
            k_policy=k_policy,
        )
    elif mode == "partial-cartesian":
        layer = CartesianTransformerLayer(**k)
    elif mode == "partial-cartesian-loose":
        layer = CartesianTransformerLayerLoose(**k)
    elif mode == "spherical-save":
        layer = E3_TransformerLayer_multi_channelwise(
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
    elif mode == "spherical-cue":
        try:
            import cuequivariance_torch  # noqa: F401
            from molecular_force_field.models.cue_layers import (
                E3_TransformerLayer_multi as E3_TransformerLayer_multi_cue_full,
            )
        except Exception as e:
            raise RuntimeError(f"cuEquivariance unavailable: {e}") from e
        layer = E3_TransformerLayer_multi_cue_full(
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
    elif mode == "spherical-save-cue":
        try:
            import cuequivariance_torch  # noqa: F401
            from molecular_force_field.models.cue_layers_channelwise import (
                E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise_cue,
            )
        except Exception as e:
            raise RuntimeError(f"cuEquivariance unavailable: {e}") from e
        layer = E3_TransformerLayer_multi_channelwise_cue(
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
    else:  # spherical
        layer = E3_TransformerLayer_multi(
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
    return layer.to(device=device, dtype=dtype)


def time_inference(
    module: nn.Module,
    graph,
    device: torch.device,
    warmup: int = 2,
    repeat: int = 5,
) -> float:
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            _ = module(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / repeat


def time_forward_backward(
    module: nn.Module,
    graph,
    device: torch.device,
    warmup: int = 2,
    repeat: int = 5,
) -> float:
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
    pos = pos.detach().requires_grad_(True)
    module.eval()
    for _ in range(warmup):
        module.zero_grad(set_to_none=True)
        if pos.grad is not None:
            pos.grad = None
        out = module(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        loss = out.sum()
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        module.zero_grad(set_to_none=True)
        if pos.grad is not None:
            pos.grad = None
        out = module(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        loss = out.sum()
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / repeat


def _sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _has_spherical_stages(module: nn.Module) -> bool:
    needed = ["e3_conv_layers", "product_3", "product_5", "proj_total", "weighted_sum"]
    return all(hasattr(module, name) for name in needed)


def _forward_spherical_stages_once(module: nn.Module, graph, device: torch.device) -> Dict[str, float]:
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph

    sort_idx = torch.argsort(edge_dst)
    edge_src = edge_src[sort_idx]
    edge_dst = edge_dst[sort_idx]
    edge_shifts = edge_shifts[sort_idx]

    _sync_if_needed(device)
    t0 = time.perf_counter()
    features = []
    f_prev = module.e3_conv_layers[0](pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    features.append(f_prev)
    for conv in module.e3_conv_layers[1:]:
        f_prev = conv(f_prev, pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        features.append(f_prev)
    _sync_if_needed(device)
    t1 = time.perf_counter()

    f_combine = torch.cat(features, dim=-1)
    f_combine_product = module.product_3(f_combine, f_combine)
    _sync_if_needed(device)
    t2 = time.perf_counter()

    T = torch.cat(features + [f_combine_product], dim=-1)
    f2_product_5 = module.product_5(T, T)
    _sync_if_needed(device)
    t3 = time.perf_counter()

    product_proj = module.proj_total(f2_product_5)
    e_out = module.weighted_sum(product_proj)
    _atom_energies = e_out.sum(dim=-1, keepdim=True)
    _sync_if_needed(device)
    t4 = time.perf_counter()

    return {
        "conv_stack": (t1 - t0) * 1000.0,
        "product_3": (t2 - t1) * 1000.0,
        "product_5": (t3 - t2) * 1000.0,
        "readout": (t4 - t3) * 1000.0,
    }


def time_spherical_stage_breakdown(
    module: nn.Module,
    graph,
    device: torch.device,
    warmup: int = 2,
    repeat: int = 5,
) -> Dict[str, float]:
    if not _has_spherical_stages(module):
        raise RuntimeError("module does not expose spherical stage attributes")

    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = _forward_spherical_stages_once(module, graph, device)

        sums = {"conv_stack": 0.0, "product_3": 0.0, "product_5": 0.0, "readout": 0.0}
        for _ in range(repeat):
            part = _forward_spherical_stages_once(module, graph, device)
            for k in sums:
                sums[k] += part[k]

    return {k: v / repeat for k, v in sums.items()}


def main():
    parser = argparse.ArgumentParser(description="Benchmark all tensor-product modes for lmax=0..6")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--atoms", type=int, default=250)
    parser.add_argument("--avg-degree", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--num-interaction", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--max-rank-other", type=int, default=1)
    parser.add_argument("--k-policy", type=str, default="k0")
    parser.add_argument("--ictd-tp-path-policy", type=str, default="full", choices=["full", "max_rank_other"])
    parser.add_argument("--ictd-tp-max-rank-other", type=int, default=None)
    parser.add_argument("--modes", type=str, nargs="+", default=ALL_MODES)
    parser.add_argument("--with-backward", action="store_true", help="Also benchmark forward+backward.")
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="Profile spherical stage breakdown: conv stack / product_3 / product_5 / readout.",
    )
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device(args.device)

    print("=" * 110, flush=True)
    print(
        f"Benchmark all tensor-product modes | device={device} dtype={args.dtype} "
        f"atoms={args.atoms} avg_degree={args.avg_degree} warmup={args.warmup} repeat={args.repeat}",
        flush=True,
    )
    print("=" * 110, flush=True)

    graph = make_dummy_graph(device, dtype, num_nodes=args.atoms, avg_degree=args.avg_degree, seed=42)

    l_values = list(range(7))
    results: Dict[str, Dict[int, Tuple[float | None, str | None]]] = {
        m: {l: (None, None) for l in l_values} for m in args.modes
    }
    results_bw: Dict[str, Dict[int, Tuple[float | None, str | None]]] = {
        m: {l: (None, None) for l in l_values} for m in args.modes
    }
    stage_results: Dict[str, Dict[int, Tuple[Dict[str, float] | None, str | None]]] = {
        m: {l: (None, None) for l in l_values} for m in args.modes
    }

    for mode in args.modes:
        print(f"\n[Mode] {mode}", flush=True)
        if mode == "spherical-cue":
            print("  NOTE: spherical-cue uses cue_layers.py (strict equivalent of e3nn_layers.py).", flush=True)
        for lmax in l_values:
            if mode == "pure-cartesian" and lmax > PURE_CARTESIAN_MAX_L:
                msg = f"skipped (pure-cartesian capped at lmax<={PURE_CARTESIAN_MAX_L})"
                results[mode][lmax] = (None, msg)
                if args.profile_stages:
                    stage_results[mode][lmax] = (None, msg)
                print(f"  l={lmax}: SKIP ({msg})", flush=True)
                continue
            cfg = ModelConfig(dtype=dtype, lmax=lmax, irreps_output_conv_channels=args.channels)
            # avoid external file dependency
            cfg.atomic_energy_keys = torch.tensor([1, 6, 7, 8], dtype=torch.long)
            cfg.atomic_energy_values = torch.tensor(
                [-430.53299511, -821.03326787, -1488.18856918, -2044.3509823], dtype=dtype
            )
            try:
                layer = build_layer(
                    mode=mode,
                    cfg=cfg,
                    device=device,
                    dtype=dtype,
                    num_interaction=args.num_interaction,
                    max_rank_other=args.max_rank_other,
                    k_policy=args.k_policy,
                    ictd_tp_path_policy=args.ictd_tp_path_policy,
                    ictd_tp_max_rank_other=args.ictd_tp_max_rank_other,
                )
                t_ms = time_inference(layer, graph, device, warmup=args.warmup, repeat=args.repeat)
                results[mode][lmax] = (t_ms, None)
                if args.profile_stages:
                    stage_modes = {"spherical", "spherical-cue", "spherical-save", "spherical-save-cue"}
                    if mode in stage_modes:
                        stage_ms = time_spherical_stage_breakdown(
                            layer, graph, device, warmup=args.warmup, repeat=args.repeat
                        )
                        stage_results[mode][lmax] = (stage_ms, None)
                    else:
                        stage_results[mode][lmax] = (None, "stage profiling only supports spherical* modes")
                if args.with_backward:
                    t_bw_ms = time_forward_backward(layer, graph, device, warmup=args.warmup, repeat=args.repeat)
                    results_bw[mode][lmax] = (t_bw_ms, None)
                    print(f"  l={lmax}: fwd={t_ms:.3f} ms/call | fwd+bwd={t_bw_ms:.3f} ms/call", flush=True)
                else:
                    print(f"  l={lmax}: {t_ms:.3f} ms/call", flush=True)
                if args.profile_stages:
                    stage_ms, stage_err = stage_results[mode][lmax]
                    if stage_ms is not None:
                        total = sum(stage_ms.values())
                        print(
                            "       stages: "
                            f"conv={stage_ms['conv_stack']:.3f} | "
                            f"p3={stage_ms['product_3']:.3f} | "
                            f"p5={stage_ms['product_5']:.3f} | "
                            f"readout={stage_ms['readout']:.3f} | "
                            f"total={total:.3f} ms",
                            flush=True,
                        )
                    elif stage_err is not None:
                        print(f"       stages: N/A ({stage_err})", flush=True)
                del layer
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:  # keep whole benchmark running
                msg = str(e).replace("\n", " ")
                results[mode][lmax] = (None, msg)
                if args.with_backward:
                    results_bw[mode][lmax] = (None, msg)
                if args.profile_stages:
                    stage_results[mode][lmax] = (None, msg)
                print(f"  l={lmax}: FAIL ({msg[:120]})", flush=True)

    print("\n" + "=" * 110, flush=True)
    print("Summary: forward (ms/call; N/A means failed/unavailable)\n", flush=True)
    header = "mode".ljust(30) + "".join([f"l={l}".rjust(12) for l in l_values])
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for mode in args.modes:
        row = mode.ljust(30)
        for l in l_values:
            t_ms, _ = results[mode][l]
            cell = "N/A" if t_ms is None else f"{t_ms:.2f}"
            row += cell.rjust(12)
        print(row, flush=True)

    if args.with_backward:
        print("\n" + "=" * 110, flush=True)
        print("Summary: forward+backward (ms/call; N/A means failed/unavailable)\n", flush=True)
        print(header, flush=True)
        print("-" * len(header), flush=True)
        for mode in args.modes:
            row = mode.ljust(30)
            for l in l_values:
                t_ms, _ = results_bw[mode][l]
                cell = "N/A" if t_ms is None else f"{t_ms:.2f}"
                row += cell.rjust(12)
            print(row, flush=True)

    if args.profile_stages:
        print("\n" + "=" * 110, flush=True)
        print("Summary: stage breakdown (forward, ms/call)\n", flush=True)
        stage_header = (
            "mode".ljust(26)
            + "lmax".rjust(6)
            + "conv".rjust(12)
            + "product_3".rjust(12)
            + "product_5".rjust(12)
            + "readout".rjust(12)
            + "total".rjust(12)
        )
        print(stage_header, flush=True)
        print("-" * len(stage_header), flush=True)
        for mode in args.modes:
            for l in l_values:
                stage_ms, err = stage_results[mode][l]
                if stage_ms is None:
                    cell = "N/A"
                    print(mode.ljust(26) + str(l).rjust(6) + cell.rjust(60), flush=True)
                    continue
                total = sum(stage_ms.values())
                row = mode.ljust(26) + str(l).rjust(6)
                row += f"{stage_ms['conv_stack']:.2f}".rjust(12)
                row += f"{stage_ms['product_3']:.2f}".rjust(12)
                row += f"{stage_ms['product_5']:.2f}".rjust(12)
                row += f"{stage_ms['readout']:.2f}".rjust(12)
                row += f"{total:.2f}".rjust(12)
                print(row, flush=True)

    print("\nFailure details:", flush=True)
    for mode in args.modes:
        for l in l_values:
            _, err = results[mode][l]
            if err is not None:
                print(f"- {mode} @ l={l}: {err}", flush=True)
    if args.profile_stages:
        for mode in args.modes:
            for l in l_values:
                _, err = stage_results[mode][l]
                if err is not None and results[mode][l][1] is None:
                    print(f"- {mode} @ l={l} [stages]: {err}", flush=True)


if __name__ == "__main__":
    main()

