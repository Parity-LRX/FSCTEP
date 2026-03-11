#!/usr/bin/env python3
"""
Find maximum number of atoms per structure such that inference time <= 1 second.

Default config: lmax=2, channels=64, num_interaction=2, float32.
Uses PureCartesianICTDTransformerLayer; optional torch.compile to match production.

Usage:
  python -m molecular_force_field.benchmark_max_atoms_1s
  python -m molecular_force_field.benchmark_max_atoms_1s --target-ms 2000 --compile
"""

from __future__ import annotations

import argparse
import time

import torch

# Fix PyTorch 2.6 weights_only=True issue with e3nn constants.pt which uses slice
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([slice])

from molecular_force_field.models import (
    E3_TransformerLayer_multi,
    CartesianTransformerLayer,
    CartesianTransformerLayerLoose,
    PureCartesianTransformerLayer,
    PureCartesianSparseTransformerLayer,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
    PureCartesianICTDTransformerLayer as PureCartesianICTDTransformerLayerFull,
)
from molecular_force_field.models.pure_cartesian_ictd_layers import (
    PureCartesianICTDTransformerLayer as PureCartesianICTDSave,
)
from molecular_force_field.models.e3nn_layers_channelwise import (
    E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise,
)


def make_dummy_graph(device, dtype, num_nodes: int, avg_degree: int = 24, seed: int = 42):
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


def measure_inference_ms(layer, graph, warmup: int, repeat: int, device, is_mace=False, backward=False):
    if is_mace:
        pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
        if backward:
            pos.requires_grad_(True)
        num_nodes = pos.shape[0]
        import e3nn.o3 as o3
        mace_data = {
            'positions': pos,
            'node_attrs': torch.ones(num_nodes, 1, device=device, dtype=pos.dtype),
            'edge_index': torch.vstack([edge_src, edge_dst]),
            'shifts': edge_shifts,
            'unit_shifts': edge_shifts,
            'cell': cell,
            'batch': batch,
            'ptr': torch.tensor([0, num_nodes], device=device, dtype=torch.long)
        }
        graph = (mace_data,)
    else:
        if backward:
            graph[0].requires_grad_(True)

    if backward:
        layer.train()
    else:
        layer.eval()

    context = torch.enable_grad() if backward else torch.no_grad()
    with context:
        for _ in range(warmup):
            if backward:
                layer.zero_grad(set_to_none=True)
            out = (layer(*graph, compute_force=False) if is_mace else layer(*graph))
            if backward:
                loss = out['energy'].sum() if is_mace else out.sum()
                loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(repeat):
            if backward:
                layer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = (layer(*graph, compute_force=False) if is_mace else layer(*graph))
            if backward:
                loss = out['energy'].sum() if is_mace else out.sum()
                loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser(
        description="Find max atoms per structure such that inference <= target ms"
    )
    parser.add_argument("--target-ms", type=float, default=1000.0,
                        help="Target max inference time in ms (default: 1000 = 1s)")
    parser.add_argument("--lmax", type=int, default=2, help="lmax (default: 2)")
    parser.add_argument("--channels", type=int, default=64,
                        help="hidden_dim_conv / channels (default: 64)")
    parser.add_argument("--num-interaction", type=int, default=2, help="num_interaction (default: 2)")
    parser.add_argument("--avg-degree", type=int, default=24,
                        help="Average edges per atom (default: 24)")
    parser.add_argument("--model", type=str, default="pure-cartesian-ictd-save", choices=['spherical', 'spherical-cue', 'spherical-save', 'spherical-save-cue', 'partial-cartesian', 'partial-cartesian-loose', 'pure-cartesian', 'pure-cartesian-sparse', 'pure-cartesian-ictd', 'pure-cartesian-ictd-save', 'mace'], help="Select the model architecture to benchmark.")
    parser.add_argument("--correction", type=int, default=3, help="MACE correlation / many-body order (default: 3)")
    parser.add_argument("--backward", action="store_true", help="Measure forward + backward pass time")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for inference/training (recommended)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5,
                        help="Repeat count per N when measuring")
    parser.add_argument("--max-atoms-cap", type=int, default=200_000,
                        help="Upper bound for binary search (default: 200000)")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"],
                        help="Compute dtype (default: float32)")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)  # Required for cuequivariance Fused TP math_dtype
    if device.type == "cuda" and dtype == torch.float32:
        torch.set_float32_matmul_precision("high")

    cfg = dict(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=args.channels,
        hidden_dim_sh=args.channels,
        hidden_dim=64,
        channel_in2=32,
        embedding_dim=16,
        max_atomvalue=10,
        output_size=8,
        num_interaction=args.num_interaction,
        lmax=args.lmax,
        ictd_tp_path_policy="full",
        internal_compute_dtype=dtype,
    )

    print(f"Config: lmax={args.lmax}, channels={args.channels}, num_interaction={args.num_interaction}, {args.dtype}")
    print(f"Target: inference <= {args.target_ms:.0f} ms ({args.target_ms/1000:.2f} s)")
    print(f"Device: {device}, avg_degree: {args.avg_degree}, compile: {args.compile}")
    print()

    if args.model == "mace":
        import numpy as np
        import mace.modules
        import e3nn.o3 as o3
        layer = mace.modules.MACE(
            r_max=5.0,
            num_bessel=8,
            num_polynomial_cutoff=5,
            max_ell=args.lmax,
            interaction_cls=mace.modules.interaction_classes['RealAgnosticResidualInteractionBlock'],
            interaction_cls_first=mace.modules.interaction_classes['RealAgnosticInteractionBlock'],
            num_interactions=args.num_interaction,
            num_elements=1,
            hidden_irreps=o3.Irreps(f'{args.channels}x0e + {args.channels}x1o + {args.channels}x2e'),
            MLP_irreps=o3.Irreps('16x0e'),
            atomic_energies=np.zeros(1),
            avg_num_neighbors=args.avg_degree,
            atomic_numbers=[1],
            correlation=args.correction,
            gate=torch.nn.functional.silu,
        ).to(device=device, dtype=dtype)
        print("Initialized MACE model.")
    else:
        if args.model == 'spherical':
            ModelClass = E3_TransformerLayer_multi
        elif args.model == 'spherical-cue':
            try:
                import cuequivariance_torch  # noqa: F401
                from molecular_force_field.models.cue_layers import (
                    E3_TransformerLayer_multi as E3_TransformerLayer_multi_cue_full,
                )
                ModelClass = E3_TransformerLayer_multi_cue_full
            except Exception as e:
                raise ImportError(
                    f"spherical-cue requires cuEquivariance: {e}\n"
                    "Install: pip install cuequivariance-torch cuequivariance-ops-torch-cu12"
                ) from e
        elif args.model == 'spherical-save':
            ModelClass = E3_TransformerLayer_multi_channelwise
        elif args.model == 'spherical-save-cue':
            try:
                import cuequivariance_torch  # noqa: F401
                from molecular_force_field.models.cue_layers_channelwise import (
                    E3_TransformerLayer_multi as E3_TransformerLayer_multi_cue,
                )
                ModelClass = E3_TransformerLayer_multi_cue
            except Exception as e:
                raise ImportError(
                    f"spherical-save-cue requires cuEquivariance: {e}\n"
                    "Install: pip install cuequivariance-torch cuequivariance-ops-torch-cu12"
                ) from e
        elif args.model == 'partial-cartesian':
            ModelClass = CartesianTransformerLayer
        elif args.model == 'partial-cartesian-loose':
            ModelClass = CartesianTransformerLayerLoose
        elif args.model == 'pure-cartesian':
            ModelClass = PureCartesianTransformerLayer
        elif args.model == 'pure-cartesian-sparse':
            ModelClass = PureCartesianSparseTransformerLayer
        elif args.model == 'pure-cartesian-ictd':
            ModelClass = PureCartesianICTDTransformerLayerFull
        elif args.model == 'pure-cartesian-ictd-save':
            ModelClass = PureCartesianICTDSave
        else:
            raise ValueError(f"Unknown model {args.model}")
            

        if 'spherical' in args.model:
            import e3nn.o3 as o3
            irreps_in = f"{args.channels}x0e + {args.channels}x1o + {args.channels}x2e"
            cfg.update({
                'irreps_input': irreps_in,
                'irreps_query': irreps_in,
                'irreps_key': irreps_in,
                'irreps_value': irreps_in,
                'irreps_output': irreps_in,
                'irreps_sh': o3.Irreps.spherical_harmonics(lmax=args.lmax),
                'hidden_dim_sh': args.channels,
                'hidden_dim': 64,  # Overridden from the dict directly
                'channel_in2': 32, # Overridden from the dict directly
            })
            cfg.pop('hidden_dim_conv', None)
            cfg.pop('lmax', None)
            
        if 'ictd' not in args.model:
            cfg.pop('ictd_tp_path_policy', None)
            cfg.pop('internal_compute_dtype', None)
        print(f"Initialized {ModelClass.__name__}.")
        layer = ModelClass(**cfg).to(device=device, dtype=dtype)

    if args.compile and device.type == "cuda":
        try:
            layer = torch.compile(layer, mode="reduce-overhead", dynamic=False)
            print("torch.compile enabled. Warming up compilation...")
            _ = measure_inference_ms(
                layer, make_dummy_graph(device, dtype, 256, args.avg_degree),
                warmup=5, repeat=3, device=device, is_mace=(args.model == "mace"), backward=args.backward
            )
            print("Done.")
        except Exception as e:
            print(f"torch.compile failed: {e}, using eager.")

    # Binary search for max N such that inference time <= target_ms
    low, high = 1, args.max_atoms_cap
    best_n = 0
    best_time = 0.0

    while low <= high:
        mid = (low + high) // 2
        graph = make_dummy_graph(device, dtype, num_nodes=mid, avg_degree=args.avg_degree)
        try:
            t_ms = measure_inference_ms(layer, graph, warmup=args.warmup, repeat=args.repeat, device=device, is_mace=(args.model == "mace"), backward=args.backward)
        except Exception as e:
            print(f"  N={mid} failed: {e}")
            high = mid - 1
            continue

        if t_ms <= args.target_ms:
            best_n = mid
            best_time = t_ms
            low = mid + 1
            print(f"  N={mid:>6}  {t_ms:>8.2f} ms  OK (best so far)")
        else:
            high = mid - 1
            print(f"  N={mid:>6}  {t_ms:>8.2f} ms  > {args.target_ms:.0f} ms")

    print()
    if best_n == 0:
        print("No N found within cap; try --max-atoms-cap or increase --target-ms.")
    else:
        pass_name = "forward+backward" if args.backward else "inference"
        print(f"Result: max atoms per structure ({pass_name} <= {args.target_ms:.0f} ms) = {best_n}")
        print(f"        measured {pass_name} time at N={best_n}: {best_time:.2f} ms")
        print(f"        edges at N={best_n}: {best_n * args.avg_degree}")


if __name__ == "__main__":
    main()
