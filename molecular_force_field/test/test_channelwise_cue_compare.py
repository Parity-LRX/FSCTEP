"""
Compare speed of channelwise spherical backends:

  - e3nn backend: tensor_product_mode="spherical-save"  (e3nn_layers_channelwise.py)
  - cuEquivariance backend: tensor_product_mode="spherical-save-cue" (cue_layers_channelwise.py)

We run a forward + backward pass on a fixed dummy graph and report:
  - parameter count
  - avg ms/iter (after warmup)
  - speedup ratio (cue / e3nn)

Usage:
  python -m molecular_force_field.test_channelwise_cue_compare --device cpu --dtype float32
  python -m molecular_force_field.test_channelwise_cue_compare --num-nodes 80 --num-edges 600 --repeat 30
  # Training-like step with force loss (double backward; matches Trainer.train_epoch graph)
  python -m molecular_force_field.test_channelwise_cue_compare --force-step
  # Print CUDA profiler table (helps locate bottlenecks)
  python -m molecular_force_field.test_channelwise_cue_compare --device cuda --force-step --profile cue
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
from torch.profiler import ProfilerActivity, profile

from molecular_force_field.utils.scatter import scatter

from molecular_force_field.utils.config import ModelConfig
from molecular_force_field.models.e3nn_layers_channelwise import (
    E3_TransformerLayer_multi as E3_TransformerLayer_multi_e3nn_cw,
)
from molecular_force_field.models.cue_layers_channelwise import (
    E3_TransformerLayer_multi as E3_TransformerLayer_multi_cue_cw,
)


@dataclass
class DummyGraph:
    pos: torch.Tensor
    A: torch.Tensor
    batch: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_shifts: torch.Tensor
    cell: torch.Tensor


def make_dummy_graph(device: torch.device, dtype: torch.dtype, *, num_nodes: int, num_edges: int) -> DummyGraph:
    torch.manual_seed(42)
    pos = torch.randn(num_nodes, 3, device=device, dtype=dtype)
    A = torch.randint(1, 6, (num_nodes,), device=device, dtype=torch.long)
    batch = torch.zeros(num_nodes, device=device, dtype=torch.long)
    edge_dst = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.long)
    edge_src = torch.randint(0, num_nodes, (num_edges,), device=device, dtype=torch.long)
    edge_shifts = torch.zeros(num_edges, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(num_nodes, -1, -1)
    return DummyGraph(pos=pos, A=A, batch=batch, edge_src=edge_src, edge_dst=edge_dst, edge_shifts=edge_shifts, cell=cell)


def count_params(m: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in m.parameters())


def _zero_grads(m: torch.nn.Module):
    for p in m.parameters():
        p.grad = None


def run_fwd_bwd(e3trans: torch.nn.Module, g: DummyGraph) -> torch.Tensor:
    """
    One training-like step:
      E_per_atom = e3trans(...)
      E_mol = scatter_sum(E_per_atom)   (single structure => scalar)
      loss = E_mol.sum()
      loss.backward()
    """
    # Fresh leaf for position grads (keeps step independent)
    pos = g.pos.detach().clone().requires_grad_(True)
    _zero_grads(e3trans)

    E_per_atom = e3trans(pos, g.A, g.batch, g.edge_src, g.edge_dst, g.edge_shifts, g.cell)  # (N, 1)
    E_mol = scatter(E_per_atom, g.batch, dim=0, reduce="sum").squeeze(-1)  # (num_mol,)
    loss = E_mol.sum()
    loss.backward()
    return loss.detach()


def run_force_step(
    e3trans: torch.nn.Module,
    g: DummyGraph,
    *,
    target_energy_mol: torch.Tensor,
    force_ref: torch.Tensor,
    a: float,
    b: float,
    force_shift_value: float,
) -> torch.Tensor:
    """
    Training-like force step that matches the compute graph in Trainer.train_epoch:
      E_per_atom -> scatter -> E_mean (per-mol)
      grads = d/dpos E_mean.sum()  (create_graph=True)
      f_pred = -grads
      loss = a*energy_loss + b*force_loss
      loss.backward()
    """
    pos = g.pos.detach().clone().requires_grad_(True)
    _zero_grads(e3trans)

    E_per_atom = e3trans(pos, g.A, g.batch, g.edge_src, g.edge_dst, g.edge_shifts, g.cell)  # (N, 1)
    E_conv_mol = scatter(E_per_atom, g.batch, dim=0, reduce="sum").squeeze(-1)  # (num_mol,)
    E_mean = E_conv_mol

    grads = torch.autograd.grad(
        E_mean.sum(),
        pos,
        create_graph=True,
        retain_graph=True,
    )[0]  # (N, 3)

    f_pred = -grads  # (N, 3)
    force_ref_scaled = force_ref * float(force_shift_value)

    criterion = torch.nn.SmoothL1Loss(beta=0.5)

    force_loss = criterion(f_pred.view(-1), force_ref_scaled.view(-1))

    num_atoms_per_mol = scatter(torch.ones_like(g.batch), g.batch, dim=0, reduce="sum").to(E_mean.dtype)
    E_avg_pred = E_mean / num_atoms_per_mol
    target_energy_avg = target_energy_mol / num_atoms_per_mol
    energy_loss = criterion(E_avg_pred, target_energy_avg)

    loss = float(a) * energy_loss + float(b) * force_loss
    loss.backward()
    return loss.detach()


def time_module(
    name: str,
    e3trans: torch.nn.Module,
    g: DummyGraph,
    *,
    warmup: int,
    repeat: int,
) -> float:
    e3trans.train()

    # Warmup (not timed)
    for _ in range(warmup):
        _ = run_fwd_bwd(e3trans, g)

    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = run_fwd_bwd(e3trans, g)
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000.0 / float(repeat)
    print(f"{name:>16s}: {avg_ms:8.2f} ms/iter", flush=True)
    return float(avg_ms)


def time_module_force(
    name: str,
    e3trans: torch.nn.Module,
    g: DummyGraph,
    *,
    target_energy_mol: torch.Tensor,
    force_ref: torch.Tensor,
    a: float,
    b: float,
    force_shift_value: float,
    warmup: int,
    repeat: int,
) -> float:
    e3trans.train()

    for _ in range(warmup):
        _ = run_force_step(
            e3trans,
            g,
            target_energy_mol=target_energy_mol,
            force_ref=force_ref,
            a=a,
            b=b,
            force_shift_value=force_shift_value,
        )

    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = run_force_step(
            e3trans,
            g,
            target_energy_mol=target_energy_mol,
            force_ref=force_ref,
            a=a,
            b=b,
            force_shift_value=force_shift_value,
        )
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000.0 / float(repeat)
    print(f"{name:>16s}: {avg_ms:8.2f} ms/iter", flush=True)
    return float(avg_ms)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64", "float", "double"])
    p.add_argument("--num-nodes", type=int, default=80)
    p.add_argument("--num-edges", type=int, default=600)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--force-step", action="store_true", help="Benchmark training-like force step (double backward).")
    p.add_argument("--a", type=float, default=1.0, help="Energy loss weight (Trainer.a).")
    p.add_argument("--b", type=float, default=1.0, help="Force loss weight (Trainer.b).")
    p.add_argument("--force-shift-value", type=float, default=1.0, help="Force scaling (Trainer.force_shift_value).")
    p.add_argument("--profile", type=str, default="none", choices=["none", "e3nn", "cue", "both"],
                   help="Print a torch.profiler table (single iteration).")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype in ("float32", "float") else torch.float64

    torch.set_default_dtype(dtype)

    config = ModelConfig(dtype=dtype)
    g = make_dummy_graph(device, dtype, num_nodes=int(args.num_nodes), num_edges=int(args.num_edges))

    # Dummy targets (single-molecule batch by default)
    num_mol = int(g.batch.max().item()) + 1 if g.batch.numel() > 0 else 1
    torch.manual_seed(123)
    target_energy_mol = torch.randn(num_mol, device=device, dtype=dtype)
    force_ref = torch.randn(int(args.num_nodes), 3, device=device, dtype=dtype)

    # Both should match the CLI construction for spherical-save / spherical-save-cue
    e3nn_layer = E3_TransformerLayer_multi_e3nn_cw(
        max_embed_radius=config.max_radius,
        main_max_radius=config.max_radius_main,
        main_number_of_basis=config.number_of_basis_main,
        irreps_input=config.get_irreps_output_conv(),
        irreps_query=config.get_irreps_query_main(),
        irreps_key=config.get_irreps_key_main(),
        irreps_value=config.get_irreps_value_main(),
        irreps_output=config.get_irreps_output_conv_2(),
        irreps_sh=config.get_irreps_sh_transformer(),
        hidden_dim_sh=config.get_hidden_dim_sh(),
        hidden_dim=config.emb_number_main_2,
        channel_in2=config.channel_in2,
        embedding_dim=config.embedding_dim,
        max_atomvalue=config.max_atomvalue,
        output_size=config.output_size,
        embed_size=config.embed_size,
        main_hidden_sizes3=config.main_hidden_sizes3,
        num_layers=config.num_layers,
        num_interaction=2,
        function_type_main=config.function_type,
        device=device,
    ).to(device)

    cue_layer = E3_TransformerLayer_multi_cue_cw(
        max_embed_radius=config.max_radius,
        main_max_radius=config.max_radius_main,
        main_number_of_basis=config.number_of_basis_main,
        irreps_input=config.get_irreps_output_conv(),
        irreps_query=config.get_irreps_query_main(),
        irreps_key=config.get_irreps_key_main(),
        irreps_value=config.get_irreps_value_main(),
        irreps_output=config.get_irreps_output_conv_2(),
        irreps_sh=config.get_irreps_sh_transformer(),
        hidden_dim_sh=config.get_hidden_dim_sh(),
        hidden_dim=config.emb_number_main_2,
        channel_in2=config.channel_in2,
        embedding_dim=config.embedding_dim,
        max_atomvalue=config.max_atomvalue,
        output_size=config.output_size,
        embed_size=config.embed_size,
        main_hidden_sizes3=config.main_hidden_sizes3,
        num_layers=config.num_layers,
        num_interaction=2,
        function_type_main=config.function_type,
        device=device,
    ).to(device)

    print(f"Device={device}, dtype={dtype}", flush=True)
    print(f"Graph: N={args.num_nodes}, E={args.num_edges}, warmup={args.warmup}, repeat={args.repeat}", flush=True)
    if args.force_step:
        print("Mode: force-step (double backward, create_graph=True)", flush=True)
    print(f"Params e3nn / cue: {count_params(e3nn_layer):,} / {count_params(cue_layer):,}", flush=True)

    def _run_one(name: str, layer: torch.nn.Module):
        if args.force_step:
            return run_force_step(
                layer,
                g,
                target_energy_mol=target_energy_mol,
                force_ref=force_ref,
                a=float(args.a),
                b=float(args.b),
                force_shift_value=float(args.force_shift_value),
            )
        return run_fwd_bwd(layer, g)

    def _maybe_profile(name: str, layer: torch.nn.Module):
        if args.profile == "none":
            return
        if args.profile not in (name, "both"):
            return
        if device.type != "cuda":
            print("[profile] CUDA profiler requested but device is not cuda; skipping.", flush=True)
            return

        # Warmup a bit to avoid capturing one-time kernel init.
        for _ in range(3):
            _ = _run_one(name, layer)
        torch.cuda.synchronize()

        acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=acts, record_shapes=False, profile_memory=False, with_stack=False) as prof:
            _ = _run_one(name, layer)
            torch.cuda.synchronize()

        print("\n" + "=" * 80, flush=True)
        print(f"[profile] {name} top ops by self CUDA time", flush=True)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30), flush=True)
        print("=" * 80 + "\n", flush=True)

    # Optional single-iteration profiler dump (before timing loops)
    _maybe_profile("e3nn", e3nn_layer)
    _maybe_profile("cue", cue_layer)

    if args.force_step:
        ms_e3nn = time_module_force(
            "e3nn_channelwise",
            e3nn_layer,
            g,
            target_energy_mol=target_energy_mol,
            force_ref=force_ref,
            a=float(args.a),
            b=float(args.b),
            force_shift_value=float(args.force_shift_value),
            warmup=int(args.warmup),
            repeat=int(args.repeat),
        )
        ms_cue = time_module_force(
            "cue_channelwise",
            cue_layer,
            g,
            target_energy_mol=target_energy_mol,
            force_ref=force_ref,
            a=float(args.a),
            b=float(args.b),
            force_shift_value=float(args.force_shift_value),
            warmup=int(args.warmup),
            repeat=int(args.repeat),
        )
    else:
        ms_e3nn = time_module("e3nn_channelwise", e3nn_layer, g, warmup=int(args.warmup), repeat=int(args.repeat))
        ms_cue = time_module("cue_channelwise", cue_layer, g, warmup=int(args.warmup), repeat=int(args.repeat))

    if ms_cue > 0:
        print(f"Speedup (e3nn/cue): {ms_e3nn / ms_cue:.3f}x", flush=True)


if __name__ == "__main__":
    main()

