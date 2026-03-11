#!/usr/bin/env python3
"""
DDP parallel inference: partition a single large structure by nodes across GPUs; sync ghost node features after each scatter.

Use case: when a single structure has many atoms and single-GPU memory is insufficient, use multiple GPUs to share memory and compute.
Limitations: inference only; no LAMMPS export, no gradients.

Usage (from project root):
  torchrun --nproc_per_node=2 -m molecular_force_field.cli.inference_ddp --atoms 100000
  torchrun --nproc_per_node=2 -m molecular_force_field.cli.inference_ddp --atoms 50000 --forces   # output energy and forces

If OpenMP SHM error occurs, set: export OMP_NUM_THREADS=1
"""

from __future__ import annotations

import argparse
import os
import re
import time

import torch
import torch.distributed as dist

from molecular_force_field.models.pure_cartesian_ictd_layers import PureCartesianICTDTransformerLayer


def _get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def _get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def partition_graph(
    num_nodes: int,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    pos: torch.Tensor,
    world_size: int,
    rank: int,
    partition_mode: str = "modulo",
):
    """
    Partition by node_id % world_size: rank r owns nodes with node_id % world_size == r.
    Returns this rank's owned_global, ghost_global (sorted), send_list, recv_list.
    Vectorized impl to avoid Python loops and .cpu().numpy() on large graphs.
    """
    dev = edge_dst.device
    if partition_mode == "modulo":
        node_owner = torch.arange(num_nodes, device=dev, dtype=torch.long) % world_size
    elif partition_mode == "spatial":
        # 1D spatial partitioning along principal axis to reduce cross-rank edges and ghost nodes.
        span = pos.max(dim=0).values - pos.min(dim=0).values
        axis = int(torch.argmax(span).item())
        coord = pos[:, axis]
        cmin = coord.min()
        cspan = (coord.max() - cmin).clamp_min(1e-8)
        rel = (coord - cmin) / cspan
        node_owner = torch.clamp((rel * world_size).to(torch.long), max=world_size - 1)
    else:
        raise ValueError(f"Unknown partition_mode: {partition_mode}")

    # Edges owned by this rank: owner(edge_dst) == rank
    mask = (node_owner[edge_dst] == rank)
    owned_global_t = torch.where(node_owner == rank)[0]
    owned_global = owned_global_t.cpu().tolist()
    # ghost = unique(edge_src[mask]) - owned
    ghost_candidates = edge_src[mask]
    unique_ghost = torch.unique(ghost_candidates)
    owned_mask = torch.zeros(num_nodes, dtype=torch.bool, device=dev)
    owned_mask[owned_global_t] = True
    ghost_global_t = unique_ghost[~owned_mask[unique_ghost]]
    ghost_global_t = torch.sort(ghost_global_t)[0]
    ghost_global = ghost_global_t.cpu().tolist()
    # recv_list[r] = ghosts owned by r
    recv_list = [ghost_global_t[node_owner[ghost_global_t] == r].cpu().tolist() for r in range(world_size)]
    # send_list[s] = global ids of this rank's owned nodes that are ghosts for s
    send_list = []
    for s in range(world_size):
        if s == rank:
            send_list.append([])
            continue
        em = (node_owner[edge_dst] == s) & (node_owner[edge_src] == rank)
        send_list.append(torch.sort(torch.unique(edge_src[em]))[0].cpu().tolist())
    return owned_global, ghost_global, send_list, recv_list


def build_local_graph(
    pos: torch.Tensor,
    A: torch.Tensor,
    batch: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_shifts: torch.Tensor,
    cell: torch.Tensor,
    owned_global: list[int],
    ghost_global: list[int],
    device: torch.device,
    dtype: torch.dtype,
):
    """Build local graph: node order [owned | ghost], keep only edges with edge_dst in owned, remap to local indices. Vectorized."""
    num_owned = len(owned_global)
    num_ghost = len(ghost_global)
    num_local = num_owned + num_ghost
    owned_t = torch.tensor(owned_global, device=device, dtype=torch.long)
    ghost_t = torch.tensor(ghost_global, device=device, dtype=torch.long)
    pos_local = torch.empty(num_local, 3, device=device, dtype=dtype)
    pos_local[:num_owned] = pos[owned_t]
    pos_local[num_owned:] = pos[ghost_t]
    A_local = torch.empty(num_local, device=device, dtype=A.dtype)
    A_local[:num_owned] = A[owned_t]
    A_local[num_owned:] = A[ghost_t]
    batch_local = torch.empty(num_local, device=device, dtype=torch.long)
    batch_local[:num_owned] = batch[owned_t]
    batch_local[num_owned:] = batch[ghost_t]

    global_to_local_tensor = torch.full((pos.size(0),), -1, device=device, dtype=torch.long)
    global_to_local_tensor[owned_t] = torch.arange(num_owned, device=device, dtype=torch.long)
    global_to_local_tensor[ghost_t] = num_owned + torch.arange(num_ghost, device=device, dtype=torch.long)
    owned_mask = torch.zeros(pos.size(0), dtype=torch.bool, device=device)
    owned_mask[owned_t] = True
    mask = owned_mask[edge_dst]
    edge_src_local = global_to_local_tensor[edge_src[mask]]
    edge_dst_local = global_to_local_tensor[edge_dst[mask]]
    edge_shifts_local = edge_shifts[mask]

    if cell.dim() == 3 and cell.size(0) == pos.size(0):
        cell_local = torch.empty(num_local, 3, 3, device=device, dtype=dtype)
        cell_local[:num_owned] = cell[owned_t]
        cell_local[num_owned:] = cell[ghost_t]
    else:
        cell_local = cell[:1].expand(num_local, -1, -1).clone().to(device=device, dtype=dtype)

    return pos_local, A_local, batch_local, edge_src_local, edge_dst_local, edge_shifts_local, cell_local


def make_sync_after_scatter(
    rank: int,
    world_size: int,
    num_owned: int,
    num_ghost: int,
    send_list: list[list[int]],
    recv_list: list[list[int]],
    owned_global: list[int],
    device: torch.device,
):
    """
    Return closure sync_after_scatter(node_features) -> node_features.
    node_features: (num_owned + num_ghost, C). After scatter only owned is correct; ghost must be received from other ranks.
    """
    # This rank's owned at local indices 0..num_owned-1; ghost at num_owned..num_owned+num_ghost-1
    # send_list[s] = global ids to send to s; convert to local owned indices for gather.
    global_to_local_owned = {g: i for i, g in enumerate(owned_global)}

    send_indices = []  # send_indices[s] = local indices (owned) to send to s
    for s in range(world_size):
        if s == rank:
            send_indices.append([])
            continue
        send_indices.append([global_to_local_owned[g] for g in send_list[s]])
    send_counts = [len(send_indices[s]) for s in range(world_size)]
    recv_counts = [len(recv_list[r]) for r in range(world_size)]
    # Prebuild index tensors to avoid rebuilding per scatter layer.
    send_indices_t = [
        torch.tensor(send_indices[s], device=device, dtype=torch.long) if send_counts[s] > 0 else None
        for s in range(world_size)
    ]

    # Ghost order fixed as concat(recv_list[0], recv_list[1], ...); write-back indices precomputed as contiguous blocks.
    recv_local_indices_t = []
    ghost_offset = 0
    for r in range(world_size):
        n = recv_counts[r]
        if n > 0:
            recv_local_indices_t.append(
                torch.arange(num_owned + ghost_offset, num_owned + ghost_offset + n, device=device, dtype=torch.long)
            )
            ghost_offset += n
        else:
            recv_local_indices_t.append(None)

    # Cache comm buffers by feat_dim to avoid per-layer allocation of large tensors.
    buf_cache: dict[tuple[torch.dtype, int], tuple[torch.Tensor, torch.Tensor, list[int], list[int]]] = {}

    def sync_after_scatter(node_features: torch.Tensor) -> torch.Tensor:
        # node_features (num_local, C)
        feat_dim = node_features.size(-1)
        dtype_f = node_features.dtype

        if world_size == 1:
            return node_features

        send_counts_elems = [c * feat_dim for c in send_counts]
        recv_counts_elems = [c * feat_dim for c in recv_counts]
        if sum(send_counts_elems) == 0 and sum(recv_counts_elems) == 0:
            return node_features

        cache_key = (dtype_f, feat_dim)
        cached = buf_cache.get(cache_key)
        send_total = sum(send_counts_elems)
        recv_total = sum(recv_counts_elems)
        if cached is None:
            send_buf = torch.empty(send_total, device=device, dtype=dtype_f)
            recv_buf = torch.empty(recv_total, device=device, dtype=dtype_f)
            buf_cache[cache_key] = (send_buf, recv_buf, send_counts_elems, recv_counts_elems)
        else:
            send_buf, recv_buf, _, _ = cached
            if send_buf.numel() != send_total:
                send_buf = torch.empty(send_total, device=device, dtype=dtype_f)
            if recv_buf.numel() != recv_total:
                recv_buf = torch.empty(recv_total, device=device, dtype=dtype_f)
            buf_cache[cache_key] = (send_buf, recv_buf, send_counts_elems, recv_counts_elems)

        so = 0
        for s in range(world_size):
            if send_counts[s] > 0:
                idx = send_indices_t[s]
                send_buf[so : so + send_counts_elems[s]] = node_features[idx].reshape(-1)
                so += send_counts_elems[s]

        dist.all_to_all_single(
            recv_buf,
            send_buf,
            recv_counts_elems,
            send_counts_elems,
            group=None,
        )

        # Write back to ghost positions
        ro = 0
        for r in range(world_size):
            if recv_counts[r] > 0:
                n = recv_counts[r]
                chunk = recv_buf[ro : ro + n * feat_dim].reshape(n, feat_dim)
                node_features[recv_local_indices_t[r]].copy_(chunk)
                ro += n * feat_dim

        return node_features

    return sync_after_scatter


class _SyncAfterScatterBackward(torch.autograd.Function):
    """Differentiable all_to_all sync wrapper: backward gradients communicated via transpose."""

    @staticmethod
    def forward(ctx, node_features, send_indices_flat, recv_ghost_local_flat, send_counts_elems, recv_counts_elems, device):
        world_size = len(send_counts_elems)
        feat_dim = node_features.size(-1)
        dtype_f = node_features.dtype
        send_buf = torch.zeros(sum(send_counts_elems), device=device, dtype=dtype_f)
        for i, idx in enumerate(send_indices_flat.tolist()):
            send_buf[i * feat_dim : (i + 1) * feat_dim] = node_features[idx].reshape(-1)
        recv_buf = torch.zeros(sum(recv_counts_elems), device=device, dtype=dtype_f)
        dist.all_to_all_single(recv_buf, send_buf, recv_counts_elems, send_counts_elems, group=None)
        out = node_features.clone()
        ro = 0
        for r in range(world_size):
            n = recv_counts_elems[r] // feat_dim
            if n > 0:
                out[recv_ghost_local_flat[ro : ro + n]] = recv_buf[ro * feat_dim : (ro + n) * feat_dim].reshape(n, feat_dim)
                ro += n
        ctx.send_indices_flat = send_indices_flat
        ctx.send_counts_elems = tuple(send_counts_elems)
        ctx.recv_counts_elems = tuple(recv_counts_elems)
        ctx.recv_ghost_local_flat = recv_ghost_local_flat
        ctx.feat_dim = feat_dim
        ctx.device = device
        return out

    @staticmethod
    def backward(ctx, grad_output):
        send_counts_elems = list(ctx.recv_counts_elems)
        recv_counts_elems = list(ctx.send_counts_elems)
        recv_ghost_local_flat = ctx.recv_ghost_local_flat
        send_indices_flat = ctx.send_indices_flat
        feat_dim = ctx.feat_dim
        device = ctx.device
        send_buf = grad_output.reshape(-1)[
            recv_ghost_local_flat.unsqueeze(1) * feat_dim + torch.arange(feat_dim, device=device)
        ].reshape(-1)
        recv_buf = torch.zeros(sum(recv_counts_elems), device=device, dtype=grad_output.dtype)
        dist.all_to_all_single(recv_buf, send_buf, recv_counts_elems, send_counts_elems, group=None)
        grad_input = grad_output.clone()
        for i, idx in enumerate(send_indices_flat.tolist()):
            grad_input[idx] = grad_input[idx] + recv_buf[i * feat_dim : (i + 1) * feat_dim].reshape(grad_input[idx].shape)
        return grad_input, None, None, None, None, None


def make_sync_after_scatter_diff(
    rank: int,
    world_size: int,
    num_owned: int,
    num_ghost: int,
    send_list: list[list[int]],
    recv_list: list[list[int]],
    owned_global: list[int],
    device: torch.device,
):
    """Differentiable sync closure for when forces are needed."""
    global_to_local_owned = {g: i for i, g in enumerate(owned_global)}
    send_indices = [[global_to_local_owned[g] for g in send_list[s]] for s in range(world_size)]
    recv_counts = [len(recv_list[r]) for r in range(world_size)]
    recv_ghost_local = []
    ghost_offset = 0
    for r in range(world_size):
        n = recv_counts[r]
        if n > 0:
            recv_ghost_local.extend(range(num_owned + ghost_offset, num_owned + ghost_offset + n))
            ghost_offset += n
    send_indices_flat = [idx for per_rank in send_indices for idx in per_rank]
    recv_ghost_local_flat = torch.tensor(recv_ghost_local, device=device, dtype=torch.long)
    send_indices_flat_t = torch.tensor(send_indices_flat, device=device, dtype=torch.long)
    send_counts = [len(send_indices[s]) for s in range(world_size)]

    def sync_after_scatter_diff(node_features: torch.Tensor) -> torch.Tensor:
        feat_dim = node_features.size(-1)
        if world_size == 1:
            return node_features
        send_counts_elems = [c * feat_dim for c in send_counts]
        recv_counts_elems = [c * feat_dim for c in recv_counts]
        if sum(send_counts_elems) == 0 and sum(recv_counts_elems) == 0:
            return node_features
        return _SyncAfterScatterBackward.apply(
            node_features,
            send_indices_flat_t,
            recv_ghost_local_flat,
            send_counts_elems,
            recv_counts_elems,
            device,
        )

    return sync_after_scatter_diff


def run_ddp_step_from_graph(
    pos: torch.Tensor,
    A: torch.Tensor,
    batch: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_shifts: torch.Tensor,
    cell: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    return_forces: bool = True,
    partition_mode: str = "modulo",
    cache: dict | None = None,
    comm_timing: dict | None = None,
) -> tuple[float, torch.Tensor] | tuple[None, None]:
    """
    All ranks run one DDP inference step on same graph (partition, forward, optional forces).
    Returns (energy, forces) valid only on rank 0; others get (None, None).
    If cache (dict) passed, same (num_nodes, num_edges) reuses partition and sync; only build_local_graph recomputed.
    If comm_timing not None, rank 0 records sync_ms (per sync_after_scatter), all_reduce_energy_ms, all_reduce_forces_ms.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_nodes_actual = pos.size(0)
    num_edges_actual = edge_src.size(0)
    use_cache = (cache is not None) and (partition_mode == "modulo")
    cache_key = (num_nodes_actual, num_edges_actual, return_forces, partition_mode)
    if use_cache and cache_key in cache:
        owned_global, ghost_global, send_list, recv_list, sync_after_scatter = cache[cache_key]
        num_owned = len(owned_global)
        num_ghost = len(ghost_global)
    else:
        owned_global, ghost_global, send_list, recv_list = partition_graph(
            num_nodes_actual, edge_src, edge_dst, pos, world_size, rank, partition_mode=partition_mode
        )
        num_owned = len(owned_global)
        num_ghost = len(ghost_global)
        if return_forces:
            sync_after_scatter = make_sync_after_scatter_diff(
                rank, world_size, num_owned, num_ghost, send_list, recv_list, owned_global, device
            )
        else:
            sync_after_scatter = make_sync_after_scatter(
                rank, world_size, num_owned, num_ghost, send_list, recv_list, owned_global, device
            )
        if use_cache:
            cache[cache_key] = (owned_global, ghost_global, send_list, recv_list, sync_after_scatter)
    if comm_timing is not None:
        comm_timing["sync_ms"] = []
        _sync_list = comm_timing["sync_ms"]

        def wrapped_sync(x):
            t0 = time.perf_counter()
            out = sync_after_scatter(x)
            _sync_list.append((time.perf_counter() - t0) * 1000.0)
            return out

        sync_to_use = wrapped_sync
    else:
        sync_to_use = sync_after_scatter
    pos_local, A_local, batch_local, edge_src_local, edge_dst_local, edge_shifts_local, cell_local = build_local_graph(
        pos, A, batch, edge_src, edge_dst, edge_shifts, cell,
        owned_global, ghost_global, device, dtype,
    )
    if return_forces:
        pos_local = pos_local.requires_grad_(True)
    model.eval()
    if return_forces:
        out = model(
            pos_local,
            A_local,
            batch_local,
            edge_src_local,
            edge_dst_local,
            edge_shifts_local,
            cell_local,
            sync_after_scatter=sync_to_use,
        )
        local_energy = out[:num_owned].sum()
        total_energy = local_energy.clone()
        if comm_timing is not None:
            dist.barrier()
            t0 = time.perf_counter()
        dist.all_reduce(total_energy, op=dist.ReduceOp.SUM)
        if comm_timing is not None and rank == 0:
            comm_timing["all_reduce_energy_ms"] = (time.perf_counter() - t0) * 1000.0
        total_energy_val = total_energy.detach().item()
        total_energy.backward()
        full_forces = torch.zeros(num_nodes_actual, 3, device=device, dtype=dtype)
        owned_t = torch.tensor(owned_global, device=device, dtype=torch.long)
        full_forces.index_copy_(0, owned_t, -pos_local.grad[:num_owned])
        if comm_timing is not None:
            dist.barrier()
            t0 = time.perf_counter()
        dist.all_reduce(full_forces, op=dist.ReduceOp.SUM)
        if comm_timing is not None and rank == 0:
            comm_timing["all_reduce_forces_ms"] = (time.perf_counter() - t0) * 1000.0
    else:
        with torch.no_grad():
            out = model(
                pos_local,
                A_local,
                batch_local,
                edge_src_local,
                edge_dst_local,
                edge_shifts_local,
                cell_local,
                sync_after_scatter=sync_to_use,
            )
        local_energy = out[:num_owned].sum()
        total_energy = local_energy.clone()
        if comm_timing is not None:
            dist.barrier()
            t0 = time.perf_counter()
        dist.all_reduce(total_energy, op=dist.ReduceOp.SUM)
        if comm_timing is not None and rank == 0:
            comm_timing["all_reduce_energy_ms"] = (time.perf_counter() - t0) * 1000.0
        total_energy_val = total_energy.item()
        full_forces = None
        if comm_timing is not None and rank == 0:
            comm_timing["all_reduce_forces_ms"] = None
    if rank != 0:
        return None, None
    return total_energy_val, full_forces


def run_one_ddp_inference_from_ase_atoms(
    atoms_or_none,
    model: torch.nn.Module,
    max_radius: float,
    device: torch.device,
    dtype: torch.dtype,
    *,
    return_forces: bool = True,
    partition_mode: str = "modulo",
    atomic_energies_dict: dict | None = None,
    cache: dict | None = None,
) -> tuple[float | None, torch.Tensor | None]:
    """
    Run one DDP inference step from ASE Atoms. **All ranks must call**:
    - rank 0 passes current atoms; others pass None.
    - rank 0 builds graph and broadcasts tensors; all ranks run run_ddp_step_from_graph.
    - If rank 0 passes None (MD end), returns (None, None).
    Returns (energy, forces); valid only on rank 0; (None, None) on exit signal.
    """
    from molecular_force_field.utils.graph_utils import radius_graph_pbc_gpu

    rank = dist.get_rank()
    if rank == 0 and atoms_or_none is not None:
        atoms = atoms_or_none
        pos = torch.tensor(atoms.get_positions(), dtype=dtype, device=device)
        A = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
        batch = torch.zeros(len(atoms), dtype=torch.long, device=device)
        if any(atoms.pbc):
            cell = torch.tensor(atoms.get_cell().array, dtype=dtype, device=device).unsqueeze(0)
            pbc = tuple(bool(x) for x in atoms.pbc)
        else:
            cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * 100.0
            pbc = (False, False, False)
        edge_src, edge_dst, edge_shifts = radius_graph_pbc_gpu(pos, max_radius, cell, pbc=pbc)
    else:
        pos = A = batch = edge_src = edge_dst = edge_shifts = cell = None
    out = broadcast_graph_from_rank0(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, device)
    if out is None:
        return None, None
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = out
    energy, forces = run_ddp_step_from_graph(
        pos, A, batch, edge_src, edge_dst, edge_shifts, cell,
        model, device, dtype, return_forces=return_forces, partition_mode=partition_mode, cache=cache,
    )
    if energy is None:
        return None, None
    if atomic_energies_dict and dist.get_rank() == 0:
        from molecular_force_field.utils.tensor_utils import map_tensor_values
        keys = torch.tensor(list(atomic_energies_dict.keys()), device=device)
        values = torch.tensor(list(atomic_energies_dict.values()), device=device, dtype=dtype)
        E_offset = map_tensor_values(A.float(), keys, values).sum().item()
        energy = energy + E_offset
    return energy, forces


def build_and_broadcast_graph(
    num_nodes: int,
    avg_degree: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    comm_timing: dict | None = None,
):
    """Rank 0 builds graph and broadcasts; all ranks get same graph. For benchmark, broadcast once per N."""
    rank = dist.get_rank()
    if rank == 0:
        torch.manual_seed(seed)
        pos = torch.randn(num_nodes, 3, device=device, dtype=dtype) * 2.0
        A = torch.randint(1, 6, (num_nodes,), device=device)
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        num_edges = num_nodes * avg_degree
        edge_dst = torch.randint(0, num_nodes, (num_edges,), device=device)
        edge_src = torch.randint(0, num_nodes, (num_edges,), device=device)
        edge_shifts = torch.zeros(num_edges, 3, device=device, dtype=dtype)
        cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(num_nodes, -1, -1)
    else:
        pos = torch.zeros(1, 3, device=device, dtype=dtype)
        A = torch.zeros(1, device=device, dtype=torch.long)
        batch = torch.zeros(1, dtype=torch.long, device=device)
        edge_src = torch.zeros(1, device=device, dtype=torch.long)
        edge_dst = torch.zeros(1, device=device, dtype=torch.long)
        edge_shifts = torch.zeros(1, 3, device=device, dtype=dtype)
        cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    return broadcast_graph_from_rank0(
        pos, A, batch, edge_src, edge_dst, edge_shifts, cell, device, comm_timing=comm_timing
    )


def run_ddp_forward_timed(
    pos: torch.Tensor,
    A: torch.Tensor,
    batch: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_shifts: torch.Tensor,
    cell: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    partition_mode: str = "modulo",
    cache: dict | None = None,
    comm_timing: dict | None = None,
) -> tuple[float, float]:
    """
    Partition + forward only on already-broadcast graph, with timing (no build/broadcast).
    With cache, same graph size reuses partition and sync. Returns (time_ms, energy), valid only on rank 0.
    If comm_timing not None, fills sync_ms, all_reduce_energy_ms, all_reduce_forces_ms (see run_ddp_step_from_graph).
    """
    dist.barrier()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    energy, _ = run_ddp_step_from_graph(
        pos, A, batch, edge_src, edge_dst, edge_shifts, cell,
        model, device, dtype, return_forces=False, partition_mode=partition_mode, cache=cache, comm_timing=comm_timing,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    dist.barrier()
    t1 = time.perf_counter()
    time_ms = (t1 - t0) * 1000.0
    if dist.get_rank() != 0:
        return 0.0, 0.0
    return time_ms, energy or 0.0


def run_one_ddp_inference(
    num_nodes: int,
    model: torch.nn.Module,
    *,
    avg_degree: int = 24,
    seed: int = 42,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    return_forces: bool = False,
    partition_mode: str = "modulo",
    output_physical_tensors: bool = False,
) -> tuple[float, float] | tuple[float, float, torch.Tensor]:
    """
    Run one DDP inference with timing. Requires dist initialized and model built on all ranks.
    Returns (time_ms, total_energy) or (time_ms, total_energy, forces) if return_forces=True;
    only rank 0 returns valid values; others get (0.0, 0.0) or (0.0, 0.0, None).
    """
    import time
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if device is None:
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        torch.manual_seed(seed)
        pos = torch.randn(num_nodes, 3, device=device, dtype=dtype) * 2.0
        A = torch.randint(1, 6, (num_nodes,), device=device)
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        num_edges = num_nodes * avg_degree
        edge_dst = torch.randint(0, num_nodes, (num_edges,), device=device)
        edge_src = torch.randint(0, num_nodes, (num_edges,), device=device)
        edge_shifts = torch.zeros(num_edges, 3, device=device, dtype=dtype)
        cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(num_nodes, -1, -1)
    else:
        pos = torch.zeros(1, 3, device=device, dtype=dtype)
        A = torch.zeros(1, device=device, dtype=torch.long)
        batch = torch.zeros(1, dtype=torch.long, device=device)
        edge_src = torch.zeros(1, device=device, dtype=torch.long)
        edge_dst = torch.zeros(1, device=device, dtype=torch.long)
        edge_shifts = torch.zeros(1, 3, device=device, dtype=dtype)
        cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)

    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = broadcast_graph_from_rank0(
        pos, A, batch, edge_src, edge_dst, edge_shifts, cell, device
    )
    num_nodes_actual = pos.size(0)
    owned_global, ghost_global, send_list, recv_list = partition_graph(
        num_nodes_actual, edge_src, edge_dst, pos, world_size, rank, partition_mode=partition_mode
    )
    num_owned = len(owned_global)
    num_ghost = len(ghost_global)
    pos_local, A_local, batch_local, edge_src_local, edge_dst_local, edge_shifts_local, cell_local = build_local_graph(
        pos, A, batch, edge_src, edge_dst, edge_shifts, cell,
        owned_global, ghost_global, device, dtype,
    )
    if return_forces:
        sync_after_scatter = make_sync_after_scatter_diff(
            rank, world_size, num_owned, num_ghost, send_list, recv_list, owned_global, device
        )
        pos_local = pos_local.requires_grad_(True)
    else:
        sync_after_scatter = make_sync_after_scatter(
            rank, world_size, num_owned, num_ghost, send_list, recv_list, owned_global, device
        )

    model.eval()
    dist.barrier()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    want_phys = output_physical_tensors and hasattr(model, "physical_tensor_heads") and model.physical_tensor_heads is not None
    fwd_kw = {"sync_after_scatter": sync_after_scatter}
    if want_phys:
        fwd_kw["return_physical_tensors"] = True
    if return_forces:
        out = model(
            pos_local,
            A_local,
            batch_local,
            edge_src_local,
            edge_dst_local,
            edge_shifts_local,
            cell_local,
            **fwd_kw,
        )
        out = out[0] if isinstance(out, tuple) else out
        local_energy = out[:num_owned].sum()
        total_energy = local_energy.clone()
        dist.all_reduce(total_energy, op=dist.ReduceOp.SUM)
        total_energy_val = total_energy.detach().item()
        total_energy.backward()
        full_forces = torch.zeros(num_nodes_actual, 3, device=device, dtype=dtype)
        owned_t = torch.tensor(owned_global, device=device, dtype=torch.long)
        full_forces.index_copy_(0, owned_t, -pos_local.grad[:num_owned])
        dist.all_reduce(full_forces, op=dist.ReduceOp.SUM)
    else:
        with torch.no_grad():
            out = model(
                pos_local,
                A_local,
                batch_local,
                edge_src_local,
                edge_dst_local,
                edge_shifts_local,
                cell_local,
                **fwd_kw,
            )
        out = out[0] if isinstance(out, tuple) else out
        local_energy = out[:num_owned].sum()
        total_energy = local_energy.clone()
        dist.all_reduce(total_energy, op=dist.ReduceOp.SUM)
        full_forces = None
    if device.type == "cuda":
        torch.cuda.synchronize()
    dist.barrier()
    t1 = time.perf_counter()
    time_ms = (t1 - t0) * 1000.0
    if rank != 0:
        return (0.0, 0.0, None) if return_forces else (0.0, 0.0)
    if return_forces:
        return time_ms, total_energy_val, full_forces
    return time_ms, total_energy.item()


def _broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """In-place broadcast: root has data, others receive into pre-allocated tensor.
    NCCL requires contiguous tensors; root side .contiguous() if needed before broadcast, returns tensor for caller.
    """
    rank = dist.get_rank()
    if rank == src and not tensor.is_contiguous():
        tensor = tensor.contiguous()
    dist.broadcast(tensor, src=src)
    return tensor


def broadcast_graph_from_rank0(
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell, device,
    comm_timing: dict | None = None,
):
    """
    Rank 0 provides graph or exit signal; tensor broadcast (no pickle).
    - rank 0 passes (pos, A, ...) to broadcast graph; (None, None, ...) to broadcast exit, all ranks return None.
    - Other ranks pass (None, None, ...), receive graph or None.
    - If comm_timing not None, rank 0 records comm_timing['broadcast_ms'].
    """
    rank = dist.get_rank()
    sizes = torch.empty(2, device=device, dtype=torch.long)
    if rank == 0:
        if pos is None:
            sizes[0], sizes[1] = 0, 0
        else:
            sizes[0], sizes[1] = pos.size(0), edge_src.size(0)
    if comm_timing is not None:
        dist.barrier()
        t0 = time.perf_counter()
    sizes = _broadcast_tensor(sizes, src=0)
    n, e = int(sizes[0].item()), int(sizes[1].item())
    if n == 0:
        return None
    dtype_pos = torch.float32
    if rank == 0:
        dtype_pos = pos.dtype
    if rank != 0:
        pos = torch.empty(n, 3, device=device, dtype=dtype_pos)
        A = torch.empty(n, device=device, dtype=torch.long)
        batch = torch.empty(n, device=device, dtype=torch.long)
        edge_src = torch.empty(e, device=device, dtype=torch.long)
        edge_dst = torch.empty(e, device=device, dtype=torch.long)
        edge_shifts = torch.empty(e, 3, device=device, dtype=dtype_pos)
        cell = torch.empty(n, 3, 3, device=device, dtype=dtype_pos)
    pos = _broadcast_tensor(pos, src=0)
    A = _broadcast_tensor(A, src=0)
    batch = _broadcast_tensor(batch, src=0)
    edge_src = _broadcast_tensor(edge_src, src=0)
    edge_dst = _broadcast_tensor(edge_dst, src=0)
    edge_shifts = _broadcast_tensor(edge_shifts, src=0)
    cell = _broadcast_tensor(cell, src=0)
    if comm_timing is not None:
        dist.barrier()
        if rank == 0:
            comm_timing["broadcast_ms"] = (time.perf_counter() - t0) * 1000.0
    return pos, A, batch, edge_src, edge_dst, edge_shifts, cell


def main():
    parser = argparse.ArgumentParser(
        description="DDP parallel inference: partition large structure by nodes, multi-GPU memory and compute"
    )
    parser.add_argument("--atoms", type=int, default=50000, help="Number of atoms (for dummy graph)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint (.pt); random weights if not provided")
    parser.add_argument("--avg-degree", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forces", action="store_true", help="Compute and output forces F=-dE/dpos")
    parser.add_argument("--backend", type=str, default=None,
                        help="dist backend: nccl (multi-GPU) or gloo (CPU/single-node); default auto by CUDA availability")
    parser.add_argument("--partition", type=str, default="modulo", choices=["modulo", "spatial"],
                        help="Graph partition: modulo (by node ID) or spatial (by coordinate principal axis)")
    parser.add_argument("--output-physical-tensors", type=str, default="auto",
                        choices=["auto", "true", "false"],
                        help="Output physical tensors: 'auto'=use checkpoint's inference_output_physical_tensors, "
                             "'true'=always, 'false'=never (MD/LAMMPS 仅需能量和力时用 false)")
    args = parser.parse_args()

    # NCCL requires setting GPU before init_process_group (avoid device unknown warning / potential hang)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    backend = args.backend or ("nccl" if torch.cuda.is_available() else "gloo")
    try:
        dist.init_process_group(backend=backend)
    except Exception as e:
        print("DDP requires torchrun, e.g.: torchrun --nproc_per_node=2 -m molecular_force_field.cli.inference_ddp --atoms 100000")
        raise SystemExit(1) from e

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', rank)}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    def _infer_physical_tensor_outputs_from_state_dict(sd: dict) -> dict | None:
        per_name: dict[str, dict[int, int]] = {}
        pat = re.compile(r"^physical_tensor_heads\.([^.]+)\.(\d+)\.weight$")
        for k, v in sd.items():
            m = pat.match(k)
            if not m:
                continue
            name = m.group(1)
            l = int(m.group(2))
            ch_out = int(v.shape[0]) if hasattr(v, "shape") and len(v.shape) >= 1 else 1
            per_name.setdefault(name, {})[l] = ch_out
        if not per_name:
            return None
        out = {}
        for name, ch_by_l in per_name.items():
            ls = sorted(ch_by_l.keys())
            out[name] = {
                "ls": ls,
                "channels_out": {l: ch_by_l[l] for l in ls},
                "reduce": "sum",
            }
        return out

    # Config and model (aligned with train/evaluate)
    cfg = dict(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=64,
        hidden_dim_sh=64,
        hidden_dim=64,
        channel_in2=32,
        embedding_dim=16,
        max_atomvalue=10,
        output_size=8,
        num_interaction=2,
        lmax=2,
        ictd_tp_path_policy="full",
        internal_compute_dtype=dtype,
    )

    ckpt = None
    physical_tensor_outputs = None
    external_tensor_rank = None
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("e3trans_state_dict") or ckpt.get("state_dict") or ckpt
        physical_tensor_outputs = ckpt.get("physical_tensor_outputs")
        if physical_tensor_outputs is None:
            physical_tensor_outputs = _infer_physical_tensor_outputs_from_state_dict(state)
        external_tensor_rank = ckpt.get("external_tensor_rank")
        if external_tensor_rank is None and "e3_conv_emb.external_tensor_scale_by_l" in state:
            external_tensor_rank = 1

    model = PureCartesianICTDTransformerLayer(
        **cfg,
        physical_tensor_outputs=physical_tensor_outputs,
        external_tensor_rank=external_tensor_rank,
    ).to(device=device, dtype=dtype)

    if ckpt is not None:
        state = ckpt.get("e3trans_state_dict") or ckpt.get("state_dict") or ckpt
        model.load_state_dict(state, strict=True)
        if rank == 0:
            print(f"Loaded checkpoint: {args.checkpoint}")
    dist.barrier()

    # Resolve output_physical_tensors: auto=from checkpoint, true/false=explicit
    if args.output_physical_tensors == "auto":
        output_physical = (ckpt or {}).get("inference_output_physical_tensors", False)
    else:
        output_physical = (args.output_physical_tensors == "true")

    if args.forces:
        time_ms, total_energy, forces = run_one_ddp_inference(
            args.atoms,
            model,
            avg_degree=args.avg_degree,
            seed=args.seed,
            device=device,
            dtype=dtype,
            return_forces=True,
            partition_mode=args.partition,
            output_physical_tensors=output_physical,
        )
        if rank == 0:
            print(f"DDP inference: {args.atoms} atoms, {time_ms:.2f} ms, total_energy={total_energy:.6f}")
            print(f"forces shape: {forces.shape}")
            print("forces (first 3 atoms):")
            print(forces[:3].cpu().numpy())
    else:
        time_ms, total_energy = run_one_ddp_inference(
            args.atoms,
            model,
            avg_degree=args.avg_degree,
            seed=args.seed,
            device=device,
            dtype=dtype,
            partition_mode=args.partition,
            output_physical_tensors=output_physical,
        )
        if rank == 0:
            print(f"DDP inference: {args.atoms} atoms, {time_ms:.2f} ms, total_energy={total_energy:.6f}")

    dist.destroy_process_group()
    return total_energy if rank == 0 else None


if __name__ == "__main__":
    main()
