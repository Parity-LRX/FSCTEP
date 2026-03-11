"""Train multiple models with different random seeds for active learning ensemble.

Supports:
* Single-process training (CPU / single GPU)
* Single-node multi-GPU DDP  (``torchrun`` with unique rendezvous port)
* Multi-node multi-GPU DDP   (``torchrun`` + ``srun``, per-model node subsets)
* **Cross-node distribution**: ``nnodes=1`` per model, but models are spread
  across multiple SLURM nodes with disjoint GPU assignments
* Parallel training of independent ensemble models on non-overlapping resources
"""

import glob
import logging
import os
import socket
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_gpu_pool() -> List[str]:
    """Return list of physical GPU IDs available on the **local** node.

    Respects ``CUDA_VISIBLE_DEVICES`` if set; otherwise queries
    ``torch.cuda.device_count()``.
    """
    env_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if env_gpus:
        return [g.strip() for g in env_gpus.split(",") if g.strip()]
    try:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def _get_slurm_nodes() -> List[str]:
    """Parse ``SLURM_NODELIST`` into a flat list of hostnames."""
    nodelist = os.environ.get("SLURM_NODELIST", "")
    if not nodelist:
        return []
    try:
        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")
    except FileNotFoundError:
        pass
    return []


def _resolve_master_addr(master_addr: str) -> str:
    """Resolve ``'auto'`` to a usable hostname / IP."""
    if master_addr != "auto":
        return master_addr
    slurm_nodes = _get_slurm_nodes()
    if slurm_nodes:
        return slurm_nodes[0]
    return socket.gethostname()


# ---------------------------------------------------------------------------
# Slot: all info needed to launch one model training subprocess
# ---------------------------------------------------------------------------

def _build_slots_local(
    n_gpu: int, gpu_pool: List[str], max_parallel: int,
    n_models: int, master_port: int,
) -> Tuple[int, List[Dict]]:
    """Build slots for single-node local training."""
    n_available = len(gpu_pool)
    if max_parallel <= 0:
        if n_available > 0 and n_gpu > 0:
            max_parallel = max(1, n_available // n_gpu)
        else:
            max_parallel = 1
    max_parallel = min(max_parallel, n_models)

    slots = []
    for s in range(max_parallel):
        if gpu_pool and n_gpu > 0:
            start = s * n_gpu
            gpu_ids = gpu_pool[start: start + n_gpu]
        else:
            gpu_ids = None
        slots.append({
            "nodelist": None,
            "gpu_ids": gpu_ids,
            "rdzv_addr": "localhost",
            "rdzv_port": master_port + s,
            "nnodes": 1,
            "tag": f"GPU {','.join(gpu_ids)}" if gpu_ids else "CPU",
        })
    return max_parallel, slots


def _build_slots_cross_node(
    n_gpu: int, gpu_pool: List[str], slurm_nodes: List[str],
    max_parallel: int, n_models: int, master_port: int,
) -> Tuple[int, List[Dict]]:
    """Build slots for cross-node distribution (nnodes=1 per model,
    models spread across multiple SLURM nodes).

    Example: 2 nodes × 8 GPU, n_gpu=4 → 4 slots total
      slot 0: node0, GPU[0,1,2,3]
      slot 1: node0, GPU[4,5,6,7]
      slot 2: node1, GPU[0,1,2,3]
      slot 3: node1, GPU[4,5,6,7]
    """
    gpus_per_node = len(gpu_pool)
    models_per_node = gpus_per_node // n_gpu if n_gpu > 0 else 1

    slots = []
    for node in slurm_nodes:
        for ls in range(models_per_node):
            start = ls * n_gpu
            gpu_ids = gpu_pool[start: start + n_gpu]
            port = master_port + len(slots)
            slots.append({
                "nodelist": [node],
                "gpu_ids": gpu_ids,
                "rdzv_addr": "localhost",
                "rdzv_port": port,
                "nnodes": 1,
                "tag": f"{node}:GPU[{','.join(gpu_ids)}]",
            })

    total = len(slots)
    if max_parallel <= 0:
        max_parallel = total
    max_parallel = min(max_parallel, n_models, total)

    logger.info(
        f"Cross-node distribution: {len(slurm_nodes)} nodes × "
        f"{models_per_node} model(s)/node = {total} slot(s), "
        f"{max_parallel} model(s) in parallel"
    )
    return max_parallel, slots


def _build_slots_multi_node(
    n_gpu: int, nnodes: int, slurm_nodes: List[str],
    max_parallel: int, n_models: int, master_port: int,
    master_addr: str, launcher: str,
) -> Tuple[int, List[Dict]]:
    """Build slots for multi-node DDP (nnodes > 1 per model)."""
    if launcher == "slurm" and slurm_nodes:
        total_nodes = len(slurm_nodes)
        if total_nodes < nnodes:
            raise RuntimeError(
                f"Requested {nnodes} nodes per model but SLURM only allocated "
                f"{total_nodes} node(s). Reduce --train-nnodes."
            )
        auto_parallel = total_nodes // nnodes
        if max_parallel <= 0:
            max_parallel = auto_parallel
        max_parallel = min(max_parallel, n_models, auto_parallel)

        slots = []
        for s in range(max_parallel):
            node_start = s * nnodes
            subset = slurm_nodes[node_start: node_start + nnodes]
            slots.append({
                "nodelist": subset,
                "gpu_ids": None,
                "rdzv_addr": subset[0],
                "rdzv_port": master_port + s,
                "nnodes": nnodes,
                "tag": f"nodes[{','.join(subset)}]×{n_gpu}G",
            })

        logger.info(
            f"Multi-node parallel: {total_nodes} total nodes, "
            f"{nnodes} nodes/model → {max_parallel} model(s) in parallel"
        )
    else:
        max_parallel = 1
        slots = [{
            "nodelist": None,
            "gpu_ids": None,
            "rdzv_addr": master_addr,
            "rdzv_port": master_port,
            "nnodes": nnodes,
            "tag": f"{nnodes}N×{n_gpu}G",
        }]

    return max_parallel, slots


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def _build_cmd(
    n_gpu: int,
    mff_args: List[str],
    *,
    nnodes: int = 1,
    rdzv_addr: str = "localhost",
    rdzv_port: int = 29500,
    nodelist: Optional[List[str]] = None,
) -> List[str]:
    """Build the subprocess command for a single model training run.

    When *nodelist* is provided, the command is wrapped with
    ``srun --nodes=N --ntasks-per-node=1 --nodelist=...``.
    """
    if n_gpu <= 1 and nnodes <= 1:
        cmd = [
            sys.executable, "-m", "molecular_force_field.cli.train",
        ] + mff_args
    else:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(n_gpu),
            "--nnodes", str(nnodes),
            "--rdzv_backend", "c10d",
            "--rdzv_endpoint", f"{rdzv_addr}:{rdzv_port}",
            "-m", "molecular_force_field.cli.train",
            "--distributed",
        ] + mff_args

    if nodelist:
        cmd = [
            "srun",
            "--nodes", str(nnodes),
            "--ntasks-per-node", "1",
            "--nodelist", ",".join(nodelist),
        ] + cmd

    return cmd


def _collect_checkpoint(ckpt_subdir: str, model_idx: int) -> str:
    """Find the saved checkpoint for *model_idx*, raise if missing."""
    pattern = os.path.join(ckpt_subdir, f"model_{model_idx}_*.pth")
    found = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if found:
        return found[0]
    fallback = os.path.join(ckpt_subdir, f"model_{model_idx}_spherical.pth")
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Checkpoint not found: {pattern}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_ensemble(
    data_dir: str,
    work_dir: str,
    n_models: int = 4,
    base_seed: int = 42,
    train_args: Optional[List[str]] = None,
    n_gpu: int = 1,
    max_parallel: int = 0,
    nnodes: int = 1,
    master_addr: str = "auto",
    master_port: int = 29500,
    launcher: str = "auto",
) -> List[str]:
    """Train *n_models* with different random seeds, optionally in parallel.

    Resource assignment modes
    -------------------------
    1. **Local** (``nnodes=1``, no SLURM multi-node):
       GPU pool on the local node is partitioned across models.

    2. **Cross-node** (``nnodes=1``, SLURM allocated > 1 node):
       Models are distributed across SLURM nodes.  Each node's GPUs
       are partitioned the same way as local mode.  E.g. 2 nodes ×
       8 GPU, ``n_gpu=4`` → 4 parallel slots (2 per node).

    3. **Multi-node** (``nnodes > 1``):
       Each model uses *nnodes* nodes.  Total SLURM nodes are split
       into disjoint subsets for parallel training.

    DDP port management: each concurrent model gets ``master_port + slot``.

    Args:
        data_dir: Data directory (processed_train.h5, processed_val.h5).
        work_dir: Output directory for checkpoints.
        n_models: Number of models to train.
        base_seed: Base seed; model *j* uses ``base_seed + j``.
        train_args: Extra args forwarded to ``mff-train``.
        n_gpu: GPUs per model **per node**.  1 = single-process, >1 = DDP.
        max_parallel: Max models trained simultaneously.  0 = auto.
        nnodes: Nodes **per model**.  1 = single-node (default).
        master_addr: Rendezvous address (``'auto'``).
        master_port: Base rendezvous port (default 29500).
        launcher: ``'auto'`` / ``'local'`` / ``'slurm'``.

    Returns:
        List of checkpoint paths (one per model).
    """
    os.makedirs(work_dir, exist_ok=True)
    train_args = train_args or []
    data_dir_abs = os.path.abspath(data_dir)
    ckpt_subdir = os.path.join(work_dir, "checkpoint")
    log_dir = os.path.join(work_dir, "train_logs")
    os.makedirs(log_dir, exist_ok=True)

    # ---- Resolve launcher ----
    if launcher == "auto":
        if os.environ.get("SLURM_JOB_ID"):
            launcher = "slurm"
        else:
            launcher = "local"

    master_addr = _resolve_master_addr(master_addr)

    # ---- GPU pool (local node) ----
    gpu_pool = _get_gpu_pool()
    n_available = len(gpu_pool)

    if n_gpu > 1 and nnodes <= 1 and n_available < n_gpu:
        raise RuntimeError(
            f"Requested {n_gpu} GPUs per model but only {n_available} GPU(s) "
            f"detected (pool={gpu_pool}). Reduce --train-n-gpu or check "
            f"CUDA_VISIBLE_DEVICES."
        )

    # ---- Build training slots ----
    slurm_nodes = _get_slurm_nodes() if launcher == "slurm" else []

    if nnodes > 1:
        max_parallel, slots = _build_slots_multi_node(
            n_gpu, nnodes, slurm_nodes,
            max_parallel, n_models, master_port, master_addr, launcher,
        )
    elif launcher == "slurm" and len(slurm_nodes) > 1 and n_available > 0 and n_gpu > 0:
        max_parallel, slots = _build_slots_cross_node(
            n_gpu, gpu_pool, slurm_nodes,
            max_parallel, n_models, master_port,
        )
    else:
        max_parallel, slots = _build_slots_local(
            n_gpu, gpu_pool, max_parallel, n_models, master_port,
        )

    mode_desc = (
        f"{n_gpu} GPU(s)/model"
        + (f" × {nnodes} node(s)" if nnodes > 1 else "")
        + f", {max_parallel} model(s) in parallel"
    )
    logger.info(f"Ensemble training: {n_models} models — {mode_desc}")

    # ---- Train in batches ----
    checkpoint_paths: List[str] = [None] * n_models  # type: ignore[list-item]

    for batch_start in range(0, n_models, max_parallel):
        batch_end = min(batch_start + max_parallel, n_models)
        batch_size = batch_end - batch_start
        procs: List[tuple] = []

        for slot_idx, j in enumerate(range(batch_start, batch_end)):
            slot = slots[slot_idx]
            seed = base_seed + j
            ckpt_path = os.path.join(work_dir, f"model_{j}.pth")
            mff_args = [
                "--data-dir", data_dir_abs,
                "--checkpoint", ckpt_path,
                "--seed", str(seed),
            ] + train_args

            cmd = _build_cmd(
                n_gpu, mff_args,
                nnodes=slot["nnodes"],
                rdzv_addr=slot["rdzv_addr"],
                rdzv_port=slot["rdzv_port"],
                nodelist=slot["nodelist"],
            )

            env = os.environ.copy()
            if slot["gpu_ids"]:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(slot["gpu_ids"])

            gpu_tag = slot["tag"]
            log_file = os.path.join(log_dir, f"model_{j}.log")
            fh = open(log_file, "w")

            if batch_size > 1:
                logger.info(
                    f"  [batch {batch_start // max_parallel}] "
                    f"Launching model {j}/{n_models} (seed={seed}, {gpu_tag}, "
                    f"rdzv_port={slot['rdzv_port']}) → {log_file}"
                )
            else:
                logger.info(
                    f"Training model {j}/{n_models} "
                    f"(seed={seed}, {gpu_tag})..."
                )

            proc = subprocess.Popen(
                cmd, cwd=work_dir, env=env,
                stdout=fh, stderr=subprocess.STDOUT,
            )
            procs.append((j, proc, fh, log_file))

        # Wait for all processes in this batch
        errors = []
        for j, proc, fh, log_file in procs:
            ret = proc.wait()
            fh.close()
            if ret != 0:
                errors.append((j, ret, log_file))
            else:
                checkpoint_paths[j] = _collect_checkpoint(ckpt_subdir, j)

        if errors:
            msgs = []
            for j, rc, lf in errors:
                tail = ""
                try:
                    with open(lf) as f:
                        lines = f.readlines()
                        tail = "".join(lines[-20:])
                except Exception:
                    pass
                msgs.append(
                    f"  model {j} (exit {rc}), log: {lf}\n{tail}"
                )
            raise RuntimeError(
                f"Training failed for {len(errors)} model(s):\n"
                + "\n".join(msgs)
            )

        if batch_size > 1:
            logger.info(
                f"  Batch {batch_start // max_parallel} done: "
                f"models {batch_start}–{batch_end - 1} completed."
            )

    return checkpoint_paths
