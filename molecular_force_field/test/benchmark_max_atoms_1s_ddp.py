#!/usr/bin/env python3
"""
双卡（或 N 卡）DDP 下：在 target 时间内单次推理能支持的最大原子数（二分搜索）。

与 benchmark_max_atoms_1s 相同配置：lmax=2, channels=64, num_interaction=2, float32。
需用 torchrun 启动，例如双卡：
  torchrun --nproc_per_node=2 -m molecular_force_field.benchmark_max_atoms_1s_ddp
  torchrun --nproc_per_node=2 -m molecular_force_field.benchmark_max_atoms_1s_ddp --target-ms 2000 --max-atoms-cap 500000
  torchrun --nproc_per_node=2 -m molecular_force_field.benchmark_max_atoms_1s_ddp --forces   # 找到最大 N 后再跑一次带力输出

测「不爆显存的最大 N」：每次测量在子进程中执行，OOM 只杀子进程，主进程继续二分。
  torchrun --nproc_per_node=2 -m molecular_force_field.benchmark_max_atoms_1s_ddp --oom-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
import datetime
import time

import torch
import torch.distributed as dist

# Fix PyTorch 2.6 weights_only=True issue with e3nn constants.pt which uses slice
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([slice])

from molecular_force_field.cli.inference_ddp import (
    build_and_broadcast_graph,
    run_ddp_forward_timed,
    run_one_ddp_inference,
)
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


def _run_measure_only(args, n: int) -> int:
    """内部测试端：隔离进程，防 OOM 崩溃。成功: 0，OOM: 1，限时超时: 2。"""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    backend = args.backend or ("nccl" if torch.cuda.is_available() else "gloo")
    
    # ==== 强力防止单卡死锁：NCCL Timeout ====
    # 如果其中一张卡 OOM 了退出了，另外一张卡最多等 15 秒也会安全退出。
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=15))
    
    rank = dist.get_rank()
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', rank)}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if torch.cuda.is_available():
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
    if args.model == 'spherical':
        ModelClass = E3_TransformerLayer_multi
    elif args.model == 'spherical-save':
        ModelClass = E3_TransformerLayer_multi_channelwise
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
    model = ModelClass(**cfg).to(device=device, dtype=dtype)
    dist.barrier()

    try:
        comm_timing = {} if args.comm_timing else None
        pos, A, batch, edge_src, edge_dst, edge_shifts, cell = build_and_broadcast_graph(
            n, args.avg_degree, args.seed, device, dtype, comm_timing=comm_timing
        )
        
        cache = {}
        times = []
        target_ms = args.target_ms
        
        for step in range(args.warmup + args.repeat):
            t_ms, _ = run_ddp_forward_timed(
                pos, A, batch, edge_src, edge_dst, edge_shifts, cell,
                model, device, dtype, cache=cache,
                comm_timing=comm_timing if (args.comm_timing and step == args.warmup) else None,
            )
            
            # 使用 DDP 同步时间（因为只有 rank 0 实际计时，且防某些节点拖慢引起 NCCL Desync）
            t_ms_t = torch.tensor([t_ms], device=device, dtype=torch.float)
            dist.all_reduce(t_ms_t, op=dist.ReduceOp.MAX)
            sync_t_ms = t_ms_t.item()
            
            # 如果某次前向耗时大于 target_ms (例如 2000 ms = 2s)，提早退出（视为不合格）
            if target_ms > 0 and sync_t_ms > target_ms:
                if rank == 0:
                     print(f"  [子卡 N={n}] 前向跑出 {sync_t_ms:.2f} ms > 设置的超时阈值 {target_ms:.0f} ms 限制，提早打断！")
                dist.destroy_process_group()
                return 2

            if step >= args.warmup:
                if rank == 0:
                    times.append(sync_t_ms)
                    
        if rank == 0:
            avg_time = sum(times) / len(times) if times else 0.0
            print(f"  [子卡 N={n}] 测试总平均跑通耗时: {avg_time:.2f} ms")
            
    except Exception as e:
        if rank == 0:
            print(f"  [子卡 N={n}] => 发生了 OOM 崩毁或其它运行时错误。")
        dist.destroy_process_group()
        return 1
        
    dist.destroy_process_group()
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="DDP 下求 inference <= target ms 的最大原子数（二分）或 --oom-only 求不爆显存的最大 N"
    )
    parser.add_argument("--model", args.model,
                "--measure-only", type=int, default=None,
                        help="内部用：只跑一次 N 原子，成功 exit 0，OOM/异常 exit 1")
    parser.add_argument("--oom-only", action="store_true",
                        help="求不爆显存 的 安全最大原子数（且满足 target ms 限制）")
    parser.add_argument("--target-ms", type=float, default=2000.0,
                        help="目标推理时间上限 ms（默认 2000 = 2s）。oom-only 下也作为耗时过滤拦截")
    parser.add_argument("--model", type=str, default="pure-cartesian-ictd-save", choices=['spherical', 'spherical-save', 'partial-cartesian', 'partial-cartesian-loose', 'pure-cartesian', 'pure-cartesian-sparse', 'pure-cartesian-ictd', 'pure-cartesian-ictd-save', 'mace'], help="Select the model architecture to benchmark.")
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--num-interaction", type=int, default=2)
    parser.add_argument("--avg-degree", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=2,
                        help="每个 N 的 warmup 次数（--oom-only 时子进程仅跑 1 次前向）")
    parser.add_argument("--repeat", type=int, default=4,
                        help="每个 N 的计时重复次数")
    parser.add_argument("--max-atoms-cap", type=int, default=400_000,
                        help="二分上界")
    parser.add_argument("--forces", action="store_true",
                        help="找到最大 N 后，再跑一次带力输出并打印（非 --oom-only 时有效）")
    parser.add_argument("--comm-timing", action="store_true",
                        help="打印通信耗时：broadcast_ms、sync_ms（每层）、all_reduce_energy_ms（非 --oom-only 时每 N 打印一次）")
    parser.add_argument("--backend", type=str, default=None)
    args = parser.parse_args()

    # -------- --measure-only：子进程单次测量，用于 OOM 探测 --------
    if args.measure_only is not None:
        exitcode = _run_measure_only(args, args.measure_only)
        sys.exit(exitcode)


    # -------- 主进程：二分搜索 --------
    use_oom_only = args.oom_only

    if use_oom_only:
        # 主进程用 gloo、不占 GPU，方便子进程独占 GPU；二分只按「是否 OOM」判断
        backend = "gloo"
        if torch.cuda.is_available():
            # 仍设置 LOCAL_RANK 以便子进程继承，主进程不用 GPU
            pass
        try:
            dist.init_process_group(backend=backend)
        except Exception as e:
            print("需用 torchrun 启动，例如: torchrun --nproc_per_node=2 -m molecular_force_field.benchmark_max_atoms_1s_ddp --oom-only")
            raise SystemExit(1) from e
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("cpu")
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
    else:
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
        try:
            backend = args.backend or ("nccl" if torch.cuda.is_available() else "gloo")
            dist.init_process_group(backend=backend)
        except Exception as e:
            print("需用 torchrun 启动，例如: torchrun --nproc_per_node=2 -m molecular_force_field.benchmark_max_atoms_1s_ddp")
            raise SystemExit(1) from e
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', rank)}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
        dtype = torch.float32
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
        if args.ictd_full:
            model = PureCartesianICTDFull(**cfg).to(device=device, dtype=dtype)
        else:
            model = PureCartesianICTDSave(**cfg).to(device=device, dtype=dtype)
        dist.barrier()

    if rank == 0:
        if use_oom_only:
            print(f"Model: {args.model}")
            print(f"Config: lmax={args.lmax}, channels={args.channels}, num_interaction={args.num_interaction}")
            print(f"Mode: --oom-only (find max N that does not OOM), world_size={world_size}")
            print(f"max_atoms_cap={args.max_atoms_cap}")
        else:
            print(f"Model: {args.model}")
            print(f"Config: lmax={args.lmax}, channels={args.channels}, num_interaction={args.num_interaction}, float32")
            print(f"Target: inference <= {args.target_ms:.0f} ms ({args.target_ms/1000:.2f} s), world_size={world_size}")
            print(f"avg_degree={args.avg_degree}, warmup={args.warmup}, repeat={args.repeat}")
        print()

    def measure_ms_in_process(n: int) -> float:
        """仅非 oom_only 时用：本进程内测量（会因 OOM 直接崩）。"""
        comm_timing = {} if args.comm_timing else None
        pos, A, batch, edge_src, edge_dst, edge_shifts, cell = build_and_broadcast_graph(
            n, args.avg_degree, args.seed, device, dtype, comm_timing=comm_timing
        )
        cache = {}
        for _ in range(args.warmup):
            run_ddp_forward_timed(
                pos, A, batch, edge_src, edge_dst, edge_shifts, cell,
                model, device, dtype, cache=cache,
            )
        times = []
        for i in range(args.repeat):
            t_ms, _ = run_ddp_forward_timed(
                pos, A, batch, edge_src, edge_dst, edge_shifts, cell,
                model, device, dtype, cache=cache,
                comm_timing=comm_timing if (args.comm_timing and i == 0) else None,
            )
            if rank == 0:
                times.append(t_ms)
        if rank == 0 and comm_timing:
            sync_ms = comm_timing.get("sync_ms", [])
            total_sync_ms = sum(sync_ms)
            print(f"  [comm_timing N={n}] broadcast_ms={comm_timing.get('broadcast_ms', 0):.2f}  "
                  f"sync_ms={[f'{x:.2f}' for x in sync_ms]} total_sync_ms={total_sync_ms:.2f}  "
                  f"all_reduce_energy_ms={comm_timing.get('all_reduce_energy_ms', 0):.2f}")
        if rank == 0:
            return sum(times) / len(times) if times else float("inf")
        return 0.0

    low, high = 1, args.max_atoms_cap
    best_n = 0
    best_time = 0.0
    if use_oom_only:
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        def spawn_measure_and_get_failed(mid: int) -> int:
            child_env = os.environ.copy()
            # 内部单独通信口，错开 29500
            child_env["MASTER_PORT"] = str(master_port + 1)
            worker_cmd = [
                sys.executable, "-m", "molecular_force_field.benchmark_max_atoms_1s_ddp",
                "--model", args.model,
                "--measure-only", str(mid),
                "--target-ms", str(args.target_ms),
                "--lmax", str(args.lmax),
                "--channels", str(args.channels),
                "--num-interaction", str(args.num_interaction),
                "--avg-degree", str(args.avg_degree),
                "--seed", str(args.seed),
                "--warmup", str(args.warmup),
                "--repeat", str(args.repeat),
            ]
            if args.backend:
                worker_cmd += ["--backend", args.backend]
            if args.comm_timing:
                worker_cmd += ["--comm-timing"]

            proc = subprocess.Popen(worker_cmd, env=child_env, stdout=sys.stdout, stderr=sys.stderr)
            ret = proc.wait()
            # 0=ok, 1=oom/err, 2=timeout
            fail_t = torch.tensor([ret], device=device, dtype=torch.long)
            dist.all_reduce(fail_t, op=dist.ReduceOp.MAX)
            return fail_t.item()
    else:
        def spawn_measure_and_get_failed(mid: int) -> bool:
            """占位，非 oom_only 时不使用。"""
            return False

    while low <= high:
        mid = (low + high) // 2
        if use_oom_only:
            ret_status = spawn_measure_and_get_failed(mid)
            if ret_status == 1:
                if rank == 0: print(f"  N={mid:>6}  => OOM 或运行异常崩毁")
                high = mid - 1
            elif ret_status == 2:
                if rank == 0: print(f"  N={mid:>6}  => 耗时超标了 (> {args.target_ms} ms)")
                high = mid - 1
            else:
                best_n = mid
                low = mid + 1
                if rank == 0: print(f"  N={mid:>6}  => 安全无 OOM，且耗时合格 (<= {args.target_ms} ms) -> 更新最佳极限")
            best_n_t = torch.tensor([best_n], device=device, dtype=torch.long)
            low_t = torch.tensor([low], device=device, dtype=torch.long)
            high_t = torch.tensor([high], device=device, dtype=torch.long)
            dist.broadcast(best_n_t, src=0)
            dist.broadcast(low_t, src=0)
            dist.broadcast(high_t, src=0)
            best_n = int(best_n_t.item())
            low = int(low_t.item())
            high = int(high_t.item())
            continue

        # 非 oom_only：本进程内测量（可能 OOM 导致整组退出）
        try:
            t_ms = measure_ms_in_process(mid)
        except Exception as e:
            if rank == 0:
                print(f"  N={mid} failed: {e}")
            dist.destroy_process_group()
            raise

        if rank == 0:
            if t_ms <= args.target_ms:
                best_n = mid
                best_time = t_ms
                low = mid + 1
                print(f"  N={mid:>6}  {t_ms:>8.2f} ms  OK (best so far)")
            else:
                high = mid - 1
                print(f"  N={mid:>6}  {t_ms:>8.2f} ms  > {args.target_ms:.0f} ms")

        best_n_t = torch.tensor([best_n], device=device, dtype=torch.long)
        low_t = torch.tensor([low], device=device, dtype=torch.long)
        high_t = torch.tensor([high], device=device, dtype=torch.long)
        dist.broadcast(best_n_t, src=0)
        dist.broadcast(low_t, src=0)
        dist.broadcast(high_t, src=0)
        best_n = int(best_n_t.item())
        low = int(low_t.item())
        high = int(high_t.item())



    if rank == 0:
        print()
        if best_n == 0:
            if use_oom_only:
                print("未在 cap 内找到既不 OOM 且跑进目标限时范围的 N")
            else:
                print("未在 cap 内找到满足条件的 N；可增大 --max-atoms-cap 或 --target-ms")
        else:
            if use_oom_only:
                print(f"Result (DDP world_size={world_size}): max atoms (No OOM & 耗时 <= {args.target_ms}ms) = {best_n}")
            else:
                print(f"Result (DDP world_size={world_size}): max atoms (inference <= {args.target_ms:.0f} ms) = {best_n}")
                print(f"  measured at N={best_n}: {best_time:.2f} ms")
                print(f"  edges at N={best_n}: {best_n * args.avg_degree}")

    if not use_oom_only and args.forces and best_n > 0:
        if rank == 0:
            print("\nRunning once with --forces at N={}...".format(best_n))
        _, total_energy, forces = run_one_ddp_inference(
            best_n,
            model,
            avg_degree=args.avg_degree,
            seed=args.seed,
            device=device,
            dtype=dtype,
            return_forces=True,
        )
        if rank == 0:
            print(f"  total_energy = {total_energy:.6f}")
            print(f"  forces shape: {forces.shape}")
            print("  forces (first 3 atoms):")
            print(forces[:3].cpu().numpy())

    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
