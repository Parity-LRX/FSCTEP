#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import torch

from molecular_force_field.models import ictd_irreps_cuda
from molecular_force_field.models.ictd_irreps import HarmonicFullyConnectedTensorProduct
from molecular_force_field.models.ictd_irreps_cuda import (
    _mix_forward_op,
    _mix_forward_reference,
    _project_forward_op,
    _project_forward_reference,
)
from molecular_force_field.models.pure_cartesian_ictd_layers import PureCartesianICTDTransformerLayer


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _zero_tp_grads(
    tp: HarmonicFullyConnectedTensorProduct,
    x1: dict[int, torch.Tensor],
    x2: dict[int, torch.Tensor],
    gates: torch.Tensor,
) -> None:
    for t in list(x1.values()) + list(x2.values()):
        if t.grad is not None:
            t.grad = None
    if gates.grad is not None:
        gates.grad = None
    if tp.weight is not None and tp.weight.grad is not None:
        tp.weight.grad = None


def _make_tp_inputs(
    *,
    batch: int,
    mul_in1: int,
    mul_in2: int,
    lmax: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    x1 = {
        l: torch.randn(batch, mul_in1, 2 * l + 1, device=device, dtype=dtype, requires_grad=requires_grad)
        for l in range(lmax + 1)
    }
    x2 = {
        l: torch.randn(batch, mul_in2, 2 * l + 1, device=device, dtype=dtype, requires_grad=requires_grad)
        for l in range(lmax + 1)
    }
    return x1, x2


def _backend_uses_native_ops(backend: str, device: torch.device) -> bool:
    if backend == "pytorch":
        return False
    if device.type != "cuda":
        return False
    if not ictd_irreps_cuda.cuda_ext_available() or not ictd_irreps_cuda.cuda_ext_has_cuda():
        return False
    if backend == "auto":
        return ictd_irreps_cuda.auto_experimental_cuda_enabled()
    return backend == "cuda_ext"


def _bench_projection_backend(
    *,
    backend: str,
    batch: int,
    mul_in1: int,
    mul_in2: int,
    num_paths: int,
    kdim: int,
    warmup: int,
    repeat: int,
    do_backward: bool,
    do_double_backward: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    m1 = 3
    m2 = 3
    a = torch.randn(batch, mul_in1, m1, device=device, dtype=dtype, requires_grad=(do_backward or do_double_backward))
    b = torch.randn(batch, mul_in2, m2, device=device, dtype=dtype, requires_grad=(do_backward or do_double_backward))
    u = torch.randn(m1 * m2, num_paths * kdim, device=device, dtype=dtype, requires_grad=(do_backward or do_double_backward))
    use_native = _backend_uses_native_ops(backend, device) and mul_in2 == 1

    for _ in range(warmup):
        y = _project_forward_op(a, b, u, num_paths) if use_native else _project_forward_reference(a, b, u, num_paths)
        loss = y.square().mean()
        if do_double_backward:
            first = torch.autograd.grad(loss, (a, b, u), create_graph=True, retain_graph=True)
            second = sum(g.square().mean() for g in first if g is not None)
            second.backward()
        elif do_backward:
            loss.backward()
        for t in (a, b, u):
            if t.grad is not None:
                t.grad = None

    _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeat):
        y = _project_forward_op(a, b, u, num_paths) if use_native else _project_forward_reference(a, b, u, num_paths)
        loss = y.square().mean()
        if do_double_backward:
            first = torch.autograd.grad(loss, (a, b, u), create_graph=True, retain_graph=True)
            second = sum(g.square().mean() for g in first if g is not None)
            second.backward()
        elif do_backward:
            loss.backward()
        for t in (a, b, u):
            if t.grad is not None:
                t.grad = None
    _sync(device)
    return (time.perf_counter() - t0) / repeat * 1000.0


def _bench_mix_backend(
    *,
    backend: str,
    batch: int,
    num_paths: int,
    kdim: int,
    ij: int,
    mul_out: int,
    warmup: int,
    repeat: int,
    do_backward: bool,
    do_double_backward: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    y = torch.randn(batch, num_paths, kdim, ij, device=device, dtype=dtype, requires_grad=(do_backward or do_double_backward))
    w = torch.randn(num_paths, mul_out, ij, device=device, dtype=dtype, requires_grad=(do_backward or do_double_backward))
    g = torch.randn(batch, num_paths, device=device, dtype=dtype, requires_grad=(do_backward or do_double_backward))
    use_native = _backend_uses_native_ops(backend, device)

    for _ in range(warmup):
        out, _ = _mix_forward_op(y, w, g) if use_native else _mix_forward_reference(y, w, g)
        loss = out.square().mean()
        if do_double_backward:
            first = torch.autograd.grad(loss, (y, w, g), create_graph=True, retain_graph=True)
            second = sum(t.square().mean() for t in first if t is not None)
            second.backward()
        elif do_backward:
            loss.backward()
        for t in (y, w, g):
            if t.grad is not None:
                t.grad = None

    _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeat):
        out, _ = _mix_forward_op(y, w, g) if use_native else _mix_forward_reference(y, w, g)
        loss = out.square().mean()
        if do_double_backward:
            first = torch.autograd.grad(loss, (y, w, g), create_graph=True, retain_graph=True)
            second = sum(t.square().mean() for t in first if t is not None)
            second.backward()
        elif do_backward:
            loss.backward()
        for t in (y, w, g):
            if t.grad is not None:
                t.grad = None
    _sync(device)
    return (time.perf_counter() - t0) / repeat * 1000.0


def _bench_tp_backend(
    *,
    backend: str,
    batch: int,
    lmax: int,
    mul_in1: int,
    mul_in2: int,
    mul_out: int,
    warmup: int,
    repeat: int,
    do_backward: bool,
    do_double_backward: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    tp = HarmonicFullyConnectedTensorProduct(
        mul_in1=mul_in1,
        mul_in2=mul_in2,
        mul_out=mul_out,
        lmax=lmax,
        internal_weights=True,
        internal_compute_dtype=dtype,
        ictd_tp_backend=backend,
    ).to(device=device, dtype=dtype)

    x1, x2 = _make_tp_inputs(
        batch=batch,
        mul_in1=mul_in1,
        mul_in2=mul_in2,
        lmax=lmax,
        device=device,
        dtype=dtype,
        requires_grad=(do_backward or do_double_backward),
    )
    gates = torch.randn(
        batch,
        tp.num_paths,
        device=device,
        dtype=dtype,
        requires_grad=(do_backward or do_double_backward),
    )

    for _ in range(warmup):
        out = tp(x1, x2, gates)
        loss = sum(v.square().mean() for v in out.values())
        if do_double_backward:
            grad_targets = tuple(x1.values()) + tuple(x2.values()) + (tp.weight, gates)
            first = torch.autograd.grad(loss, grad_targets, create_graph=True, retain_graph=True)
            second = sum(g.square().mean() for g in first if g is not None)
            second.backward()
        elif do_backward:
            loss.backward()
        _zero_tp_grads(tp, x1, x2, gates)

    _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = tp(x1, x2, gates)
        loss = sum(v.square().mean() for v in out.values())
        if do_double_backward:
            grad_targets = tuple(x1.values()) + tuple(x2.values()) + (tp.weight, gates)
            first = torch.autograd.grad(loss, grad_targets, create_graph=True, retain_graph=True)
            second = sum(g.square().mean() for g in first if g is not None)
            second.backward()
        elif do_backward:
            loss.backward()
        _zero_tp_grads(tp, x1, x2, gates)
    _sync(device)
    return (time.perf_counter() - t0) / repeat * 1000.0


def _make_graph(num_nodes: int, avg_degree: int, device: torch.device, dtype: torch.dtype):
    pos = torch.randn(num_nodes, 3, device=device, dtype=dtype)
    A = torch.randint(1, 6, (num_nodes,), device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    num_edges = num_nodes * avg_degree
    edge_dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_src = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_shifts = torch.zeros(num_edges, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    return pos, A, batch, edge_src, edge_dst, edge_shifts, cell


def _bench_model(
    model: torch.nn.Module,
    graph,
    *,
    warmup: int,
    repeat: int,
    do_backward: bool,
    device: torch.device,
) -> float:
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph

    for _ in range(warmup):
        p = pos.detach().clone().requires_grad_(do_backward)
        model.zero_grad(set_to_none=True)
        out = model(p, A, batch, edge_src, edge_dst, edge_shifts, cell)
        if do_backward:
            out.sum().backward()
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(repeat):
        p = pos.detach().clone().requires_grad_(do_backward)
        model.zero_grad(set_to_none=True)
        out = model(p, A, batch, edge_src, edge_dst, edge_shifts, cell)
        if do_backward:
            out.sum().backward()
    _sync(device)
    return (time.perf_counter() - t0) / repeat * 1000.0


def _bench_model_force_step(
    model: torch.nn.Module,
    graph,
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> float:
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = graph

    for _ in range(warmup):
        p = pos.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        energy = model(p, A, batch, edge_src, edge_dst, edge_shifts, cell).sum()
        forces = -torch.autograd.grad(energy, p, create_graph=True, retain_graph=True)[0]
        loss = forces.square().mean()
        loss.backward()
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(repeat):
        p = pos.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        energy = model(p, A, batch, edge_src, edge_dst, edge_shifts, cell).sum()
        forces = -torch.autograd.grad(energy, p, create_graph=True, retain_graph=True)[0]
        loss = forces.square().mean()
        loss.backward()
    _sync(device)
    return (time.perf_counter() - t0) / repeat * 1000.0


def _print_ext_status() -> None:
    print("ICTD TP extension status")
    print(f"  cuda_ext_available = {ictd_irreps_cuda.cuda_ext_available()}")
    print(f"  cuda_ext_has_cuda  = {ictd_irreps_cuda.cuda_ext_has_cuda()}")
    print(f"  auto_experimental = {ictd_irreps_cuda.auto_experimental_cuda_enabled()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ICTD TP backends: pytorch vs auto vs cuda_ext")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--batch", type=int, default=20000, help="Batch/edge count for TP microbenchmark")
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--mul-in1", type=int, default=32)
    parser.add_argument("--mul-in2", type=int, default=1)
    parser.add_argument("--mul-out", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--skip-fwdbwd", action="store_true", help="Skip forward+backward TP benchmark")
    parser.add_argument("--include-double-backward", action="store_true", help="Also benchmark TP double-backward and model force-step")
    parser.add_argument("--backends", nargs="+", default=["pytorch", "auto", "cuda_ext"])
    parser.add_argument("--full-model", action="store_true", help="Also benchmark a small ICTD full model")
    parser.add_argument("--nodes", type=int, default=128)
    parser.add_argument("--avg-degree", type=int, default=24)
    parser.add_argument("--num-interaction", type=int, default=2)
    parser.add_argument("--hidden-dim-conv", type=int, default=32)
    parser.add_argument("--output-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    print(f"Device: {device}")
    print(f"Dtype:  {dtype}")
    _print_ext_status()
    print()

    if device.type != "cuda":
        print("WARNING: not running on CUDA; 'auto' should fall back and 'cuda_ext' should fail.", flush=True)
        print()
    elif "auto" in args.backends and not ictd_irreps_cuda.auto_experimental_cuda_enabled():
        print(
            "NOTE: 'auto' experimental CUDA TP is disabled. "
            "Set ICTD_ENABLE_EXPERIMENTAL_CUDA_TP=1 to let 'auto' try the custom backend.",
            flush=True,
        )
    print()

    print("Projection microbenchmark")
    if args.include_double_backward:
        print(f"{'backend':<10} {'forward_ms':>12} {'fwdbwd_ms':>12} {'dblbwd_ms':>12} {'status':>18}")
        print("-" * 72)
    else:
        print(f"{'backend':<10} {'forward_ms':>12} {'fwdbwd_ms':>12} {'status':>18}")
        print("-" * 58)
    for backend in args.backends:
        fwd_ms = bwd_ms = dbl_ms = None
        status = "ok"
        try:
            fwd_ms = _bench_projection_backend(
                backend=backend,
                batch=args.batch,
                mul_in1=args.mul_in1,
                mul_in2=args.mul_in2,
                num_paths=max(1, args.mul_out // 4),
                kdim=3,
                warmup=args.warmup,
                repeat=args.repeat,
                do_backward=False,
                do_double_backward=False,
                device=device,
                dtype=dtype,
            )
            if not args.skip_fwdbwd:
                bwd_ms = _bench_projection_backend(
                    backend=backend,
                    batch=args.batch,
                    mul_in1=args.mul_in1,
                    mul_in2=args.mul_in2,
                    num_paths=max(1, args.mul_out // 4),
                    kdim=3,
                    warmup=max(3, args.warmup // 2),
                    repeat=max(5, args.repeat // 2),
                    do_backward=True,
                    do_double_backward=False,
                    device=device,
                    dtype=dtype,
                )
            if args.include_double_backward:
                dbl_ms = _bench_projection_backend(
                    backend=backend,
                    batch=args.batch,
                    mul_in1=args.mul_in1,
                    mul_in2=args.mul_in2,
                    num_paths=max(1, args.mul_out // 4),
                    kdim=3,
                    warmup=max(2, args.warmup // 3),
                    repeat=max(3, args.repeat // 6),
                    do_backward=False,
                    do_double_backward=True,
                    device=device,
                    dtype=dtype,
                )
        except Exception as exc:  # pragma: no cover
            status = f"error: {type(exc).__name__}"
        fwd_text = f"{fwd_ms:12.3f}" if fwd_ms is not None else f"{'-':>12}"
        bwd_text = f"{bwd_ms:12.3f}" if bwd_ms is not None else f"{'-':>12}"
        if args.include_double_backward:
            dbl_text = f"{dbl_ms:12.3f}" if dbl_ms is not None else f"{'-':>12}"
            print(f"{backend:<10} {fwd_text} {bwd_text} {dbl_text} {status:>18}")
        else:
            print(f"{backend:<10} {fwd_text} {bwd_text} {status:>18}")

    print()
    print("Mix microbenchmark")
    if args.include_double_backward:
        print(f"{'backend':<10} {'forward_ms':>12} {'fwdbwd_ms':>12} {'dblbwd_ms':>12} {'status':>18}")
        print("-" * 72)
    else:
        print(f"{'backend':<10} {'forward_ms':>12} {'fwdbwd_ms':>12} {'status':>18}")
        print("-" * 58)
    for backend in args.backends:
        fwd_ms = bwd_ms = dbl_ms = None
        status = "ok"
        try:
            num_paths = max(1, args.mul_out // 4)
            fwd_ms = _bench_mix_backend(
                backend=backend,
                batch=args.batch,
                num_paths=num_paths,
                kdim=3,
                ij=args.mul_in1 * args.mul_in2,
                mul_out=args.mul_out,
                warmup=args.warmup,
                repeat=args.repeat,
                do_backward=False,
                do_double_backward=False,
                device=device,
                dtype=dtype,
            )
            if not args.skip_fwdbwd:
                bwd_ms = _bench_mix_backend(
                    backend=backend,
                    batch=args.batch,
                    num_paths=num_paths,
                    kdim=3,
                    ij=args.mul_in1 * args.mul_in2,
                    mul_out=args.mul_out,
                    warmup=max(3, args.warmup // 2),
                    repeat=max(5, args.repeat // 2),
                    do_backward=True,
                    do_double_backward=False,
                    device=device,
                    dtype=dtype,
                )
            if args.include_double_backward:
                dbl_ms = _bench_mix_backend(
                    backend=backend,
                    batch=args.batch,
                    num_paths=num_paths,
                    kdim=3,
                    ij=args.mul_in1 * args.mul_in2,
                    mul_out=args.mul_out,
                    warmup=max(2, args.warmup // 3),
                    repeat=max(3, args.repeat // 6),
                    do_backward=False,
                    do_double_backward=True,
                    device=device,
                    dtype=dtype,
                )
        except Exception as exc:  # pragma: no cover
            status = f"error: {type(exc).__name__}"
        fwd_text = f"{fwd_ms:12.3f}" if fwd_ms is not None else f"{'-':>12}"
        bwd_text = f"{bwd_ms:12.3f}" if bwd_ms is not None else f"{'-':>12}"
        if args.include_double_backward:
            dbl_text = f"{dbl_ms:12.3f}" if dbl_ms is not None else f"{'-':>12}"
            print(f"{backend:<10} {fwd_text} {bwd_text} {dbl_text} {status:>18}")
        else:
            print(f"{backend:<10} {fwd_text} {bwd_text} {status:>18}")

    print()
    print("TP microbenchmark")
    print(
        f"  batch={args.batch} lmax={args.lmax} mul_in1={args.mul_in1} "
        f"mul_in2={args.mul_in2} mul_out={args.mul_out}"
    )
    print(f"  warmup={args.warmup} repeat={args.repeat}")
    print()
    if args.include_double_backward:
        print(f"{'backend':<10} {'forward_ms':>12} {'fwdbwd_ms':>12} {'dblbwd_ms':>12} {'status':>18}")
        print("-" * 72)
    else:
        print(f"{'backend':<10} {'forward_ms':>12} {'fwdbwd_ms':>12} {'status':>18}")
        print("-" * 58)

    for backend in args.backends:
        fwd_ms = None
        bwd_ms = None
        dbl_ms = None
        status = "ok"
        try:
            fwd_ms = _bench_tp_backend(
                backend=backend,
                batch=args.batch,
                lmax=args.lmax,
                mul_in1=args.mul_in1,
                mul_in2=args.mul_in2,
                mul_out=args.mul_out,
                warmup=args.warmup,
                repeat=args.repeat,
                do_backward=False,
                do_double_backward=False,
                device=device,
                dtype=dtype,
            )
            if not args.skip_fwdbwd:
                bwd_ms = _bench_tp_backend(
                    backend=backend,
                    batch=args.batch,
                    lmax=args.lmax,
                    mul_in1=args.mul_in1,
                    mul_in2=args.mul_in2,
                    mul_out=args.mul_out,
                    warmup=max(3, args.warmup // 2),
                    repeat=max(5, args.repeat // 2),
                    do_backward=True,
                    do_double_backward=False,
                    device=device,
                    dtype=dtype,
                )
            if args.include_double_backward:
                dbl_ms = _bench_tp_backend(
                    backend=backend,
                    batch=args.batch,
                    lmax=args.lmax,
                    mul_in1=args.mul_in1,
                    mul_in2=args.mul_in2,
                    mul_out=args.mul_out,
                    warmup=max(2, args.warmup // 3),
                    repeat=max(3, args.repeat // 6),
                    do_backward=False,
                    do_double_backward=True,
                    device=device,
                    dtype=dtype,
                )
        except Exception as exc:  # pragma: no cover - benchmark failure path
            status = f"error: {type(exc).__name__}"
        fwd_text = f"{fwd_ms:12.3f}" if fwd_ms is not None else f"{'-':>12}"
        bwd_text = f"{bwd_ms:12.3f}" if bwd_ms is not None else f"{'-':>12}"
        if args.include_double_backward:
            dbl_text = f"{dbl_ms:12.3f}" if dbl_ms is not None else f"{'-':>12}"
            print(f"{backend:<10} {fwd_text} {bwd_text} {dbl_text} {status:>18}")
        else:
            print(f"{backend:<10} {fwd_text} {bwd_text} {status:>18}")

    if not args.full_model:
        return

    print()
    print("Full model benchmark")
    print(
        f"  nodes={args.nodes} avg_degree={args.avg_degree} "
        f"num_interaction={args.num_interaction} hidden_dim_conv={args.hidden_dim_conv}"
    )
    print()
    if args.include_double_backward:
        print(f"{'backend':<10} {'forward_ms':>12} {'fwdbwd_ms':>12} {'force_step_ms':>14} {'params':>12} {'status':>12}")
        print("-" * 82)
    else:
        print(f"{'backend':<10} {'forward_ms':>12} {'fwdbwd_ms':>12} {'params':>12} {'status':>12}")
        print("-" * 66)

    graph = _make_graph(args.nodes, args.avg_degree, device, dtype)
    common = dict(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=args.hidden_dim_conv,
        hidden_dim_sh=16,
        hidden_dim=32,
        channel_in2=32,
        embedding_dim=16,
        max_atomvalue=10,
        output_size=args.output_size,
        embed_size=[64, 64],
        main_hidden_sizes3=[64, 64],
        num_layers=1,
        num_interaction=args.num_interaction,
        lmax=args.lmax,
        internal_compute_dtype=dtype,
    )

    for backend in args.backends:
        fwd_ms = None
        bwd_ms = None
        force_step_ms = None
        n_params = None
        status = "ok"
        try:
            model = PureCartesianICTDTransformerLayer(
                **common,
                ictd_tp_backend=backend,
            ).to(device=device, dtype=dtype)
            n_params = sum(p.numel() for p in model.parameters())
            model.eval()
            fwd_ms = _bench_model(
                model,
                graph,
                warmup=max(3, args.warmup // 2),
                repeat=max(5, args.repeat // 4),
                do_backward=False,
                device=device,
            )
            bwd_ms = _bench_model(
                model,
                graph,
                warmup=max(2, args.warmup // 3),
                repeat=max(3, args.repeat // 6),
                do_backward=True,
                device=device,
            )
            if args.include_double_backward:
                force_step_ms = _bench_model_force_step(
                    model,
                    graph,
                    warmup=max(1, args.warmup // 4),
                    repeat=max(2, args.repeat // 8),
                    device=device,
                )
        except Exception as exc:  # pragma: no cover - benchmark failure path
            status = f"{type(exc).__name__}"
        fwd_text = f"{fwd_ms:12.3f}" if fwd_ms is not None else f"{'-':>12}"
        bwd_text = f"{bwd_ms:12.3f}" if bwd_ms is not None else f"{'-':>12}"
        param_text = f"{n_params:12d}" if n_params is not None else f"{'-':>12}"
        if args.include_double_backward:
            force_text = f"{force_step_ms:14.3f}" if force_step_ms is not None else f"{'-':>14}"
            print(f"{backend:<10} {fwd_text} {bwd_text} {force_text} {param_text} {status:>12}")
        else:
            print(f"{backend:<10} {fwd_text} {bwd_text} {param_text} {status:>12}")


if __name__ == "__main__":
    main()
