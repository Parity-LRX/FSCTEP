#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time

import torch
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])
from e3nn import o3

import cuequivariance as cue
import cuequivariance_torch as cuet

from molecular_force_field.models.ictd_irreps import (
    HarmonicFullyConnectedTensorProduct,
    HarmonicFullyConnectedTensorProductO3,
)


def parity_for_l(l: int) -> int:
    return 1 if l % 2 == 0 else -1


def make_e3nn_irreps(mul: int, lmax: int) -> o3.Irreps:
    return o3.Irreps(" + ".join(f"{mul}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1)))


def make_cue_irreps(mul: int, lmax: int):
    return cue.Irreps("O3", [(mul, (l, parity_for_l(l))) for l in range(lmax + 1)])


def make_ictd_inputs(batch: int, mul: int, lmax: int, device: torch.device, dtype: torch.dtype):
    return {
        l: torch.randn(batch, mul, 2 * l + 1, device=device, dtype=dtype, requires_grad=True)
        for l in range(lmax + 1)
    }


def make_ictd_o3_inputs(batch: int, mul: int, lmax: int, device: torch.device, dtype: torch.dtype):
    return {
        (l, parity_for_l(l)): torch.randn(batch, mul, 2 * l + 1, device=device, dtype=dtype, requires_grad=True)
        for l in range(lmax + 1)
    }


def reset_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def peak_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def measure_dense_module(module, x1, x2, *, repeat: int, warmup: int, device: torch.device):
    for _ in range(warmup):
        out = module(x1, x2)
        out.sum().backward()
        module.zero_grad(set_to_none=True)
        if x1.grad is not None:
            x1.grad = None
        if x2.grad is not None:
            x2.grad = None
    reset_cuda(device)
    times = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = module(x1, x2)
        out.sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)
        module.zero_grad(set_to_none=True)
        if x1.grad is not None:
            x1.grad = None
        if x2.grad is not None:
            x2.grad = None
    return statistics.median(times), peak_mb(device)


def measure_dict_module(module, x1, x2, *, repeat: int, warmup: int, device: torch.device):
    def loss_fn(out):
        if isinstance(out, dict):
            return sum(v.sum() for v in out.values())
        raise TypeError(type(out))

    for _ in range(warmup):
        out = module(x1, x2)
        loss_fn(out).backward()
        module.zero_grad(set_to_none=True)
        for v in x1.values():
            if v.grad is not None:
                v.grad = None
        for v in x2.values():
            if v.grad is not None:
                v.grad = None
    reset_cuda(device)
    times = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = module(x1, x2)
        loss_fn(out).backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)
        module.zero_grad(set_to_none=True)
        for v in x1.values():
            if v.grad is not None:
                v.grad = None
        for v in x2.values():
            if v.grad is not None:
                v.grad = None
    return statistics.median(times), peak_mb(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--mul", type=int, default=64)
    p.add_argument("--lmax-max", type=int, default=6)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if device.type == "cuda" and dtype == torch.float32:
        torch.set_float32_matmul_precision("high")

    for lmax in range(args.lmax_max + 1):
        irreps_e3nn = make_e3nn_irreps(args.mul, lmax)
        irreps_cue = make_cue_irreps(args.mul, lmax)

        x1_e3 = irreps_e3nn.randn(args.batch, -1).to(device=device, dtype=dtype).requires_grad_(True)
        x2_e3 = irreps_e3nn.randn(args.batch, -1).to(device=device, dtype=dtype).requires_grad_(True)
        mod_e3 = o3.FullyConnectedTensorProduct(irreps_e3nn, irreps_e3nn, irreps_e3nn).to(device=device, dtype=dtype)
        e3_ms, e3_mem = measure_dense_module(mod_e3, x1_e3, x2_e3, repeat=args.repeat, warmup=args.warmup, device=device)

        x1_cue = torch.randn(args.batch, irreps_cue.dim, device=device, dtype=dtype, requires_grad=True)
        x2_cue = torch.randn(args.batch, irreps_cue.dim, device=device, dtype=dtype, requires_grad=True)
        mod_cue = cuet.FullyConnectedTensorProduct(
            irreps_cue,
            irreps_cue,
            irreps_cue,
            layout=cue.ir_mul,
        ).to(device=device, dtype=dtype)
        cue_ms, cue_mem = measure_dense_module(mod_cue, x1_cue, x2_cue, repeat=args.repeat, warmup=args.warmup, device=device)

        x1_ictd = make_ictd_inputs(args.batch, args.mul, lmax, device, dtype)
        x2_ictd = make_ictd_inputs(args.batch, args.mul, lmax, device, dtype)
        mod_ictd = HarmonicFullyConnectedTensorProduct(
            mul_in1=args.mul,
            mul_in2=args.mul,
            mul_out=args.mul,
            lmax=lmax,
            internal_weights=True,
            path_policy="full",
            internal_compute_dtype=dtype,
        ).to(device=device, dtype=dtype)
        ictd_ms, ictd_mem = measure_dict_module(mod_ictd, x1_ictd, x2_ictd, repeat=args.repeat, warmup=args.warmup, device=device)

        active_irreps = [(l, parity_for_l(l)) for l in range(lmax + 1)]
        x1_o3 = make_ictd_o3_inputs(args.batch, args.mul, lmax, device, dtype)
        x2_o3 = make_ictd_o3_inputs(args.batch, args.mul, lmax, device, dtype)
        mod_o3 = HarmonicFullyConnectedTensorProductO3(
            mul_in1=args.mul,
            mul_in2=args.mul,
            mul_out=args.mul,
            lmax=lmax,
            active_irreps=active_irreps,
            internal_weights=True,
            path_policy="full",
            internal_compute_dtype=dtype,
        ).to(device=device, dtype=dtype)
        o3_ms, o3_mem = measure_dict_module(mod_o3, x1_o3, x2_o3, repeat=args.repeat, warmup=args.warmup, device=device)

        print(
            f"lmax={lmax} "
            f"e3nn_ms={e3_ms:.3f} e3nn_mb={e3_mem:.1f} "
            f"cueq_ms={cue_ms:.3f} cueq_mb={cue_mem:.1f} "
            f"ictd_ms={ictd_ms:.3f} ictd_mb={ictd_mem:.1f} "
            f"ictd_o3_ms={o3_ms:.3f} ictd_o3_mb={o3_mem:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
