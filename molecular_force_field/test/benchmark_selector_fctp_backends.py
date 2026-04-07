#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
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


def make_e3nn_hidden(mul: int, lmax: int) -> o3.Irreps:
    return o3.Irreps(" + ".join(f"{mul}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1)))


def make_cue_hidden(mul: int, lmax: int):
    return cue.Irreps("O3", [(mul, (l, parity_for_l(l))) for l in range(lmax + 1)])


def make_ictd_hidden(batch: int, mul: int, lmax: int, device: torch.device, dtype: torch.dtype):
    return {
        l: torch.randn(batch, mul, 2 * l + 1, device=device, dtype=dtype, requires_grad=True)
        for l in range(lmax + 1)
    }


def make_ictd_hidden_o3(batch: int, mul: int, lmax: int, device: torch.device, dtype: torch.dtype):
    return {
        (l, parity_for_l(l)): torch.randn(batch, mul, 2 * l + 1, device=device, dtype=dtype, requires_grad=True)
        for l in range(lmax + 1)
    }


def make_selector_dense(batch: int, num_elements: int, device: torch.device, dtype: torch.dtype):
    idx = torch.randint(0, num_elements, (batch,), device=device)
    x = torch.zeros(batch, num_elements, device=device, dtype=dtype)
    x.scatter_(1, idx[:, None], 1.0)
    x.requires_grad_(True)
    return x


def make_selector_ictd(batch: int, num_elements: int, device: torch.device, dtype: torch.dtype):
    idx = torch.randint(0, num_elements, (batch,), device=device)
    x = torch.zeros(batch, num_elements, 1, device=device, dtype=dtype)
    x.scatter_(1, idx[:, None, None], 1.0)
    x.requires_grad_(True)
    return {0: x}


def make_selector_ictd_o3(batch: int, num_elements: int, device: torch.device, dtype: torch.dtype):
    idx = torch.randint(0, num_elements, (batch,), device=device)
    x = torch.zeros(batch, num_elements, 1, device=device, dtype=dtype)
    x.scatter_(1, idx[:, None, None], 1.0)
    x.requires_grad_(True)
    return {(0, 1): x}


def zero_dense_grads(module, *tensors):
    module.zero_grad(set_to_none=True)
    for t in tensors:
        if t.grad is not None:
            t.grad = None


def zero_dict_grads(module, *items):
    module.zero_grad(set_to_none=True)
    for item in items:
        if isinstance(item, dict):
            for t in item.values():
                if t.grad is not None:
                    t.grad = None
        elif hasattr(item, "grad") and item.grad is not None:
            item.grad = None


def loss_from_output(out) -> torch.Tensor:
    if isinstance(out, dict):
        return sum(v.square().mean() for v in out.values())
    return out.square().mean()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def measure(fn, *, warmup: int, repeat: int, device: torch.device) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    sync(device)
    reset_peak(device)
    times = []
    for _ in range(repeat):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), peak_mb(device)


def bench_e3nn(batch: int, mul: int, lmax: int, num_elements: int, device: torch.device, dtype: torch.dtype, warmup: int, repeat: int):
    irreps1 = make_e3nn_hidden(mul, lmax)
    irreps2 = o3.Irreps(f"{num_elements}x0e")
    mod = o3.FullyConnectedTensorProduct(irreps1, irreps2, irreps1).to(device=device, dtype=dtype)
    x1 = irreps1.randn(batch, -1).to(device=device, dtype=dtype).requires_grad_(True)
    x2 = make_selector_dense(batch, num_elements, device, dtype)

    def run():
        zero_dense_grads(mod, x1, x2)
        loss = loss_from_output(mod(x1, x2))
        loss.backward()

    return measure(run, warmup=warmup, repeat=repeat, device=device), sum(p.numel() for p in mod.parameters())


def bench_cueq(batch: int, mul: int, lmax: int, num_elements: int, device: torch.device, dtype: torch.dtype, warmup: int, repeat: int):
    irreps1 = make_cue_hidden(mul, lmax)
    irreps2 = cue.Irreps("O3", [(num_elements, (0, 1))])
    mod = cuet.FullyConnectedTensorProduct(
        irreps1,
        irreps2,
        irreps1,
        layout=cue.ir_mul,
    ).to(device=device, dtype=dtype)
    x1 = torch.randn(batch, irreps1.dim, device=device, dtype=dtype, requires_grad=True)
    x2 = make_selector_dense(batch, num_elements, device, dtype)

    def run():
        zero_dense_grads(mod, x1, x2)
        loss = loss_from_output(mod(x1, x2))
        loss.backward()

    return measure(run, warmup=warmup, repeat=repeat, device=device), sum(p.numel() for p in mod.parameters())


def bench_ictd(batch: int, mul: int, lmax: int, num_elements: int, device: torch.device, dtype: torch.dtype, warmup: int, repeat: int):
    mod = HarmonicFullyConnectedTensorProduct(
        mul_in1=mul,
        mul_in2=num_elements,
        mul_out=mul,
        lmax=lmax,
        internal_weights=True,
        path_policy="full",
        internal_compute_dtype=dtype,
    ).to(device=device, dtype=dtype)
    x1 = make_ictd_hidden(batch, mul, lmax, device, dtype)
    x2 = make_selector_ictd(batch, num_elements, device, dtype)

    def run():
        zero_dict_grads(mod, x1, x2)
        loss = loss_from_output(mod(x1, x2))
        loss.backward()

    return measure(run, warmup=warmup, repeat=repeat, device=device), sum(p.numel() for p in mod.parameters())


def bench_ictd_o3(batch: int, mul: int, lmax: int, num_elements: int, device: torch.device, dtype: torch.dtype, warmup: int, repeat: int):
    active_irreps = [(l, parity_for_l(l)) for l in range(lmax + 1)]
    mod = HarmonicFullyConnectedTensorProductO3(
        mul_in1=mul,
        mul_in2=num_elements,
        mul_out=mul,
        lmax=lmax,
        active_irreps=active_irreps,
        internal_weights=True,
        path_policy="full",
        internal_compute_dtype=dtype,
    ).to(device=device, dtype=dtype)
    x1 = make_ictd_hidden_o3(batch, mul, lmax, device, dtype)
    x2 = make_selector_ictd_o3(batch, num_elements, device, dtype)

    def run():
        zero_dict_grads(mod, x1, x2)
        loss = loss_from_output(mod(x1, x2))
        loss.backward()

    return measure(run, warmup=warmup, repeat=repeat, device=device), sum(p.numel() for p in mod.parameters())


def plot_results(rows: list[dict], out_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    series = [
        ("e3nn", "e3nn selector-FCTP"),
        ("cueq", "cueq selector-FCTP"),
        ("ictd", "ictd selector-FCTP"),
        ("ictd_o3", "ictd-o3 selector-FCTP"),
    ]
    ax_speed, ax_mem = axes
    for key, label in series:
        xs = [r["lmax"] for r in rows]
        ys_ms = [r[f"{key}_ms"] for r in rows]
        ys_mb = [r[f"{key}_mb"] for r in rows]
        ax_speed.plot(xs, ys_ms, marker="o", label=label)
        ax_mem.plot(xs, ys_mb, marker="o", label=label)
    ax_speed.set_title("MACE-like Selector FCTP Speed")
    ax_mem.set_title("MACE-like Selector FCTP Memory")
    ax_speed.set_xlabel("lmax")
    ax_mem.set_xlabel("lmax")
    ax_speed.set_ylabel("ms")
    ax_mem.set_ylabel("Peak MB")
    ax_speed.set_yscale("log")
    ax_speed.grid(True, alpha=0.3)
    ax_mem.grid(True, alpha=0.3)
    ax_speed.legend()
    ax_mem.legend()
    fig.suptitle("MACE-like Selector/Skip FCTP in FP32 on GPU")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--mul", type=int, default=64)
    p.add_argument("--lmax-max", type=int, default=6)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--num-elements", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--out-json", default="/home/rebuild/.bench_selector_fctp_backends.jsonl")
    p.add_argument("--out-png", default="/home/rebuild/selector_fctp_backends.png")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if device.type == "cuda" and dtype == torch.float32:
        torch.set_float32_matmul_precision("high")

    rows: list[dict] = []
    for lmax in range(args.lmax_max + 1):
        (e3nn_ms, e3nn_mb), e3nn_params = bench_e3nn(args.batch, args.mul, lmax, args.num_elements, device, dtype, args.warmup, args.repeat)
        (cueq_ms, cueq_mb), cueq_params = bench_cueq(args.batch, args.mul, lmax, args.num_elements, device, dtype, args.warmup, args.repeat)
        (ictd_ms, ictd_mb), ictd_params = bench_ictd(args.batch, args.mul, lmax, args.num_elements, device, dtype, args.warmup, args.repeat)
        (ictd_o3_ms, ictd_o3_mb), ictd_o3_params = bench_ictd_o3(args.batch, args.mul, lmax, args.num_elements, device, dtype, args.warmup, args.repeat)
        row = {
            "lmax": lmax,
            "e3nn_ms": e3nn_ms,
            "e3nn_mb": e3nn_mb,
            "e3nn_params": e3nn_params,
            "cueq_ms": cueq_ms,
            "cueq_mb": cueq_mb,
            "cueq_params": cueq_params,
            "ictd_ms": ictd_ms,
            "ictd_mb": ictd_mb,
            "ictd_params": ictd_params,
            "ictd_o3_ms": ictd_o3_ms,
            "ictd_o3_mb": ictd_o3_mb,
            "ictd_o3_params": ictd_o3_params,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    out_json = Path(args.out_json)
    out_json.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    plot_results(rows, Path(args.out_png))


if __name__ == "__main__":
    main()
