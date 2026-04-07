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
if not hasattr(torch, "compiler"):
    class _CompilerShim:
        @staticmethod
        def disable(fn):
            return fn
    torch.compiler = _CompilerShim()  # type: ignore[attr-defined]

from e3nn import o3
import cuequivariance as cue
import cuequivariance_torch as cuet

from molecular_force_field.models.e3nn_layers_channelwise import tp_out_irreps_with_instructions
from molecular_force_field.models.ictd_irreps import (
    HarmonicChannelWiseTensorProduct,
    HarmonicChannelWiseTensorProductO3,
)


def parity_for_l(l: int) -> int:
    return 1 if l % 2 == 0 else -1


def make_e3nn_irreps1(mul: int, lmax: int) -> o3.Irreps:
    return o3.Irreps(" + ".join(f"{mul}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1)))


def make_e3nn_irreps2(lmax: int) -> o3.Irreps:
    return o3.Irreps(" + ".join(f"1x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(lmax + 1)))


def make_cue_irreps1(mul: int, lmax: int):
    return cue.Irreps("O3", [(mul, (l, parity_for_l(l))) for l in range(lmax + 1)])


def make_cue_irreps2(lmax: int):
    return cue.Irreps("O3", [(1, (l, parity_for_l(l))) for l in range(lmax + 1)])


def make_ictd_inputs(batch: int, mul: int, lmax: int, device: torch.device, dtype: torch.dtype):
    x1 = {
        l: torch.randn(batch, mul, 2 * l + 1, device=device, dtype=dtype, requires_grad=True)
        for l in range(lmax + 1)
    }
    x2 = {
        l: torch.randn(batch, 1, 2 * l + 1, device=device, dtype=dtype, requires_grad=True)
        for l in range(lmax + 1)
    }
    return x1, x2


def make_ictd_o3_inputs(batch: int, mul: int, lmax: int, device: torch.device, dtype: torch.dtype):
    x1 = {
        (l, parity_for_l(l)): torch.randn(batch, mul, 2 * l + 1, device=device, dtype=dtype, requires_grad=True)
        for l in range(lmax + 1)
    }
    x2 = {
        (l, parity_for_l(l)): torch.randn(batch, 1, 2 * l + 1, device=device, dtype=dtype, requires_grad=True)
        for l in range(lmax + 1)
    }
    return x1, x2


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


def bench_e3nn(batch: int, mul: int, lmax: int, device: torch.device, dtype: torch.dtype, warmup: int, repeat: int):
    irreps1 = make_e3nn_irreps1(mul, lmax)
    irreps2 = make_e3nn_irreps2(lmax)
    irreps_mid, instructions = tp_out_irreps_with_instructions(irreps1, irreps2, irreps1)
    mod = o3.TensorProduct(
        irreps1,
        irreps2,
        irreps_mid,
        instructions=instructions,
        shared_weights=False,
        internal_weights=False,
    ).to(device=device, dtype=dtype)
    x1 = irreps1.randn(batch, -1).to(device=device, dtype=dtype).requires_grad_(True)
    x2 = irreps2.randn(batch, -1).to(device=device, dtype=dtype).requires_grad_(True)
    w = torch.randn(batch, mod.weight_numel, device=device, dtype=dtype, requires_grad=True)

    def run():
        zero_dense_grads(mod, x1, x2, w)
        loss = loss_from_output(mod(x1, x2, w))
        loss.backward()

    return measure(run, warmup=warmup, repeat=repeat, device=device), mod.weight_numel


def bench_cueq(batch: int, mul: int, lmax: int, device: torch.device, dtype: torch.dtype, warmup: int, repeat: int):
    irreps1 = make_cue_irreps1(mul, lmax)
    irreps2 = make_cue_irreps2(lmax)
    filter_out = [ir for (_mul, ir) in irreps1]
    mod = cuet.ChannelWiseTensorProduct(
        irreps1,
        irreps2,
        filter_irreps_out=filter_out,
        shared_weights=False,
        internal_weights=False,
        layout=cue.ir_mul,
        device=device,
        dtype=dtype,
        method="uniform_1d" if device.type == "cuda" else "naive",
    ).to(device=device, dtype=dtype)
    x1 = torch.randn(batch, irreps1.dim, device=device, dtype=dtype, requires_grad=True)
    x2 = torch.randn(batch, irreps2.dim, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(batch, mod.weight_numel, device=device, dtype=dtype, requires_grad=True)

    def run():
        zero_dense_grads(mod, x1, x2, w)
        loss = loss_from_output(mod(x1, x2, w))
        loss.backward()

    return measure(run, warmup=warmup, repeat=repeat, device=device), mod.weight_numel


def bench_ictd(batch: int, mul: int, lmax: int, device: torch.device, dtype: torch.dtype, warmup: int, repeat: int):
    mod = HarmonicChannelWiseTensorProduct(
        mul_in1=mul,
        mul_in2=1,
        mul_out=mul,
        lmax=lmax,
        internal_weights=True,
        path_policy="full",
        internal_compute_dtype=dtype,
    ).to(device=device, dtype=dtype)
    x1, x2 = make_ictd_inputs(batch, mul, lmax, device, dtype)
    gates = torch.randn(batch, mod.num_paths, device=device, dtype=dtype, requires_grad=True)

    def run():
        zero_dict_grads(mod, x1, x2, gates)
        loss = loss_from_output(mod(x1, x2, gates))
        loss.backward()

    return measure(run, warmup=warmup, repeat=repeat, device=device), mod.num_paths


def bench_ictd_o3(batch: int, mul: int, lmax: int, device: torch.device, dtype: torch.dtype, warmup: int, repeat: int):
    active_irreps = [(l, parity_for_l(l)) for l in range(lmax + 1)]
    mod = HarmonicChannelWiseTensorProductO3(
        mul_in1=mul,
        mul_in2=1,
        mul_out=mul,
        lmax=lmax,
        active_irreps=active_irreps,
        internal_weights=True,
        path_policy="full",
        internal_compute_dtype=dtype,
    ).to(device=device, dtype=dtype)
    x1, x2 = make_ictd_o3_inputs(batch, mul, lmax, device, dtype)
    gates = torch.randn(batch, mod.num_paths, device=device, dtype=dtype, requires_grad=True)

    def run():
        zero_dict_grads(mod, x1, x2, gates)
        loss = loss_from_output(mod(x1, x2, gates))
        loss.backward()

    return measure(run, warmup=warmup, repeat=repeat, device=device), mod.num_paths


def plot_results(rows: list[dict], out_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    series = [
        ("e3nn", "e3nn channelwise"),
        ("cueq", "cueq channelwise"),
        ("ictd", "ictd channelwise"),
        ("ictd_o3", "ictd-o3 channelwise"),
    ]
    ax_speed, ax_mem = axes
    for key, label in series:
        xs = [r["lmax"] for r in rows]
        ys_ms = [r[f"{key}_ms"] for r in rows]
        ys_mb = [r[f"{key}_mb"] for r in rows]
        ax_speed.plot(xs, ys_ms, marker="o", label=label)
        ax_mem.plot(xs, ys_mb, marker="o", label=label)
    ax_speed.set_title("ChannelWiseTP Speed")
    ax_mem.set_title("ChannelWiseTP Memory")
    ax_speed.set_xlabel("lmax")
    ax_mem.set_xlabel("lmax")
    ax_speed.set_ylabel("ms")
    ax_mem.set_ylabel("Peak MB")
    ax_speed.set_yscale("log")
    ax_speed.grid(True, alpha=0.3)
    ax_mem.grid(True, alpha=0.3)
    ax_speed.legend()
    ax_mem.legend()
    fig.suptitle("Conv-style ChannelWiseTP Backends in FP32 on GPU")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--mul", type=int, default=64)
    p.add_argument("--lmax-max", type=int, default=6)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out-json", default="/home/rebuild/.bench_channelwise_tp_backends.jsonl")
    p.add_argument("--out-png", default="/home/rebuild/channelwise_tp_backends.png")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if device.type == "cuda" and dtype == torch.float32:
        torch.set_float32_matmul_precision("high")

    rows: list[dict] = []
    for lmax in range(args.lmax_max + 1):
        (e3nn_ms, e3nn_mb), e3nn_weight_numel = bench_e3nn(args.batch, args.mul, lmax, device, dtype, args.warmup, args.repeat)
        (cueq_ms, cueq_mb), cueq_weight_numel = bench_cueq(args.batch, args.mul, lmax, device, dtype, args.warmup, args.repeat)
        (ictd_ms, ictd_mb), ictd_num_paths = bench_ictd(args.batch, args.mul, lmax, device, dtype, args.warmup, args.repeat)
        (ictd_o3_ms, ictd_o3_mb), ictd_o3_num_paths = bench_ictd_o3(args.batch, args.mul, lmax, device, dtype, args.warmup, args.repeat)
        row = {
            "lmax": lmax,
            "e3nn_ms": e3nn_ms,
            "e3nn_mb": e3nn_mb,
            "e3nn_weight_numel": e3nn_weight_numel,
            "cueq_ms": cueq_ms,
            "cueq_mb": cueq_mb,
            "cueq_weight_numel": cueq_weight_numel,
            "ictd_ms": ictd_ms,
            "ictd_mb": ictd_mb,
            "ictd_num_paths": ictd_num_paths,
            "ictd_o3_ms": ictd_o3_ms,
            "ictd_o3_mb": ictd_o3_mb,
            "ictd_o3_num_paths": ictd_o3_num_paths,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    out_json = Path(args.out_json)
    out_json.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    plot_results(rows, Path(args.out_png))


if __name__ == "__main__":
    main()
