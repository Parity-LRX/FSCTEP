"""
Benchmark: HarmonicElementwiseProduct("0e") vs old _irreps_elementwise_tensor_product_0e.

Tests multiple (lmax, channels, batch) configurations on CPU and GPU (if available).
Reports per-call latency (ms) and relative speedup.
"""
import time
import math
import torch
import torch.nn as nn

from molecular_force_field.models.ictd_irreps import HarmonicElementwiseProduct


# ── reference implementations (copied from pure_cartesian_ictd_layers_full.py) ──

def _split_irreps(x: torch.Tensor, channels: int, lmax: int) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    idx = 0
    for l in range(lmax + 1):
        d = channels * (2 * l + 1)
        blk = x[..., idx: idx + d]
        idx += d
        out[l] = blk.view(*x.shape[:-1], channels, 2 * l + 1)
    return out


def _old_elementwise_tp_0e(x1, x2, channels, lmax):
    b1 = _split_irreps(x1, channels, lmax)
    b2 = _split_irreps(x2, channels, lmax)
    outs = []
    for l in range(lmax + 1):
        outs.append((b1[l] * b2[l]).sum(dim=-1) / ((2 * l + 1) ** 0.5))
    return torch.cat(outs, dim=-1)


# ── benchmark helper ──

def bench(fn, warmup=50, repeats=200, sync_cuda=False):
    for _ in range(warmup):
        fn()
    if sync_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if sync_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / repeats * 1000  # ms


# ── main ──

def run_config(lmax, channels, batch, num_interaction, device):
    dim = channels * (lmax + 1) ** 2
    sync = device.type == "cuda"

    torch.manual_seed(0)
    # Simulate per-feature product (the actual pattern in PureCartesianICTDTransformerLayer)
    features = [torch.randn(batch, dim, device=device) for _ in range(num_interaction)]

    product_5 = HarmonicElementwiseProduct(lmax=lmax, mul=channels, irreps_out="0e").to(device)

    # Pre-split for new path (fair comparison: split is done once in forward)
    features_split = [_split_irreps(f, channels, lmax) for f in features]

    def old_fn():
        invs = [_old_elementwise_tp_0e(f, f, channels, lmax) for f in features]
        return torch.cat(invs, dim=-1)

    def new_fn():
        invs = [product_5(b, b) for b in features_split]
        return torch.cat(invs, dim=-1)

    def new_fn_with_split():
        invs = []
        for f in features:
            b = _split_irreps(f, channels, lmax)
            invs.append(product_5(b, b))
        return torch.cat(invs, dim=-1)

    t_old = bench(old_fn, sync_cuda=sync)
    t_new = bench(new_fn, sync_cuda=sync)
    t_new_split = bench(new_fn_with_split, sync_cuda=sync)
    return t_old, t_new, t_new_split


def main():
    configs = [
        # (lmax, channels, batch, num_interaction)
        (2, 16, 64, 2),
        (2, 16, 256, 2),
        (2, 32, 64, 2),
        (2, 32, 256, 2),
        (2, 16, 64, 3),
        (2, 16, 256, 3),
        (3, 16, 64, 2),
        (3, 16, 256, 2),
        (4, 16, 64, 2),
        (4, 16, 256, 2),
    ]

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    header = f"{'device':<6} {'lmax':>4} {'ch':>4} {'batch':>5} {'n_int':>5} │ {'old(ms)':>9} {'new(ms)':>9} {'new+split':>10} │ {'speedup':>8} {'sp(+split)':>10}"
    sep = "─" * len(header)

    print(sep)
    print(header)
    print(sep)

    for device in devices:
        for lmax, ch, batch, n_int in configs:
            t_old, t_new, t_new_split = run_config(lmax, ch, batch, n_int, device)
            speedup = t_old / t_new if t_new > 0 else float("inf")
            speedup_s = t_old / t_new_split if t_new_split > 0 else float("inf")
            print(
                f"{str(device):<6} {lmax:>4} {ch:>4} {batch:>5} {n_int:>5} │ "
                f"{t_old:>9.3f} {t_new:>9.3f} {t_new_split:>10.3f} │ "
                f"{speedup:>7.2f}x {speedup_s:>9.2f}x"
            )
        print(sep)


if __name__ == "__main__":
    main()
