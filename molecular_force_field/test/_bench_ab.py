#!/usr/bin/env python3
"""A/B benchmark: full model forward+backward for sparse modes.
No dependency on new helper functions so it works on both old and new code."""
import os, sys, time, gc
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import torch
from molecular_force_field.utils.config import ModelConfig
from molecular_force_field.utils.graph_utils import radius_graph_pbc_gpu

def build_graph(N=256, device="cpu", dtype=torch.float32):
    torch.manual_seed(42)
    pos = torch.randn(N, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype) * 12.0
    species = torch.zeros(N, dtype=torch.long, device=device)
    edge_src, edge_dst, shift_vecs = radius_graph_pbc_gpu(pos, 5.0, cell)
    return pos, cell, species, edge_src, edge_dst, shift_vecs

def bench_mode(mode, N=256, device="cpu", warmup=3, repeat=7):
    pos, cell, species, edge_src, edge_dst, shift_vecs = build_graph(N, device)
    cfg = ModelConfig(
        tensor_product_mode=mode,
        num_species=1, channels=32, Lmax=2, num_interaction=3,
        cutoff=5.0, use_long_range=False,
    )
    from molecular_force_field.models import build_model
    model = build_model(cfg).to(device)
    model.train()

    def run():
        pos_r = pos.clone().requires_grad_(True)
        energy, forces, stress, extras = model(
            species, pos_r, cell, edge_src, edge_dst, shift_vecs,
        )
        loss = energy.sum() + forces.sum()
        loss.backward(create_graph=False)
        return loss

    # warmup
    for _ in range(warmup):
        run()
    gc.collect()

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)

    med = sorted(times)[len(times)//2] * 1000
    return med

if __name__ == "__main__":
    for mode in ["pure-cartesian-sparse", "pure-cartesian-sparse-save"]:
        t = bench_mode(mode)
        print(f"  {mode:40s} fwd+bwd median: {t:.1f} ms")
