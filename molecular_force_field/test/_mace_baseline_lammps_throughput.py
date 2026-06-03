#!/usr/bin/env python3
"""Throughput impact of the long-range module on the inference step.

Times the per-step inference compute (forward + 1st-order force) with long-range
OFF vs ON, per backend, at ~1024 atoms / channels=64 / fp32 — the dominant compute
of a LAMMPS step. This measures the TOTAL long-range compute (short-range +
long-range energy + long-range force); in LAMMPS the long-range may be split (cheap
source head in the TorchScript core + heavy reciprocal in the C++ solver), but the
total per-step long-range work is roughly the same either way. A definitive
end-to-end LAMMPS MD timing (with the C++ reciprocal solver) is a separate step.

Run: python -m molecular_force_field.test._mace_baseline_lammps_throughput
"""
from __future__ import annotations

import time

import torch

from molecular_force_field.test.bench_ictd_fix_trainstep import make_fixed_graph, SPECIES
from molecular_force_field.models.pure_cartesian_ictd_mace_baseline import (
    PureCartesianICTDMaceBaseline,
)

_SPEC = torch.tensor(list(SPECIES))


def make_periodic_graph(n, box, cutoff, dtype, device, seed=11):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(n, 3, generator=g, dtype=torch.float64) * box
    cell = torch.eye(3, dtype=torch.float64) * box
    A = _SPEC[torch.randint(0, len(SPECIES), (n,), generator=g)]
    d = pos[:, None, :] - pos[None, :, :]
    shift = -torch.round(d / box)
    dist = (d + shift * box).norm(dim=-1)
    mask = (dist <= cutoff) & (dist > 1e-6)
    src, dst = mask.nonzero(as_tuple=True)
    edge_shifts = shift[src, dst]
    batch = torch.zeros(n, dtype=torch.long)
    f = lambda t, dt=dtype: t.to(device=device, dtype=dt)
    return (f(pos), A.to(device), batch.to(device), src.to(device), dst.to(device),
            f(edge_shifts), f(cell.unsqueeze(0)))


def _build(dtype, device, channels=64, **lr):
    return PureCartesianICTDMaceBaseline(
        max_embed_radius=5.0, main_number_of_basis=8, hidden_dim_conv=channels,
        atomic_numbers=list(SPECIES), max_atomvalue=10, num_interaction=2, lmax=2,
        function_type_main="bessel", polynomial_cutoff_p=6, save_contraction_order=3,
        avg_num_neighbors=40.0, internal_compute_dtype=dtype, device=device, **lr,
    ).to(device=device, dtype=dtype)


def _time(model, graph, dev, warm=8, it=30):
    def step():
        pos = graph[0].detach().clone().requires_grad_(True)
        out = model(pos, *graph[1:])
        e = (out[0] if isinstance(out, tuple) else out).sum()
        torch.autograd.grad(e, pos)  # 1st-order force (inference; no double-backward)

    for _ in range(warm):
        step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    N = 1024
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"[thru] N={N} channels=64 fp32 gpu={gpu}")

    g_open = make_fixed_graph(num_nodes=N, avg_degree=40, dtype=dtype, device=device, seed=7)
    g_per = make_periodic_graph(N, 24.0, 5.0, dtype, device)
    print(f"[thru] open edges={g_open[3].numel()}  periodic edges={g_per[3].numel()}")

    groups = [
        ("OPEN (nonperiodic)", g_open, [
            ("OFF (short-range only)", {}),
            ("latent-coulomb/dense", dict(long_range_mode="latent-coulomb", long_range_backend="dense_pairwise", long_range_boundary="nonperiodic")),
            ("latent-coulomb/tree_fmm", dict(long_range_mode="latent-coulomb", long_range_backend="tree_fmm", long_range_boundary="nonperiodic")),
            ("isolated-far-field-v2", dict(long_range_mode="isolated-far-field-v2", long_range_boundary="nonperiodic")),
        ]),
        ("PERIODIC", g_per, [
            ("OFF (short-range only)", {}),
            ("reciprocal/direct_kspace kmax2", dict(long_range_mode="reciprocal-spectral-v1", long_range_reciprocal_backend="direct_kspace", long_range_boundary="periodic", long_range_kmax=2)),
            ("reciprocal/mesh_fft mesh16", dict(long_range_mode="reciprocal-spectral-v1", long_range_reciprocal_backend="mesh_fft", long_range_boundary="periodic", long_range_mesh_size=16)),
            ("reciprocal/mesh_fft mesh32", dict(long_range_mode="reciprocal-spectral-v1", long_range_reciprocal_backend="mesh_fft", long_range_boundary="periodic", long_range_mesh_size=32)),
        ]),
    ]
    for gname, graph, configs in groups:
        base_ms = None
        print(f"\n[thru] === {gname} graph ===")
        for label, lr in configs:
            try:
                torch.manual_seed(0)
                m = _build(dtype, device, **lr); m.eval()
                ms = _time(m, graph, device)
                if base_ms is None:
                    base_ms = ms
                ov = ms / base_ms if base_ms else float("nan")
                print(f"[thru] {label:32s} {ms:8.3f} ms/step  overhead={ov:5.2f}x (+{(ov-1)*100:4.0f}%)")
            except Exception as ex:
                print(f"[thru] {label:32s} FAIL: {type(ex).__name__}: {str(ex)[:70]}")
    print("[thru] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
