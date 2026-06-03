#!/usr/bin/env python3
"""Long-range mode SUPPORT MATRIX for PureCartesianICTDMaceBaseline.

For every long-range mode / backend / boundary: build, forward (finite energy),
forces (1st-order grad finite), equivariance (where well-defined), and the FORCE
DOUBLE-BACKWARD (the real test for force-loss training; FFT/FMM kernels are the
risk). Non-periodic modes use the open-box graph; reciprocal modes use a small
periodic box. Each config is isolated in try/except so one failure doesn't abort.

Run: python -m molecular_force_field.test._mace_baseline_lr_matrix
"""
from __future__ import annotations

import torch

from molecular_force_field.test.bench_ictd_fix_trainstep import (
    make_fixed_graph,
    random_rotation,
    SPECIES,
)
from molecular_force_field.models.pure_cartesian_ictd_mace_baseline import (
    PureCartesianICTDMaceBaseline,
)

_SPEC = torch.tensor(list(SPECIES))


def make_periodic_graph(n, box, cutoff, dtype, device, seed=11):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(n, 3, generator=g, dtype=torch.float64) * box
    cell = torch.eye(3, dtype=torch.float64) * box
    A = _SPEC[torch.randint(0, len(SPECIES), (n,), generator=g)]
    src, dst, sh = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pos[j] - pos[i]
            shift = -torch.round(d / box)
            if (d + shift * box).norm() <= cutoff:
                src.append(i); dst.append(j); sh.append(shift)
    edge_src = torch.tensor(src, dtype=torch.long)
    edge_dst = torch.tensor(dst, dtype=torch.long)
    edge_shifts = torch.stack(sh) if sh else torch.zeros(0, 3, dtype=torch.float64)
    batch = torch.zeros(n, dtype=torch.long)
    f = lambda t, dt=dtype: t.to(device=device, dtype=dt)
    return (f(pos), A.to(device), batch.to(device), edge_src.to(device),
            edge_dst.to(device), f(edge_shifts), f(cell.unsqueeze(0)))


def _build(dtype, device, **lr):
    return PureCartesianICTDMaceBaseline(
        max_embed_radius=5.0, main_number_of_basis=8, hidden_dim_conv=16,
        atomic_numbers=list(SPECIES), max_atomvalue=10, num_interaction=2, lmax=2,
        function_type_main="bessel", polynomial_cutoff_p=6, save_contraction_order=3,
        avg_num_neighbors=24.0, internal_compute_dtype=dtype, device=device, **lr,
    ).to(device=device, dtype=dtype)


def _energy(model, graph, pos):
    out = model(pos, *graph[1:])
    return (out[0] if isinstance(out, tuple) else out).sum()


def check(label, lr, graph, periodic, skip_equiv, device, dtype):
    r = {"label": label, "build": "-", "fwd": "-", "force": "-", "equiv": "-", "dbl": "-"}
    try:
        m = _build(dtype, device, **lr); m.eval(); r["build"] = "OK"
    except Exception as ex:
        r["build"] = f"FAIL:{type(ex).__name__}"; return r
    try:
        pos = graph[0].detach().clone().requires_grad_(True)
        e = _energy(m, graph, pos)
        r["fwd"] = "OK" if torch.isfinite(e) else "naninf"
        f = torch.autograd.grad(e, pos, create_graph=False, allow_unused=True)[0]
        r["force"] = "None" if f is None else ("OK" if torch.isfinite(f).all() else "naninf")
    except Exception as ex:
        r["force"] = f"FAIL:{type(ex).__name__}"
    if skip_equiv:
        r["equiv"] = "N/A"
    else:
        try:
            R = random_rotation(dtype=dtype).to(device)
            if periodic:
                graph_r = (graph[0] @ R.T, graph[1], graph[2], graph[3], graph[4],
                           graph[5], graph[6] @ R.transpose(-1, -2))
            else:
                graph_r = (graph[0] @ R.T,) + tuple(graph[1:])
            e0 = _energy(m, graph, graph[0])
            er = _energy(m, graph_r, graph_r[0])
            r["equiv"] = f"{(er - e0).abs().item():.1e}"
        except Exception as ex:
            r["equiv"] = f"FAIL:{type(ex).__name__}"
    try:
        pos = graph[0].detach().clone().requires_grad_(True)
        e = _energy(m, graph, pos)
        fr = torch.autograd.grad(e, pos, create_graph=True, allow_unused=True)[0]
        if fr is None:
            r["dbl"] = "Fnone"
        else:
            (fr ** 2).mean().backward()
            gs = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
            r["dbl"] = "OK" if (gs == gs and gs < 1e30) else "naninf"
    except Exception as ex:
        r["dbl"] = f"FAIL:{type(ex).__name__}"
    return r


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    g_open = make_fixed_graph(num_nodes=24, avg_degree=12, dtype=dtype, device=device, seed=7)
    g_per = make_periodic_graph(20, 8.0, 5.0, dtype, device)
    print(f"[matrix] open-box edges={g_open[3].numel()}  periodic-box edges={g_per[3].numel()}")

    configs = [
        ("latent-coulomb/dense/nonperiodic",
         dict(long_range_mode="latent-coulomb", long_range_backend="dense_pairwise", long_range_boundary="nonperiodic"), g_open, False, False),
        ("latent-coulomb/tree_fmm/nonperiodic",
         dict(long_range_mode="latent-coulomb", long_range_backend="tree_fmm", long_range_boundary="nonperiodic"), g_open, False, False),
        ("isolated-far-field-v1/nonperiodic",
         dict(long_range_mode="isolated-far-field-v1", long_range_boundary="nonperiodic"), g_open, False, False),
        ("isolated-far-field-v2/nonperiodic",
         dict(long_range_mode="isolated-far-field-v2", long_range_boundary="nonperiodic"), g_open, False, False),
        ("reciprocal-spectral/direct_kspace/periodic",
         dict(long_range_mode="reciprocal-spectral-v1", long_range_reciprocal_backend="direct_kspace", long_range_boundary="periodic", long_range_kmax=2), g_per, True, False),
        ("reciprocal-spectral/mesh_fft/periodic",
         dict(long_range_mode="reciprocal-spectral-v1", long_range_reciprocal_backend="mesh_fft", long_range_boundary="periodic", long_range_mesh_size=16), g_per, True, False),
        ("reciprocal-spectral/mesh_fft/slab",
         dict(long_range_mode="reciprocal-spectral-v1", long_range_reciprocal_backend="mesh_fft", long_range_boundary="slab", long_range_mesh_size=16, long_range_slab_padding_factor=2), g_per, False, True),
    ]
    for label, lr, graph, periodic, skip_equiv in configs:
        r = check(label, lr, graph, periodic, skip_equiv, device, dtype)
        print(f"[matrix] {r['label']:44s} build={r['build']:>12s} fwd={r['fwd']:>7s} "
              f"force={r['force']:>7s} equiv={r['equiv']:>12s} dbl-bwd={r['dbl']:>12s}")
    print("[matrix] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
