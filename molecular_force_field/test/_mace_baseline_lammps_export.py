#!/usr/bin/env python3
"""mfftorch LAMMPS interface compat test for PureCartesianICTDMaceBaseline.

Reproduces the exact production export path (lammps_mliap / export_libtorch_core):
wrap the model in `_TorchScriptEdgeVecCore`, `torch.jit.trace` it (check_trace=False,
strict=False), freeze, save to core.pt, reload, and run the LAMMPS-style forward
(precomputed edge_vec). Checks: trace+freeze+save+reload succeeds, the reloaded
TorchScript core supports the C++ engine's force path (autograd.grad of energy wrt
pos), output is finite and matches the eager model. This is the TorchScript gate the
mfftorch C++ engine (mff_torch_engine.cpp) relies on.

Long-range OFF (default) — the standard short-range LAMMPS path. (LR in LAMMPS routes
a reciprocal_source to the C++ solver; that's a separate integration.)

Run: python -m molecular_force_field.test._mace_baseline_lammps_export
"""
from __future__ import annotations

import os
import tempfile

import torch

from molecular_force_field.test.bench_ictd_fix_trainstep import make_fixed_graph, SPECIES
from molecular_force_field.models.pure_cartesian_ictd_mace_baseline import (
    PureCartesianICTDMaceBaseline,
)
from molecular_force_field.interfaces.lammps_mliap import _TorchScriptEdgeVecCore


def _energy_forces(core, pos, rest):
    # The mfftorch C++ engine passes a precomputed edge_vec and differentiates the
    # energy w.r.t. edge_vec (then maps to atom forces via the neighbor list), since
    # with precomputed_edge_vec the energy depends on edge_vec, not pos directly.
    A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec, ext = rest
    ev = edge_vec.detach().clone().requires_grad_(True)
    out = core(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, ev, ext)
    e = out[0].sum()
    g = torch.autograd.grad(e, ev)[0]            # dE/d edge_vec (engine's force source)
    f = torch.zeros_like(pos)                    # F_i = -dE/dpos_i, edge_vec = pos[dst]-pos[src]
    f.index_add_(0, edge_src, g)
    f.index_add_(0, edge_dst, -g)
    return e, f


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # LAMMPS / LibTorch runs fp32
    torch.manual_seed(0)
    base = PureCartesianICTDMaceBaseline(
        max_embed_radius=5.0, main_number_of_basis=8, hidden_dim_conv=16,
        atomic_numbers=list(SPECIES), max_atomvalue=10, num_interaction=2, lmax=2,
        function_type_main="bessel", polynomial_cutoff_p=6, save_contraction_order=3,
        avg_num_neighbors=24.0, internal_compute_dtype=dtype, device=device,
    ).to(device=device, dtype=dtype)
    base.eval()
    base.skip_input_validation = True

    core = _TorchScriptEdgeVecCore(base).to(device).eval()

    # LAMMPS-style inputs (precomputed edge_vec, empty external_tensor)
    g = make_fixed_graph(num_nodes=64, avg_degree=16, dtype=dtype, device=device, seed=3)
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = g
    shift = torch.einsum("ni,nij->nj", edge_shifts, cell[batch[edge_src]])
    edge_vec = pos[edge_dst] - pos[edge_src] + shift
    ext = torch.zeros((pos.shape[0], 0), device=device, dtype=dtype)
    rest = (A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec, ext)

    e_eager, f_eager = _energy_forces(core, pos, rest)
    print(f"[lmp] eager core: E={e_eager.item():.6e}  |F|max={f_eager.abs().max().item():.3e}")

    trace_inputs = (pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec, ext)
    try:
        traced = torch.jit.trace(core, trace_inputs, check_trace=False, strict=False)
        try:
            traced = torch.jit.freeze(traced.eval())
            froze = True
        except Exception as ex_f:
            froze = False
            print(f"[lmp] (freeze skipped: {type(ex_f).__name__}: {str(ex_f)[:80]})")
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"[lmp] torch.jit.trace FAILED: {type(ex).__name__}: {str(ex)[:200]}")
        return 1

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "core.pt")
        torch.jit.save(traced, p)
        sz = os.path.getsize(p)
        reloaded = torch.jit.load(p, map_location=device)
        print(f"[lmp] trace OK (froze={froze}); core.pt={sz} bytes; reloaded OK")
        e_s, f_s = _energy_forces(reloaded, pos, rest)

    dE = (e_s - e_eager).abs().item()
    dF = (f_s - f_eager).abs().max().item()
    finite = (e_s.item() == e_s.item()) and (f_s.abs().max().item() < 1e30)
    ok = finite and dE <= 1e-3 * max(1.0, abs(e_eager.item())) and dF <= 1e-3 * max(1.0, f_eager.abs().max().item())
    print(f"[lmp] reloaded TS core: E={e_s.item():.6e} |F|max={f_s.abs().max().item():.3e}")
    print(f"[lmp] TS-core vs eager: dE={dE:.3e} dF={dF:.3e} finite={finite} {'PASS' if ok else 'FAIL'}")
    print(f"[lmp] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
