#!/usr/bin/env python3
"""torch.compile (Inductor) test for ictd-fix FUSION *inference*: forward + forces via
a SINGLE backward (autograd.grad, create_graph=False) — the MD/LAMMPS force-eval, NOT
the training double-backward. Reports graph-break count, numerical correctness vs eager,
and speedup at several atom counts. This is the Python proxy for whether AOTInductor
(the C++-deployable Inductor path) is worth wiring into the LAMMPS engine: AOTInductor
uses the same Inductor backend, so if torch.compile can't fuse/help the fusion forward
here, AOTInductor won't either.

Run: python -m molecular_force_field.test._compile_infer_test
"""
from __future__ import annotations

import time
import torch
import torch._dynamo as dynamo

from molecular_force_field.test.bench_ictd_fix_trainstep import build_model, make_fixed_graph


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def forces(model, pos, rest):
    e = model(pos, *rest)
    if isinstance(e, tuple):
        e = e[0]
    g = torch.autograd.grad(e.sum(), pos, create_graph=False)[0]
    return -g


def time_fn(fn, n, dev):
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    _sync(dev)
    return (time.perf_counter() - t0) * 1e3 / n


def run(atoms, degree, dtype, dev):
    torch.manual_seed(0)
    model = build_model(channels=64, lmax=2, num_interaction=2, route="fusion",
                        product_backend="ictd-pure-u", dtype=dtype, device=dev)
    model.eval()
    model.skip_input_validation = True
    g = make_fixed_graph(num_nodes=atoms, avg_degree=degree, dtype=dtype, device=dev)
    pos0 = g[0].detach().clone().requires_grad_(True)
    rest = tuple(g[1:])

    # eager reference + timing
    F_e = forces(model, pos0, rest)
    eager_ms = time_fn(lambda: forces(model, pos0, rest), 20, dev)

    # forward graph-break count (Dynamo explain)
    dynamo.reset()
    try:
        expl = dynamo.explain(model)(pos0, *rest)
        nbreaks = getattr(expl, "graph_break_count", getattr(expl, "graph_breaks", "?"))
        ngraphs = getattr(expl, "graph_count", "?")
    except Exception as ex:
        nbreaks = f"explain_err:{type(ex).__name__}"
        ngraphs = "?"

    # torch.compile (inductor) forward + AOTAutograd single backward
    dynamo.reset()
    cmodel = torch.compile(model, dynamic=False)
    ok = True
    err = ""
    comp_ms = float("nan")
    relerr = float("nan")
    try:
        t_c0 = time.perf_counter()
        for _ in range(3):  # warmup triggers compile of fwd + bwd
            _ = forces(cmodel, pos0, rest)
        _sync(dev)
        compile_s = time.perf_counter() - t_c0
        F_c = forces(cmodel, pos0, rest)
        d = (F_c - F_e).abs().max().item()
        s = F_e.abs().max().item() + 1e-30
        relerr = d / s
        comp_ms = time_fn(lambda: forces(cmodel, pos0, rest), 20, dev)
    except Exception as ex:
        ok = False
        err = f"{type(ex).__name__}: {str(ex)[:140]}"
        compile_s = float("nan")

    su = (eager_ms / comp_ms) if (comp_ms == comp_ms and comp_ms > 0) else float("nan")
    print(f"[ci] atoms={atoms} edges={atoms*degree} graphs={ngraphs} breaks={nbreaks} | "
          f"eager={eager_ms:.2f}ms compiled={comp_ms:.2f}ms speedup={su:.2f}x | "
          f"force_relerr={relerr:.2e} compile_time={compile_s:.1f}s ok={ok} {err}")


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float32
    dynamo.config.cache_size_limit = 256
    print(f"[ci] device={dev} dtype={dt} route=fusion(ictd-pure-u) torch={torch.__version__}")
    for atoms in [512, 1024, 2048]:
        try:
            run(atoms, 16, dt, dev)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(f"[ci] atoms={atoms} FAILED {type(ex).__name__}: {str(ex)[:150]}")
    print("AOTI_INFER_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
