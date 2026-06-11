#!/usr/bin/env python3
"""AOTInductor export + load + verify for FSCETP LAMMPS-inference acceleration (A-phase).

Flattens forward + 1st-order force-autograd via make_fx (reusing
``makefx_compile.trace_and_compile_force(do_compile=False)``), hands the flat
GraphModule to ``torch.export.export``, AOTInductor-compiles it to a ``.pt2``
package, loads it back, and checks the loaded model's ``(energy, force)`` match
eager + measures inference speedup.

This is the Python-only validation step of the LAMMPS-AOTI plan (no C++/LAMMPS
yet): it answers "can the FSCETP model be AOTI-compiled, and how much does
inference speed up?" before investing in the C++ pair_style. Force is traced
INTO the graph (model.forward + autograd.grad over positions -> force output),
which is the settled approach from NequIP/Allegro + DeepMD (torch.export does
not emit a standalone backward; AOTI is inference-only).

Usage (on the 4090):
  python -m molecular_force_field.test.bench_aoti_export \
      --route fusion --atoms 256 --attn-heads 1 --dtype float32 --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from molecular_force_field.test.bench_ictd_fix_trainstep import build_model, make_fixed_graph, SPECIES
from molecular_force_field.training.makefx_compile import trace_and_compile_force


class _E0Wrap(torch.nn.Module):
    """Add E0(Z) to the bare model's per-atom energies so the exported .pt2 returns ABSOLUTE energies
    (a true drop-in for an E0-embedded TorchScript core). E0 is a per-atom constant -> forces are
    unchanged (its gradient w.r.t. positions is zero)."""

    def __init__(self, model, e0_lut):
        super().__init__()
        self.model = model
        self.register_buffer("e0_lut", e0_lut)

    def forward(self, pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        out = self.model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
        e_atom = out[0] if isinstance(out, tuple) else out
        e0 = self.e0_lut[A].to(e_atom.dtype).reshape(e_atom.shape)
        return e_atom + e0


def force_compute_fn_factory(model, *, training: bool):
    """compute_fn returning (E_atom, force=-dE/dpos) -- force traced into the graph."""
    def compute_fn(pos, A, batch, edge_src, edge_dst, edge_shifts, cell):
        p = pos.detach().requires_grad_(True)
        out = model(p, A, batch, edge_src, edge_dst, edge_shifts, cell)
        e_atom = out[0] if isinstance(out, tuple) else out
        grad = torch.autograd.grad(e_atom.sum(), p, create_graph=training)[0]
        return e_atom, -grad
    return compute_fn


def _aoti_compile(exported, out_path):
    """torch 2.x AOTInductor compile+package; tolerate API moves across versions."""
    try:
        from torch._inductor import aoti_compile_and_package
        return aoti_compile_and_package(exported, package_path=out_path)
    except TypeError:
        # older signature took the package path positionally / via options
        from torch._inductor import aoti_compile_and_package
        return aoti_compile_and_package(exported, out_path)


def _aoti_load(path, device):
    """Load an AOTI .pt2 back into a callable; tolerate API moves."""
    last = None
    for loader in ("aoti_load_package",):
        try:
            from torch._inductor import aoti_load_package  # type: ignore
            return aoti_load_package(path)
        except Exception as e:  # pragma: no cover
            last = e
    # fallback: torch._export.aot_load (older) returns a runner callable
    try:
        from torch._export import aot_load  # type: ignore
        return aot_load(path, device)
    except Exception as e:
        raise RuntimeError(f"no working AOTI loader (last: {last} / {e})")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _random_rotation(dtype, device, seed=12345):
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = torch.randn(3, 3, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(m)
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q.to(device=device, dtype=dtype)


def _time(fn, device, iters, warmup):
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) * 1e3 / iters


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route", default="fusion", choices=["baseline", "fusion"])
    p.add_argument("--atoms", type=int, default=256)
    p.add_argument("--degree", type=int, default=24)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--lmax", type=int, default=2)
    p.add_argument("--num-interaction", type=int, default=2)
    p.add_argument("--attn-heads", dest="attn_heads", type=int, default=1)
    p.add_argument("--contraction-order", dest="contraction_order", type=int, default=2)
    p.add_argument("--product-backend", default="ictd-pure-u")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint", default=None,
                   help="path to a trained .pth checkpoint to export (via LAMMPS_MLIAP_MFF.from_checkpoint). "
                        "When set, the REAL trained weights/dtype/species are used instead of a random model, "
                        "so the produced .pt2 is the one LAMMPS will actually run.")
    p.add_argument("--elements", default="H,C,N,O",
                   help="comma-separated element symbols for the checkpoint (from_checkpoint requires element_types; "
                        "the validation graph's species fall back to these if the model can't report its own)")
    p.add_argument("--avg-num-neighbors", dest="avg_num_neighbors", type=float, default=None,
                   help="message-normalization constant the weights were trained under (model divides messages "
                        "by it). For ictd-fix it is auto-computed from the training data and is NOT saved in the "
                        "checkpoint, so pass the TRAINING value (logged as 'Computed average number of neighbors') "
                        "or the deployed energies/forces are wrong (from_checkpoint else falls back to 14.38).")
    p.add_argument("--embed-e0", dest="embed_e0", action="store_true",
                   help="add E0(Z) atomic reference energies into the exported per-atom energy, so the .pt2 returns "
                        "ABSOLUTE energy (a drop-in for an E0-embedded TorchScript core). Forces are unaffected.")
    p.add_argument("--out", default="/tmp/fscetp_aoti.pt2")
    p.add_argument("--fallback", default=None,
                   help="path to an N-flexible TorchScript core (.pt) the LAMMPS engine should fall back to "
                        "when ntotal exceeds this .pt2's baked N (a ghost-count spike). Written into <out>.meta.")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dynamic", action="store_true",
                   help="export with dynamic_shapes (Dim for atom/edge counts) -> one .pt2 for all "
                        "system sizes (needed for LAMMPS' per-frame-varying N)")
    p.add_argument("--vary-degree", dest="vary_degree", type=int, default=0,
                   help="with --dynamic: also call the .pt2 at this avg degree (!= --degree) -> a "
                        "different EDGE count at the SAME atom count, proving one .pt2 handles varying "
                        "neighbor count (the thing that actually varies in LAMMPS NVE/NVT, where N is fixed)")
    p.add_argument("--no-equiv", dest="no_equiv", action="store_true",
                   help="skip the rotation-equivariance gate on the loaded .pt2")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    # Hard-off TF32: keep full float32 matmul precision. TF32 would drop matmul to ~1e-3
    # (orthogonal to compilation, but a real precision knob) -- we want numerics unchanged.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    if args.checkpoint:
        # Real deployment path: export the actual trained checkpoint. from_checkpoint rebuilds the
        # architecture from the checkpoint's hyperparameters and loads the (EMA) weights; the bare
        # energy model lives at obj.wrapper.model (a PureCartesianICTDFix, ZBL-wrapped only if the
        # checkpoint configured ZBL -- same 7-arg forward either way). The checkpoint's dtype/species
        # OVERRIDE the CLI so the .pt2 matches what LAMMPS will run.
        from molecular_force_field.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
        element_types = [s.strip() for s in args.elements.split(",") if s.strip()]
        obj = LAMMPS_MLIAP_MFF.from_checkpoint(
            checkpoint_path=args.checkpoint, element_types=element_types, device=args.device,
            avg_num_neighbors=args.avg_num_neighbors,
        )
        model = obj.wrapper.model
        model.skip_input_validation = True  # set on the BARE model (the A.max().item() guard lives here,
                                            # not on the E0 wrapper) so make_fx tracing stays data-independent
        dtype = next(model.parameters()).dtype  # honor the trained dtype (likely float64)
        species_z = [int(z) for z in (getattr(model, "atomic_numbers", None) or SPECIES) if int(z) > 0] or list(SPECIES)
        print(f"[aoti] loaded checkpoint {args.checkpoint}  trained_dtype={dtype}  species={species_z}  "
              f"avg_num_neighbors={getattr(model, 'avg_num_neighbors', None)}")
        if args.embed_e0:
            from molecular_force_field.cli.export_libtorch_core import _e0_lut_from_keys_values
            aek = obj.wrapper.atomic_energy_keys.detach().cpu()
            aev = obj.wrapper.atomic_energy_values.detach().cpu()
            lut = _e0_lut_from_keys_values(aek, aev, dtype=dtype, device=device)
            model = _E0Wrap(model, lut).to(device=device)
            print(f"[aoti] embedded E0(Z) for Z={aek.tolist()} -> absolute-energy .pt2")
    else:
        model = build_model(
            channels=args.channels, lmax=args.lmax, num_interaction=args.num_interaction,
            route=args.route, product_backend=args.product_backend, dtype=dtype, device=device,
            correlation=args.contraction_order, attn_heads=args.attn_heads,
        )
        species_z = list(SPECIES)
    model.eval()
    model.skip_input_validation = True

    def _apply_species(g):
        # When the checkpoint's element set differs from the harness default, re-draw the species
        # vector A from it (deterministic seed, so positions/edges -- and thus the traced shapes --
        # are unchanged; only the atomic numbers feeding the embedding change). Using a Z the model
        # was NOT trained on would index its embedding out of range -> CUDA device-side assert, so
        # EVERY graph fed to the model (incl. the VARY-E proof) must go through this.
        if species_z == list(SPECIES):
            return g
        n = int(g[0].shape[0])
        gA = torch.Generator(device="cpu").manual_seed(42)
        sp = torch.tensor(species_z, dtype=torch.long)
        A_new = sp[torch.randint(0, len(species_z), (n,), generator=gA)].to(device)
        return (g[0], A_new) + tuple(g[2:])

    graph = _apply_species(make_fixed_graph(num_nodes=args.atoms, avg_degree=args.degree, dtype=dtype, device=device))
    example_inputs = (graph[0],) + tuple(graph[1:])

    print(f"[aoti] device={device} dtype={dtype} atoms={args.atoms} edges={args.atoms*args.degree} "
          f"route={args.route} attn_heads={args.attn_heads} torch={torch.__version__}")

    # ---- eager reference ----
    eager_fn = force_compute_fn_factory(model, training=False)
    e_ref, f_ref = eager_fn(*example_inputs)
    e_ref = e_ref.detach(); f_ref = f_ref.detach()
    print(f"[aoti] eager energy_sum={e_ref.sum().item():.6e}  force_absmax={f_ref.abs().max().item():.6e}")

    # ---- make_fx flatten (force into the graph) ----
    try:
        gm = trace_and_compile_force(
            model, example_inputs, training=False,
            compute_fn=force_compute_fn_factory(model, training=False),
            do_compile=False,
        )
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"[aoti] make_fx FLATTEN FAILED: {type(ex).__name__}: {ex}")
        return 1
    e_gm, f_gm = gm(*example_inputs)
    dE_gm = (e_gm - e_ref).abs().max().item(); dF_gm = (f_gm - f_ref).abs().max().item()
    print(f"[aoti] flat gm vs eager: dE={dE_gm:.3e} dF={dF_gm:.3e}  (should be ~0, bit-identical)")

    # ---- torch.export (optionally with dynamic_shapes for varying atom/edge counts) ----
    dyn = None
    if args.dynamic:
        from torch.export import Dim
        # FSCETP's make_fx flatten BAKES the atom count N (the many view(*x.shape[:-1], C, 2l+1)
        # in the ICTD layers, same root cause as the training-side per-N cache) -- torch.export
        # confirms n_atoms/n_mol are constants. But the EDGE count E stays symbolic (edge ops are
        # scatter/index_select). This is exactly right for LAMMPS: in NVE/NVT the atom count is
        # FIXED (conserved), and what varies frame-to-frame is the neighbor (edge) count as atoms
        # move. So make ONLY E dynamic; N and num_mol stay static (one .pt2 per system size).
        # inputs order: (pos[N,3], A[N], batch[N], edge_src[E], edge_dst[E], edge_shifts[E,3], cell[M,3,3])
        Edim = Dim("n_edges", min=2)
        dyn = (None, None, None, {0: Edim}, {0: Edim}, {0: Edim}, None)
    try:
        exported = torch.export.export(gm, tuple(example_inputs), dynamic_shapes=dyn)
        print(f"[aoti] torch.export.export OK (dynamic={args.dynamic})")
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"[aoti] torch.export FAILED: {type(ex).__name__}: {ex}")
        return 1

    # ---- AOTInductor compile + package ----
    t0 = time.perf_counter()
    try:
        pt2 = _aoti_compile(exported, args.out)
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"[aoti] aoti_compile_and_package FAILED: {type(ex).__name__}: {ex}")
        return 1
    print(f"[aoti] compiled .pt2 -> {pt2}  ({time.perf_counter()-t0:.1f}s)")

    # Sidecar metadata the LAMMPS engine reads: baked atom count N (-> pad ntotal up to it), a valid
    # padding species, and an optional N-flexible TorchScript fallback for ntotal > N spikes.
    meta_path = str(args.out) + ".meta"
    pad_z = int(species_z[0]) if species_z else 1
    with open(meta_path, "w") as mf:
        mf.write(f"nmax {args.atoms}\n")
        mf.write(f"pad_z {pad_z}\n")
        if args.fallback:
            mf.write(f"fallback {args.fallback}\n")
    print(f"[aoti] wrote {meta_path}  (nmax={args.atoms} pad_z={pad_z} fallback={args.fallback})")

    # ---- load back + verify numerics ----
    try:
        loaded = _aoti_load(pt2, device)
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"[aoti] aoti load FAILED: {type(ex).__name__}: {ex}")
        return 1
    out = loaded(*example_inputs)
    e_a, f_a = out if isinstance(out, (tuple, list)) else (out, None)
    dE = (e_a - e_ref).abs().max().item()
    dF = (f_a - f_ref).abs().max().item() if f_a is not None else float("nan")
    escale = e_ref.abs().max().item() + 1e-30; fscale = f_ref.abs().max().item() + 1e-30
    tol = 1e-9 if dtype == torch.float64 else 3e-3
    ok = (dE / escale <= tol) and (dF / fscale <= tol)
    print(f"[aoti] AOTI vs eager: dE={dE:.3e} (rel {dE/escale:.2e})  dF={dF:.3e} (rel {dF/fscale:.2e})")
    print(f"[aoti] NUMERICAL-MATCH {'PASS' if ok else 'FAIL'} (rel tol {tol:.1e})")

    # ---- equivariance gate on the loaded .pt2 (HARD constraint): rotate positions;
    #      energy must be invariant and force covariant: f(x R^T) == f(x) R^T. ----
    if not args.no_equiv:
        R = _random_rotation(dtype, device)
        rest = tuple(example_inputs[1:])
        e0, f0 = loaded(example_inputs[0], *rest)
        e1, f1 = loaded(example_inputs[0] @ R.T, *rest)
        e_inv = (e1.sum() - e0.sum()).abs().item()
        f_cov = (f1 - f0 @ R.T).abs().max().item()
        e_sc = e0.abs().sum().item() + 1e-30
        f_sc = f0.abs().max().item() + 1e-30
        eqtol = 1e-9 if dtype == torch.float64 else 1e-3
        eq_ok = (e_inv / e_sc <= eqtol) and (f_cov / f_sc <= eqtol)
        print(f"[aoti] EQUIVARIANCE |E(Rx)-E(x)|={e_inv:.3e} (rel {e_inv/e_sc:.2e})  "
              f"|F(Rx)-F(x)R^T|={f_cov:.3e} (rel {f_cov/f_sc:.2e})  "
              f"{'PASS' if eq_ok else 'FAIL'} (rel tol {eqtol:.1e})")
        ok = ok and eq_ok
        del e0, f0, e1, f1

    # ---- dynamic-E proof: call the SAME .pt2 at a different EDGE count (same N) ----
    if args.dynamic and args.vary_degree > 0 and args.vary_degree != args.degree:
        g2 = _apply_species(make_fixed_graph(num_nodes=args.atoms, avg_degree=args.vary_degree, dtype=dtype, device=device))
        inp2 = (g2[0],) + tuple(g2[1:])
        e2e, f2e = eager_fn(*inp2); e2e = e2e.detach(); f2e = f2e.detach()
        try:
            e2a, f2a = loaded(*inp2)
            d2e = (e2a - e2e).abs().max().item(); d2f = (f2a - f2e).abs().max().item()
            v_ok = (d2e / (e2e.abs().max().item() + 1e-30) <= tol) and (d2f / (f2e.abs().max().item() + 1e-30) <= tol)
            print(f"[aoti] VARY-E: exported@{args.atoms*args.degree} called@{args.atoms*args.vary_degree} edges "
                  f"(same {args.atoms} atoms) -> dE={d2e:.3e} dF={d2f:.3e}  "
                  f"{'PASS (one .pt2 handles varying neighbor count)' if v_ok else 'FAIL'}")
            ok = ok and v_ok
        except Exception as ex:
            print(f"[aoti] VARY-E call FAILED: {type(ex).__name__}: {ex}")
            ok = False
        del e2e, f2e

    # Free the trace/export/compile intermediates before timing. Otherwise on large
    # float64 systems the script holds eager refs + flat gm + ExportedProgram +
    # compiled package + loaded model all at once, and the eager timing run then
    # OOMs (it is not an AOTI limitation -- the loaded .pt2 alone is small).
    del exported, gm, e_gm, f_gm, e_a, f_a
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ---- speed: eager forward+force vs AOTI ----
    t_eager = _time(lambda: eager_fn(*example_inputs), device, args.iters, args.warmup)
    t_aoti = _time(lambda: loaded(*example_inputs), device, args.iters, args.warmup)
    print(f"[aoti] eager fwd+force : {t_eager:8.3f} ms")
    print(f"[aoti] AOTI .pt2       : {t_aoti:8.3f} ms")
    print(f"[aoti] SPEEDUP         : {t_eager / max(t_aoti,1e-9):6.2f}x")
    if device.type == "cuda":
        print(f"[aoti] peak CUDA mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
