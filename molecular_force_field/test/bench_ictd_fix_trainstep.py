#!/usr/bin/env python3
"""Correctness + timing harness for the pure-cartesian-ictd-fix training step.

This isolates the exact training-step compute used by the trainer
(trainer.py: forward -> forces = -autograd.grad(E, pos, create_graph=True)
-> loss(energy, forces) -> loss.backward()) on a fixed-shape synthetic batch.

It serves two purposes for the training-speed work:

  --check : numerical + equivariance gate. Runs in float64, verifies energy is
            rotation-invariant and forces are rotation-covariant, and saves /
            compares a numerical reference (energy + forces) so that any
            acceleration change can be proven to not alter the numbers.

  --bench : times the full double-backward train step on a fixed-shape batch
            (CUDA-graph-friendly), with a forward/backward breakdown and an
            optional torch.profiler trace to reveal launch-bound vs compute-bound
            behaviour. This decides which acceleration direction pays off.

Usage:
  python -m molecular_force_field.test.bench_ictd_fix_trainstep --check
  python -m molecular_force_field.test.bench_ictd_fix_trainstep --check --save-ref /tmp/ref.pt
  python -m molecular_force_field.test.bench_ictd_fix_trainstep --check --compare-ref /tmp/ref.pt
  python -m molecular_force_field.test.bench_ictd_fix_trainstep --bench --atoms 512 --profile
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from molecular_force_field.models.pure_cartesian_ictd_fix import PureCartesianICTDFix
from molecular_force_field.utils.config import ModelConfig

SPECIES = (1, 6, 7, 8)


def build_model(
    *,
    channels: int,
    lmax: int,
    num_interaction: int,
    route: str,
    product_backend: str,
    dtype: torch.dtype,
    device: torch.device,
    correlation: int = 2,
    attn_heads: int = 0,
    **extra,
) -> PureCartesianICTDFix:
    cfg = ModelConfig(dtype=dtype)
    cfg.channel_in = channels
    cfg.irreps_output_conv_channels = channels
    cfg.lmax = lmax
    model = PureCartesianICTDFix(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=channels,
        hidden_dim_sh=cfg.get_hidden_dim_sh(),
        hidden_dim=64,
        channel_in2=32,
        embedding_dim=16,
        max_atomvalue=10,
        atomic_numbers=list(SPECIES),
        output_size=8,
        embed_size=cfg.embed_size,
        main_hidden_sizes3=cfg.main_hidden_sizes3,
        num_layers=cfg.num_layers,
        num_interaction=num_interaction,
        function_type_main="bessel",
        lmax=lmax,
        ictd_fix_route=route,
        ictd_fix_product_backend=product_backend,
        ictd_fix_fusion_scale_init=1.0,
        ictd_fix_fusion_heads=1,
        ictd_fix_interaction_attn_heads=attn_heads,
        save_contraction_order=correlation,
        avg_num_neighbors=float(24),
        internal_compute_dtype=dtype,
        device=device,
        **extra,
    ).to(device=device, dtype=dtype)
    return model


def make_fixed_graph(
    *,
    num_nodes: int,
    avg_degree: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 42,
):
    """Deterministic fixed-shape single-molecule graph (no self-loops)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = (torch.randn(num_nodes, 3, generator=g, dtype=torch.float64) * 2.5)
    species = torch.tensor(SPECIES, dtype=torch.long)
    A = species[torch.randint(0, len(SPECIES), (num_nodes,), generator=g)]
    num_edges = num_nodes * avg_degree
    edge_src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    edge_dst = torch.randint(0, num_nodes, (num_edges,), generator=g)
    # remove self loops by nudging dst deterministically
    loop = edge_src == edge_dst
    edge_dst[loop] = (edge_dst[loop] + 1) % num_nodes
    edge_shifts = torch.zeros(num_edges, 3, dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 100.0  # big box, no PBC wrap
    batch = torch.zeros(num_nodes, dtype=torch.long)

    pos = pos.to(device=device, dtype=dtype)
    A = A.to(device=device)
    batch = batch.to(device=device)
    edge_src = edge_src.to(device=device)
    edge_dst = edge_dst.to(device=device)
    edge_shifts = edge_shifts.to(device=device, dtype=dtype)
    cell = cell.to(device=device, dtype=dtype)
    return pos, A, batch, edge_src, edge_dst, edge_shifts, cell


def forward_energy_atom(model, pos, graph):
    _, A, batch, edge_src, edge_dst, edge_shifts, cell = graph
    out = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    if isinstance(out, tuple):
        out = out[0]
    return out  # [N, 1] per-atom energy


def compute_energy_forces(model, graph, *, create_graph: bool):
    pos = graph[0].detach().clone().requires_grad_(True)
    g2 = (pos,) + tuple(graph[1:])
    e_atom = forward_energy_atom(model, pos, g2)
    energy = e_atom.sum()
    grad = torch.autograd.grad(energy, pos, create_graph=create_graph)[0]
    forces = -grad
    return energy, forces, e_atom


def random_rotation(dtype=torch.float64, seed=20260503) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = torch.randn(3, 3, generator=g, dtype=dtype)
    q, _ = torch.linalg.qr(m)
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def summarize(t: torch.Tensor) -> dict:
    t = t.detach().double().flatten()
    return {
        "mean": t.mean().item(),
        "std": t.std().item(),
        "absmax": t.abs().max().item(),
        "l2": t.norm().item(),
    }


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_check(args) -> int:
    device = torch.device(args.device)
    dtype = torch.float64
    torch.manual_seed(0)
    model = build_model(
        channels=args.channels, lmax=args.lmax, num_interaction=args.num_interaction,
        route=args.route, product_backend=args.product_backend, dtype=dtype, device=device,
        correlation=args.contraction_order, attn_heads=args.attn_heads,
    )
    model.eval()
    graph = make_fixed_graph(num_nodes=args.atoms, avg_degree=args.degree, dtype=dtype, device=device)

    energy, forces, _ = compute_energy_forces(model, graph, create_graph=False)

    # ---- equivariance: rotate positions, energy invariant, forces covariant ----
    R = random_rotation(dtype=dtype).to(device)
    graph_rot = (graph[0] @ R.T,) + tuple(graph[1:])
    energy_r, forces_r, _ = compute_energy_forces(model, graph_rot, create_graph=False)

    e_err = (energy_r - energy).abs().item()
    # pos_rot = pos @ R.T  =>  forces_rot should equal forces @ R.T
    f_cov_err = (forces_r - forces @ R.T).abs().max().item()
    f_scale = forces.abs().max().item()

    print(f"[check] device={device} dtype={dtype} atoms={args.atoms} edges={args.atoms*args.degree} "
          f"channels={args.channels} lmax={args.lmax} L={args.num_interaction} route={args.route} "
          f"backend={args.product_backend}")
    print(f"[check] energy={energy.item():.10e}  |E(Rx)-E(x)|={e_err:.3e}")
    print(f"[check] force absmax={f_scale:.6e}  equivariance |F(Rx)-F(x)R^T|_inf={f_cov_err:.3e}")
    print(f"[check] energy summary {summarize(energy.reshape(1))}")
    print(f"[check] forces summary {summarize(forces)}")

    tol = 1e-8 * max(1.0, abs(energy.item()))
    f_tol = 1e-8 * max(1.0, f_scale)
    ok = (e_err <= tol) and (f_cov_err <= f_tol)
    print(f"[check] EQUIVARIANCE {'PASS' if ok else 'FAIL'} (e_tol={tol:.2e}, f_tol={f_tol:.2e})")

    if args.stress:
        def _stress(g):
            p = g[0].detach().clone().requires_grad_(True)
            A_, b_, es_, ed_, esh_, c_ = g[1], g[2], g[3], g[4], g[5], g[6]
            nm = int(b_.max().item()) + 1
            strn = torch.zeros(nm, 3, 3, device=device, dtype=dtype, requires_grad=True)
            pin, cin = _apply_strain(p, c_, b_, strn)
            out = model(pin, A_, b_, es_, ed_, esh_, cin)
            if isinstance(out, tuple):
                out = out[0]
            sg = torch.autograd.grad(out.sum(), strn, create_graph=False)[0]
            vol = _det3x3(c_).abs().clamp_min(1e-10)
            return sg / vol.view(-1, 1, 1)
        stress = _stress(graph)
        # rotate BOTH positions and cell; stress is rank-2: S(Rx) = R S(x) R^T
        graph_rot_full = (graph[0] @ R.T, graph[1], graph[2], graph[3], graph[4], graph[5], graph[6] @ R.T)
        stress_r = _stress(graph_rot_full)
        s_cov_err = (stress_r - torch.matmul(torch.matmul(R, stress), R.transpose(-1, -2))).abs().max().item()
        s_scale = stress.abs().max().item()
        s_tol = 1e-8 * max(1.0, s_scale)
        s_ok = s_cov_err <= s_tol
        print(f"[check] stress absmax={s_scale:.6e}  equivariance |S(Rx)-R S(x) R^T|_inf={s_cov_err:.3e}")
        print(f"[check] STRESS-EQUIVARIANCE {'PASS' if s_ok else 'FAIL'} (s_tol={s_tol:.2e})")
        ok = ok and s_ok

    if args.save_ref:
        torch.save({"energy": energy.detach().cpu(), "forces": forces.detach().cpu(),
                    "meta": vars(args)}, args.save_ref)
        print(f"[check] saved reference -> {args.save_ref}")

    if args.compare_ref:
        ref = torch.load(args.compare_ref, map_location="cpu", weights_only=False)
        de = (energy.detach().cpu() - ref["energy"]).abs().max().item()
        df = (forces.detach().cpu() - ref["forces"]).abs().max().item()
        print(f"[check] vs ref {args.compare_ref}: dE_inf={de:.3e}  dF_inf={df:.3e}")
        num_ok = (de <= 1e-9 * max(1.0, abs(energy.item()))) and (df <= 1e-9 * max(1.0, f_scale))
        print(f"[check] NUMERICAL-MATCH {'PASS' if num_ok else 'FAIL'}")
        ok = ok and num_ok

    return 0 if ok else 1


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_section(fn, *, device, iters, warmup, label=None):
    for _ in range(warmup):
        fn()
    _sync(device)
    ts = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        ts.append((time.perf_counter() - t0) * 1e3)
    if label is not None:
        # Print the FULL per-iter series + stats so steady-state is auditable:
        # a recompile or cache miss shows up as a latency spike, and a flat
        # series proves the reported median is the post-warmup steady speed.
        import statistics as _st
        srt = sorted(ts)
        med = srt[len(srt) // 2]
        spread = (max(ts) - min(ts)) / max(med, 1e-9)
        print(f"[timing:{label}] n={len(ts)} warmup={warmup} min={min(ts):.3f} "
              f"median={med:.3f} mean={_st.mean(ts):.3f} max={max(ts):.3f} "
              f"std={_st.pstdev(ts):.3f} ms  (max-min)/median={spread:.2f}")
        print(f"[timing:{label}] per-iter ms: " + " ".join(f"{t:.2f}" for t in ts))
    ts.sort()
    return ts[len(ts) // 2]  # median ms


def run_bench(args) -> int:
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(0)
    model = build_model(
        channels=args.channels, lmax=args.lmax, num_interaction=args.num_interaction,
        route=args.route, product_backend=args.product_backend, dtype=dtype, device=device,
        correlation=args.contraction_order, attn_heads=args.attn_heads,
    )
    model.train()
    graph = make_fixed_graph(num_nodes=args.atoms, avg_degree=args.degree, dtype=dtype, device=device)
    params = list(model.parameters())
    f_target = torch.zeros(args.atoms, 3, device=device, dtype=dtype)
    e_target = torch.zeros((), device=device, dtype=dtype)
    force_weight = 10.0

    def forward_only():
        pos = graph[0].detach().clone().requires_grad_(True)
        g2 = (pos,) + tuple(graph[1:])
        e_atom = forward_energy_atom(model, pos, g2)
        return e_atom.sum()

    def forward_and_forces():
        pos = graph[0].detach().clone().requires_grad_(True)
        g2 = (pos,) + tuple(graph[1:])
        e_atom = forward_energy_atom(model, pos, g2)
        energy = e_atom.sum()
        grad = torch.autograd.grad(energy, pos, create_graph=True)[0]
        return energy, -grad

    def full_train_step():
        for p in params:
            p.grad = None
        pos = graph[0].detach().clone().requires_grad_(True)
        g2 = (pos,) + tuple(graph[1:])
        e_atom = forward_energy_atom(model, pos, g2)
        energy = e_atom.sum()
        grad = torch.autograd.grad(energy, pos, create_graph=True)[0]
        forces = -grad
        e_loss = (energy - e_target) ** 2
        f_loss = torch.mean((forces - f_target) ** 2)
        loss = e_loss + force_weight * f_loss
        loss.backward()
        return loss

    print(f"[bench] device={device} dtype={dtype} atoms={args.atoms} edges={args.atoms*args.degree} "
          f"channels={args.channels} lmax={args.lmax} L={args.num_interaction} route={args.route} "
          f"backend={args.product_backend}")
    if device.type == "cuda":
        print(f"[bench] gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")

    fwd = _time_section(forward_only, device=device, iters=args.iters, warmup=args.warmup)
    fwd_grad = _time_section(lambda: forward_and_forces(), device=device, iters=args.iters, warmup=args.warmup)
    full = _time_section(full_train_step, device=device, iters=args.iters, warmup=args.warmup)
    print(f"[bench] forward-only            : {fwd:8.3f} ms")
    print(f"[bench] forward + forces(grad)  : {fwd_grad:8.3f} ms   (1st backward, create_graph=True)")
    print(f"[bench] full step (+2nd backward): {full:8.3f} ms   (= train step)")
    print(f"[bench] => forces-grad adds {fwd_grad - fwd:.3f} ms; 2nd backward adds {full - fwd_grad:.3f} ms")
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"[bench] peak CUDA mem: {mem:.2f} GB")

    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        acts = [ProfilerActivity.CPU]
        if device.type == "cuda":
            acts.append(ProfilerActivity.CUDA)
        for _ in range(3):
            full_train_step()
        _sync(device)
        with profile(activities=acts, record_shapes=False, with_stack=False) as prof:
            for _ in range(args.profile_iters):
                full_train_step()
            _sync(device)
        sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
        print("\n[profile] top ops by", sort_key)
        print(prof.key_averages().table(sort_by=sort_key, row_limit=25))
        # launch-bound signal: count of kernel launches vs total time
        evts = prof.key_averages()
        ncalls = sum(e.count for e in evts)
        print(f"[profile] total profiled op-calls over {args.profile_iters} steps: {ncalls} "
              f"(~{ncalls/args.profile_iters:.0f}/step)")
        if args.trace:
            prof.export_chrome_trace(args.trace)
            print(f"[profile] chrome trace -> {args.trace}")

    return 0


def _det3x3(m: torch.Tensor) -> torch.Tensor:
    """Explicit 3x3 determinant (capture-safe; avoids cuSOLVER in torch.det). m: [...,3,3]."""
    return (
        m[..., 0, 0] * (m[..., 1, 1] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 1])
        - m[..., 0, 1] * (m[..., 1, 0] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 0])
        + m[..., 0, 2] * (m[..., 1, 0] * m[..., 2, 1] - m[..., 1, 1] * m[..., 2, 0])
    )


def _apply_strain(pos: torch.Tensor, cell: torch.Tensor, batch: torch.Tensor, strain: torch.Tensor):
    """Mirror the trainer's stress strain deformation: deformation = I + sym(strain),
    pos_input = pos @ deformation_per_atom, cell_input = cell @ deformation. Returns
    (pos_input, cell_input). strain has requires_grad so stress = dE/dstrain."""
    sym = 0.5 * (strain + strain.transpose(-1, -2))
    I3 = torch.eye(3, device=pos.device, dtype=pos.dtype)
    deform = I3 + sym  # [M,3,3]
    pos_input = torch.einsum("ni,nij->nj", pos, deform[batch])
    cell_input = torch.bmm(cell, deform)
    return pos_input, cell_input


def _flat_grads(params) -> torch.Tensor:
    gs = [p.grad.detach().flatten() for p in params if p.grad is not None]
    return torch.cat(gs) if gs else torch.zeros(1)


def run_cudagraph(args) -> int:
    """Capture the full double-backward train step as a CUDA graph and prove it
    replays bit-for-bit (within atomic-scatter tolerance) vs eager, then time it."""
    device = torch.device(args.device)
    if device.type != "cuda":
        print("[cudagraph] requires CUDA"); return 1
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(0)
    model = build_model(
        channels=args.channels, lmax=args.lmax, num_interaction=args.num_interaction,
        route=args.route, product_backend=args.product_backend, dtype=dtype, device=device,
        correlation=args.contraction_order, attn_heads=args.attn_heads,
    )
    model.eval()
    model.skip_input_validation = True  # remove host syncs so forward is capturable
    graph = make_fixed_graph(num_nodes=args.atoms, avg_degree=args.degree, dtype=dtype, device=device)

    static_pos = graph[0].detach().clone().requires_grad_(True)
    rest = tuple(graph[1:])  # (A, batch, edge_src, edge_dst, edge_shifts, cell) -- fixed
    f_tgt = torch.zeros(args.atoms, 3, device=device, dtype=dtype)
    e_tgt = torch.zeros((), device=device, dtype=dtype)
    fw = 10.0
    params = [p for p in model.parameters() if p.requires_grad]

    def capture_step():
        e_atom = model(static_pos, *rest)
        if isinstance(e_atom, tuple):
            e_atom = e_atom[0]
        energy = e_atom.sum()
        gpos = torch.autograd.grad(energy, static_pos, create_graph=True)[0]
        forces = -gpos
        loss = (energy - e_tgt) ** 2 + fw * ((forces - f_tgt) ** 2).mean()
        loss.backward()
        return loss.detach().clone()

    def zero_all():
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        if static_pos.grad is not None:
            static_pos.grad.zero_()

    print(f"[cudagraph] device={device} dtype={dtype} atoms={args.atoms} edges={args.atoms*args.degree} "
          f"channels={args.channels} lmax={args.lmax} L={args.num_interaction} route={args.route}")
    print(f"[cudagraph] gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")

    # ---------- eager reference ----------
    for p in params:
        p.grad = None
    static_pos.grad = None
    loss_eager = capture_step()
    grads_eager = _flat_grads(params).clone()

    # ---------- capture ----------
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(args.warmup):
            zero_all()
            capture_step()
    torch.cuda.current_stream().wait_stream(s)
    zero_all()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            static_loss = capture_step()
    except Exception as e:
        import traceback
        print(f"[cudagraph] CAPTURE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    print("[cudagraph] capture OK")

    # ---------- replay + numerical compare ----------
    zero_all()
    g.replay()
    torch.cuda.synchronize()
    loss_graph = static_loss.clone()
    grads_graph = _flat_grads(params).clone()

    dloss = (loss_graph - loss_eager).abs().item()
    rloss = dloss / (loss_eager.abs().item() + 1e-30)
    dgrad = (grads_graph - grads_eager).abs().max().item()
    gscale = grads_eager.abs().max().item()
    rgrad = dgrad / (gscale + 1e-30)
    print(f"[cudagraph] loss eager={loss_eager.item():.10e} graph={loss_graph.item():.10e} "
          f"|d|={dloss:.3e} rel={rloss:.3e}")
    print(f"[cudagraph] grads: max|d|={dgrad:.3e} scale={gscale:.3e} rel={rgrad:.3e} (n={grads_eager.numel()})")
    tol = 1e-10 if dtype == torch.float64 else 2e-4
    ok = (rloss <= tol) and (rgrad <= tol)
    print(f"[cudagraph] NUMERICAL-MATCH {'PASS' if ok else 'FAIL'} (rel tol={tol:.1e})")

    # ---------- timing: eager full step vs graph replay ----------
    def eager_step():
        zero_all()
        capture_step()

    def graph_step():
        zero_all()
        g.replay()

    t_eager = _time_section(eager_step, device=device, iters=args.iters, warmup=args.warmup)
    t_graph = _time_section(graph_step, device=device, iters=args.iters, warmup=args.warmup)
    print(f"[cudagraph] eager full step : {t_eager:8.3f} ms")
    print(f"[cudagraph] graph replay    : {t_graph:8.3f} ms")
    print(f"[cudagraph] SPEEDUP         : {t_eager / max(t_graph,1e-9):6.2f}x")
    return 0 if ok else 1


def _count_graph_breaks() -> int:
    try:
        from torch._dynamo.utils import counters
        return sum(counters["graph_break"].values())
    except Exception:
        return -1


def _compiled_autograd_ctx(compiler_fn):
    import torch._dynamo.compiled_autograd as ca
    fn = getattr(ca, "enable", None) or getattr(ca, "_enable")
    return fn(compiler_fn)


def _param_vec(params) -> torch.Tensor:
    return torch.cat([p.detach().double().flatten() for p in params])


def run_cudagraph_train(args) -> int:
    """Fixed-shape CUDA-graph TRAINING benchmark: capture the full train-step
    compute (forward -> forces via create_graph -> loss -> backward) as a CUDA
    graph, then run N optimizer steps by replaying it (optimizer.step stays eager).
    Compares the loss trajectory + final weights vs an identical eager training run
    (numerically identical replay), and reports ms/step speedup. Requires fixed
    shapes; CUDA-graph does NOT need the fusion-bias freeze (the None bug is
    compiled-autograd-only), so all params train."""
    import copy
    device = torch.device(args.device)
    if device.type != "cuda":
        print("[cudagraph-train] requires CUDA"); return 1
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(0)
    model = build_model(
        channels=args.channels, lmax=args.lmax, num_interaction=args.num_interaction,
        route=args.route, product_backend=args.product_backend, dtype=dtype, device=device,
        correlation=args.contraction_order, attn_heads=args.attn_heads,
    )
    model.train()
    model.skip_input_validation = True
    graph = make_fixed_graph(num_nodes=args.atoms, avg_degree=args.degree, dtype=dtype, device=device)
    static_pos = graph[0].detach().clone().requires_grad_(True)
    rest = tuple(graph[1:])
    f_tgt = torch.zeros(args.atoms, 3, device=device, dtype=dtype)
    e_tgt = torch.zeros((), device=device, dtype=dtype)
    fw = 10.0
    lr = 1e-3
    N = int(args.train_steps)
    params = [p for p in model.parameters() if p.requires_grad]
    init_state = copy.deepcopy(model.state_dict())

    def compute_step():
        e_atom = model(static_pos, *rest)
        if isinstance(e_atom, tuple):
            e_atom = e_atom[0]
        energy = e_atom.sum()
        gpos = torch.autograd.grad(energy, static_pos, create_graph=True)[0]
        forces = -gpos
        loss = (energy - e_tgt) ** 2 + fw * ((forces - f_tgt) ** 2).mean()
        loss.backward()
        return loss.detach().clone()

    def zero_all():
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        if static_pos.grad is not None:
            static_pos.grad.zero_()

    print(f"[cudagraph-train] device={device} dtype={dtype} atoms={args.atoms} edges={args.atoms*args.degree} "
          f"channels={args.channels} lmax={args.lmax} route={args.route} steps={N}")
    print(f"[cudagraph-train] gpu={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    # ---------- EAGER reference training ----------
    model.load_state_dict(init_state)
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(3):  # warmup (perturbs params)
        opt.zero_grad(set_to_none=True); compute_step(); opt.step()
    model.load_state_dict(init_state)
    opt = torch.optim.Adam(params, lr=lr)
    eager_losses = []
    _sync(device); t0 = time.perf_counter()
    for _ in range(N):
        opt.zero_grad(set_to_none=True)
        loss = compute_step()
        opt.step()
        eager_losses.append(loss.item())
    _sync(device); t_eager = (time.perf_counter() - t0) * 1e3 / N
    eager_final = _param_vec(params).clone()

    # ---------- GRAPHED training ----------
    model.load_state_dict(init_state)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(args.warmup):
            zero_all(); compute_step()
    torch.cuda.current_stream().wait_stream(s)
    zero_all()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            static_loss = compute_step()
    except Exception as e:
        import traceback
        print(f"[cudagraph-train] CAPTURE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    print("[cudagraph-train] capture OK")
    model.load_state_dict(init_state)  # reset params to init (warmup did no optimizer step; safe-reset)
    opt_g = torch.optim.Adam(params, lr=lr)
    graph_losses = []
    _sync(device); t0 = time.perf_counter()
    for _ in range(N):
        zero_all()       # in-place: keep static grad buffers
        g.replay()
        opt_g.step()
        graph_losses.append(static_loss.item())
    _sync(device); t_graph = (time.perf_counter() - t0) * 1e3 / N
    graph_final = _param_vec(params).clone()

    # ---------- compare ----------
    traj_dev = max(abs(a - b) for a, b in zip(eager_losses, graph_losses))
    lscale = max(abs(x) for x in eager_losses) + 1e-30
    dparam = (graph_final - eager_final).abs().max().item()
    pscale = eager_final.abs().max().item() + 1e-30
    print(f"[cudagraph-train] loss[0]  eager={eager_losses[0]:.6e} graph={graph_losses[0]:.6e}")
    print(f"[cudagraph-train] loss[-1] eager={eager_losses[-1]:.6e} graph={graph_losses[-1]:.6e}")
    print(f"[cudagraph-train] max loss-traj |d|={traj_dev:.3e} (rel {traj_dev/lscale:.3e}) over {N} steps")
    print(f"[cudagraph-train] final weights max|d|={dparam:.3e} (rel {dparam/pscale:.3e})")
    tol = 1e-9 if dtype == torch.float64 else 5e-3
    ok = (dparam / pscale) <= tol
    print(f"[cudagraph-train] NUMERICAL-MATCH {'PASS' if ok else 'FAIL'} (rel tol={tol:.1e})")
    print(f"[cudagraph-train] eager  : {t_eager:8.3f} ms/step")
    print(f"[cudagraph-train] graph  : {t_graph:8.3f} ms/step")
    print(f"[cudagraph-train] SPEEDUP: {t_eager / max(t_graph,1e-9):6.2f}x")
    return 0 if ok else 1


def run_compile(args) -> int:
    """Test torch.compile (forward) and optional compiled-autograd (through the
    create_graph double-backward) at the compute-bound large-atom regime.
    Gate numerics + equivariance; report graph breaks and speedup vs eager."""
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 256
    device = torch.device(args.device)
    if device.type != "cuda":
        print("[compile] requires CUDA"); return 1
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.manual_seed(0)
    model = build_model(
        channels=args.channels, lmax=args.lmax, num_interaction=args.num_interaction,
        route=args.route, product_backend=args.product_backend, dtype=dtype, device=device,
        correlation=args.contraction_order, attn_heads=args.attn_heads,
    )
    model.eval()
    model.skip_input_validation = True
    if getattr(args, "freeze_fusion_bias", False):
        nfz = 0
        for n, p in model.named_parameters():
            if n.endswith(".bias") and "fusion_readouts" in n:
                p.requires_grad_(False)
                nfz += 1
        print(f"[compile] froze {nfz} fusion_readouts bias params (additive energy terms; "
              f"None 2nd-order force grad blocks compiled-autograd)")
    graph = make_fixed_graph(num_nodes=args.atoms, avg_degree=args.degree, dtype=dtype, device=device)
    base_pos = graph[0].detach().clone()
    rest = tuple(graph[1:])   # (A, batch, edge_src, edge_dst, edge_shifts, cell)
    batch_idx = rest[1]
    cell0 = rest[-1]
    num_mols = int(batch_idx.max().item()) + 1
    f_tgt = torch.zeros(args.atoms, 3, device=device, dtype=dtype)
    e_tgt = torch.zeros((), device=device, dtype=dtype)
    s_tgt = torch.zeros(num_mols, 3, 3, device=device, dtype=dtype)
    fw = 10.0
    sw = float(args.stress_weight)

    params = [p for p in model.parameters() if p.requires_grad]

    def make_step(fwd, use_ca, ca_compiler):
        def step():
            for p in params:
                p.grad = None
            pos = base_pos.detach().clone().requires_grad_(True)
            if args.stress:
                strain = torch.zeros(num_mols, 3, 3, device=device, dtype=dtype, requires_grad=True)
                pos_in, cell_in = _apply_strain(pos, cell0, batch_idx, strain)
                e_atom = fwd(pos_in, rest[0], batch_idx, rest[2], rest[3], rest[4], cell_in)
            else:
                strain = None
                e_atom = fwd(pos, *rest)
            if isinstance(e_atom, tuple):
                e_atom = e_atom[0]
            energy = e_atom.sum()
            targets = [pos] if strain is None else [pos, strain]
            grads = torch.autograd.grad(energy, targets, create_graph=True)
            forces = -grads[0]
            loss = (energy - e_tgt) ** 2 + fw * ((forces - f_tgt) ** 2).mean()
            if strain is not None:
                volume = _det3x3(cell0).abs().clamp_min(1e-10)
                stress = grads[1] / volume.view(-1, 1, 1)
                loss = loss + sw * ((stress - s_tgt) ** 2).mean()
            if use_ca:
                with _compiled_autograd_ctx(ca_compiler):
                    loss.backward()
            else:
                loss.backward()
            return loss.detach().clone(), forces.detach().clone()
        return step

    print(f"[compile] device={device} dtype={dtype} atoms={args.atoms} edges={args.atoms*args.degree} "
          f"channels={args.channels} lmax={args.lmax} L={args.num_interaction} tf32={args.tf32} "
          f"fullgraph={args.fullgraph} compiled_autograd={args.compiled_autograd} mode={args.compile_mode} "
          f"stress={args.stress}")

    # ---- eager reference ----
    eager_step = make_step(model, False, None)
    loss_e, forces_e = eager_step()
    grads_e = _flat_grads(params).clone()

    # ---- compile ----
    dynamo.reset()
    try:
        from torch._dynamo.utils import counters
        counters.clear()
    except Exception:
        pass
    _ca_mode = None if args.compile_mode == "default" else args.compile_mode

    def ca_compiler(gm, **kw):
        return torch.compile(gm, dynamic=False, mode=_ca_mode)

    if args.compiled_autograd:
        # Forward stays EAGER: torch.compile's AOTAutograd backward cannot be
        # double-differentiated ("does not support double backward"). compiled-
        # autograd instead traces+compiles the backward graph at runtime, which
        # does support the create_graph double-backward.
        comp_step = make_step(model, True, ca_compiler)
    else:
        cmodel = torch.compile(model, dynamic=False, fullgraph=args.fullgraph,
                               mode=(None if args.compile_mode == "default" else args.compile_mode))
        comp_step = make_step(cmodel, False, ca_compiler)
    cold_ms = []
    t_compile0 = time.perf_counter()
    try:
        for _ in range(5):  # triggers compilation; time each iter to expose the compile cliff
            _sync(device)
            _c0 = time.perf_counter()
            comp_step()
            _sync(device)
            cold_ms.append((time.perf_counter() - _c0) * 1e3)
    except Exception as ex:
        import traceback
        print(f"[compile] COMPILED STEP FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        return 1
    compile_secs = time.perf_counter() - t_compile0
    gb = _count_graph_breaks()
    print(f"[compile] compiled OK; warmup+compile {compile_secs:.1f}s; graph_breaks={gb}")
    print(f"[compile] cold per-iter ms (compile lands on iter0, then converges to steady): "
          + " ".join(f"{t:.1f}" for t in cold_ms))

    # ---- numerics ----
    loss_c, forces_c = comp_step()
    grads_c = _flat_grads(params).clone()
    dloss = (loss_c - loss_e).abs().item()
    rloss = dloss / (loss_e.abs().item() + 1e-30)
    dgrad = (grads_c - grads_e).abs().max().item()
    gscale = grads_e.abs().max().item()
    rgrad = dgrad / (gscale + 1e-30)
    dforce = (forces_c - forces_e).abs().max().item()
    print(f"[compile] loss eager={loss_e.item():.8e} comp={loss_c.item():.8e} rel={rloss:.3e}")
    print(f"[compile] grads max|d|={dgrad:.3e} rel={rgrad:.3e} | forces max|d|={dforce:.3e}")
    tol = 1e-9 if dtype == torch.float64 else 3e-3
    ok = (rloss <= tol) and (rgrad <= tol)
    print(f"[compile] NUMERICAL-MATCH {'PASS' if ok else 'FAIL'} (rel tol={tol:.1e})")

    # ---- timing (steady-state: warmup excluded, full per-iter series printed) ----
    _ca_lbl = "compiled-autograd" if args.compiled_autograd else "compiled"
    t_eager = _time_section(lambda: eager_step(), device=device, iters=args.iters, warmup=args.warmup, label="eager")
    t_comp = _time_section(lambda: comp_step(), device=device, iters=args.iters, warmup=args.warmup, label=_ca_lbl)
    print(f"[compile] eager full step : {t_eager:8.3f} ms")
    print(f"[compile] compiled step   : {t_comp:8.3f} ms")
    print(f"[compile] SPEEDUP         : {t_eager / max(t_comp,1e-9):6.2f}x  (steady-state median, compile excluded)")
    return 0 if ok else 1


def run_makefx(args) -> int:
    """make_fx-compile route: flatten the forward + inner force-autograd into one
    FX graph (so dE/dx is ordinary ops, not a hidden autograd call), then
    torch.compile it -- letting Inductor do a single ordinary backward over the
    flat graph for the optimizer step, sidestepping the second-order limit that
    blocks a direct torch.compile of the train step.

    Gates numerics (loss / forces / param-grads) + equivariance vs eager, then
    times it. On CPU (or --makefx-no-compile) it skips Inductor and validates
    only the flatten+strip+rebuild stages -- this proves param-grad connectivity
    and detach-strip correctness without a GPU."""
    from molecular_force_field.training.makefx_compile import trace_and_compile_force

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    do_compile = (device.type == "cuda") and (not args.makefx_no_compile)
    if do_compile:
        import torch._dynamo as dynamo
        dynamo.config.cache_size_limit = 256
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.manual_seed(0)
    model = build_model(
        channels=args.channels, lmax=args.lmax, num_interaction=args.num_interaction,
        route=args.route, product_backend=args.product_backend, dtype=dtype, device=device,
        correlation=args.contraction_order, attn_heads=args.attn_heads,
    )
    model.eval()
    model.skip_input_validation = True
    graph = make_fixed_graph(num_nodes=args.atoms, avg_degree=args.degree, dtype=dtype, device=device)
    base_pos = graph[0].detach().clone()
    rest = tuple(graph[1:])
    example_inputs = (base_pos,) + rest
    f_tgt = torch.zeros(args.atoms, 3, device=device, dtype=dtype)
    e_tgt = torch.zeros((), device=device, dtype=dtype)
    fw = 10.0
    params = [p for p in model.parameters() if p.requires_grad]

    print(f"[makefx] device={device} dtype={dtype} atoms={args.atoms} edges={args.atoms*args.degree} "
          f"channels={args.channels} lmax={args.lmax} L={args.num_interaction} route={args.route} "
          f"attn_heads={args.attn_heads} compile={do_compile}")
    if device.type == "cuda":
        print(f"[makefx] gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")

    def loss_from(energy, forces):
        return (energy - e_tgt) ** 2 + fw * ((forces - f_tgt) ** 2).mean()

    def eager_compute(pos_base):
        pos = pos_base.detach().clone().requires_grad_(True)
        e_atom = forward_energy_atom(model, pos, (pos,) + rest)
        energy = e_atom.sum()
        grad = torch.autograd.grad(energy, pos, create_graph=True)[0]
        return energy, -grad

    # ---- eager reference (numbers + param grads) ----
    for p in params:
        p.grad = None
    e_ref, f_ref = eager_compute(base_pos)
    loss_ref = loss_from(e_ref, f_ref)
    loss_ref.backward()
    grads_ref = _flat_grads(params).clone()
    f_ref = f_ref.detach().clone()
    loss_ref_v = loss_ref.detach().clone()

    # ---- build the make_fx (compiled or flattened) callable ----
    try:
        compute = trace_and_compile_force(
            model, example_inputs, training=True, do_compile=do_compile,
        )
    except Exception as ex:
        import traceback
        print(f"[makefx] TRACE/COMPILE FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        return 1

    def makefx_step(pos_base):
        for p in params:
            p.grad = None
        energy, forces = compute(pos_base, *rest)
        loss = loss_from(energy, forces)
        loss.backward()
        return loss.detach().clone(), forces.detach().clone()

    # warm up (compilation lands on the first call when do_compile)
    try:
        loss_c, forces_c = makefx_step(base_pos)
    except Exception as ex:
        import traceback
        print(f"[makefx] COMPILED STEP FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        return 1
    grads_c = _flat_grads(params).clone()

    # ---- numerics vs eager ----
    dloss = (loss_c - loss_ref_v).abs().item()
    rloss = dloss / (loss_ref_v.abs().item() + 1e-30)
    dforce = (forces_c - f_ref).abs().max().item()
    fscale = f_ref.abs().max().item()
    dgrad = (grads_c - grads_ref).abs().max().item()
    gscale = grads_ref.abs().max().item()
    rgrad = dgrad / (gscale + 1e-30)
    gnz = grads_c.abs().max().item()
    print(f"[makefx] loss eager={loss_ref_v.item():.8e} makefx={loss_c.item():.8e} rel={rloss:.3e}")
    print(f"[makefx] forces max|d|={dforce:.3e} scale={fscale:.3e} rel={dforce/(fscale+1e-30):.3e}")
    print(f"[makefx] param-grads max|d|={dgrad:.3e} scale={gscale:.3e} rel={rgrad:.3e} "
          f"(makefx grad absmax={gnz:.3e}, n={grads_ref.numel()})")
    if gnz <= 0.0:
        print("[makefx] WARNING: make_fx param grads are all ZERO -> detach strip likely severed "
              "the force-loss -> theta path. NUMERICAL-MATCH would be a false pass.")
    tol = 1e-9 if dtype == torch.float64 else 3e-3
    ok = (rloss <= tol) and (rgrad <= tol) and (gnz > 0.0)
    print(f"[makefx] NUMERICAL-MATCH {'PASS' if ok else 'FAIL'} (rel tol={tol:.1e})")

    # ---- equivariance: rotate inputs, energy invariant + forces covariant ----
    if dtype == torch.float64:
        R = random_rotation(dtype=dtype).to(device)
        pos_rot = base_pos @ R.T
        e_c0, f_c0 = compute(base_pos, *rest)
        e_cR, f_cR = compute(pos_rot, *rest)
        e_err = (e_cR - e_c0).abs().item()
        f_cov_err = (f_cR - f_c0 @ R.T).abs().max().item()
        f_sc = f_c0.abs().max().item()
        etol = 1e-8 * max(1.0, abs(e_c0.item()))
        ftol = 1e-8 * max(1.0, f_sc)
        eq_ok = (e_err <= etol) and (f_cov_err <= ftol)
        print(f"[makefx] equivariance |E(Rx)-E(x)|={e_err:.3e} |F(Rx)-F(x)R^T|_inf={f_cov_err:.3e} "
              f"{'PASS' if eq_ok else 'FAIL'}")
        ok = ok and eq_ok

    # ---- timing (only when actually compiled) ----
    if do_compile:
        def eager_step():
            for p in params:
                p.grad = None
            e_, f_ = eager_compute(base_pos)
            loss_from(e_, f_).backward()
        torch.cuda.reset_peak_memory_stats()
        t_eager = _time_section(eager_step, device=device, iters=args.iters, warmup=args.warmup, label="eager")
        mem_eager = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        t_mfx = _time_section(lambda: makefx_step(base_pos), device=device, iters=args.iters, warmup=args.warmup, label="makefx")
        mem_mfx = torch.cuda.max_memory_allocated() / 1e9
        print(f"[makefx] eager full step : {t_eager:8.3f} ms")
        print(f"[makefx] makefx step     : {t_mfx:8.3f} ms")
        print(f"[makefx] SPEEDUP         : {t_eager / max(t_mfx,1e-9):6.2f}x  (steady-state median, compile excluded)")
        print(f"[makefx] peak CUDA mem   : eager={mem_eager:.2f} GB  makefx={mem_mfx:.2f} GB  "
              f"ratio={mem_mfx / max(mem_eager, 1e-9):.2f}x")
    return 0 if ok else 1


def run_func(args) -> int:
    """Compute the training-step gradients via torch.func (functional VJP) instead
    of the autograd engine. forces = -func.grad(E, pos); param grads = func.grad of
    the loss (the mixed 2nd derivative). This uses a different mechanism than the
    autograd engine's AccumulateGrad, so it may sidestep the compiled-autograd
    None-accumulate bug. Optionally torch.compile the whole functional double-grad."""
    from torch.func import grad, grad_and_value, functional_call
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 256
    # The 75 graph breaks in the func+compile path are "skip __call__" of non-inlined
    # nn.Modules (from functional_call), not model code. Inlining them lets dynamo
    # trace the whole forward into one graph so Inductor can fuse the elementwise chains.
    for _cfg in ("inline_inbuilt_nn_modules", "allow_rnn"):
        if hasattr(dynamo.config, _cfg) and _cfg == "inline_inbuilt_nn_modules":
            setattr(dynamo.config, _cfg, True)
    device = torch.device(args.device)
    if device.type != "cuda":
        print("[func] requires CUDA"); return 1
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(0)
    model = build_model(
        channels=args.channels, lmax=args.lmax, num_interaction=args.num_interaction,
        route=args.route, product_backend=args.product_backend, dtype=dtype, device=device,
        correlation=args.contraction_order, attn_heads=args.attn_heads,
    )
    model.eval()
    model.skip_input_validation = True
    graph = make_fixed_graph(num_nodes=args.atoms, avg_degree=args.degree, dtype=dtype, device=device)
    base_pos = graph[0].detach().clone()
    rest = tuple(graph[1:])
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    f_tgt = torch.zeros(args.atoms, 3, device=device, dtype=dtype)
    e_tgt = torch.zeros((), device=device, dtype=dtype)
    fw = 10.0
    pnames = list(params.keys())

    def energy_sum(p, pos):
        out = functional_call(model, (p, buffers), (pos,) + rest)
        if isinstance(out, tuple):
            out = out[0]
        return out.sum()

    def loss_fn(p, pos):
        neg_forces, e = grad_and_value(energy_sum, argnums=1)(p, pos)
        forces = -neg_forces
        return (e - e_tgt) ** 2 + fw * ((forces - f_tgt) ** 2).mean()

    grad_params = grad(loss_fn, argnums=0)

    print(f"[func] device={device} dtype={dtype} atoms={args.atoms} route={args.route} "
          f"channels={args.channels} lmax={args.lmax} compile_func={args.compile_func}")

    # ---- autograd reference (the shipped trainer path) ----
    for v in params.values():
        pass
    ref_model = model
    for p in ref_model.parameters():
        p.grad = None
    pos_r = base_pos.detach().clone().requires_grad_(True)
    e_atom = ref_model(pos_r, *rest)
    if isinstance(e_atom, tuple):
        e_atom = e_atom[0]
    energy_r = e_atom.sum()
    gpos = torch.autograd.grad(energy_r, pos_r, create_graph=True)[0]
    loss_r = (energy_r - e_tgt) ** 2 + fw * ((-gpos - f_tgt) ** 2).mean()
    loss_r.backward()
    ref = {k: v.grad.detach().clone() for k, v in ref_model.named_parameters()}

    # ---- eager torch.func ----
    try:
        gf = grad_params(params, base_pos)
    except Exception as ex:
        import traceback
        print(f"[func] EAGER torch.func FAILED: {type(ex).__name__}: {ex}")
        traceback.print_exc()
        return 1
    torch.cuda.synchronize()
    df = max((gf[k] - ref[k]).abs().max().item() for k in pnames)
    scale = max(ref[k].abs().max().item() for k in pnames)
    print(f"[func] eager torch.func vs autograd: max|d|={df:.3e} scale={scale:.3e} rel={df/(scale+1e-30):.3e}")
    ok_eager = df <= (1e-9 if dtype == torch.float64 else 3e-4) * max(1.0, scale)
    print(f"[func] eager NUMERICAL-MATCH {'PASS' if ok_eager else 'FAIL'}")

    ok = ok_eager
    fn_to_time = grad_params
    if args.compile_func:
        dynamo.reset()
        compiled = torch.compile(grad_params, dynamic=False)
        try:
            for _ in range(3):
                compiled(params, base_pos)
            torch.cuda.synchronize()
        except Exception as ex:
            import traceback
            print(f"[func] COMPILED torch.func FAILED: {type(ex).__name__}: {ex}")
            traceback.print_exc()
            print("[func] (eager torch.func still usable; compile is the extra step)")
            return 0 if ok_eager else 1
        gc = compiled(params, base_pos)
        torch.cuda.synchronize()
        dc = max((gc[k] - ref[k]).abs().max().item() for k in pnames)
        print(f"[func] compiled torch.func vs autograd: max|d|={dc:.3e} rel={dc/(scale+1e-30):.3e}")
        ok = ok and (dc <= (1e-9 if dtype == torch.float64 else 3e-3) * max(1.0, scale))
        print(f"[func] compiled NUMERICAL-MATCH {'PASS' if dc <= 3e-3*max(1.0,scale) else 'FAIL'}")
        fn_to_time = compiled

    # ---- timing vs autograd full step ----
    def autograd_step():
        for p in ref_model.parameters():
            p.grad = None
        pos = base_pos.detach().clone().requires_grad_(True)
        ea = ref_model(pos, *rest)
        if isinstance(ea, tuple):
            ea = ea[0]
        en = ea.sum()
        gp = torch.autograd.grad(en, pos, create_graph=True)[0]
        ll = (en - e_tgt) ** 2 + fw * ((-gp - f_tgt) ** 2).mean()
        ll.backward()

    def func_step():
        fn_to_time(params, base_pos)

    t_ag = _time_section(autograd_step, device=device, iters=args.iters, warmup=args.warmup)
    t_fn = _time_section(func_step, device=device, iters=args.iters, warmup=args.warmup)
    print(f"[func] autograd full step : {t_ag:8.3f} ms")
    print(f"[func] torch.func step    : {t_fn:8.3f} ms")
    print(f"[func] SPEEDUP            : {t_ag / max(t_fn,1e-9):6.2f}x")
    return 0 if ok else 1


def run_paths(args) -> int:
    """Print the ICTD-U symmetric-contraction path counts (each U's last axis = the
    number of independent coupling paths/weights for that degree+output_l), so the
    correlation->2 path reduction is quantified. No model build / no CUDA needed."""
    from molecular_force_field.models.ictd_irreps import ictd_u_matrix_so3
    lmax = args.lmax
    print(f"[paths] lmax={lmax}  ICTD-U symmetric-contraction path counts (U last-axis = #paths)")
    per_deg: dict = {}
    for L in range(lmax + 1):
        cells = []
        for d in range(1, 4):
            U = ictd_u_matrix_so3(lmax=lmax, output_l=L, correlation=d, dtype=torch.float64)
            n = int(U.shape[-1])
            per_deg.setdefault(d, {})[L] = n
            cells.append(f"deg{d}={n:>6d} U{tuple(U.shape)}")
        print(f"[paths] output_l={L}: " + "   ".join(cells))
    sum_d = {d: sum(per_deg[d].values()) for d in per_deg}
    print(f"[paths] per-degree total paths (summed over output_l 0..{lmax}): "
          + "  ".join(f"deg{d}={sum_d[d]}" for d in sorted(sum_d)))
    print(f"[paths] cubic vs quadratic term: deg3={sum_d[3]} vs deg2={sum_d[2]} "
          f"= {sum_d[3]/max(sum_d[2],1):.2f}x more paths in the degree-3 term")
    tot2 = sum_d[1] + sum_d[2]
    tot3 = sum_d[1] + sum_d[2] + sum_d[3]
    print(f"[paths] whole-model paths  corr=2: {tot2}   corr=3: {tot3}   "
          f"corr3/corr2 = {tot3/max(tot2,1):.2f}x  (operands: deg2=4 OK for fused_tp, deg3=5 NOT)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="run numerical + equivariance check")
    p.add_argument("--bench", action="store_true", help="run timing / profiling")
    p.add_argument("--cudagraph", action="store_true", help="capture full step as CUDA graph; compare + time")
    p.add_argument("--cudagraph-train", dest="cudagraph_train", action="store_true", help="fixed-shape CUDA-graph TRAINING benchmark (N optimizer steps via replay)")
    p.add_argument("--train-steps", type=int, default=30, help="number of optimizer steps for --cudagraph-train")
    p.add_argument("--compile", dest="do_compile", action="store_true", help="torch.compile forward (+optional compiled-autograd); compare + time")
    p.add_argument("--fullgraph", action="store_true", help="torch.compile fullgraph=True")
    p.add_argument("--compiled-autograd", action="store_true", help="compile the backward via compiled-autograd")
    p.add_argument("--compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--tf32", action="store_true", help="allow TF32 matmul (changes numerics)")
    p.add_argument("--makefx", action="store_true", help="make_fx-flatten the forward+force-autograd then torch.compile it (second-order via flat backward); compare+time")
    p.add_argument("--makefx-no-compile", dest="makefx_no_compile", action="store_true", help="with --makefx: skip Inductor, validate only the flatten+strip+rebuild stages vs eager (CPU-friendly)")
    p.add_argument("--func", action="store_true", help="compute train-step grads via torch.func (functional VJP); compare+time")
    p.add_argument("--paths", action="store_true", help="print ICTD-U symmetric-contraction path counts per degree (no model build / no CUDA)")
    p.add_argument("--compile-func", action="store_true", help="torch.compile the torch.func double-grad")
    p.add_argument("--freeze-fusion-bias", action="store_true", help="freeze fusion_readouts biases (unblocks compiled-autograd on fusion)")
    p.add_argument("--stress", action="store_true", help="also train stress (dE/dstrain): adds strain deformation + stress loss")
    p.add_argument("--stress-weight", type=float, default=1.0, help="stress loss weight (with --stress)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--atoms", type=int, default=256)
    p.add_argument("--degree", type=int, default=24)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--lmax", type=int, default=2)
    p.add_argument("--num-interaction", type=int, default=2)
    p.add_argument("--contraction-order", dest="contraction_order", type=int, default=2,
                   help="symmetric-contraction correlation/body-order (degree). 3=production default; "
                        "2=cheaper (back in fused_tp's bilinear zone, far fewer paths)")
    p.add_argument("--route", default="baseline", choices=["baseline", "fusion"])
    p.add_argument("--attn-heads", dest="attn_heads", type=int, default=0,
                   help="interaction neighbor-attention heads (0=off; trainer flag "
                        "--ictd-fix-interaction-attn-heads)")
    p.add_argument("--product-backend", default="ictd-pure-u")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--profile-iters", type=int, default=10)
    p.add_argument("--trace", default=None)
    p.add_argument("--save-ref", default=None)
    p.add_argument("--compare-ref", default=None)
    args = p.parse_args()

    if not (args.check or args.bench or args.cudagraph or args.cudagraph_train or args.do_compile or args.func or args.paths or args.makefx):
        args.check = True

    rc = 0
    if args.paths:
        rc |= run_paths(args)
    if args.check:
        rc |= run_check(args)
    if args.bench:
        rc |= run_bench(args)
    if args.cudagraph:
        rc |= run_cudagraph(args)
    if args.cudagraph_train:
        rc |= run_cudagraph_train(args)
    if args.do_compile:
        rc |= run_compile(args)
    if args.func:
        rc |= run_func(args)
    if args.makefx:
        rc |= run_makefx(args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
