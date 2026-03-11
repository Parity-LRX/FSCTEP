"""
Compare ICTD operator optimization (HarmonicFullyConnectedTensorProduct forward) before/after.

This test builds two identical PureCartesianICTDTransformerLayer instances:
  - "old": uses the baseline implementation from `molecular_force_field.models.ictd_irreps`
  - "new": swaps TP modules to an optimized implementation defined in THIS script

It reports:
  - parameter counts
  - forward+backward speed
  - output numerics consistency (old vs new)
  - simple O(3) invariance check for total energy (sum over atoms)

Run:
  python -m molecular_force_field.test_ictd_tp_opt_compare
  python -m molecular_force_field.test_ictd_tp_opt_compare --device cuda --dtype float32
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import sys
import torch
import torch.nn as nn
import torch.autograd

from molecular_force_field.models.ictd_irreps import HarmonicFullyConnectedTensorProduct
from molecular_force_field.models.pure_cartesian_ictd_layers import PureCartesianICTDTransformerLayer


def _param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def make_dummy_graph(device: torch.device, dtype: torch.dtype, num_nodes: int = 128, avg_degree: int = 24, seed: int = 42):
    torch.manual_seed(seed)
    pos = torch.randn(num_nodes, 3, device=device, dtype=dtype) * 2.0
    A = torch.randint(1, 6, (num_nodes,), device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    num_edges = num_nodes * avg_degree
    edge_dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_src = torch.randint(0, num_nodes, (num_edges,), device=device)
    edge_shifts = torch.zeros(num_edges, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(num_nodes, -1, -1)
    return pos, A, batch, edge_src, edge_dst, edge_shifts, cell


def _loss_for_timing(module: nn.Module, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, *, second_deriv: bool):
    """
    Build a scalar loss.

    - second_deriv=False: loss = sum(E_per_atom)
    - second_deriv=True : loss = ||forces||^2 where forces = -dE/dpos, then backward() triggers 2nd derivatives
    """
    out = module(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    E = out.sum()
    if not second_deriv:
        return E
    # forces = -dE/dpos
    forces = -torch.autograd.grad(E, pos, create_graph=True, retain_graph=True)[0]
    return (forces ** 2).sum()

def _cudagraph_step_begin_if_available():
    """
    When torch.compile uses CUDA graphs, outputs are often reused across iterations.
    Marking step boundaries prevents "accessing tensor output ... overwritten" errors.
    """
    try:
        # PyTorch 2.4+ typically exposes this
        torch.compiler.cudagraph_mark_step_begin()  # type: ignore[attr-defined]
    except Exception:
        pass


def time_forward_backward(
    module: nn.Module,
    pos,
    A,
    batch,
    edge_src,
    edge_dst,
    edge_shifts,
    cell,
    *,
    warmup: int = 3,
    repeat: int = 20,
    second_deriv: bool = False,
):
    module.train()
    for _ in range(warmup):
        if next(module.parameters()).is_cuda:
            _cudagraph_step_begin_if_available()
        module.zero_grad(set_to_none=True)
        if pos.grad is not None:
            pos.grad = None
        loss = _loss_for_timing(module, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, second_deriv=second_deriv)
        loss.backward()
    if next(module.parameters()).is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        if next(module.parameters()).is_cuda:
            _cudagraph_step_begin_if_available()
        module.zero_grad(set_to_none=True)
        if pos.grad is not None:
            pos.grad = None
        loss = _loss_for_timing(module, pos, A, batch, edge_src, edge_dst, edge_shifts, cell, second_deriv=second_deriv)
        loss.backward()
    if next(module.parameters()).is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat


def time_inference(
    module: nn.Module,
    pos,
    A,
    batch,
    edge_src,
    edge_dst,
    edge_shifts,
    cell,
    *,
    warmup: int = 10,
    repeat: int = 100,
):
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            if next(module.parameters()).is_cuda:
                _cudagraph_step_begin_if_available()
            _ = module(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    if next(module.parameters()).is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeat):
            if next(module.parameters()).is_cuda:
                _cudagraph_step_begin_if_available()
            _ = module(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    if next(module.parameters()).is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat


def random_orthogonal(device: torch.device, dtype: torch.dtype, *, reflect: bool = False, seed: int = 0) -> torch.Tensor:
    """
    Create a deterministic-ish random O(3) matrix.

    We can do QR on GPU just fine; the only caveat is the RNG generator must match the tensor device.
    This helper does:
      - CUDA: generate M on CUDA with a CUDA generator, then QR on CUDA
      - CPU : generate M on CPU, then QR on CPU
    If CUDA generator is not supported by the installed PyTorch build, we fall back to CPU.
    """
    if device.type == "cuda":
        try:
            g = torch.Generator(device="cuda")
            g.manual_seed(seed)
            M = torch.randn(3, 3, generator=g, device=device, dtype=torch.float64)
            Q, _R = torch.linalg.qr(M)
        except Exception:
            # Conservative fallback (works everywhere)
            g = torch.Generator(device="cpu")
            g.manual_seed(seed)
            M_cpu = torch.randn(3, 3, generator=g, device="cpu", dtype=torch.float64)
            Q, _R = torch.linalg.qr(M_cpu)
    else:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        M_cpu = torch.randn(3, 3, generator=g, device="cpu", dtype=torch.float64)
        Q, _R = torch.linalg.qr(M_cpu)
    # Fix sign to make Q a proper rotation first
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    if reflect:
        # Apply a reflection (det = -1)
        Q[:, 0] = -Q[:, 0]
    return Q.to(device=device, dtype=dtype)


class OptimizedHarmonicFullyConnectedTensorProduct(HarmonicFullyConnectedTensorProduct):
    """
    Optimized forward (used only in this test):
      - avoid materializing t_mn (..., i, j, m1, m2) by two-step contraction with U reshaped to (m1,m2,K)
      - batch channel mixing across paths inside each (l1,l2) group (GPU-friendly)

    This targets the common usage in `pure_cartesian_ictd_layers.py`:
      internal_weights=True and weights is per-path scalar gates (..., num_paths).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pidx_cache_by_device: dict[str, list[torch.Tensor]] = {}

    def _get_group_pidx_list(self, device: torch.device) -> list[torch.Tensor]:
        key = str(device)
        cached = self._pidx_cache_by_device.get(key)
        if cached is not None:
            return cached
        pidx_list: list[torch.Tensor] = []
        for g in self._groups:
            p_indices = g["p_indices"]  # type: ignore[assignment]
            pidx_list.append(torch.tensor(list(p_indices), dtype=torch.long, device=device))
        self._pidx_cache_by_device[key] = pidx_list
        return pidx_list

    def forward(self, x1, x2, weights=None):  # type: ignore[override]
        sample = next(iter(x1.values()))
        batch_shape = sample.shape[:-2]
        device = sample.device
        dtype = sample.dtype
        compute_dtype = self.internal_compute_dtype

        if not self.internal_weights:
            return super().forward(x1, x2, weights)
        if weights is not None and weights.shape[-1] not in (self.num_paths, self.weight_numel):
            return super().forward(x1, x2, weights)
        # We only optimize the common "gates" case: (..., num_paths)
        if weights is not None and weights.shape[-1] == self.weight_numel:
            return super().forward(x1, x2, weights)

        if weights is not None and (weights.device != device or weights.dtype != dtype):
            weights = weights.to(device=device, dtype=dtype)

        out = {l: torch.zeros(*batch_shape, self.mul_out, 2 * l + 1, device=device, dtype=dtype) for l in range(self.lmax + 1)}
        proj_list = self._get_proj_group_list(device=device, dtype=dtype)
        pidx_list = self._get_group_pidx_list(device=device)
        w_param = self.weight  # (P, o, i, j)

        for g_idx, g in enumerate(self._groups):
            l1 = int(g["l1"])
            l2 = int(g["l2"])
            segments = g["segments"]
            k_total = int(g["k_total"])

            a = x1.get(l1)
            b = x2.get(l2)
            if a is None or b is None:
                continue

            m1 = 2 * l1 + 1
            m2 = 2 * l2 + 1
            U = proj_list[g_idx]  # (m1*m2, K_total)
            a_comp = a.to(dtype=compute_dtype) if a.dtype != compute_dtype else a
            b_comp = b.to(dtype=compute_dtype) if b.dtype != compute_dtype else b

            # Two-step contraction (avoid t_mn) implemented with matmul-friendly reshapes:
            # tmp[..., j, m1, K] = sum_{m2} b[..., j, m2] * U[m1, m2, K]
            # y  [..., i, j, K]  = sum_{m1} a[..., i, m1] * tmp[..., j, m1, K]
            U3 = U.view(m1, m2, k_total)
            # (m2, m1*K)
            U_flat = U3.permute(1, 0, 2).reshape(m2, m1 * k_total)
            # (..., j, m2) @ (m2, m1*K) -> (..., j, m1*K) -> (..., j, m1, K)
            tmp_flat = torch.matmul(b_comp, U_flat)  # (..., j, m1*K)
            tmp = tmp_flat.view(*batch_shape, self.mul_in2, m1, k_total)
            # (..., i, m1) @ (..., m1, j*K) -> (..., i, j*K) -> (..., i, j, K)
            tmp_m1_jk = tmp.permute(*range(len(batch_shape)), 2, 1, 3).reshape(*batch_shape, m1, self.mul_in2 * k_total)
            y_flat = torch.matmul(a_comp, tmp_m1_jk)  # (..., i, j*K)
            y = y_flat.view(*batch_shape, self.mul_in1, self.mul_in2, k_total)

            # Batch channel mixing across paths in this group
            pidx = pidx_list[g_idx]  # (P_g,)
            W_stack = w_param.index_select(0, pidx)  # (P_g, o, i, j)
            W_stack_comp = W_stack.to(dtype=compute_dtype) if W_stack.dtype != compute_dtype else W_stack
            if self.mul_in2 == 1:
                # common in our models: mul_in2=1 (edge-only). Reduce a dimension for speed.
                y2 = y.squeeze(-2)  # (..., i, K)
                W2 = W_stack_comp.squeeze(-1)  # (P_g, o, i)
                out_group = torch.einsum("...ik,poi->...pok", y2, W2)  # (..., P_g, o, K)
            else:
                out_group = torch.einsum("...ijk,poij->...pok", y, W_stack_comp)  # (..., P_g, o, K)
            out_group = out_group.to(dtype=dtype) if out_group.dtype != dtype else out_group
            if weights is not None:
                gate_g = weights[..., pidx]  # (..., P_g)
                out_group = out_group * gate_g[..., :, None, None]

            for seg_idx, (_p_idx, l3, s, e) in enumerate(segments):
                out[int(l3)] = out[int(l3)] + out_group[..., seg_idx, :, int(s): int(e)]

        return out


@dataclass
class CompareResult:
    params: int
    time_s: float
    energy: float


def _swap_tp_modules_to_optimized(layer: PureCartesianICTDTransformerLayer) -> PureCartesianICTDTransformerLayer:
    # conv1 TP
    tp = layer.e3_conv_emb.tp2
    assert isinstance(tp, HarmonicFullyConnectedTensorProduct)
    new_tp = OptimizedHarmonicFullyConnectedTensorProduct(
        mul_in1=tp.mul_in1,
        mul_in2=tp.mul_in2,
        mul_out=tp.mul_out,
        lmax=tp.lmax,
        internal_weights=True,
        allowed_paths=None,
        path_policy="full",
        max_rank_other=None,
        internal_compute_dtype=tp.internal_compute_dtype,
    ).to(device=next(tp.parameters()).device, dtype=next(tp.parameters()).dtype)
    new_tp.weight.data.copy_(tp.weight.data)
    layer.e3_conv_emb.tp2 = new_tp

    # conv2..convN TPs
    new_layers = nn.ModuleList()
    for tp2 in layer.tp2_layers:
        assert isinstance(tp2, HarmonicFullyConnectedTensorProduct)
        new_tp2 = OptimizedHarmonicFullyConnectedTensorProduct(
            mul_in1=tp2.mul_in1,
            mul_in2=tp2.mul_in2,
            mul_out=tp2.mul_out,
            lmax=tp2.lmax,
            internal_weights=True,
            allowed_paths=None,
            path_policy="full",
            max_rank_other=None,
            internal_compute_dtype=tp2.internal_compute_dtype,
        ).to(device=next(tp2.parameters()).device, dtype=next(tp2.parameters()).dtype)
        new_tp2.weight.data.copy_(tp2.weight.data)
        new_layers.append(new_tp2)
    layer.tp2_layers = new_layers
    return layer


def _maybe_compile(module: nn.Module, *, which: str, compile_target: str, mode: str, fullgraph: bool, dynamic: bool):
    """
    Optionally wrap module in torch.compile.
    Returns (module_or_compiled, compiled_ok: bool, err_msg: str|None).
    """
    if compile_target not in (which, "both"):
        return module, False, None
    if not hasattr(torch, "compile"):
        return module, False, "torch.compile not available in this PyTorch"
    try:
        compiled = torch.compile(module, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
        return compiled, True, None
    except Exception as e:
        # Keep test runnable even if compile fails for higher-order grads / unsupported ops
        return module, False, f"torch.compile failed: {e}"

def _precache_ictd_cuda_kernels(
    module: nn.Module,
    pos: torch.Tensor,
    A: torch.Tensor,
    batch: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_shifts: torch.Tensor,
    cell: torch.Tensor,
):
    """
    Warm up caches that otherwise introduce CPU work / graph breaks during torch.compile tracing:
      - direction_harmonics_fast: builds _dir_proj_cpu_f64(l) on CPU then .to(cuda) once per (device,dtype,l)
      - HarmonicFullyConnectedTensorProduct: builds CG/projection caches on first use then keeps on device

    Running one eager forward is enough to populate these caches for the current device/dtype/lmax.
    """
    module.eval()
    with torch.no_grad():
        if next(module.parameters()).is_cuda:
            _cudagraph_step_begin_if_available()
        _ = module(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)


def _time_with_fallback(
    label: str,
    compiled_module: nn.Module,
    eager_module: nn.Module,
    pos,
    A,
    batch,
    edge_src,
    edge_dst,
    edge_shifts,
    cell,
    *,
    warmup: int,
    second_deriv: bool,
) -> tuple[float, str | None]:
    """
    Time compiled module; if it fails at runtime (inductor/codegen/etc.), fall back to eager.
    Returns (time_s, runtime_compile_error_msg_or_None).
    """
    try:
        t = time_forward_backward(
            compiled_module,
            pos,
            A,
            batch,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
            warmup=warmup,
            second_deriv=second_deriv,
        )
        return t, None
    except Exception as e:
        # Fallback to eager
        t = time_forward_backward(
            eager_module,
            pos,
            A,
            batch,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
            warmup=3,
            second_deriv=second_deriv,
        )
        return t, f"{label} runtime failed, fell back to eager: {e}"


def _time_inference_with_fallback(
    label: str,
    compiled_module: nn.Module,
    eager_module: nn.Module,
    pos,
    A,
    batch,
    edge_src,
    edge_dst,
    edge_shifts,
    cell,
    *,
    warmup: int,
    repeat: int,
) -> tuple[float, str | None]:
    try:
        t = time_inference(
            compiled_module,
            pos,
            A,
            batch,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
            warmup=warmup,
            repeat=repeat,
        )
        return t, None
    except Exception as e:
        t = time_inference(
            eager_module,
            pos,
            A,
            batch,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
            warmup=max(3, warmup // 2),
            repeat=repeat,
        )
        return t, f"{label} runtime failed, fell back to eager: {e}"


def run_compare(
    device: torch.device,
    dtype: torch.dtype,
    *,
    compile_target: str,
    compile_mode: str,
    compile_fullgraph: bool,
    compile_dynamic: bool,
    compile_on_cpu: bool,
    precache: bool,
    second_deriv: bool,
    inference: bool,
    warmup: int,
    repeat: int,
):
    cfg = dict(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=32,
        hidden_dim_sh=32,
        hidden_dim=64,
        channel_in2=32,
        embedding_dim=16,
        max_atomvalue=10,
        output_size=8,
        num_interaction=3,
        lmax=2,
        ictd_tp_path_policy="full",
        ictd_tp_max_rank_other=None,
        internal_compute_dtype=torch.float64,  # keep default behavior
    )

    # Build OLD (baseline)
    layer_old = PureCartesianICTDTransformerLayer(**cfg).to(device=device, dtype=dtype)
    # Build NEW (same weights) then swap TP modules to optimized forward
    layer_new = PureCartesianICTDTransformerLayer(**cfg).to(device=device, dtype=dtype)
    layer_new.load_state_dict(layer_old.state_dict())
    layer_new = _swap_tp_modules_to_optimized(layer_new)

    # Keep eager refs for fallback
    layer_old_eager = layer_old
    layer_new_eager = layer_new

    # Optional torch.compile (script-only)
    if device.type == "cpu" and compile_target != "none" and not compile_on_cpu:
        layer_old, old_compiled, old_compile_err = layer_old, False, "skipped (cpu)"
        layer_new, new_compiled, new_compile_err = layer_new, False, "skipped (cpu)"
    else:
        layer_old, old_compiled, old_compile_err = _maybe_compile(
            layer_old,
            which="old",
            compile_target=compile_target,
            mode=compile_mode,
            fullgraph=compile_fullgraph,
            dynamic=compile_dynamic,
        )
        layer_new, new_compiled, new_compile_err = _maybe_compile(
            layer_new,
            which="new",
            compile_target=compile_target,
            mode=compile_mode,
            fullgraph=compile_fullgraph,
            dynamic=compile_dynamic,
        )

    # Params (should match)
    p_new = _param_count(layer_new)
    p_old = _param_count(layer_old)

    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = make_dummy_graph(device, dtype)
    if not inference:
        pos.requires_grad_(True)

    # Optional: precache CUDA-side ICTD projections before torch.compile to avoid CPU work inside tracing.
    if precache and device.type == "cuda" and compile_target != "none":
        _precache_ictd_cuda_kernels(layer_old_eager, pos.detach(), A, batch, edge_src, edge_dst, edge_shifts, cell)
        _precache_ictd_cuda_kernels(layer_new_eager, pos.detach(), A, batch, edge_src, edge_dst, edge_shifts, cell)

    # Speed
    if inference:
        t_new, new_runtime_err = _time_inference_with_fallback(
            "new",
            layer_new,
            layer_new_eager,
            pos,
            A,
            batch,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
            warmup=warmup,
            repeat=repeat,
        )
        t_old, old_runtime_err = _time_inference_with_fallback(
            "old",
            layer_old,
            layer_old_eager,
            pos,
            A,
            batch,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
            warmup=warmup,
            repeat=repeat,
        )
    else:
        # Increase warmup a bit when compiling to amortize graph capture/inductor warmups.
        warmup_train = max(warmup, 6) if (old_compiled or new_compiled) else max(warmup, 3)
        t_new, new_runtime_err = _time_with_fallback(
            "new",
            layer_new,
            layer_new_eager,
            pos,
            A,
            batch,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
            warmup=warmup_train,
            second_deriv=second_deriv,
        )
        t_old, old_runtime_err = _time_with_fallback(
            "old",
            layer_old,
            layer_old_eager,
            pos,
            A,
            batch,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
            warmup=warmup_train,
            second_deriv=second_deriv,
        )

    # Numerics: same input, compare energy sums
    with torch.no_grad():
        # IMPORTANT: If torch.compile uses CUDA graphs, compiled outputs can be overwritten by subsequent runs.
        # For correctness checks, use eager modules (no cudagraph reuse) and/or clone immediately.
        out_new = layer_new_eager(pos, A, batch, edge_src, edge_dst, edge_shifts, cell).detach().clone()
        out_old = layer_old_eager(pos, A, batch, edge_src, edge_dst, edge_shifts, cell).detach().clone()
        e_new = out_new.sum().item()
        e_old = out_old.sum().item()
        max_abs = float(torch.max(torch.abs(out_new - out_old)).item())

    # Equivariance: total energy invariance under random O(3) (rotation + reflection)
    R = random_orthogonal(device, dtype, reflect=False, seed=123)
    Rr = random_orthogonal(device, dtype, reflect=True, seed=456)

    def transform(pos0, cell0, RR):
        return pos0 @ RR.T, cell0 @ RR.T

    pos_R, cell_R = transform(pos, cell, R)
    pos_Rr, cell_Rr = transform(pos, cell, Rr)

    with torch.no_grad():
        # Use eager version for the same cudagraph output-reuse reason as above.
        E = layer_new_eager(pos, A, batch, edge_src, edge_dst, edge_shifts, cell).sum()
        E_R = layer_new_eager(pos_R, A, batch, edge_src, edge_dst, edge_shifts, cell_R).sum()
        E_Rr = layer_new_eager(pos_Rr, A, batch, edge_src, edge_dst, edge_shifts, cell_Rr).sum()
        rel_R = (E - E_R).abs() / (E.abs() + E_R.abs() + 1e-12)
        rel_Rr = (E - E_Rr).abs() / (E.abs() + E_Rr.abs() + 1e-12)

    return {
        "params_new": p_new,
        "params_old": p_old,
        "time_new_s": t_new,
        "time_old_s": t_old,
        "speedup_old_over_new": t_old / max(t_new, 1e-12),
        "energy_new": e_new,
        "energy_old": e_old,
        "max_abs_diff": max_abs,
        "rel_invariance_rot": float(rel_R.item()),
        "rel_invariance_reflect": float(rel_Rr.item()),
        "compiled_old": old_compiled,
        "compiled_new": new_compiled,
        "compile_err_old": old_compile_err,
        "compile_err_new": new_compile_err,
        "runtime_err_old": old_runtime_err,
        "runtime_err_new": new_runtime_err,
    }


def main():
    # Ensure users can always see *some* output (helps debug remote runs / buffering).
    try:
        sys.stdout.reconfigure(line_buffering=True)  # py>=3.7
    except Exception:
        pass
    print(f"[test_ictd_tp_opt_compare] running: {__file__}", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", choices=["float32", "float64", "float", "double"], default="float32")
    parser.add_argument("--compile", choices=["none", "old", "new", "both"], default="none",
                        help="Apply torch.compile to old/new/both (default: none).")
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="default",
                        help="torch.compile mode (default: default).")
    parser.add_argument("--compile-fullgraph", action="store_true",
                        help="Pass fullgraph=True to torch.compile (may fail more often).")
    parser.add_argument("--compile-dynamic", action="store_true",
                        help="Pass dynamic=True to torch.compile.")
    parser.add_argument("--compile-on-cpu", action="store_true",
                        help="Allow torch.compile on CPU (default: off; CPU inductor may fail depending on toolchain).")
    parser.add_argument("--precache", action="store_true",
                        help="Pre-run one eager forward on CUDA before compiling to warm caches (recommended).")
    parser.add_argument("--second-deriv", action="store_true",
                        help="Time a second-derivative loss: forces=-dE/dpos, loss=||forces||^2 (tests double backward).")
    parser.add_argument("--inference", action="store_true",
                        help="Time inference only (forward under torch.no_grad).")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations for timing (default: 10).")
    parser.add_argument("--repeat", type=int, default=100,
                        help="Repeat iterations for inference timing (default: 100).")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.dtype in ("float32", "float"):
        dtype = torch.float32
    else:
        dtype = torch.float64

    out = run_compare(
        device,
        dtype,
        compile_target=args.compile,
        compile_mode=args.compile_mode,
        compile_fullgraph=args.compile_fullgraph,
        compile_dynamic=args.compile_dynamic,
        compile_on_cpu=args.compile_on_cpu,
        precache=args.precache,
        second_deriv=args.second_deriv,
        inference=args.inference,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    print(f"Device={device}, dtype={dtype}", flush=True)
    if args.compile != "none":
        print(
            f"Compile target: {args.compile}, mode={args.compile_mode}, fullgraph={args.compile_fullgraph}, dynamic={args.compile_dynamic}",
            flush=True,
        )
        if args.precache and device.type == "cuda":
            print("Precache: ON (eager warmup before compile)", flush=True)
        if out.get("compile_err_old"):
            print(f"  old compile: FAILED ({out['compile_err_old']})", flush=True)
        else:
            print(f"  old compile: {'ON' if out.get('compiled_old') else 'OFF'}", flush=True)
        if out.get("compile_err_new"):
            print(f"  new compile: FAILED ({out['compile_err_new']})", flush=True)
        else:
            print(f"  new compile: {'ON' if out.get('compiled_new') else 'OFF'}", flush=True)
        if out.get("runtime_err_old"):
            print(f"  old runtime: {out['runtime_err_old']}", flush=True)
        if out.get("runtime_err_new"):
            print(f"  new runtime: {out['runtime_err_new']}", flush=True)
    if args.second_deriv:
        print("Timing uses second-derivative loss (force-squared).", flush=True)
    if args.inference:
        print("Timing uses inference-only (forward under torch.no_grad).", flush=True)
    print(
        f"Params old/new: {out['params_old']:,} / {out['params_new']:,}  (ratio new/old={out['params_new']/max(out['params_old'],1):.6f})",
        flush=True,
    )
    print(
        f"Time old/new: {out['time_old_s']*1000:.2f} ms / {out['time_new_s']*1000:.2f} ms  (speedup={out['speedup_old_over_new']:.2f}x)",
        flush=True,
    )
    print(f"Energy old/new (sum): {out['energy_old']:.6f} / {out['energy_new']:.6f}", flush=True)
    print(f"Max abs diff (per-atom output): {out['max_abs_diff']:.3e}", flush=True)
    print(f"O(3) invariance rel err (rotation): {out['rel_invariance_rot']:.3e}", flush=True)
    print(f"O(3) invariance rel err (reflection): {out['rel_invariance_reflect']:.3e}", flush=True)


if __name__ == "__main__":
    main()

