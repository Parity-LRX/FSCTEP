"""
Triton-fused sparse tensor product for O(3) pure-Cartesian GNN.

Fuses the 6 sparse paths of PureCartesianTensorProductO3Sparse for the common
Lmax=2 / k_policy="k0" / assume_pseudo_zero=True / allow_epsilon=False case.

Paths (all k_delta=0, no epsilon):
  (0,0)->0  scalar x scalar -> scalar
  (0,1)->1  scalar x vector -> vector
  (0,2)->2  scalar x matrix -> matrix
  (1,0)->1  vector x scalar -> vector
  (1,1)->2  vector x vector -> matrix
  (2,0)->2  matrix x scalar -> matrix

Feature layout (full O(3) graded, C channels, Lmax=2):
  [s0_L0(C), s0_L1(3C), s0_L2(9C), s1_L0(C), s1_L1(3C), s1_L2(9C)]
  total_dim = C * 26
  s=0 part = first C*13 elements (this kernel only touches s=0)

Weight layout: [E, 6*Cout*C1*C2] (per-sample, 6 paths concatenated)

Strategy:
  Forward: Triton kernel on CUDA, processing several output channels per launch
  Backward:
    - first-order: explicit PyTorch formulas
    - higher-order: recompute composite forward and use autograd.grad
"""

from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


_S0_PATHS = (
    (0, 0, 0, 1.0),
    (0, 1, 1, 1.0),
    (0, 2, 2, 1.0),
    (1, 0, 1, 1.0),
    (1, 1, 2, 1.0),
    (2, 0, 2, 1.0),
)

_COMPILED_INFERENCE_FN: Callable | None = None
_COMPILED_INFERENCE_DISABLED = False


def _rank_shape(L: int):
    return tuple([3] * L)


def _split_s0(x: torch.Tensor, C: int, Lmax: int = 2):
    """Split the s=0 prefix into per-rank blocks."""
    batch_shape = x.shape[:-1]
    blocks = {}
    idx = 0
    for L in range(Lmax + 1):
        d = C * (3 ** L)
        if L == 0:
            blocks[L] = x[..., idx : idx + d].reshape(*batch_shape, C)
        else:
            blocks[L] = x[..., idx : idx + d].reshape(*batch_shape, C, *_rank_shape(L))
        idx += d
    return blocks


def _merge_s0(blocks, C: int, Lmax: int = 2):
    """Merge per-rank blocks back into a flat s=0 tensor."""
    parts = []
    batch_shape = blocks[0].shape[:-1]
    for L in range(Lmax + 1):
        t = blocks[L]
        parts.append(t.reshape(*batch_shape, C * (3 ** L)))
    return torch.cat(parts, dim=-1)


def _split_s0_rank_blocks(x: torch.Tensor, C: int):
    """Split the flat s=0 prefix into scalar/vector/matrix blocks for Lmax=2."""
    s = x[:, :C]
    v = x[:, C : 4 * C].view(x.shape[0], C, 3)
    m = x[:, 4 * C : 13 * C].view(x.shape[0], C, 9)
    return s, v, m


def _sparse_tp_inference_bmm_s0(
    t1_s0: torch.Tensor,
    t2_s0: torch.Tensor,
    w_flat: torch.Tensor,
    C1: int,
    C2: int,
    Cout: int,
) -> torch.Tensor:
    """
    Inference-only forward using batched GEMMs.

    This path stays in pure PyTorch so `torch.compile` can optimize it well.
    """
    E = t1_s0.shape[0]
    t1s, t1v, t1m = _split_s0_rank_blocks(t1_s0, C1)
    t2s, t2v, t2m = _split_s0_rank_blocks(t2_s0, C2)
    WBS = Cout * C1 * C2
    w0, w1, w2, w3, w4, w5 = [
        w_flat[:, i * WBS : (i + 1) * WBS].view(E, Cout, C1, C2) for i in range(6)
    ]

    tmp0 = torch.bmm(w0.reshape(E, Cout * C1, C2), t2s.unsqueeze(-1)).view(E, Cout, C1)
    out_s = torch.bmm(tmp0, t1s.unsqueeze(-1)).squeeze(-1)

    alpha1 = torch.bmm(
        w1.permute(0, 1, 3, 2).reshape(E, Cout * C2, C1),
        t1s.unsqueeze(-1),
    ).view(E, Cout, C2)
    out_v = torch.bmm(alpha1, t2v)
    beta3 = torch.bmm(w3.reshape(E, Cout * C1, C2), t2s.unsqueeze(-1)).view(E, Cout, C1)
    out_v = out_v + torch.bmm(beta3, t1v)

    alpha2 = torch.bmm(
        w2.permute(0, 1, 3, 2).reshape(E, Cout * C2, C1),
        t1s.unsqueeze(-1),
    ).view(E, Cout, C2)
    out_m = torch.bmm(alpha2, t2m)
    beta5 = torch.bmm(w5.reshape(E, Cout * C1, C2), t2s.unsqueeze(-1)).view(E, Cout, C1)
    out_m = out_m + torch.bmm(beta5, t1m)

    w4r = w4.reshape(E, Cout * C1, C2)
    cols = []
    for j in range(3):
        tmp_j = torch.bmm(w4r, t2v[:, :, j].unsqueeze(-1)).view(E, Cout, C1)
        cols.append(torch.bmm(tmp_j, t1v))
    out_m = out_m + torch.stack(cols, dim=-1).reshape(E, Cout, 9)

    return torch.cat([out_s, out_v.reshape(E, Cout * 3), out_m.reshape(E, Cout * 9)], dim=-1)


def _get_compiled_inference_fn() -> Callable | None:
    global _COMPILED_INFERENCE_FN, _COMPILED_INFERENCE_DISABLED
    if _COMPILED_INFERENCE_DISABLED or not hasattr(torch, 'compile'):
        return None
    if _COMPILED_INFERENCE_FN is None:
        try:
            _COMPILED_INFERENCE_FN = torch.compile(
                _sparse_tp_inference_bmm_s0,
                mode='max-autotune-no-cudagraphs',
                fullgraph=False,
            )
        except Exception:
            _COMPILED_INFERENCE_DISABLED = True
            return None
    return _COMPILED_INFERENCE_FN


if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_E": 32, "BLOCK_COUT": 2}, num_warps=2),
            triton.Config({"BLOCK_E": 64, "BLOCK_COUT": 2}, num_warps=4),
            triton.Config({"BLOCK_E": 64, "BLOCK_COUT": 4}, num_warps=4),
            triton.Config({"BLOCK_E": 128, "BLOCK_COUT": 2}, num_warps=4),
            triton.Config({"BLOCK_E": 128, "BLOCK_COUT": 4}, num_warps=8),
        ],
        key=["C1", "C2", "Cout"],
    )
    @triton.jit
    def _sparse_tp_fwd_kernel(
        out_ptr,
        t1_ptr,
        t2_ptr,
        w_ptr,
        E,
        Cout,
        C1: tl.constexpr,
        C2: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_COUT: tl.constexpr,
    ):
        """
        Grid: (ceil(E/BLOCK_E), ceil(Cout/BLOCK_COUT))
        Each program handles one edge block and one output-channel block.
        """
        pid_e = tl.program_id(0)
        pid_c = tl.program_id(1)

        e_off = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
        c_off = pid_c * BLOCK_COUT + tl.arange(0, BLOCK_COUT)
        e_mask = e_off < E
        c_mask = c_off < Cout
        ec_mask = c_mask[:, None] & e_mask[None, :]

        T1D: tl.constexpr = C1 * 13
        T2D: tl.constexpr = C2 * 13
        WBS: tl.constexpr = C1 * C2
        OD = Cout * 13
        WD = 6 * Cout * WBS

        t1_base = t1_ptr + e_off * T1D
        t2_base = t2_ptr + e_off * T2D
        w_base = w_ptr + e_off * WD
        out_base = out_ptr + e_off * OD
        cout_stride = c_off[:, None] * WBS

        acc_s = tl.zeros([BLOCK_COUT, BLOCK_E], dtype=tl.float32)
        acc_v0 = tl.zeros([BLOCK_COUT, BLOCK_E], dtype=tl.float32)
        acc_v1 = tl.zeros([BLOCK_COUT, BLOCK_E], dtype=tl.float32)
        acc_v2 = tl.zeros([BLOCK_COUT, BLOCK_E], dtype=tl.float32)
        acc_m0 = tl.zeros([BLOCK_COUT, BLOCK_E], dtype=tl.float32)
        acc_m1 = tl.zeros([BLOCK_COUT, BLOCK_E], dtype=tl.float32)
        acc_m2 = tl.zeros([BLOCK_COUT, BLOCK_E], dtype=tl.float32)
        acc_m3 = tl.zeros([BLOCK_COUT, BLOCK_E], dtype=tl.float32)
        acc_m4 = tl.zeros([BLOCK_COUT, BLOCK_E], dtype=tl.float32)
        acc_m5 = tl.zeros([BLOCK_COUT, BLOCK_E], dtype=tl.float32)
        acc_m6 = tl.zeros([BLOCK_COUT, BLOCK_E], dtype=tl.float32)
        acc_m7 = tl.zeros([BLOCK_COUT, BLOCK_E], dtype=tl.float32)
        acc_m8 = tl.zeros([BLOCK_COUT, BLOCK_E], dtype=tl.float32)

        for a in range(C1):
            t1_s = tl.load(t1_base + a, mask=e_mask, other=0.0)
            t1_v0 = tl.load(t1_base + C1 + a * 3 + 0, mask=e_mask, other=0.0)
            t1_v1 = tl.load(t1_base + C1 + a * 3 + 1, mask=e_mask, other=0.0)
            t1_v2 = tl.load(t1_base + C1 + a * 3 + 2, mask=e_mask, other=0.0)
            t1_m0 = tl.load(t1_base + 4 * C1 + a * 9 + 0, mask=e_mask, other=0.0)
            t1_m1 = tl.load(t1_base + 4 * C1 + a * 9 + 1, mask=e_mask, other=0.0)
            t1_m2 = tl.load(t1_base + 4 * C1 + a * 9 + 2, mask=e_mask, other=0.0)
            t1_m3 = tl.load(t1_base + 4 * C1 + a * 9 + 3, mask=e_mask, other=0.0)
            t1_m4 = tl.load(t1_base + 4 * C1 + a * 9 + 4, mask=e_mask, other=0.0)
            t1_m5 = tl.load(t1_base + 4 * C1 + a * 9 + 5, mask=e_mask, other=0.0)
            t1_m6 = tl.load(t1_base + 4 * C1 + a * 9 + 6, mask=e_mask, other=0.0)
            t1_m7 = tl.load(t1_base + 4 * C1 + a * 9 + 7, mask=e_mask, other=0.0)
            t1_m8 = tl.load(t1_base + 4 * C1 + a * 9 + 8, mask=e_mask, other=0.0)

            for b in range(C2):
                t2_s = tl.load(t2_base + b, mask=e_mask, other=0.0)
                t2_v0 = tl.load(t2_base + C2 + b * 3 + 0, mask=e_mask, other=0.0)
                t2_v1 = tl.load(t2_base + C2 + b * 3 + 1, mask=e_mask, other=0.0)
                t2_v2 = tl.load(t2_base + C2 + b * 3 + 2, mask=e_mask, other=0.0)
                t2_m0 = tl.load(t2_base + 4 * C2 + b * 9 + 0, mask=e_mask, other=0.0)
                t2_m1 = tl.load(t2_base + 4 * C2 + b * 9 + 1, mask=e_mask, other=0.0)
                t2_m2 = tl.load(t2_base + 4 * C2 + b * 9 + 2, mask=e_mask, other=0.0)
                t2_m3 = tl.load(t2_base + 4 * C2 + b * 9 + 3, mask=e_mask, other=0.0)
                t2_m4 = tl.load(t2_base + 4 * C2 + b * 9 + 4, mask=e_mask, other=0.0)
                t2_m5 = tl.load(t2_base + 4 * C2 + b * 9 + 5, mask=e_mask, other=0.0)
                t2_m6 = tl.load(t2_base + 4 * C2 + b * 9 + 6, mask=e_mask, other=0.0)
                t2_m7 = tl.load(t2_base + 4 * C2 + b * 9 + 7, mask=e_mask, other=0.0)
                t2_m8 = tl.load(t2_base + 4 * C2 + b * 9 + 8, mask=e_mask, other=0.0)

                w_idx = cout_stride + a * C2 + b
                w0 = tl.load(w_base[None, :] + 0 * Cout * WBS + w_idx, mask=ec_mask, other=0.0)
                w1 = tl.load(w_base[None, :] + 1 * Cout * WBS + w_idx, mask=ec_mask, other=0.0)
                w2 = tl.load(w_base[None, :] + 2 * Cout * WBS + w_idx, mask=ec_mask, other=0.0)
                w3 = tl.load(w_base[None, :] + 3 * Cout * WBS + w_idx, mask=ec_mask, other=0.0)
                w4 = tl.load(w_base[None, :] + 4 * Cout * WBS + w_idx, mask=ec_mask, other=0.0)
                w5 = tl.load(w_base[None, :] + 5 * Cout * WBS + w_idx, mask=ec_mask, other=0.0)

                t1_s_b = t1_s[None, :]
                t2_s_b = t2_s[None, :]
                t1_v0_b = t1_v0[None, :]
                t1_v1_b = t1_v1[None, :]
                t1_v2_b = t1_v2[None, :]
                t2_v0_b = t2_v0[None, :]
                t2_v1_b = t2_v1[None, :]
                t2_v2_b = t2_v2[None, :]
                t1_m0_b = t1_m0[None, :]
                t1_m1_b = t1_m1[None, :]
                t1_m2_b = t1_m2[None, :]
                t1_m3_b = t1_m3[None, :]
                t1_m4_b = t1_m4[None, :]
                t1_m5_b = t1_m5[None, :]
                t1_m6_b = t1_m6[None, :]
                t1_m7_b = t1_m7[None, :]
                t1_m8_b = t1_m8[None, :]
                t2_m0_b = t2_m0[None, :]
                t2_m1_b = t2_m1[None, :]
                t2_m2_b = t2_m2[None, :]
                t2_m3_b = t2_m3[None, :]
                t2_m4_b = t2_m4[None, :]
                t2_m5_b = t2_m5[None, :]
                t2_m6_b = t2_m6[None, :]
                t2_m7_b = t2_m7[None, :]
                t2_m8_b = t2_m8[None, :]

                acc_s += w0 * t1_s_b * t2_s_b

                w1_t1s = w1 * t1_s_b
                acc_v0 += w1_t1s * t2_v0_b
                acc_v1 += w1_t1s * t2_v1_b
                acc_v2 += w1_t1s * t2_v2_b

                w2_t1s = w2 * t1_s_b
                acc_m0 += w2_t1s * t2_m0_b
                acc_m1 += w2_t1s * t2_m1_b
                acc_m2 += w2_t1s * t2_m2_b
                acc_m3 += w2_t1s * t2_m3_b
                acc_m4 += w2_t1s * t2_m4_b
                acc_m5 += w2_t1s * t2_m5_b
                acc_m6 += w2_t1s * t2_m6_b
                acc_m7 += w2_t1s * t2_m7_b
                acc_m8 += w2_t1s * t2_m8_b

                w3_t2s = w3 * t2_s_b
                acc_v0 += w3_t2s * t1_v0_b
                acc_v1 += w3_t2s * t1_v1_b
                acc_v2 += w3_t2s * t1_v2_b

                w4_v0 = w4 * t1_v0_b
                w4_v1 = w4 * t1_v1_b
                w4_v2 = w4 * t1_v2_b
                acc_m0 += w4_v0 * t2_v0_b
                acc_m1 += w4_v0 * t2_v1_b
                acc_m2 += w4_v0 * t2_v2_b
                acc_m3 += w4_v1 * t2_v0_b
                acc_m4 += w4_v1 * t2_v1_b
                acc_m5 += w4_v1 * t2_v2_b
                acc_m6 += w4_v2 * t2_v0_b
                acc_m7 += w4_v2 * t2_v1_b
                acc_m8 += w4_v2 * t2_v2_b

                w5_t2s = w5 * t2_s_b
                acc_m0 += w5_t2s * t1_m0_b
                acc_m1 += w5_t2s * t1_m1_b
                acc_m2 += w5_t2s * t1_m2_b
                acc_m3 += w5_t2s * t1_m3_b
                acc_m4 += w5_t2s * t1_m4_b
                acc_m5 += w5_t2s * t1_m5_b
                acc_m6 += w5_t2s * t1_m6_b
                acc_m7 += w5_t2s * t1_m7_b
                acc_m8 += w5_t2s * t1_m8_b

        out_scalar = out_base[None, :] + c_off[:, None]
        tl.store(out_scalar, acc_s, mask=ec_mask)

        v_off = out_base[None, :] + Cout + c_off[:, None] * 3
        tl.store(v_off + 0, acc_v0, mask=ec_mask)
        tl.store(v_off + 1, acc_v1, mask=ec_mask)
        tl.store(v_off + 2, acc_v2, mask=ec_mask)

        m_off = out_base[None, :] + 4 * Cout + c_off[:, None] * 9
        tl.store(m_off + 0, acc_m0, mask=ec_mask)
        tl.store(m_off + 1, acc_m1, mask=ec_mask)
        tl.store(m_off + 2, acc_m2, mask=ec_mask)
        tl.store(m_off + 3, acc_m3, mask=ec_mask)
        tl.store(m_off + 4, acc_m4, mask=ec_mask)
        tl.store(m_off + 5, acc_m5, mask=ec_mask)
        tl.store(m_off + 6, acc_m6, mask=ec_mask)
        tl.store(m_off + 7, acc_m7, mask=ec_mask)
        tl.store(m_off + 8, acc_m8, mask=ec_mask)


def _sparse_tp_composite_s0(t1_s0, t2_s0, w_flat, C1, C2, Cout):
    """Composite PyTorch forward on the s=0 blocks only."""
    Lmax = 2
    E = t1_s0.shape[0]
    A = _split_s0(t1_s0, C1, Lmax)
    B = _split_s0(t2_s0, C2, Lmax)

    out = {
        L: t1_s0.new_zeros((E, Cout) if L == 0 else (E, Cout, *_rank_shape(L)))
        for L in range(Lmax + 1)
    }

    WBS = Cout * C1 * C2
    w_idx = 0
    for L1, L2, Lout, nrm in _S0_PATHS:
        Wp = w_flat[:, w_idx : w_idx + WBS].view(E, Cout, C1, C2)
        w_idx += WBS
        t1 = A[L1]
        t2 = B[L2]
        if L1 == 0 and L2 == 0:
            r = torch.einsum("ecab,ea,eb->ec", Wp, t1, t2)
        elif L1 == 0:
            idx = "ijk"[:L2]
            r = torch.einsum(f"ecab,ea,eb{idx}->ec{idx}", Wp, t1, t2)
        elif L2 == 0:
            idx = "ijk"[:L1]
            r = torch.einsum(f"ecab,ea{idx},eb->ec{idx}", Wp, t1, t2)
        else:
            i1, i2 = "ijk"[:L1], "lmn"[:L2]
            io = i1 + i2
            r = torch.einsum(f"ecab,ea{i1},eb{i2}->ec{io}", Wp, t1, t2) * nrm
        out[Lout] = out[Lout] + r

    return _merge_s0(out, Cout, Lmax)


class _SparseTPFused(torch.autograd.Function):
    """
    Forward: Triton fused kernel on CUDA.
    Backward:
      - first-order: explicit formulas
      - higher-order: recompute composite forward and let autograd build the graph
    """

    @staticmethod
    def forward(ctx, t1_s0, t2_s0, w_flat, C1, C2, Cout):
        E = t1_s0.shape[0]
        out_s0 = torch.zeros(E, Cout * 13, device=t1_s0.device, dtype=t1_s0.dtype)
        grid = lambda META: (
            triton.cdiv(E, META["BLOCK_E"]),
            triton.cdiv(Cout, META["BLOCK_COUT"]),
        )
        _sparse_tp_fwd_kernel[grid](out_s0, t1_s0, t2_s0, w_flat, E=E, Cout=Cout, C1=C1, C2=C2)
        ctx.save_for_backward(t1_s0, t2_s0, w_flat)
        ctx.C1, ctx.C2, ctx.Cout = C1, C2, Cout
        return out_s0

    @staticmethod
    def backward(ctx, grad_output):
        t1_s0, t2_s0, w_flat = ctx.saved_tensors
        C1, C2, Cout = ctx.C1, ctx.C2, ctx.Cout

        if torch.is_grad_enabled():
            grads = [None, None, None]
            inputs = []
            slots = []
            for slot, (need_grad, tensor) in enumerate(
                zip(ctx.needs_input_grad[:3], (t1_s0, t2_s0, w_flat))
            ):
                if need_grad:
                    inputs.append(tensor)
                    slots.append(slot)
            if inputs:
                with torch.enable_grad():
                    out = _sparse_tp_composite_s0(t1_s0, t2_s0, w_flat, C1, C2, Cout)
                    grad_vals = torch.autograd.grad(
                        out,
                        tuple(inputs),
                        grad_output,
                        create_graph=True,
                        allow_unused=True,
                    )
                for slot, grad in zip(slots, grad_vals):
                    grads[slot] = grad
            return grads[0], grads[1], grads[2], None, None, None

        E = t1_s0.shape[0]
        Lmax = 2
        WBS = Cout * C1 * C2

        A = _split_s0(t1_s0, C1, Lmax)
        B = _split_s0(t2_s0, C2, Lmax)
        G = _split_s0(grad_output, Cout, Lmax)

        grad_t1 = {L: t1_s0.new_zeros(A[L].shape) for L in range(Lmax + 1)}
        grad_t2 = {L: t2_s0.new_zeros(B[L].shape) for L in range(Lmax + 1)}
        grad_w_parts: List[torch.Tensor] = []

        w_idx = 0
        for L1, L2, Lout, nrm in _S0_PATHS:
            Wp = w_flat[:, w_idx : w_idx + WBS].view(E, Cout, C1, C2)
            w_idx += WBS
            t1 = A[L1]
            t2 = B[L2]
            g = G[Lout]

            if L1 == 0 and L2 == 0:
                gw = torch.einsum("ec,ea,eb->ecab", g, t1, t2)
                gt1 = torch.einsum("ecab,ec,eb->ea", Wp, g, t2)
                gt2 = torch.einsum("ecab,ec,ea->eb", Wp, g, t1)
            elif L1 == 0:
                idx = "ijk"[:L2]
                gw = torch.einsum(f"ec{idx},ea,eb{idx}->ecab", g, t1, t2)
                gt1 = torch.einsum(f"ecab,ec{idx},eb{idx}->ea", Wp, g, t2)
                gt2 = torch.einsum(f"ecab,ec{idx},ea->eb{idx}", Wp, g, t1)
            elif L2 == 0:
                idx = "ijk"[:L1]
                gw = torch.einsum(f"ec{idx},ea{idx},eb->ecab", g, t1, t2)
                gt1 = torch.einsum(f"ecab,ec{idx},eb->ea{idx}", Wp, g, t2)
                gt2 = torch.einsum(f"ecab,ec{idx},ea{idx}->eb", Wp, g, t1)
            else:
                i1, i2 = "ijk"[:L1], "lmn"[:L2]
                io = i1 + i2
                gw = torch.einsum(f"ec{io},ea{i1},eb{i2}->ecab", g, t1, t2) * nrm
                gt1 = torch.einsum(f"ecab,ec{io},eb{i2}->ea{i1}", Wp, g, t2) * nrm
                gt2 = torch.einsum(f"ecab,ec{io},ea{i1}->eb{i2}", Wp, g, t1) * nrm

            grad_w_parts.append(gw.reshape(E, WBS))
            grad_t1[L1] = grad_t1[L1] + gt1
            grad_t2[L2] = grad_t2[L2] + gt2

        grad_t1_flat = _merge_s0(grad_t1, C1, Lmax)
        grad_t2_flat = _merge_s0(grad_t2, C2, Lmax)
        grad_w = torch.cat(grad_w_parts, dim=-1)
        return grad_t1_flat, grad_t2_flat, grad_w, None, None, None


def _einsum_fallback(t1_s0, t2_s0, w_flat, C1, C2, Cout):
    """Composite fallback used on CPU or when Triton is unavailable."""
    return _sparse_tp_composite_s0(t1_s0, t2_s0, w_flat, C1, C2, Cout)


def sparse_tp_fused(
    x1: torch.Tensor,
    x2: torch.Tensor,
    w_flat: torch.Tensor,
    C1: int,
    C2: int,
    Cout: int,
) -> torch.Tensor:
    """
    Sparse tensor product for the common Lmax=2 / k0 / pseudo-zero case.

    Behavior by mode:
      - inference/no_grad on CUDA: compiled batched-GEMM fast path
      - grad-enabled on CUDA: Triton autograd path
      - fallback: composite PyTorch reference
    """
    E = x1.shape[0]
    t1_s0 = x1[..., : C1 * 13].contiguous()
    t2_s0 = x2[..., : C2 * 13].contiguous()
    w_flat = w_flat.contiguous()

    compiled_inference = None
    if x1.is_cuda and (not torch.is_grad_enabled()):
        compiled_inference = _get_compiled_inference_fn()

    if compiled_inference is not None:
        out_s0 = compiled_inference(t1_s0, t2_s0, w_flat, C1, C2, Cout)
    elif HAS_TRITON and x1.is_cuda and torch.is_grad_enabled():
        out_s0 = _SparseTPFused.apply(t1_s0, t2_s0, w_flat, C1, C2, Cout)
    else:
        out_s0 = _einsum_fallback(t1_s0, t2_s0, w_flat, C1, C2, Cout)

    out = torch.zeros(E, Cout * 26, device=x1.device, dtype=x1.dtype)
    out[..., : Cout * 13] = out_s0
    return out
