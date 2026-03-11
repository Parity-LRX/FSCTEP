"""
Triton CUDA kernels for ictd_irreps (direction harmonics + tensor product).

Optional dependency: triton. If not available or not on CUDA, ictd_irreps falls back to PyTorch.
- Direction harmonics: fused monomial evaluation + projection.
- Tensor product (FlashTP-style): fused outer-product + CG projection to reduce memory traffic
  and kernel launches; path-aggregated execution by (l1,l2) group.
"""

from __future__ import annotations

import os

import torch

_TRITON_AVAILABLE = False
triton = None
tl = None
try:
    import triton
    import triton.language as tl  # noqa: F811
    _TRITON_AVAILABLE = triton is not None
except Exception:
    pass

# Direction harmonics Triton kernel limits (auto-fallback to PyTorch above these).
# Computed from _TRITON_LMAX_LIMIT: sym_dim(L) = (L+1)(L+2)/2, K = 2L+1.
_TRITON_LMAX_LIMIT = 12
_D_MAX = (_TRITON_LMAX_LIMIT + 1) * (_TRITON_LMAX_LIMIT + 2) // 2  # 91
_K_MAX = 2 * _TRITON_LMAX_LIMIT + 1                                 # 25

# Tensor product: D = m1*m2, m_i = 2*l+1 <= _K_MAX
_TP_D_MAX = _K_MAX * _K_MAX                                         # 625
_TP_K_MAX = 512
# Sparse TP: max nnz per group (fall back to dense if nnz > this)
_TP_SPARSE_MAX_NNZ = 1024
# Set ICTD_USE_TRITON_TP=1 to use Triton fused TP kernel (default: 1 when CUDA available)
_USE_TRITON_TP = os.environ.get("ICTD_USE_TRITON_TP", "1") == "1"


if _TRITON_AVAILABLE and triton is not None and tl is not None:

    @triton.jit
    def _direction_harmonics_kernel(
        n_ptr,
        exps_ptr,
        coefs_ptr,
        P_ptr,
        out_ptr,
        N: tl.constexpr,
        D: tl.constexpr,
        K: tl.constexpr,
        stride_n0,
        stride_n1,
        stride_exps0,
        stride_exps1,
        stride_P0,
        stride_P1,
        stride_out0,
        stride_out1,
    ):
        """One program per row: parallel over N, vectorized load of P[d,:] for matmul."""
        row_idx = tl.program_id(0)
        if row_idx >= N:
            return
        n0 = tl.load(n_ptr + row_idx * stride_n0)
        n1 = tl.load(n_ptr + row_idx * stride_n0 + stride_n1)
        n2 = tl.load(n_ptr + row_idx * stride_n0 + 2 * stride_n1)
        acc = tl.zeros((K,), dtype=out_ptr.dtype.element_ty)
        for d in range(D):
            a = tl.load(exps_ptr + d * stride_exps0)
            b = tl.load(exps_ptr + d * stride_exps0 + stride_exps1)
            c = tl.load(exps_ptr + d * stride_exps0 + 2 * stride_exps1)
            term = tl.load(coefs_ptr + d)
            term = term * tl.math.pow(n0, tl.cast(a, n0.dtype))
            term = term * tl.math.pow(n1, tl.cast(b, n1.dtype))
            term = term * tl.math.pow(n2, tl.cast(c, n2.dtype))
            off_k = tl.arange(0, K)
            p_row = tl.load(P_ptr + d * stride_P0 + off_k * stride_P1)
            acc += term * p_row
        tl.store(out_ptr + row_idx * stride_out0 + tl.arange(0, K) * stride_out1, acc)


def direction_harmonics_triton(
    n: torch.Tensor,
    l: int,
    exps: torch.Tensor,
    coefs: torch.Tensor,
    P: torch.Tensor,
) -> torch.Tensor | None:
    """
    CUDA path: fused monomial + projection for direction harmonics.
    n: (N, 3), l >= 1. exps, coefs, P from _dir_monomial_exps_coefs and _dir_proj_cpu_f64.
    Returns (N, 2l+1) or None to fall back to PyTorch.
    """
    if not _TRITON_AVAILABLE or triton is None or not n.is_cuda or n.dim() != 2 or l == 0:
        return None
    N, three = n.shape
    if three != 3:
        return None
    D = exps.shape[0]
    K = P.shape[1]
    if D > _D_MAX or K > _K_MAX:
        return None
    n = n.contiguous()
    exps = exps.to(device=n.device, dtype=torch.int32).contiguous()
    coefs = coefs.to(device=n.device, dtype=n.dtype).contiguous()
    P = P.contiguous()
    out = torch.empty((N, K), device=n.device, dtype=n.dtype)
    _direction_harmonics_kernel[(N,)](
        n,
        exps,
        coefs,
        P,
        out,
        N=N,
        D=D,
        K=K,
        stride_n0=n.stride(0),
        stride_n1=n.stride(1),
        stride_exps0=exps.stride(0),
        stride_exps1=exps.stride(1),
        stride_P0=P.stride(0),
        stride_P1=P.stride(1),
        stride_out0=out.stride(0),
        stride_out1=out.stride(1),
    )
    return out

    # -------------------------------------------------------------------------
    # FlashTP-style: fused outer-product + CG projection for tensor product
    # -------------------------------------------------------------------------

    @triton.jit
    def _tp_outer_proj_kernel_impl(
        a_ptr,
        b_ptr,
        U_ptr,
        y_ptr,
        B: tl.constexpr,
        num_i: tl.constexpr,
        num_j: tl.constexpr,
        M1: tl.constexpr,
        M2: tl.constexpr,
        D: tl.constexpr,
        K: tl.constexpr,
        stride_a_b,
        stride_a_i,
        stride_a_m,
        stride_b_b,
        stride_b_j,
        stride_b_m,
        stride_U_d,
        stride_U_k,
        stride_y_b,
        stride_y_i,
        stride_y_j,
        stride_y_k,
    ):
        """
        FlashTP-style: one program per (b,i,j). y = (a outer b) @ U without writing t_flat.
        Loop over d: acc += a[m1]*b[m2] * U[d,:]; no intermediate t_vec in global memory.
        """
        pid = tl.program_id(0)
        num_ij = num_i * num_j
        if pid >= B * num_ij:
            return
        b = pid // num_ij
        ij = pid % num_ij
        i_idx = ij // num_j
        j_idx = ij % num_j

        off_a = b * stride_a_b + i_idx * stride_a_i
        off_b = b * stride_b_b + j_idx * stride_b_j

        acc = tl.zeros((K,), dtype=U_ptr.dtype.element_ty)
        for d in range(D):
            m1_idx = d // M2
            m2_idx = d % M2
            t_val = tl.load(a_ptr + off_a + m1_idx * stride_a_m) * tl.load(b_ptr + off_b + m2_idx * stride_b_m)
            off_u = d * stride_U_d + tl.arange(0, K) * stride_U_k
            u_row = tl.load(U_ptr + off_u, mask=tl.arange(0, K) < K)
            acc += t_val * u_row
        off_y = b * stride_y_b + i_idx * stride_y_i + j_idx * stride_y_j
        tl.store(y_ptr + off_y + tl.arange(0, K) * stride_y_k, acc, mask=tl.arange(0, K) < K)

    # -------------------------------------------------------------------------
    # Sparse CG: fused outer-product + sparse projection (only nnz ops per (b,i,j))
    # -------------------------------------------------------------------------

    @triton.jit
    def _tp_sparse_outer_proj_kernel(
        a_ptr,
        b_ptr,
        d_idx_ptr,
        k_idx_ptr,
        vals_ptr,
        nnz_ptr,
        y_ptr,
        B: tl.constexpr,
        num_i: tl.constexpr,
        num_j: tl.constexpr,
        M1: tl.constexpr,
        M2: tl.constexpr,
        K: tl.constexpr,
        MAX_NNZ: tl.constexpr,
        stride_a_b,
        stride_a_i,
        stride_a_m,
        stride_b_b,
        stride_b_j,
        stride_b_m,
        stride_y_b,
        stride_y_i,
        stride_y_j,
        stride_y_k,
    ):
        """One program per (b,i,j). acc[k] += t_val * v for each (d,k,v) in sparse U; nnz loaded at runtime."""
        pid = tl.program_id(0)
        num_ij = num_i * num_j
        if pid >= B * num_ij:
            return
        b = pid // num_ij
        ij = pid % num_ij
        i_idx = ij // num_j
        j_idx = ij % num_j

        off_a = b * stride_a_b + i_idx * stride_a_i
        off_b = b * stride_b_b + j_idx * stride_b_j

        nnz_val = tl.load(nnz_ptr)
        acc = tl.zeros((K,), dtype=vals_ptr.dtype.element_ty)
        for i in range(MAX_NNZ):
            mask = i < nnz_val
            d = tl.load(d_idx_ptr + i, mask=mask, other=0)
            k = tl.load(k_idx_ptr + i, mask=mask, other=0)
            v = tl.load(vals_ptr + i, mask=mask, other=0.0)
            m1_idx = d // M2
            m2_idx = d % M2
            t_val = tl.load(a_ptr + off_a + m1_idx * stride_a_m) * tl.load(b_ptr + off_b + m2_idx * stride_b_m)
            add_val = tl.where(mask, t_val * v, 0.0)
            acc = tl.where(tl.arange(0, K) == k, acc + add_val, acc)
        off_y = b * stride_y_b + i_idx * stride_y_i + j_idx * stride_y_j
        tl.store(y_ptr + off_y + tl.arange(0, K) * stride_y_k, acc, mask=tl.arange(0, K) < K)

    def _tp_fused_outer_proj(
        a: torch.Tensor,
        b: torch.Tensor,
        U: torch.Tensor,
        m1: int,
        m2: int,
    ) -> torch.Tensor | None:
        """
        Fused outer-product + projection: y[b,i,j,:] = (a[b,i,:] outer b[b,j,:]) @ U.
        a: (B, num_i, m1), b: (B, num_j, m2), U: (D, K) with D=m1*m2.
        Returns y (B, num_i, num_j, K) or None to fall back to PyTorch.
        """
        if not _TRITON_AVAILABLE or not _USE_TRITON_TP or not a.is_cuda or a.dim() != 3 or b.dim() != 3:
            return None
        D = m1 * m2
        K = U.shape[1]
        if D > _TP_D_MAX or K > _TP_K_MAX:
            return None
        B, num_i, _ = a.shape
        _, num_j, _ = b.shape
        a = a.contiguous()
        b = b.contiguous()
        U = U.contiguous()
        y = torch.empty((B, num_i, num_j, K), device=a.device, dtype=a.dtype)
        grid = (triton.cdiv(B * num_i * num_j, 1),)
        _tp_outer_proj_kernel_impl[grid](
            a,
            b,
            U,
            y,
            B=B,
            num_i=num_i,
            num_j=num_j,
            M1=m1,
            M2=m2,
            D=D,
            K=K,
            stride_a_b=a.stride(0),
            stride_a_i=a.stride(1),
            stride_a_m=a.stride(2),
            stride_b_b=b.stride(0),
            stride_b_j=b.stride(1),
            stride_b_m=b.stride(2),
            stride_U_d=U.stride(0),
            stride_U_k=U.stride(1),
            stride_y_b=y.stride(0),
            stride_y_i=y.stride(1),
            stride_y_j=y.stride(2),
            stride_y_k=y.stride(3),
        )
        return y

    def _tp_fused_outer_proj_sparse(
        a: torch.Tensor,
        b: torch.Tensor,
        d_idx: torch.Tensor,
        k_idx: torch.Tensor,
        vals: torch.Tensor,
        m1: int,
        m2: int,
        K: int,
    ) -> torch.Tensor | None:
        """
        Fused outer-product + sparse projection: y = (a outer b) @ U_sparse.
        d_idx, k_idx: int32 indices (nnz,) into U; vals: (nnz,) non-zero values.
        Returns y (B, num_i, num_j, K) or None to fall back.
        """
        if not _TRITON_AVAILABLE or not _USE_TRITON_TP or not a.is_cuda or a.dim() != 3 or b.dim() != 3:
            return None
        nnz = vals.numel()
        if nnz == 0 or nnz > _TP_SPARSE_MAX_NNZ:
            return None
        if K > _TP_K_MAX or m1 * m2 > _TP_D_MAX:
            return None
        B, num_i, _ = a.shape
        _, num_j, _ = b.shape
        a = a.contiguous()
        b = b.contiguous()
        # Ensure indices are int32 and on same device
        d_idx = d_idx.to(device=a.device, dtype=torch.int32).contiguous()
        k_idx = k_idx.to(device=a.device, dtype=torch.int32).contiguous()
        vals = vals.to(device=a.device, dtype=a.dtype).contiguous()
        # Pad to MAX_NNZ so kernel can always load (mask handles i >= nnz)
        MAX_NNZ = _TP_SPARSE_MAX_NNZ
        if nnz < MAX_NNZ:
            d_idx = torch.nn.functional.pad(d_idx, (0, MAX_NNZ - nnz), value=0)
            k_idx = torch.nn.functional.pad(k_idx, (0, MAX_NNZ - nnz), value=0)
            vals = torch.nn.functional.pad(vals, (0, MAX_NNZ - nnz), value=0.0)
        nnz_t = torch.tensor([nnz], device=a.device, dtype=torch.int32)

        y = torch.empty((B, num_i, num_j, K), device=a.device, dtype=a.dtype)
        grid = (triton.cdiv(B * num_i * num_j, 1),)
        _tp_sparse_outer_proj_kernel[grid](
            a,
            b,
            d_idx,
            k_idx,
            vals,
            nnz_t,
            y,
            B=B,
            num_i=num_i,
            num_j=num_j,
            M1=m1,
            M2=m2,
            K=K,
            MAX_NNZ=MAX_NNZ,
            stride_a_b=a.stride(0),
            stride_a_i=a.stride(1),
            stride_a_m=a.stride(2),
            stride_b_b=b.stride(0),
            stride_b_j=b.stride(1),
            stride_b_m=b.stride(2),
            stride_y_b=y.stride(0),
            stride_y_i=y.stride(1),
            stride_y_j=y.stride(2),
            stride_y_k=y.stride(3),
        )
        return y

    # -------------------------------------------------------------------------
    # Fused outer-product + projection + channel mixing (single kernel, no y write-back)
    # -------------------------------------------------------------------------
    _TP_FUSED_PROJ_MIX_NUM_PATHS_MAX = 16
    _TP_FUSED_PROJ_MIX_MAX_KDIM = 16

    @triton.jit
    def _tp_fused_outer_proj_channel_mix_kernel(
        a_ptr,
        b_ptr,
        U_ptr,
        num_paths_ptr,
        W_stack_ptr,
        seg_s_ptr,
        seg_e_ptr,
        out_buf_ptr,
        B: tl.constexpr,
        num_i: tl.constexpr,
        num_j: tl.constexpr,
        M1: tl.constexpr,
        M2: tl.constexpr,
        D: tl.constexpr,
        K_total: tl.constexpr,
        O: tl.constexpr,
        NUM_PATHS_MAX: tl.constexpr,
        stride_a_b,
        stride_a_i,
        stride_a_m,
        stride_b_b,
        stride_b_j,
        stride_b_m,
        stride_U_d,
        stride_U_k,
        stride_W_p,
        stride_W_o,
        stride_W_ij,
        stride_buf_b,
        stride_buf_p,
        stride_buf_o,
        stride_buf_k,
        MAX_KDIM: tl.constexpr,
    ):
        """
        One program per (b, ij). Compute y = (a outer b) @ U in registers; then for each path p,
        segment [s,e): atomic_add out_buf[b,p,o,s+k] += W[p,o,ij]*y[s+k]. No global y write.
        num_paths_ptr: pointer to int32 scalar (num_paths).
        """
        pid = tl.program_id(0)
        num_ij = num_i * num_j
        if pid >= B * num_ij:
            return
        b = pid // num_ij
        ij = pid % num_ij
        i_idx = ij // num_j
        j_idx = ij % num_j

        off_a = b * stride_a_b + i_idx * stride_a_i
        off_b = b * stride_b_b + j_idx * stride_b_j

        # 1) y = (a outer b) @ U in registers
        acc = tl.zeros((K_total,), dtype=U_ptr.dtype.element_ty)
        for d in range(D):
            m1_idx = d // M2
            m2_idx = d % M2
            t_val = tl.load(a_ptr + off_a + m1_idx * stride_a_m) * tl.load(b_ptr + off_b + m2_idx * stride_b_m)
            off_u = d * stride_U_d + tl.arange(0, K_total) * stride_U_k
            u_row = tl.load(U_ptr + off_u, mask=tl.arange(0, K_total) < K_total)
            acc += t_val * u_row

        num_paths = tl.load(num_paths_ptr)
        for p in range(NUM_PATHS_MAX):
            if p < num_paths:
                s = tl.load(seg_s_ptr + p)
                e = tl.load(seg_e_ptr + p)
                kdim = e - s
                for k in range(MAX_KDIM):
                    if k < kdim:
                        y_val = acc[s + k]
                        for o_idx in range(O):
                            w_val = tl.load(W_stack_ptr + p * stride_W_p + o_idx * stride_W_o + ij * stride_W_ij)
                            out_off = b * stride_buf_b + p * stride_buf_p + o_idx * stride_buf_o + (s + k) * stride_buf_k
                            tl.atomic_add(out_buf_ptr + out_off, w_val * y_val)

    def _tp_fused_outer_proj_channel_mix(
        a: torch.Tensor,
        b: torch.Tensor,
        U: torch.Tensor,
        W_stack: torch.Tensor,
        segments: list,
        k_total: int,
        mul_out: int,
        m1: int,
        m2: int,
    ) -> torch.Tensor | None:
        """
        Fused outer-product + projection + channel mixing. One kernel, no y write-back.
        a: (B, num_i, m1), b: (B, num_j, m2), U: (D, k_total), W_stack: (num_paths, o, ij).
        segments: list of (p_idx, l3, s, e). Returns out_buf (B, num_paths, o, k_total) for caller to scatter to out[l3].
        """
        if not _TRITON_AVAILABLE or not _USE_TRITON_TP or not a.is_cuda or a.dim() != 3 or b.dim() != 3:
            return None
        num_paths = len(segments)
        if num_paths > _TP_FUSED_PROJ_MIX_NUM_PATHS_MAX:
            return None
        D = m1 * m2
        K_total = k_total
        if D > _TP_D_MAX or K_total > _TP_K_MAX:
            return None
        B, num_i, _ = a.shape
        _, num_j, _ = b.shape
        o = mul_out
        a = a.contiguous()
        b = b.contiguous()
        U = U.contiguous()
        W_stack = W_stack.contiguous()
        device = a.device
        dtype = a.dtype

        seg_s = torch.zeros(num_paths, device=device, dtype=torch.int32)
        seg_e = torch.zeros(num_paths, device=device, dtype=torch.int32)
        for p, (_p_idx, _l3, s, e) in enumerate(segments):
            seg_s[p] = int(s)
            seg_e[p] = int(e)
        num_paths_t = torch.tensor([num_paths], device=device, dtype=torch.int32)

        out_buf = torch.zeros((B, num_paths, o, K_total), device=device, dtype=dtype)
        grid = (triton.cdiv(B * num_i * num_j, 1),)
        NUM_PATHS_MAX = _TP_FUSED_PROJ_MIX_NUM_PATHS_MAX
        MAX_KDIM = _TP_FUSED_PROJ_MIX_MAX_KDIM
        _tp_fused_outer_proj_channel_mix_kernel[grid](
            a,
            b,
            U,
            num_paths_t,
            W_stack,
            seg_s,
            seg_e,
            out_buf,
            B=B,
            num_i=num_i,
            num_j=num_j,
            M1=m1,
            M2=m2,
            D=D,
            K_total=K_total,
            O=o,
            NUM_PATHS_MAX=NUM_PATHS_MAX,
            stride_a_b=a.stride(0),
            stride_a_i=a.stride(1),
            stride_a_m=a.stride(2),
            stride_b_b=b.stride(0),
            stride_b_j=b.stride(1),
            stride_b_m=b.stride(2),
            stride_U_d=U.stride(0),
            stride_U_k=U.stride(1),
            stride_W_p=W_stack.stride(0),
            stride_W_o=W_stack.stride(1),
            stride_W_ij=W_stack.stride(2),
            stride_buf_b=out_buf.stride(0),
            stride_buf_p=out_buf.stride(1),
            stride_buf_o=out_buf.stride(2),
            stride_buf_k=out_buf.stride(3),
            MAX_KDIM=MAX_KDIM,
        )
        return out_buf
