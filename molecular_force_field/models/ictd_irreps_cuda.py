from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional, Tuple

import torch

_SUPPORTED_ICTD_TP_BACKENDS = {"auto", "pytorch", "cuda_ext"}
_AUTO_EXPERIMENTAL_ENV = "ICTD_ENABLE_EXPERIMENTAL_CUDA_TP"

try:
    from molecular_force_field import _C_ictd_tp as _ictd_tp_ext
except Exception:  # pragma: no cover - optional extension
    _ictd_tp_ext = None


def normalize_ictd_tp_backend(backend: str | None) -> str:
    value = str(backend or "auto").strip().lower()
    if value not in _SUPPORTED_ICTD_TP_BACKENDS:
        raise ValueError(
            f"Unsupported ictd_tp_backend={backend!r}; "
            f"expected one of {sorted(_SUPPORTED_ICTD_TP_BACKENDS)}"
        )
    return value


def cuda_ext_available() -> bool:
    required = (
        "project_bucket_forward",
        "project_bucket_transpose_a",
        "project_bucket_transpose_b",
        "project_bucket_transpose_u",
        "mix_bucket_forward",
        "mix_bucket_transpose_y",
        "mix_bucket_transpose_w",
        "mix_bucket_transpose_g",
        "bucketed_tp_forward",
    )
    return _ictd_tp_ext is not None and all(hasattr(_ictd_tp_ext, name) for name in required)


def cuda_ext_has_cuda() -> bool:
    return bool(
        _ictd_tp_ext is not None
        and hasattr(_ictd_tp_ext, "has_cuda")
        and _ictd_tp_ext.has_cuda()
    )


def auto_experimental_cuda_enabled() -> bool:
    return os.environ.get(_AUTO_EXPERIMENTAL_ENV, "0") == "1"


@contextmanager
def _exact_cuda_matmul_mode(sample: torch.Tensor):
    if sample.device.type != "cuda":
        yield
        return
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def grouped_tp_cuda_ext_support_reason(
    *,
    backend: str,
    sample: torch.Tensor,
    compute_dtype: torch.dtype,
    internal_weights: bool,
    weights: torch.Tensor | None,
) -> Tuple[bool, str]:
    normalized = normalize_ictd_tp_backend(backend)
    if normalized == "pytorch":
        return False, "backend_forced_to_pytorch"
    if normalized == "auto" and not auto_experimental_cuda_enabled():
        return False, "auto_experimental_cuda_disabled"
    if not cuda_ext_available():
        return False, "cuda_extension_unavailable"
    if not cuda_ext_has_cuda():
        return False, "cuda_extension_built_without_cuda"
    if sample.device.type != "cuda":
        return False, "sample_not_on_cuda"
    if sample.dtype not in (torch.float32, torch.float64):
        return False, f"sample_dtype_{sample.dtype}_unsupported"
    if compute_dtype not in (torch.float32, torch.float64):
        return False, f"compute_dtype_{compute_dtype}_unsupported"
    if sample.dtype != compute_dtype:
        return False, "sample_dtype_must_match_compute_dtype"
    if not internal_weights:
        return False, "external_full_weights_not_supported"
    if weights is not None and weights.dim() != 2:
        return False, "gates_must_be_flattened_to_2d"
    return True, "supported"


def ensure_grouped_tp_cuda_ext_supported(
    *,
    backend: str,
    sample: torch.Tensor,
    compute_dtype: torch.dtype,
    internal_weights: bool,
    weights: torch.Tensor | None,
) -> bool:
    supported, reason = grouped_tp_cuda_ext_support_reason(
        backend=backend,
        sample=sample,
        compute_dtype=compute_dtype,
        internal_weights=internal_weights,
        weights=weights,
    )
    if supported:
        return True
    if normalize_ictd_tp_backend(backend) == "cuda_ext":
        raise RuntimeError(f"ictd_tp_backend='cuda_ext' is unavailable for this call: {reason}")
    return False


def bucketed_tp_forward(
    *,
    backend: str,
    a: torch.Tensor,
    b: torch.Tensor,
    U_bucket: torch.Tensor,
    W_stack: torch.Tensor,
    gates: Optional[torch.Tensor],
    compute_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if not ensure_grouped_tp_cuda_ext_supported(
        backend=backend,
        sample=a,
        compute_dtype=compute_dtype,
        internal_weights=True,
        weights=gates,
    ):
        return None

    if gates is None:
        gates = a.new_ones((a.shape[0], W_stack.shape[0]), dtype=a.dtype)
    else:
        gates = gates.contiguous()

    if b.shape[1] != 1:
        if normalize_ictd_tp_backend(backend) == "cuda_ext":
            raise RuntimeError("ictd_tp_backend='cuda_ext' is unavailable for this call: mul_in2_not_supported")
        return None

    a_c = a.contiguous()
    b_c = b.contiguous()
    u_c = U_bucket.contiguous()
    w_c = W_stack.contiguous()
    num_paths = int(W_stack.shape[0])

    with _exact_cuda_matmul_mode(a_c):
        if not torch.is_grad_enabled():
            y = _ictd_tp_ext.project_bucket_forward(a_c, b_c, u_c, num_paths)
        else:
            y = _project_forward_op(a_c, b_c, u_c, num_paths)
        out, _ = _mix_forward_op(y, w_c, gates)
    return out


def _project_forward_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    u_bucket: torch.Tensor,
    num_paths: int,
) -> torch.Tensor:
    batch, mul_in1, m1 = a.shape
    _, mul_in2, m2 = b.shape
    pk = u_bucket.shape[1]
    kdim = pk // num_paths
    y = torch.einsum("bim,bjn,mnp->bijp", a, b, u_bucket.view(m1, m2, pk))
    return y.view(batch, mul_in1 * mul_in2, num_paths, kdim).permute(0, 2, 3, 1).contiguous()


def _project_forward_op(
    a: torch.Tensor,
    b: torch.Tensor,
    u_bucket: torch.Tensor,
    num_paths: int,
) -> torch.Tensor:
    if (
        _ictd_tp_ext is not None
        and a.is_cuda
        and b.is_cuda
        and u_bucket.is_cuda
        and b.shape[1] == 1
    ):
        return _ProjectForwardNative.apply(a, b, u_bucket, num_paths)
    return _project_forward_reference(a, b, u_bucket, num_paths)


def _project_transpose_reference(
    grad_y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    u_bucket: torch.Tensor,
    *,
    need_a: bool,
    need_b: bool,
    need_u: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    batch, num_paths, kdim, _ = grad_y.shape
    mul_in1 = a.shape[1]
    mul_in2 = b.shape[1]
    m1 = a.shape[2]
    m2 = b.shape[2]
    pk = num_paths * kdim
    grad_y_5d = grad_y.view(batch, num_paths, kdim, mul_in1, mul_in2)
    grad_y_ijp = grad_y_5d.permute(0, 3, 4, 1, 2).contiguous().view(batch, mul_in1, mul_in2, pk)
    u_3d = u_bucket.view(m1, m2, pk)

    grad_a = torch.einsum("bijp,bjn,mnp->bim", grad_y_ijp, b, u_3d) if need_a else None
    grad_b = torch.einsum("bijp,bim,mnp->bjn", grad_y_ijp, a, u_3d) if need_b else None
    grad_u = (
        torch.einsum("bijp,bim,bjn->mnp", grad_y_ijp, a, b).reshape(m1 * m2, pk)
        if need_u
        else None
    )
    return grad_a, grad_b, grad_u


def _project_transpose_op(
    grad_y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    u_bucket: torch.Tensor,
    *,
    need_a: bool,
    need_b: bool,
    need_u: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    return _project_transpose_reference(
        grad_y,
        a,
        b,
        u_bucket,
        need_a=need_a,
        need_b=need_b,
        need_u=need_u,
    )


def _mix_forward_reference(
    y: torch.Tensor,
    w_stack: torch.Tensor,
    gates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_per = torch.einsum("bpkq,poq->bpok", y, w_stack)
    out = (out_per * gates[:, :, None, None]).sum(dim=1)
    return out, out_per


def _mix_forward_op(
    y: torch.Tensor,
    w_stack: torch.Tensor,
    gates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return _mix_forward_reference(y, w_stack, gates)


def _mix_transpose_reference(
    grad_out: torch.Tensor,
    y: torch.Tensor,
    w_stack: torch.Tensor,
    gates: torch.Tensor,
    *,
    need_y: bool,
    need_w: bool,
    need_g: bool,
    out_per: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    grad_out_gated = grad_out[:, None, :, :] * gates[:, :, None, None]
    grad_y = torch.einsum("bpok,poq->bpkq", grad_out_gated, w_stack) if need_y else None
    grad_w = torch.einsum("bpok,bpkq->poq", grad_out_gated, y) if need_w else None
    grad_g = None
    if need_g:
        if out_per is None:
            out_per = torch.einsum("bpkq,poq->bpok", y, w_stack)
        grad_g = (grad_out[:, None, :, :] * out_per).sum(dim=(2, 3))
    return grad_y, grad_w, grad_g


def _mix_transpose_op(
    grad_out: torch.Tensor,
    y: torch.Tensor,
    w_stack: torch.Tensor,
    gates: torch.Tensor,
    *,
    need_y: bool,
    need_w: bool,
    need_g: bool,
    out_per: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    return _mix_transpose_reference(
        grad_out,
        y,
        w_stack,
        gates,
        need_y=need_y,
        need_w=need_w,
        need_g=need_g,
        out_per=out_per,
    )


class _ProjectForwardNative(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a: torch.Tensor,
        b: torch.Tensor,
        u_bucket: torch.Tensor,
        num_paths: int,
    ) -> torch.Tensor:
        ctx.num_paths = int(num_paths)
        ctx.save_for_backward(a, b, u_bucket)
        return _ictd_tp_ext.project_bucket_forward(a, b, u_bucket, int(num_paths))

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        a, b, u_bucket = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        need_a, need_b, need_u, _ = ctx.needs_input_grad
        grad_a, grad_b, grad_u = _project_transpose_reference(
            grad_out, a, b, u_bucket,
            need_a=need_a, need_b=need_b, need_u=need_u,
        )
        return grad_a, grad_b, grad_u, None
