from __future__ import annotations

import pytest
import torch

from molecular_force_field.models import ictd_irreps_cuda
from molecular_force_field.models.ictd_irreps import (
    HarmonicFullyConnectedTensorProduct,
    HarmonicFullyConnectedTensorProductO3,
)
from molecular_force_field.models.ictd_irreps_cuda import (
    _mix_forward_reference,
    _mix_forward_op,
    _mix_transpose_op,
    _mix_transpose_reference,
    _project_forward_op,
    _project_forward_reference,
    _project_transpose_op,
    _project_transpose_reference,
)
from molecular_force_field.models.pure_cartesian_ictd_layers import PureCartesianICTDTransformerLayer
from molecular_force_field.models.pure_cartesian_ictd_layers_o3 import PureCartesianICTDO3TransformerLayer


def _make_so3_inputs(batch: int, mul1: int, mul2: int, lmax: int, dtype: torch.dtype):
    g = torch.Generator().manual_seed(1234)
    x1 = {
        l: torch.randn(batch, mul1, 2 * l + 1, generator=g, dtype=dtype)
        for l in range(lmax + 1)
    }
    x2 = {
        l: torch.randn(batch, mul2, 2 * l + 1, generator=g, dtype=dtype)
        for l in range(lmax + 1)
    }
    return x1, x2


def _make_o3_inputs(
    batch: int,
    mul1: int,
    mul2: int,
    active_irreps: list[tuple[int, int]],
    dtype: torch.dtype,
):
    g = torch.Generator().manual_seed(4321)
    x1 = {
        key: torch.randn(batch, mul1, 2 * key[0] + 1, generator=g, dtype=dtype)
        for key in active_irreps
    }
    x2 = {
        key: torch.randn(batch, mul2, 2 * key[0] + 1, generator=g, dtype=dtype)
        for key in active_irreps
    }
    return x1, x2


def test_so3_tp_auto_backend_falls_back_to_pytorch_on_cpu():
    lmax = 2
    mul1 = 4
    mul2 = 1
    mul_out = 3
    batch = 7
    dtype = torch.float32
    x1, x2 = _make_so3_inputs(batch, mul1, mul2, lmax, dtype)

    tp_ref = HarmonicFullyConnectedTensorProduct(
        mul_in1=mul1,
        mul_in2=mul2,
        mul_out=mul_out,
        lmax=lmax,
        internal_weights=True,
        internal_compute_dtype=dtype,
        ictd_tp_backend="pytorch",
    )
    tp_auto = HarmonicFullyConnectedTensorProduct(
        mul_in1=mul1,
        mul_in2=mul2,
        mul_out=mul_out,
        lmax=lmax,
        internal_weights=True,
        internal_compute_dtype=dtype,
        ictd_tp_backend="auto",
    )
    tp_auto.load_state_dict(tp_ref.state_dict())

    g = torch.Generator().manual_seed(99)
    gates = torch.randn(batch, tp_ref.num_paths, generator=g, dtype=dtype)
    out_ref = tp_ref(x1, x2, gates)
    out_auto = tp_auto(x1, x2, gates)
    for l in range(lmax + 1):
        torch.testing.assert_close(out_auto[l], out_ref[l], rtol=1e-5, atol=1e-6)


def test_o3_tp_auto_backend_falls_back_to_pytorch_on_cpu():
    active_irreps = [(0, 1), (1, -1), (2, 1)]
    lmax = 2
    mul1 = 3
    mul2 = 1
    mul_out = 2
    batch = 5
    dtype = torch.float32
    x1, x2 = _make_o3_inputs(batch, mul1, mul2, active_irreps, dtype)

    tp_ref = HarmonicFullyConnectedTensorProductO3(
        mul_in1=mul1,
        mul_in2=mul2,
        mul_out=mul_out,
        lmax=lmax,
        active_irreps=active_irreps,
        internal_weights=True,
        internal_compute_dtype=dtype,
        ictd_tp_backend="pytorch",
    )
    tp_auto = HarmonicFullyConnectedTensorProductO3(
        mul_in1=mul1,
        mul_in2=mul2,
        mul_out=mul_out,
        lmax=lmax,
        active_irreps=active_irreps,
        internal_weights=True,
        internal_compute_dtype=dtype,
        ictd_tp_backend="auto",
    )
    tp_auto.load_state_dict(tp_ref.state_dict())

    g = torch.Generator().manual_seed(77)
    gates = torch.randn(batch, tp_ref.num_paths, generator=g, dtype=dtype)
    out_ref = tp_ref(x1, x2, gates)
    out_auto = tp_auto(x1, x2, gates)
    for key in active_irreps:
        torch.testing.assert_close(out_auto[key], out_ref[key], rtol=1e-5, atol=1e-6)


def test_explicit_cuda_ext_backend_raises_on_unsupported_cpu_path():
    tp = HarmonicFullyConnectedTensorProduct(
        mul_in1=2,
        mul_in2=1,
        mul_out=2,
        lmax=1,
        internal_weights=True,
        internal_compute_dtype=torch.float32,
        ictd_tp_backend="cuda_ext",
    )
    x1, x2 = _make_so3_inputs(batch=3, mul1=2, mul2=1, lmax=1, dtype=torch.float32)
    gates = torch.randn(3, tp.num_paths, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="cuda_ext"):
        tp(x1, x2, gates)


def test_auto_backend_requires_experimental_env_before_considering_cuda(monkeypatch: pytest.MonkeyPatch):
    class _StubExt:
        @staticmethod
        def has_cuda() -> bool:
            return True

        @staticmethod
        def _unexpected(*args, **kwargs):
            raise AssertionError("extension op should not be called in support probing")

        project_bucket_forward = _unexpected
        project_bucket_transpose_a = _unexpected
        project_bucket_transpose_b = _unexpected
        project_bucket_transpose_u = _unexpected
        mix_bucket_forward = _unexpected
        mix_bucket_transpose_y = _unexpected
        mix_bucket_transpose_w = _unexpected
        mix_bucket_transpose_g = _unexpected
        bucketed_tp_forward = _unexpected

    monkeypatch.setattr(ictd_irreps_cuda, "_ictd_tp_ext", _StubExt())
    sample = torch.zeros(2, 3, 3, dtype=torch.float32)

    monkeypatch.delenv("ICTD_ENABLE_EXPERIMENTAL_CUDA_TP", raising=False)
    supported, reason = ictd_irreps_cuda.grouped_tp_cuda_ext_support_reason(
        backend="auto",
        sample=sample,
        compute_dtype=torch.float32,
        internal_weights=True,
        weights=None,
    )
    assert supported is False
    assert reason == "auto_experimental_cuda_disabled"

    monkeypatch.setenv("ICTD_ENABLE_EXPERIMENTAL_CUDA_TP", "1")
    supported, reason = ictd_irreps_cuda.grouped_tp_cuda_ext_support_reason(
        backend="auto",
        sample=sample,
        compute_dtype=torch.float32,
        internal_weights=True,
        weights=None,
    )
    assert supported is False
    assert reason == "sample_not_on_cuda"


def test_fp64_is_a_supported_cuda_dtype_in_principle(monkeypatch: pytest.MonkeyPatch):
    class _StubExt:
        @staticmethod
        def has_cuda() -> bool:
            return True

        @staticmethod
        def _unexpected(*args, **kwargs):
            raise AssertionError("extension op should not be called in support probing")

        project_bucket_forward = _unexpected
        project_bucket_transpose_a = _unexpected
        project_bucket_transpose_b = _unexpected
        project_bucket_transpose_u = _unexpected
        mix_bucket_forward = _unexpected
        mix_bucket_transpose_y = _unexpected
        mix_bucket_transpose_w = _unexpected
        mix_bucket_transpose_g = _unexpected
        bucketed_tp_forward = _unexpected

    monkeypatch.setattr(ictd_irreps_cuda, "_ictd_tp_ext", _StubExt())
    sample = torch.zeros(2, 3, 3, dtype=torch.float64, device="cpu")
    monkeypatch.setenv("ICTD_ENABLE_EXPERIMENTAL_CUDA_TP", "1")
    supported, reason = ictd_irreps_cuda.grouped_tp_cuda_ext_support_reason(
        backend="auto",
        sample=sample,
        compute_dtype=torch.float64,
        internal_weights=True,
        weights=None,
    )
    assert supported is False
    assert reason == "sample_not_on_cuda"


def test_model_constructors_thread_ictd_tp_backend():
    so3_model = PureCartesianICTDTransformerLayer(
        max_embed_radius=4.0,
        main_max_radius=4.0,
        main_number_of_basis=4,
        hidden_dim_conv=4,
        hidden_dim_sh=4,
        hidden_dim=16,
        num_interaction=2,
        lmax=2,
        internal_compute_dtype=torch.float32,
        ictd_tp_backend="pytorch",
    )
    assert so3_model.e3_conv_emb.tp2.ictd_tp_backend == "pytorch"
    assert so3_model.tp2_layers[0].ictd_tp_backend == "pytorch"

    o3_model = PureCartesianICTDO3TransformerLayer(
        max_embed_radius=4.0,
        main_max_radius=4.0,
        main_number_of_basis=4,
        hidden_dim_conv=4,
        hidden_dim_sh=4,
        hidden_dim=16,
        num_interaction=2,
        lmax=2,
        internal_compute_dtype=torch.float32,
        ictd_tp_backend="pytorch",
    )
    assert o3_model.e3_conv_emb.tp2.ictd_tp_backend == "pytorch"
    assert o3_model.tp2_layers[0].ictd_tp_backend == "pytorch"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reference_project_mix_operator_family_supports_double_backward(dtype):
    B = 2
    mul_in1 = 2
    mul_in2 = 1
    mul_out = 3
    m1 = 3
    m2 = 3
    num_paths = 2
    kdim = 2

    g = torch.Generator().manual_seed(2027)
    a = torch.randn(B, mul_in1, m1, generator=g, dtype=dtype, requires_grad=True)
    b = torch.randn(B, mul_in2, m2, generator=g, dtype=dtype, requires_grad=True)
    U = torch.randn(m1 * m2, num_paths * kdim, generator=g, dtype=dtype, requires_grad=True)
    W = torch.randn(num_paths, mul_out, mul_in1 * mul_in2, generator=g, dtype=dtype, requires_grad=True)
    gates = torch.randn(B, num_paths, generator=g, dtype=dtype, requires_grad=True)

    y = _project_forward_reference(a, b, U, num_paths)
    out, out_per = _mix_forward_reference(y, W, gates)
    loss = out.square().sum()

    grad_out = torch.autograd.grad(loss, out, create_graph=True)[0]
    grad_y, grad_w, grad_g = _mix_transpose_reference(
        grad_out, y, W, gates, need_y=True, need_w=True, need_g=True, out_per=out_per
    )
    grad_a, grad_b, grad_u = _project_transpose_reference(
        grad_y, a, b, U, need_a=True, need_b=True, need_u=True
    )

    second = torch.autograd.grad(
        grad_a.sum() + grad_b.sum() + grad_u.sum() + grad_w.sum() + grad_g.sum(),
        (a, b, U, W, gates),
        allow_unused=False,
    )
    assert all(t is not None for t in second)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 1e-5, 1e-6),
        (torch.float64, 1e-10, 1e-12),
    ],
)
@pytest.mark.skipif(
    not ictd_irreps_cuda.cuda_ext_available(),
    reason="ICTD TP extension not built",
)
def test_bucketed_tp_extension_supports_grad_and_double_backward(dtype, rtol, atol):
    B = 3
    mul_in1 = 2
    mul_in2 = 1
    mul_out = 4
    m1 = 3
    m2 = 3
    num_paths = 2
    kdim = 3

    g = torch.Generator().manual_seed(2026)
    device = torch.device("cuda" if ictd_irreps_cuda.cuda_ext_has_cuda() else "cpu")
    a = torch.randn(B, mul_in1, m1, generator=g, dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(B, mul_in2, m2, generator=g, dtype=dtype, device=device, requires_grad=True)
    U = torch.randn(m1 * m2, num_paths * kdim, generator=g, dtype=dtype, device=device, requires_grad=True)
    W = torch.randn(num_paths, mul_out, mul_in1 * mul_in2, generator=g, dtype=dtype, device=device, requires_grad=True)
    gates = torch.randn(B, num_paths, generator=g, dtype=dtype, device=device, requires_grad=True)

    y_ext = _project_forward_op(a, b, U, num_paths)
    out_ext, out_per_ext = _mix_forward_op(y_ext, W, gates)

    y = torch.einsum("bim,bjn,mnk->bijk", a, b, U.view(m1, m2, num_paths * kdim))
    y = y.view(B, mul_in1 * mul_in2, num_paths, kdim).permute(0, 2, 3, 1).contiguous()
    out_per = torch.einsum("bpki,pio->bpok", y, W.permute(0, 2, 1).contiguous())
    out_ref = (out_per * gates.view(B, num_paths, 1, 1)).sum(dim=1)

    torch.testing.assert_close(y_ext, y, rtol=rtol, atol=atol)
    torch.testing.assert_close(out_ext, out_ref, rtol=rtol, atol=atol)
    if out_per_ext is not None:
        torch.testing.assert_close(out_per_ext, out_per, rtol=rtol, atol=atol)

    loss = out_ext.square().sum()
    grad_a, grad_b, grad_u, grad_w, grad_g = torch.autograd.grad(
        loss, (a, b, U, W, gates), create_graph=True
    )
    second = torch.autograd.grad(
        grad_a.sum() + grad_b.sum() + grad_u.sum() + grad_w.sum() + grad_g.sum(),
        (a, b, U, W, gates),
        allow_unused=False,
    )
    assert all(t is not None for t in second)

    grad_out = torch.autograd.grad(loss, out_ext, create_graph=True)[0]
    grad_y_ext, grad_w_ext, grad_g_ext = _mix_transpose_op(
        grad_out, y_ext, W, gates, need_y=True, need_w=True, need_g=True, out_per=out_per_ext
    )
    grad_y_ref, grad_w_ref, grad_g_ref = _mix_transpose_reference(
        grad_out, y, W, gates, need_y=True, need_w=True, need_g=True, out_per=out_per
    )
    torch.testing.assert_close(grad_y_ext, grad_y_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(grad_w_ext, grad_w_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(grad_g_ext, grad_g_ref, rtol=rtol, atol=atol)

    grad_a_ext, grad_b_ext, grad_u_ext = _project_transpose_op(
        grad_y_ext, a, b, U, need_a=True, need_b=True, need_u=True
    )
    grad_a_ref, grad_b_ref, grad_u_ref = _project_transpose_reference(
        grad_y_ref, a, b, U, need_a=True, need_b=True, need_u=True
    )
    torch.testing.assert_close(grad_a_ext, grad_a_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(grad_b_ext, grad_b_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(grad_u_ext, grad_u_ref, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Native operator family parity tests
# ---------------------------------------------------------------------------

_CUDA_EXT_SKIP = pytest.mark.skipif(
    not ictd_irreps_cuda.cuda_ext_available() or not ictd_irreps_cuda.cuda_ext_has_cuda(),
    reason="ICTD TP CUDA extension not built or no CUDA",
)

_OP_PARITY_PARAMS = pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 1e-5, 1e-6),
        (torch.float64, 1e-10, 1e-12),
    ],
)


def _tp_inputs_cuda(
    B: int,
    mul_in1: int,
    mul_in2: int,
    mul_out: int,
    m1: int,
    m2: int,
    num_paths: int,
    kdim: int,
    dtype: torch.dtype,
    requires_grad: bool = False,
    seed: int = 42,
):
    g = torch.Generator().manual_seed(seed)
    device = torch.device("cuda")
    a = torch.randn(B, mul_in1, m1, generator=g, dtype=dtype, device=device, requires_grad=requires_grad)
    b = torch.randn(B, mul_in2, m2, generator=g, dtype=dtype, device=device, requires_grad=requires_grad)
    U = torch.randn(m1 * m2, num_paths * kdim, generator=g, dtype=dtype, device=device, requires_grad=requires_grad)
    W = torch.randn(num_paths, mul_out, mul_in1 * mul_in2, generator=g, dtype=dtype, device=device, requires_grad=requires_grad)
    gates = torch.randn(B, num_paths, generator=g, dtype=dtype, device=device, requires_grad=requires_grad)
    return a, b, U, W, gates


@_CUDA_EXT_SKIP
@_OP_PARITY_PARAMS
def test_native_project_forward_vs_reference(dtype, rtol, atol):
    B, I, m1, m2, P, K = 4, 3, 3, 3, 2, 3
    a, b, U, W, gates = _tp_inputs_cuda(B, I, 1, 4, m1, m2, P, K, dtype, seed=100)
    y_native = _project_forward_op(a, b, U, P)
    y_ref = _project_forward_reference(a, b, U, P)
    torch.testing.assert_close(y_native, y_ref, rtol=rtol, atol=atol)


@_CUDA_EXT_SKIP
@_OP_PARITY_PARAMS
def test_native_project_adjoint_family_vs_reference(dtype, rtol, atol):
    B, I, m1, m2, P, K = 4, 3, 3, 3, 2, 3
    a, b, U, W, gates = _tp_inputs_cuda(B, I, 1, 4, m1, m2, P, K, dtype, seed=101)
    y_ref = _project_forward_reference(a, b, U, P)
    grad_y = torch.randn_like(y_ref)

    ga_nat, gb_nat, gu_nat = _project_transpose_op(
        grad_y, a, b, U, need_a=True, need_b=True, need_u=True,
    )
    ga_ref, gb_ref, gu_ref = _project_transpose_reference(
        grad_y, a, b, U, need_a=True, need_b=True, need_u=True,
    )
    torch.testing.assert_close(ga_nat, ga_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(gb_nat, gb_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(gu_nat, gu_ref, rtol=rtol, atol=atol)


@_CUDA_EXT_SKIP
@_OP_PARITY_PARAMS
def test_native_mix_forward_vs_reference(dtype, rtol, atol):
    B, I, P, K, O = 4, 3, 2, 3, 4
    g_gen = torch.Generator().manual_seed(102)
    device = torch.device("cuda")
    y = torch.randn(B, P, K, I, generator=g_gen, dtype=dtype, device=device)
    w = torch.randn(P, O, I, generator=g_gen, dtype=dtype, device=device)
    gates = torch.randn(B, P, generator=g_gen, dtype=dtype, device=device)

    out_native, _ = _mix_forward_op(y, w, gates)
    out_ref, _ = _mix_forward_reference(y, w, gates)
    torch.testing.assert_close(out_native, out_ref, rtol=rtol, atol=atol)


@_CUDA_EXT_SKIP
@_OP_PARITY_PARAMS
def test_native_mix_adjoint_family_vs_reference(dtype, rtol, atol):
    B, I, P, K, O = 4, 3, 2, 3, 4
    g_gen = torch.Generator().manual_seed(103)
    device = torch.device("cuda")
    y = torch.randn(B, P, K, I, generator=g_gen, dtype=dtype, device=device)
    w = torch.randn(P, O, I, generator=g_gen, dtype=dtype, device=device)
    gates = torch.randn(B, P, generator=g_gen, dtype=dtype, device=device)
    grad_out = torch.randn(B, O, K, generator=g_gen, dtype=dtype, device=device)

    gy_nat, gw_nat, gg_nat = _mix_transpose_op(
        grad_out, y, w, gates, need_y=True, need_w=True, need_g=True,
    )
    gy_ref, gw_ref, gg_ref = _mix_transpose_reference(
        grad_out, y, w, gates, need_y=True, need_w=True, need_g=True,
    )
    torch.testing.assert_close(gy_nat, gy_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(gw_nat, gw_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(gg_nat, gg_ref, rtol=rtol, atol=atol)


@_CUDA_EXT_SKIP
@_OP_PARITY_PARAMS
def test_native_first_order_derivative_parity(dtype, rtol, atol):
    """First-order gradients of a, b, U, W, g from native TP match reference."""
    B, I, m1, m2, P, K, O = 3, 2, 3, 3, 2, 3, 4
    a, b, U, W, gates = _tp_inputs_cuda(B, I, 1, O, m1, m2, P, K, dtype, requires_grad=True, seed=200)

    y_nat = _project_forward_op(a, b, U, P)
    out_nat, _ = _mix_forward_op(y_nat, W, gates)
    loss_nat = out_nat.square().sum()
    grads_nat = torch.autograd.grad(loss_nat, (a, b, U, W, gates), create_graph=False)

    a2 = a.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    U2 = U.detach().clone().requires_grad_(True)
    W2 = W.detach().clone().requires_grad_(True)
    g2 = gates.detach().clone().requires_grad_(True)

    y_ref = _project_forward_reference(a2, b2, U2, P)
    out_ref, _ = _mix_forward_reference(y_ref, W2, g2)
    loss_ref = out_ref.square().sum()
    grads_ref = torch.autograd.grad(loss_ref, (a2, b2, U2, W2, g2), create_graph=False)

    for gn, gr in zip(grads_nat, grads_ref):
        torch.testing.assert_close(gn, gr, rtol=rtol, atol=atol)


@_CUDA_EXT_SKIP
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 5e-5, 5e-6),
        (torch.float64, 1e-10, 1e-12),
    ],
)
def test_native_double_backward_parity(dtype, rtol, atol):
    """Double-backward (create_graph) from native TP matches reference."""
    B, I, m1, m2, P, K, O = 3, 2, 3, 3, 2, 3, 4
    a, b, U, W, gates = _tp_inputs_cuda(B, I, 1, O, m1, m2, P, K, dtype, requires_grad=True, seed=300)

    y_nat = _project_forward_op(a, b, U, P)
    out_nat, _ = _mix_forward_op(y_nat, W, gates)
    loss_nat = out_nat.square().sum()
    first_nat = torch.autograd.grad(loss_nat, (a, b, U, W, gates), create_graph=True)
    dbl_nat = torch.autograd.grad(
        sum(g.sum() for g in first_nat),
        (a, b, U, W, gates),
        allow_unused=False,
    )

    a2 = a.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    U2 = U.detach().clone().requires_grad_(True)
    W2 = W.detach().clone().requires_grad_(True)
    g2 = gates.detach().clone().requires_grad_(True)

    y_ref = _project_forward_reference(a2, b2, U2, P)
    out_ref, _ = _mix_forward_reference(y_ref, W2, g2)
    loss_ref = out_ref.square().sum()
    first_ref = torch.autograd.grad(loss_ref, (a2, b2, U2, W2, g2), create_graph=True)
    dbl_ref = torch.autograd.grad(
        sum(g.sum() for g in first_ref),
        (a2, b2, U2, W2, g2),
        allow_unused=False,
    )

    for gn, gr in zip(dbl_nat, dbl_ref):
        assert gn is not None and gr is not None
        torch.testing.assert_close(gn, gr, rtol=rtol, atol=atol)


@_CUDA_EXT_SKIP
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 5e-5, 5e-6),
        (torch.float64, 1e-10, 1e-12),
    ],
)
def test_native_force_loss_parity(dtype, rtol, atol):
    """Force-loss path (energy → forces via create_graph → loss.backward) parity."""
    B, I, m1, m2, P, K, O = 3, 2, 3, 3, 2, 3, 4
    a, b, U, W, gates = _tp_inputs_cuda(B, I, 1, O, m1, m2, P, K, dtype, requires_grad=True, seed=400)

    y_nat = _project_forward_op(a, b, U, P)
    out_nat, _ = _mix_forward_op(y_nat, W, gates)
    energy_nat = out_nat.sum()
    force_nat = -torch.autograd.grad(energy_nat, a, create_graph=True, retain_graph=True)[0]
    force_loss_nat = force_nat.square().mean()
    param_grads_nat = torch.autograd.grad(
        force_loss_nat, (a, b, U, W, gates), allow_unused=True,
    )

    a2 = a.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    U2 = U.detach().clone().requires_grad_(True)
    W2 = W.detach().clone().requires_grad_(True)
    g2 = gates.detach().clone().requires_grad_(True)

    y_ref = _project_forward_reference(a2, b2, U2, P)
    out_ref, _ = _mix_forward_reference(y_ref, W2, g2)
    energy_ref = out_ref.sum()
    force_ref = -torch.autograd.grad(energy_ref, a2, create_graph=True, retain_graph=True)[0]
    force_loss_ref = force_ref.square().mean()
    param_grads_ref = torch.autograd.grad(
        force_loss_ref, (a2, b2, U2, W2, g2), allow_unused=True,
    )

    torch.testing.assert_close(force_nat, force_ref, rtol=rtol, atol=atol)
    for gn, gr in zip(param_grads_nat, param_grads_ref):
        if gn is None and gr is None:
            continue
        assert gn is not None and gr is not None
        torch.testing.assert_close(gn, gr, rtol=rtol, atol=atol)


@_CUDA_EXT_SKIP
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 5e-5, 5e-6),
        (torch.float64, 1e-10, 1e-12),
    ],
)
def test_fused_project_dbl_bwd_kernels_vs_reference(dtype, rtol, atol):
    """Each fused Project double-backward kernel matches reference computed via autograd."""
    B, I, m1, m2, P, K = 3, 2, 3, 3, 2, 3
    g = torch.Generator().manual_seed(500)
    device = torch.device("cuda")
    h = torch.randn(B, P, K, I, generator=g, dtype=dtype, device=device)
    a = torch.randn(B, I, m1, generator=g, dtype=dtype, device=device, requires_grad=True)
    b = torch.randn(B, 1, m2, generator=g, dtype=dtype, device=device, requires_grad=True)
    U = torch.randn(m1 * m2, P * K, generator=g, dtype=dtype, device=device, requires_grad=True)

    ga_ref = _project_transpose_reference(h, a, b, U, need_a=True, need_b=False, need_u=False)[0]
    gb_ref = _project_transpose_reference(h, a, b, U, need_a=False, need_b=True, need_u=False)[1]
    gu_ref = _project_transpose_reference(h, a, b, U, need_a=False, need_b=False, need_u=True)[2]

    gga = torch.randn_like(ga_ref)
    ggb = torch.randn_like(gb_ref)
    ggu = torch.randn_like(gu_ref)

    from molecular_force_field.models.ictd_irreps_cuda import _ictd_tp_ext

    nat_gh = _ictd_tp_ext.project_dbl_bwd_grad_h(a, gga, b, ggb, U, ggu, P)
    nat_ga = _ictd_tp_ext.project_dbl_bwd_grad_a(h, b, ggb, U, ggu)
    nat_gb = _ictd_tp_ext.project_dbl_bwd_grad_b(h, a, gga, U, ggu)
    nat_gu = _ictd_tp_ext.project_dbl_bwd_grad_u(h, a, gga, b, ggb)

    ref_gh = (
        _project_forward_reference(gga, b, U, P)
        + _project_forward_reference(a, ggb, U, P)
        + _project_forward_reference(a, b, ggu, P)
    )
    ref_ga = (
        _project_transpose_reference(h, a, ggb, U, need_a=True, need_b=False, need_u=False)[0]
        + _project_transpose_reference(h, a, b, ggu, need_a=True, need_b=False, need_u=False)[0]
    )
    ref_gb = (
        _project_transpose_reference(h, gga, b, U, need_a=False, need_b=True, need_u=False)[1]
        + _project_transpose_reference(h, a, b, ggu, need_a=False, need_b=True, need_u=False)[1]
    )
    ref_gu = (
        _project_transpose_reference(h, gga, b, U, need_a=False, need_b=False, need_u=True)[2]
        + _project_transpose_reference(h, a, ggb, U, need_a=False, need_b=False, need_u=True)[2]
    )

    torch.testing.assert_close(nat_gh, ref_gh, rtol=rtol, atol=atol)
    torch.testing.assert_close(nat_ga, ref_ga, rtol=rtol, atol=atol)
    torch.testing.assert_close(nat_gb, ref_gb, rtol=rtol, atol=atol)
    torch.testing.assert_close(nat_gu, ref_gu, rtol=rtol, atol=atol)


@_CUDA_EXT_SKIP
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 5e-5, 5e-6),
        (torch.float64, 1e-10, 1e-12),
    ],
)
def test_fused_mix_dbl_bwd_kernels_vs_reference(dtype, rtol, atol):
    """Each fused Mix double-backward kernel matches reference computed via autograd."""
    B, I, P, K, O = 3, 2, 2, 3, 4
    g_gen = torch.Generator().manual_seed(600)
    device = torch.device("cuda")
    g_out = torch.randn(B, O, K, generator=g_gen, dtype=dtype, device=device)
    y = torch.randn(B, P, K, I, generator=g_gen, dtype=dtype, device=device)
    w = torch.randn(P, O, I, generator=g_gen, dtype=dtype, device=device)
    gates = torch.randn(B, P, generator=g_gen, dtype=dtype, device=device)

    ggy = torch.randn_like(y)
    ggw = torch.randn_like(w)
    ggg = torch.randn_like(gates)

    from molecular_force_field.models.ictd_irreps_cuda import _ictd_tp_ext

    nat_go = _ictd_tp_ext.mix_dbl_bwd_grad_g_out(y, ggy, w, ggw, gates, ggg)
    nat_y2 = _ictd_tp_ext.mix_dbl_bwd_grad_y(g_out, w, ggw, gates, ggg)
    nat_w2 = _ictd_tp_ext.mix_dbl_bwd_grad_w(g_out, y, ggy, gates, ggg)
    nat_g2 = _ictd_tp_ext.mix_dbl_bwd_grad_g(g_out, y, ggy, w, ggw)

    ref_go, _ = _mix_forward_reference(ggy, w, gates)
    tmp, _ = _mix_forward_reference(y, ggw, gates)
    ref_go = ref_go + tmp
    tmp, _ = _mix_forward_reference(y, w, ggg)
    ref_go = ref_go + tmp

    ref_y2 = (
        _mix_transpose_reference(g_out, y, ggw, gates, need_y=True, need_w=False, need_g=False)[0]
        + _mix_transpose_reference(g_out, y, w, ggg, need_y=True, need_w=False, need_g=False)[0]
    )
    ref_w2 = (
        _mix_transpose_reference(g_out, ggy, w, gates, need_y=False, need_w=True, need_g=False)[1]
        + _mix_transpose_reference(g_out, y, w, ggg, need_y=False, need_w=True, need_g=False)[1]
    )
    ref_g2 = (
        _mix_transpose_reference(g_out, ggy, w, gates, need_y=False, need_w=False, need_g=True)[2]
        + _mix_transpose_reference(g_out, y, ggw, gates, need_y=False, need_w=False, need_g=True)[2]
    )

    torch.testing.assert_close(nat_go, ref_go, rtol=rtol, atol=atol)
    torch.testing.assert_close(nat_y2, ref_y2, rtol=rtol, atol=atol)
    torch.testing.assert_close(nat_w2, ref_w2, rtol=rtol, atol=atol)
    torch.testing.assert_close(nat_g2, ref_g2, rtol=rtol, atol=atol)
