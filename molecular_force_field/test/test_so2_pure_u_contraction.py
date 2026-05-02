from __future__ import annotations

import pytest
import torch

from molecular_force_field.models.ictd_irreps_2d import so2_irrep_dim
from molecular_force_field.models.pure_cartesian_ictd_fix_so2 import (
    NodeLocalO2SymmetricContraction,
    PureCartesianICTDFixSO2,
)


def _make_local(lmax: int, batch: int, channels: int, dtype: torch.dtype = torch.float64):
    torch.manual_seed(7)
    return {
        (l, m): torch.randn(batch, channels, so2_irrep_dim(m), dtype=dtype)
        for l in range(lmax + 1)
        for m in range(l + 1)
    }


@pytest.mark.parametrize("lmax", [0, 1, 2, 3])
@pytest.mark.parametrize("nu", [1, 2, 3])
def test_strict_pure_u_tensor_shapes(lmax: int, nu: int) -> None:
    contract = NodeLocalO2SymmetricContraction(
        in_channels=4,
        hidden_channels=4,
        lmax=lmax,
        correlation=3,
        max_atomvalue=4,
        contraction_combine="path-free",
    )
    one_copy_dim = sum(2 * l + 1 for l in range(lmax + 1))
    for key in contract.get_active_keys():
        l, m = key
        U = contract.get_u_tensor(nu, key)
        assert U.shape[:-1] == (so2_irrep_dim(m),) + (one_copy_dim,) * nu
        assert U.shape[-1] > 0
        if l == 0 or nu > 1:
            assert U.abs().sum().item() > 0.0


@pytest.mark.parametrize("lmax", [1, 2, 3])
def test_strict_pure_u_paths_respect_so3_triangle(lmax: int) -> None:
    contract = NodeLocalO2SymmetricContraction(
        in_channels=4,
        hidden_channels=4,
        lmax=lmax,
        correlation=2,
        max_atomvalue=4,
        contraction_combine="path-free",
    )
    for l_out, m_out in contract.get_active_keys():
        del m_out
        for l1, l2, l3 in contract.get_u_tensor_paths(2, (l_out, 0)):
            assert l3 == l_out
            assert abs(l1 - l2) <= l3 <= l1 + l2


def test_strict_pure_u_has_odd_scalar_branch() -> None:
    contract = NodeLocalO2SymmetricContraction(
        in_channels=4,
        hidden_channels=4,
        lmax=2,
        correlation=2,
        max_atomvalue=4,
        contraction_combine="path-free",
    )
    assert contract.has_parity_branch()
    U_odd_scalar = contract.get_u_tensor(1, (1, 0))
    assert U_odd_scalar.abs().sum().item() > 0.0


def test_atom_types_change_strict_pure_u_output() -> None:
    torch.manual_seed(0)
    contract = NodeLocalO2SymmetricContraction(
        in_channels=8,
        hidden_channels=8,
        lmax=2,
        correlation=3,
        max_atomvalue=5,
        contraction_combine="path-free",
    ).double()
    local = _make_local(2, batch=4, channels=8)
    out_a = contract(local, torch.tensor([0, 1, 2, 3]))
    out_b = contract(local, torch.tensor([1, 0, 3, 2]))
    diff = max((out_a[k] - out_b[k]).abs().max().item() for k in out_a)
    assert diff > 1e-6


def test_backward_grad_reaches_strict_pure_u_params() -> None:
    contract = NodeLocalO2SymmetricContraction(
        in_channels=8,
        hidden_channels=8,
        lmax=2,
        correlation=3,
        max_atomvalue=5,
        contraction_combine="path-free",
    ).double()
    local = _make_local(2, batch=4, channels=8)
    out = contract(local, torch.tensor([0, 1, 2, 3]))
    loss = sum(v.pow(2).sum() for v in out.values())
    loss.backward()
    grad_names = [
        name for name, p in contract.named_parameters()
        if p.grad is not None and p.grad.abs().max().item() > 0.0
    ]
    assert any("pure_u" in name and "weight" in name for name in grad_names)


def _build_graph(N: int, dtype, seed=11):
    torch.manual_seed(seed)
    pos = torch.randn(N, 3, dtype=dtype) * 1.5
    A = torch.tensor([1, 6, 8, 1, 6, 7, 8, 1][:N])
    batch = torch.zeros(N, dtype=torch.long)
    edges = [
        (i, j)
        for i in range(N)
        for j in range(N)
        if i != j and torch.norm(pos[i] - pos[j]) < 4.0
    ]
    edge_src = torch.tensor([e[0] for e in edges], dtype=torch.long)
    edge_dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
    edge_shifts = torch.zeros(edge_src.numel(), 3, dtype=dtype)
    cell = torch.eye(3, dtype=dtype).unsqueeze(0)
    return pos, A, batch, edge_src, edge_dst, edge_shifts, cell


@pytest.mark.parametrize("route", ["baseline", "fusion"])
def test_full_model_invariance_strict_pure_u(route: str) -> None:
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    try:
        model = PureCartesianICTDFixSO2(
            max_embed_radius=4.0,
            main_max_radius=4.0,
            main_number_of_basis=8,
            hidden_dim_conv=16,
            hidden_dim_sh=None,
            hidden_dim=None,
            num_interaction=3,
            lmax=2,
            main_hidden_sizes3=[32],
            atomic_numbers=[1, 6, 7, 8],
            save_contraction_order=2,
            ictd_fix_route=route,
            ictd_fix_contraction_combine="path-free",
            ictd_fix_interaction_scale="mace-rms",
            device=torch.device("cpu"),
        ).double()
        model.eval()
        pos, A, batch, edge_src, edge_dst, edge_shifts, cell = _build_graph(8, torch.float64)
        with torch.no_grad():
            out_orig = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
            out_trans = model(pos + 1.0, A, batch, edge_src, edge_dst, edge_shifts, cell)
        assert (out_orig - out_trans).abs().max().item() < 1e-10

        torch.manual_seed(42)
        W = torch.randn(3, 3, dtype=torch.float64)
        Q, _ = torch.linalg.qr(W)
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        with torch.no_grad():
            out_rot = model(pos @ Q.T, A, batch, edge_src, edge_dst, edge_shifts, cell)
        assert (out_orig - out_rot).abs().max().item() < 1e-10
    finally:
        torch.set_default_dtype(torch.float32)


def test_pipeline_alignment_node_embedding_then_interactions() -> None:
    model = PureCartesianICTDFixSO2(
        max_embed_radius=4.0,
        main_max_radius=4.0,
        main_number_of_basis=8,
        hidden_dim_conv=8,
        hidden_dim_sh=None,
        hidden_dim=None,
        num_interaction=3,
        lmax=2,
        main_hidden_sizes3=[16],
        atomic_numbers=[1, 6, 7, 8],
        save_contraction_order=2,
        ictd_fix_route="baseline",
        ictd_fix_contraction_combine="path-free",
        device=torch.device("cpu"),
    )
    assert isinstance(model.node_embedding, torch.nn.Linear)
    assert len(model.interactions) == model.num_interaction
    assert model.interactions[0].input_lmax == 0
    assert model.interactions[-1].target_lmax == 0
    assert not hasattr(model, "local_conv_emb")
    assert not hasattr(model, "post_local_linears")
