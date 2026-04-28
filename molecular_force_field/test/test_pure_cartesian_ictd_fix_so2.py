from __future__ import annotations

import pytest
import torch

from molecular_force_field.models.pure_cartesian_ictd_fix_so2 import (
    NodeLocalO2SymmetricContraction,
    PureCartesianICTDFixSO2,
)


def _make_dummy_graph(
    device: torch.device,
    dtype: torch.dtype,
    *,
    num_nodes: int = 48,
    avg_degree: int = 12,
    seed: int = 123,
):
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


def _run_model_smoke(device: torch.device, route: str):
    dtype = torch.float32
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = _make_dummy_graph(device, dtype)
    pos = pos.requires_grad_(True)
    model = PureCartesianICTDFixSO2(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=16,
        hidden_dim_sh=16,
        hidden_dim=16,
        lmax=2,
        num_interaction=3,
        ictd_fix_route=route,
        save_contraction_order=3,
        internal_compute_dtype=None,
    ).to(device=device, dtype=dtype)
    out = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    assert out.shape == (pos.shape[0], 1)
    loss = out.square().mean()
    loss.backward()
    assert pos.grad is not None
    assert torch.isfinite(out).all()
    assert torch.isfinite(pos.grad).all()

    out2, feats = model(
        pos.detach(),
        A,
        batch,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        return_combined_features=True,
    )
    assert out2.shape == (pos.shape[0], 1)
    assert feats.shape[0] == pos.shape[0]
    assert torch.isfinite(feats).all()


def test_pure_cartesian_ictd_fix_so2_cpu_smoke() -> None:
    _run_model_smoke(torch.device("cpu"), "baseline")
    _run_model_smoke(torch.device("cpu"), "fusion")


def test_fix_so2_contraction_uses_o2_complete_stack() -> None:
    model = PureCartesianICTDFixSO2(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=16,
        hidden_dim_sh=16,
        hidden_dim=16,
        lmax=2,
        num_interaction=3,
        ictd_fix_route="fusion",
        save_contraction_order=3,
        internal_compute_dtype=None,
    )
    assert isinstance(model.products[-1].symmetric_contractions, NodeLocalO2SymmetricContraction)
    assert isinstance(model.multiple_contraction_mix.symmetric_contractions, NodeLocalO2SymmetricContraction)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pure_cartesian_ictd_fix_so2_cuda_smoke() -> None:
    _run_model_smoke(torch.device("cuda"), "baseline")
    _run_model_smoke(torch.device("cuda"), "fusion")
