from __future__ import annotations

from pathlib import Path

import pytest
import torch

from molecular_force_field.models.pure_cartesian_ictd_layers import (
    PureCartesianICTDTransformerLayer,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_o2 import (
    PureCartesianICTDO2TransformerLayer,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_so2 import (
    HarmonicLocalO2ContractionTensorProduct,
    LocalMultipleContractionSO2,
    PureCartesianICTDSO2TransformerLayer,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_so2 import (
    LocalMultipleContractionO2,
)


def _make_dummy_graph(
    device: torch.device,
    dtype: torch.dtype,
    *,
    num_nodes: int = 48,
    avg_degree: int = 12,
    seed: int = 42,
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


def _run_smoke(model_cls, *, save_readout_mode: str):
    device = torch.device("cpu")
    dtype = torch.float32
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = _make_dummy_graph(device, dtype)
    pos = pos.requires_grad_(True)

    model = model_cls(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=16,
        hidden_dim_sh=16,
        hidden_dim=16,
        lmax=2,
        num_interaction=3,
        save_readout_mode=save_readout_mode,
        internal_compute_dtype=torch.float64,
    ).to(device=device, dtype=dtype)

    out = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    assert out.shape == (pos.shape[0], 1)
    loss = out.sum()
    loss.backward()
    assert pos.grad is not None
    assert torch.isfinite(out).all()
    assert torch.isfinite(pos.grad).all()

    out_sync = model(
        pos.detach(),
        A,
        batch,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        sync_after_scatter=lambda x: x,
    )
    assert out_sync.shape == out.shape
    return model


def _run_smoke_on_device(model_cls, device: torch.device, *, save_readout_mode: str):
    dtype = torch.float32
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = _make_dummy_graph(device, dtype)
    pos = pos.requires_grad_(True)
    model = model_cls(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=16,
        hidden_dim_sh=16,
        hidden_dim=16,
        lmax=2,
        num_interaction=3,
        save_readout_mode=save_readout_mode,
        internal_compute_dtype=torch.float64,
    ).to(device=device, dtype=dtype)
    out = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    loss = out.square().mean()
    loss.backward()
    assert out.shape == (pos.shape[0], 1)
    assert pos.grad is not None
    assert torch.isfinite(out).all()
    assert torch.isfinite(pos.grad).all()
    return model


def test_pure_cartesian_ictd_so2_smoke() -> None:
    model = _run_smoke(PureCartesianICTDSO2TransformerLayer, save_readout_mode="multiple-contraction")
    assert isinstance(model.multiple_contraction_last, LocalMultipleContractionSO2)


def test_pure_cartesian_ictd_o2_smoke() -> None:
    model = _run_smoke(PureCartesianICTDO2TransformerLayer, save_readout_mode="multiple-contraction")
    assert isinstance(model.multiple_contraction_last, LocalMultipleContractionO2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_so3_so2_o2_cuda_smoke() -> None:
    device = torch.device("cuda")
    _run_smoke_on_device(PureCartesianICTDTransformerLayer, device, save_readout_mode="multiple-contraction")
    _run_smoke_on_device(PureCartesianICTDSO2TransformerLayer, device, save_readout_mode="multiple-contraction")
    _run_smoke_on_device(PureCartesianICTDO2TransformerLayer, device, save_readout_mode="multiple-contraction")


def test_no_e3nn_runtime_rotation_residuals_in_so2_module() -> None:
    src = Path("/home/rebuild/molecular_force_field/models/pure_cartesian_ictd_layers_so2.py").read_text(encoding="utf-8")
    forbidden = [
        "D_from_angles",
        "D_from_matrix",
        "spherical_harmonics",
        "_ictd_to_e3",
        "_e3_to_ictd",
        "_stable_e3nn_angles",
    ]
    for token in forbidden:
        assert token not in src


def test_local_o2_hidden_0o_branch_is_active() -> None:
    mod = HarmonicLocalO2ContractionTensorProduct(
        channels=2,
        lmax=2,
        internal_compute_dtype=torch.float64,
        normalization="none",
    )
    shape_scalar = (3, 2 * 3, 1)
    shape_f1 = (3, 2 * 2, 2)
    shape_f2 = (3, 2 * 1, 2)
    x1 = {
        ("scalar", 1): torch.zeros(shape_scalar, dtype=torch.float64),
        ("scalar", -1): torch.zeros(shape_scalar, dtype=torch.float64),
        ("freq", 1): torch.zeros(shape_f1, dtype=torch.float64),
        ("freq", 2): torch.zeros(shape_f2, dtype=torch.float64),
    }
    x2 = {
        ("scalar", 1): torch.zeros(shape_scalar, dtype=torch.float64),
        ("scalar", -1): torch.zeros(shape_scalar, dtype=torch.float64),
        ("freq", 1): torch.zeros(shape_f1, dtype=torch.float64),
        ("freq", 2): torch.zeros(shape_f2, dtype=torch.float64),
    }
    x1[("freq", 1)][..., 0] = 1.0
    x2[("freq", 1)][..., 1] = 1.0
    out = mod(x1, x2)
    assert out[("scalar", -1)].abs().sum().item() > 0.0
