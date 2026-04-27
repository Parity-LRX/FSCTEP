from __future__ import annotations

import math

import pytest
import torch

from molecular_force_field.models.node_local_so2_helpers import (
    build_node_local_frames,
    node_local_so2_to_so3_blocks,
    pack_node_local_so2,
    so3_blocks_to_node_local_so2,
    unpack_node_local_so2,
)
from molecular_force_field.models.pure_cartesian_ictd_layers import _merge_irreps, _split_irreps
from molecular_force_field.models.pure_cartesian_ictd_layers_node_so2 import (
    NodeLocalSO2SymmetricContraction,
    PureCartesianICTDNodeSO2TransformerLayer,
)


def _rotation_z(theta: float, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)


def _make_blocks(batch: int = 4, channels: int = 3, lmax: int = 2, dtype: torch.dtype = torch.float64):
    g = torch.Generator().manual_seed(1234)
    return {
        l: torch.randn(batch, channels, 2 * l + 1, generator=g, dtype=dtype)
        for l in range(lmax + 1)
    }


def _make_local(batch: int = 5, channels: int = 3, lmax: int = 2, dtype: torch.dtype = torch.float64):
    g = torch.Generator().manual_seed(4321)
    return {
        (l, m): torch.randn(batch, channels, 1 if m == 0 else 2, generator=g, dtype=dtype)
        for l in range(lmax + 1)
        for m in range(l + 1)
    }


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


def test_build_node_local_frames_rotation_behavior() -> None:
    edge_vec = torch.tensor(
        [
            [1.2, 0.1, 0.3],
            [0.0, 1.3, 0.4],
            [0.2, 0.3, 1.1],
            [1.0, 1.0, 0.5],
        ],
        dtype=torch.float64,
    )
    edge_dst = torch.zeros(edge_vec.size(0), dtype=torch.long)
    G = build_node_local_frames(edge_vec, edge_dst, 1, compute_dtype=torch.float64)
    R = _rotation_z(0.7)
    edge_vec_rot = edge_vec @ R.transpose(-1, -2)
    G_rot = build_node_local_frames(edge_vec_rot, edge_dst, 1, compute_dtype=torch.float64)
    torch.testing.assert_close(G_rot[0] @ R, G[0], rtol=1e-8, atol=1e-8)
    I = G[0].transpose(-1, -2) @ G[0]
    torch.testing.assert_close(I, torch.eye(3, dtype=torch.float64), rtol=1e-8, atol=1e-8)


def test_so3_node_local_so2_roundtrip_l0_l2() -> None:
    blocks = _make_blocks(dtype=torch.float64)
    G = _rotation_z(0.3).unsqueeze(0).expand(blocks[0].shape[0], -1, -1).contiguous()
    local = so3_blocks_to_node_local_so2(blocks, G, 2, compute_dtype=torch.float64)
    rebuilt = node_local_so2_to_so3_blocks(local, G, 2, out_dtype=torch.float64, compute_dtype=torch.float64)
    for l in range(3):
        torch.testing.assert_close(rebuilt[l], blocks[l], rtol=1e-8, atol=1e-8)


def test_node_local_pack_unpack_roundtrip() -> None:
    local = _make_local(dtype=torch.float64)
    packed = pack_node_local_so2(local, 2)
    rebuilt = unpack_node_local_so2(packed, 2)
    for key in local:
        torch.testing.assert_close(rebuilt[key], local[key], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("correlation", [1, 2, 3])
def test_node_local_so2_symmetric_contraction_shape_and_grad(correlation: int) -> None:
    batch = 4
    in_channels = 3
    hidden_channels = 2
    lmax = 2
    local = {
        key: value.clone().requires_grad_(True)
        for key, value in _make_local(batch=batch, channels=in_channels, lmax=lmax, dtype=torch.float64).items()
    }
    atom_types = torch.randint(1, 6, (batch,), dtype=torch.long)
    mod = NodeLocalSO2SymmetricContraction(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        lmax=lmax,
        correlation=correlation,
    )
    out = mod(local, atom_types)
    flat = torch.cat([out[(l, m)].reshape(batch, -1) for l in range(lmax + 1) for m in range(l + 1)], dim=-1)
    loss = flat.square().mean()
    loss.backward()
    for key in out:
        assert out[key].shape[-2] == hidden_channels
        assert torch.isfinite(out[key]).all()
    for key in local:
        assert local[key].grad is not None
        assert torch.isfinite(local[key].grad).all()


def _run_model_smoke(device: torch.device, mode: str):
    dtype = torch.float32
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = _make_dummy_graph(device, dtype)
    pos = pos.requires_grad_(True)
    model = PureCartesianICTDNodeSO2TransformerLayer(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=16,
        hidden_dim_sh=16,
        hidden_dim=16,
        lmax=2,
        num_interaction=3,
        save_readout_mode=mode,
        internal_compute_dtype=torch.float64,
    ).to(device=device, dtype=dtype)
    out = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    assert out.shape == (pos.shape[0], 1)
    loss = out.square().mean()
    loss.backward()
    assert pos.grad is not None
    assert torch.isfinite(out).all()
    assert torch.isfinite(pos.grad).all()


def test_node_local_so2_model_cpu_smoke() -> None:
    _run_model_smoke(torch.device("cpu"), "elementwise-scalar")
    _run_model_smoke(torch.device("cpu"), "multiple-contraction")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_node_local_so2_model_cuda_smoke() -> None:
    _run_model_smoke(torch.device("cuda"), "elementwise-scalar")
    _run_model_smoke(torch.device("cuda"), "multiple-contraction")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_node_local_so2_output_scale_matches_so3_order_of_magnitude() -> None:
    from molecular_force_field.models.pure_cartesian_ictd_layers import PureCartesianICTDTransformerLayer

    device = torch.device("cuda")
    dtype = torch.float32
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = _make_dummy_graph(device, dtype, num_nodes=32, avg_degree=8)
    so3 = PureCartesianICTDTransformerLayer(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=16,
        hidden_dim_sh=16,
        hidden_dim=16,
        lmax=2,
        num_interaction=3,
        save_readout_mode="multiple-contraction",
        internal_compute_dtype=torch.float64,
    ).to(device=device, dtype=dtype)
    node_so2 = PureCartesianICTDNodeSO2TransformerLayer(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=8,
        hidden_dim_conv=16,
        hidden_dim_sh=16,
        hidden_dim=16,
        lmax=2,
        num_interaction=3,
        save_readout_mode="multiple-contraction",
        internal_compute_dtype=torch.float64,
    ).to(device=device, dtype=dtype)
    out_so3 = so3(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    out_node = node_so2(pos, A, batch, edge_src, edge_dst, edge_shifts, cell)
    ratio = float(out_node.std().item() / max(out_so3.std().item(), 1e-8))
    assert 0.01 < ratio < 100.0
