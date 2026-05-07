from __future__ import annotations

import pytest
import torch

from molecular_force_field.models.pure_cartesian_ictd_fix_so2 import (
    PureCartesianICTDFixSO2,
    SO2PureUContraction,
    SO2PureUSymmetricContraction,
    _FastSO2PathWeightedTP,
    _build_source_feature_moments_flat_so2,
    _local_so2_full_change_of_basis,
    _local_to_flat_so2,
    _project_fusion_features_multihead_so2,
    _so3_blocks_to_node_local_so2_fast,
    _strict_so3_u_tensor_in_local_so2_basis,
)
from molecular_force_field.models.pure_cartesian_ictd_escn_so2 import (
    ICTDS2GridActivation,
    ICTDSO2Convolution,
    PureCartesianICTDESCNSO2,
    _pack_m_major_so2,
    _unpack_m_major_so2,
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


def test_pure_cartesian_ictd_escn_so2_cpu_smoke() -> None:
    dtype = torch.float32
    device = torch.device("cpu")
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = _make_dummy_graph(
        device,
        dtype,
        num_nodes=16,
        avg_degree=4,
    )
    pos = pos.requires_grad_(True)
    model = PureCartesianICTDESCNSO2(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=4,
        hidden_dim_conv=8,
        hidden_dim_sh=8,
        hidden_dim=8,
        lmax=1,
        num_interaction=2,
        ictd_fix_route="fusion",
        ictd_fix_fusion_heads=4,
        ictd_fix_fusion_head_weight_mode="softmax",
        ictd_fix_readout_head_scale_init=0.3,
        ictd_fix_readout_head_scale_trainable=True,
        save_contraction_order=2,
        internal_compute_dtype=None,
    ).to(device=device, dtype=dtype)
    out, feats = model(
        pos,
        A,
        batch,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        return_combined_features=True,
    )
    assert out.shape == (pos.shape[0], 1)
    assert feats.shape[0] == pos.shape[0]
    assert len(model.blocks) == 2
    assert not hasattr(model, "products")
    assert not hasattr(model, "fusion_readouts")
    loss = out.square().mean()
    loss.backward()
    assert pos.grad is not None
    assert torch.isfinite(out).all()
    assert torch.isfinite(feats).all()
    assert torch.isfinite(pos.grad).all()


def _random_so2_local(num_nodes: int, channels: int, lmax: int, *, dtype: torch.dtype) -> dict[tuple[int, int], torch.Tensor]:
    local: dict[tuple[int, int], torch.Tensor] = {}
    for l in range(lmax + 1):
        local[(l, 0)] = torch.randn(num_nodes, channels, 1, dtype=dtype)
        for m in range(1, l + 1):
            local[(l, m)] = torch.randn(num_nodes, channels, 2, dtype=dtype)
    return local


def _rotate_so2_local_z(
    local: dict[tuple[int, int], torch.Tensor],
    theta: float,
) -> dict[tuple[int, int], torch.Tensor]:
    out: dict[tuple[int, int], torch.Tensor] = {}
    for (l, m), block in local.items():
        if m == 0:
            out[(l, m)] = block.clone()
            continue
        c = torch.cos(block.new_tensor(float(m) * theta))
        s = torch.sin(block.new_tensor(float(m) * theta))
        x = block[..., 0]
        y = block[..., 1]
        out[(l, m)] = torch.stack([c * x - s * y, s * x + c * y], dim=-1)
    return out


def test_escn_so2_m_major_pack_roundtrip() -> None:
    torch.manual_seed(0)
    local = _random_so2_local(5, 4, 3, dtype=torch.float32)
    packed = _pack_m_major_so2(local, lmax=3, mmax=3)
    restored = _unpack_m_major_so2(packed, lmax=3, channels=4, mmax=3)
    assert list(packed) == [0, 1, 2, 3]
    assert packed[0].shape == (5, 4, 4, 1)
    assert packed[1].shape == (5, 3, 4, 2)
    for key in local:
        assert torch.equal(local[key], restored[key])


def test_escn_so2_convolution_block_equivariance_smoke() -> None:
    torch.manual_seed(1)
    local = _random_so2_local(7, 3, 2, dtype=torch.float32)
    conv = ICTDSO2Convolution(in_channels=3, out_channels=5, lmax=2, external_weights=False)
    theta = 0.41
    y_rotated_input = conv(_rotate_so2_local_z(local, theta))
    y_rotated_output = _rotate_so2_local_z(conv(local), theta)
    for key in y_rotated_output:
        assert torch.allclose(y_rotated_input[key], y_rotated_output[key], atol=2e-6, rtol=2e-6)


def test_escn_s2_grid_activation_forward_backward_finite() -> None:
    torch.manual_seed(2)
    local = _random_so2_local(6, 4, 2, dtype=torch.float32)
    local = {key: value.requires_grad_(True) for key, value in local.items()}
    act = ICTDS2GridActivation(channels=4, lmax=2, grid_points=64)
    out = act(local)
    loss = sum(value.square().mean() for value in out.values())
    loss.backward()
    for key in local:
        assert out[key].shape == local[key].shape
        assert torch.isfinite(out[key]).all()
        assert local[key].grad is not None
        assert torch.isfinite(local[key].grad).all()


def test_pure_cartesian_ictd_escn_so2_z_rotation_energy_force_smoke() -> None:
    torch.manual_seed(3)
    dtype = torch.float32
    device = torch.device("cpu")
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = _make_dummy_graph(
        device,
        dtype,
        num_nodes=10,
        avg_degree=3,
        seed=3,
    )
    model = PureCartesianICTDESCNSO2(
        main_max_radius=5.0,
        main_number_of_basis=4,
        hidden_dim_conv=6,
        hidden_dim_sh=6,
        hidden_dim=6,
        lmax=1,
        num_interaction=2,
        ictd_fix_route="fusion",
        ictd_fix_fusion_heads=4,
        ictd_fix_fusion_head_weight_mode="softmax",
        ictd_fix_readout_head_scale_init=0.3,
        ictd_fix_readout_head_scale_trainable=True,
        save_contraction_order=2,
        internal_compute_dtype=None,
    ).to(device=device, dtype=dtype)
    theta = 0.23
    c = torch.cos(torch.tensor(theta, dtype=dtype))
    s = torch.sin(torch.tensor(theta, dtype=dtype))
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
    pos1 = pos.detach().clone().requires_grad_(True)
    pos2 = (pos.detach() @ R.T).requires_grad_(True)

    e1 = model(pos1, A, batch, edge_src, edge_dst, edge_shifts, cell)
    e2 = model(pos2, A, batch, edge_src, edge_dst, edge_shifts, cell)
    g1 = torch.autograd.grad(e1.sum(), pos1, create_graph=False)[0]
    g2 = torch.autograd.grad(e2.sum(), pos2, create_graph=False)[0]

    assert torch.allclose(e1, e2, atol=2e-3, rtol=2e-3)
    assert torch.allclose(g2, g1 @ R.T, atol=5e-2, rtol=5e-2)


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
    assert isinstance(model.products[-1].symmetric_contractions, SO2PureUSymmetricContraction)
    assert isinstance(model.multiple_contraction_mix.symmetric_contractions, SO2PureUSymmetricContraction)


def test_fix_so2_fusion_supports_multiple_readout_heads() -> None:
    model = PureCartesianICTDFixSO2(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=4,
        hidden_dim_conv=8,
        hidden_dim_sh=8,
        hidden_dim=8,
        lmax=1,
        num_interaction=2,
        ictd_fix_route="fusion",
        ictd_fix_fusion_heads=4,
        ictd_fix_fusion_head_weight_mode="softmax",
        save_contraction_order=2,
        internal_compute_dtype=None,
    )
    assert model.ictd_fix_fusion_heads == 4
    assert len(model.fusion_readouts) == 4
    assert len(model.layer_energy_readouts) + 1 + len(model.fusion_readouts) == 6
    assert model.fusion_readout is model.fusion_readouts[0]
    assert model.fusion_head_logits.shape == (4,)
    assert model.fusion_head_weights is None
    assert model.fusion_energy_scale is not None


def test_fix_so2_interaction_initialization_matches_so3_scale_conventions() -> None:
    model = PureCartesianICTDFixSO2(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=4,
        hidden_dim_conv=8,
        hidden_dim_sh=8,
        hidden_dim=8,
        lmax=2,
        num_interaction=2,
        ictd_fix_route="fusion",
        ictd_fix_fusion_heads=4,
        save_contraction_order=2,
        internal_compute_dtype=None,
    )

    assert torch.allclose(
        model.interactions[0].recouple.weight,
        torch.ones_like(model.interactions[0].recouple.weight),
    )
    assert torch.allclose(
        model.interactions[1].recouple.weight,
        torch.ones_like(model.interactions[1].recouple.weight),
    )

    for weight in model.interactions[0].message_selector.weights.values():
        eye = torch.eye(weight.shape[-1], dtype=weight.dtype).expand(weight.shape[0], -1, -1)
        assert torch.allclose(weight, eye)

    for weight in model.interactions[1].self_connection.weights.values():
        eye = torch.eye(weight.shape[-1], dtype=weight.dtype).expand(weight.shape[0], -1, -1)
        assert torch.allclose(weight, eye)

    for adapter in model.products[0].linear.adapters.values():
        assert torch.allclose(adapter.weight, torch.eye(adapter.weight.shape[0], dtype=adapter.weight.dtype))
    assert torch.allclose(
        model.products[1].linear.weight,
        torch.eye(model.products[1].linear.weight.shape[0], dtype=model.products[1].linear.weight.dtype),
    )
    assert torch.allclose(
        model.products[1].sc_scale,
        torch.tensor(3.0, dtype=model.products[1].sc_scale.dtype),
    )


def test_fix_so2_mace_rms_interaction_scale_is_wired() -> None:
    model = PureCartesianICTDFixSO2(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=4,
        hidden_dim_conv=8,
        hidden_dim_sh=8,
        hidden_dim=8,
        lmax=2,
        num_interaction=2,
        ictd_fix_route="fusion",
        ictd_fix_interaction_scale="mace-rms",
        save_contraction_order=2,
        internal_compute_dtype=None,
    )

    assert torch.allclose(
        model.interactions[0].message_output_scale.log_scale.detach().exp(),
        torch.tensor([0.625, 0.561, 0.540], dtype=model.interactions[0].message_output_scale.log_scale.dtype),
    )
    assert torch.allclose(
        model.interactions[1].message_output_scale.log_scale.detach().exp(),
        torch.tensor([0.489, 0.745, 0.741], dtype=model.interactions[1].message_output_scale.log_scale.dtype),
    )
    assert torch.allclose(
        model.interactions[1].sc_output_scale.log_scale.detach().exp(),
        torch.tensor([0.342, 0.5, 0.5], dtype=model.interactions[1].sc_output_scale.log_scale.dtype),
    )


def test_fast_so2_path_weighted_tp_matches_reference() -> None:
    from molecular_force_field.models.ictd_irreps_2d import HarmonicPathWeightedTensorProductSO2

    torch.manual_seed(11)
    channels = 5
    mmax = 3
    ref = HarmonicPathWeightedTensorProductSO2(channels=channels, mmax=mmax)
    fast = _FastSO2PathWeightedTP(channels=channels, mmax=mmax)
    with torch.no_grad():
        fast.weight.copy_(ref.weight)
    x1 = {
        m: torch.randn(7, channels, 1 if m == 0 else 2)
        for m in range(mmax + 1)
    }
    x2 = {
        m: torch.randn(7, channels, 1 if m == 0 else 2)
        for m in range(mmax + 1)
    }
    out_ref = ref(x1, x2)
    out_fast = fast(x1, x2)
    assert out_ref.keys() == out_fast.keys()
    for key in out_ref:
        assert torch.allclose(out_fast[key], out_ref[key], atol=1e-6, rtol=1e-6)


def test_flat_so2_source_feature_moments_match_reference() -> None:
    from molecular_force_field.models.pure_cartesian_ictd_layers_node_so2 import build_source_feature_moments

    torch.manual_seed(13)
    lmax = 2
    channels = 4
    basis = 3
    num_nodes = 6
    num_edges = 18
    local = {
        (l, m): torch.randn(num_nodes, channels, 1 if m == 0 else 2)
        for l in range(lmax + 1)
        for m in range(l + 1)
    }
    edge_src = torch.randint(0, num_nodes, (num_edges,))
    edge_dst = torch.randint(0, num_nodes, (num_edges,))
    edge_radial = torch.randn(num_edges, basis)
    edge_mask = torch.randint(0, 2, (num_edges, 1), dtype=torch.float32)
    neighbor_count = torch.zeros(num_nodes)
    neighbor_count.scatter_add_(0, edge_dst, edge_mask.squeeze(-1))
    neighbor_count = neighbor_count.clamp(min=1.0)

    ref = build_source_feature_moments(
        local,
        edge_radial=edge_radial,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_mask=edge_mask,
        neighbor_count=neighbor_count,
        lmax=lmax,
    )
    fast = _build_source_feature_moments_flat_so2(
        local,
        edge_radial=edge_radial,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_mask=edge_mask,
        neighbor_count=neighbor_count,
        lmax=lmax,
    )
    assert ref.keys() == fast.keys()
    for key in ref:
        assert torch.allclose(fast[key], ref[key], atol=1e-6, rtol=1e-6)


def test_fast_so3_to_node_local_so2_matches_reference() -> None:
    from molecular_force_field.models.node_local_so2_helpers import so3_blocks_to_node_local_so2

    torch.manual_seed(17)
    lmax = 3
    num_nodes = 5
    channels = 4
    blocks = {
        l: torch.randn(num_nodes, channels, 2 * l + 1)
        for l in range(lmax + 1)
    }
    q, _ = torch.linalg.qr(torch.randn(num_nodes, 3, 3))
    det = torch.det(q)
    q = torch.where((det < 0).view(-1, 1, 1), q * torch.tensor([-1.0, 1.0, 1.0]).view(1, 1, 3), q)

    ref = so3_blocks_to_node_local_so2(blocks, q, lmax, compute_dtype=torch.float32)
    fast = _so3_blocks_to_node_local_so2_fast(blocks, q, lmax, compute_dtype=torch.float32)
    assert ref.keys() == fast.keys()
    for key in ref:
        assert torch.allclose(fast[key], ref[key], atol=1e-5, rtol=1e-5)


def test_multihead_fusion_projection_matches_per_head_projection() -> None:
    torch.manual_seed(23)
    model = PureCartesianICTDFixSO2(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=4,
        hidden_dim_conv=8,
        hidden_dim_sh=8,
        hidden_dim=8,
        lmax=1,
        num_interaction=2,
        ictd_fix_route="fusion",
        ictd_fix_fusion_heads=4,
        ictd_fix_fusion_head_weight_mode="softmax",
        main_hidden_sizes3=[8],
        save_contraction_order=2,
        internal_compute_dtype=None,
    )
    features = torch.randn(6, model.fusion_readouts[0].proj_total.input_norm.normalized_shape[0])
    ref = torch.stack(
        [readout.project_features(features) for readout in model.fusion_readouts],
        dim=0,
    )
    fast = _project_fusion_features_multihead_so2(model.fusion_readouts, features)
    assert torch.allclose(fast, ref, atol=1e-6, rtol=1e-6)


def test_pure_cartesian_ictd_fix_so2_force_training_backward_smoke() -> None:
    dtype = torch.float32
    device = torch.device("cpu")
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = _make_dummy_graph(
        device,
        dtype,
        num_nodes=12,
        avg_degree=6,
    )
    pos = pos.requires_grad_(True)
    model = PureCartesianICTDFixSO2(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=4,
        hidden_dim_conv=8,
        hidden_dim_sh=8,
        hidden_dim=8,
        lmax=1,
        num_interaction=2,
        ictd_fix_route="fusion",
        ictd_fix_fusion_heads=2,
        save_contraction_order=2,
        internal_compute_dtype=None,
    ).to(device=device, dtype=dtype)
    energy = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell).sum()
    force_grad = torch.autograd.grad(energy, pos, create_graph=True, retain_graph=True)[0]
    loss = energy.square() + force_grad.square().mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_fix_so2_trainable_readout_head_scales_receive_grad() -> None:
    dtype = torch.float32
    device = torch.device("cpu")
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = _make_dummy_graph(
        device,
        dtype,
        num_nodes=10,
        avg_degree=5,
    )
    model = PureCartesianICTDFixSO2(
        max_embed_radius=5.0,
        main_max_radius=5.0,
        main_number_of_basis=4,
        hidden_dim_conv=8,
        hidden_dim_sh=8,
        hidden_dim=8,
        lmax=1,
        num_interaction=2,
        ictd_fix_route="fusion",
        ictd_fix_fusion_heads=2,
        ictd_fix_readout_head_scale_init=0.1,
        ictd_fix_readout_head_scale_trainable=True,
        save_contraction_order=2,
        internal_compute_dtype=None,
    ).to(device=device, dtype=dtype)

    assert model.readout_head_scales is not None
    assert model.readout_head_scales.requires_grad
    assert torch.allclose(model.readout_head_scales.detach(), torch.full((2,), 0.1, dtype=dtype))

    energy = model(pos, A, batch, edge_src, edge_dst, edge_shifts, cell).sum()
    energy.backward()

    assert model.readout_head_scales.grad is not None
    assert torch.isfinite(model.readout_head_scales.grad).all()


def test_so2_pure_u_contraction_preserves_channel_l_layout() -> None:
    torch.manual_seed(7)
    channels = 3
    lmax = 2
    num_nodes = 5
    num_elements = 4
    module = SO2PureUSymmetricContraction(
        num_elements=num_elements,
        channels=channels,
        lmax=lmax,
        target_lmax=lmax,
        correlation=2,
    )
    local = {
        (l, m): torch.randn(num_nodes, channels, 1 if m == 0 else 2)
        for l in range(lmax + 1)
        for m in range(l + 1)
    }
    atom_types = torch.tensor([0, 1, 2, 3, 1])
    node_attrs = torch.nn.functional.one_hot(atom_types, num_classes=num_elements).to(
        dtype=next(iter(local.values())).dtype
    )

    out = module(local, node_attrs)
    x = _local_to_flat_so2(local, lmax)
    expected_flat = torch.cat(
        [
            contraction(x, node_attrs).view(num_nodes, channels, 2 * l + 1)
            for l, contraction in enumerate(module.contractions)
        ],
        dim=-1,
    )

    offset = 0
    for l in range(lmax + 1):
        for m in range(l + 1):
            dim = 1 if m == 0 else 2
            actual = out[(l, m)].reshape(num_nodes, channels, dim)
            expected = expected_flat[:, :, offset : offset + dim]
            assert torch.allclose(actual, expected)
            offset += dim


def test_so2_pure_u_contraction_matches_so3_u_after_local_basis_change() -> None:
    from molecular_force_field.models.ictd_irreps import ictd_u_matrix_so3
    from molecular_force_field.models.pure_cartesian_ictd_fix import _ICTDPureUContraction

    torch.manual_seed(31)
    lmax = 2
    output_l = 2
    correlation = 2
    channels = 2
    num_elements = 3
    num_nodes = 4
    dtype = torch.float32

    u_so3 = {
        nu: ictd_u_matrix_so3(
            lmax=lmax,
            output_l=output_l,
            correlation=nu,
            irrep_normalization="component",
            dtype=dtype,
        )
        for nu in range(1, correlation + 1)
    }
    u_local = {
        nu: _strict_so3_u_tensor_in_local_so2_basis(
            lmax=lmax,
            output_l=output_l,
            correlation=nu,
            dtype=dtype,
        )
        for nu in range(1, correlation + 1)
    }
    so3 = _ICTDPureUContraction(
        u_tensors=u_so3,
        output_l=output_l,
        num_elements=num_elements,
        num_features=channels,
    )
    so2 = SO2PureUContraction(
        u_tensors=u_local,
        output_l=output_l,
        num_elements=num_elements,
        num_features=channels,
        lmax=lmax,
    )
    with torch.no_grad():
        so2.weights_max.copy_(so3.weights_max)
        for dst, src in zip(so2.weights, so3.weights):
            dst.copy_(src)

    dim = (lmax + 1) ** 2
    x_so3 = torch.randn(num_nodes, channels, dim, dtype=dtype)
    q_all = _local_so2_full_change_of_basis(lmax, dtype=dtype)
    x_local = x_so3 @ q_all
    atom_types = torch.tensor([0, 1, 2, 1])
    node_attrs = torch.nn.functional.one_hot(atom_types, num_classes=num_elements).to(dtype=dtype)

    out_so3 = so3(x_so3, node_attrs).view(num_nodes, channels, 2 * output_l + 1)
    q_out = _local_so2_full_change_of_basis(output_l, dtype=dtype)[
        output_l * output_l : (output_l + 1) * (output_l + 1),
        output_l * output_l : (output_l + 1) * (output_l + 1),
    ]
    expected_local = out_so3 @ q_out
    actual_local = so2(x_local, node_attrs).view(num_nodes, channels, 2 * output_l + 1)
    assert torch.allclose(actual_local, expected_local, atol=5e-5, rtol=5e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pure_cartesian_ictd_fix_so2_cuda_smoke() -> None:
    _run_model_smoke(torch.device("cuda"), "baseline")
    _run_model_smoke(torch.device("cuda"), "fusion")
