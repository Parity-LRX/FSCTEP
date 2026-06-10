from __future__ import annotations

import torch
import torch.nn.functional as F

from molecular_force_field.models.ictd_irreps import direction_harmonics_all
from molecular_force_field.models.pure_cartesian_ictd_fix import (
    PureCartesianICTDFix,
    _concat_so3_states_by_l,
)
from molecular_force_field.models.pure_cartesian_ictd_layers import _split_irreps


def _all_directed_edges(num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    src = []
    dst = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)


def _toy_graph(dtype: torch.dtype = torch.float64):
    pos = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.81, 0.13, 0.07],
            [0.19, 0.92, 0.21],
            [0.27, 0.18, 1.03],
            [1.07, 0.84, 0.47],
        ],
        dtype=dtype,
    )
    atomic_numbers = torch.tensor([1, 6, 8, 1, 6], dtype=torch.long)
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    edge_src, edge_dst = _all_directed_edges(pos.shape[0])
    edge_shifts = torch.zeros(edge_src.shape[0], 3, dtype=dtype)
    cell = torch.eye(3, dtype=dtype).unsqueeze(0) * 8.0
    return pos, atomic_numbers, batch, edge_src, edge_dst, edge_shifts, cell


def _random_rotation(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260503)
    mat = torch.randn(3, 3, generator=generator, dtype=dtype)
    q, _ = torch.linalg.qr(mat)
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _right_action_matrices(rotation: torch.Tensor, lmax: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1729)
    xyz = torch.randn(256, 3, generator=generator, dtype=rotation.dtype)
    xyz = F.normalize(xyz, dim=-1)
    xyz_rot = xyz @ rotation.T
    y = direction_harmonics_all(xyz, lmax)
    y_rot = direction_harmonics_all(xyz_rot, lmax)
    return [torch.linalg.lstsq(y[l], y_rot[l]).solution.contiguous() for l in range(lmax + 1)]


def _small_fix_model(
    *,
    route: str,
    readout_head_scale_trainable: bool = False,
    lmax: int = 1,
    gmix_output_lmax: int | None = None,
) -> PureCartesianICTDFix:
    return PureCartesianICTDFix(
        max_embed_radius=4.0,
        main_max_radius=4.0,
        main_number_of_basis=3,
        hidden_dim_conv=2,
        hidden_dim_sh=2,
        hidden_dim=2,
        channel_in2=2,
        embedding_dim=4,
        max_atomvalue=8,
        atomic_numbers=(1, 6, 8),
        output_size=4,
        embed_size=[4],
        main_hidden_sizes3=[4],
        num_layers=1,
        num_interaction=2,
        function_type_main="bessel",
        lmax=lmax,
        ictd_fix_route=route,
        ictd_fix_product_backend="ictd-pure-u",
        ictd_fix_fusion_scale_init=1.0,
        ictd_fix_fusion_heads=1,
        ictd_fix_readout_head_scale_trainable=readout_head_scale_trainable,
        ictd_fix_gmix_output_lmax=gmix_output_lmax,
        save_contraction_order=2,
        avg_num_neighbors=4.0,
    ).to(dtype=torch.float64)


def test_concat_so3_states_by_l_groups_channels_within_each_l() -> None:
    torch.manual_seed(0)
    dtype = torch.float64
    num_nodes = 3
    channels = 2
    lmax = 2
    states = [
        torch.randn(num_nodes, channels * (lmax + 1) ** 2, dtype=dtype),
        torch.randn(num_nodes, channels * (lmax + 1) ** 2, dtype=dtype),
        torch.randn(num_nodes, channels * (lmax + 1) ** 2, dtype=dtype),
    ]

    out = _concat_so3_states_by_l(states, channels, lmax)
    out_blocks = _split_irreps(out, channels * len(states), lmax)
    state_blocks = [_split_irreps(state, channels, lmax) for state in states]

    for l in range(lmax + 1):
        expected = torch.cat([blocks[l] for blocks in state_blocks], dim=-2)
        torch.testing.assert_close(out_blocks[l], expected, atol=0.0, rtol=0.0)


def test_fusion_mix_output_is_so3_equivariant_after_layer_state_concat_fix() -> None:
    torch.manual_seed(11)
    model = _small_fix_model(route="fusion")
    model.eval()
    captured: list[torch.Tensor] = []

    def capture_g_mix(_module, _inputs, output):
        captured.append(output.detach())

    handle = model.multiple_contraction_mix.register_forward_hook(capture_g_mix)
    try:
        graph = _toy_graph()
        pos, atomic_numbers, batch, edge_src, edge_dst, edge_shifts, cell = graph
        with torch.no_grad():
            energy = model(pos, atomic_numbers, batch, edge_src, edge_dst, edge_shifts, cell)
        g_mix = captured.pop()

        rotation = _random_rotation()
        d_mats = _right_action_matrices(rotation, model.lmax)
        with torch.no_grad():
            energy_rot = model(pos @ rotation.T, atomic_numbers, batch, edge_src, edge_dst, edge_shifts, cell)
        g_mix_rot = captured.pop()
    finally:
        handle.remove()

    assert g_mix.abs().max().item() > 1.0e-12
    assert energy.abs().max().item() > 1.0e-12
    assert energy_rot.abs().max().item() > 1.0e-12

    channels = model.save_multiple_mix_channels
    blocks = _split_irreps(g_mix, channels, model.lmax)
    blocks_rot = _split_irreps(g_mix_rot, channels, model.lmax)
    for l in range(model.lmax + 1):
        expected = torch.matmul(blocks[l], d_mats[l])
        torch.testing.assert_close(blocks_rot[l], expected, atol=1.0e-8, rtol=1.0e-8)


def test_gmix_output_lmax_default_matches_lmax_byte_identical() -> None:
    """gmix_output_lmax=None resolves to lmax => byte-identical to gmix_output_lmax=lmax."""
    torch.manual_seed(7)
    m_default = _small_fix_model(route="fusion", lmax=1, gmix_output_lmax=None)
    torch.manual_seed(7)
    m_explicit = _small_fix_model(route="fusion", lmax=1, gmix_output_lmax=1)
    assert m_default.ictd_fix_gmix_output_lmax == 1
    assert m_default.multiple_contraction_mix.output_lmax == 1
    graph = _toy_graph()
    pos, atomic_numbers, batch, edge_src, edge_dst, edge_shifts, cell = graph
    with torch.no_grad():
        e_d = m_default(pos, atomic_numbers, batch, edge_src, edge_dst, edge_shifts, cell)
        e_e = m_explicit(pos, atomic_numbers, batch, edge_src, edge_dst, edge_shifts, cell)
    torch.testing.assert_close(e_d, e_e, atol=0.0, rtol=0.0)


def _check_gmix_higher_output_lmax(lmax: int, gmix_output_lmax: int) -> None:
    """gmix contraction emitting output_lmax > input lmax must be (a) non-trivial at the
    new l, (b) SO(3)-equivariant per block, (c) energy-invariant + force-covariant end-to-end."""
    assert gmix_output_lmax > lmax
    torch.manual_seed(11)
    model = _small_fix_model(route="fusion", lmax=lmax, gmix_output_lmax=gmix_output_lmax)
    model.eval()
    assert model.multiple_contraction_mix.output_lmax == gmix_output_lmax

    captured: list[torch.Tensor] = []

    def capture_g_mix(_module, _inputs, output):
        captured.append(output.detach())

    graph = _toy_graph()
    pos, atomic_numbers, batch, edge_src, edge_dst, edge_shifts, cell = graph
    rotation = _random_rotation()
    d_mats = _right_action_matrices(rotation, gmix_output_lmax)

    handle = model.multiple_contraction_mix.register_forward_hook(capture_g_mix)
    try:
        pos_req = pos.detach().clone().requires_grad_(True)
        energy = model(pos_req, atomic_numbers, batch, edge_src, edge_dst, edge_shifts, cell)
        forces = -torch.autograd.grad(energy.sum(), pos_req)[0]
        g_mix = captured.pop()

        pos_rot = (pos @ rotation.T).detach().clone().requires_grad_(True)
        energy_rot = model(pos_rot, atomic_numbers, batch, edge_src, edge_dst, edge_shifts, cell)
        forces_rot = -torch.autograd.grad(energy_rot.sum(), pos_rot)[0]
        g_mix_rot = captured.pop()
    finally:
        handle.remove()

    for t in (energy, forces, g_mix, energy_rot, forces_rot, g_mix_rot):
        assert torch.isfinite(t).all()

    channels = model.save_multiple_mix_channels
    blocks = _split_irreps(g_mix, channels, gmix_output_lmax)
    blocks_rot = _split_irreps(g_mix_rot, channels, gmix_output_lmax)

    # the new higher-l block actually carries signal (it is doing something)
    assert blocks[gmix_output_lmax].abs().max().item() > 1.0e-9

    # equivariance of EVERY gmix output block, including the new higher l
    for l in range(gmix_output_lmax + 1):
        expected = torch.matmul(blocks[l], d_mats[l])
        torch.testing.assert_close(blocks_rot[l], expected, atol=1.0e-8, rtol=1.0e-8)

    # end-to-end: energy invariance + force covariance F(pos@R.T) == F(pos) @ R.T
    torch.testing.assert_close(energy_rot, energy, atol=1.0e-8, rtol=1.0e-8)
    torch.testing.assert_close(forces_rot, forces @ rotation.T, atol=1.0e-7, rtol=1.0e-7)


def test_gmix_output_lmax_l1_to_l2_equivariant_nonzero() -> None:
    _check_gmix_higher_output_lmax(lmax=1, gmix_output_lmax=2)


def test_gmix_output_lmax_l2_to_l3_equivariant_nonzero() -> None:
    _check_gmix_higher_output_lmax(lmax=2, gmix_output_lmax=3)


def test_pure_cartesian_ictd_fix_baseline_route_forward_backward_smoke() -> None:
    torch.manual_seed(13)
    model = _small_fix_model(route="baseline")
    graph = _toy_graph()
    pos, atomic_numbers, batch, edge_src, edge_dst, edge_shifts, cell = graph
    pos = pos.detach().requires_grad_(True)

    energy = model(pos, atomic_numbers, batch, edge_src, edge_dst, edge_shifts, cell)
    loss = energy.sum()
    loss.backward()

    assert torch.isfinite(energy).all()
    assert pos.grad is not None
    assert torch.isfinite(pos.grad).all()


def test_pure_cartesian_ictd_fix_readout_head_scales_receive_gradients() -> None:
    torch.manual_seed(17)
    model = _small_fix_model(route="baseline", readout_head_scale_trainable=True)
    graph = _toy_graph()
    pos, atomic_numbers, batch, edge_src, edge_dst, edge_shifts, cell = graph

    energy = model(pos, atomic_numbers, batch, edge_src, edge_dst, edge_shifts, cell)
    loss = energy.square().sum()
    loss.backward()

    assert model.readout_head_scales is not None
    assert model.readout_head_scales.shape == (2,)
    torch.testing.assert_close(
        model.readout_head_scales.detach(),
        torch.ones(2, dtype=torch.float64),
        atol=0.0,
        rtol=0.0,
    )
    assert model.readout_head_scales.grad is not None
    assert torch.isfinite(model.readout_head_scales.grad).all()
    assert model.readout_head_scales.grad.abs().max().item() > 0.0
